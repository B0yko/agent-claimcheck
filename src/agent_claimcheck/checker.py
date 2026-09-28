"""`Checker`: turn traces into `CheckResult`s with one configured detector.

The pipeline is the same regardless of which detector is configured:
`detector_view` (redaction) -> `resolve_claims` (structured claims win,
otherwise pattern extraction) -> skip if there is no success claim ->
`detector.score` -> calibrate (an ensemble already calibrated its own
deciding component; a plain detector is calibrated here) -> `gate` -> one
`CheckResult`. `CheckResult` is this project's other public schema, next to
`agent-trace/v1`: every one this module builds is validated against
`schemas/claimcheck-result-v1.json` before `dump_result` hands back its
canonical JSON line.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import httpx
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field

from agent_claimcheck import resources
from agent_claimcheck.calibration import CalibratorSet
from agent_claimcheck.claims import ClaimExtractor, ResolvedClaim, success_claims
from agent_claimcheck.config import Config, load_config, resolve_api_key
from agent_claimcheck.config import cache_dir as _cache_dir
from agent_claimcheck.config import ledger_path as _ledger_path
from agent_claimcheck.detectors.base import Detector, DetectorOutput, Reason, get_detector
from agent_claimcheck.detectors.classifier import ClassifierDetector, load_artifact
from agent_claimcheck.detectors.ensemble import CascadeDetector, CascadeOfflineDetector
from agent_claimcheck.detectors.judge import JudgeDetector
from agent_claimcheck.detectors.rules import RulesDetector
from agent_claimcheck.gate import Thresholds, UnknownPriceError, confidence, gate
from agent_claimcheck.judge.cache import JudgeCache
from agent_claimcheck.judge.render import DEFAULT_PROMPT_NAME
from agent_claimcheck.ledger import Budget, Ledger, Price, PriceBook
from agent_claimcheck.redact import DetectorView, detector_view, resolve_claims
from agent_claimcheck.rules.engine import RulePack, builtin_packs, load_rule_pack
from agent_claimcheck.schema import Domain, Trace, merge_probes

#: Reported as `p_success` when nothing else (a fitted `CalibratorSet`, an
#: artifact) supplies a base rate.
DEFAULT_BASE_RATE = 0.6

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_RESULT_SCHEMA_REL = "schemas/claimcheck-result-v1.json"


class CheckResult(BaseModel):
    """One trace's outcome, published as `claimcheck-result/v1`."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True, serialize_by_alias=True
    )

    schema_: Literal["claimcheck-result/v1"] = Field(alias="schema")
    trace_id: str
    domain: Domain
    detector: str
    verdict: Literal["verified", "false_success", "unverifiable", "skipped"]
    #: The gate's value: calibrated when a calibrator exists, else raw. Null
    #: only for `skipped` (no detector ever ran).
    p_success: float | None
    p_success_raw: float | None
    calibrated: bool
    abstain: bool
    abstain_reason: str | None
    confidence: float | None
    reasons: list[Reason]
    claims: list[ResolvedClaim]
    detectors: list[DetectorOutput]
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False


def _result_validator() -> Draft202012Validator:
    schema_path = resources.path(_RESULT_SCHEMA_REL)
    with schema_path.open("r", encoding="utf-8") as fh:
        schema_dict: dict[str, Any] = json.load(fh)
    return Draft202012Validator(schema_dict)


def _json_path(absolute_path: Iterable[Any]) -> str:
    parts = ["$"]
    for segment in absolute_path:
        if isinstance(segment, int):
            parts[-1] += f"[{segment}]"
        else:
            parts.append(f".{segment}")
    return "".join(parts)


def dump_result(result: CheckResult) -> str:
    """Serialize one `CheckResult` to a canonical JSON line, validating it
    against `schemas/claimcheck-result-v1.json` first.
    """
    obj = result.model_dump(mode="json", exclude_none=False)
    errors = sorted(_result_validator().iter_errors(obj), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        raise ValueError(
            f"invalid claimcheck-result: {_json_path(first.absolute_path)}: {first.message}"
        )
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def builtin_calibrators() -> CalibratorSet:
    """The packaged `models/calibrators-v1.json`."""
    return CalibratorSet.load(_MODELS_DIR / "calibrators-v1.json")


class Checker:
    """Runs one configured detector over traces and yields `CheckResult`s."""

    def __init__(
        self,
        detector: str = "cascade-offline",
        *,
        config: Config | None = None,
        rules: Sequence[str] = (),
        prompt: str | Path | None = None,
        calibration: str | Path | CalibratorSet | None = None,
        thresholds: Thresholds | None = None,
        max_usd: float | None = None,
        price: Price | None = None,
        concurrency: int | None = None,
        use_cache: bool = True,
    ) -> None:
        self.config = config or Config()
        self.use_cache = use_cache
        self.thresholds = thresholds or Thresholds(
            verified=self.config.gate.verified, false_success=self.config.gate.false_success
        )
        self.non_success_types = tuple(self.config.rules.non_success_types)

        pack_paths = [*self.config.rules.packs, *rules]
        self.packs: list[RulePack] = [
            *builtin_packs().values(),
            *(load_rule_pack(p) for p in pack_paths),
        ]
        self.extractor = ClaimExtractor(self.packs, self.non_success_types)

        # Precedence: the argument (or --calibration), then [classifier]
        # calibration from the config, then the packaged set.
        if calibration is None:
            calibration = self.config.classifier.calibration
        self.calibrators: CalibratorSet | None
        if isinstance(calibration, CalibratorSet):
            self.calibrators = calibration
            self.calibrators_builtin = False
        elif calibration is not None:
            self.calibrators = CalibratorSet.load(calibration)
            self.calibrators_builtin = False
        else:
            self.calibrators = builtin_calibrators()
            self.calibrators_builtin = True

        self.max_usd = max_usd if max_usd is not None else self.config.budget.max_usd
        self._price_override = price
        self._price_book: PriceBook | None = None
        self.concurrency = concurrency if concurrency is not None else self.config.judge.concurrency
        self.run_id = uuid.uuid4().hex[:12]
        self.ledger: Ledger | None = None

        self.detector_name = detector
        self._detector = self._build_detector(detector, prompt)

    @property
    def detector(self) -> Detector:
        """The built detector instance this `Checker` scores traces with."""
        return self._detector

    @classmethod
    def from_config(cls, path: str | Path, **kwargs: Any) -> Checker:
        """Build a `Checker` from a `claimcheck.toml` file."""
        return cls(config=load_config(path), **kwargs)

    def require_price(self) -> None:
        """Raise `UnknownPriceError` when this detector calls the judge and its model has no
        known price (an explicit `price`, then the config, then the API's model listing).

        `JudgeDetector` itself still abstains `unknown_price` trace by trace; a caller
        that would rather refuse a whole run up front calls this once after building the
        `Checker`. Detectors that never call the judge always pass.
        """
        if self._price_book is None:
            return
        model = self.config.judge.model or ""
        try:
            self._price_book.list_price(model)
        except UnknownPriceError:
            raise UnknownPriceError(
                f"no price known for judge model {model!r}: pass --price-in and --price-out, "
                "or set price_in_per_m and price_out_per_m under [judge]"
            ) from None

    def _base_rate(self) -> float:
        return self.calibrators.base_rate if self.calibrators is not None else DEFAULT_BASE_RATE

    def _build_rules(self) -> RulesDetector:
        return RulesDetector(self.packs, self.non_success_types, base_rate=self._base_rate())

    def _build_classifier(self) -> ClassifierDetector:
        return ClassifierDetector(load_artifact(self.config.classifier.model), guard=True)

    def _build_judge(self, prompt: str | Path | None) -> JudgeDetector:
        jc = self.config.judge
        if not jc.model:
            raise ValueError(
                "a judge model must be configured (CLAIMCHECK_MODEL, or [judge].model)"
            )
        configured_key = os.environ.get(jc.api_key_env) if jc.api_key_env else None
        api_key = resolve_api_key(jc.base_url, configured_key)

        ledger_file = _ledger_path(self.config.ledger)
        ledger = Ledger(ledger_file)
        self.ledger = ledger

        config_prices = None
        if jc.price_in_per_m is not None and jc.price_out_per_m is not None:
            config_prices = {jc.model: Price(jc.price_in_per_m, jc.price_out_per_m)}
        price_known = config_prices is not None or self._price_override is not None
        price_book = PriceBook(
            base_url=jc.base_url,
            api_key=api_key,
            config_prices=config_prices,
            override=self._price_override,
            client=None if price_known else httpx.Client(),
        )
        self._price_book = price_book
        budget = Budget(max_usd=self.max_usd, ledger=ledger, cap=self.config.ledger_cap_usd)

        return JudgeDetector(
            model=jc.model,
            base_url=jc.base_url,
            api_key=api_key,
            prompt=prompt or jc.prompt or DEFAULT_PROMPT_NAME,
            temperature=jc.temperature,
            max_tokens=jc.max_tokens,
            json_mode=jc.json_mode,
            timeout_s=jc.timeout_s,
            cache=JudgeCache(_cache_dir()) if self.use_cache else None,
            budget=budget,
            price_book=price_book,
            ledger=ledger,
            run_id=self.run_id,
            base_rate=self._base_rate(),
        )

    def _build_detector(self, name: str, prompt: str | Path | None) -> Detector:
        if name == "rules":
            return self._build_rules()
        if name == "classifier":
            return self._build_classifier()
        if name == "judge":
            return self._build_judge(prompt)
        if name == "cascade-offline":
            return CascadeOfflineDetector(
                self._build_rules(), self._build_classifier(), calibrators=self.calibrators
            )
        if name == "cascade":
            return CascadeDetector(
                self._build_rules(), self._build_judge(prompt), calibrators=self.calibrators
            )
        try:
            # Baselines, and any other name registered with
            # `register_detector` (a default-constructible custom
            # detector), for extensibility.
            return get_detector(name)
        except KeyError:
            raise ValueError(f"unknown detector {name!r}") from None

    def _skip_result(self, trace: Trace, view: DetectorView) -> CheckResult:
        return CheckResult(
            schema="claimcheck-result/v1",
            trace_id=trace.trace_id,
            domain=trace.task.domain,
            detector=self.detector_name,
            verdict="skipped",
            p_success=None,
            p_success_raw=None,
            calibrated=False,
            abstain=True,
            abstain_reason="no_success_claim",
            confidence=None,
            reasons=[
                Reason(
                    claim=None,
                    outcome="no_success_claim",
                    step=None,
                    detail="the trace has no success claim",
                )
            ],
            claims=list(view.final_claim.claims),
            detectors=[],
            cost_usd=0.0,
            latency_ms=0.0,
            cached=False,
        )

    def _score_one(self, trace: Trace) -> CheckResult:
        view = resolve_claims(detector_view(trace), self.extractor)
        if not success_claims(view.final_claim.claims, self.non_success_types):
            return self._skip_result(trace, view)

        raw_output = self._detector.score(view)
        if getattr(self._detector, "is_ensemble", False):
            final_output = raw_output
        else:
            p_calibrated = (
                self.calibrators.apply(raw_output.detector, raw_output)
                if self.calibrators is not None
                else None
            )
            final_output = (
                raw_output.model_copy(update={"p_calibrated": p_calibrated})
                if p_calibrated is not None
                else raw_output
            )

        p_used = (
            final_output.p_calibrated
            if final_output.p_calibrated is not None
            else final_output.p_success
        )
        verdict = gate(p_used, final_output.abstain, self.thresholds)

        return CheckResult(
            schema="claimcheck-result/v1",
            trace_id=trace.trace_id,
            domain=trace.task.domain,
            detector=final_output.detector,
            verdict=verdict,
            p_success=p_used,
            p_success_raw=final_output.p_success,
            calibrated=final_output.p_calibrated is not None,
            abstain=final_output.abstain,
            abstain_reason=final_output.abstain_reason,
            confidence=confidence(verdict, p_used),
            reasons=list(final_output.reasons)
            or [
                Reason(
                    claim=None,
                    outcome=final_output.detector,
                    step=None,
                    detail="the detector cited no evidence",
                )
            ],
            claims=list(view.final_claim.claims),
            detectors=[final_output],
            cost_usd=final_output.cost_usd,
            latency_ms=final_output.latency_ms,
            cached=final_output.cached,
        )

    def check(
        self, traces: Iterable[Trace], probes: str | Path | None = None
    ) -> Iterator[CheckResult]:
        """Score `traces`, yielding one `CheckResult` per trace, in order."""
        trace_list = list(traces)
        if probes is not None:
            trace_list, _warnings = merge_probes(trace_list, probes)

        if getattr(self._detector, "concurrent", False):
            workers = max(1, self.concurrency)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                yield from executor.map(self._score_one, trace_list)
        else:
            for trace in trace_list:
                yield self._score_one(trace)
