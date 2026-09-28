"""The offline bench run: every non-judge detector, scored once, recorded.

`run_offline` never makes a network call. It retrains `classifier-lr` on the
train split (asserting the result matches the shipped artifact, so a silent
drift between the packaged model and its own training code would fail
loudly), fits the `rules` and `classifier-lr` calibrators the same way
`train --calibrate rules` does, fits three leave-one-domain-out classifiers,
times every offline detector, and writes the recorded files a report can
later read back with no fitting and no timing of its own.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from agent_claimcheck.calibration import CalibratorSet, fit_calibrator
from agent_claimcheck.claims import ClaimExtractor, success_claims
from agent_claimcheck.detectors.base import Detector, DetectorOutput
from agent_claimcheck.detectors.baselines import AnyErrorDetector, TrustAgentDetector
from agent_claimcheck.detectors.classifier import (
    ClassifierDetector,
    load_artifact,
    oof_predictions,
    train_lr,
)
from agent_claimcheck.detectors.rules import RulesDetector
from agent_claimcheck.redact import DetectorView, detector_view, resolve_claims
from agent_claimcheck.rules.engine import OUTCOME_ORDER, builtin_packs
from agent_claimcheck.schema import Trace

#: The three domains the benchmark generator produces; leave-one-domain-out
#: trains on the other two's train traces and scores this one's test split.
LODO_DOMAINS: tuple[str, ...] = ("booking", "crm", "coding")

#: Absolute tolerance for comparing the freshly retrained classifier against
#: the shipped `models/lr-v1.json` artifact.
ARTIFACT_TOLERANCE = 1e-6

#: `run_offline`'s own file names, all written under its `out_dir`.
PREDICTIONS_FILE = "offline-predictions.jsonl"
TIMINGS_FILE = "offline-timings.jsonl"
LODO_MODELS_FILE = "lodo-models.json"
CALIBRATION_FILE = "calibration.json"
RUN_FILE = "run.json"


def _dump_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _dump_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", "utf-8")


@dataclass(frozen=True)
class _Prepared:
    """One trace, projected and resolved once, kept alongside its split."""

    trace: Trace
    view: DetectorView
    split: str
    #: 1 = ground-truth success, 0 = failure (the classifier's own target
    #: convention; metrics.py's opposite "positive = failure" convention is
    #: applied by the report, not here).
    success: int


def _prepare(traces: Sequence[Trace], split: str, extractor: ClaimExtractor) -> list[_Prepared]:
    prepared: list[_Prepared] = []
    for t in traces:
        view = resolve_claims(detector_view(t), extractor)
        if not success_claims(view.final_claim.claims):
            continue
        gt = t.ground_truth
        if gt is None or gt.outcome == "unknown":
            continue
        prepared.append(_Prepared(t, view, split, 1 if gt.outcome == "success" else 0))
    return prepared


def _assert_matches_shipped(retrained: dict[str, Any], shipped: dict[str, Any]) -> None:
    for key in ("means", "scales", "coef"):
        a, b = retrained[key], shipped[key]
        if len(a) != len(b) or any(
            abs(x - y) > ARTIFACT_TOLERANCE for x, y in zip(a, b, strict=True)
        ):
            raise AssertionError(
                f"retrained classifier-lr {key!r} does not match the shipped lr-v1.json artifact"
            )
    if abs(retrained["intercept"] - shipped["intercept"]) > ARTIFACT_TOLERANCE:
        raise AssertionError(
            "retrained classifier-lr intercept does not match the shipped lr-v1.json artifact"
        )
    if retrained["C"] != shipped["C"]:
        raise AssertionError(
            "retrained classifier-lr C does not match the shipped lr-v1.json artifact"
        )


def _rule_outcome(output: DetectorOutput) -> str | None:
    claims_detail = output.details.get("claims")
    if not claims_detail:
        return None
    outcome: str = min((c["outcome"] for c in claims_detail), key=OUTCOME_ORDER.index)
    return outcome


def _ground_truth_fields(trace: Trace) -> dict[str, Any]:
    gt = trace.ground_truth
    assert gt is not None
    details = gt.details
    return {
        "outcome": gt.outcome,
        "injection": details.get("injection", "none"),
        "evidence": details.get("evidence"),
        "variant": details.get("variant"),
    }


def _prediction_row(
    detector_key: str,
    item: _Prepared,
    output: DetectorOutput,
    p_cal: float | None,
) -> dict[str, Any]:
    return {
        "detector": detector_key,
        "trace_id": item.trace.trace_id,
        "split": item.split,
        "domain": item.trace.task.domain,
        **_ground_truth_fields(item.trace),
        "p_raw": output.p_success,
        "p_cal": p_cal,
        "abstain": output.abstain,
        "abstain_reason": output.abstain_reason,
        "rule_outcome": _rule_outcome(output) if detector_key == "rules" else None,
    }


def _score_timed(
    detector: Detector, items: Sequence[_Prepared]
) -> list[tuple[DetectorOutput, float]]:
    results: list[tuple[DetectorOutput, float]] = []
    for item in items:
        start = perf_counter()
        output = detector.score(item.view)
        elapsed_ms = (perf_counter() - start) * 1000.0
        results.append((output, elapsed_ms))
    return results


def run_offline(
    train: Sequence[Trace],
    test: Sequence[Trace],
    out_dir: str | Path,
    *,
    seed: int = 0,
    command: str = "agent-claimcheck bench --offline",
    date: str = "",
    hardware: str = "unspecified",
    package_version: str = "",
    dataset_sha256: str = "",
) -> None:
    """Score every offline detector on `train` + `test`, write the recorded
    files `report.build_report` later reads back.

    `train`/`test` are meant to be the packaged `bench:train`/`bench:test`
    splits: the classifier retrained here is asserted to match the shipped
    `models/lr-v1.json` artifact byte-for-byte within a tight tolerance,
    which only holds for that exact split.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    extractor = ClaimExtractor(list(builtin_packs().values()))
    train_items = _prepare(train, "train", extractor)
    test_items = _prepare(test, "test", extractor)
    all_items = [*train_items, *test_items]

    train_views = [i.view for i in train_items]
    train_labels = [i.success for i in train_items]
    train_domains = [i.trace.task.domain for i in train_items]

    shipped_artifact = load_artifact()
    retrained = train_lr(train_views, train_labels, train_domains, seed=seed)
    _assert_matches_shipped(retrained, shipped_artifact)

    detectors: list[tuple[str, Detector]] = [
        ("trust-agent", TrustAgentDetector()),
        ("any-error", AnyErrorDetector()),
        ("rules", RulesDetector()),
        ("classifier-lr", ClassifierDetector(shipped_artifact, guard=True)),
    ]

    raw_by_detector: dict[str, dict[str, DetectorOutput]] = {}
    timings: list[dict[str, Any]] = []
    for key, detector in detectors:
        timed = _score_timed(detector, all_items)
        raw_by_detector[key] = {}
        for item, (output, latency_ms) in zip(all_items, timed, strict=True):
            raw_by_detector[key][item.trace.trace_id] = output
            timings.append(
                {"detector": key, "trace_id": item.trace.trace_id, "latency_ms": latency_ms}
            )

    # --- calibrators: rules on its own non-abstaining train raw scores,
    # classifier-lr on out-of-fold train predictions (never in-sample). ---
    rules_train_p: list[float] = []
    rules_train_labels: list[int] = []
    for item in train_items:
        output = raw_by_detector["rules"][item.trace.trace_id]
        if not output.abstain:
            rules_train_p.append(output.p_success)
            rules_train_labels.append(item.success)

    calibrators: dict[str, Any] = {}
    if rules_train_p:
        calibrators["rules"] = fit_calibrator(
            "rules", rules_train_p, rules_train_labels, fitted_on="bench:train"
        )

    oof = oof_predictions(train_views, train_labels, seed=seed)
    calibrators["classifier-lr"] = fit_calibrator(
        "classifier-lr", oof, train_labels, fitted_on="bench:train"
    )

    calibrator_set = CalibratorSet(
        version=1,
        fitted_on="bench:train",
        base_rate=sum(train_labels) / len(train_labels),
        calibrators=calibrators,
    )
    calibrator_set.save(out_path / CALIBRATION_FILE)

    predictions: list[dict[str, Any]] = []
    for key, _detector in detectors:
        for item in all_items:
            output = raw_by_detector[key][item.trace.trace_id]
            p_cal = calibrator_set.apply(key, output)
            predictions.append(_prediction_row(key, item, output, p_cal))

    # --- classifier-lr:oof, train only, no calibration (it IS the fit input) ---
    for item, oof_p in zip(train_items, oof, strict=True):
        fake_output = DetectorOutput(detector="classifier-lr:oof", p_success=oof_p, abstain=False)
        predictions.append(_prediction_row("classifier-lr:oof", item, fake_output, None))

    # --- leave-one-domain-out ---
    lodo_models: dict[str, Any] = {}
    for held_domain in LODO_DOMAINS:
        other_items = [i for i in train_items if i.trace.task.domain != held_domain]
        lodo_artifact = train_lr(
            [i.view for i in other_items],
            [i.success for i in other_items],
            [i.trace.task.domain for i in other_items],
            seed=seed,
        )
        lodo_models[held_domain] = lodo_artifact
        lodo_detector = ClassifierDetector(lodo_artifact, guard=False)
        lodo_key = f"classifier-lr:lodo:{held_domain}"
        for item in test_items:
            if item.trace.task.domain != held_domain:
                continue
            output = lodo_detector.score(item.view)
            predictions.append(_prediction_row(lodo_key, item, output, None))

    _dump_jsonl(predictions, out_path / PREDICTIONS_FILE)
    _dump_jsonl(timings, out_path / TIMINGS_FILE)
    _dump_json(lodo_models, out_path / LODO_MODELS_FILE)

    run_meta = {
        "command": command,
        "date": date,
        "hardware": hardware,
        "concurrency": 1,
        "package_version": package_version,
        "dataset_sha256": dataset_sha256,
        "judges": [],
        "total_spend_usd": 0.0,
    }
    _dump_json(run_meta, out_path / RUN_FILE)
