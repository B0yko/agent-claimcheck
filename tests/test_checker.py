"""`Checker`: the pipeline from traces to `CheckResult`s."""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from factory import probe, tool_call, tool_result, trace

from agent_claimcheck.calibration import Calibrator, CalibratorSet
from agent_claimcheck.checker import Checker, CheckResult, dump_result
from agent_claimcheck.config import ClassifierConfig, Config, JudgeConfig, load_config
from agent_claimcheck.detectors.base import DetectorOutput, Reason, register_detector
from agent_claimcheck.gate import Thresholds, UnknownPriceError
from agent_claimcheck.ledger import Price


def _booked_trace(trace_id: str, *, with_probe: bool = True) -> object:
    steps = [
        tool_call(
            0,
            "calendar.create_event",
            {
                "start": "2026-01-01T01:00:00+00:00",
                "duration_min": 30,
                "attendees": ["a@example.test"],
            },
        ),
        tool_result(
            1,
            "calendar.create_event",
            ok=True,
            output={
                "event_id": "e1",
                "start": "2026-01-01T01:00:00+00:00",
                "attendees": ["a@example.test"],
            },
        ),
    ]
    if with_probe:
        steps.append(
            probe(
                2,
                "calendar.get_event",
                ok=True,
                output={
                    "status": "confirmed",
                    "start": "2026-01-01T01:00:00+00:00",
                    "attendees": ["a@example.test"],
                },
            )
        )
    return trace(
        trace_id,
        "booking",
        steps,
        text="Booked it.",
        claims=[
            (
                "booked",
                {
                    "start": "2026-01-01T01:00:00+00:00",
                    "timezone": "UTC",
                    "attendee_email": "a@example.test",
                    "duration_min": 30,
                },
            )
        ],
    )


def _no_claim_trace(trace_id: str) -> object:
    return trace(trace_id, "booking", [], text="I couldn't book it.", claims=[("blocked", {})])


def test_check_yields_results_in_input_order() -> None:
    checker = Checker("rules")
    traces = [_booked_trace("t3"), _booked_trace("t1"), _no_claim_trace("t2")]
    results = list(checker.check(traces))
    assert [r.trace_id for r in results] == ["t3", "t1", "t2"]


def test_skipped_result_shape() -> None:
    checker = Checker("rules")
    [result] = list(checker.check([_no_claim_trace("t1")]))
    assert result.verdict == "skipped"
    assert result.p_success is None
    assert result.p_success_raw is None
    assert result.abstain is True
    assert result.abstain_reason == "no_success_claim"
    assert result.reasons[0].outcome == "no_success_claim"
    assert result.detectors == []
    dump_result(result)  # validates against the schema


def test_rules_probe_supported_verifies() -> None:
    checker = Checker("rules")
    [result] = list(checker.check([_booked_trace("t1")]))
    assert result.verdict == "verified"
    assert result.p_success_raw == pytest.approx(0.97)
    dump_result(result)


def test_gate_uses_calibrated_p_when_a_calibrator_exists() -> None:
    # sigmoid(0.05 * logit(0.97) - 1.0) ~= 0.30: pulls the raw 0.97
    # (verified on its own) down into the unverifiable band.
    calibrators = CalibratorSet(
        version=1,
        fitted_on="test",
        base_rate=0.6,
        calibrators={
            "rules": Calibrator(
                method="platt",
                a=0.05,
                b=-1.0,
                clip=1e-6,
                n=10,
                n_pos=5,
                fitted_on="test",
                detector="rules",
            )
        },
    )
    checker = Checker("rules", calibration=calibrators)
    [result] = list(checker.check([_booked_trace("t1")]))
    assert result.p_success_raw == pytest.approx(0.97)
    assert result.calibrated is True
    assert result.p_success != pytest.approx(0.97)
    assert result.verdict == "unverifiable"  # calibrated p pulled below 0.80


def test_calibrators_builtin_flag() -> None:
    assert Checker("rules").calibrators_builtin is True
    calibrators = CalibratorSet(version=1, fitted_on="x", base_rate=0.6, calibrators={})
    assert Checker("rules", calibration=calibrators).calibrators_builtin is False


def test_custom_thresholds_change_the_gate() -> None:
    checker = Checker("rules", thresholds=Thresholds(verified=0.99, false_success=0.01))
    [result] = list(checker.check([_booked_trace("t1")]))
    assert result.verdict == "unverifiable"  # 0.97 raw < 0.99 verified threshold


def test_probes_merge_changes_a_result() -> None:
    checker = Checker("rules")
    base = list(checker.check([_booked_trace("t1", with_probe=False)]))[0]
    assert base.verdict == "unverifiable"  # receipt_only, 0.70

    probes_path = _write_probes_file("t1")
    with_probe = list(
        Checker("rules").check([_booked_trace("t1", with_probe=False)], probes=probes_path)
    )[0]
    assert with_probe.verdict == "verified"


def _write_probes_file(trace_id: str) -> Path:
    line = {
        "trace_id": trace_id,
        "name": "calendar.get_event",
        "ok": True,
        "output": {
            "status": "confirmed",
            "start": "2026-01-01T01:00:00+00:00",
            "attendees": ["a@example.test"],
        },
        "ts": "2026-01-01T00:05:00+00:00",
    }
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with open(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return Path(path)


def test_unknown_detector_raises() -> None:
    with pytest.raises(ValueError, match="unknown detector"):
        Checker("not-a-real-detector")


def test_from_config_reads_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "claimcheck.toml"
    config_path.write_text("[gate]\nverified = 0.99\nfalse_success = 0.01\n", encoding="utf-8")
    checker = Checker.from_config(config_path, detector="rules")
    [result] = list(checker.check([_booked_trace("t1")]))
    assert result.verdict == "unverifiable"  # 0.97 < 0.99


class _ConcurrentEchoDetector:
    """Records which thread scored each trace, to prove `Checker.check`
    uses a thread pool for a `concurrent = True` detector."""

    name = "concurrent-echo"
    concurrent = True

    def __init__(self) -> None:
        self.threads: set[int] = set()
        self.lock = threading.Lock()

    def score(self, trace: object) -> DetectorOutput:
        with self.lock:
            self.threads.add(threading.get_ident())
        return DetectorOutput(
            detector=self.name,
            p_success=0.9,
            abstain=False,
            reasons=[Reason(claim=None, outcome="x", step=None, detail="x")],
        )


def test_concurrent_detector_runs_on_a_thread_pool_but_preserves_order() -> None:
    echo = _ConcurrentEchoDetector()
    register_detector("concurrent-echo", lambda: echo)
    checker = Checker("concurrent-echo", concurrency=4)
    traces = [_booked_trace(f"t{i}") for i in range(8)]
    results = list(checker.check(traces))
    assert [r.trace_id for r in results] == [f"t{i}" for i in range(8)]
    assert all(r.verdict == "verified" for r in results)


def test_check_result_round_trips_through_dump_result() -> None:
    checker = Checker("rules")
    [result] = list(checker.check([_booked_trace("t1")]))
    assert isinstance(result, CheckResult)
    line = dump_result(result)
    assert '"schema":"claimcheck-result/v1"' in line


# ------------------------------------------------------------- judge pricing --


def _empty_listing(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every `Checker`-built HTTP client see an empty `/models` listing."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(200, json={"data": []})

    real_client = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        kwargs["transport"] = kwargs.get("transport") or httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr("agent_claimcheck.checker.httpx.Client", client)
    return requested


def _judge_config(**judge: object) -> Config:
    return Config(judge=JudgeConfig(model="test/fake-model", **judge))  # type: ignore[arg-type]


def test_require_price_raises_when_the_judge_model_has_no_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _empty_listing(monkeypatch)
    checker = Checker("judge", config=_judge_config())
    with pytest.raises(UnknownPriceError) as info:
        checker.require_price()
    assert "no price known for judge model 'test/fake-model'" in info.value.args[0]
    assert requested  # the listing was consulted


def test_require_price_passes_with_an_explicit_price_or_a_configured_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _empty_listing(monkeypatch)
    Checker("judge", config=_judge_config(), price=Price(1.0, 2.0)).require_price()
    Checker(
        "cascade", config=_judge_config(price_in_per_m=1.0, price_out_per_m=2.0)
    ).require_price()
    assert requested == []  # a known price never touches the network


def test_an_explicit_price_beats_the_configured_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _empty_listing(monkeypatch)
    checker = Checker(
        "judge",
        config=_judge_config(price_in_per_m=1.0, price_out_per_m=1.0),
        price=Price(5.0, 6.0),
    )
    assert checker._price_book is not None
    assert checker._price_book.list_price("test/fake-model") == Price(5.0, 6.0)


def test_require_price_is_a_no_op_for_detectors_without_a_judge() -> None:
    Checker("rules").require_price()
    Checker("cascade-offline").require_price()


def test_judge_detector_still_abstains_per_trace_without_a_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _empty_listing(monkeypatch)
    monkeypatch.setenv("CLAIMCHECK_API_KEY", "test-key")
    checker = Checker("judge", config=_judge_config(), use_cache=False)
    [result] = list(checker.check([_booked_trace("t1")]))
    assert result.abstain_reason == "unknown_price"
    assert result.verdict == "unverifiable"


# ------------------------------------------------------ configured calibration --


def _rules_calibrators(a: float, b: float) -> CalibratorSet:
    return CalibratorSet(
        version=1,
        fitted_on="test",
        base_rate=0.6,
        calibrators={
            "rules": Calibrator(
                method="platt",
                a=a,
                b=b,
                clip=1e-6,
                n=10,
                n_pos=5,
                fitted_on="test",
                detector="rules",
            )
        },
    )


def test_config_calibration_is_used_when_no_argument_is_given(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    _rules_calibrators(0.05, -1.0).save(path)
    config = Config(classifier=ClassifierConfig(calibration=str(path)))

    checker = Checker("rules", config=config)
    assert checker.calibrators_builtin is False
    [result] = list(checker.check([_booked_trace("t1")]))
    # The configured set pulls the raw 0.97 down (sigmoid(0.05 * logit(0.97) - 1.0) ~ 0.30);
    # the built-in set would leave it verified.
    assert result.p_success_raw == pytest.approx(0.97)
    assert result.calibrated is True
    assert result.p_success == pytest.approx(0.30, abs=0.01)
    assert result.verdict == "unverifiable"


def test_config_calibration_is_read_from_the_toml_file(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    _rules_calibrators(0.05, -1.0).save(path)
    toml = tmp_path / "claimcheck.toml"
    toml.write_text(f'[classifier]\ncalibration = "{path}"\n', encoding="utf-8")

    checker = Checker.from_config(toml, detector="rules")
    assert checker.calibrators_builtin is False
    assert checker.calibrators is not None
    assert checker.calibrators.fitted_on == "test"


def test_calibration_argument_beats_the_config(tmp_path: Path) -> None:
    configured = tmp_path / "configured.json"
    _rules_calibrators(0.05, -1.0).save(configured)
    config = Config(classifier=ClassifierConfig(calibration=str(configured)))

    explicit = _rules_calibrators(1.0, 0.0)
    checker = Checker("rules", config=config, calibration=explicit)
    assert checker.calibrators is explicit
    [result] = list(checker.check([_booked_trace("t1")]))
    assert result.p_success == pytest.approx(0.97, abs=0.001)  # a=1, b=0 is the identity
    assert result.verdict == "verified"


def test_builtin_calibrators_when_neither_argument_nor_config_names_one() -> None:
    assert Checker("rules", config=load_config(None)).calibrators_builtin is True
    assert Checker("rules", config=Config()).calibrators_builtin is True
