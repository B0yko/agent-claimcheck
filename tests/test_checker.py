"""`Checker`: the pipeline from traces to `CheckResult`s."""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

import pytest
from factory import probe, tool_call, tool_result, trace

from agent_claimcheck.calibration import Calibrator, CalibratorSet
from agent_claimcheck.checker import Checker, CheckResult, dump_result
from agent_claimcheck.detectors.base import DetectorOutput, Reason, register_detector
from agent_claimcheck.gate import Thresholds


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
