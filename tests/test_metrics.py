"""`metrics.py`: every function against a hand-computed value."""

from __future__ import annotations

import pytest
from factory import tool_call, tool_result, trace

from agent_claimcheck import metrics
from agent_claimcheck.checker import CheckResult
from agent_claimcheck.detectors.base import Reason
from agent_claimcheck.schema import GroundTruth


def test_auroc_perfect_separation() -> None:
    assert metrics.auroc([1, 1, 0, 0], [0.9, 0.8, 0.4, 0.3]) == pytest.approx(1.0)


def test_auroc_worst_separation() -> None:
    assert metrics.auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)


def test_auroc_ties_averaged() -> None:
    # positives score 3, 2; negatives score 2, 1 -> one tie pair counts 0.5.
    assert metrics.auroc([1, 1, 0, 0], [3, 2, 2, 1]) == pytest.approx(0.875)


def test_auroc_chance_with_a_coin_flip_tie() -> None:
    assert metrics.auroc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)


def test_auroc_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        metrics.auroc([1, 1], [0.9, 0.8])


def test_ece_hand_computed() -> None:
    p = [0.05, 0.05, 0.95, 0.95]
    y = [0, 0, 1, 1]
    assert metrics.ece(p, y) == pytest.approx(0.05)


def test_ece_perfect_calibration_is_zero() -> None:
    p = [0.0, 1.0]
    y = [0, 1]
    assert metrics.ece(p, y) == pytest.approx(0.0)


def test_ece_empty_is_zero() -> None:
    assert metrics.ece([], []) == 0.0


def test_brier_hand_computed() -> None:
    p = [0.05, 0.05, 0.95, 0.95]
    y = [0, 0, 1, 1]
    assert metrics.brier(p, y) == pytest.approx(0.0025)


def test_brier_empty_is_zero() -> None:
    assert metrics.brier([], []) == 0.0


def test_extremes_share_hand_computed() -> None:
    assert metrics.extremes_share([0.01, 0.5, 0.99, 0.5]) == pytest.approx(0.5)


def test_extremes_share_respects_bounds() -> None:
    assert metrics.extremes_share([0.05, 0.95]) == pytest.approx(0.0)  # inclusive edges excluded


def test_bootstrap_ci_is_deterministic() -> None:
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    s = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    lo1, hi1 = metrics.bootstrap_ci(y, s, n_resamples=200, seed=0)
    lo2, hi2 = metrics.bootstrap_ci(y, s, n_resamples=200, seed=0)
    assert (lo1, hi1) == (lo2, hi2)
    assert 0.0 <= lo1 <= hi1 <= 1.0


def test_bootstrap_ci_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        metrics.bootstrap_ci([1, 1], [0.9, 0.8])


def test_decision_stats_hand_computed() -> None:
    verdicts = ["false_success", "verified", "unverifiable", "verified"]
    labels = [1, 0, 1, 0]
    stats = metrics.decision_stats(verdicts, labels)
    assert stats["n"] == 4
    assert stats["n_success"] == 2
    assert stats["n_failure"] == 2
    assert stats["coverage"] == pytest.approx(0.75)
    assert stats["accuracy_on_decided"] == pytest.approx(1.0)
    assert stats["caught"] == 1
    assert stats["missed"] == 0
    assert stats["false_alarms"] == 0
    assert stats["sent_to_review"] == 1
    assert stats["confusion"] == {
        "success": {"verified": 2, "false_success": 0, "unverifiable": 0},
        "failure": {"verified": 0, "false_success": 1, "unverifiable": 1},
    }


def test_decision_stats_missed_and_false_alarm() -> None:
    # A false success marked verified (missed) and a genuine success marked
    # false_success (a false alarm).
    stats = metrics.decision_stats(["verified", "false_success"], [1, 0])
    assert stats["missed"] == 1
    assert stats["false_alarms"] == 1
    assert stats["caught"] == 0
    assert stats["accuracy_on_decided"] == pytest.approx(0.0)


def _result(
    trace_id: str, domain: str, verdict: str, p: float | None, p_raw: float | None
) -> CheckResult:
    return CheckResult(
        schema="claimcheck-result/v1",  # type: ignore[call-arg]
        trace_id=trace_id,
        domain=domain,  # type: ignore[arg-type]
        detector="rules",
        verdict=verdict,  # type: ignore[arg-type]
        p_success=p,
        p_success_raw=p_raw,
        calibrated=False,
        abstain=False,
        abstain_reason=None,
        confidence=None,
        reasons=[Reason(claim=None, outcome="x", step=None, detail="x")],
        claims=[],
        detectors=[],
        cost_usd=0.0,
        latency_ms=0.0,
        cached=False,
    )


def _labelled_trace(trace_id: str, outcome: str) -> object:
    t = trace(
        trace_id,
        "booking",
        [tool_call(0, "calendar.create_event"), tool_result(1, "calendar.create_event")],
    )
    return t.model_copy(
        update={"ground_truth": GroundTruth(outcome=outcome, checked_by="state_probe")}
    )  # type: ignore[arg-type]


def test_evaluate_excludes_skipped_and_unlabelled() -> None:
    results = [
        _result("t1", "booking", "verified", 0.9, 0.9),
        _result("t2", "booking", "false_success", 0.05, 0.05),
        _result("t3", "booking", "skipped", None, None),
        _result("t4", "booking", "unverifiable", 0.5, 0.5),
    ]
    traces = [
        _labelled_trace("t1", "success"),
        _labelled_trace("t2", "failure"),
        _labelled_trace("t3", "success"),
        _labelled_trace("t4", "unknown"),
    ]
    evaluation = metrics.evaluate(results, traces)  # type: ignore[arg-type]
    assert evaluation["n"] == 2  # t3 skipped, t4 has no usable ground truth
    assert evaluation["n_success"] == 1
    assert evaluation["n_failure"] == 1
    assert evaluation["auroc"]["value"] == pytest.approx(1.0)


def test_evaluate_empty_when_nothing_scorable() -> None:
    evaluation = metrics.evaluate([], [])
    assert evaluation["n"] == 0
    assert evaluation["auroc"] is None
