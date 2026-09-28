"""`cascade-offline` and `cascade`: decision logic, isolated from the real
rules/classifier/judge detectors with small stand-ins, so these tests never
depend on the trained classifier artifact or a network call.
"""

from __future__ import annotations

from typing import Any

from factory import trace

from agent_claimcheck.calibration import Calibrator, CalibratorSet
from agent_claimcheck.detectors.base import DetectorOutput, Reason
from agent_claimcheck.detectors.ensemble import CascadeDetector, CascadeOfflineDetector
from agent_claimcheck.redact import detector_view


class _Stub:
    """A detector stand-in that returns a fixed output, or raises if scored
    when the test expects it never to be called."""

    def __init__(self, name: str, output: DetectorOutput | None = None) -> None:
        self.name = name
        self._output = output
        self.calls = 0

    def score(self, trace: Any) -> DetectorOutput:
        self.calls += 1
        if self._output is None:
            raise AssertionError(f"{self.name} should not have been scored")
        return self._output


def _reason(detail: str) -> list[Reason]:
    return [Reason(claim="booked", outcome="x", step=1, detail=detail)]


def _view() -> Any:
    t = trace("t1", "booking", [], text="Booked.", claims=[("booked", {})])
    return detector_view(t)


def _output(detector: str, p: float, *, abstain: bool = False, reason: str = "x") -> DetectorOutput:
    return DetectorOutput(
        detector=detector,
        p_success=p,
        abstain=abstain,
        reasons=_reason(reason),
        cost_usd=0.01,
        latency_ms=5.0,
    )


def test_cascade_offline_uses_rules_when_contradicted_and_never_calls_classifier() -> None:
    rules = _Stub("rules", _output("rules", 0.03, reason="contradicted"))
    classifier = _Stub("classifier-lr", None)  # would raise if scored
    ensemble = CascadeOfflineDetector(rules, classifier)

    out = ensemble.score(_view())
    assert out.p_success == 0.03
    assert out.details["decided_by"] == "rules"
    assert classifier.calls == 0
    assert out.components == [rules.score(_view())]  # same shape


def test_cascade_offline_defers_to_classifier_when_rules_abstains() -> None:
    rules = _Stub("rules", _output("rules", 0.6, abstain=True, reason="no_rule"))
    classifier = _Stub("classifier-lr", _output("classifier-lr", 0.5))
    ensemble = CascadeOfflineDetector(rules, classifier)

    out = ensemble.score(_view())
    assert out.p_success == 0.5
    assert out.details["decided_by"] == "classifier-lr"
    assert classifier.calls == 1


def test_cascade_offline_defers_to_classifier_on_receipt_only() -> None:
    rules = _Stub("rules", _output("rules", 0.70, reason="receipt_only"))
    classifier = _Stub("classifier-lr", _output("classifier-lr", 0.4))
    ensemble = CascadeOfflineDetector(rules, classifier)

    out = ensemble.score(_view())
    assert out.details["decided_by"] == "classifier-lr"
    assert out.p_success == 0.4


def test_cascade_offline_uses_rules_when_probe_supported() -> None:
    rules = _Stub("rules", _output("rules", 0.97, reason="probe_supported"))
    classifier = _Stub("classifier-lr", None)
    ensemble = CascadeOfflineDetector(rules, classifier)

    out = ensemble.score(_view())
    assert out.details["decided_by"] == "rules"
    assert out.p_success == 0.97


def test_cascade_offline_composes_calibration_for_each_component_it_runs() -> None:
    rules = _Stub("rules", _output("rules", 0.70, reason="receipt_only"))
    classifier = _Stub("classifier-lr", _output("classifier-lr", 0.4))
    calibrators = CalibratorSet(
        version=1,
        fitted_on="test",
        base_rate=0.6,
        calibrators={
            "rules": Calibrator(
                method="platt",
                a=1.0,
                b=0.0,
                clip=1e-6,
                n=10,
                n_pos=5,
                fitted_on="test",
                detector="rules",
            ),
            "classifier-lr": Calibrator(
                method="platt",
                a=2.0,
                b=0.0,
                clip=1e-6,
                n=10,
                n_pos=5,
                fitted_on="test",
                detector="classifier-lr",
            ),
        },
    )
    ensemble = CascadeOfflineDetector(rules, classifier, calibrators=calibrators)
    out = ensemble.score(_view())

    assert out.p_calibrated is not None
    # The deciding (classifier) component's calibrated value, not rules'.
    expected = calibrators.apply("classifier-lr", _output("classifier-lr", 0.4))
    assert out.p_calibrated == expected
    # Both components carry their own calibrated value.
    assert out.components[0].p_calibrated == calibrators.apply("rules", _output("rules", 0.70))
    assert out.components[1].p_calibrated == expected


def test_cascade_offline_cost_and_latency_sum_only_components_actually_run() -> None:
    rules = _Stub("rules", _output("rules", 0.03))  # conclusive: cost 0.01, latency 5.0
    classifier = _Stub("classifier-lr", None)
    ensemble = CascadeOfflineDetector(rules, classifier)
    out = ensemble.score(_view())
    assert out.cost_usd == 0.01
    assert out.latency_ms == 5.0


def test_cascade_uses_rules_when_conclusive_and_never_calls_judge() -> None:
    rules = _Stub("rules", _output("rules", 0.05, reason="unsupported"))
    judge = _Stub("judge:m", None)
    ensemble = CascadeDetector(rules, judge)

    out = ensemble.score(_view())
    assert out.details == {"decided_by": "rules", "sent_to_judge": False}
    assert judge.calls == 0
    assert out.p_success == 0.05


def test_cascade_sends_to_judge_when_rules_inconclusive() -> None:
    rules = _Stub("rules", _output("rules", 0.6, abstain=True, reason="no_rule"))
    judge = _Stub("judge:m", _output("judge:m", 0.8))
    ensemble = CascadeDetector(rules, judge)

    out = ensemble.score(_view())
    assert out.details == {"decided_by": "judge", "sent_to_judge": True}
    assert judge.calls == 1
    assert out.p_success == 0.8
    assert out.cost_usd == 0.02  # rules 0.01 + judge 0.01
    assert out.latency_ms == 10.0


def test_cascade_is_marked_concurrent_and_is_ensemble() -> None:
    assert CascadeDetector.concurrent is True
    assert CascadeDetector.is_ensemble is True
    assert CascadeOfflineDetector.is_ensemble is True
    assert not hasattr(CascadeOfflineDetector, "concurrent")
