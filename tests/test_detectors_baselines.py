"""`trust-agent` and `any-error`: constant baselines, never abstain."""

from __future__ import annotations

from factory import tool_call, tool_result, trace

from agent_claimcheck.detectors.base import get_detector
from agent_claimcheck.detectors.baselines import AnyErrorDetector, TrustAgentDetector
from agent_claimcheck.redact import detector_view


def test_trust_agent_always_verifies() -> None:
    t = trace("t1", "booking", [], text="Booked.", claims=[("booked", {})])
    output = TrustAgentDetector().score(detector_view(t))
    assert output.p_success == 1.0
    assert output.abstain is False


def test_any_error_reports_success_with_no_failed_result() -> None:
    t = trace(
        "t1",
        "booking",
        [tool_call(0, "calendar.create_event"), tool_result(1, "calendar.create_event", ok=True)],
        text="Booked.",
        claims=[("booked", {})],
    )
    output = AnyErrorDetector().score(detector_view(t))
    assert output.p_success == 1.0


def test_any_error_reports_failure_when_any_result_failed() -> None:
    t = trace(
        "t1",
        "booking",
        [
            tool_call(0, "calendar.create_event"),
            tool_result(1, "calendar.create_event", ok=False, error="409: taken"),
        ],
        text="Booked.",
        claims=[("booked", {})],
    )
    output = AnyErrorDetector().score(detector_view(t))
    assert output.p_success == 0.0
    assert output.reasons[0].step == 1


def test_any_error_reports_failure_on_error_without_ok_false() -> None:
    t = trace(
        "t1",
        "booking",
        [
            tool_call(0, "calendar.create_event"),
            tool_result(1, "calendar.create_event", ok=True, error="soft warning"),
        ],
        text="Booked.",
        claims=[("booked", {})],
    )
    output = AnyErrorDetector().score(detector_view(t))
    assert output.p_success == 0.0


def test_both_baselines_registered() -> None:
    assert get_detector("trust-agent").name == "trust-agent"
    assert get_detector("any-error").name == "any-error"
