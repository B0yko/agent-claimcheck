"""The rules detector: aggregation, abstention, and the browser example."""

from __future__ import annotations

from factory import probe, tool_call, tool_result, trace

from agent_claimcheck.claims import ClaimExtractor, success_claims
from agent_claimcheck.detectors.rules import RulesDetector
from agent_claimcheck.redact import detector_view, resolve_claims
from agent_claimcheck.rules.engine import builtin_packs
from agent_claimcheck.schema import load_traces

PACKS = list(builtin_packs().values())


def test_abstains_with_base_rate_when_no_rule_applies() -> None:
    t = trace("t1", "browser", [tool_call(0, "browser.click")], claims=[("done", {})])
    detector = RulesDetector(packs=PACKS, base_rate=0.6)
    output = detector.score(detector_view(t))
    assert output.abstain is True
    assert output.abstain_reason == "no_rule"
    assert output.p_success == 0.6


def test_scores_the_worst_outcome_across_claims() -> None:
    t = trace(
        "t2",
        "booking",
        [
            tool_call(
                0,
                "calendar.create_event",
                args={"duration_min": 30, "attendees": ["a@example.test"]},
            ),
            tool_result(
                1,
                "calendar.create_event",
                output={
                    "event_id": "e1",
                    "start": "2026-01-01T00:00:00Z",
                    "attendees": ["a@example.test"],
                },
            ),
            tool_call(2, "email.send_invite", args={"to": ["a@example.test"]}),
            tool_result(
                3,
                "email.send_invite",
                ok=False,
                error="404: unknown event",
                output={},
            ),
        ],
        claims=[
            (
                "booked",
                {
                    "start": "2026-01-01T00:00:00Z",
                    "duration_min": 30,
                    "attendee_email": "a@example.test",
                },
            ),
            ("invite_sent", {"attendee_email": "a@example.test"}),
        ],
    )
    detector = RulesDetector(packs=PACKS)
    output = detector.score(detector_view(t))
    # "invite_sent" is contradicted (its result failed); that is worse than
    # "booked"'s receipt_only, so the trace score reflects the worst claim.
    assert output.p_success == 0.03
    assert output.abstain is False
    outcomes = {r.claim: r.outcome for r in output.reasons}
    assert outcomes["invite_sent"] == "contradicted"
    assert outcomes["booked"] == "receipt_only"


def test_details_carries_one_claim_entry_per_success_claim() -> None:
    t = trace(
        "t3",
        "booking",
        [tool_call(0, "calendar.create_event")],
        claims=[("booked", {}), ("failed", {})],
    )
    detector = RulesDetector(packs=PACKS)
    output = detector.score(detector_view(t))
    claim_types = [c["claim"] for c in output.details["claims"]]
    # "failed" is a non-success type and is excluded before scoring.
    assert claim_types == ["booked"]


def test_reasons_are_one_line_each() -> None:
    t = trace("t4", "booking", [tool_call(0, "calendar.create_event")], claims=[("booked", {})])
    detector = RulesDetector(packs=PACKS)
    output = detector.score(detector_view(t))
    for reason in output.reasons:
        assert len(reason.detail) <= 200
        assert "\n" not in reason.detail


def test_example_browser_rules_abstain_for_every_trace_with_a_success_claim() -> None:
    traces = load_traces("example:browser")
    detector = RulesDetector(packs=PACKS)
    checked = 0
    for t in traces:
        view = detector_view(t)
        if not success_claims(view.final_claim.claims):
            continue
        checked += 1
        output = detector.score(view)
        assert output.abstain is True
        assert output.abstain_reason == "no_rule"
    assert checked > 0


# --- crm.yaml stage_changed ignores "stage" used as an ordinary word ---


def test_bare_stage_word_does_not_spuriously_downgrade_a_real_update() -> None:
    # "stage" as ordinary language ("first stage of onboarding") must not be
    # extracted as a spurious stage_changed claim that drags a real,
    # probe-confirmed "updated" claim down to false_success.
    t = trace(
        "t5",
        "crm",
        [
            tool_call(0, "crm.update_contact", args={"record_id": "c1"}),
            tool_result(
                1,
                "crm.update_contact",
                output={"record_id": "c1", "updated_fields": ["notes"]},
            ),
            probe(2, "crm.read_record", output={"fields": {"notes": "onboarded"}}),
        ],
        text=("Updated the contact's notes. This closes out the first stage of onboarding."),
    )
    view = resolve_claims(detector_view(t), ClaimExtractor(PACKS))
    detector = RulesDetector(packs=PACKS)
    output = detector.score(view)
    # The extracted claim has no subject, so every value check is skipped and
    # the claim is capped at receipt_only; the bare word "stage" must not add
    # an unsupported stage_changed claim on top.
    assert output.p_success == 0.70
    assert {c.type for c in view.final_claim.claims} == {"updated"}
