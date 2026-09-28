"""Claim resolution: structured vs. extracted, the negation guard, success_claims."""

from __future__ import annotations

import pytest
from factory import message, tool_call, trace

from agent_claimcheck.claims import ClaimExtractor, ResolvedClaim, success_claims
from agent_claimcheck.redact import detector_view, resolve_claims
from agent_claimcheck.rules.engine import builtin_packs

PACKS = list(builtin_packs().values())


def _extractor() -> ClaimExtractor:
    return ClaimExtractor(PACKS)


def test_structured_claims_win_over_extraction() -> None:
    t = trace(
        "t1",
        "booking",
        [tool_call(0, "calendar.create_event")],
        text="Booked.",
        claims=[("booked", {"start": "2026-01-01T00:00:00Z"})],
    )
    view = resolve_claims(detector_view(t), _extractor())
    assert view.claims_extracted is False
    assert [c.type for c in view.final_claim.claims] == ["booked"]
    assert view.final_claim.claims[0].source == "structured"


def test_extraction_runs_when_no_structured_claims() -> None:
    t = trace("t2", "booking", [tool_call(0, "calendar.create_event")], text="Booked the slot.")
    view = resolve_claims(detector_view(t), _extractor())
    assert view.claims_extracted is True
    assert [c.type for c in view.final_claim.claims] == ["booked"]
    assert view.final_claim.claims[0].source == "extracted"


def test_negation_guard_drops_the_match() -> None:
    phrases = [
        "I couldn't book the meeting.",
        "Unable to book the meeting.",
        "Failed to book the meeting.",
        "I did not book the meeting.",
        "I haven't booked the meeting.",
        "Booking is not yet confirmed.",
    ]
    for text in phrases:
        claims = _extractor().extract(
            text, domain="booking", steps=[tool_call(0, "calendar.create_event")]
        )
        assert claims == [], text


def _booking_claims(text: str) -> list[ResolvedClaim]:
    return _extractor().extract(
        text, domain="booking", steps=[tool_call(0, "calendar.create_event")]
    )


@pytest.mark.parametrize(
    "text",
    [
        "The meeting wasn't booked.",
        "It was not booked.",
        "It isn't booked yet.",
        "The meeting is not booked.",
        "Both meetings weren't booked.",
        "The meetings were not booked.",
        "The slots aren't booked.",
        "The slots are not booked.",
        "The meeting hasn't been booked.",
        "The meeting has not been booked.",
        "The meetings haven't been booked.",
        "The meetings have not been booked.",
        "The meeting didn't get booked.",
        "The meeting did not get booked.",
        "The meeting wasn\u2019t booked.",
        "The slot was NOT booked.",
        "Sorry, but the requested slot wasn't free, so nothing is booked.",
    ],
)
def test_passive_and_copula_negation_before_the_match_drops_it(text: str) -> None:
    assert _booking_claims(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        "Booked, and nothing is not working.",
        "Booked the slot, and it wasn't easy.",
        "Confirmed. It wasn't easy, but it is done.",
        "The notebook slot is booked.",
        "That was nothing special: booked.",
        "Booked, and the old slot isn't needed any more.",
    ],
)
def test_negation_after_the_match_or_in_another_word_keeps_the_claim(text: str) -> None:
    assert [c.type for c in _booking_claims(text)] == ["booked"], text


def test_not_yet_anywhere_in_the_sentence_drops_the_match() -> None:
    claims = _extractor().extract(
        "Confirmed, not yet final though.",
        domain="booking",
        steps=[tool_call(0, "calendar.create_event")],
    )
    assert claims == []


def test_generic_done_fallback_used_only_when_nothing_else_matched() -> None:
    # No booking/crm/coding pack applies in the browser domain, no tool calls.
    claims = _extractor().extract("Done.", domain="browser", steps=[])
    assert [c.type for c in claims] == ["done"]
    assert claims[0].source == "extracted"


def test_generic_fallback_is_not_used_when_a_domain_pattern_matched() -> None:
    claims = _extractor().extract(
        "Booked the meeting, all set.",
        domain="booking",
        steps=[tool_call(0, "calendar.create_event")],
    )
    assert [c.type for c in claims] == ["booked"]


def test_first_non_negated_match_wins_subject_merged_from_later_matches() -> None:
    # Two "booked" style sentences: the first has no captured fields, but the
    # merge only fills missing subject keys and never overrides.
    claims = _extractor().extract(
        "Scheduled the call. Confirmed.",
        domain="booking",
        steps=[tool_call(0, "calendar.create_event")],
    )
    assert len(claims) == 1
    assert claims[0].type == "booked"
    assert claims[0].span == (0, 9)  # "Scheduled" is the first non-negated match


def test_success_claims_filters_non_success_types() -> None:
    claims = [
        ResolvedClaim(type="booked", subject={}, source="structured"),
        ResolvedClaim(type="failed", subject={}, source="structured"),
        ResolvedClaim(type="blocked", subject={}, source="structured"),
    ]
    assert [c.type for c in success_claims(claims)] == ["booked"]


def test_no_success_claim_gives_an_empty_list() -> None:
    claims = [ResolvedClaim(type="gave_up", subject={}, source="structured")]
    assert success_claims(claims) == []


def test_extraction_has_no_applicable_pack_and_no_generic_pattern_matches() -> None:
    t = trace(
        "t3", "browser", [message(0, "agent", "Still working on it.")], text="Still working on it."
    )
    view = resolve_claims(detector_view(t), _extractor())
    assert view.final_claim.claims == []
    assert view.claims_extracted is True
