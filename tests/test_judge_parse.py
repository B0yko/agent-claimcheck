"""Parsing the judge's JSON answer."""

from __future__ import annotations

import json

import pytest

from agent_claimcheck.judge.parse import ParseError, parse_judgment

VALID = {
    "p_success": 0.9,
    "failure_kind": "none",
    "evidence_steps": [0, 1],
    "rationale": "the receipt and probe both confirm the booking",
}


def test_parses_bare_json_object() -> None:
    parsed = parse_judgment(json.dumps(VALID), valid_steps={0, 1, 2})
    assert parsed.p_success == 0.9
    assert parsed.failure_kind == "none"
    assert parsed.evidence_steps == [0, 1]
    assert parsed.rationale_truncated is False
    assert parsed.invalid_citation is False


def test_parses_fenced_json() -> None:
    content = "```json\n" + json.dumps(VALID) + "\n```"
    parsed = parse_judgment(content, valid_steps={0, 1})
    assert parsed.p_success == 0.9


def test_parses_fenced_json_without_language_tag() -> None:
    content = "```\n" + json.dumps(VALID) + "\n```"
    parsed = parse_judgment(content, valid_steps={0, 1})
    assert parsed.p_success == 0.9


def test_claims_key_tolerated() -> None:
    obj = {**VALID, "claims": [{"claim": "booked", "supporting_steps": [0]}]}
    parsed = parse_judgment(json.dumps(obj), valid_steps={0, 1})
    assert parsed.claims == [{"claim": "booked", "supporting_steps": [0]}]


def test_unparseable_text_abstains_parse_error() -> None:
    with pytest.raises(ParseError):
        parse_judgment("not json at all", valid_steps=set())


def test_missing_key_raises_parse_error() -> None:
    obj = dict(VALID)
    del obj["rationale"]
    with pytest.raises(ParseError):
        parse_judgment(json.dumps(obj), valid_steps={0, 1})


def test_p_success_out_of_range_raises_parse_error() -> None:
    obj = {**VALID, "p_success": 1.5}
    with pytest.raises(ParseError):
        parse_judgment(json.dumps(obj), valid_steps={0, 1})


def test_p_success_bool_rejected() -> None:
    obj = {**VALID, "p_success": True}
    with pytest.raises(ParseError):
        parse_judgment(json.dumps(obj), valid_steps={0, 1})


def test_unknown_failure_kind_raises_parse_error() -> None:
    obj = {**VALID, "failure_kind": "made_up_kind"}
    with pytest.raises(ParseError):
        parse_judgment(json.dumps(obj), valid_steps={0, 1})


def test_evidence_steps_bool_in_list_rejected() -> None:
    obj = {**VALID, "evidence_steps": [0, True]}
    with pytest.raises(ParseError):
        parse_judgment(json.dumps(obj), valid_steps={0, 1})


def test_evidence_steps_not_a_list_rejected() -> None:
    obj = {**VALID, "evidence_steps": "0,1"}
    with pytest.raises(ParseError):
        parse_judgment(json.dumps(obj), valid_steps={0, 1})


def test_top_level_array_rejected() -> None:
    with pytest.raises(ParseError):
        parse_judgment(json.dumps([VALID]), valid_steps={0, 1})


def test_invalid_citation_flagged_not_abstain() -> None:
    obj = {**VALID, "evidence_steps": [0, 99]}
    parsed = parse_judgment(json.dumps(obj), valid_steps={0, 1})
    assert parsed.invalid_citation is True
    assert parsed.p_success == 0.9


def test_rationale_truncated_at_300_chars() -> None:
    obj = {**VALID, "rationale": "x" * 400}
    parsed = parse_judgment(json.dumps(obj), valid_steps={0, 1})
    assert parsed.rationale_truncated is True
    assert len(parsed.rationale) == 300


def test_rationale_within_limit_not_truncated() -> None:
    obj = {**VALID, "rationale": "x" * 300}
    parsed = parse_judgment(json.dumps(obj), valid_steps={0, 1})
    assert parsed.rationale_truncated is False
    assert len(parsed.rationale) == 300
