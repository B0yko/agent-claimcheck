"""The declarative rule engine: loading, checks, applicability, outcomes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from factory import probe, tool_call, tool_result

from agent_claimcheck.rules.engine import (
    Check,
    ClaimOutcome,
    Rule,
    RulePack,
    RulePackError,
    builtin_packs,
    evaluate_claim,
    load_rule_pack,
    pack_applies,
)
from agent_claimcheck.schema import Step

# --- loading -----------------------------------------------------------


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pack.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_python_object_tag_is_rejected(tmp_path: Path) -> None:
    text = 'pack: !!python/object:os.system ["echo hi"]\nversion: 1\nclaims: {}\n'
    with pytest.raises(RulePackError):
        load_rule_pack(_write(tmp_path, text))


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    text = "pack: x\nversion: 1\nclaims: {}\nfrobnicate: true\n"
    with pytest.raises(RulePackError):
        load_rule_pack(_write(tmp_path, text))


def test_unknown_check_key_is_rejected(tmp_path: Path) -> None:
    text = (
        "pack: x\nversion: 1\nclaims:\n"
        "  done:\n    action: x.do\n    receipt: {ok: {frobnicate: true}}\n"
    )
    with pytest.raises(RulePackError):
        load_rule_pack(_write(tmp_path, text))


def test_bad_regex_in_claim_patterns_is_rejected(tmp_path: Path) -> None:
    text = "pack: x\nversion: 1\nclaim_patterns:\n  done: ['(unclosed']\nclaims: {}\n"
    with pytest.raises(RulePackError):
        load_rule_pack(_write(tmp_path, text))


def test_bad_regex_in_matches_check_is_rejected(tmp_path: Path) -> None:
    text = (
        "pack: x\nversion: 1\nclaims:\n"
        "  done:\n    action: x.do\n    receipt: {sha: {matches: '(unclosed'}}\n"
    )
    with pytest.raises(RulePackError):
        load_rule_pack(_write(tmp_path, text))


def test_args_path_receipt_check_loads_and_reads_the_calls_args(tmp_path: Path) -> None:
    text = (
        "pack: x\nversion: 1\ntools: ['x.do']\nclaims:\n"
        "  done:\n    action: x.do\n    receipt: {args.path: {equals: 'a.txt'}}\n"
    )
    pack = load_rule_pack(_write(tmp_path, text))
    steps = [
        tool_call(0, "x.do", args={"path": "a.txt"}),
        tool_result(1, "x.do", output={}),
    ]
    outcome = evaluate_claim({}, "done", domain="x", steps=steps, packs=[pack])
    assert outcome.outcome == "receipt_only"
    assert outcome.checks[0].path == "args.path"
    assert outcome.checks[0].result == "pass"


# --- helpers for direct engine-level tests ------------------------------


def _pack(rule: Rule, *, tools: tuple[str, ...] = ("x.*",), pack_name: str = "x") -> RulePack:
    return RulePack(
        pack=pack_name, version=1, tools=tools, claim_patterns={}, claims={"done": rule}
    )


def _evaluate(
    rule: Rule, subject: dict[str, Any], steps: list[Step], domain: str = "x"
) -> ClaimOutcome:
    return evaluate_claim(subject, "done", domain=domain, steps=steps, packs=[_pack(rule)])


# --- checks: operators and normalisers ----------------------------------


def test_exists_true_and_false() -> None:
    rule = Rule("x.do", receipt={"a": Check("exists", True), "b": Check("exists", False)})
    steps = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"a": 1})]
    outcome = _evaluate(rule, {}, steps)
    assert outcome.outcome == "receipt_only"
    assert {c.path: c.result for c in outcome.checks} == {"a": "pass", "b": "pass"}


def test_equals_plain_value() -> None:
    rule = Rule("x.do", receipt={"status": Check("equals", "confirmed")})
    steps = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"status": "confirmed"})]
    assert _evaluate(rule, {}, steps).outcome == "receipt_only"

    steps_bad = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"status": "pending"})]
    assert _evaluate(rule, {}, steps_bad).outcome == "contradicted"


def test_in_operator() -> None:
    rule = Rule("x.do", receipt={"status": Check("in", ["confirmed", "tentative"])})
    steps = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"status": "tentative"})]
    assert _evaluate(rule, {}, steps).outcome == "receipt_only"


def test_contains_substring_and_membership() -> None:
    rule_str = Rule("x.do", receipt={"msg": Check("contains", "ok")})
    steps_str = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"msg": "it is ok now"})]
    assert _evaluate(rule_str, {}, steps_str).outcome == "receipt_only"

    rule_list = Rule("x.do", receipt={"emails": Check("contains", "a@example.test")})
    steps_list = [
        tool_call(0, "x.do"),
        tool_result(1, "x.do", output={"emails": ["a@example.test", "b@example.test"]}),
    ]
    assert _evaluate(rule_list, {}, steps_list).outcome == "receipt_only"


def test_matches_regex() -> None:
    rule = Rule("x.do", receipt={"sha": Check("matches", re.compile("^[0-9a-f]{7}$"))})
    steps = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"sha": "abc1234"})]
    assert _evaluate(rule, {}, steps).outcome == "receipt_only"

    steps_bad = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"sha": "not-a-sha"})]
    assert _evaluate(rule, {}, steps_bad).outcome == "contradicted"


def test_datetime_equality_is_timezone_aware() -> None:
    rule = Rule("x.do", receipt={"start": Check("equals", "{subject.start}", as_="datetime")})
    steps = [
        tool_call(0, "x.do"),
        tool_result(1, "x.do", output={"start": "2026-03-02T17:00:00Z"}),
    ]
    outcome = _evaluate(rule, {"start": "2026-03-02T19:00:00+02:00"}, steps)
    assert outcome.outcome == "receipt_only"
    assert outcome.checks[0].result == "pass"


def test_email_normaliser_is_case_and_space_insensitive() -> None:
    rule = Rule("x.do", receipt={"to": Check("equals", "{subject.email}", as_="email")})
    steps = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"to": " A@Example.Test "})]
    outcome = _evaluate(rule, {"email": "a@example.test"}, steps)
    assert outcome.checks[0].result == "pass"


def test_number_normaliser_tolerance() -> None:
    rule = Rule("x.do", receipt={"n": Check("equals", "{subject.n}", as_="number")})
    steps = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"n": 30.0000000001})]
    outcome = _evaluate(rule, {"n": 30}, steps)
    assert outcome.checks[0].result == "pass"


def test_string_normaliser_strips_and_casefolds() -> None:
    rule = Rule("x.do", receipt={"s": Check("equals", "{subject.s}", as_="string")})
    steps = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"s": "  Foo "})]
    outcome = _evaluate(rule, {"s": "foo"}, steps)
    assert outcome.checks[0].result == "pass"


def test_value_that_cannot_be_normalised_fails() -> None:
    rule = Rule("x.do", receipt={"start": Check("equals", "2026-01-01T00:00:00Z", as_="datetime")})
    steps = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"start": 12345})]
    outcome = _evaluate(rule, {}, steps)
    assert outcome.checks[0].result == "fail"


# --- interpolation and missing_subject ----------------------------------


def test_missing_subject_skip_that_leaves_no_other_check_caps_at_receipt_only() -> None:
    # Both the only receipt check and the only probe check need a subject
    # field the claim never carried, so every check is skipped and the
    # outcome is capped below probe_supported.
    rule = Rule(
        "x.do",
        receipt={"start": Check("equals", "{subject.start}", as_="datetime")},
        probe="x.get",
        probe_checks={"end": Check("equals", "{subject.end}", as_="datetime")},
    )
    steps = [
        tool_call(0, "x.do"),
        tool_result(1, "x.do", output={"start": "2026-01-01T00:00:00Z"}),
        probe(2, "x.get", output={"end": "2026-01-01T01:00:00Z"}),
    ]
    outcome = _evaluate(rule, {}, steps)  # no subject at all
    assert outcome.outcome == "receipt_only"
    assert outcome.missing_subject is True
    assert all(c.result == "skipped" for c in outcome.checks)


def test_missing_subject_alongside_a_passing_check_does_not_cap() -> None:
    # The receipt's exists check is unaffected by the missing subject, so the
    # claim still reaches probe_supported; only the skipped check is flagged.
    rule = Rule(
        "x.do",
        receipt={"event_id": Check("exists", True)},
        probe="x.get",
        probe_checks={"start": Check("equals", "{subject.start}", as_="datetime")},
    )
    steps = [
        tool_call(0, "x.do"),
        tool_result(1, "x.do", output={"event_id": "e1"}),
        probe(2, "x.get", output={"start": "2026-01-01T00:00:00Z"}),
    ]
    outcome = _evaluate(rule, {}, steps)  # no subject.start at all
    assert outcome.outcome == "probe_supported"
    assert outcome.missing_subject is True


# --- applicability -------------------------------------------------------


def test_pack_applies_via_tools_glob() -> None:
    pack = RulePack("booking", 1, ("calendar.*",), {}, {})
    steps = [tool_call(0, "calendar.create_event")]
    assert pack_applies(pack, "other", steps) is True


def test_pack_does_not_apply_for_unfamiliar_tools_with_domain_match() -> None:
    pack = RulePack("booking", 1, ("calendar.*",), {}, {})
    steps = [tool_call(0, "unrelated.tool")]
    assert pack_applies(pack, "booking", steps) is False


def test_pack_applies_with_zero_calls_and_matching_domain() -> None:
    pack = RulePack("booking", 1, ("calendar.*",), {}, {})
    assert pack_applies(pack, "booking", []) is True


def test_zero_calls_and_domain_gives_unsupported_not_unknown() -> None:
    rule = Rule("calendar.create_event", receipt={"event_id": Check("exists", True)})
    pack = RulePack("booking", 1, ("calendar.*",), {}, {"booked": rule})
    outcome = evaluate_claim({}, "booked", domain="booking", steps=[], packs=[pack])
    assert outcome.outcome == "unsupported"


def test_unfamiliar_tools_gives_unknown() -> None:
    rule = Rule("calendar.create_event", receipt={"event_id": Check("exists", True)})
    pack = RulePack("booking", 1, ("calendar.*",), {}, {"booked": rule})
    steps = [tool_call(0, "unrelated.tool")]
    outcome = evaluate_claim({}, "booked", domain="booking", steps=steps, packs=[pack])
    assert outcome.outcome == "unknown"


def test_no_pack_defines_the_claim_type_is_unknown() -> None:
    outcome = evaluate_claim({}, "booked", domain="booking", steps=[], packs=[])
    assert outcome.outcome == "unknown"
    assert outcome.step is None


# --- outcomes with step citations ---------------------------------------


def test_contradicted_cites_the_failing_result_step() -> None:
    rule = Rule("x.do", receipt={"ok_field": Check("exists", True)})
    steps = [
        tool_call(0, "x.do"),
        tool_result(1, "x.do", ok=False, error="409 conflict: taken", output={}),
    ]
    outcome = _evaluate(rule, {}, steps)
    assert outcome.outcome == "contradicted"
    assert outcome.step == 1


def test_unsupported_when_no_matching_call() -> None:
    # "x.other" keeps the pack applicable (it matches the pack's tools
    # glob), but no call matches this claim's own action, "x.do".
    rule = Rule("x.do", receipt={})
    outcome = _evaluate(rule, {}, [tool_call(0, "x.other")])
    assert outcome.outcome == "unsupported"
    assert outcome.step is None


def test_unsupported_when_matching_call_has_no_result() -> None:
    rule = Rule("x.do", receipt={})
    outcome = _evaluate(rule, {}, [tool_call(0, "x.do")])
    assert outcome.outcome == "unsupported"


def test_receipt_only_when_no_probe_declared() -> None:
    rule = Rule("x.do", receipt={"a": Check("exists", True)})
    steps = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"a": 1})]
    outcome = _evaluate(rule, {}, steps)
    assert outcome.outcome == "receipt_only"
    assert outcome.step == 1


def test_probe_supported_cites_the_probe_step() -> None:
    rule = Rule("x.do", receipt={"a": Check("exists", True)}, probe="x.get", probe_checks={})
    steps = [
        tool_call(0, "x.do"),
        tool_result(1, "x.do", output={"a": 1}),
        probe(2, "x.get", ok=True, output={}),
    ]
    outcome = _evaluate(rule, {}, steps)
    assert outcome.outcome == "probe_supported"
    assert outcome.step == 2


def test_probe_missing_falls_back_to_receipt_only() -> None:
    rule = Rule("x.do", receipt={"a": Check("exists", True)}, probe="x.get", probe_checks={})
    steps = [tool_call(0, "x.do"), tool_result(1, "x.do", output={"a": 1})]
    outcome = _evaluate(rule, {}, steps)
    assert outcome.outcome == "receipt_only"
    assert outcome.step == 1


def test_probe_contradicts_cites_the_probe_step() -> None:
    rule = Rule(
        "x.do",
        receipt={"a": Check("exists", True)},
        probe="x.get",
        probe_checks={"status": Check("equals", "confirmed")},
    )
    steps = [
        tool_call(0, "x.do"),
        tool_result(1, "x.do", output={"a": 1}),
        probe(2, "x.get", output={"status": "cancelled"}),
    ]
    outcome = _evaluate(rule, {}, steps)
    assert outcome.outcome == "contradicted"
    assert outcome.step == 2


def test_last_matching_probe_decides() -> None:
    rule = Rule(
        "x.do",
        receipt={"a": Check("exists", True)},
        probe="x.get",
        probe_checks={"status": Check("equals", "confirmed")},
    )
    steps = [
        tool_call(0, "x.do"),
        tool_result(1, "x.do", output={"a": 1}),
        probe(2, "x.get", output={"status": "cancelled"}),
        probe(3, "x.get", output={"status": "confirmed"}),
    ]
    outcome = _evaluate(rule, {}, steps)
    assert outcome.outcome == "probe_supported"
    assert outcome.step == 3


def test_retry_after_a_failed_call_reaches_probe_supported() -> None:
    rule = Rule(
        "x.do",
        receipt={"a": Check("exists", True)},
        probe="x.get",
        probe_checks={"status": Check("equals", "confirmed")},
    )
    steps = [
        tool_call(0, "x.do"),
        tool_result(1, "x.do", ok=False, error="429: rate limited", output={}),
        tool_call(2, "x.do"),
        tool_result(3, "x.do", output={"a": 1}),
        probe(4, "x.get", output={"status": "confirmed"}),
    ]
    outcome = _evaluate(rule, {}, steps)
    assert outcome.outcome == "probe_supported"
    assert outcome.step == 4


# --- reviewer findings: pack rules missing subject-vs-evidence checks ---


def test_booking_booked_rule_catches_wrong_attendee() -> None:
    # `booked`'s subject is {start, timezone, attendee_email,
    # duration_min}. A booking made for the wrong attendee must not reach
    # probe_supported just because the receipt/probe otherwise look fine.
    packs = builtin_packs()
    steps = [
        tool_call(
            0,
            "calendar.create_event",
            args={"duration_min": 30, "attendees": ["bob@example.test"]},
        ),
        tool_result(
            1,
            "calendar.create_event",
            output={
                "event_id": "e1",
                "start": "2026-01-01T00:00:00Z",
                "attendees": ["bob@example.test"],
            },
        ),
        probe(
            2,
            "calendar.get_event",
            output={
                "status": "confirmed",
                "start": "2026-01-01T00:00:00Z",
                "attendees": ["bob@example.test"],
                "invites_sent": ["bob@example.test"],
            },
        ),
    ]
    subject = {
        "start": "2026-01-01T00:00:00Z",
        "duration_min": 30,
        "attendee_email": "alice@example.test",
    }
    outcome = evaluate_claim(
        subject, "booked", domain="booking", steps=steps, packs=list(packs.values())
    )
    assert outcome.outcome != "probe_supported"


def test_crm_updated_rule_catches_wrong_field_value() -> None:
    # A right-record-wrong-value CRM update must not reach probe_supported:
    # the rule must compare the claimed field values, not just their
    # existence.
    packs = builtin_packs()
    steps = [
        tool_call(0, "crm.update_contact", args={"record_id": "c1"}),
        tool_result(
            1,
            "crm.update_contact",
            output={"record_id": "c1", "updated_fields": {"phone": "+1-555-9999"}},
        ),
        probe(2, "crm.read_record", output={"fields": {"phone": "+1-555-9999"}}),
    ]
    subject = {
        "object": "contact",
        "record_id": "c1",
        "fields": {"phone": "+1-555-0100"},
    }
    outcome = evaluate_claim(
        subject, "updated", domain="crm", steps=steps, packs=list(packs.values())
    )
    assert outcome.outcome != "probe_supported"


def test_coding_tests_passed_rule_catches_a_narrower_suite() -> None:
    # Claiming the full suite passed while only a subset ran must not reach
    # probe_supported: the rule must compare subject.suite/count, not just
    # the probe's failed == 0.
    packs = builtin_packs()
    steps = [
        tool_call(0, "shell.run", args={"command": "pytest tests/test_login.py"}),
        tool_result(1, "shell.run", output={"exit_code": 0}),
        probe(
            2,
            "ci.run_tests",
            output={"suite": "tests/test_login.py", "total": 3, "passed": 3, "failed": 0},
        ),
    ]
    subject = {"suite": "full", "count": 42}
    outcome = evaluate_claim(
        subject, "tests_passed", domain="coding", steps=steps, packs=list(packs.values())
    )
    assert outcome.outcome != "probe_supported"


# --- reviewer finding: empty check list vacuously "all skipped" --------


def test_probe_supported_when_rule_declares_zero_checks() -> None:
    # A rule with an empty receipt and no probe_checks is valid per the pack
    # schema. When the probe genuinely confirms the claim (ok, no error), an
    # empty check list must not be treated as "every check was skipped".
    rule = Rule("x.do", receipt={}, probe="x.get", probe_checks={})
    steps = [
        tool_call(0, "x.do"),
        tool_result(1, "x.do", output={}),
        probe(2, "x.get", ok=True, output={}),
    ]
    outcome = _evaluate(rule, {}, steps)
    assert outcome.outcome == "probe_supported"


def test_booking_booked_rule_still_verifies_the_right_attendee() -> None:
    packs = builtin_packs()
    steps = [
        tool_call(
            0,
            "calendar.create_event",
            args={"duration_min": 30, "attendees": ["alice@example.test"]},
        ),
        tool_result(
            1,
            "calendar.create_event",
            output={
                "event_id": "e1",
                "start": "2026-01-01T00:00:00Z",
                "attendees": ["alice@example.test"],
            },
        ),
        probe(
            2,
            "calendar.get_event",
            output={
                "status": "confirmed",
                "start": "2026-01-01T00:00:00Z",
                "attendees": ["alice@example.test"],
                "invites_sent": ["alice@example.test"],
            },
        ),
    ]
    subject = {
        "start": "2026-01-01T00:00:00Z",
        "duration_min": 30,
        "attendee_email": "alice@example.test",
    }
    outcome = evaluate_claim(
        subject, "booked", domain="booking", steps=steps, packs=list(packs.values())
    )
    assert outcome.outcome == "probe_supported"


def test_crm_updated_rule_still_verifies_the_right_value() -> None:
    packs = builtin_packs()
    steps = [
        tool_call(0, "crm.update_contact", args={"record_id": "c1"}),
        tool_result(
            1,
            "crm.update_contact",
            output={"record_id": "c1", "updated_fields": {"phone": "+1-555-0100"}},
        ),
        probe(2, "crm.read_record", output={"fields": {"phone": "+1-555-0100"}}),
    ]
    subject = {"object": "contact", "record_id": "c1", "fields": {"phone": "+1-555-0100"}}
    outcome = evaluate_claim(
        subject, "updated", domain="crm", steps=steps, packs=list(packs.values())
    )
    assert outcome.outcome == "probe_supported"


def test_coding_tests_passed_rule_still_verifies_the_full_suite() -> None:
    packs = builtin_packs()
    steps = [
        tool_call(0, "shell.run", args={"command": "pytest"}),
        tool_result(1, "shell.run", output={"exit_code": 0}),
        probe(2, "ci.run_tests", output={"suite": "full", "total": 42, "passed": 42, "failed": 0}),
    ]
    subject = {"suite": "full", "count": 42}
    outcome = evaluate_claim(
        subject, "tests_passed", domain="coding", steps=steps, packs=list(packs.values())
    )
    assert outcome.outcome == "probe_supported"


# --- the documented async-accepted (202) receipts are receipts too ----


def test_crm_updated_accepts_the_async_accepted_receipt() -> None:
    packs = builtin_packs()
    steps = [
        tool_call(0, "crm.update_contact", args={"record_id": "c1"}),
        tool_result(
            1,
            "crm.update_contact",
            output={"record_id": "c1", "job_id": "job-1", "status": "accepted"},
        ),
    ]
    subject = {"object": "contact", "record_id": "c1", "fields": {"phone": "+1-555-0100"}}
    outcome = evaluate_claim(
        subject, "updated", domain="crm", steps=steps, packs=list(packs.values())
    )
    assert outcome.outcome == "receipt_only"


def test_coding_fixed_accepts_the_async_accepted_receipt() -> None:
    packs = builtin_packs()
    steps = [
        tool_call(0, "fs.write_file", args={"path": "src/app.py", "content": "x"}),
        tool_result(
            1,
            "fs.write_file",
            output={"path": "src/app.py", "job_id": "job-2", "status": "pending"},
        ),
    ]
    outcome = evaluate_claim(
        {"file": "src/app.py"}, "fixed", domain="coding", steps=steps, packs=list(packs.values())
    )
    assert outcome.outcome == "receipt_only"
