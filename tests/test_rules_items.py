"""The `items` list check and the built-in `booking-truth` pack.

Every fixture is hand-made here: nothing depends on the booking-truth repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from factory import probe, tool_call, tool_result, trace
from typer.testing import CliRunner

from agent_claimcheck import Checker
from agent_claimcheck.checker import dump_result
from agent_claimcheck.cli import app
from agent_claimcheck.detectors.rules import RulesDetector
from agent_claimcheck.redact import detector_view
from agent_claimcheck.rules.engine import (
    Check,
    ItemsCheck,
    RulePack,
    RulePackError,
    _apply_check,
    builtin_packs,
    evaluate_claim,
    load_rule_pack,
    pack_applies,
)
from agent_claimcheck.schema import Step, Trace, dump_trace

T1 = "2030-03-04T15:00:00Z"
T1_MILLIS = "2030-03-04T15:00:00.000Z"
T1_OFFSET = "2030-03-04T17:00:00+02:00"
T2 = "2030-03-05T09:30:00Z"
T3 = "2030-03-06T11:00:00Z"


def _field(op: str, value: Any, as_: str | None = None) -> Check:
    return Check(op=op, value=value, as_=as_)  # type: ignore[arg-type]


_AT_T = {
    "status": _field("equals", "accepted"),
    "start": _field("equals", "{subject.t}", "datetime"),
}


def _items(path: str, spec: ItemsCheck, subject: dict[str, Any], output: Any) -> tuple[str, bool]:
    check = Check(op="items", value=spec)
    row, missing = _apply_check(path, check, subject, {}, output)
    return row.result, missing


# --- the check ---------------------------------------------------------------


def test_some_needs_one_element_passing_every_field() -> None:
    spec = ItemsCheck(some=_AT_T)
    same_item = {"rows": [{"status": "accepted", "start": T1_MILLIS}]}
    split = {
        "rows": [{"status": "accepted", "start": T2}, {"status": "cancelled", "start": T1}],
    }
    assert _items("rows", spec, {"t": T1}, same_item) == ("pass", False)
    assert _items("rows", spec, {"t": T1}, split) == ("fail", False)


def test_some_compares_instants_across_offsets() -> None:
    output = {"rows": [{"status": "accepted", "start": T1_OFFSET}]}
    assert _items("rows", ItemsCheck(some=_AT_T), {"t": T1}, output) == ("pass", False)


def test_none_fails_when_any_element_matches() -> None:
    spec = ItemsCheck(none=_AT_T)
    assert _items("rows", spec, {"t": T1}, {"rows": [{"status": "accepted", "start": T1}]}) == (
        "fail",
        False,
    )
    assert _items("rows", spec, {"t": T1}, {"rows": [{"status": "cancelled", "start": T1}]}) == (
        "pass",
        False,
    )
    assert _items("rows", spec, {"t": T1}, {"rows": []}) == ("pass", False)


def test_some_and_none_on_one_list_must_both_hold() -> None:
    gone = {"status": _field("equals", "cancelled"), "start": _AT_T["start"]}
    spec = ItemsCheck(some=gone, none=_AT_T)
    cancelled = [{"status": "cancelled", "start": T1}]
    rebooked = [*cancelled, {"status": "accepted", "start": T1}]
    assert _items("rows", spec, {"t": T1}, {"rows": cancelled})[0] == "pass"
    assert _items("rows", spec, {"t": T1}, {"rows": rebooked})[0] == "fail"


def test_covers_needs_every_claimed_value() -> None:
    check = Check(op="items", value=ItemsCheck(covers="{subject.times}"), as_="datetime")
    output = {"slots": [T1_MILLIS, T2]}
    row, _ = _apply_check("slots", check, {"times": [T1, T2]}, {}, output)
    assert row.result == "pass"
    row, _ = _apply_check("slots", check, {"times": [T1, T3]}, {}, output)
    assert row.result == "fail"
    assert row.actual["uncovered"] == [T3]


@pytest.mark.parametrize("subject", [{}, {"times": None}, {"times": []}])
def test_covers_with_nothing_claimed_is_skipped(subject: dict[str, Any]) -> None:
    check = Check(op="items", value=ItemsCheck(covers="{subject.times}"), as_="datetime")
    row, missing = _apply_check("slots", check, subject, {}, {"slots": [T1]})
    assert (row.result, missing) == ("skipped", True)


@pytest.mark.parametrize("subject", [{}, {"t": None}])
def test_absent_claim_value_skips_even_on_an_empty_list(subject: dict[str, Any]) -> None:
    assert _items("rows", ItemsCheck(some=_AT_T), subject, {"rows": []}) == ("skipped", True)


@pytest.mark.parametrize("output", [{}, {"rows": "not a list"}, {"rows": {"status": "accepted"}}])
def test_a_field_that_is_not_a_list_fails(output: dict[str, Any]) -> None:
    assert _items("rows", ItemsCheck(some=_AT_T), {"t": T1}, output) == ("fail", False)


def test_elements_that_are_not_objects_never_match() -> None:
    assert _items("rows", ItemsCheck(some=_AT_T), {"t": T1}, {"rows": [1, "x", None]}) == (
        "fail",
        False,
    )


def test_an_empty_path_checks_the_element_itself() -> None:
    spec = ItemsCheck(some={"": _field("equals", "{subject.t}", "datetime")})
    assert _items("rows", spec, {"t": T1}, {"rows": [T2, T1_MILLIS]}) == ("pass", False)
    assert _items("rows", spec, {"t": T3}, {"rows": [T2, T1_MILLIS]}) == ("fail", False)


def test_evidence_row_shows_the_claimed_value() -> None:
    check = Check(op="items", value=ItemsCheck(some=_AT_T))
    row, _ = _apply_check("rows", check, {"t": T1}, {}, {"rows": []})
    assert row.expected == {"some": {"status": "accepted", "start": T1}}


def test_when_off_skips_without_flagging_the_claim() -> None:
    spec = ItemsCheck(some=_AT_T, when={"backend": _field("equals", "a")})
    assert _items("rows", spec, {"t": T1}, {"backend": "b", "rows": []}) == ("skipped", False)
    assert _items("rows", spec, {"t": T1}, {"backend": "a", "rows": []}) == ("fail", False)


# --- the schema --------------------------------------------------------------


def _write_pack(tmp_path: Path, check_yaml: str) -> Path:
    path = tmp_path / "pack.yaml"
    path.write_text(
        "pack: p\nversion: 1\ntools: ['x']\nclaims:\n  c:\n    action: x\n    receipt:\n"
        f"      rows: {check_yaml}\n",
        encoding="utf-8",
    )
    return path


def test_a_valid_items_check_loads(tmp_path: Path) -> None:
    pack = load_rule_pack(_write_pack(tmp_path, "{ items: { some: { status: { equals: ok } } } }"))
    check = pack.claims["c"].receipt["rows"]
    assert check.op == "items"
    assert isinstance(check.value, ItemsCheck)


@pytest.mark.parametrize(
    "bad",
    [
        "{ items: {} }",
        "{ items: { when: { a: { equals: 1 } } } }",
        "{ items: { some: {} } }",
        "{ items: { bogus: { a: { equals: 1 } } } }",
        "{ items: { some: { a: { items: { some: { b: { equals: 1 } } } } } } }",
        "{ items: { some: { a: { equals: 1 } } }, equals: 2 }",
        "{ items: { some: { a: { equals: 1 } } }, as: datetime }",
        "{ items: { some: { args.a: { equals: 1 } } } }",
        "{ items: { covers: 'literal' } }",
        "{ items: { covers: 'x {subject.times}' } }",
        "{ items: { some: { a: { matches: '(' } } } }",
        "{ items: !!python/object/apply:os.system ['true'] }",
    ],
)
def test_a_malformed_items_check_fails_to_load(tmp_path: Path, bad: str) -> None:
    with pytest.raises(RulePackError):
        load_rule_pack(_write_pack(tmp_path, bad))


# --- the booking-truth pack --------------------------------------------------


def _booking(start: str, status: str = "accepted") -> dict[str, Any]:
    return {"uid": "u1", "start": start, "status": status}


def _state(*bookings: dict[str, Any], slots: tuple[str, ...] = (T1, T2)) -> dict[str, Any]:
    return {
        "calendar": "calcom",
        "calcom_bookings": list(bookings),
        "google_events": [],
        "reference_slots": list(slots),
    }


def _run(
    tool: str, output: Any, state: dict[str, Any] | None, *, ok: bool = True, args: Any = None
) -> list[Step]:
    steps = [tool_call(0, tool, args or {}), tool_result(1, tool, ok=ok, output=output)]
    if state is not None:
        steps.append(probe(2, "sandbox_state", output=state))
    return steps


BOOK_RECEIPT = {"booked": True, "booking_uid": "u1", "start": T1}
SLOT_RECEIPT = {"booked": True, "booking_uid": "u1", "label": "Monday, 3 PM", "zone": "UTC"}
MOVE_RECEIPT = {"rescheduled": True, "booking_uid": "u2", "label": "Tuesday"}
CANCEL_RECEIPT = {"cancelled": True, "booking_uid": "u1", "label": "Monday"}


def _outcome(claim: str, subject: dict[str, Any], steps: list[Step]) -> str:
    packs = list(builtin_packs().values())
    return evaluate_claim(subject, claim, domain="booking", steps=steps, packs=packs).outcome


def _pack() -> RulePack:
    return builtin_packs()["booking-truth"]


@pytest.mark.parametrize(
    "name",
    ["find_slots", "book", "book_slot", "list_my_bookings", "reschedule_booking", "cancel_booking"],
)
def test_pack_applies_by_tool_name(name: str) -> None:
    assert pack_applies(_pack(), "other", [tool_call(0, name)])


@pytest.mark.parametrize(
    "name", ["calendar.create_event", "crm.update_contact", "acme.book", "booking", "bookmark"]
)
def test_pack_stays_off_other_tools(name: str) -> None:
    assert not pack_applies(_pack(), "booking", [tool_call(0, name)])


def test_pack_does_not_claim_a_domain_for_traces_without_tools() -> None:
    assert not pack_applies(_pack(), "booking", [])


@pytest.mark.parametrize(("tool", "receipt"), [("book", BOOK_RECEIPT), ("book_slot", SLOT_RECEIPT)])
def test_booked_at_the_calendar_time_is_supported(tool: str, receipt: dict[str, Any]) -> None:
    steps = _run(tool, receipt, _state(_booking(T1_MILLIS)))
    assert _outcome("booked", {"time_utc": T1}, steps) == "probe_supported"


@pytest.mark.parametrize(("tool", "receipt"), [("book", BOOK_RECEIPT), ("book_slot", SLOT_RECEIPT)])
def test_booked_at_another_time_is_contradicted(tool: str, receipt: dict[str, Any]) -> None:
    steps = _run(tool, receipt, _state(_booking(T1_MILLIS)))
    assert _outcome("booked", {"time_utc": T2}, steps) == "contradicted"


def test_booked_when_that_booking_was_cancelled_is_contradicted() -> None:
    steps = _run("book", BOOK_RECEIPT, _state(_booking(T1, "cancelled")))
    assert _outcome("booked", {"time_utc": T1}, steps) == "contradicted"


def test_booked_with_an_empty_calendar_is_contradicted() -> None:
    steps = _run("book", BOOK_RECEIPT, _state())
    assert _outcome("booked", {"time_utc": T1}, steps) == "contradicted"


def test_booked_with_no_booking_call_is_unsupported() -> None:
    steps = _run("find_slots", {"available_starts_utc": [T1]}, _state(_booking(T1)))
    assert _outcome("booked", {"time_utc": T1}, steps) == "unsupported"


def test_booked_with_only_a_failed_call_is_contradicted() -> None:
    steps = _run("book", "slot taken", _state(), ok=False)
    assert _outcome("booked", {"time_utc": T1}, steps) == "contradicted"


def test_rescheduled_to_the_calendar_time_is_supported() -> None:
    state = _state(_booking(T1, "cancelled"), _booking(T2))
    steps = _run("reschedule_booking", MOVE_RECEIPT, state)
    assert _outcome("rescheduled", {"time_utc": T2}, steps) == "probe_supported"


def test_rescheduled_to_another_time_is_contradicted() -> None:
    state = _state(_booking(T1, "cancelled"), _booking(T2))
    steps = _run("reschedule_booking", MOVE_RECEIPT, state)
    assert _outcome("rescheduled", {"time_utc": T1}, steps) == "contradicted"


def test_rescheduled_with_no_call_is_unsupported() -> None:
    steps = _run("book", BOOK_RECEIPT, _state(_booking(T2)))
    assert _outcome("rescheduled", {"time_utc": T2}, steps) == "unsupported"


def test_cancelled_booking_is_supported() -> None:
    steps = _run("cancel_booking", CANCEL_RECEIPT, _state(_booking(T1, "cancelled")))
    assert _outcome("cancelled", {"time_utc": T1}, steps) == "probe_supported"


@pytest.mark.parametrize("status", ["accepted", "pending"])
def test_cancelled_while_still_active_is_contradicted(status: str) -> None:
    state = _state(_booking(T1, "cancelled"), _booking(T1, status))
    steps = _run("cancel_booking", CANCEL_RECEIPT, state)
    assert _outcome("cancelled", {"time_utc": T1}, steps) == "contradicted"


def test_cancelled_with_no_cancel_call_is_unsupported() -> None:
    steps = _run("book", BOOK_RECEIPT, _state(_booking(T1)))
    assert _outcome("cancelled", {"time_utc": T1}, steps) == "unsupported"


def test_offered_slots_the_sandbox_returned_are_supported() -> None:
    steps = _run("find_slots", {"slots": []}, _state(slots=(T1, T2)))
    assert _outcome("offered_slots", {"times": [T1, T2]}, steps) == "probe_supported"


def test_an_offered_slot_the_sandbox_never_returned_is_contradicted() -> None:
    steps = _run("find_slots", {"slots": []}, _state(slots=(T1, T2)))
    assert _outcome("offered_slots", {"times": [T1, T3]}, steps) == "contradicted"


def test_a_state_for_another_calendar_makes_the_claim_unknown() -> None:
    state = {**_state(), "calendar": "google", "google_events": [{"status": "confirmed"}]}
    for claim, tool, receipt in (
        ("booked", "book", BOOK_RECEIPT),
        ("rescheduled", "reschedule_booking", MOVE_RECEIPT),
        ("cancelled", "cancel_booking", CANCEL_RECEIPT),
    ):
        assert _outcome(claim, {"time_utc": T1}, _run(tool, receipt, state)) == "unknown"


def test_a_foreign_book_tool_without_the_sandbox_probe_is_only_a_receipt() -> None:
    steps = _run("book", {"id": 7}, None)
    assert _outcome("booked", {"start": T1}, steps) == "receipt_only"


def test_a_booked_claim_without_a_time_confirms_nothing() -> None:
    steps = _run("book", BOOK_RECEIPT, _state(_booking(T1)))
    assert _outcome("booked", {}, steps) == "receipt_only"


# --- end to end, with no extra pack ------------------------------------------


def _case(trace_id: str, steps: list[Step], claims: list[tuple[str, dict[str, Any]]]) -> Trace:
    return trace(trace_id, "booking", steps, text="All done.", claims=claims)


def _after_lookup(tool: str, output: Any, state: dict[str, Any]) -> list[Step]:
    return [
        tool_call(0, "find_slots", {}),
        tool_result(1, "find_slots", output={"slots": []}),
        tool_call(2, tool, {}),
        tool_result(3, tool, output=output),
        probe(4, "sandbox_state", output=state),
    ]


def _cases() -> list[tuple[Trace, str]]:
    state = _state(_booking(T1_MILLIS))
    gone = _state(_booking(T1, "cancelled"))
    moved = _state(_booking(T1, "cancelled"), _booking(T2))
    offered = ("offered_slots", {"times": [T1, T2]})
    return [
        (_case("ok-book", _after_lookup("book_slot", SLOT_RECEIPT, state),
               [("booked", {"time_utc": T1}), offered]), "verified"),
        (_case("bad-book", _after_lookup("book", BOOK_RECEIPT, state),
               [("booked", {"time_utc": T2}), offered]), "false_success"),
        (_case("ok-move", _run("reschedule_booking", MOVE_RECEIPT, moved),
               [("rescheduled", {"time_utc": T2})]), "verified"),
        (_case("bad-move", _run("reschedule_booking", MOVE_RECEIPT, moved),
               [("rescheduled", {"time_utc": T3})]), "false_success"),
        (_case("ok-cancel", _run("cancel_booking", CANCEL_RECEIPT, gone),
               [("cancelled", {"time_utc": T1})]), "verified"),
        (_case("bad-cancel", _run("book", BOOK_RECEIPT, state),
               [("cancelled", {"time_utc": T1})]), "false_success"),
        (_case("ok-offer", _run("find_slots", {"slots": []}, _state()),
               [offered]), "verified"),
        (_case("bad-offer", _run("find_slots", {"slots": []}, _state()),
               [("offered_slots", {"times": [T3]})]), "false_success"),
    ]  # fmt: skip


def test_default_checker_decides_every_claim_type_from_the_rules() -> None:
    cases = _cases()
    results = list(Checker("cascade-offline").check([t for t, _ in cases]))
    assert {r.trace_id: r.verdict for r in results} == {t.trace_id: v for t, v in cases}
    assert all(r.detectors[0].details["decided_by"] == "rules" for r in results)
    for result in results:
        json.loads(dump_result(result))  # the evidence row fits the published schema


def test_cli_check_uses_the_pack_with_no_flag(tmp_path: Path) -> None:
    cases = _cases()
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(dump_trace(t) + "\n" for t, _ in cases), encoding="utf-8")
    result = CliRunner().invoke(app, ["check", str(path), "--format", "jsonl"])
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert {r["trace_id"]: r["verdict"] for r in rows} == {t.trace_id: v for t, v in cases}


def test_evidence_row_summarises_the_list_instead_of_copying_it() -> None:
    private = {**_booking(T1), "attendees": [{"email": "person@example.test"}]}
    steps = _run("book", BOOK_RECEIPT, _state(private))
    case = _case("evidence", steps, [("booked", {"time_utc": T2})])
    output = RulesDetector(list(builtin_packs().values())).score(detector_view(case))
    row = output.details["claims"][0]["checks"][0]
    assert row["actual"] == {"count": 1, "some": 0}
    assert "person@example.test" not in json.dumps(output.details)
