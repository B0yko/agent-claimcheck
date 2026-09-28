"""`TraceStore`, the inspector payload and review appending."""

from __future__ import annotations

import json
from pathlib import Path

from factory import message, tool_call, tool_result, trace

from agent_claimcheck.checker import Checker
from agent_claimcheck.claims import ClaimExtractor
from agent_claimcheck.rules.engine import builtin_packs
from agent_claimcheck.schema import GroundTruth, load_traces_report
from agent_claimcheck.server.state import (
    append_review,
    build_record,
    build_store,
    inspector_payload,
)


def _extractor() -> ClaimExtractor:
    return ClaimExtractor(list(builtin_packs().values()))


def test_queue_is_sorted_by_distance_from_the_gate_fence() -> None:
    checker = Checker("cascade-offline")
    traces = [
        trace(
            "booking-near",
            "booking",
            [message(0, "agent", "Not sure what happened.")],
            text="Not sure what happened.",
            claims=[("booked", {})],
        ),
        trace(
            "booking-far",
            "booking",
            [message(0, "agent", "Something odd occurred.")],
            text="Something odd occurred.",
            claims=[("booked", {})],
        ),
    ]
    results = {r.trace_id: r for r in checker.check(traces)}
    store = build_store(traces, results, checker.extractor)

    queue = store.queue()
    assert all(r.result.verdict == "unverifiable" for r in queue)
    distances = [abs((r.result.p_success or 0.5) - 0.5) for r in queue]
    assert distances == sorted(distances)


def test_overview_counts_by_verdict_and_domain() -> None:
    checker = Checker("cascade-offline")
    traces = [
        trace(
            "booking-ok",
            "booking",
            [
                tool_call(0, "calendar.create_event", {"title": "x"}),
                tool_result(1, "calendar.create_event", ok=True, output={"status": "confirmed"}),
            ],
            text="Booked.",
            claims=[("booked", {})],
        ),
    ]
    results = {r.trace_id: r for r in checker.check(traces)}
    store = build_store(traces, results, checker.extractor)
    overview = store.overview()
    assert sum(sum(by_domain.values()) for by_domain in overview.values()) == 1


def test_inspector_payload_never_carries_ground_truth_or_meta() -> None:
    checker = Checker("cascade-offline")
    labelled = trace(
        "booking-01",
        "booking",
        [message(0, "agent", "All set.")],
        text="All set.",
        claims=[("booked", {})],
    ).model_copy(
        update={
            "ground_truth": GroundTruth(outcome="success", checked_by="state_probe", details={}),
            "meta": {"secret": "canary-value-should-never-leak"},
        }
    )
    results = {r.trace_id: r for r in checker.check([labelled])}
    record = build_record(labelled, results["booking-01"], checker.extractor)
    payload = inspector_payload(record)

    blob = json.dumps(payload)
    assert "canary-value-should-never-leak" not in blob
    assert "ground_truth" not in blob
    assert "checked_by" not in blob
    assert payload["trace_id"] == "booking-01"
    assert payload["verdict"] in ("verified", "false_success", "unverifiable", "skipped")


def test_append_review_writes_a_validated_agent_trace_with_human_ground_truth(
    tmp_path: Path,
) -> None:
    checker = Checker("cascade-offline")
    traces = [
        trace(
            "crm-01",
            "crm",
            [message(0, "agent", "Updated the contact.")],
            text="Updated the contact.",
            claims=[("updated", {"object": "contact", "record_id": "r1"})],
        ),
        trace(
            "crm-02",
            "crm",
            [message(0, "agent", "Updated the contact.")],
            text="Updated the contact.",
            claims=[("updated", {"object": "contact", "record_id": "r2"})],
        ),
    ]
    results = {r.trace_id: r for r in checker.check(traces)}
    record_1 = build_record(traces[0], results["crm-01"], checker.extractor)
    record_2 = build_record(traces[1], results["crm-02"], checker.extractor)

    reviews_path = tmp_path / "reviews.jsonl"
    append_review(reviews_path, record_1, decision="verified", note="looks solid")
    append_review(reviews_path, record_2, decision="false_success", note="")

    lines = reviews_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    report = load_traces_report(reviews_path)
    assert not report.errors
    first, second = report.traces
    assert first.ground_truth is not None
    assert first.ground_truth.outcome == "success"
    assert first.ground_truth.checked_by == "human"
    assert first.ground_truth.details["decision"] == "verified"
    assert first.ground_truth.details["note"] == "looks solid"
    assert second.ground_truth is not None
    assert second.ground_truth.outcome == "failure"
