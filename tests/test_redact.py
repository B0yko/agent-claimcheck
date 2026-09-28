"""The detector view must never leak ground_truth, meta or source."""

from __future__ import annotations

import json

from agent_claimcheck.redact import detector_view, resolve_claims
from agent_claimcheck.schema import Trace

CANARY = "canary-3f9c7a2b-do-not-leak"

TRACE_WITH_CANARY = {
    "schema": "agent-trace/v1",
    "trace_id": "t1",
    "source": "test/0.0.1",
    "task": {"id": "task-1", "domain": "booking", "instruction": "book a table"},
    "steps": [
        {
            "i": 0,
            "ts": "2026-01-01T00:00:00+00:00",
            "kind": "tool_call",
            "role": "agent",
            "name": "calendar.create_event",
        },
    ],
    "final_claim": {"text": "Booked.", "claims": [{"type": "booked", "subject": {}}]},
    "ground_truth": {
        "outcome": "success",
        "checked_by": "human",
        "details": {"note": CANARY},
    },
    "meta": {"note": CANARY},
}


def test_detector_view_drops_ground_truth_meta_and_source() -> None:
    trace = Trace.model_validate(TRACE_WITH_CANARY)
    view = detector_view(trace)

    assert not hasattr(view, "ground_truth")
    assert not hasattr(view, "meta")
    assert not hasattr(view, "source")


def test_canary_never_reaches_the_detector_view_json() -> None:
    trace = Trace.model_validate(TRACE_WITH_CANARY)
    view = detector_view(trace)
    resolved = resolve_claims(view)

    view_json = json.dumps(view.model_dump(mode="json"))
    resolved_json = json.dumps(resolved.model_dump(mode="json"))

    assert CANARY not in view_json
    assert CANARY not in resolved_json

    # Sanity: the canary really was present on the source trace.
    trace_json = json.dumps(trace.model_dump(mode="json"))
    assert CANARY in trace_json
