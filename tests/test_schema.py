"""Schema validation, trace I/O, probe merge and pairing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_claimcheck.schema import (
    Step,
    Trace,
    TraceValidationError,
    dump_trace,
    load_traces,
    load_traces_report,
    pair_results,
)

VALID_TRACE = {
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
            "args": {"start": "2026-01-02T19:00:00+00:00"},
        },
        {
            "i": 1,
            "ts": "2026-01-01T00:00:01+00:00",
            "kind": "tool_result",
            "role": "tool",
            "name": "calendar.create_event",
            "ok": True,
            "output": {"event_id": "e1"},
        },
    ],
    "final_claim": {"text": "Booked.", "claims": [{"type": "booked", "subject": {}}]},
    "ground_truth": {"outcome": "success", "checked_by": "human", "details": {}},
}


def _write_jsonl(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_valid_trace_round_trips(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, [json.dumps(VALID_TRACE)])
    report = load_traces_report(path)
    assert report.errors == []
    assert len(report.traces) == 1
    trace = report.traces[0]
    assert trace.trace_id == "t1"

    dumped = dump_trace(trace)
    assert json.loads(dumped)["trace_id"] == "t1"


def test_bad_domain_reports_json_path(tmp_path: Path) -> None:
    bad = json.loads(json.dumps(VALID_TRACE))
    bad["task"]["domain"] = "spaceship"
    path = _write_jsonl(tmp_path, [json.dumps(bad)])

    report = load_traces_report(path)
    assert report.traces == []
    assert len(report.errors) == 1
    err = report.errors[0]
    assert err.line_no == 1
    assert err.json_path == "$.task.domain"
    assert "spaceship" in err.message


def test_missing_required_field_reports_json_path(tmp_path: Path) -> None:
    bad = json.loads(json.dumps(VALID_TRACE))
    del bad["source"]
    path = _write_jsonl(tmp_path, [json.dumps(bad)])

    report = load_traces_report(path)
    assert report.traces == []
    assert any(e.json_path == "$" and "source" in e.message for e in report.errors)


def test_additional_property_rejected(tmp_path: Path) -> None:
    bad = json.loads(json.dumps(VALID_TRACE))
    bad["unexpected_field"] = 1
    path = _write_jsonl(tmp_path, [json.dumps(bad)])

    report = load_traces_report(path)
    assert report.traces == []
    assert len(report.errors) == 1


def test_step_i_must_strictly_increase(tmp_path: Path) -> None:
    bad = json.loads(json.dumps(VALID_TRACE))
    bad["steps"][1]["i"] = 0  # same as steps[0], not strictly increasing
    path = _write_jsonl(tmp_path, [json.dumps(bad)])

    report = load_traces_report(path)
    assert report.traces == []
    assert any(
        e.json_path == "$.steps[1].i" and "strictly increasing" in e.message for e in report.errors
    )


def test_duplicate_trace_id_reported(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, [json.dumps(VALID_TRACE), json.dumps(VALID_TRACE)])

    report = load_traces_report(path)
    assert len(report.traces) == 1
    assert len(report.errors) == 1
    err = report.errors[0]
    assert err.line_no == 2
    assert err.json_path == "$.trace_id"
    assert "duplicate" in err.message


def test_non_strict_skips_bad_lines(tmp_path: Path) -> None:
    bad = json.loads(json.dumps(VALID_TRACE))
    bad["trace_id"] = "t2"
    bad["task"]["domain"] = "spaceship"
    path = _write_jsonl(tmp_path, [json.dumps(VALID_TRACE), json.dumps(bad)])

    traces = load_traces(path, strict=False)
    assert [t.trace_id for t in traces] == ["t1"]


def test_strict_aborts_on_first_bad_line(tmp_path: Path) -> None:
    bad = json.loads(json.dumps(VALID_TRACE))
    bad["trace_id"] = "t2"
    bad["task"]["domain"] = "spaceship"
    path = _write_jsonl(tmp_path, [json.dumps(VALID_TRACE), json.dumps(bad)])

    with pytest.raises(TraceValidationError) as excinfo:
        load_traces(path, strict=True)
    assert excinfo.value.errors[0].line_no == 2


def test_invalid_json_line(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, ["not json"])
    report = load_traces_report(path)
    assert report.traces == []
    assert "invalid JSON" in report.errors[0].message


def test_probe_merge_continues_i_and_sets_environment_role(tmp_path: Path) -> None:
    trace_path = _write_jsonl(tmp_path, [json.dumps(VALID_TRACE)])
    probes_path = tmp_path / "probes.jsonl"
    probes_path.write_text(
        json.dumps(
            {
                "trace_id": "t1",
                "name": "calendar.get_event",
                "args": {},
                "ok": True,
                "output": {"status": "confirmed"},
                "error": None,
                "ts": "2026-01-01T00:00:02+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = load_traces_report(trace_path, probes=probes_path)
    assert report.warnings == []
    trace = report.traces[0]
    assert len(trace.steps) == 3
    probe_step = trace.steps[-1]
    assert probe_step.i == 2  # continues after the last step's i == 1
    assert probe_step.kind == "state_probe"
    assert probe_step.role == "environment"
    assert probe_step.name == "calendar.get_event"


def test_probe_for_unknown_trace_id_is_a_warning(tmp_path: Path) -> None:
    trace_path = _write_jsonl(tmp_path, [json.dumps(VALID_TRACE)])
    probes_path = tmp_path / "probes.jsonl"
    probes_path.write_text(
        json.dumps(
            {
                "trace_id": "does-not-exist",
                "name": "calendar.get_event",
                "ok": True,
                "ts": "2026-01-01T00:00:02+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = load_traces_report(trace_path, probes=probes_path)
    assert len(report.traces) == 1
    assert len(report.traces[0].steps) == 2  # unmatched probe was not appended
    assert len(report.warnings) == 1
    assert "does-not-exist" in report.warnings[0]


def test_pair_results_nearest_preceding_call_and_orphan() -> None:
    steps = [
        Step(i=0, ts="2026-01-01T00:00:00+00:00", kind="tool_call", role="agent", name="a"),
        Step(i=1, ts="2026-01-01T00:00:01+00:00", kind="tool_result", role="tool", name="a"),
        Step(i=2, ts="2026-01-01T00:00:02+00:00", kind="tool_call", role="agent", name="a"),
        Step(i=3, ts="2026-01-01T00:00:03+00:00", kind="tool_result", role="tool", name="a"),
        Step(i=4, ts="2026-01-01T00:00:04+00:00", kind="tool_result", role="tool", name="b"),
    ]
    pairing = pair_results(steps)
    assert pairing == {1: 0, 3: 2, 4: None}


def test_dump_trace_validates_before_returning(tmp_path: Path) -> None:
    trace = Trace.model_validate(VALID_TRACE)
    dumped = dump_trace(trace)
    # dump_trace's own schema validation must accept what it just produced.
    path = _write_jsonl(tmp_path, [dumped])
    reloaded = load_traces_report(path)
    assert reloaded.errors == []
    assert reloaded.traces[0].trace_id == "t1"
