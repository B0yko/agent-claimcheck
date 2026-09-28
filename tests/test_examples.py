"""The packaged example sets."""

from __future__ import annotations

from agent_claimcheck.schema import load_traces_report


def test_browser_demo_has_24_valid_traces() -> None:
    report = load_traces_report("example:browser")
    assert report.errors == []
    assert len(report.traces) == 24
    assert all(t.task.domain == "browser" for t in report.traces)
    assert len({t.trace_id for t in report.traces}) == 24
