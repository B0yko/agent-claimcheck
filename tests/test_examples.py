"""The packaged example sets."""

from __future__ import annotations

from collections import Counter

from agent_claimcheck.schema import load_traces_report


def test_browser_demo_has_24_valid_traces() -> None:
    report = load_traces_report("example:browser")
    assert report.errors == []
    assert len(report.traces) == 24
    assert all(t.task.domain == "browser" for t in report.traces)
    assert len({t.trace_id for t in report.traces}) == 24


def test_mixed_has_12_valid_traces_4_per_domain() -> None:
    report = load_traces_report("example:mixed")
    assert report.errors == []
    assert len(report.traces) == 12
    assert len({t.trace_id for t in report.traces}) == 12
    by_domain = Counter(t.task.domain for t in report.traces)
    assert by_domain == {"booking": 4, "crm": 4, "coding": 4}
