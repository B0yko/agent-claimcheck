"""The Python API snippet runs verbatim, character for character."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_claimcheck.resources import path as resource_path


def test_python_api_snippet_runs_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "traces.jsonl").write_bytes(resource_path("examples/traces.jsonl").read_bytes())
    (tmp_path / "probes.jsonl").write_bytes(resource_path("examples/probes.jsonl").read_bytes())
    monkeypatch.chdir(tmp_path)

    from agent_claimcheck import load_traces, Checker  # noqa: I001 - kept verbatim, as documented

    checker = Checker(detector="cascade-offline")  # or Checker.from_config("claimcheck.toml")
    for r in checker.check(load_traces("traces.jsonl"), probes="probes.jsonl"):
        print(r.trace_id, r.verdict, r.p_success, r.confidence, r.reasons[0].detail)

    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert len(lines) == 12
    assert lines[0].startswith("booking-01 ")
