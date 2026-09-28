"""`agent-claimcheck validate`: exit codes and error reporting."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_claimcheck.cli import app

runner = CliRunner()

VALID_TRACE = {
    "schema": "agent-trace/v1",
    "trace_id": "t1",
    "source": "test/0.0.1",
    "task": {"id": "task-1", "domain": "booking", "instruction": "book a table"},
    "steps": [],
    "final_claim": {"text": "Booked.", "claims": [{"type": "booked", "subject": {}}]},
}


def test_validate_exits_0_on_a_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "good.jsonl"
    path.write_text(json.dumps(VALID_TRACE) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0


def test_validate_exits_2_on_an_invalid_file(tmp_path: Path) -> None:
    bad = json.loads(json.dumps(VALID_TRACE))
    bad["task"]["domain"] = "spaceship"
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 2
    assert "$.task.domain" in result.output


def test_validate_exits_2_on_a_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "missing.jsonl")])
    assert result.exit_code == 2


def test_validate_exits_2_on_a_usage_error() -> None:
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 2


def test_validate_example_browser_alias() -> None:
    result = runner.invoke(app, ["validate", "example:browser"])
    assert result.exit_code == 0
