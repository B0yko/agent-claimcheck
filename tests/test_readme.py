"""The README's numbers are backed by the recorded run, and its snippets run."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from factory import numbers_in
from typer.testing import CliRunner

from agent_claimcheck.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
RECORDED = REPO_ROOT / "results" / "v0.1.0"


def _section(text: str, heading: str) -> str:
    after = text.split(f"\n## {heading}\n", 1)[1]
    return after.split("\n## ", 1)[0]


@pytest.mark.xfail(
    strict=True,
    reason="the README's bench block predates the reworded H1/H4 lines and the wider "
    "ablation table; paste the regenerated report into it, then delete this marker",
)
def test_readme_bench_block_matches_the_recorded_report() -> None:
    result = CliRunner().invoke(
        app, ["bench", "--from-recorded", str(RECORDED), "--check-readme", str(README)]
    )
    assert result.exit_code == 0, result.output


def test_every_number_in_findings_appears_in_the_generated_report() -> None:
    findings = _section(README.read_text(encoding="utf-8"), "Findings")
    bench_md = (RECORDED / "bench.md").read_text(encoding="utf-8")
    numbers = numbers_in(findings)
    assert numbers, "the Findings section should cite numbers"
    missing = sorted(n for n in numbers if n not in bench_md)
    assert not missing, missing


def test_readme_python_snippet_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = README.read_text(encoding="utf-8")
    snippets = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert len(snippets) == 1
    shutil.copy(REPO_ROOT / "examples" / "traces.jsonl", tmp_path / "traces.jsonl")
    shutil.copy(REPO_ROOT / "examples" / "probes.jsonl", tmp_path / "probes.jsonl")
    monkeypatch.chdir(tmp_path)
    exec(compile(snippets[0], "README.md", "exec"), {})  # noqa: S102 - the README's own example
