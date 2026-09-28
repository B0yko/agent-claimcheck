"""Tests for the `agent-claimcheck bench` command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_claimcheck.bench.report import README_END, README_START
from agent_claimcheck.cli import app

runner = CliRunner()


def test_requires_exactly_one_mode() -> None:
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 2

    result = runner.invoke(app, ["bench", "--offline", "--from-recorded", "x", "--out", "y"])
    assert result.exit_code == 2


def test_offline_requires_out(tmp_path: Path) -> None:
    result = runner.invoke(app, ["bench", "--offline"])
    assert result.exit_code == 2
    assert "--out" in result.output


def test_offline_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "recorded"
    result = runner.invoke(app, ["bench", "--offline", "--out", str(out)])
    assert result.exit_code == 0, result.output

    for name in ("bench.json", "bench.md", "reliability.svg", "histogram.svg", "run.json"):
        assert (out / name).exists()

    bench_json = json.loads((out / "bench.json").read_text(encoding="utf-8"))
    assert bench_json["dataset"]["n_total"] == 300
    assert "rules" in bench_json["detectors"]
    assert "### Table A" in result.output


def test_from_recorded_reproduces_the_offline_run(
    offline_recorded_dir: Path, tmp_path: Path
) -> None:
    first_out = tmp_path / "replay-1"
    second_out = tmp_path / "replay-2"
    for out in (first_out, second_out):
        result = runner.invoke(
            app, ["bench", "--from-recorded", str(offline_recorded_dir), "--out", str(out)]
        )
        assert result.exit_code == 0, result.output

    for name in ("bench.md", "bench.json", "reliability.svg", "histogram.svg"):
        assert (first_out / name).read_text(encoding="utf-8") == (second_out / name).read_text(
            encoding="utf-8"
        )


def test_check_readme_pass_and_fail(offline_recorded_dir: Path, tmp_path: Path) -> None:
    first = runner.invoke(
        app, ["bench", "--from-recorded", str(offline_recorded_dir), "--out", str(tmp_path / "r")]
    )
    assert first.exit_code == 0
    bench_md = (tmp_path / "r" / "bench.md").read_text(encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(
        f"# project\n\n{README_START}\nstale content\n{README_END}\n", encoding="utf-8"
    )
    failing = runner.invoke(
        app,
        [
            "bench",
            "--from-recorded",
            str(offline_recorded_dir),
            "--check-readme",
            str(readme),
        ],
    )
    assert failing.exit_code == 1
    assert "stale content" in failing.output

    readme.write_text(
        f"# project\n\n{README_START}\n{bench_md.strip()}\n{README_END}\n", encoding="utf-8"
    )
    passing = runner.invoke(
        app,
        [
            "bench",
            "--from-recorded",
            str(offline_recorded_dir),
            "--check-readme",
            str(readme),
        ],
    )
    assert passing.exit_code == 0
