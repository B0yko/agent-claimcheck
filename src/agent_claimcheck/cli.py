"""The `agent-claimcheck` command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from agent_claimcheck.resources import ResourceNotFoundError
from agent_claimcheck.schema import TraceValidationError, load_traces_report

app = typer.Typer(add_completion=False, no_args_is_help=True)
dataset_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
console = Console()
error_console = Console(stderr=True)


@app.callback()
def _callback() -> None:
    """Check AI agent success claims against trace evidence."""


@app.command()
def validate(
    file: str = typer.Argument(..., help="Trace file (JSON Lines) path or packaged alias."),
    strict: bool = typer.Option(False, "--strict", help="Abort on the first invalid line."),
    probes: str | None = typer.Option(
        None, "--probes", help="Probes JSON Lines file to merge into matching traces."
    ),
) -> None:
    """Validate a trace file against the agent-trace/v1 schema."""
    try:
        report = load_traces_report(file, strict=strict, probes=probes)
    except TraceValidationError as exc:
        for err in exc.errors:
            console.print(f"line {err.line_no}: {err.json_path}: {err.message}")
        raise typer.Exit(code=2) from None
    except (OSError, ResourceNotFoundError) as exc:
        error_console.print(f"error: {exc}")
        raise typer.Exit(code=2) from None

    for err in report.errors:
        console.print(f"line {err.line_no}: {err.json_path}: {err.message}")
    for warning in report.warnings:
        console.print(f"warning: {warning}")

    raise typer.Exit(code=2 if report.errors else 0)


@dataset_app.command("generate")
def dataset_generate(
    seed: int = typer.Option(20260924, "--seed", help="Seed for the deterministic generator."),
    out: str = typer.Option("benchmark/v1", "--out", help="Output directory."),
) -> None:
    """Generate the synthetic benchmark: traces, manifest and dataset card."""
    from agent_claimcheck.bench.generator import generate

    dataset = generate(seed)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces.train.jsonl").write_text(dataset.train_jsonl, encoding="utf-8")
    (out_dir / "traces.test.jsonl").write_text(dataset.test_jsonl, encoding="utf-8")
    (out_dir / "manifest.json").write_text(dataset.manifest_json, encoding="utf-8")
    (out_dir / "DATASET_CARD.md").write_text(dataset.card_markdown, encoding="utf-8")
    console.print(
        f"wrote {out_dir}/traces.train.jsonl, traces.test.jsonl, manifest.json, DATASET_CARD.md"
    )


@dataset_app.command("validate")
def dataset_validate(
    dir_: str | None = typer.Argument(
        None, help="Benchmark directory (default: the packaged benchmark)."
    ),
) -> None:
    """Validate a generated benchmark directory against its manifest and the design."""
    from agent_claimcheck.bench.generator.validate import validate_dataset

    failures = validate_dataset(dir_)
    for failure in failures:
        console.print(failure)
    raise typer.Exit(code=2 if failures else 0)


def main() -> None:
    """Entry point for the `agent-claimcheck` script: usage errors exit 2."""
    try:
        app()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive fallback
        error_console.print(f"error: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
