"""The `agent-claimcheck` command-line interface."""

from __future__ import annotations

import typer
from rich.console import Console

from agent_claimcheck.resources import ResourceNotFoundError
from agent_claimcheck.schema import TraceValidationError, load_traces_report

app = typer.Typer(add_completion=False, no_args_is_help=True)
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
