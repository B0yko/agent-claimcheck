"""The `agent-claimcheck` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from agent_claimcheck.resources import ResourceNotFoundError
from agent_claimcheck.schema import TraceValidationError, load_traces_report

#: `train --calibrate` keys accepted by this build.
_SUPPORTED_CALIBRATE_KEYS = ("rules",)

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


@app.command()
def train(
    input_file: str = typer.Argument(
        ..., help="Labelled trace file (JSON Lines) path or packaged alias."
    ),
    out: str = typer.Option(
        ..., "--out", help="Output directory for lr-v1.json and calibration.json."
    ),
    calibrate: str | None = typer.Option(
        None,
        "--calibrate",
        help="Comma-separated raw-score calibrators to fit in addition to classifier-lr.",
    ),
) -> None:
    """Train the classifier-lr artifact and its calibrator(s) on labelled traces."""
    from agent_claimcheck.calibration import CalibratorSet, fit_calibrator
    from agent_claimcheck.claims import ClaimExtractor, success_claims
    from agent_claimcheck.detectors.classifier import oof_predictions, train_lr
    from agent_claimcheck.detectors.rules import RulesDetector
    from agent_claimcheck.redact import DetectorView, detector_view, resolve_claims
    from agent_claimcheck.rules.engine import builtin_packs

    calibrate_keys = [k.strip() for k in calibrate.split(",") if k.strip()] if calibrate else []
    for key in calibrate_keys:
        if key not in _SUPPORTED_CALIBRATE_KEYS:
            error_console.print(
                f"error: --calibrate does not support {key!r} yet "
                f"(only {', '.join(_SUPPORTED_CALIBRATE_KEYS)!r})"
            )
            raise typer.Exit(code=2)

    try:
        report = load_traces_report(input_file)
    except (OSError, ResourceNotFoundError) as exc:
        error_console.print(f"error: {exc}")
        raise typer.Exit(code=2) from None
    for err in report.errors:
        console.print(f"line {err.line_no}: {err.json_path}: {err.message}")

    extractor = ClaimExtractor(list(builtin_packs().values()))
    views: list[DetectorView] = []
    labels: list[int] = []
    domains: list[str] = []
    n_unlabelled = 0
    n_no_claim = 0
    for t in report.traces:
        if t.ground_truth is None or t.ground_truth.outcome == "unknown":
            n_unlabelled += 1
            continue
        view = resolve_claims(detector_view(t), extractor)
        if not success_claims(view.final_claim.claims):
            n_no_claim += 1
            continue
        views.append(view)
        labels.append(1 if t.ground_truth.outcome == "success" else 0)
        domains.append(t.task.domain)

    console.print(
        f"training on {len(views)} traces "
        f"(skipped {n_unlabelled} unlabelled, {n_no_claim} without a success claim)"
    )
    if not views:
        error_console.print("error: no labelled traces with a success claim to train on")
        raise typer.Exit(code=2)

    artifact = train_lr(views, labels, domains, seed=0)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lr-v1.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    oof = oof_predictions(views, labels, seed=0)
    calibrators = {
        "classifier-lr": fit_calibrator("classifier-lr", oof, labels, fitted_on=str(input_file))
    }
    if "rules" in calibrate_keys:
        rules_detector = RulesDetector()
        raw_p: list[float] = []
        raw_labels: list[int] = []
        for view, label in zip(views, labels, strict=True):
            rules_output = rules_detector.score(view)
            if rules_output.abstain:
                continue
            raw_p.append(rules_output.p_success)
            raw_labels.append(label)
        if raw_p:
            calibrators["rules"] = fit_calibrator(
                "rules", raw_p, raw_labels, fitted_on=str(input_file)
            )

    calibrator_set = CalibratorSet(
        version=1,
        fitted_on=str(input_file),
        base_rate=sum(labels) / len(labels),
        calibrators=calibrators,
    )
    calibrator_set.save(out_dir / "calibration.json")
    console.print(f"wrote {out_dir}/lr-v1.json, {out_dir}/calibration.json")


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
