"""The `agent-claimcheck` command-line interface."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from agent_claimcheck.gate import UnknownPriceError
from agent_claimcheck.ledger import Price
from agent_claimcheck.resources import ResourceNotFoundError
from agent_claimcheck.schema import TraceValidationError, load_traces_report

#: `train --calibrate` keys accepted by this build.
_SUPPORTED_CALIBRATE_KEYS = ("rules", "judge")

#: `train` needs at least this many labelled traces of each outcome: the
#: classifier's cross-validation splits every outcome into five folds.
_MIN_PER_OUTCOME = 5

#: Raw scores this close are one value as far as a calibrator can tell.
_SCORE_DECIMALS = 9

_Verdict = Literal["verified", "false_success", "unverifiable", "skipped"]

#: Verdicts `check --fail-on` and the summary line may name.
_VERDICTS: tuple[_Verdict, ...] = ("verified", "false_success", "unverifiable", "skipped")

app = typer.Typer(add_completion=False, no_args_is_help=True)
dataset_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
console = Console()
#: Diagnostics are single lines that scripts grep: never wrapped at the
#: terminal width, and never read as rich markup (a `[judge]` table name or an
#: exception's bracketed text would otherwise vanish).
error_console = Console(stderr=True, soft_wrap=True, markup=False)


@app.callback()
def _callback() -> None:
    """Check AI agent success claims against trace evidence."""


def _distinct_scores(raw_p: list[float]) -> int:
    return len({round(p, _SCORE_DECIMALS) for p in raw_p})


def _calibration_skipped(name: str, raw_p: list[float]) -> bool:
    """Print a note and return True when `raw_p` cannot support a calibrator."""
    distinct = _distinct_scores(raw_p)
    if distinct >= 2:
        return False
    console.print(
        f"note: skipped the {name} calibrator: its raw scores take {distinct} distinct value(s), "
        "and a calibrator needs at least 2",
        soft_wrap=True,
    )
    return True


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
    config: str | None = typer.Option(
        None, "--config", help="claimcheck.toml path (only read for --calibrate judge)."
    ),
) -> None:
    """Train the classifier-lr artifact and its calibrator(s) on labelled traces."""
    from agent_claimcheck.calibration import Calibrator, CalibratorSet, fit_calibrator
    from agent_claimcheck.checker import Checker
    from agent_claimcheck.claims import ClaimExtractor, success_claims
    from agent_claimcheck.config import load_config
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
    n_success = sum(labels)
    n_failure = len(labels) - n_success
    if min(n_success, n_failure) < _MIN_PER_OUTCOME:
        error_console.print(
            f"error: need at least {_MIN_PER_OUTCOME} labelled traces of each outcome to train, "
            f"got {n_success} success and {n_failure} failure "
            "(among those with a success claim)"
        )
        raise typer.Exit(code=2)

    artifact = train_lr(views, labels, domains, seed=0)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lr-v1.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    oof = oof_predictions(views, labels, seed=0)
    calibrators: dict[str, Calibrator] = {}
    if not _calibration_skipped("classifier-lr", oof):
        calibrators["classifier-lr"] = fit_calibrator(
            "classifier-lr", oof, labels, fitted_on=str(input_file)
        )
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
        if not _calibration_skipped("rules", raw_p):
            calibrators["rules"] = fit_calibrator(
                "rules", raw_p, raw_labels, fitted_on=str(input_file)
            )

    if "judge" in calibrate_keys:
        try:
            judge_checker = Checker("judge", config=load_config(config))
        except (OSError, ValueError) as exc:
            error_console.print(f"error: {exc}")
            raise typer.Exit(code=2) from None
        judge_detector = judge_checker.detector
        judge_raw_p: list[float] = []
        judge_raw_labels: list[int] = []
        for view, label in zip(views, labels, strict=True):
            judge_output = judge_detector.score(view)
            if judge_output.abstain:
                continue
            judge_raw_p.append(judge_output.p_success)
            judge_raw_labels.append(label)
        console.print(
            f"judge {judge_detector.name}: scored {len(judge_raw_p)}/{len(views)} traces "
            f"(the rest abstained)"
        )
        if not _calibration_skipped(judge_detector.name, judge_raw_p):
            calibrators[judge_detector.name] = fit_calibrator(
                judge_detector.name, judge_raw_p, judge_raw_labels, fitted_on=str(input_file)
            )

    calibrator_set = CalibratorSet(
        version=1,
        fitted_on=str(input_file),
        base_rate=sum(labels) / len(labels),
        calibrators=calibrators,
    )
    calibrator_set.save(out_dir / "calibration.json")
    console.print(f"wrote {out_dir}/lr-v1.json, {out_dir}/calibration.json")


@app.command()
def check(
    input_file: str = typer.Argument(
        ..., metavar="INPUT", help="Trace file (JSON Lines) path or packaged alias."
    ),
    config: str | None = typer.Option(None, "--config", help="claimcheck.toml path."),
    probes: str | None = typer.Option(
        None, "--probes", help="Probes JSON Lines file to merge into matching traces."
    ),
    detector: str = typer.Option(
        "cascade-offline",
        "--detector",
        help="Detector to run: cascade-offline, rules, classifier, judge or cascade.",
    ),
    rules: list[str] = typer.Option(  # noqa: B008 - typer's documented repeatable-option idiom
        [], "--rules", help="Extra rule pack YAML file(s), on top of the built-in packs."
    ),
    prompt: str | None = typer.Option(
        None, "--prompt", help="Custom judge prompt file (judge/cascade only)."
    ),
    calibration: str | None = typer.Option(
        None,
        "--calibration",
        help="CalibratorSet JSON file (overrides [classifier] calibration and the built-in set).",
    ),
    out: str | None = typer.Option(
        None, "--out", help="Write every result as a JSON Lines file at this path."
    ),
    format_: str = typer.Option(
        "table", "--format", help="How results print to stdout: table, json or jsonl."
    ),
    fail_on: str = typer.Option(
        "false_success",
        "--fail-on",
        help="Comma-separated verdicts (verified, false_success, unverifiable, skipped) "
        "that make the exit code 1.",
    ),
    max_usd: float | None = typer.Option(
        None, "--max-usd", help="Live judge budget cap in USD for this run."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Neither read nor write the judge response cache."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Abort with exit 2 on the first invalid trace line."
    ),
    price_in: float | None = typer.Option(
        None,
        "--price-in",
        help="Judge price per million input tokens (with --price-out); beats the config.",
    ),
    price_out: float | None = typer.Option(
        None,
        "--price-out",
        help="Judge price per million output tokens (with --price-in); beats the config.",
    ),
) -> None:
    """Check success claims against trace evidence and gate each trace.

    Exit codes: 0 when nothing matched --fail-on; 1 when a verdict named by
    --fail-on occurred; 2 on a usage or input error, including any trace line
    that failed validation (unless --strict aborts earlier) and an input with
    no valid trace. Exit 2 wins over exit 1.
    """
    from agent_claimcheck.checker import Checker, dump_result
    from agent_claimcheck.config import load_config
    from agent_claimcheck.judge.render import PromptError
    from agent_claimcheck.rules.engine import RulePackError

    if format_ not in ("table", "json", "jsonl"):
        error_console.print(f"error: --format must be one of table, json, jsonl, got {format_!r}")
        raise typer.Exit(code=2)

    fail_on_set = {v.strip() for v in fail_on.split(",") if v.strip()}
    unknown_verdicts = fail_on_set - set(_VERDICTS)
    if unknown_verdicts:
        error_console.print(
            f"error: --fail-on has unknown verdict(s): {', '.join(sorted(unknown_verdicts))}"
        )
        raise typer.Exit(code=2)

    price = _price_from_flags(price_in, price_out)

    try:
        report = load_traces_report(input_file, strict=strict)
    except TraceValidationError as exc:
        for err in exc.errors:
            error_console.print(f"line {err.line_no}: {err.json_path}: {err.message}")
        error_console.print("error: aborting on the first invalid line (--strict)")
        raise typer.Exit(code=2) from None
    except (OSError, ResourceNotFoundError) as exc:
        error_console.print(f"error: {exc}")
        raise typer.Exit(code=2) from None
    for err in report.errors:
        error_console.print(f"line {err.line_no}: {err.json_path}: {err.message}")
    if not report.traces:
        error_console.print(f"error: no valid traces loaded from {input_file}")
        raise typer.Exit(code=2)

    try:
        checker = Checker(
            detector,
            config=load_config(config),
            rules=rules,
            prompt=prompt,
            calibration=calibration,
            max_usd=max_usd,
            price=price,
            use_cache=not no_cache,
        )
        checker.require_price()
    except UnknownPriceError as exc:
        error_console.print(f"error: {exc.args[0]}")
        raise typer.Exit(code=2) from None
    except (OSError, ValueError, RulePackError, PromptError, ResourceNotFoundError) as exc:
        error_console.print(f"error: {exc}")
        raise typer.Exit(code=2) from None

    try:
        results = list(checker.check(report.traces, probes=probes))
    except (OSError, ResourceNotFoundError) as exc:
        error_console.print(f"error: {exc}")
        raise typer.Exit(code=2) from None

    calibrators_applied = any(r.calibrated for r in results)
    if (
        checker.calibrators_builtin
        and calibrators_applied
        and input_file not in ("bench:train", "bench:test")
    ):
        error_console.print(
            "Note: built-in calibrators were fitted on the synthetic benchmark; "
            "fit your own with `agent-claimcheck train`."
        )

    counts = Counter(r.verdict for r in results)
    summary = f"{len(results)} traces: " + ", ".join(
        f"{counts[v]} {v}" for v in _VERDICTS if counts.get(v)
    )

    if format_ == "table":
        table = Table("trace_id", "domain", "verdict", "p_success", "top reason")
        for r in results:
            p_text = "" if r.p_success is None else f"{r.p_success:.2f}"
            top_reason = r.reasons[0].detail if r.reasons else ""
            table.add_row(r.trace_id, r.domain, r.verdict, p_text, top_reason)
        console.print(table)
        console.print(summary)
    elif format_ == "jsonl":
        for r in results:
            print(dump_result(r))
        error_console.print(summary)
    else:
        print(json.dumps([json.loads(dump_result(r)) for r in results], indent=2))
        error_console.print(summary)

    if out is not None:
        lines = [dump_result(r) for r in results]
        Path(out).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if report.errors:
        error_console.print(
            f"error: {len(report.errors)} invalid trace line(s) skipped; "
            "exiting 2 (pass --strict to abort on the first one)"
        )
        raise typer.Exit(code=2)
    raise typer.Exit(code=1 if fail_on_set & set(counts) else 0)


def _price_from_flags(price_in: float | None, price_out: float | None) -> Price | None:
    """A `Price` from `--price-in`/`--price-out`: both or neither, never negative."""
    if price_in is None and price_out is None:
        return None
    if price_in is None or price_out is None:
        error_console.print("error: --price-in and --price-out must be given together")
        raise typer.Exit(code=2)
    if price_in < 0 or price_out < 0:
        error_console.print("error: --price-in and --price-out must not be negative")
        raise typer.Exit(code=2)
    return Price(price_in_per_m=price_in, price_out_per_m=price_out)


def _run_bench_live(
    *,
    judges: str | None,
    ablation_judge: str | None,
    ablation_prompt: str,
    run_name: str | None,
    max_usd: float | None,
    concurrency: int,
    limit: int | None,
    dry_run: bool,
    allow_partial: bool,
    price_in: float | None,
    price_out: float | None,
    hardware: str | None,
    out: str | None,
) -> str:
    """Validate `bench --live`'s options, then run the dry-run or the live run.

    Returns the output directory for the caller's shared report-building
    tail. `--dry-run` prints its plan and exits before returning.
    """
    import hashlib
    import time
    from datetime import UTC, datetime

    from agent_claimcheck import __version__
    from agent_claimcheck.bench.live import (
        BudgetPreflightError,
        dry_run_plan,
        format_dry_run,
        format_spend_summary,
        run_live,
    )
    from agent_claimcheck.config import cache_dir, load_config, resolve_api_key
    from agent_claimcheck.config import ledger_path as resolve_ledger_path
    from agent_claimcheck.resources import path as resource_path

    missing = [
        name
        for name, value in (
            ("--judges", judges),
            ("--ablation-judge", ablation_judge),
            ("--run-name", run_name),
            ("--max-usd", max_usd),
        )
        if value is None
    ]
    if not dry_run:
        if out is None:
            missing.append("--out")
        if hardware is None:
            missing.append("--hardware")
    if missing:
        error_console.print(f"error: --live requires {', '.join(missing)}")
        raise typer.Exit(code=2)
    assert judges is not None
    assert ablation_judge is not None
    assert run_name is not None
    assert max_usd is not None

    judges_list = [j.strip() for j in judges.split(",") if j.strip()]
    if not judges_list:
        error_console.print("error: --judges must name at least one model")
        raise typer.Exit(code=2)

    price_override = _price_from_flags(price_in, price_out)

    cfg = load_config(None)
    base_url = cfg.judge.base_url
    api_key = resolve_api_key(base_url)

    train_full = load_traces_report("bench:train").traces
    test_full = load_traces_report("bench:test").traces

    try:
        if dry_run:
            rows, grand_total = dry_run_plan(
                train_full,
                test_full,
                judges=judges_list,
                ablation_judge=ablation_judge,
                ablation_prompt=ablation_prompt,
                limit=limit,
                base_url=base_url,
                api_key=api_key,
                price_override=price_override,
            )
            console.print(format_dry_run(rows, grand_total))
            raise typer.Exit(code=0)

        assert out is not None
        assert hardware is not None
        test_bytes = resource_path("bench:test").read_bytes()
        dataset_sha256 = hashlib.sha256(test_bytes).hexdigest()
        date = datetime.now(UTC).date().isoformat()
        command = (
            f"agent-claimcheck bench --live --judges {judges} "
            f"--ablation-judge {ablation_judge} --ablation-prompt {ablation_prompt} "
            f"--run-name {run_name} --max-usd {max_usd} --concurrency {concurrency}"
            + (f" --limit {limit}" if limit is not None else "")
            + (" --allow-partial" if allow_partial else "")
            + f" --out {out}"
        )
        start = time.perf_counter()
        summary = run_live(
            train_full,
            test_full,
            out,
            judges=judges_list,
            ablation_judge=ablation_judge,
            ablation_prompt=ablation_prompt,
            run_name=run_name,
            max_usd=max_usd,
            concurrency=concurrency,
            limit=limit,
            allow_partial=allow_partial,
            base_url=base_url,
            api_key=api_key,
            price_override=price_override,
            timeout_s=cfg.judge.timeout_s,
            ledger_path=resolve_ledger_path(cfg.ledger),
            ledger_cap_usd=cfg.ledger_cap_usd,
            cache_dir=cache_dir(),
            command=command,
            date=date,
            hardware=hardware,
            package_version=__version__,
            dataset_sha256=dataset_sha256,
        )
    except BudgetPreflightError as exc:
        error_console.print(f"error: {exc}")
        raise typer.Exit(code=2) from None
    except UnknownPriceError as exc:
        error_console.print(f"error: {exc}")
        raise typer.Exit(code=2) from None

    console.print(f"live run finished in {time.perf_counter() - start:.1f}s")
    console.print(format_spend_summary(summary))
    if summary.budget_exhausted:
        error_console.print(
            "warning: budget exhausted mid-run; remaining calls abstained (budget_exhausted)"
        )
    return out


@app.command()
def bench(
    offline: bool = typer.Option(
        False, "--offline", help="Score every offline detector on the packaged benchmark."
    ),
    from_recorded: str | None = typer.Option(
        None,
        "--from-recorded",
        help="Recorded directory or alias (e.g. recorded:v0.1.0) to replay.",
    ),
    live: bool = typer.Option(
        False, "--live", help="Run live judge benchmarks against a chat-completions API."
    ),
    judges: str | None = typer.Option(
        None, "--judges", help="Comma-separated judge model ids (--live)."
    ),
    ablation_judge: str | None = typer.Option(
        None, "--ablation-judge", help="Model id the prompt ablation runs on (--live)."
    ),
    ablation_prompt: str = typer.Option(
        "claim-by-claim",
        "--ablation-prompt",
        help="Prompt name (or path) the ablation judge uses (--live).",
    ),
    run_name: str | None = typer.Option(
        None, "--run-name", help="Run identifier, recorded on every ledger line (--live)."
    ),
    max_usd: float | None = typer.Option(
        None, "--max-usd", help="Budget cap in USD for this run (--live)."
    ),
    concurrency: int = typer.Option(
        8, "--concurrency", help="Threads scoring one judge's calls at a time (--live)."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Score only the first N traces per split (--live)."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print call counts and worst-case reservations, send nothing (--live).",
    ),
    allow_partial: bool = typer.Option(
        False,
        "--allow-partial",
        help="Run even when the dry-run reservation exceeds --max-usd (--live).",
    ),
    price_in: float | None = typer.Option(
        None,
        "--price-in",
        help="Override price per million input tokens for every judge (--live).",
    ),
    price_out: float | None = typer.Option(
        None,
        "--price-out",
        help="Override price per million output tokens for every judge (--live).",
    ),
    hardware: str | None = typer.Option(
        None, "--hardware", help="Hardware string recorded in run.json (--live)."
    ),
    out: str | None = typer.Option(
        None,
        "--out",
        help="Output directory (required with --offline/--live; optional with --from-recorded).",
    ),
    check_readme: str | None = typer.Option(
        None, "--check-readme", help="README to check the bench:start/bench:end block against."
    ),
) -> None:
    """Run or replay the benchmark and render the README numbers."""
    import hashlib
    import time
    from datetime import UTC, datetime

    from agent_claimcheck import __version__
    from agent_claimcheck.bench.report import (
        build_report,
        check_readme_diff,
        histogram_svg_for,
        reliability_svg_for,
        verify_judge_requests,
    )
    from agent_claimcheck.bench.runner import run_offline
    from agent_claimcheck.resources import path as resource_path

    if sum((offline, from_recorded is not None, live)) != 1:
        error_console.print("error: pass exactly one of --offline, --from-recorded or --live")
        raise typer.Exit(code=2)
    if dry_run and not live:
        error_console.print("error: --dry-run requires --live")
        raise typer.Exit(code=2)

    if offline:
        if out is None:
            error_console.print("error: --offline requires --out")
            raise typer.Exit(code=2)
        train = load_traces_report("bench:train").traces
        test = load_traces_report("bench:test").traces
        test_bytes = resource_path("bench:test").read_bytes()
        dataset_sha256 = hashlib.sha256(test_bytes).hexdigest()
        start = time.perf_counter()
        run_offline(
            train,
            test,
            out,
            command="agent-claimcheck bench --offline --out " + out,
            date=datetime.now(UTC).date().isoformat(),
            package_version=__version__,
            dataset_sha256=dataset_sha256,
        )
        console.print(
            f"scored {len(train)} train + {len(test)} test traces in {out} "
            f"({time.perf_counter() - start:.1f}s)"
        )
        recorded_dir = out
    elif live:
        recorded_dir = _run_bench_live(
            judges=judges,
            ablation_judge=ablation_judge,
            ablation_prompt=ablation_prompt,
            run_name=run_name,
            max_usd=max_usd,
            concurrency=concurrency,
            limit=limit,
            dry_run=dry_run,
            allow_partial=allow_partial,
            price_in=price_in,
            price_out=price_out,
            hardware=hardware,
            out=out,
        )
    else:
        assert from_recorded is not None
        recorded_dir = from_recorded
        mismatches = verify_judge_requests(recorded_dir)
        if mismatches:
            for m in mismatches:
                error_console.print(f"error: {m}")
            raise typer.Exit(code=2)

    bench_json, bench_md = build_report(recorded_dir)

    if out is not None:
        write_dir = Path(out)
        write_dir.mkdir(parents=True, exist_ok=True)
        (write_dir / "bench.json").write_text(
            json.dumps(bench_json, indent=2, sort_keys=True, ensure_ascii=False) + "\n", "utf-8"
        )
        (write_dir / "bench.md").write_text(bench_md, encoding="utf-8")
        (write_dir / "reliability.svg").write_text(
            reliability_svg_for(recorded_dir), encoding="utf-8"
        )
        (write_dir / "histogram.svg").write_text(histogram_svg_for(recorded_dir), encoding="utf-8")

    console.print(bench_md)

    if check_readme is not None:
        diff = check_readme_diff(check_readme, bench_md)
        if diff is not None:
            console.print(diff)
            raise typer.Exit(code=1)


@app.command()
def serve(
    inputs: list[str] = typer.Argument(  # noqa: B008 - typer's documented variadic-argument idiom
        ..., metavar="INPUT", help="Trace file(s) (JSON Lines) path or packaged alias."
    ),
    results: str | None = typer.Option(
        None, "--results", help="Precomputed claimcheck-result/v1 JSON Lines file."
    ),
    reviews: str = typer.Option(
        "claimcheck-reviews.jsonl", "--reviews", help="Reviews JSON Lines file to append to."
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8765, "--port", help="Bind port."),
    max_usd: float | None = typer.Option(
        None,
        "--max-usd",
        help="Live judge budget cap in USD (default: CLAIMCHECK_MAX_USD, then the config's "
        "[budget] max_usd, then 1.0).",
    ),
    config: str | None = typer.Option(None, "--config", help="claimcheck.toml path."),
) -> None:
    """Serve the review dashboard over the named trace inputs."""
    from agent_claimcheck.server.app import serve as serve_dashboard

    try:
        serve_dashboard(
            inputs,
            results=results,
            reviews=Path(reviews),
            host=host,
            port=port,
            max_usd=max_usd,
            config=config,
        )
    except (OSError, ValueError, ResourceNotFoundError) as exc:
        error_console.print(f"error: {exc}")
        raise typer.Exit(code=2) from None
    except KeyboardInterrupt:
        raise typer.Exit(code=0) from None


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
