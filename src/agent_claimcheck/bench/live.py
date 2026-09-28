"""Live judge orchestration: a dry-run cost estimate and the real judged run.

Nothing here changes the recorded-file contract `bench/report.py` already
reads: a live run writes exactly the record shapes `bench/runner.py`'s
`run_offline` writes for the offline detectors, plus `judge-records.jsonl`
in the shape `report.py` expects, so `bench --from-recorded` on a live run's
output directory reproduces its `bench.md` byte for byte.

`--limit` only restricts which traces the judges are asked to score (a cheap
smoke run). The offline detectors always score the full packaged benchmark,
because `run_offline` asserts its retrained classifier matches the shipped
artifact, which only holds for the complete split.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agent_claimcheck.calibration import CalibratorSet, fit_calibrator
from agent_claimcheck.claims import ClaimExtractor, success_claims
from agent_claimcheck.detectors.base import DetectorOutput
from agent_claimcheck.detectors.judge import JudgeDetector
from agent_claimcheck.judge.cache import JudgeCache
from agent_claimcheck.judge.render import (
    DEFAULT_PROMPT_NAME,
    JudgeSpec,
    Prompt,
    load_prompt,
    openrouter_extra_body,
    render_request,
    request_sha256,
)
from agent_claimcheck.ledger import Budget, Ledger, Price, PriceBook, reservation_usd
from agent_claimcheck.redact import DetectorView, detector_view, resolve_claims
from agent_claimcheck.rules.engine import builtin_packs
from agent_claimcheck.schema import Trace

JUDGE_RECORDS_FILE = "judge-records.jsonl"
RUN_FILE = "run.json"


class BudgetPreflightError(Exception):
    """The dry-run reservation total exceeds `--max-usd` without `--allow-partial`."""


@dataclass(frozen=True)
class _Item:
    """One trace, projected once and kept alongside its split and label."""

    trace_id: str
    split: str
    view: DetectorView
    success: int  # 1 = ground-truth success, 0 = failure.


@dataclass(frozen=True)
class JudgeRun:
    """One (model, prompt) combination every selected trace is scored with."""

    model: str
    prompt: Prompt
    key: str
    spec: JudgeSpec
    is_ablation: bool = False


@dataclass(frozen=True)
class ReservationRow:
    """One run's dry-run call count and worst-case reservation total."""

    key: str
    model: str
    prompt_name: str
    n_calls: int
    total_usd: float


@dataclass(frozen=True)
class LiveSummary:
    """What `bench --live`'s closing spend line reports."""

    calls: int
    cached: int
    total_actual_usd: float
    ledger_total_usd: float
    budget_exhausted: bool


def _dump_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _dump_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", "utf-8")


def _sanitize_command_paths(command: str) -> str:
    """Rewrite every absolute-path token in `command` (as recorded in
    `run.json`) so it carries no other user's home directory or machine
    layout: a path under the current directory becomes relative to it, a
    path under the home directory is shortened to `~/...`, and anything
    else (already relative, or outside both) is left untouched.
    """
    cwd = Path.cwd()
    home = Path.home()
    tokens = []
    for token in command.split(" "):
        if token.startswith("/"):
            candidate = Path(token)
            try:
                tokens.append(str(candidate.relative_to(cwd)))
                continue
            except ValueError:
                pass
            try:
                rel_home = candidate.relative_to(home)
                tokens.append("~" if str(rel_home) == "." else f"~/{rel_home}")
                continue
            except ValueError:
                pass
        tokens.append(token)
    return " ".join(tokens)


def _prepare_items(traces: Sequence[Trace], split: str, extractor: ClaimExtractor) -> list[_Item]:
    items: list[_Item] = []
    for t in traces:
        view = resolve_claims(detector_view(t), extractor)
        if not success_claims(view.final_claim.claims):
            continue
        gt = t.ground_truth
        if gt is None or gt.outcome == "unknown":
            continue
        items.append(
            _Item(
                trace_id=t.trace_id,
                split=split,
                view=view,
                success=1 if gt.outcome == "success" else 0,
            )
        )
    return items


def _limited(traces: Sequence[Trace], limit: int | None) -> list[Trace]:
    traces = list(traces)
    return traces[:limit] if limit is not None else traces


def _detector_key(model: str, prompt: Prompt) -> str:
    if prompt.name == DEFAULT_PROMPT_NAME:
        return f"judge:{model}"
    return f"judge:{model}:{prompt.name}"


def _spec_for(model: str, prompt: Prompt, price_book: PriceBook, base_url: str) -> JudgeSpec:
    supports_reasoning = price_book.reasoning_supported(model)
    extra_body = openrouter_extra_body(
        base_url, json_mode=True, supports_reasoning=supports_reasoning
    )
    return JudgeSpec(
        model=model, temperature=0.0, max_tokens=400, json_mode=True, extra_body=extra_body
    )


def build_runs(
    judges: Sequence[str],
    ablation_judge: str,
    ablation_prompt: str,
    price_book: PriceBook,
    base_url: str,
) -> list[JudgeRun]:
    """The primary judges (claim-audit) plus the ablation judge/prompt."""
    runs: list[JudgeRun] = []
    audit_prompt = load_prompt(DEFAULT_PROMPT_NAME)
    for model in judges:
        spec = _spec_for(model, audit_prompt, price_book, base_url)
        runs.append(
            JudgeRun(
                model=model,
                prompt=audit_prompt,
                key=_detector_key(model, audit_prompt),
                spec=spec,
            )
        )
    ab_prompt = load_prompt(ablation_prompt)
    ab_spec = _spec_for(ablation_judge, ab_prompt, price_book, base_url)
    runs.append(
        JudgeRun(
            model=ablation_judge,
            prompt=ab_prompt,
            key=_detector_key(ablation_judge, ab_prompt),
            spec=ab_spec,
            is_ablation=True,
        )
    )
    return runs


def plan_reservations(
    runs: Sequence[JudgeRun], items: Sequence[_Item], price_book: PriceBook
) -> tuple[list[ReservationRow], float]:
    """Worst-case reservation per run (max per-endpoint price), and the grand total."""
    rows: list[ReservationRow] = []
    grand_total = 0.0
    for run in runs:
        max_price = price_book.max_price(run.model)
        total = 0.0
        for item in items:
            body = render_request(item.view, run.prompt, run.spec)
            request_json = json.dumps(
                body, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            total += reservation_usd(request_json, run.spec.max_tokens, max_price)
        rows.append(ReservationRow(run.key, run.model, run.prompt.name, len(items), total))
        grand_total += total
    return rows, grand_total


def format_dry_run(rows: Sequence[ReservationRow], grand_total: float) -> str:
    lines = [
        f"{r.key} ({r.prompt_name}): {r.n_calls} calls, worst-case reservation ${r.total_usd:.6f}"
        for r in rows
    ]
    lines.append(f"grand total: ${grand_total:.6f}")
    return "\n".join(lines)


def format_spend_summary(summary: LiveSummary) -> str:
    return (
        f"live run: {summary.calls} calls ({summary.cached} cached), "
        f"${summary.total_actual_usd:.6f} actual this run, "
        f"${summary.ledger_total_usd:.6f} ledger total"
    )


def dry_run_plan(
    train_full: Sequence[Trace],
    test_full: Sequence[Trace],
    *,
    judges: Sequence[str],
    ablation_judge: str,
    ablation_prompt: str,
    limit: int | None,
    base_url: str,
    api_key: str | None,
    price_override: Price | None,
    transport: httpx.BaseTransport | None = None,
) -> tuple[list[ReservationRow], float]:
    """Call counts and worst-case reservations for `--live --dry-run`. Sends nothing."""
    extractor = ClaimExtractor(list(builtin_packs().values()))
    items = _prepare_items(_limited(train_full, limit), "train", extractor) + _prepare_items(
        _limited(test_full, limit), "test", extractor
    )
    with httpx.Client(transport=transport) as client:
        price_book = PriceBook(
            base_url=base_url, api_key=api_key, override=price_override, client=client
        )
        runs = build_runs(judges, ablation_judge, ablation_prompt, price_book, base_url)
        return plan_reservations(runs, items, price_book)


def _score_concurrent(
    detector: JudgeDetector, items: Sequence[_Item], concurrency: int
) -> list[DetectorOutput]:
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(detector.score, [item.view for item in items]))


def _judges_meta(
    runs: Sequence[JudgeRun], price_book: PriceBook, date: str
) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run.model in meta:
            continue
        list_price = price_book.list_price(run.model)
        max_price = price_book.max_price(run.model)
        meta[run.model] = {
            "id": run.model,
            "price_in_per_m": list_price.price_in_per_m,
            "price_out_per_m": list_price.price_out_per_m,
            "max_price_in_per_m": max_price.price_in_per_m,
            "max_price_out_per_m": max_price.price_out_per_m,
            "price_date": date,
            "json_mode": run.spec.json_mode,
            "supports_reasoning": price_book.reasoning_supported(run.model),
        }
    return meta


def run_live(
    train_full: Sequence[Trace],
    test_full: Sequence[Trace],
    out_dir: str | Path,
    *,
    judges: Sequence[str],
    ablation_judge: str,
    ablation_prompt: str,
    run_name: str,
    max_usd: float,
    concurrency: int,
    limit: int | None,
    allow_partial: bool,
    base_url: str,
    api_key: str | None,
    price_override: Price | None,
    timeout_s: float,
    ledger_path: str | Path,
    ledger_cap_usd: float | None,
    cache_dir: str | Path,
    command: str,
    date: str,
    hardware: str,
    package_version: str,
    dataset_sha256: str,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] | None = None,
) -> LiveSummary:
    """Score every selected trace with every judge, then the ablation judge.

    Raises `BudgetPreflightError` before writing anything when the dry-run
    reservation total exceeds `max_usd` and `allow_partial` is false. A
    reservation that runs out of budget mid-run is handled by `JudgeDetector`
    itself (the remaining calls abstain `budget_exhausted`); this function
    still writes every file and reports the exhaustion in the summary.
    """
    extractor = ClaimExtractor(list(builtin_packs().values()))
    items = _prepare_items(_limited(train_full, limit), "train", extractor) + _prepare_items(
        _limited(test_full, limit), "test", extractor
    )

    with httpx.Client(transport=transport) as price_client:
        price_book = PriceBook(
            base_url=base_url, api_key=api_key, override=price_override, client=price_client
        )
        runs = build_runs(judges, ablation_judge, ablation_prompt, price_book, base_url)
        _reservation_rows, grand_total = plan_reservations(runs, items, price_book)
        judges_meta = _judges_meta(runs, price_book, date)

    if grand_total > max_usd and not allow_partial:
        raise BudgetPreflightError(
            f"dry-run reservation ${grand_total:.6f} exceeds --max-usd ${max_usd:.2f}; "
            "pass --allow-partial to run until the budget stops it"
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(ledger_path)
    ledger_total_before = ledger.total_actual()
    budget = Budget(max_usd, ledger=ledger, cap=ledger_cap_usd)
    cache = JudgeCache(cache_dir)

    judge_wall_clock_s: dict[str, float] = {}
    records: list[dict[str, Any]] = []

    for run in runs:
        detector = JudgeDetector(
            model=run.model,
            base_url=base_url,
            api_key=api_key,
            prompt=run.prompt.name,
            temperature=run.spec.temperature,
            max_tokens=run.spec.max_tokens,
            json_mode=run.spec.json_mode,
            timeout_s=timeout_s,
            extra_body=run.spec.extra_body,
            cache=cache,
            read_cache=False,
            budget=budget,
            price_book=price_book,
            ledger=ledger,
            run_id=run_name,
            transport=transport,
            sleep=sleep,
        )
        try:
            started = time.perf_counter()
            outputs = _score_concurrent(detector, items, concurrency)
            judge_wall_clock_s[run.key] = time.perf_counter() - started
        finally:
            detector.close()

        for item, output in zip(items, outputs, strict=True):
            body = render_request(item.view, run.prompt, run.spec)
            details = output.details
            records.append(
                {
                    "trace_id": item.trace_id,
                    "split": item.split,
                    "detector": run.key,
                    "model": run.model,
                    "prompt_name": run.prompt.name,
                    "prompt_version": run.prompt.version,
                    "prompt_sha256": run.prompt.sha256,
                    "request_sha256": request_sha256(body),
                    "parsed": details.get("parsed"),
                    "p_raw": output.p_success,
                    "abstain": output.abstain,
                    "abstain_reason": output.abstain_reason,
                    "raw_text": details.get("raw_response"),
                    "usage": details.get("usage", {}),
                    "cost_usd": output.cost_usd,
                    "latency_ms": output.latency_ms,
                    "cached": output.cached,
                    "attempts": details.get("attempts", 0),
                    "invalid_citation": details.get("invalid_citation", False),
                }
            )

    _dump_jsonl(records, out_path / JUDGE_RECORDS_FILE)

    from agent_claimcheck.bench.runner import run_offline

    run_offline(
        train_full,
        test_full,
        out_path,
        seed=0,
        command=command,
        date=date,
        hardware=hardware,
        package_version=package_version,
        dataset_sha256=dataset_sha256,
    )

    calibration_path = out_path / "calibration.json"
    calibrator_set = CalibratorSet.load(calibration_path)
    success_by_train_trace = {i.trace_id: i.success for i in items if i.split == "train"}
    for run in runs:
        train_rows = [
            r
            for r in records
            if r["detector"] == run.key and r["split"] == "train" and not r["abstain"]
        ]
        if not train_rows:
            continue
        raw_p = [r["p_raw"] for r in train_rows]
        labels = [success_by_train_trace[r["trace_id"]] for r in train_rows]
        calibrator_set.calibrators[run.key] = fit_calibrator(
            run.key, raw_p, labels, fitted_on="bench:train"
        )
    calibrator_set.save(calibration_path)

    ledger_total_after = ledger.total_actual()
    total_actual_this_run = ledger_total_after - ledger_total_before
    n_calls = len(records)
    n_cached = sum(1 for r in records if r["cached"])
    budget_exhausted = any(r["abstain_reason"] == "budget_exhausted" for r in records)

    run_meta = {
        "command": _sanitize_command_paths(command),
        "date": date,
        "hardware": hardware,
        "concurrency": concurrency,
        "package_version": package_version,
        "dataset_sha256": dataset_sha256,
        "judges": [judges_meta[m] for m in sorted(judges_meta)],
        "total_spend_usd": total_actual_this_run,
        "judge_wall_clock_s": judge_wall_clock_s,
        "run_name": run_name,
    }
    _dump_json(run_meta, out_path / RUN_FILE)

    return LiveSummary(
        calls=n_calls,
        cached=n_cached,
        total_actual_usd=total_actual_this_run,
        ledger_total_usd=ledger_total_after,
        budget_exhausted=budget_exhausted,
    )
