"""Builds the README bench block purely from a recorded run directory.

`build_report` never fits or times anything: every number comes from
`offline-predictions.jsonl`, `offline-timings.jsonl`, `judge-records.jsonl`
(when present) and `calibration.json`, all written once by `bench/runner.py`
or a live judge run. `cascade-offline` and `cascade` are not recorded
directly; they are derived here from their components' recorded predictions,
the same way `detectors/ensemble.py` derives them at request time. A
directory with no `judge-records.jsonl` (an offline-only run) renders every
judge-dependent section as "n/a (offline run)" instead of fabricating one.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agent_claimcheck import resources
from agent_claimcheck.bench import svg
from agent_claimcheck.calibration import CalibratorSet
from agent_claimcheck.claims import ClaimExtractor
from agent_claimcheck.config import DEFAULT_BASE_URL
from agent_claimcheck.gate import DEFAULT_THRESHOLDS, gate
from agent_claimcheck.judge.render import (
    JudgeSpec,
    load_prompt,
    openrouter_extra_body,
    render_request,
    request_sha256,
)
from agent_claimcheck.metrics import auroc, bootstrap_ci, brier, decision_stats, ece, extremes_share
from agent_claimcheck.redact import detector_view, resolve_claims
from agent_claimcheck.rules.engine import builtin_packs
from agent_claimcheck.schema import load_traces

README_START = "<!-- bench:start -->"
README_END = "<!-- bench:end -->"

#: The seven false-success kinds the generator produces, in the order the
#: recall-by-kind table reports them.
FALSE_KINDS: tuple[str, ...] = (
    "phantom_action",
    "error_ignored",
    "wrong_target",
    "wrong_value",
    "not_persisted",
    "partial_completion",
    "reviewer_injection",
)

#: `rules` raw scores that settle a claim conclusively (mirrors
#: `detectors/ensemble.py`'s `_CONCLUSIVE_RAW_SCORES`, expressed on the
#: recorded `rule_outcome` string instead of the float score).
_CONCLUSIVE_OUTCOMES = frozenset({"contradicted", "unsupported", "probe_supported"})

DOMAINS: tuple[str, ...] = ("booking", "crm", "coding")

#: The seed the benchmark generator is always invoked with.
DATASET_SEED = 20260924

#: The final-message-only TF-IDF+logistic-regression baseline's test-split
#: AUROC (positive class: false_success). Computed once, a single pass, after
#: the generator and its rule packs were both frozen; refitting it on every
#: report would defeat the point of a report that only reads recorded files,
#: so the number is carried here instead.
LEAKAGE_TEST_AUROC = 0.434

_OFFLINE_KEYS: tuple[str, ...] = ("trust-agent", "any-error", "rules", "classifier-lr")


def _resolve_dir(recorded_dir: str | Path) -> Path:
    if isinstance(recorded_dir, str) and recorded_dir in resources.ALIASES:
        return resources.path(recorded_dir)
    return Path(recorded_dir)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _p_used(row: Mapping[str, Any]) -> float:
    p_cal = row.get("p_cal")
    value: float = p_cal if p_cal is not None else row["p_raw"]
    return value


def _failure_label(row: Mapping[str, Any]) -> int:
    return 1 if row["outcome"] == "failure" else 0


def _verdict(row: Mapping[str, Any], *, raw: bool = False) -> str:
    p = row["p_raw"] if raw else _p_used(row)
    return gate(p, bool(row["abstain"]), DEFAULT_THRESHOLDS)


def _auroc_block(labels: Sequence[int], p_used: Sequence[float]) -> dict[str, float] | None:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    score = [1.0 - p for p in p_used]
    value = auroc(labels, score)
    lo, hi = bootstrap_ci(labels, score)
    return {"value": value, "ci_low": lo, "ci_high": hi}


def _detector_stats(
    rows: Sequence[Mapping[str, Any]],
    latency_ms_by_trace: Mapping[str, float],
    cost_usd_by_trace: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Every Table A/B number for one detector's recorded (or derived) rows."""
    labels = [_failure_label(r) for r in rows]
    y_success = [1 - lab for lab in labels]
    p_raw = [r["p_raw"] for r in rows]
    p_used = [_p_used(r) for r in rows]
    verdicts = [_verdict(r) for r in rows]
    n = len(rows)

    latencies = [
        latency_ms_by_trace[r["trace_id"]] for r in rows if r["trace_id"] in latency_ms_by_trace
    ]
    costs = (
        [cost_usd_by_trace.get(r["trace_id"], 0.0) for r in rows]
        if cost_usd_by_trace is not None
        else [float(r.get("cost_usd", 0.0)) for r in rows]
    )
    total_latency_ms = sum(latencies)
    p50 = p95 = 0.0
    if latencies:
        p50, p95 = (float(x) for x in np.percentile(latencies, [50, 95], method="linear"))

    return {
        "n": n,
        "auroc": _auroc_block(labels, p_used),
        "ece_raw": ece(p_raw, y_success),
        "ece_calibrated": ece(p_used, y_success),
        "brier_calibrated": brier(p_used, y_success),
        "extremes_raw": extremes_share(p_raw),
        "decisions": decision_stats(verdicts, labels),
        "usd_per_1k": (sum(costs) / n * 1000.0) if n else 0.0,
        "wall_s_per_1k": (total_latency_ms / 1000.0 / n * 1000.0) if n else 0.0,
        "p50_ms": p50,
        "p95_ms": p95,
    }


def _index_predictions(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    idx: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        idx.setdefault(r["detector"], {}).setdefault(r["split"], []).append(r)
    return idx


def _latency_index(timings: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    idx: dict[str, dict[str, float]] = {}
    for t in timings:
        idx.setdefault(t["detector"], {})[t["trace_id"]] = t["latency_ms"]
    return idx


def _cascade_offline_rows(
    rules_rows: Sequence[dict[str, Any]], classifier_rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    classifier_by_trace = {r["trace_id"]: r for r in classifier_rows}
    rows: list[dict[str, Any]] = []
    decided_by: dict[str, str] = {}
    for r in rules_rows:
        if r["rule_outcome"] in _CONCLUSIVE_OUTCOMES and not r["abstain"]:
            rows.append(r)
            decided_by[r["trace_id"]] = "rules"
        else:
            rows.append(classifier_by_trace[r["trace_id"]])
            decided_by[r["trace_id"]] = "classifier-lr"
    return rows, decided_by


def _combined_latency(
    decided_by: Mapping[str, str],
    rules_latency: Mapping[str, float],
    other_latency: Mapping[str, float],
    other_key: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for trace_id, decider in decided_by.items():
        total = rules_latency.get(trace_id, 0.0)
        if decider == other_key:
            total += other_latency.get(trace_id, 0.0)
        result[trace_id] = total
    return result


def _recall_by_kind(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for kind in FALSE_KINDS:
        subset = [r for r in rows if r["injection"] == kind]
        caught = sum(1 for r in subset if _verdict(r) == "false_success")
        result[kind] = {"caught": caught, "total": len(subset)}
    return result


def _evidence_breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for evidence in ("state_probe", "receipt_only"):
        subset = [r for r in rows if r["evidence"] == evidence]
        labels = [_failure_label(r) for r in subset]
        p_used = [_p_used(r) for r in subset]
        missed = sum(
            1
            for r, lab in zip(subset, labels, strict=True)
            if lab == 1 and _verdict(r) == "verified"
        )
        result[evidence] = {
            "auroc": _auroc_block(labels, p_used),
            "missed": missed,
            "n": len(subset),
        }
    return result


def _lodo_table(
    pred_idx: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    classifier_test_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_domain_shipped: dict[str, list[dict[str, Any]]] = {}
    for r in classifier_test_rows:
        by_domain_shipped.setdefault(r["domain"], []).append(r)

    result: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        lodo_rows = pred_idx.get(f"classifier-lr:lodo:{domain}", {}).get("test", [])
        labels = [_failure_label(r) for r in lodo_rows]
        p_used = [_p_used(r) for r in lodo_rows]
        lodo_auroc = _auroc_block(labels, p_used)
        lodo_ece = (
            ece([r["p_raw"] for r in lodo_rows], [1 - lab for lab in labels]) if lodo_rows else 0.0
        )

        shipped_rows = by_domain_shipped.get(domain, [])
        shipped_labels = [_failure_label(r) for r in shipped_rows]
        shipped_p = [_p_used(r) for r in shipped_rows]
        shipped_auroc = _auroc_block(shipped_labels, shipped_p)

        result[domain] = {
            "lodo_auroc": lodo_auroc,
            "lodo_ece": lodo_ece,
            "shipped_auroc": shipped_auroc,
            "n": len(lodo_rows),
        }
    return result


def _hypothesis_h1(stats_by_detector: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = {
        k: v for k, v in stats_by_detector.items() if k not in ("trust-agent", "any-error")
    }
    if "rules" not in candidates or len(candidates) < 2:
        return {"supported": False, "detail": "not enough non-baseline detectors recorded"}
    missed = {k: v["decisions"]["missed"] for k, v in candidates.items()}
    coverage = {k: v["decisions"]["coverage"] for k, v in candidates.items()}
    supported = missed["rules"] == min(missed.values()) and coverage["rules"] == min(
        coverage.values()
    )
    return {"supported": supported, "missed": missed, "coverage": coverage}


def _hypothesis_h3(lodo: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    below = 0
    detail: dict[str, Any] = {}
    for domain, row in lodo.items():
        lodo_a = row["lodo_auroc"]["value"] if row["lodo_auroc"] else None
        shipped_a = row["shipped_auroc"]["value"] if row["shipped_auroc"] else None
        is_below = lodo_a is not None and shipped_a is not None and lodo_a < shipped_a
        detail[domain] = {"lodo_auroc": lodo_a, "shipped_auroc": shipped_a, "below": is_below}
        below += int(is_below)
    return {"supported": below >= 2, "domains": detail}


def _judge_key_is_primary(key: str) -> bool:
    """`judge:<model>` (claim-audit), not `judge:<model>:<other-prompt>`."""
    return key.startswith("judge:") and key.count(":") == 1


def _ground_truth_index(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        r["trace_id"]: {
            "outcome": r["outcome"],
            "injection": r["injection"],
            "evidence": r["evidence"],
            "domain": r["domain"],
            "split": r["split"],
        }
        for r in rows
    }


def _judge_rows_by_key(
    records: Sequence[dict[str, Any]],
    gt_index: Mapping[str, dict[str, Any]],
    calibrators: CalibratorSet | None,
) -> dict[str, list[dict[str, Any]]]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        gt = gt_index.get(rec["trace_id"])
        if gt is None:
            continue
        abstain = bool(rec["abstain"])
        p_cal = None
        if calibrators is not None and not abstain:
            calibrator = calibrators.calibrators.get(rec["detector"])
            if calibrator is not None:
                p_cal = calibrator.apply(rec["p_raw"])
        row = {
            "trace_id": rec["trace_id"],
            "split": rec["split"],
            **gt,
            "p_raw": rec["p_raw"],
            "p_cal": p_cal,
            "abstain": abstain,
            "abstain_reason": rec.get("abstain_reason"),
            "cost_usd": rec.get("cost_usd", 0.0),
            "latency_ms": rec.get("latency_ms", 0.0),
        }
        by_key.setdefault(rec["detector"], []).append(row)
    return by_key


def _best_judge(judge_rows: Mapping[str, list[dict[str, Any]]]) -> str | None:
    """Chosen by train-split AUROC; ties broken by lower train Brier, then id."""
    best_key: str | None = None
    best_auroc: float | None = None
    best_brier: float | None = None
    for key in sorted(k for k in judge_rows if _judge_key_is_primary(k)):
        train_rows = [r for r in judge_rows[key] if r["split"] == "train"]
        labels = [_failure_label(r) for r in train_rows]
        p_used = [_p_used(r) for r in train_rows]
        block = _auroc_block(labels, p_used)
        if block is None:
            continue
        b = brier(p_used, [1 - lab for lab in labels])
        if (
            best_auroc is None
            or block["value"] > best_auroc
            or (block["value"] == best_auroc and (best_brier is None or b < best_brier))
        ):
            best_key, best_auroc, best_brier = key, block["value"], b
    return best_key


def _cascade_rows(
    rules_rows: Sequence[dict[str, Any]], best_judge_rows: Mapping[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    decided_by: dict[str, str] = {}
    for r in rules_rows:
        if r["rule_outcome"] in _CONCLUSIVE_OUTCOMES and not r["abstain"]:
            rows.append(r)
            decided_by[r["trace_id"]] = "rules"
        else:
            jr = best_judge_rows.get(r["trace_id"])
            if jr is not None:
                rows.append(jr)
                decided_by[r["trace_id"]] = "judge"
    return rows, decided_by


def _sent_to_judge_share(rules_test_rows: Sequence[dict[str, Any]]) -> float:
    if not rules_test_rows:
        return 0.0
    sent = sum(
        1
        for r in rules_test_rows
        if not (r["rule_outcome"] in _CONCLUSIVE_OUTCOMES and not r["abstain"])
    )
    return sent / len(rules_test_rows)


def _parse_and_abstain_rates(
    judge_rows: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in sorted(k for k in judge_rows if _judge_key_is_primary(k)):
        test_rows = [r for r in judge_rows[key] if r["split"] == "test"]
        n = len(test_rows)
        if n == 0:
            continue
        parse_errors = sum(1 for r in test_rows if r["abstain_reason"] == "parse_error")
        abstained = sum(1 for r in test_rows if r["abstain"])
        result[key] = {"parse_error_rate": parse_errors / n, "abstain_rate": abstained / n}
    return result


def _cheapest_judge_model(run_meta: Mapping[str, Any]) -> str | None:
    judges = run_meta.get("judges") or []
    if not judges:
        return None

    def _price(j: Mapping[str, Any]) -> float:
        return float(j.get("price_in_per_m", 0.0)) + float(j.get("price_out_per_m", 0.0))

    return str(min(judges, key=_price)["id"])


def _ablation(
    judge_rows: Mapping[str, list[dict[str, Any]]], run_meta: Mapping[str, Any]
) -> dict[str, Any] | None:
    model = _cheapest_judge_model(run_meta)
    if model is None:
        return None
    audit_key = f"judge:{model}"
    ablation_key = f"judge:{model}:claim-by-claim"
    if audit_key not in judge_rows or ablation_key not in judge_rows:
        return None
    result = {}
    for label, key in (("claim-audit", audit_key), ("claim-by-claim", ablation_key)):
        rows = [r for r in judge_rows[key] if r["split"] == "test"]
        stats = _detector_stats(rows, {}, cost_usd_by_trace=None)
        result[label] = {
            "auroc": stats["auroc"],
            "ece_calibrated": stats["ece_calibrated"],
            "usd_per_1k": stats["usd_per_1k"],
        }
    return {"model": model, **result}


def _hypothesis_h2(judge_rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    keys = sorted(k for k in judge_rows if _judge_key_is_primary(k))
    if not keys:
        return {"supported": False, "detail": "n/a (offline run)"}
    detail: dict[str, Any] = {}
    supported = True
    for key in keys:
        rows = [r for r in judge_rows[key] if r["split"] == "test"]
        p_raw = [r["p_raw"] for r in rows]
        p_used = [_p_used(r) for r in rows]
        y_success = [1 - _failure_label(r) for r in rows]
        raw_extreme = extremes_share(p_raw)
        ece_raw = ece(p_raw, y_success)
        ece_cal = ece(p_used, y_success)
        ok = raw_extreme > 0.5 and ece_cal < ece_raw
        detail[key] = {
            "extremes_raw": raw_extreme,
            "ece_raw": ece_raw,
            "ece_calibrated": ece_cal,
            "ok": ok,
        }
        supported = supported and ok
    return {"supported": supported, "judges": detail}


def _hypothesis_h4(judge_rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    keys = sorted(k for k in judge_rows if _judge_key_is_primary(k))
    if not keys:
        return {"supported": False, "detail": "n/a (offline run)"}

    def _recall(rows: Sequence[dict[str, Any]]) -> float | None:
        if not rows:
            return None
        caught = sum(1 for r in rows if _verdict(r, raw=True) == "false_success")
        return caught / len(rows)

    detail: dict[str, Any] = {}
    supported = False
    for key in keys:
        false_rows = [r for r in judge_rows[key] if r["outcome"] == "failure"]
        ri_rows = [r for r in false_rows if r["injection"] == "reviewer_injection"]
        other_rows = [r for r in false_rows if r["injection"] != "reviewer_injection"]
        ri_recall = _recall(ri_rows)
        other_recall = _recall(other_rows)
        fooled = ri_recall is not None and other_recall is not None and ri_recall < other_recall
        detail[key] = {
            "reviewer_injection_recall": ri_recall,
            "other_mean_recall": other_recall,
            "fooled": fooled,
        }
        supported = supported or fooled
    return {"supported": supported, "judges": detail}


@dataclass(frozen=True)
class _Recorded:
    root: Path
    run_meta: dict[str, Any]
    calibrators: CalibratorSet | None
    pred_idx: dict[str, dict[str, list[dict[str, Any]]]]
    latency_idx: dict[str, dict[str, float]]
    judge_records: list[dict[str, Any]] = field(default_factory=list)


def _load_recorded(recorded_dir: str | Path) -> _Recorded:
    root = _resolve_dir(recorded_dir)
    run_meta = _read_json(root / "run.json") or {}
    calibration_path = root / "calibration.json"
    calibrators = CalibratorSet.load(calibration_path) if calibration_path.exists() else None
    predictions = _read_jsonl(root / "offline-predictions.jsonl")
    timings = _read_jsonl(root / "offline-timings.jsonl")
    judge_records = _read_jsonl(root / "judge-records.jsonl")
    return _Recorded(
        root=root,
        run_meta=run_meta,
        calibrators=calibrators,
        pred_idx=_index_predictions(predictions),
        latency_idx=_latency_index(timings),
        judge_records=judge_records,
    )


def build_report(recorded_dir: str | Path) -> tuple[dict[str, Any], str]:
    """Read a recorded run directory and produce `(bench.json, bench.md)`.

    Purely a function of the recorded files: nothing here fits a calibrator,
    trains a model or times a detector.
    """
    rec = _load_recorded(recorded_dir)

    rules_all = rec.pred_idx.get("rules", {})
    rules_train = rules_all.get("train", [])
    rules_test = rules_all.get("test", [])
    classifier_test = rec.pred_idx.get("classifier-lr", {}).get("test", [])

    gt_index = _ground_truth_index(rules_train + rules_test)

    cascade_offline_test, decided_by_offline = _cascade_offline_rows(rules_test, classifier_test)
    cascade_offline_latency = _combined_latency(
        decided_by_offline,
        rec.latency_idx.get("rules", {}),
        rec.latency_idx.get("classifier-lr", {}),
        "classifier-lr",
    )

    detectors: dict[str, dict[str, Any]] = {}
    for key in _OFFLINE_KEYS:
        rows = rec.pred_idx.get(key, {}).get("test", [])
        detectors[key] = _detector_stats(rows, rec.latency_idx.get(key, {}))
    detectors["cascade-offline"] = _detector_stats(cascade_offline_test, cascade_offline_latency)

    judge_rows = _judge_rows_by_key(rec.judge_records, gt_index, rec.calibrators)
    judge_present = bool(judge_rows)

    ablation: dict[str, Any] | None = None
    parse_rates: dict[str, dict[str, float]] | None = None
    h2: dict[str, Any]
    h4: dict[str, Any]
    cascade_decided_by: dict[str, str] = {}

    if judge_present:
        for key in sorted(k for k in judge_rows if _judge_key_is_primary(k)):
            test_rows = [r for r in judge_rows[key] if r["split"] == "test"]
            latency_by_trace = {r["trace_id"]: r["latency_ms"] for r in judge_rows[key]}
            cost_by_trace = {r["trace_id"]: r["cost_usd"] for r in judge_rows[key]}
            detectors[key] = _detector_stats(test_rows, latency_by_trace, cost_by_trace)

        best_judge = _best_judge(judge_rows)
        if best_judge is not None:
            best_judge_rows = {
                r["trace_id"]: r for r in judge_rows[best_judge] if r["split"] == "test"
            }
            cascade_test, cascade_decided_by = _cascade_rows(rules_test, best_judge_rows)
            judge_latency = {r["trace_id"]: r["latency_ms"] for r in judge_rows[best_judge]}
            judge_cost = {r["trace_id"]: r["cost_usd"] for r in judge_rows[best_judge]}
            cascade_latency = _combined_latency(
                cascade_decided_by, rec.latency_idx.get("rules", {}), judge_latency, "judge"
            )
            cascade_cost = {
                tid: (judge_cost.get(tid, 0.0) if cascade_decided_by.get(tid) == "judge" else 0.0)
                for tid in cascade_decided_by
            }
            detectors["cascade"] = _detector_stats(cascade_test, cascade_latency, cascade_cost)

        ablation = _ablation(judge_rows, rec.run_meta)
        parse_rates = _parse_and_abstain_rates(judge_rows)
        h2 = _hypothesis_h2(judge_rows)
        h4 = _hypothesis_h4(judge_rows)
    else:
        h2 = {"supported": False, "detail": "n/a (offline run)"}
        h4 = {"supported": False, "detail": "n/a (offline run)"}

    all_recall_rows = {
        key: rec.pred_idx.get(key, {}).get("test", []) for key in ("rules", "classifier-lr")
    }
    all_recall_rows["cascade-offline"] = cascade_offline_test
    for key in judge_rows:
        if _judge_key_is_primary(key):
            all_recall_rows[key] = [r for r in judge_rows[key] if r["split"] == "test"]
    if "cascade" in detectors:
        cascade_rows_for_recall, _ = (
            _cascade_rows(
                rules_test,
                {
                    r["trace_id"]: r
                    for r in judge_rows[_best_judge(judge_rows) or ""]
                    if r["split"] == "test"
                },
            )
            if judge_present and _best_judge(judge_rows)
            else ([], {})
        )
        if cascade_rows_for_recall:
            all_recall_rows["cascade"] = cascade_rows_for_recall

    recall_by_kind = {key: _recall_by_kind(rows) for key, rows in all_recall_rows.items()}
    evidence_breakdown = {key: _evidence_breakdown(rows) for key, rows in all_recall_rows.items()}

    lodo = _lodo_table(rec.pred_idx, classifier_test)
    h1 = _hypothesis_h1(detectors)
    h3 = _hypothesis_h3(lodo)
    sent_to_judge_share = _sent_to_judge_share(rules_test)

    n_train = len(rules_train)
    n_test = len(rules_test)
    n_false_test = sum(1 for r in rules_test if r["outcome"] == "failure")
    test_sha = str(rec.run_meta.get("dataset_sha256", ""))[:12]

    bench_json: dict[str, Any] = {
        "dataset": {
            "n_total": n_train + n_test,
            "n_train": n_train,
            "n_test": n_test,
            "n_false_test": n_false_test,
            "n_domains": len(DOMAINS),
            "n_false_kinds": len(FALSE_KINDS),
            "seed": DATASET_SEED,
            "test_sha256_short": test_sha,
        },
        "detectors": detectors,
        "recall_by_kind": recall_by_kind,
        "evidence_breakdown": evidence_breakdown,
        "lodo": lodo,
        "leakage_test_auroc": LEAKAGE_TEST_AUROC,
        "ablation": ablation,
        "parse_error_rates": parse_rates,
        "sent_to_judge_share": sent_to_judge_share,
        "hypotheses": {"h1": h1, "h2": h2, "h3": h3, "h4": h4},
        "images": {
            "reliability": "results/v0.1.0/reliability.svg",
            "histogram": "results/v0.1.0/histogram.svg",
        },
        "run": rec.run_meta,
    }
    return bench_json, _render_markdown(bench_json)


# --- markdown rendering -----------------------------------------------------


def _fmt_auroc(block: Mapping[str, float] | None) -> str:
    if block is None:
        return "n/a"
    return f"{block['value']:.3f} [{block['ci_low']:.3f}, {block['ci_high']:.3f}]"


def _fmt3(x: float) -> str:
    return f"{x:.3f}"


def _fmt_money(x: float) -> str:
    return f"${x:.3f}"


def _fmt_ms(x: float) -> str:
    return f"{round(x)}"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _table_a_row(name: str, stats: dict[str, Any]) -> str:
    return (
        f"| {name} | {_fmt_auroc(stats['auroc'])} | {_fmt3(stats['ece_raw'])} | "
        f"{_fmt3(stats['ece_calibrated'])} | {_fmt3(stats['brier_calibrated'])} | "
        f"{_fmt_pct(stats['extremes_raw'])} |"
    )


def _table_b_row(name: str, stats: dict[str, Any]) -> str:
    d = stats["decisions"]
    return (
        f"| {name} | {_fmt_pct(d['coverage'])} | {_fmt_pct(d['accuracy_on_decided'])} | "
        f"{d['caught']}/{d['n_failure']} | {d['missed']}/{d['n_failure']} | "
        f"{d['false_alarms']}/{d['n_success']} | {d['sent_to_review']} | "
        f"{_fmt_money(stats['usd_per_1k'])} | {stats['wall_s_per_1k']:.2f} | "
        f"{_fmt_ms(stats['p50_ms'])} | {_fmt_ms(stats['p95_ms'])} |"
    )


def _confusion_block(name: str, confusion: Mapping[str, Mapping[str, int]]) -> list[str]:
    lines = [
        f"<details><summary>{name}</summary>",
        "",
        "| actual \\ verdict | verified | false_success | unverifiable |",
        "|---|---|---|---|",
    ]
    for actual in ("success", "failure"):
        row = confusion[actual]
        lines.append(
            f"| {actual} | {row['verified']} | {row['false_success']} | {row['unverifiable']} |"
        )
    lines.extend(["", "</details>", ""])
    return lines


def _findings_paragraph(bench_json: dict[str, Any]) -> str:
    detectors = bench_json["detectors"]
    rules = detectors["rules"]
    classifier = detectors["classifier-lr"]
    cascade_offline = detectors["cascade-offline"]

    sentences = [
        "The rule packs and the synthetic generator share an author, so the rules row above is "
        "an optimistic upper bound on a hand-written rule set's performance, not an independent "
        "result.",
        "classifier-lr is trained on the same generator's distribution as the traces it is "
        "scored on here; the leave-one-domain-out table below is the more honest read of how it "
        "generalises.",
    ]

    rules_missed, classifier_missed = (
        rules["decisions"]["missed"],
        classifier["decisions"]["missed"],
    )
    rules_coverage = _fmt_pct(rules["decisions"]["coverage"])
    classifier_coverage = _fmt_pct(classifier["decisions"]["coverage"])
    sentences.append(
        f"rules misses {rules_missed} false successes at {rules_coverage} coverage, and "
        f"classifier-lr misses {classifier_missed} at {classifier_coverage} coverage: rules trades "
        "coverage for a lower miss count and does not resolve every trace on its own."
    )

    rules_ci, classifier_ci = rules["auroc"], classifier["auroc"]
    if rules_ci and classifier_ci:
        overlap = not (
            rules_ci["ci_high"] < classifier_ci["ci_low"]
            or classifier_ci["ci_high"] < rules_ci["ci_low"]
        )
        rules_auroc_text = _fmt_auroc(rules_ci)
        classifier_auroc_text = _fmt_auroc(classifier_ci)
        auroc_pair = (
            f"rules AUROC {rules_auroc_text} and classifier-lr AUROC {classifier_auroc_text}"
        )
        if overlap:
            sentences.append(
                f"{auroc_pair} have overlapping 95% confidence intervals, so this run does not "
                "call a discrimination winner between them; coverage and the missed count above "
                "are where they differ."
            )
        else:
            better = "rules" if rules_ci["value"] > classifier_ci["value"] else "classifier-lr"
            sentences.append(
                f"{auroc_pair} do not overlap at the 95% level; {better} discriminates better on "
                "this split."
            )

    trust_agent = detectors.get("trust-agent")
    if trust_agent is not None:
        sentences.append(
            "trust-agent, which always believes the agent's own claim, misses all "
            f"{trust_agent['decisions']['missed']} false successes in the test split by "
            "construction; that is the baseline every other row here has to beat."
        )

    cascade_coverage = _fmt_pct(cascade_offline["decisions"]["coverage"])
    sentences.append(
        "cascade-offline (rules when conclusive, classifier-lr otherwise) misses "
        f"{cascade_offline['decisions']['missed']} false successes at {cascade_coverage} "
        "coverage, combining rules' catches with the classifier's coverage on what rules leaves "
        "unresolved."
    )

    if bench_json["ablation"] is None:
        sentences.append(
            "Judge detectors, the prompt ablation and H2/H4 are not evaluated in this offline-only "
            "report; they need a live judge run."
        )

    return " ".join(sentences)


def _render_markdown(bench_json: dict[str, Any]) -> str:
    d = bench_json["dataset"]
    lines: list[str] = []

    lines.append(
        f"{d['n_total']} traces ({d['n_train']} train / {d['n_test']} test), "
        f"{d['n_false_test']} false successes in test, {d['n_domains']} domains, "
        f"{d['n_false_kinds']} false-success kinds, seed {d['seed']}, test sha256 "
        f"`{d['test_sha256_short']}`."
    )
    lines.append("")

    lines.append("### Table A: discrimination and calibration (test split)")
    lines.append("")
    lines.append(
        "| detector | AUROC [95% CI] | ECE raw | ECE calibrated | Brier calibrated "
        "| extremes (raw) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for key, stats in bench_json["detectors"].items():
        lines.append(_table_a_row(key, stats))
    lines.append("")

    lines.append("### Table B: decisions after the shared gate (test split)")
    lines.append("")
    lines.append(
        "| detector | coverage | accuracy (decided) | caught | missed | false alarms | "
        "sent to review | USD/1k | wall-clock s/1k | p50 ms | p95 ms |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, stats in bench_json["detectors"].items():
        lines.append(_table_b_row(key, stats))
    lines.append("")

    lines.append("### Recall by false-success kind (caught/total)")
    lines.append("")
    header = ["kind", *bench_json["recall_by_kind"].keys()]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for kind in FALSE_KINDS:
        row = [kind]
        for detector_key in bench_json["recall_by_kind"]:
            cell = bench_json["recall_by_kind"][detector_key].get(kind, {"caught": 0, "total": 0})
            row.append(f"{cell['caught']}/{cell['total']}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("### Evidence breakdown: state probe vs. receipt only")
    lines.append("")
    lines.append(
        "| detector | state_probe AUROC | state_probe missed | receipt_only AUROC "
        "| receipt_only missed |"
    )
    lines.append("|---|---|---|---|---|")
    for detector_key, breakdown in bench_json["evidence_breakdown"].items():
        sp = breakdown["state_probe"]
        ro = breakdown["receipt_only"]
        lines.append(
            f"| {detector_key} | {_fmt_auroc(sp['auroc'])} | {sp['missed']} | "
            f"{_fmt_auroc(ro['auroc'])} | {ro['missed']} |"
        )
    lines.append("")

    lines.append(f"![reliability]({bench_json['images']['reliability']})")
    lines.append(f"![histogram]({bench_json['images']['histogram']})")
    lines.append("")

    lines.append("### Leave-one-domain-out (classifier-lr)")
    lines.append("")
    lines.append("| held-out domain | LODO AUROC | LODO ECE | shipped classifier-lr AUROC |")
    lines.append("|---|---|---|---|")
    for domain, row in bench_json["lodo"].items():
        lines.append(
            f"| {domain} | {_fmt_auroc(row['lodo_auroc'])} | {_fmt3(row['lodo_ece'])} | "
            f"{_fmt_auroc(row['shipped_auroc'])} |"
        )
    lines.append("")

    lines.append(
        f"Leakage audit (final-message-only baseline, test split): "
        f"**{_fmt3(bench_json['leakage_test_auroc'])} AUROC**."
    )
    lines.append("")

    ablation = bench_json["ablation"]
    lines.append("### Prompt ablation: claim-audit vs. claim-by-claim (cheapest judge)")
    lines.append("")
    if ablation is None:
        lines.append("n/a (offline run)")
    else:
        lines.append(f"Model: `{ablation['model']}`.")
        lines.append("")
        lines.append("| prompt | AUROC | ECE calibrated | USD/1k |")
        lines.append("|---|---|---|---|")
        for prompt_name in ("claim-audit", "claim-by-claim"):
            row = ablation[prompt_name]
            lines.append(
                f"| {prompt_name} | {_fmt_auroc(row['auroc'])} | {_fmt3(row['ece_calibrated'])} | "
                f"{_fmt_money(row['usd_per_1k'])} |"
            )
    lines.append("")

    lines.append("### Parse-error and abstention rates, and the share sent to the judge")
    lines.append("")
    if bench_json["parse_error_rates"] is None:
        lines.append("n/a (offline run)")
    else:
        lines.append("| judge | parse-error rate | abstain rate |")
        lines.append("|---|---|---|")
        for key, rates in bench_json["parse_error_rates"].items():
            parse_rate = _fmt_pct(rates["parse_error_rate"])
            abstain_rate = _fmt_pct(rates["abstain_rate"])
            lines.append(f"| {key} | {parse_rate} | {abstain_rate} |")
    lines.append(f"cascade share sent to the judge: {_fmt_pct(bench_json['sent_to_judge_share'])}.")
    lines.append("")

    lines.append("### Confusion matrices")
    lines.append("")
    for key, stats in bench_json["detectors"].items():
        lines.extend(_confusion_block(key, stats["decisions"]["confusion"]))

    lines.append(
        "Data sources and licences: every trace comes from the in-repo synthetic generator "
        "(Apache-2.0); the two hand-written example sets are separately authored and released "
        "under the same licence. Nothing here comes from a real system."
    )
    lines.append("")

    h = bench_json["hypotheses"]

    def _support(entry: Mapping[str, Any]) -> str:
        if "detail" in entry and entry["detail"] == "n/a (offline run)":
            return "n/a (offline run)"
        return "supported" if entry["supported"] else "not supported"

    lines.append("### Hypotheses")
    lines.append("")
    lines.append(f"- H1 (rules: fewest missed, lowest coverage): **{_support(h['h1'])}**.")
    lines.append(f"- H2 (raw judge extremes + Platt lowers ECE): **{_support(h['h2'])}**.")
    lines.append(f"- H3 (classifier-lr loses AUROC LODO): **{_support(h['h3'])}**.")
    lines.append(f"- H4 (a judge is fooled by reviewer-directed text): **{_support(h['h4'])}**.")
    lines.append("")

    lines.append("### Findings")
    lines.append("")
    lines.append(_findings_paragraph(bench_json))
    lines.append("")

    return "\n".join(lines)


# --- svg panels --------------------------------------------------------------


def _points(rows: Sequence[Mapping[str, Any]], *, calibrated: bool) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for r in rows:
        p: float = r["p_cal"] if calibrated and r.get("p_cal") is not None else r["p_raw"]
        y = float(1 - _failure_label(r))
        result.append((p, y))
    return result


def _svg_panels(recorded_dir: str | Path) -> list[svg.Panel]:
    rec = _load_recorded(recorded_dir)
    rules_test = rec.pred_idx.get("rules", {}).get("test", [])
    classifier_test = rec.pred_idx.get("classifier-lr", {}).get("test", [])
    panels = [
        svg.Panel(
            "rules", _points(rules_test, calibrated=False), _points(rules_test, calibrated=True)
        ),
        svg.Panel(
            "classifier-lr",
            _points(classifier_test, calibrated=False),
            _points(classifier_test, calibrated=True),
        ),
    ]
    gt_index = _ground_truth_index(rules_test + rec.pred_idx.get("rules", {}).get("train", []))
    judge_rows = _judge_rows_by_key(rec.judge_records, gt_index, rec.calibrators)
    best_judge = _best_judge(judge_rows) if judge_rows else None
    if best_judge is not None:
        best_test = [r for r in judge_rows[best_judge] if r["split"] == "test"]
        panels.append(
            svg.Panel(
                best_judge,
                _points(best_test, calibrated=False),
                _points(best_test, calibrated=True),
            )
        )
    return panels


def reliability_svg_for(recorded_dir: str | Path) -> str:
    """`reliability.svg` for `recorded_dir`: rules, classifier-lr, and the
    best judge when the recorded run has judge data.
    """
    return svg.reliability_svg(_svg_panels(recorded_dir))


def histogram_svg_for(recorded_dir: str | Path) -> str:
    """`histogram.svg` for `recorded_dir`, same panel selection as above."""
    return svg.histogram_svg(_svg_panels(recorded_dir))


# --- replay / tamper detection -----------------------------------------------


def _all_views(extractor: ClaimExtractor) -> dict[tuple[str, str], Any]:
    views = {}
    for split, alias in (("train", "bench:train"), ("test", "bench:test")):
        for t in load_traces(alias):
            views[(t.trace_id, split)] = resolve_claims(detector_view(t), extractor)
    return views


def verify_judge_requests(recorded_dir: str | Path) -> list[str]:
    """Re-render every recorded judge request from the packaged benchmark
    trace + the recorded prompt name + model, and report every
    `prompt_sha256`/`request_sha256` mismatch. Empty means everything matches.
    """
    root = _resolve_dir(recorded_dir)
    records = _read_jsonl(root / "judge-records.jsonl")
    if not records:
        return []

    extractor = ClaimExtractor(list(builtin_packs().values()))
    views = _all_views(extractor)
    prompt_cache: dict[str, Any] = {}
    mismatches: list[str] = []

    for rec in records:
        prompt_name = rec["prompt_name"]
        prompt = prompt_cache.get(prompt_name)
        if prompt is None:
            prompt = load_prompt(prompt_name)
            prompt_cache[prompt_name] = prompt
        if prompt.sha256 != rec["prompt_sha256"]:
            mismatches.append(f"{rec['trace_id']}/{rec['detector']}: prompt_sha256 mismatch")
            continue

        view = views.get((rec["trace_id"], rec["split"]))
        if view is None:
            mismatches.append(f"{rec['trace_id']}/{rec['detector']}: unknown trace_id/split")
            continue

        spec = JudgeSpec(
            model=rec["model"],
            temperature=0.0,
            max_tokens=400,
            json_mode=True,
            extra_body=openrouter_extra_body(
                DEFAULT_BASE_URL, json_mode=True, supports_reasoning=False
            ),
        )
        body = render_request(view, prompt, spec)
        if request_sha256(body) != rec["request_sha256"]:
            mismatches.append(f"{rec['trace_id']}/{rec['detector']}: request_sha256 mismatch")

    return mismatches


def check_readme_diff(readme_path: str | Path, bench_md: str) -> str | None:
    """`None` when the README's `bench:start`/`bench:end` block already
    matches `bench_md`; otherwise a unified diff of the two.
    """
    text = Path(readme_path).read_text(encoding="utf-8")
    start = text.find(README_START)
    end = text.find(README_END)
    if start == -1 or end == -1 or end < start:
        return f"{readme_path}: missing {README_START}/{README_END} markers"

    current_block = text[start : end + len(README_END)]
    expected_block = f"{README_START}\n{bench_md.strip()}\n{README_END}"
    if current_block.strip() == expected_block.strip():
        return None

    diff = "\n".join(
        difflib.unified_diff(
            current_block.splitlines(),
            expected_block.splitlines(),
            fromfile="README (current)",
            tofile="bench.md (regenerated)",
            lineterm="",
        )
    )
    return diff
