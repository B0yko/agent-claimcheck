"""Detector metrics: discrimination, calibration and decision statistics.

Every function here is a plain function of parallel arrays (labels, scores,
verdicts), so each one is testable against a hand-computed value without a
`Checker` or a benchmark in the loop. `evaluate` is the one entry point that
joins a detector's `CheckResult`s back to their traces' ground truth and
assembles all of them.

Convention throughout: the positive class for discrimination is
`false_success` (the agent claimed success but the outcome was a failure),
scored as `1 - p_success`. `y_success` (used by `ece`/`brier`) is the
opposite sense: 1 for an actually-successful trace, 0 for a failure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from agent_claimcheck.checker import CheckResult
    from agent_claimcheck.schema import Trace

#: `evaluate`'s bootstrap CI settings (fixed, not configurable).
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
ECE_BINS = 10
EXTREME_LOW = 0.05
EXTREME_HIGH = 0.95


def _rank_average(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties replaced by the average rank of the tied group."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        # Ranks i..j (0-based) are 1-based (i+1)..(j+1); their average:
        ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """AUROC by the Mann-Whitney rank-sum method, ties averaged.

    `y_true` is 1 for the positive class, 0 otherwise. Requires at least one
    of each class.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    n_pos = float(y.sum())
    n_neg = float(len(y)) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auroc requires both classes to be present")
    ranks = _rank_average(s)
    sum_ranks_pos = float(ranks[y == 1.0].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def bootstrap_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """A 95% CI for `auroc(y_true, y_score)`: `n_resamples` resamples, each
    stratified by label (drawn with replacement within each class at the
    class's own size), `numpy.random.default_rng(seed)`, percentile 2.5/97.5.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    pos_idx = np.flatnonzero(y == 1.0)
    neg_idx = np.flatnonzero(y == 0.0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("bootstrap_ci requires both classes to be present")

    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=float)
    for k in range(n_resamples):
        pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pos_sample, neg_sample])
        values[k] = auroc(y[idx].tolist(), s[idx].tolist())

    lo, hi = np.percentile(values, [2.5, 97.5], method="linear")
    return float(lo), float(hi)


def ece(p_success: Sequence[float], y_success: Sequence[int], *, n_bins: int = ECE_BINS) -> float:
    """Expected calibration error: `n_bins` equal-width bins on `p_success`
    over `[0, 1]`, weighted by bin count, against the bin's actual success
    rate. The last bin's upper edge is inclusive.
    """
    p = np.asarray(p_success, dtype=float)
    y = np.asarray(y_success, dtype=float)
    total = len(p)
    if total == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_error = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        total_error += (count / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total_error


def brier(p_success: Sequence[float], y_success: Sequence[int]) -> float:
    """Mean squared error between `p_success` and the actual outcome."""
    p = np.asarray(p_success, dtype=float)
    y = np.asarray(y_success, dtype=float)
    if len(p) == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def extremes_share(
    p_success: Sequence[float], *, low: float = EXTREME_LOW, high: float = EXTREME_HIGH
) -> float:
    """Share of `p_success` values below `low` or above `high`."""
    p = np.asarray(p_success, dtype=float)
    if len(p) == 0:
        return 0.0
    return float(np.mean((p < low) | (p > high)))


def decision_stats(verdicts: Sequence[str], labels: Sequence[int]) -> dict[str, Any]:
    """Gate decision statistics: `labels` is 1 for an actual false success
    (label `failure`), 0 for an actual genuine success, aligned with
    `verdicts`.
    """
    n = len(verdicts)
    n_failure = sum(labels)
    n_success = n - n_failure

    confusion: dict[str, dict[str, int]] = {
        "success": {"verified": 0, "false_success": 0, "unverifiable": 0},
        "failure": {"verified": 0, "false_success": 0, "unverifiable": 0},
    }
    caught = 0
    missed = 0
    false_alarms = 0
    sent_to_review = 0
    correct = 0
    decided = 0

    for verdict, label in zip(verdicts, labels, strict=True):
        actual = "failure" if label == 1 else "success"
        confusion[actual][verdict] += 1
        if verdict == "unverifiable":
            sent_to_review += 1
            continue
        decided += 1
        if actual == "failure":
            if verdict == "false_success":
                caught += 1
                correct += 1
            elif verdict == "verified":
                missed += 1
        else:
            if verdict == "verified":
                correct += 1
            elif verdict == "false_success":
                false_alarms += 1

    return {
        "n": n,
        "n_success": n_success,
        "n_failure": n_failure,
        "coverage": (decided / n) if n else 0.0,
        "accuracy_on_decided": (correct / decided) if decided else 0.0,
        "caught": caught,
        "missed": missed,
        "false_alarms": false_alarms,
        "sent_to_review": sent_to_review,
        "confusion": confusion,
    }


def _empty_evaluation() -> dict[str, Any]:
    return {
        "n": 0,
        "n_success": 0,
        "n_failure": 0,
        "auroc": None,
        "ece_raw": 0.0,
        "ece_calibrated": 0.0,
        "brier_raw": 0.0,
        "brier_calibrated": 0.0,
        "extremes_raw": 0.0,
        "extremes_calibrated": 0.0,
        "decisions": decision_stats([], []),
    }


def evaluate(results: Sequence[CheckResult], traces: Sequence[Trace]) -> dict[str, Any]:
    """Join one detector's `CheckResult`s to `traces`' ground truth and
    compute every metric above.

    `skipped` results are excluded, as are results whose trace has no usable
    ground truth (missing, or `outcome: "unknown"`): both leave no label to
    score against. An abstaining detector's `p_success` is already its
    reported base rate (never calibrated), so it is scored like any other
    result, per the base-rate convention every detector uses for abstention.
    """
    by_id = {t.trace_id: t for t in traces}

    labels: list[int] = []  # 1 = false success (failure), 0 = genuine success
    p_raw: list[float] = []
    p_used: list[float] = []  # calibrated when the result carries one, else raw
    verdicts: list[str] = []

    for r in results:
        if r.verdict == "skipped":
            continue
        trace = by_id.get(r.trace_id)
        if trace is None or trace.ground_truth is None or trace.ground_truth.outcome == "unknown":
            continue
        if r.p_success_raw is None or r.p_success is None:
            continue
        labels.append(1 if trace.ground_truth.outcome == "failure" else 0)
        p_raw.append(r.p_success_raw)
        p_used.append(r.p_success)
        verdicts.append(r.verdict)

    if not labels:
        return _empty_evaluation()

    y_success = [1 - label for label in labels]
    n_success = sum(y_success)
    n_failure = len(labels) - n_success

    result: dict[str, Any] = {
        "n": len(labels),
        "n_success": n_success,
        "n_failure": n_failure,
        "ece_raw": ece(p_raw, y_success),
        "ece_calibrated": ece(p_used, y_success),
        "brier_raw": brier(p_raw, y_success),
        "brier_calibrated": brier(p_used, y_success),
        "extremes_raw": extremes_share(p_raw),
        "extremes_calibrated": extremes_share(p_used),
        "decisions": decision_stats(verdicts, labels),
    }

    score = [1.0 - p for p in p_used]
    if n_success > 0 and n_failure > 0:
        value = auroc(labels, score)
        lo, hi = bootstrap_ci(labels, score)
        result["auroc"] = {"value": value, "ci_low": lo, "ci_high": hi}
    else:
        result["auroc"] = None

    return result
