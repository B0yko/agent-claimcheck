"""The classifier detector: standardised logistic regression over the
generic features in `features.py`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from agent_claimcheck.detectors.base import DetectorOutput, Reason, register_detector
from agent_claimcheck.features import FEATURES, extract
from agent_claimcheck.redact import DetectorView

#: C values tried by the CV grid search, narrowest to widest.
C_GRID: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)

#: Reported as `p_success` when the classifier abstains.
DEFAULT_BASE_RATE = 0.6

_FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _feature_matrix(views: Sequence[DetectorView]) -> np.ndarray:
    rows = [extract(v) for v in views]
    return np.array([[row[name] for name in _FEATURE_NAMES] for row in rows], dtype=float)


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population mean/std (ddof 0); a zero-variance feature gets scale 1.0."""
    means = x.mean(axis=0)
    scales = x.std(axis=0, ddof=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    return (x - means) / scales, means, scales


def _fit_one(x: np.ndarray, y: np.ndarray, c: float) -> LogisticRegression:
    # L2 penalty (the default: l1_ratio=0.0), lbfgs.
    model = LogisticRegression(C=c, solver="lbfgs", tol=1e-10, max_iter=10000)
    model.fit(x, y)
    return model


def _positive_proba(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(x.shape[0])
    result: np.ndarray = proba[:, classes.index(1)]
    return result


def _select_c(x_std: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, list[float]]:
    """The C in `C_GRID` with the lowest 5-fold mean log-loss; same folds for
    every C (a fresh `StratifiedKFold(..., random_state=seed)` reproduces
    identical splits every time it is asked to split the same `x`, `y`).
    """
    mean_loglosses: list[float] = []
    for c in C_GRID:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_losses: list[float] = []
        for train_idx, test_idx in splitter.split(x_std, y):
            model = _fit_one(x_std[train_idx], y[train_idx], c)
            p1 = _positive_proba(model, x_std[test_idx])
            fold_losses.append(float(log_loss(y[test_idx], p1, labels=[0, 1])))
        mean_loglosses.append(float(np.mean(fold_losses)))
    best_idx = int(np.argmin(mean_loglosses))
    return C_GRID[best_idx], mean_loglosses


def train_lr(
    views: Sequence[DetectorView],
    labels: Sequence[int],
    domains: Sequence[str],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit the classifier on `views` (target = success) and return the
    `claimcheck-lr/v1` JSON artifact: `StandardScaler`
    followed by `LogisticRegression`, `C` chosen by 5-fold CV log-loss.
    """
    x = _feature_matrix(views)
    y = np.array(labels, dtype=int)
    x_std, means, scales = _standardise(x)
    best_c, cv_logloss = _select_c(x_std, y, seed)
    model = _fit_one(x_std, y, best_c)
    return {
        "format": "claimcheck-lr/v1",
        "features": list(_FEATURE_NAMES),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "C": best_c,
        "cv": {"grid": list(C_GRID), "logloss": cv_logloss},
        "training_domains": sorted(set(domains)),
        "base_rate": float(np.mean(y)) if len(y) else DEFAULT_BASE_RATE,
        "n_train": len(views),
        "seed": seed,
    }


def oof_predictions(
    views: Sequence[DetectorView],
    labels: Sequence[int],
    *,
    seed: int = 0,
) -> list[float]:
    """5-fold `StratifiedKFold(shuffle=True, random_state=seed)` out-of-fold
    `p_success`, for fitting the `classifier-lr` calibrator:
    the same C-selection as `train_lr`, fit on 4 folds, scored on the held-
    out one.
    """
    x = _feature_matrix(views)
    y = np.array(labels, dtype=int)
    x_std, _means, _scales = _standardise(x)
    best_c, _cv_logloss = _select_c(x_std, y, seed)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(views))
    for train_idx, test_idx in splitter.split(x_std, y):
        model = _fit_one(x_std[train_idx], y[train_idx], best_c)
        oof[test_idx] = _positive_proba(model, x_std[test_idx])
    return oof.tolist()


def fit_lodo(
    views: Sequence[DetectorView],
    labels: Sequence[int],
    domains: Sequence[str],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a classifier artifact the same way `train_lr` does, meant to be
    called with only two domains' worth of train traces (leave-one-domain-
    out): the caller filters `views`/`labels`/`domains` to those two domains
    and scores the held-out domain's test traces with `guard=False`.
    """
    return train_lr(views, labels, domains, seed=seed)


def load_artifact(path: str | Path | None = None) -> dict[str, Any]:
    """Load a `claimcheck-lr/v1` JSON artifact; defaults to the packaged
    `models/lr-v1.json`. JSON only, per ADR 0003 — never pickle or joblib.
    """
    resolved = Path(path) if path is not None else _MODELS_DIR / "lr-v1.json"
    with resolved.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
        return data


class ClassifierDetector:
    """Scores a trace with a standardised logistic regression."""

    name = "classifier-lr"

    def __init__(self, artifact: dict[str, Any], *, guard: bool = True) -> None:
        self.artifact = artifact
        self.guard = guard
        self.features: list[str] = list(artifact["features"])
        self.means = np.array(artifact["means"], dtype=float)
        self.scales = np.array(artifact["scales"], dtype=float)
        self.coef = np.array(artifact["coef"], dtype=float)
        self.intercept = float(artifact["intercept"])
        self.training_domains: set[str] = set(artifact["training_domains"])
        self.base_rate = float(artifact["base_rate"])

    def _abstain(self, trace: DetectorView) -> DetectorOutput | None:
        if not self.guard:
            return None
        has_result = any(s.kind == "tool_result" for s in trace.steps)
        if trace.task.domain in self.training_domains and has_result:
            return None
        return DetectorOutput(
            detector=self.name,
            p_success=self.base_rate,
            abstain=True,
            abstain_reason="out_of_distribution",
            reasons=[
                Reason(
                    claim=None,
                    outcome="out_of_distribution",
                    step=None,
                    detail="trace domain or missing tool results are outside the training set",
                )
            ],
        )

    def score(self, trace: DetectorView) -> DetectorOutput:
        abstained = self._abstain(trace)
        if abstained is not None:
            return abstained

        row = extract(trace)
        raw = np.array([row[name] for name in self.features], dtype=float)
        standardised = (raw - self.means) / self.scales
        contributions = self.coef * standardised
        logit = float(contributions.sum() + self.intercept)
        p = float(1.0 / (1.0 + np.exp(-logit)))

        order = np.argsort(-np.abs(contributions))
        top = [{"feature": self.features[i], "value": float(contributions[i])} for i in order[:10]]
        reasons: list[Reason] = []
        if len(order):
            top_i = int(order[0])
            reasons.append(
                Reason(
                    claim=None,
                    outcome="classifier-lr",
                    step=None,
                    detail=(
                        f"strongest signal: {self.features[top_i]} ({contributions[top_i]:+.3f})"
                    )[:200],
                )
            )

        return DetectorOutput(
            detector=self.name,
            p_success=p,
            abstain=False,
            reasons=reasons,
            details={"contributions": top},
        )


register_detector("classifier-lr", ClassifierDetector)
