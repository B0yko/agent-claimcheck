"""Platt scaling: the only v0.1 calibrator (ADR 0006).

A one-dimensional logistic regression on the clipped logit of a detector's
raw `p_success`, fitted by Newton's method to convergence on smoothed
targets (Platt, 1999). `CalibratorSet` is the small JSON file `train` writes
and `check` loads: one `Calibrator` per detector key (`rules`,
`classifier-lr`), plus the base rate abstentions keep instead of being
calibrated.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from agent_claimcheck.detectors.base import DetectorOutput

#: Logit clip: `p_success` is clamped to `[clip, 1 - clip]` before the logit.
DEFAULT_CLIP = 1e-6
_MAX_ITERS = 200
_CONVERGENCE = 1e-12


def _clip(p: np.ndarray, clip: float) -> np.ndarray:
    result: np.ndarray = np.clip(p, clip, 1.0 - clip)
    return result


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Platt:
    """A fitted Platt calibrator: `p' = sigmoid(a * x + b)`."""

    a: float
    b: float
    clip: float = DEFAULT_CLIP

    def apply(self, p: float) -> float:
        x = _logit(_clip(np.array([p], dtype=float), self.clip))
        return float(_sigmoid(self.a * x[0] + self.b))


def fit_platt(
    raw_p: Sequence[float], labels: Sequence[int], *, clip: float = DEFAULT_CLIP
) -> Platt:
    """Fit `a, b` by Newton's method on `x = logit(clip(raw_p))`, against
    Platt's smoothed targets (`t+ = (N+ + 1)/(N+ + 2)`, `t- = 1/(N- + 2)`).
    Deterministic: stops when `|delta_a| < 1e-12` and `|delta_b| < 1e-12`,
    at most 200 iterations.
    """
    x = _logit(_clip(np.asarray(raw_p, dtype=float), clip))
    y = np.asarray(labels, dtype=float)
    n_pos = float(y.sum())
    n_neg = float(len(y)) - n_pos
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    t = np.where(y == 1.0, t_pos, t_neg)

    a, b = 0.0, 0.0
    for _ in range(_MAX_ITERS):
        z = a * x + b
        p = _sigmoid(z)
        w = p * (1.0 - p)
        w = np.where(w < 1e-12, 1e-12, w)

        grad_a = float(np.sum((p - t) * x))
        grad_b = float(np.sum(p - t))
        h_aa = float(np.sum(w * x * x))
        h_ab = float(np.sum(w * x))
        h_bb = float(np.sum(w))

        det = h_aa * h_bb - h_ab * h_ab
        if det == 0.0:
            break
        delta_a = (h_bb * grad_a - h_ab * grad_b) / det
        delta_b = (h_aa * grad_b - h_ab * grad_a) / det
        a -= delta_a
        b -= delta_b
        if abs(delta_a) < _CONVERGENCE and abs(delta_b) < _CONVERGENCE:
            break

    return Platt(a=a, b=b, clip=clip)


@dataclass(frozen=True)
class Calibrator:
    """One detector's fitted Platt calibrator, plus its provenance."""

    method: str
    a: float
    b: float
    clip: float
    n: int
    n_pos: int
    fitted_on: str
    detector: str

    def apply(self, p: float) -> float:
        return Platt(a=self.a, b=self.b, clip=self.clip).apply(p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "a": self.a,
            "b": self.b,
            "clip": self.clip,
            "n": self.n,
            "n_pos": self.n_pos,
            "fitted_on": self.fitted_on,
            "detector": self.detector,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Calibrator:
        return Calibrator(
            method=d["method"],
            a=float(d["a"]),
            b=float(d["b"]),
            clip=float(d["clip"]),
            n=int(d["n"]),
            n_pos=int(d["n_pos"]),
            fitted_on=d["fitted_on"],
            detector=d["detector"],
        )


def fit_calibrator(
    detector: str,
    raw_p: Sequence[float],
    labels: Sequence[int],
    *,
    fitted_on: str,
    clip: float = DEFAULT_CLIP,
) -> Calibrator:
    """Fit one detector's Platt calibrator on its non-abstaining raw scores."""
    platt = fit_platt(raw_p, labels, clip=clip)
    return Calibrator(
        method="platt",
        a=platt.a,
        b=platt.b,
        clip=clip,
        n=len(raw_p),
        n_pos=int(sum(labels)),
        fitted_on=fitted_on,
        detector=detector,
    )


@dataclass(frozen=True)
class CalibratorSet:
    """The `calibration.json` file: one calibrator per detector key."""

    version: int
    fitted_on: str
    base_rate: float
    calibrators: dict[str, Calibrator] = field(default_factory=dict)

    def apply(self, detector: str, output: DetectorOutput) -> float | None:
        """Calibrated `p_success` for `output`, or `None` ("n/a") when no
        calibrator is registered for `detector`. Abstaining outputs are never
        calibrated: they keep their own (base-rate) `p_success`.
        """
        if output.abstain:
            return None
        calibrator = self.calibrators.get(detector)
        if calibrator is None:
            return None
        return calibrator.apply(output.p_success)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fitted_on": self.fitted_on,
            "base_rate": self.base_rate,
            "calibrators": {key: cal.to_dict() for key, cal in self.calibrators.items()},
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CalibratorSet:
        return CalibratorSet(
            version=int(d["version"]),
            fitted_on=d["fitted_on"],
            base_rate=float(d["base_rate"]),
            calibrators={
                key: Calibrator.from_dict(v) for key, v in d.get("calibrators", {}).items()
            },
        )

    @staticmethod
    def load(path: str | Path) -> CalibratorSet:
        with Path(path).open("r", encoding="utf-8") as fh:
            return CalibratorSet.from_dict(json.load(fh))

    def save(self, path: str | Path) -> None:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        Path(path).write_text(text, encoding="utf-8")
