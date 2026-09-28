"""Tests for Platt scaling and the calibrator set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agent_claimcheck.calibration import (
    Calibrator,
    CalibratorSet,
    Platt,
    fit_calibrator,
    fit_platt,
)
from agent_claimcheck.detectors.base import DetectorOutput


def test_platt_recovers_known_parameters_on_synthetic_data() -> None:
    rng = np.random.default_rng(0)
    true_a, true_b = 2.0, -0.5
    xs = np.linspace(-3, 3, 25)
    raw_p: list[float] = []
    labels: list[int] = []
    for x in xs:
        p_true = 1.0 / (1.0 + np.exp(-(true_a * x + true_b)))
        outcomes = rng.random(80) < p_true
        for outcome in outcomes:
            raw_p.append(float(1.0 / (1.0 + np.exp(-x))))  # logit(raw_p) == x, exactly
            labels.append(int(outcome))

    fitted = fit_platt(raw_p, labels)
    assert fitted.a == pytest.approx(true_a, abs=0.3)
    assert fitted.b == pytest.approx(true_b, abs=0.3)


def test_fit_platt_is_deterministic() -> None:
    raw_p = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    labels = [0, 0, 0, 1, 1, 1, 1]
    first = fit_platt(raw_p, labels)
    second = fit_platt(raw_p, labels)
    assert first == second


def test_platt_does_not_diverge_on_perfectly_separable_data() -> None:
    """ADR 0006's whole point: target smoothing keeps `a` finite even when
    the raw scores perfectly separate the two classes.
    """
    raw_p = [0.01, 0.02, 0.03, 0.97, 0.98, 0.99]
    labels = [0, 0, 0, 1, 1, 1]
    fitted = fit_platt(raw_p, labels)
    assert np.isfinite(fitted.a)
    assert np.isfinite(fitted.b)
    assert fitted.a > 0


def test_platt_apply_is_monotonic_in_raw_p() -> None:
    platt = Platt(a=1.5, b=0.2)
    values = [platt.apply(p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert values == sorted(values)


def test_calibrator_json_round_trips() -> None:
    calibrator = fit_calibrator("rules", [0.05, 0.7, 0.97], [0, 1, 1], fitted_on="bench:train")
    restored = Calibrator.from_dict(calibrator.to_dict())
    assert restored == calibrator


def test_calibrator_set_save_and_load_round_trip(tmp_path: Path) -> None:
    calibrator_set = CalibratorSet(
        version=1,
        fitted_on="bench:train",
        base_rate=0.6,
        calibrators={
            "rules": fit_calibrator("rules", [0.05, 0.7, 0.97], [0, 1, 1], fitted_on="bench:train"),
            "classifier-lr": fit_calibrator(
                "classifier-lr", [0.2, 0.5, 0.8], [0, 1, 1], fitted_on="bench:train"
            ),
        },
    )
    path = tmp_path / "calibration.json"
    calibrator_set.save(path)
    loaded = CalibratorSet.load(path)
    assert loaded == calibrator_set
    # .json files: indent=2, sort_keys, trailing newline.
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.startswith("{\n")


def test_calibrator_set_apply_returns_none_for_an_unregistered_detector() -> None:
    calibrator_set = CalibratorSet(version=1, fitted_on="bench:train", base_rate=0.6)
    output = DetectorOutput(detector="trust-agent", p_success=1.0, abstain=False, reasons=[])
    assert calibrator_set.apply("trust-agent", output) is None


def test_abstaining_outputs_are_never_calibrated() -> None:
    calibrator_set = CalibratorSet(
        version=1,
        fitted_on="bench:train",
        base_rate=0.6,
        calibrators={
            "rules": fit_calibrator("rules", [0.05, 0.7, 0.97], [0, 1, 1], fitted_on="bench:train")
        },
    )
    output = DetectorOutput(
        detector="rules", p_success=0.6, abstain=True, abstain_reason="no_rule", reasons=[]
    )
    assert calibrator_set.apply("rules", output) is None


def test_apply_calibrates_a_non_abstaining_output() -> None:
    calibrator_set = CalibratorSet(
        version=1,
        fitted_on="bench:train",
        base_rate=0.6,
        calibrators={
            "rules": fit_calibrator("rules", [0.05, 0.7, 0.97], [0, 1, 1], fitted_on="bench:train")
        },
    )
    output = DetectorOutput(detector="rules", p_success=0.7, abstain=False, reasons=[])
    calibrated = calibrator_set.apply("rules", output)
    assert calibrated is not None
    assert 0.0 <= calibrated <= 1.0
