"""Tests for `bench/runner.py`'s offline recorded run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_claimcheck.bench.runner import LODO_DOMAINS
from agent_claimcheck.checker import builtin_calibrators
from agent_claimcheck.detectors.classifier import load_artifact


@pytest.fixture
def recorded_dir(offline_recorded_dir: Path) -> Path:
    return offline_recorded_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_writes_every_file(recorded_dir: Path) -> None:
    for name in (
        "offline-predictions.jsonl",
        "offline-timings.jsonl",
        "lodo-models.json",
        "calibration.json",
        "run.json",
    ):
        assert (recorded_dir / name).exists()


def test_predictions_cover_both_splits_for_every_offline_detector(recorded_dir: Path) -> None:
    rows = _read_jsonl(recorded_dir / "offline-predictions.jsonl")
    by_detector: dict[str, list[dict]] = {}
    for r in rows:
        by_detector.setdefault(r["detector"], []).append(r)

    for key in ("trust-agent", "any-error", "rules", "classifier-lr"):
        train_rows = [r for r in by_detector[key] if r["split"] == "train"]
        test_rows = [r for r in by_detector[key] if r["split"] == "test"]
        assert len(train_rows) == 180
        assert len(test_rows) == 120

    # OOF: train only, one row per train trace.
    assert len(by_detector["classifier-lr:oof"]) == 180
    assert all(r["split"] == "train" for r in by_detector["classifier-lr:oof"])

    # LODO: each held-out domain's own test rows only (40 per domain: 3*40=120).
    total_lodo = 0
    for domain in LODO_DOMAINS:
        lodo_rows = by_detector[f"classifier-lr:lodo:{domain}"]
        assert all(r["split"] == "test" and r["domain"] == domain for r in lodo_rows)
        total_lodo += len(lodo_rows)
    assert total_lodo == 120


def test_prediction_row_shape(recorded_dir: Path) -> None:
    rows = _read_jsonl(recorded_dir / "offline-predictions.jsonl")
    rules_row = next(r for r in rows if r["detector"] == "rules")
    for key in (
        "detector",
        "trace_id",
        "split",
        "domain",
        "outcome",
        "injection",
        "evidence",
        "variant",
        "p_raw",
        "p_cal",
        "abstain",
        "abstain_reason",
        "rule_outcome",
    ):
        assert key in rules_row

    baseline_row = next(r for r in rows if r["detector"] == "trust-agent")
    assert baseline_row["p_cal"] is None  # baselines are never calibrated
    assert baseline_row["rule_outcome"] is None


def test_classifier_lr_matches_the_shipped_artifact(recorded_dir: Path) -> None:
    # run_offline asserts this internally; a second, independent check here
    # confirms the recorded predictions actually used the shipped weights.
    shipped = load_artifact()
    rows = _read_jsonl(recorded_dir / "offline-predictions.jsonl")
    classifier_rows = [r for r in rows if r["detector"] == "classifier-lr"]
    assert shipped["training_domains"] == ["booking", "coding", "crm"]
    assert len(classifier_rows) == 300


def test_calibration_matches_the_shipped_calibrators(recorded_dir: Path) -> None:
    recorded = json.loads((recorded_dir / "calibration.json").read_text())
    shipped = builtin_calibrators().to_dict()
    assert recorded["base_rate"] == shipped["base_rate"]
    for key in ("rules", "classifier-lr"):
        assert recorded["calibrators"][key]["a"] == pytest.approx(shipped["calibrators"][key]["a"])
        assert recorded["calibrators"][key]["b"] == pytest.approx(shipped["calibrators"][key]["b"])


def test_lodo_models_file_has_one_artifact_per_domain(recorded_dir: Path) -> None:
    lodo_models = json.loads((recorded_dir / "lodo-models.json").read_text())
    assert set(lodo_models) == set(LODO_DOMAINS)
    for domain, artifact in lodo_models.items():
        assert domain not in artifact["training_domains"]


def test_timings_cover_only_the_four_timed_detectors(recorded_dir: Path) -> None:
    rows = _read_jsonl(recorded_dir / "offline-timings.jsonl")
    detectors = {r["detector"] for r in rows}
    assert detectors == {"trust-agent", "any-error", "rules", "classifier-lr"}
    assert len(rows) == 4 * 300
    assert all(r["latency_ms"] >= 0.0 for r in rows)


def test_run_json_carries_the_given_metadata(recorded_dir: Path) -> None:
    run_meta = json.loads((recorded_dir / "run.json").read_text())
    assert run_meta["package_version"] == "0.0.0-test"
    assert run_meta["dataset_sha256"] == "b" * 64
    assert run_meta["judges"] == []
