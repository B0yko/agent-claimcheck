"""The recorded v0.1.0 run: every fitted artifact it ships can be re-derived.

`bench --from-recorded` replays the recorded calibrators and leave-one-domain-out
models instead of refitting them, so the report is byte-identical across
platforms. These tests refit both from the run's own inputs and check that they
agree with the recorded values within 1e-6.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agent_claimcheck.calibration import fit_platt
from agent_claimcheck.claims import ClaimExtractor
from agent_claimcheck.detectors.classifier import train_lr
from agent_claimcheck.redact import detector_view, resolve_claims
from agent_claimcheck.rules.engine import builtin_packs
from agent_claimcheck.schema import load_traces

RECORDED = Path(__file__).resolve().parents[1] / "results" / "v0.1.0"
BUILTIN_CALIBRATORS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "agent_claimcheck"
    / "models"
    / "calibrators-v1.json"
)


def _jsonl(name: str) -> list[dict[str, Any]]:
    with (RECORDED / name).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _train_labels() -> dict[str, int]:
    return {
        t.trace_id: 1 if t.ground_truth is not None and t.ground_truth.outcome == "success" else 0
        for t in load_traces("bench:train")
    }


def test_recorded_calibrators_refit_within_tolerance() -> None:
    calibrators = json.loads((RECORDED / "calibration.json").read_text(encoding="utf-8"))
    labels = _train_labels()
    predictions = _jsonl("offline-predictions.jsonl")
    judge_records = _jsonl("judge-records.jsonl")
    sources = {"rules": "rules", "classifier-lr": "classifier-lr:oof"}

    for key, recorded in calibrators["calibrators"].items():
        if key in sources:
            rows = [
                r
                for r in predictions
                if r["detector"] == sources[key] and r["split"] == "train" and not r["abstain"]
            ]
        else:
            rows = [
                r
                for r in judge_records
                if r["detector"] == key and r["split"] == "train" and not r["abstain"]
            ]
        assert rows, key
        platt = fit_platt([r["p_raw"] for r in rows], [labels[r["trace_id"]] for r in rows])
        assert platt.a == pytest.approx(recorded["a"], abs=1e-6), key
        assert platt.b == pytest.approx(recorded["b"], abs=1e-6), key
        assert recorded["n"] == len(rows), key


def test_recorded_lodo_models_retrain_within_tolerance() -> None:
    recorded = json.loads((RECORDED / "lodo-models.json").read_text(encoding="utf-8"))
    extractor = ClaimExtractor(list(builtin_packs().values()))
    train = load_traces("bench:train")
    for held_out, artifact in recorded.items():
        kept = [t for t in train if t.task.domain != held_out]
        retrained = train_lr(
            [resolve_claims(detector_view(t), extractor) for t in kept],
            [1 if t.ground_truth and t.ground_truth.outcome == "success" else 0 for t in kept],
            [t.task.domain for t in kept],
            seed=artifact["seed"],
        )
        assert retrained["features"] == artifact["features"]
        assert retrained["C"] == artifact["C"]
        for key in ("coef", "means", "scales"):
            assert np.array(retrained[key]) == pytest.approx(np.array(artifact[key]), abs=1e-6)
        assert retrained["intercept"] == pytest.approx(artifact["intercept"], abs=1e-6)


def test_builtin_calibrators_are_the_recorded_ones() -> None:
    recorded = json.loads((RECORDED / "calibration.json").read_text(encoding="utf-8"))
    builtin = json.loads(BUILTIN_CALIBRATORS.read_text(encoding="utf-8"))
    assert builtin == recorded
