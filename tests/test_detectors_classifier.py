"""Tests for the classifier-lr detector and trainer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from factory import probe, tool_call, tool_result, trace

from agent_claimcheck.claims import ClaimExtractor, success_claims
from agent_claimcheck.detectors.classifier import (
    ClassifierDetector,
    oof_predictions,
    train_lr,
)
from agent_claimcheck.features import extract
from agent_claimcheck.redact import detector_view, resolve_claims
from agent_claimcheck.rules.engine import builtin_packs
from agent_claimcheck.schema import load_traces

REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def _genuine(i: int) -> object:
    return trace(
        f"g{i}",
        "booking",
        [
            tool_call(0, "calendar.create_event", args={"start": "2026-03-02T09:00:00Z"}),
            tool_result(1, "calendar.create_event", output={"event_id": f"e{i}", "id": f"e{i}"}),
            probe(2, "calendar.get_event", output={"status": "confirmed"}),
        ],
        text="Booked for 2026-03-02T09:00:00Z.",
        claims=[("booked", {"start": "2026-03-02T09:00:00Z"})],
    )


def _false(i: int) -> object:
    return trace(
        f"f{i}",
        "booking",
        [
            tool_call(0, "calendar.create_event", args={"start": "2026-03-02T09:00:00Z"}),
            tool_result(1, "calendar.create_event", ok=False, error="409: conflict"),
        ],
        text="Booked for 2026-03-02T09:00:00Z.",
        claims=[("booked", {"start": "2026-03-02T09:00:00Z"})],
    )


def _views_labels_domains(n: int = 10) -> tuple[list[object], list[int], list[str]]:
    traces = [_genuine(i) for i in range(n)] + [_false(i) for i in range(n)]
    views = [detector_view(t) for t in traces]
    labels = [1] * n + [0] * n
    domains = ["booking"] * (2 * n)
    return views, labels, domains


def test_no_pickle_or_joblib_import_anywhere_in_src() -> None:
    """ADR 0003: model artifacts are JSON only, never pickle or joblib."""
    for path in REPO_SRC.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import pickle"), path
            assert not stripped.startswith("import joblib"), path
            assert "from pickle" not in stripped, path
            assert "from joblib" not in stripped, path


def test_train_lr_artifact_shape() -> None:
    views, labels, domains = _views_labels_domains()
    artifact = train_lr(views, labels, domains, seed=0)
    assert artifact["format"] == "claimcheck-lr/v1"
    assert len(artifact["coef"]) == len(artifact["features"])
    assert len(artifact["means"]) == len(artifact["features"])
    assert len(artifact["scales"]) == len(artifact["features"])
    assert artifact["training_domains"] == ["booking"]
    assert artifact["C"] in artifact["cv"]["grid"]
    assert len(artifact["cv"]["logloss"]) == len(artifact["cv"]["grid"])
    assert artifact["n_train"] == len(views)


def test_contributions_plus_intercept_equal_the_logit() -> None:
    views, labels, domains = _views_labels_domains()
    artifact = train_lr(views, labels, domains, seed=0)
    detector = ClassifierDetector(artifact, guard=False)

    view = detector_view(_genuine(0))
    row = extract(view)
    raw = np.array([row[name] for name in artifact["features"]], dtype=float)
    means = np.array(artifact["means"], dtype=float)
    scales = np.array(artifact["scales"], dtype=float)
    coef = np.array(artifact["coef"], dtype=float)
    full_contributions = coef * ((raw - means) / scales)
    expected_logit = float(full_contributions.sum() + artifact["intercept"])

    output = detector.score(view)
    actual_logit = math.log(output.p_success / (1.0 - output.p_success))
    assert actual_logit == pytest.approx(expected_logit, abs=1e-9)


def test_contributions_details_kept_to_ten_sorted_by_magnitude() -> None:
    views, labels, domains = _views_labels_domains()
    artifact = train_lr(views, labels, domains, seed=0)
    detector = ClassifierDetector(artifact, guard=False)
    output = detector.score(detector_view(_genuine(0)))
    contributions = output.details["contributions"]
    assert len(contributions) <= 10
    values = [abs(c["value"]) for c in contributions]
    assert values == sorted(values, reverse=True)


def test_ood_guard_abstains_on_unknown_domain() -> None:
    views, labels, domains = _views_labels_domains()
    artifact = train_lr(views, labels, domains, seed=0)
    detector = ClassifierDetector(artifact, guard=True)

    other_domain = trace(
        "crm1",
        "crm",
        [
            tool_call(0, "crm.update_contact", args={"record_id": "c1"}),
            tool_result(1, "crm.update_contact", output={"record_id": "c1"}),
        ],
        claims=[("updated", {})],
    )
    output = detector.score(detector_view(other_domain))
    assert output.abstain is True
    assert output.abstain_reason == "out_of_distribution"
    assert output.p_success == artifact["base_rate"]


def test_ood_guard_abstains_without_tool_results() -> None:
    views, labels, domains = _views_labels_domains()
    artifact = train_lr(views, labels, domains, seed=0)
    detector = ClassifierDetector(artifact, guard=True)

    no_results = trace(
        "b1", "booking", [tool_call(0, "calendar.create_event", args={})], claims=[("booked", {})]
    )
    output = detector.score(detector_view(no_results))
    assert output.abstain is True
    assert output.abstain_reason == "out_of_distribution"


def test_guard_false_scores_out_of_distribution_traces_anyway() -> None:
    views, labels, domains = _views_labels_domains()
    artifact = train_lr(views, labels, domains, seed=0)
    detector = ClassifierDetector(artifact, guard=False)

    other_domain = trace(
        "crm2",
        "crm",
        [
            tool_call(0, "crm.update_contact", args={"record_id": "c1"}),
            tool_result(1, "crm.update_contact", output={"record_id": "c1"}),
        ],
        claims=[("updated", {})],
    )
    output = detector.score(detector_view(other_domain))
    assert output.abstain is False


def test_example_browser_abstains_for_every_trace() -> None:
    views, labels, domains = _views_labels_domains()
    artifact = train_lr(views, labels, domains, seed=0)
    detector = ClassifierDetector(artifact, guard=True)

    extractor = ClaimExtractor(list(builtin_packs().values()))
    checked = 0
    for t in load_traces("example:browser"):
        view = resolve_claims(detector_view(t), extractor)
        if not success_claims(view.final_claim.claims):
            continue
        checked += 1
        output = detector.score(view)
        assert output.abstain is True
        assert output.abstain_reason == "out_of_distribution"
    assert checked > 0


def test_oof_predictions_are_one_per_trace_and_bounded() -> None:
    views, labels, domains = _views_labels_domains()
    oof = oof_predictions(views, labels, seed=0)
    assert len(oof) == len(views)
    assert all(0.0 <= p <= 1.0 for p in oof)


def test_oof_predictions_are_deterministic() -> None:
    views, labels, _domains = _views_labels_domains()
    first = oof_predictions(views, labels, seed=0)
    second = oof_predictions(views, labels, seed=0)
    assert first == second
