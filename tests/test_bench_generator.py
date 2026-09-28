"""Tests for the benchmark generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from agent_claimcheck import resources
from agent_claimcheck.bench.generator import generate
from agent_claimcheck.bench.generator._blocked_words import BLOCKED_SYLLABLE_WORDS
from agent_claimcheck.bench.generator.leakage import (
    leakage_auroc,
    leakage_auroc_train_test,
    rank_auroc,
)
from agent_claimcheck.bench.generator.plan import DOMAINS, FALSE_KINDS, GENUINE_KINDS, full_plan
from agent_claimcheck.bench.generator.pools import build_pools
from agent_claimcheck.bench.generator.validate import _domain_template_texts, validate_dataset
from agent_claimcheck.claims import ClaimExtractor, success_claims
from agent_claimcheck.cli import app
from agent_claimcheck.features import FEATURES
from agent_claimcheck.redact import detector_view, resolve_claims
from agent_claimcheck.rules.engine import builtin_packs
from agent_claimcheck.schema import Trace

SEED = 20260924
runner = CliRunner()


def _parse_lines(text: str) -> list[Trace]:
    return [Trace.model_validate(json.loads(line)) for line in text.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def dataset():
    return generate(SEED)


@pytest.fixture(scope="module")
def all_traces(dataset) -> list[Trace]:
    return _parse_lines(dataset.train_jsonl) + _parse_lines(dataset.test_jsonl)


def test_byte_identical_regeneration(dataset) -> None:
    again = generate(SEED)
    assert dataset.train_jsonl == again.train_jsonl
    assert dataset.test_jsonl == again.test_jsonl
    assert dataset.manifest_json == again.manifest_json
    assert dataset.card_markdown == again.card_markdown


def test_matches_the_committed_benchmark(dataset) -> None:
    train_path = resources.path("bench:train")
    test_path = resources.path("bench:test")
    manifest_path = train_path.parent / "manifest.json"
    assert dataset.train_jsonl == train_path.read_text(encoding="utf-8")
    assert dataset.test_jsonl == test_path.read_text(encoding="utf-8")
    assert dataset.manifest_json == manifest_path.read_text(encoding="utf-8")


def test_manifest_sha256_matches(dataset) -> None:
    manifest = json.loads(dataset.manifest_json)
    train_sha = hashlib.sha256(dataset.train_jsonl.encode("utf-8")).hexdigest()
    test_sha = hashlib.sha256(dataset.test_jsonl.encode("utf-8")).hexdigest()
    assert manifest["sha256"]["traces.train.jsonl"] == train_sha
    assert manifest["sha256"]["traces.test.jsonl"] == test_sha


def test_all_cell_counts_match_the_design(dataset) -> None:
    manifest = json.loads(dataset.manifest_json)
    totals: dict[tuple[str, str, str], int] = {}
    for key, count in manifest["counts"].items():
        _split, domain, class_, kind, _evidence = key.split("/")
        cell = (domain, class_, kind)
        totals[cell] = totals.get(cell, 0) + count
    expected = {("genuine", k): n for k, n in GENUINE_KINDS} | {
        ("false", k): n for k, n in FALSE_KINDS
    }
    for domain in DOMAINS:
        for (class_, kind), n in expected.items():
            assert totals.get((domain, class_, kind)) == n, (domain, class_, kind)


def test_probe_and_structured_fractions_per_cell() -> None:
    plan = full_plan(SEED)
    for domain in DOMAINS:
        specs = plan[domain]
        for class_, kinds in (("genuine", GENUINE_KINDS), ("false", FALSE_KINDS)):
            for kind, n in kinds:
                cell = [s for s in specs if s.class_ == class_ and s.kind == kind]
                assert len(cell) == n
                expected_probe = round(n / 2) if kind == "not_persisted" else round(2 * n / 3)
                assert sum(s.probe for s in cell) == expected_probe
                assert sum(s.structured for s in cell) == round(0.8 * n)


def test_split_invariants(dataset) -> None:
    manifest = json.loads(dataset.manifest_json)
    for domain in DOMAINS:
        n_genuine_test = sum(
            v for k, v in manifest["counts"].items() if k.startswith(f"test/{domain}/genuine/")
        )
        n_false_test = sum(
            v for k, v in manifest["counts"].items() if k.startswith(f"test/{domain}/false/")
        )
        assert n_genuine_test == 24
        assert n_false_test == 16
        for kind, _n in FALSE_KINDS:
            count = sum(
                v
                for k, v in manifest["counts"].items()
                if k.startswith(f"test/{domain}/false/{kind}/")
            )
            assert count >= 2, (domain, kind)


def test_train_test_pools_are_disjoint() -> None:
    pools = build_pools(SEED)
    assert set(pools.names.train).isdisjoint(pools.names.test)
    assert set(pools.companies.train).isdisjoint(pools.companies.test)
    assert set(pools.titles.train).isdisjoint(pools.titles.test)
    assert set(pools.files.train).isdisjoint(pools.files.test)


def test_ids_carry_no_label(all_traces: list[Trace]) -> None:
    by_domain: dict[str, list[Trace]] = {}
    for t in all_traces:
        by_domain.setdefault(t.task.domain, []).append(t)
    for domain, traces in by_domain.items():
        ordered = sorted(traces, key=lambda t: int(t.trace_id.split("-")[1]))
        assert [t.trace_id for t in ordered] == [f"{domain}-{i:03d}" for i in range(1, 101)]
        quartile = len(ordered) // 4
        for q in range(4):
            chunk = ordered[q * quartile : (q + 1) * quartile]
            outcomes = {t.ground_truth.outcome for t in chunk if t.ground_truth}
            assert len(outcomes) > 1, f"{domain} quartile {q}: ids correlate with the label"


def test_steps_are_between_4_and_18(all_traces: list[Trace]) -> None:
    for t in all_traces:
        assert 4 <= len(t.steps) <= 18, t.trace_id
        assert t.steps[0].kind == "message"
        assert t.steps[0].role == "user"


def test_every_trace_has_a_success_claim(all_traces: list[Trace]) -> None:
    extractor = ClaimExtractor(list(builtin_packs().values()))
    for t in all_traces:
        view = resolve_claims(detector_view(t), extractor)
        assert success_claims(view.final_claim.claims), f"{t.trace_id}: no success claim"


def test_benign_and_injected_phrases_share_one_pool(all_traces: list[Trace]) -> None:
    pools = build_pools(SEED)
    phrase_pool = list(pools.reviewer_phrases)
    genuine_hits = 0
    false_hits = 0
    for t in all_traces:
        blob = json.dumps(t.model_dump(mode="json"))
        if any(p in blob for p in phrase_pool):
            assert t.ground_truth is not None
            if t.ground_truth.outcome == "success":
                genuine_hits += 1
            else:
                false_hits += 1
    assert genuine_hits == 12  # 4 benign genuine traces x 3 domains
    assert false_hits == 12  # 4 reviewer_injection traces x 3 domains


def test_hedged_template_rate_within_10_points_across_classes() -> None:
    plan = full_plan(SEED)
    genuine_flags = []
    false_flags = []
    for domain in DOMAINS:
        for s in plan[domain]:
            (genuine_flags if s.class_ == "genuine" else false_flags).append(s.hedged)
    genuine_rate = sum(genuine_flags) / len(genuine_flags) * 100
    false_rate = sum(false_flags) / len(false_flags) * 100
    assert abs(genuine_rate - false_rate) <= 10


def test_leakage_train_auroc_is_at_most_0_65(all_traces: list[Trace]) -> None:
    train = [t for t in all_traces if t.meta and t.meta.get("split") == "train"]
    texts = [t.final_claim.text or "" for t in train]
    labels = [0 if t.ground_truth and t.ground_truth.outcome == "success" else 1 for t in train]
    auroc = leakage_auroc(texts, labels)
    assert auroc <= 0.65, auroc


def test_card_reports_the_frozen_test_leakage_auroc(all_traces: list[Trace], dataset) -> None:
    """The generator freeze: the one-time test-split, final-
    message-only leakage AUROC, fitted on the full train split.
    """
    train = [t for t in all_traces if t.meta and t.meta.get("split") == "train"]
    test = [t for t in all_traces if t.meta and t.meta.get("split") == "test"]
    train_texts = [t.final_claim.text or "" for t in train]
    train_labels = [
        0 if t.ground_truth and t.ground_truth.outcome == "success" else 1 for t in train
    ]
    test_texts = [t.final_claim.text or "" for t in test]
    test_labels = [0 if t.ground_truth and t.ground_truth.outcome == "success" else 1 for t in test]
    auroc = leakage_auroc_train_test(train_texts, train_labels, test_texts, test_labels)
    assert auroc <= 0.65, auroc
    assert f"{auroc:.3f} AUROC** on test" in dataset.card_markdown


def test_card_reports_single_feature_auroc_for_every_classifier_feature(dataset) -> None:
    assert "Single-feature AUROC (train)" in dataset.card_markdown
    for spec in FEATURES:
        assert f"`{spec.name}`" in dataset.card_markdown.split("Single-feature AUROC")[1]


def test_step_count_does_not_strongly_predict_the_label(all_traces: list[Trace]) -> None:
    """Regression guard: a trace's raw step count must not, on its own,
    nearly determine the label. Before short false scenarios were padded
    with neutral narration, this was as high as 0.975 (booking).
    """
    train = [t for t in all_traces if t.meta and t.meta.get("split") == "train"]
    scores = np.array([-len(t.steps) for t in train], dtype=float)
    labels = np.array(
        [0 if t.ground_truth and t.ground_truth.outcome == "success" else 1 for t in train]
    )
    assert rank_auroc(scores, labels) <= 0.65


def test_syllable_entity_pools_avoid_real_dictionary_words() -> None:
    """Names and company roots are built from syllables; none of them may
    collide with a real, plausible dictionary word.
    """
    pools = build_pools(SEED)
    for full_name in list(pools.names.train) + list(pools.names.test):
        for part in full_name.split(" "):
            assert part.lower() not in BLOCKED_SYLLABLE_WORDS, full_name
    for company in list(pools.companies.train) + list(pools.companies.test):
        root = company.rsplit(" ", 1)[0]
        assert root.lower() not in BLOCKED_SYLLABLE_WORDS, company


def test_train_test_templates_are_disjoint() -> None:
    """`dataset validate`'s disjointness scan also covers instruction and
    final-message templates, not just entity pools.
    """
    pools = build_pools(SEED)
    for domain in DOMAINS:
        train_templates, test_templates = _domain_template_texts(domain, pools)
        assert set(train_templates).isdisjoint(test_templates)


def test_dataset_validate_exits_0_on_the_committed_set() -> None:
    result = runner.invoke(app, ["dataset", "validate"])
    assert result.exit_code == 0, result.output


def test_dataset_validate_exits_2_on_a_tampered_copy(tmp_path: Path, dataset) -> None:
    out = tmp_path / "tampered"
    out.mkdir()
    (out / "traces.train.jsonl").write_text(dataset.train_jsonl, encoding="utf-8")
    (out / "traces.test.jsonl").write_text(
        dataset.test_jsonl.replace('"outcome":"success"', '"outcome":"failure"', 1),
        encoding="utf-8",
    )
    (out / "manifest.json").write_text(dataset.manifest_json, encoding="utf-8")

    result = runner.invoke(app, ["dataset", "validate", str(out)])
    assert result.exit_code == 2


def test_dataset_generate_cli_writes_the_four_files(tmp_path: Path) -> None:
    out = tmp_path / "gen"
    result = runner.invoke(app, ["dataset", "generate", "--seed", str(SEED), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "traces.train.jsonl").exists()
    assert (out / "traces.test.jsonl").exists()
    assert (out / "manifest.json").exists()
    assert (out / "DATASET_CARD.md").exists()
    assert validate_dataset(out) == []
