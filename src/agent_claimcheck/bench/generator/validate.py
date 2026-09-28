"""`agent-claimcheck dataset validate`: checks a generated benchmark directory
against its own manifest and against the fixed design.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent_claimcheck import resources
from agent_claimcheck.bench.generator.leakage import leakage_auroc
from agent_claimcheck.bench.generator.plan import DOMAINS, FALSE_KINDS, GENUINE_KINDS
from agent_claimcheck.bench.generator.pools import build_pools
from agent_claimcheck.claims import ClaimExtractor, success_claims
from agent_claimcheck.redact import detector_view, resolve_claims
from agent_claimcheck.rules.engine import builtin_packs
from agent_claimcheck.schema import Trace, iter_trace_lines

_DESIGN_TOTALS: dict[tuple[str, str], int] = {("genuine", kind): n for kind, n in GENUINE_KINDS} | {
    ("false", kind): n for kind, n in FALSE_KINDS
}

_LEAKAGE_THRESHOLD = 0.65


def _resolve_paths(dir_path: str | Path | None) -> tuple[Path, Path, Path]:
    if dir_path is None:
        train_path = resources.path("bench:train")
        test_path = resources.path("bench:test")
        manifest_path = train_path.parent / "manifest.json"
    else:
        base = Path(dir_path)
        train_path = base / "traces.train.jsonl"
        test_path = base / "traces.test.jsonl"
        manifest_path = base / "manifest.json"
    return train_path, test_path, manifest_path


def _load_traces(path: Path, label: str, failures: list[str]) -> list[Trace] | None:
    traces: list[Trace] = []
    ok = True
    for result in iter_trace_lines(path):
        if result.errors:
            ok = False
            for err in result.errors:
                failures.append(f"{label}:{err.line_no}: {err.json_path}: {err.message}")
        elif result.trace is not None:
            traces.append(result.trace)
    return traces if ok else None


def _cell(t: Trace) -> tuple[str, str, str, str, str] | None:
    if t.ground_truth is None or t.meta is None:
        return None
    split = t.meta.get("split")
    injection = t.ground_truth.details.get("injection", "none")
    class_ = "genuine" if t.ground_truth.outcome == "success" else "false"
    kind = injection if class_ == "false" else _outcome_kind(t)
    evidence = t.ground_truth.details.get("evidence")
    if not isinstance(split, str) or not isinstance(kind, str) or not isinstance(evidence, str):
        return None
    return (split, t.task.domain, class_, kind, evidence)


def _outcome_kind(t: Trace) -> str:
    assert t.ground_truth is not None
    variant = t.ground_truth.details.get("variant")
    return variant if isinstance(variant, str) else ""


def validate_dataset(dir_path: str | Path | None = None) -> list[str]:
    """Validate a generated benchmark directory. Returns a list of failure
    messages; an empty list means the dataset is valid.
    """
    failures: list[str] = []
    train_path, test_path, manifest_path = _resolve_paths(dir_path)

    if not manifest_path.exists():
        return [f"missing manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_traces = _load_traces(train_path, "train", failures)
    test_traces = _load_traces(test_path, "test", failures)
    if train_traces is None or test_traces is None:
        return failures

    train_bytes = train_path.read_bytes()
    test_bytes = test_path.read_bytes()
    train_sha = hashlib.sha256(train_bytes).hexdigest()
    test_sha = hashlib.sha256(test_bytes).hexdigest()
    expected = manifest.get("sha256", {})
    expected_train_sha = expected.get("traces.train.jsonl")
    if expected_train_sha != train_sha:
        failures.append(
            f"sha256 mismatch: traces.train.jsonl (expected {expected_train_sha}, got {train_sha})"
        )
    expected_test_sha = expected.get("traces.test.jsonl")
    if expected_test_sha != test_sha:
        failures.append(
            f"sha256 mismatch: traces.test.jsonl (expected {expected_test_sha}, got {test_sha})"
        )

    computed_counts: dict[str, int] = {}
    for t in train_traces + test_traces:
        cell = _cell(t)
        if cell is None:
            failures.append(
                f"{t.trace_id}: missing or malformed ground_truth/meta for cell bucketing"
            )
            continue
        key = "/".join(cell)
        computed_counts[key] = computed_counts.get(key, 0) + 1

    manifest_counts = manifest.get("counts", {})
    if computed_counts != manifest_counts:
        only_computed = set(computed_counts) - set(manifest_counts)
        only_manifest = set(manifest_counts) - set(computed_counts)
        differing = {
            k
            for k in set(computed_counts) & set(manifest_counts)
            if computed_counts[k] != manifest_counts[k]
        }
        for k in sorted(only_computed | only_manifest | differing):
            failures.append(
                f"count mismatch for cell {k!r}: computed={computed_counts.get(k)} "
                f"manifest={manifest_counts.get(k)}"
            )

    design_totals: dict[tuple[str, str, str], int] = {}
    for key, count in computed_counts.items():
        _split, domain, class_, kind, _evidence = key.split("/")
        cell_key = (domain, class_, kind)
        design_totals[cell_key] = design_totals.get(cell_key, 0) + count
    for domain in DOMAINS:
        for (class_, kind), expected_n in _DESIGN_TOTALS.items():
            actual_n = design_totals.get((domain, class_, kind), 0)
            if actual_n != expected_n:
                failures.append(
                    f"design count mismatch: {domain}/{class_}/{kind} "
                    f"expected {expected_n}, got {actual_n}"
                )

    for domain in DOMAINS:
        domain_train = [t for t in train_traces if t.task.domain == domain]
        domain_test = [t for t in test_traces if t.task.domain == domain]
        if len(domain_test) != 40:
            failures.append(f"{domain}: expected 40 test traces, got {len(domain_test)}")
        if len(domain_train) != 60:
            failures.append(f"{domain}: expected 60 train traces, got {len(domain_train)}")
        n_genuine_test = sum(
            1 for t in domain_test if t.ground_truth and t.ground_truth.outcome == "success"
        )
        n_false_test = len(domain_test) - n_genuine_test
        if n_genuine_test != 24:
            failures.append(f"{domain}: expected 24 genuine test traces, got {n_genuine_test}")
        if n_false_test != 16:
            failures.append(f"{domain}: expected 16 false test traces, got {n_false_test}")
        false_kind_test_counts: dict[str, int] = {}
        for t in domain_test:
            if t.ground_truth and t.ground_truth.outcome == "failure":
                kind = t.ground_truth.details.get("injection", "?")
                false_kind_test_counts[kind] = false_kind_test_counts.get(kind, 0) + 1
        for kind, _n in FALSE_KINDS:
            if false_kind_test_counts.get(kind, 0) < 2:
                failures.append(
                    f"{domain}: false kind {kind!r} has fewer than 2 test traces "
                    f"({false_kind_test_counts.get(kind, 0)})"
                )

    packs = list(builtin_packs().values())
    extractor = ClaimExtractor(packs)
    n_test_checked = 0
    for t in train_traces:
        view = resolve_claims(detector_view(t), extractor)
        if not success_claims(view.final_claim.claims):
            failures.append(f"{t.trace_id}: no success claim (skipped)")
    for t in test_traces:
        view = resolve_claims(detector_view(t), extractor)
        if not success_claims(view.final_claim.claims):
            failures.append(f"{t.trace_id}: no success claim (skipped)")
        else:
            n_test_checked += 1
    if n_test_checked != len(test_traces):
        failures.append(f"expected all {len(test_traces)} test traces to have a success claim")

    seed = manifest.get("seed")
    if isinstance(seed, int):
        pools = build_pools(seed)
        pool_lists = {
            "booking": (pools.names, pools.companies, pools.titles),
            "crm": (pools.names, pools.companies, pools.titles),
            "coding": (pools.files, pools.titles),
        }
        for domain in DOMAINS:
            train_only: set[str] = set()
            test_only: set[str] = set()
            for pool in pool_lists[domain]:
                train_only.update(pool.train)
                test_only.update(pool.test)
            for t in test_traces:
                if t.task.domain != domain:
                    continue
                blob = json.dumps(t.model_dump(mode="json"))
                for entity in train_only:
                    if entity in blob:
                        failures.append(
                            f"{t.trace_id}: test trace contains a train-only entity {entity!r}"
                        )
            for t in train_traces:
                if t.task.domain != domain:
                    continue
                blob = json.dumps(t.model_dump(mode="json"))
                for entity in test_only:
                    if entity in blob:
                        failures.append(
                            f"{t.trace_id}: train trace contains a test-only entity {entity!r}"
                        )

    train_texts = [t.final_claim.text or "" for t in train_traces]
    train_labels = [
        0 if t.ground_truth is not None and t.ground_truth.outcome == "success" else 1
        for t in train_traces
    ]
    auroc = leakage_auroc(train_texts, train_labels)
    if auroc > _LEAKAGE_THRESHOLD:
        failures.append(f"leakage audit: train AUROC {auroc:.3f} exceeds {_LEAKAGE_THRESHOLD}")

    return failures
