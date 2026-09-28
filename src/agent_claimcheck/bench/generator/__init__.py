"""The benchmark generator: `generate(seed) -> GeneratedDataset`.

Order of work: plan all 300 cells, assign each a split,
render each trace using only its split's entity/template pools, shuffle
within each domain, assign `<domain>-<nnn>` ids, then write.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from agent_claimcheck.bench.generator import booking, coding, crm
from agent_claimcheck.bench.generator.card import render_card
from agent_claimcheck.bench.generator.common import sub_rng
from agent_claimcheck.bench.generator.leakage import leakage_auroc
from agent_claimcheck.bench.generator.plan import DOMAINS, TraceSpec, full_plan
from agent_claimcheck.bench.generator.pools import build_pools
from agent_claimcheck.schema import Trace, dump_trace

GENERATOR_VERSION = "claimcheck-bench/1"

_RENDERERS: dict[str, Callable[..., Trace]] = {
    "booking": booking.render,
    "crm": crm.render,
    "coding": coding.render,
}


@dataclass(frozen=True)
class GeneratedDataset:
    train_jsonl: str
    test_jsonl: str
    manifest_json: str
    card_markdown: str


def _assign_ids(
    seed: int, domain: str, rendered: list[tuple[TraceSpec, Trace]]
) -> list[tuple[TraceSpec, Trace]]:
    rng = sub_rng(seed, domain, "shuffle")
    order = list(range(len(rendered)))
    rng.shuffle(order)
    out = []
    for position, idx in enumerate(order, start=1):
        spec, trace = rendered[idx]
        new_id = f"{domain}-{position:03d}"
        new_trace = trace.model_copy(
            update={
                "trace_id": new_id,
                "task": trace.task.model_copy(update={"id": f"{new_id}-task"}),
            }
        )
        out.append((spec, new_trace))
    return out


def generate(seed: int) -> GeneratedDataset:
    pools = build_pools(seed)
    plan = full_plan(seed)

    train_traces: list[Trace] = []
    test_traces: list[Trace] = []
    manifest_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    false_examples: dict[str, str] = {}
    trace_offset = 0

    for domain in DOMAINS:
        specs = plan[domain]
        rendered: list[tuple[TraceSpec, Trace]] = []
        for spec in specs:
            trace = _RENDERERS[domain](seed, spec, pools, trace_offset)
            trace_offset += 1
            rendered.append((spec, trace))

        for spec, trace in _assign_ids(seed, domain, rendered):
            evidence = "state_probe" if spec.probe else "receipt_only"
            manifest_counts[(spec.split, domain, spec.class_, spec.kind, evidence)] += 1
            if spec.class_ == "false" and spec.kind not in false_examples:
                false_examples[spec.kind] = trace.final_claim.text or ""
            if spec.split == "train":
                train_traces.append(trace)
            else:
                test_traces.append(trace)

    train_traces.sort(key=lambda t: t.trace_id)
    test_traces.sort(key=lambda t: t.trace_id)

    train_jsonl = "".join(dump_trace(t) + "\n" for t in train_traces)
    test_jsonl = "".join(dump_trace(t) + "\n" for t in test_traces)

    train_sha = hashlib.sha256(train_jsonl.encode("utf-8")).hexdigest()
    test_sha = hashlib.sha256(test_jsonl.encode("utf-8")).hexdigest()

    counts = {"/".join(cell): count for cell, count in sorted(manifest_counts.items())}
    manifest = {
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "counts": counts,
        "sha256": {"traces.train.jsonl": train_sha, "traces.test.jsonl": test_sha},
    }
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    train_texts = [t.final_claim.text or "" for t in train_traces]
    train_labels = [
        0 if t.ground_truth is not None and t.ground_truth.outcome == "success" else 1
        for t in train_traces
    ]
    auroc = leakage_auroc(train_texts, train_labels)

    card_markdown = render_card(
        seed=seed, manifest=manifest, leakage_train_auroc=auroc, examples=false_examples
    )

    return GeneratedDataset(
        train_jsonl=train_jsonl,
        test_jsonl=test_jsonl,
        manifest_json=manifest_json,
        card_markdown=card_markdown,
    )
