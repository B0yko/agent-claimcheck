"""Cell planning: how many traces of each shape, and which split each one
lands in, decided BEFORE any trace is rendered.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from agent_claimcheck.bench.generator.common import Split, sub_rng

DOMAINS: tuple[str, ...] = ("booking", "crm", "coding")

GENUINE_KINDS: tuple[tuple[str, int], ...] = (("clean", 25), ("recovered", 20), ("noisy", 15))
_GENUINE_TEST_COUNT: dict[str, int] = {"clean": 10, "recovered": 8, "noisy": 6}

FALSE_CANONICAL_KINDS: tuple[str, ...] = (
    "phantom_action",
    "error_ignored",
    "wrong_target",
    "wrong_value",
    "not_persisted",
    "partial_completion",
)
FALSE_KINDS: tuple[tuple[str, int], ...] = tuple((k, 6) for k in FALSE_CANONICAL_KINDS) + (
    ("reviewer_injection", 4),
)

#: Underlying failure kinds a `reviewer_injection` instance can wrap.
REVIEWER_INJECTION_UNDERLYING: tuple[str, ...] = (
    "error_ignored",
    "phantom_action",
    "not_persisted",
    "error_ignored",
)


@dataclass(frozen=True)
class TraceSpec:
    """One planned trace: its cell membership and cross-cutting flags."""

    domain: str
    class_: str  # "genuine" | "false"
    kind: str
    cell_index: int  # index within (domain, kind), 0-based
    split: Split
    probe: bool
    structured: bool
    hedged: bool
    reviewer_phrase: bool
    wrong_variant: str | None = None  # "restated" | "repeated", wrong_target/wrong_value only
    underlying: str | None = None  # reviewer_injection only


def _bumped_kinds(seed: int, domain_index: int) -> frozenset[str]:
    """The 2 (of 6) canonical false kinds whose test count is bumped to 3 for
    this domain, via a seeded rotation so every kind is bumped in exactly one
    of the 3 domains.
    """
    rng = sub_rng(seed, "split-rotation")
    order = list(FALSE_CANONICAL_KINDS)
    rng.shuffle(order)
    start = (2 * domain_index) % len(order)
    return frozenset({order[start], order[(start + 1) % len(order)]})


def _pick(rng_key: tuple[str, ...], seed: int, n: int, k: int) -> frozenset[int]:
    rng = sub_rng(seed, *rng_key)
    return frozenset(rng.sample(range(n), k))


def _cell_specs(
    seed: int,
    domain: str,
    domain_index: int,
    class_: str,
    kind: str,
    n: int,
    test_count: int,
) -> list[TraceSpec]:
    probe_count = n if kind == "not_persisted" else round(2 * n / 3)
    structured_count = round(0.8 * n)
    hedge_count = round(0.25 * n)

    test_idx = _pick((domain, class_, kind, "split"), seed, n, test_count)
    probe_idx = _pick((domain, class_, kind, "probe"), seed, n, probe_count)
    structured_idx = _pick((domain, class_, kind, "structured"), seed, n, structured_count)
    hedge_idx = _pick((domain, class_, kind, "hedge"), seed, n, hedge_count)

    wrong_variant_by_idx: dict[int, str] = {}
    if kind in ("wrong_target", "wrong_value"):
        restated_idx = _pick((domain, class_, kind, "variant"), seed, n, n // 2)
        for i in range(n):
            wrong_variant_by_idx[i] = "restated" if i in restated_idx else "repeated"

    underlying_by_idx: dict[int, str] = {}
    if kind == "reviewer_injection":
        rng = sub_rng(seed, domain, "reviewer_injection", "underlying")
        pool = list(REVIEWER_INJECTION_UNDERLYING)
        rng.shuffle(pool)
        for i in range(n):
            underlying_by_idx[i] = pool[i % len(pool)]

    specs = []
    for i in range(n):
        specs.append(
            TraceSpec(
                domain=domain,
                class_=class_,
                kind=kind,
                cell_index=i,
                split="test" if i in test_idx else "train",
                probe=i in probe_idx,
                structured=i in structured_idx,
                hedged=i in hedge_idx,
                reviewer_phrase=False,
                wrong_variant=wrong_variant_by_idx.get(i),
                underlying=underlying_by_idx.get(i),
            )
        )
    return specs


#: How many of the 60 genuine traces per domain carry a benign reviewer-note
#: phrase, spread across variants: 2 clean, 1 recovered, 1 noisy.
_BENIGN_PER_VARIANT: dict[str, int] = {"clean": 2, "recovered": 1, "noisy": 1}


def domain_plan(seed: int, domain: str, domain_index: int) -> list[TraceSpec]:
    """The 100 `TraceSpec`s for one domain (60 genuine + 40 false)."""
    specs: list[TraceSpec] = []

    for kind, n in GENUINE_KINDS:
        cell = _cell_specs(
            seed, domain, domain_index, "genuine", kind, n, _GENUINE_TEST_COUNT[kind]
        )
        benign_n = _BENIGN_PER_VARIANT[kind]
        benign_idx = _pick((domain, "genuine", kind, "benign"), seed, n, benign_n)
        cell = [replace(s, reviewer_phrase=s.cell_index in benign_idx) for s in cell]
        specs.extend(cell)

    bumped = _bumped_kinds(seed, domain_index)
    for kind, n in FALSE_KINDS:
        test_count = 2 if kind == "reviewer_injection" else (3 if kind in bumped else 2)
        cell = _cell_specs(seed, domain, domain_index, "false", kind, n, test_count)
        if kind == "reviewer_injection":
            cell = [replace(s, reviewer_phrase=True) for s in cell]
        specs.extend(cell)

    return specs


def full_plan(seed: int) -> dict[str, list[TraceSpec]]:
    """All 300 planned traces, grouped by domain."""
    return {domain: domain_plan(seed, domain, idx) for idx, domain in enumerate(DOMAINS)}
