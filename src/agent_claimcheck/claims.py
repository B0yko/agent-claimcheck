"""Turning `final_claim` into resolved claims.

Structured claims (`final_claim.claims`) win when present. Otherwise a
`ClaimExtractor` runs pattern extraction over `final_claim.text`, using the
`claim_patterns` of whichever rule packs apply to the trace, falling
back to a generic `done` pattern only when nothing else matched.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_claimcheck.config import DEFAULT_NON_SUCCESS_TYPES
from agent_claimcheck.rules.engine import RulePack, pack_applies
from agent_claimcheck.schema import Step

#: Phrases that drop a match when they appear before it in the same
#: sentence; "not yet" also drops a match anywhere in the sentence.
_NEGATION_PHRASES: tuple[str, ...] = (
    "couldn't",
    "could not",
    "can't",
    "cannot",
    "unable to",
    "failed to",
    "not yet",
    "did not",
    "didn't",
    "wasn't able",
    "was not able",
    "haven't",
    "have not",
    "never",
    "no longer",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class ResolvedClaim(BaseModel):
    """One resolved success/non-success claim: structured or extracted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    subject: dict[str, Any] = Field(default_factory=dict)
    source: Literal["structured", "extracted"]
    span: tuple[int, int] | None = None


def success_claims(
    claims: Sequence[ResolvedClaim],
    non_success_types: Sequence[str] = DEFAULT_NON_SUCCESS_TYPES,
) -> list[ResolvedClaim]:
    """Claims whose type is not one of `non_success_types`."""
    excluded = set(non_success_types)
    return [c for c in claims if c.type not in excluded]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return spans


def _is_negated(match: re.Match[str], text: str, sentences: Sequence[tuple[int, int]]) -> bool:
    pos = match.start()
    sent_start, sent_end = next(((s, e) for s, e in sentences if s <= pos < e), (0, len(text)))
    sentence = text[sent_start:sent_end].lower()
    if "not yet" in sentence:
        return True
    before = text[sent_start:pos].lower()
    return any(phrase in before for phrase in _NEGATION_PHRASES if phrase != "not yet")


def _extract_with_packs(text: str, packs: Sequence[RulePack]) -> list[ResolvedClaim]:
    patterns_by_type: dict[str, list[re.Pattern[str]]] = {}
    for pack in packs:
        for claim_type, patterns in pack.claim_patterns.items():
            patterns_by_type.setdefault(claim_type, []).extend(patterns)

    sentences = _sentence_spans(text)
    claims: list[ResolvedClaim] = []
    for claim_type, type_patterns in patterns_by_type.items():
        matches: list[re.Match[str]] = []
        for pattern in type_patterns:
            matches.extend(pattern.finditer(text))
        matches.sort(key=lambda m: m.start())

        non_negated = [m for m in matches if not _is_negated(m, text, sentences)]
        if not non_negated:
            continue

        first = non_negated[0]
        subject = {k: v for k, v in first.groupdict().items() if v is not None}
        for later in non_negated[1:]:
            for key, value in later.groupdict().items():
                if value is not None and key not in subject:
                    subject[key] = value

        claims.append(
            ResolvedClaim(
                type=claim_type,
                subject=subject,
                source="extracted",
                span=(first.start(), first.end()),
            )
        )
    return claims


class ClaimExtractor:
    """Pattern-based claim extraction, applicability-gated.

    `packs` is normally `rules.engine.builtin_packs().values()` plus any
    user-supplied packs. A pack named `generic` is treated specially: its
    patterns are tried only as a fallback, when no applicable domain pack's
    patterns matched anything, regardless of its own applicability.
    """

    def __init__(
        self,
        packs: Sequence[RulePack],
        non_success_types: Sequence[str] = DEFAULT_NON_SUCCESS_TYPES,
    ) -> None:
        self.packs = list(packs)
        self.non_success_types = tuple(non_success_types)

    def extract(self, text: str, *, domain: str, steps: Sequence[Step]) -> list[ResolvedClaim]:
        """Extract claims from `text`, given the trace's domain and steps."""
        domain_packs = [p for p in self.packs if p.pack != "generic"]
        applicable = [p for p in domain_packs if pack_applies(p, domain, steps)]

        claims = _extract_with_packs(text, applicable)
        if claims:
            return claims

        generic_pack = next((p for p in self.packs if p.pack == "generic"), None)
        if generic_pack is None:
            return []
        return _extract_with_packs(text, [generic_pack])
