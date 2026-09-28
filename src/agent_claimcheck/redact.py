"""The one projection detectors are allowed to see.

`ground_truth`, `meta` and `source` never reach a detector. Everything that
scores a trace, from the rule engine to the LLM judge, must be built on top
of `DetectorView`, never on `Trace` directly, so a label can never leak into
a score.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent_claimcheck.claims import ClaimExtractor, ResolvedClaim
from agent_claimcheck.schema import Step, Task, Trace


class ViewFinalClaim(BaseModel):
    """`final_claim` as detectors see it: text plus resolved claims."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str | None
    claims: list[ResolvedClaim] = Field(default_factory=list)


class DetectorView(BaseModel):
    """Exactly what a detector may look at: no ground truth, no metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    task: Task
    steps: list[Step]
    final_claim: ViewFinalClaim
    claims_extracted: bool = False


def detector_view(trace: Trace) -> DetectorView:
    """Project a trace down to what detectors are allowed to see.

    `trace_id` is kept for bookkeeping (logging, joining results back to
    traces); judge prompts must never render it. Structured claims (when
    `final_claim.claims` is non-empty) are wrapped as resolved claims right
    away; otherwise `final_claim.claims` stays empty until `resolve_claims`
    runs pattern extraction.
    """
    claims = [
        ResolvedClaim(type=c.type, subject=c.subject, source="structured")
        for c in trace.final_claim.claims
    ]
    return DetectorView(
        trace_id=trace.trace_id,
        task=trace.task,
        steps=trace.steps,
        final_claim=ViewFinalClaim(text=trace.final_claim.text, claims=claims),
    )


def resolve_claims(view: DetectorView, extractor: ClaimExtractor | None = None) -> DetectorView:
    """Return a copy whose `final_claim.claims` are the resolved claims.

    Structured claims (already present on `view`) win and are returned
    unchanged. Otherwise, when an extractor is given, pattern extraction
    runs over `final_claim.text` and `view.claims_extracted` is set.
    """
    if view.final_claim.claims or extractor is None:
        return view
    extracted = extractor.extract(
        view.final_claim.text or "", domain=view.task.domain, steps=view.steps
    )
    new_final_claim = view.final_claim.model_copy(update={"claims": extracted})
    return view.model_copy(update={"final_claim": new_final_claim, "claims_extracted": True})
