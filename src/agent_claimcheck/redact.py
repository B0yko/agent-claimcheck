"""The one projection detectors are allowed to see.

`ground_truth`, `meta` and `source` never reach a detector. Everything that
scores a trace, from the rule engine to the LLM judge, must be built on top
of `DetectorView`, never on `Trace` directly, so a label can never leak into
a score.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_claimcheck.schema import FinalClaim, Step, Task, Trace


class DetectorView(BaseModel):
    """Exactly what a detector may look at: no ground truth, no metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    task: Task
    steps: list[Step]
    final_claim: FinalClaim


def detector_view(trace: Trace) -> DetectorView:
    """Project a trace down to what detectors are allowed to see.

    `trace_id` is kept for bookkeeping (logging, joining results back to
    traces); judge prompts must never render it.
    """
    return DetectorView(
        trace_id=trace.trace_id,
        task=trace.task,
        steps=trace.steps,
        final_claim=trace.final_claim,
    )


def resolve_claims(view: DetectorView, extractor: Any = None) -> DetectorView:
    """Return a view whose `final_claim.claims` are resolved claims.

    Claim resolution (structured claims from `final_claim.claims`, falling
    back to pattern extraction over `final_claim.text` when that list is
    empty) is added by the claims module. Until then this returns the view
    unchanged.
    """
    del extractor
    return view
