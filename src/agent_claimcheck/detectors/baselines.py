"""Baseline detectors: `trust-agent` and `any-error`.

Both are constant-ish rules with no learned or fitted part, so they never
abstain and are never calibrated (a `CalibratorSet` simply has no entry for
either key).
"""

from __future__ import annotations

from agent_claimcheck.detectors.base import DetectorOutput, Reason, register_detector
from agent_claimcheck.redact import DetectorView


class TrustAgentDetector:
    """Always trusts the agent's own claim: `p_success = 1.0`."""

    name = "trust-agent"

    def score(self, trace: DetectorView) -> DetectorOutput:
        return DetectorOutput(
            detector=self.name,
            p_success=1.0,
            abstain=False,
            reasons=[
                Reason(
                    claim=None,
                    outcome="trust-agent",
                    step=None,
                    detail="always trusts the agent's own final claim",
                )
            ],
        )


class AnyErrorDetector:
    """`p_success = 0.0` when any `tool_result` step failed, else `1.0`."""

    name = "any-error"

    def score(self, trace: DetectorView) -> DetectorOutput:
        failed = next(
            (s for s in trace.steps if s.kind == "tool_result" and (s.ok is False or s.error)),
            None,
        )
        if failed is not None:
            return DetectorOutput(
                detector=self.name,
                p_success=0.0,
                abstain=False,
                reasons=[
                    Reason(
                        claim=None,
                        outcome="any-error",
                        step=failed.i,
                        detail="at least one tool_result step failed",
                    )
                ],
            )
        return DetectorOutput(
            detector=self.name,
            p_success=1.0,
            abstain=False,
            reasons=[
                Reason(
                    claim=None,
                    outcome="any-error",
                    step=None,
                    detail="no tool_result step failed",
                )
            ],
        )


register_detector("trust-agent", TrustAgentDetector)
register_detector("any-error", AnyErrorDetector)
