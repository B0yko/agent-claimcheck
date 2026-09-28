"""The rules detector: deterministic claim-evidence rules."""

from __future__ import annotations

from collections.abc import Sequence

from agent_claimcheck.claims import success_claims
from agent_claimcheck.config import DEFAULT_NON_SUCCESS_TYPES
from agent_claimcheck.detectors.base import DetectorOutput, Reason, register_detector
from agent_claimcheck.redact import DetectorView
from agent_claimcheck.rules.engine import OUTCOME_ORDER, RULE_SCORES, RulePack, evaluate_claim

#: Reported as `p_success` when the rules detector abstains (no applicable
#: rule pack defines any of the trace's success claims).
DEFAULT_BASE_RATE = 0.6


class RulesDetector:
    """Scores a trace by evaluating every success claim's rule outcome."""

    name = "rules"

    def __init__(
        self,
        packs: Sequence[RulePack] = (),
        non_success_types: Sequence[str] = DEFAULT_NON_SUCCESS_TYPES,
        base_rate: float = DEFAULT_BASE_RATE,
    ) -> None:
        self.packs = list(packs)
        self.non_success_types = tuple(non_success_types)
        self.base_rate = base_rate

    def score(self, trace: DetectorView) -> DetectorOutput:
        claims = success_claims(trace.final_claim.claims, self.non_success_types)
        outcomes = [
            evaluate_claim(
                c.subject,
                c.type,
                domain=trace.task.domain,
                steps=trace.steps,
                packs=self.packs,
            )
            for c in claims
        ]

        if not outcomes:
            return DetectorOutput(
                detector=self.name,
                p_success=self.base_rate,
                abstain=True,
                abstain_reason="no_rule",
                reasons=[
                    Reason(claim=None, outcome="unknown", step=None, detail="no success claims")
                ],
            )

        # OUTCOME_ORDER lists worst-first, so the worst outcome has the
        # smallest index.
        worst = min(outcomes, key=lambda o: OUTCOME_ORDER.index(o.outcome))
        reasons = [
            Reason(claim=o.claim_type, outcome=o.outcome, step=o.step, detail=o.detail[:200])
            for o in outcomes
        ]
        details = {"claims": [o.as_dict() for o in outcomes]}

        score = RULE_SCORES[worst.outcome]
        if score is None:
            return DetectorOutput(
                detector=self.name,
                p_success=self.base_rate,
                abstain=True,
                abstain_reason="no_rule",
                reasons=reasons,
                details=details,
            )
        return DetectorOutput(
            detector=self.name,
            p_success=score,
            abstain=False,
            reasons=reasons,
            details=details,
        )


register_detector("rules", RulesDetector)
