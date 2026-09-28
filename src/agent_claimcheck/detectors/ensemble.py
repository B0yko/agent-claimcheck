"""The two ensembles: `cascade-offline` (rules, then classifier) and
`cascade` (rules, then judge).

Both compose their own components' calibration: unlike a single detector,
whose calibrated `p_success` a `Checker` fills in from its `CalibratorSet`
after `score()` returns, an ensemble calibrates each component it actually
runs *before* deciding, because the decision (`decided_by`) and the raw/
calibrated `p_success` it reports both belong to whichever component
decided, not to the ensemble as a whole. A `Checker` recognises this via
`is_ensemble = True` and skips its own calibration step for these outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_claimcheck.detectors.base import Detector, DetectorOutput, register_detector
from agent_claimcheck.redact import DetectorView
from agent_claimcheck.rules.engine import RULE_SCORES

if TYPE_CHECKING:
    from agent_claimcheck.calibration import CalibratorSet

#: Rules raw scores that settle a claim conclusively: the ensemble takes the
#: rules output as-is instead of falling through to the next component.
_CONCLUSIVE_RAW_SCORES = frozenset(
    RULE_SCORES[outcome] for outcome in ("contradicted", "unsupported", "probe_supported")
)


def _rules_conclusive(rules_output: DetectorOutput) -> bool:
    return not rules_output.abstain and rules_output.p_success in _CONCLUSIVE_RAW_SCORES


def _calibrate(output: DetectorOutput, calibrators: CalibratorSet | None) -> DetectorOutput:
    if calibrators is None:
        return output
    p_calibrated = calibrators.apply(output.detector, output)
    if p_calibrated is None:
        return output
    return output.model_copy(update={"p_calibrated": p_calibrated})


class CascadeOfflineDetector:
    """Rules decide when conclusive; otherwise the classifier decides."""

    name = "cascade-offline"
    is_ensemble = True

    def __init__(
        self,
        rules: Detector,
        classifier: Detector,
        *,
        calibrators: CalibratorSet | None = None,
    ) -> None:
        self._rules = rules
        self._classifier = classifier
        self._calibrators = calibrators

    def score(self, trace: DetectorView) -> DetectorOutput:
        rules_output = _calibrate(self._rules.score(trace), self._calibrators)

        if _rules_conclusive(rules_output):
            deciding, decided_by, components = rules_output, "rules", [rules_output]
        else:
            classifier_output = _calibrate(self._classifier.score(trace), self._calibrators)
            deciding = classifier_output
            decided_by = "classifier-lr"
            components = [rules_output, classifier_output]

        return DetectorOutput(
            detector=self.name,
            p_success=deciding.p_success,
            abstain=deciding.abstain,
            abstain_reason=deciding.abstain_reason,
            p_calibrated=deciding.p_calibrated,
            reasons=deciding.reasons,
            cost_usd=sum(c.cost_usd for c in components),
            latency_ms=sum(c.latency_ms for c in components),
            cached=False,
            components=components,
            details={"decided_by": decided_by},
        )


class CascadeDetector:
    """Rules decide when conclusive; otherwise the LLM judge decides.

    `concurrent = True`: the judge component may make a network call, so a
    `Checker` scores traces for this ensemble on its thread pool.
    """

    name = "cascade"
    is_ensemble = True
    concurrent = True

    def __init__(
        self,
        rules: Detector,
        judge: Detector,
        *,
        calibrators: CalibratorSet | None = None,
    ) -> None:
        self._rules = rules
        self._judge = judge
        self._calibrators = calibrators

    def score(self, trace: DetectorView) -> DetectorOutput:
        rules_output = _calibrate(self._rules.score(trace), self._calibrators)

        if _rules_conclusive(rules_output):
            return DetectorOutput(
                detector=self.name,
                p_success=rules_output.p_success,
                abstain=rules_output.abstain,
                abstain_reason=rules_output.abstain_reason,
                p_calibrated=rules_output.p_calibrated,
                reasons=rules_output.reasons,
                cost_usd=rules_output.cost_usd,
                latency_ms=rules_output.latency_ms,
                cached=False,
                components=[rules_output],
                details={"decided_by": "rules", "sent_to_judge": False},
            )

        judge_output = _calibrate(self._judge.score(trace), self._calibrators)
        return DetectorOutput(
            detector=self.name,
            p_success=judge_output.p_success,
            abstain=judge_output.abstain,
            abstain_reason=judge_output.abstain_reason,
            p_calibrated=judge_output.p_calibrated,
            reasons=judge_output.reasons,
            cost_usd=rules_output.cost_usd + judge_output.cost_usd,
            latency_ms=rules_output.latency_ms + judge_output.latency_ms,
            cached=judge_output.cached,
            components=[rules_output, judge_output],
            details={"decided_by": "judge", "sent_to_judge": True},
        )


def _cascade_offline_factory(
    rules: Detector | None = None,
    classifier: Detector | None = None,
    calibrators: CalibratorSet | None = None,
    **_ignored: Any,
) -> Detector:
    from agent_claimcheck.detectors.classifier import ClassifierDetector, load_artifact
    from agent_claimcheck.detectors.rules import RulesDetector

    return CascadeOfflineDetector(
        rules or RulesDetector(),
        classifier or ClassifierDetector(load_artifact()),
        calibrators=calibrators,
    )


def _cascade_factory(
    rules: Detector,
    judge: Detector,
    calibrators: CalibratorSet | None = None,
    **_ignored: Any,
) -> Detector:
    return CascadeDetector(rules, judge, calibrators=calibrators)


register_detector("cascade-offline", _cascade_offline_factory)
register_detector("cascade", _cascade_factory)
