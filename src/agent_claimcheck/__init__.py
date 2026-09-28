"""agent-claimcheck: check AI agent success claims against trace evidence."""

from agent_claimcheck import metrics
from agent_claimcheck.checker import Checker, CheckResult
from agent_claimcheck.detectors.base import Detector, DetectorOutput, register_detector
from agent_claimcheck.rules.engine import RulePack, load_rule_pack
from agent_claimcheck.schema import load_traces

__version__ = "0.1.1"

__all__ = [
    "CheckResult",
    "Checker",
    "Detector",
    "DetectorOutput",
    "RulePack",
    "__version__",
    "load_rule_pack",
    "load_traces",
    "metrics",
    "register_detector",
]
