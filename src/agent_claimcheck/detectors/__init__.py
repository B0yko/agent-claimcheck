"""Built-in detectors register themselves with `base.register_detector` on
import.
"""

from __future__ import annotations

from agent_claimcheck.detectors import classifier as _classifier  # noqa: F401
from agent_claimcheck.detectors import rules as _rules  # noqa: F401

__all__: list[str] = []
