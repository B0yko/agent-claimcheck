"""The detector protocol every scorer implements, and its registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent_claimcheck.redact import DetectorView


class Reason(BaseModel):
    """One line of evidence behind a detector's score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: str | None
    outcome: str
    step: int | None
    detail: str = Field(max_length=200)


class DetectorOutput(BaseModel):
    """What every detector returns, before calibration and the gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector: str
    p_success: float
    abstain: bool
    abstain_reason: str | None = None
    p_calibrated: float | None = None
    reasons: list[Reason] = Field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    components: list[DetectorOutput] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class Detector(Protocol):
    """`score` must be thread-safe: judges run several traces concurrently."""

    name: str

    def score(self, trace: DetectorView) -> DetectorOutput: ...


_REGISTRY: dict[str, Callable[..., Detector]] = {}


def register_detector(name: str, factory: Callable[..., Detector]) -> None:
    """Register a detector factory under `name` (e.g. `"rules"`)."""
    _REGISTRY[name] = factory


def get_detector(name: str, **cfg: Any) -> Detector:
    """Build a registered detector by name."""
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"no detector registered as {name!r}") from exc
    return factory(**cfg)
