"""The shared decision gate: one pure function per detector output.

The gate is deliberately plain, dependency-free code, identical for every
detector. A model estimates a probability; this module decides whether that
estimate is actable. Kept separate from any single detector so the
comparison between them stays fair.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["verified", "false_success", "unverifiable"]


@dataclass(frozen=True)
class Thresholds:
    """Gate thresholds. Defaults are fixed before any evaluation (ADR 0004)."""

    verified: float = 0.80
    false_success: float = 0.20

    def __post_init__(self) -> None:
        if not (0 <= self.false_success < self.verified <= 1):
            raise ValueError(
                "thresholds must satisfy 0 <= false_success < verified <= 1, got "
                f"false_success={self.false_success}, verified={self.verified}"
            )


DEFAULT_THRESHOLDS = Thresholds()


def gate(p_success: float, abstain: bool, t: Thresholds = DEFAULT_THRESHOLDS) -> Verdict:
    """Turn a probability and an abstain flag into a verdict. Pure."""
    if abstain:
        return "unverifiable"
    if p_success >= t.verified:
        return "verified"
    if p_success <= t.false_success:
        return "false_success"
    return "unverifiable"


def confidence(verdict: Verdict, p_success: float) -> float | None:
    """How confident the gate's own verdict is, or `None` when it abstained."""
    if verdict == "verified":
        return p_success
    if verdict == "false_success":
        return 1 - p_success
    return None


class UnknownPriceError(KeyError):
    """Raised when a model has no known price. Never silently free."""


def price_for(model: str, table: Mapping[str, Mapping[str, float]]) -> Mapping[str, float]:
    """Look up a model's price row in `table`, raising for an unknown model."""
    try:
        return table[model]
    except KeyError as exc:
        raise UnknownPriceError(f"no price configured for model {model!r}") from exc


def cost_usd(
    price_in_per_m: float, price_out_per_m: float, tokens_in: int, tokens_out: int
) -> float:
    """Cost in USD for a given token count at the given per-million prices."""
    return (tokens_in * price_in_per_m + tokens_out * price_out_per_m) / 1_000_000
