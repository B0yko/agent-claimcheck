"""The pure gate: thresholds, boundaries, abstain and cost accounting."""

from __future__ import annotations

import pytest

from agent_claimcheck.gate import (
    DEFAULT_THRESHOLDS,
    Thresholds,
    UnknownPriceError,
    confidence,
    cost_usd,
    gate,
    price_for,
)


def test_default_thresholds_are_080_020() -> None:
    assert DEFAULT_THRESHOLDS.verified == 0.80
    assert DEFAULT_THRESHOLDS.false_success == 0.20


def test_gate_verified_at_and_above_threshold() -> None:
    assert gate(0.80, False) == "verified"
    assert gate(0.95, False) == "verified"
    assert gate(1.0, False) == "verified"


def test_gate_false_success_at_and_below_threshold() -> None:
    assert gate(0.20, False) == "false_success"
    assert gate(0.05, False) == "false_success"
    assert gate(0.0, False) == "false_success"


def test_gate_unverifiable_in_between() -> None:
    assert gate(0.21, False) == "unverifiable"
    assert gate(0.5, False) == "unverifiable"
    assert gate(0.79, False) == "unverifiable"


def test_gate_abstain_always_unverifiable() -> None:
    assert gate(0.99, True) == "unverifiable"
    assert gate(0.01, True) == "unverifiable"


def test_gate_respects_custom_thresholds() -> None:
    t = Thresholds(verified=0.9, false_success=0.1)
    assert gate(0.85, False, t) == "unverifiable"
    assert gate(0.9, False, t) == "verified"
    assert gate(0.1, False, t) == "false_success"


def test_thresholds_validate_ordering() -> None:
    with pytest.raises(ValueError):
        Thresholds(verified=0.2, false_success=0.5)
    with pytest.raises(ValueError):
        Thresholds(verified=1.5, false_success=0.2)
    with pytest.raises(ValueError):
        Thresholds(verified=0.8, false_success=-0.1)


def test_confidence() -> None:
    assert confidence("verified", 0.9) == 0.9
    assert confidence("false_success", 0.1) == pytest.approx(0.9)
    assert confidence("unverifiable", 0.5) is None


def test_unknown_model_price_raises() -> None:
    # An unknown model is an error, never a silent price of 0.0.
    with pytest.raises(UnknownPriceError):
        price_for("no-such-model", {"known-model": {"in": 1.0, "out": 2.0}})


def test_known_model_price_is_returned() -> None:
    table = {"known-model": {"in": 1.0, "out": 2.0}}
    assert price_for("known-model", table) == {"in": 1.0, "out": 2.0}


def test_cost_usd() -> None:
    assert cost_usd(1.0, 2.0, 1_000_000, 500_000) == 2.0
    assert cost_usd(0.0, 0.0, 1_000, 1_000) == 0.0
