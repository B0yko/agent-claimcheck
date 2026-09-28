"""Shared plumbing for the benchmark generator: RNG streams, the clock, and
step/trace builders. Nothing here is domain-specific.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from agent_claimcheck.schema import Claim, FinalClaim, GroundTruth, Step, Task, Trace

GENERATOR_VERSION = "claimcheck-bench/1"
SOURCE = "agent-claimcheck/0.1.0"

#: Fixed epoch every generated trace's step timestamps are offset from.
BASE_DATE = datetime(2026, 3, 2, 9, 0, 0, tzinfo=UTC)

Split = Literal["train", "test"]


def sub_rng(seed: int, *parts: str) -> random.Random:
    """A reproducible sub-stream RNG, derived from `seed` and `parts`."""
    return random.Random(f"{seed}:{':'.join(parts)}")


class Clock:
    """Deterministic, strictly increasing RFC 3339 timestamps for one trace."""

    def __init__(self, trace_offset: int) -> None:
        self._start = BASE_DATE + timedelta(seconds=trace_offset * 600)

    def at(self, step_i: int) -> str:
        return (self._start + timedelta(seconds=step_i)).isoformat()


def message(i: int, role: str, content: str, ts: str) -> Step:
    return Step(i=i, ts=ts, kind="message", role=role, content=content)  # type: ignore[arg-type]


def tool_call(i: int, name: str, args: dict[str, Any], ts: str) -> Step:
    return Step(i=i, ts=ts, kind="tool_call", role="agent", name=name, args=args)


def tool_result(
    i: int,
    name: str,
    ts: str,
    *,
    ok: bool,
    output: Any,
    error: str | None = None,
) -> Step:
    return Step(
        i=i, ts=ts, kind="tool_result", role="tool", name=name, ok=ok, output=output, error=error
    )


def state_probe(
    i: int,
    name: str,
    ts: str,
    *,
    ok: bool,
    output: Any,
    error: str | None = None,
) -> Step:
    return Step(
        i=i,
        ts=ts,
        kind="state_probe",
        role="environment",
        name=name,
        ok=ok,
        output=output,
        error=error,
    )


def hex_id(rng: random.Random, prefix: str, n: int = 8) -> str:
    digits = "0123456789abcdef"
    return prefix + "".join(rng.choice(digits) for _ in range(n))


def rfc3339(date: str, time: str, offset: str) -> str:
    """`date` `YYYY-MM-DD`, `time` `HH:MM`, `offset` e.g. `+02:00`."""
    return f"{date}T{time}:00{offset}"


def add_minutes(ts: str, minutes: int) -> str:
    dt = datetime.fromisoformat(ts) + timedelta(minutes=minutes)
    return dt.isoformat()


def add_days(date: str, days: int) -> str:
    dt = datetime.fromisoformat(date) + timedelta(days=days)
    return dt.date().isoformat()


class StepBuilder:
    """Accumulates one trace's steps in order, tracking `i` and the clock."""

    def __init__(self, trace_offset: int) -> None:
        self._clock = Clock(trace_offset)
        self._i = 0
        self.steps: list[Step] = []

    def message(self, role: str, content: str) -> None:
        self.steps.append(message(self._i, role, content, self._clock.at(self._i)))
        self._i += 1

    def call(self, name: str, args: dict[str, Any]) -> None:
        self.steps.append(tool_call(self._i, name, args, self._clock.at(self._i)))
        self._i += 1

    def result(self, name: str, *, ok: bool, output: Any, error: str | None = None) -> None:
        self.steps.append(
            tool_result(self._i, name, self._clock.at(self._i), ok=ok, output=output, error=error)
        )
        self._i += 1

    def probe(self, name: str, *, ok: bool, output: Any, error: str | None = None) -> None:
        self.steps.append(
            state_probe(self._i, name, self._clock.at(self._i), ok=ok, output=output, error=error)
        )
        self._i += 1


def build_trace(
    *,
    trace_id: str,
    domain: str,
    instruction: str,
    steps: list[Step],
    final_text: str,
    final_claims: list[tuple[str, dict[str, Any]]],
    outcome: Literal["success", "failure"],
    injection: str,
    variant: str,
    evidence: Literal["state_probe", "receipt_only"],
    split: Split,
) -> Trace:
    """Assemble one `Trace` with `ground_truth`/`meta` set."""
    return Trace(
        schema="agent-trace/v1",
        trace_id=trace_id,
        source=SOURCE,
        task=Task(id=f"{trace_id}-task", domain=domain, instruction=instruction),  # type: ignore[arg-type]
        steps=steps,
        final_claim=FinalClaim(
            text=final_text,
            claims=[Claim(type=t, subject=s) for t, s in final_claims],
        ),
        ground_truth=GroundTruth(
            outcome=outcome,
            checked_by="state_probe",
            details={"injection": injection, "evidence": evidence, "variant": variant},
        ),
        meta={"generator": GENERATOR_VERSION, "split": split},
    )
