"""Small hand-made trace and step builders, shared by the tests.

Not the benchmark generator: these build a handful of steps directly, for
tests that need one specific shape of evidence.
"""

from __future__ import annotations

from typing import Any

from agent_claimcheck.schema import Claim, FinalClaim, Step, Task, Trace


def step(
    i: int,
    kind: str,
    role: str,
    *,
    name: str | None = None,
    args: dict[str, Any] | None = None,
    ok: bool | None = None,
    output: Any = None,
    error: str | dict[str, Any] | None = None,
    content: str | None = None,
    ts: str | None = None,
) -> Step:
    return Step(
        i=i,
        ts=ts or f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00",
        kind=kind,  # type: ignore[arg-type]
        role=role,  # type: ignore[arg-type]
        name=name,
        content=content,
        args=args,
        ok=ok,
        output=output,
        error=error,
    )


def message(i: int, role: str, content: str, *, ts: str | None = None) -> Step:
    return step(i, "message", role, content=content, ts=ts)


def tool_call(
    i: int, name: str, args: dict[str, Any] | None = None, *, ts: str | None = None
) -> Step:
    return step(i, "tool_call", "agent", name=name, args=args, ts=ts)


def tool_result(
    i: int,
    name: str,
    *,
    ok: bool = True,
    output: Any = None,
    error: str | dict[str, Any] | None = None,
    ts: str | None = None,
) -> Step:
    return step(i, "tool_result", "tool", name=name, ok=ok, output=output, error=error, ts=ts)


def probe(
    i: int,
    name: str,
    *,
    ok: bool = True,
    output: Any = None,
    error: str | dict[str, Any] | None = None,
    ts: str | None = None,
) -> Step:
    return step(
        i, "state_probe", "environment", name=name, ok=ok, output=output, error=error, ts=ts
    )


def trace(
    trace_id: str,
    domain: str,
    steps: list[Step],
    *,
    text: str | None = None,
    claims: list[tuple[str, dict[str, Any]]] | None = None,
    instruction: str = "do the task",
) -> Trace:
    return Trace(
        schema="agent-trace/v1",
        trace_id=trace_id,
        source="test/0.0.1",
        task=Task(id=f"{trace_id}-task", domain=domain, instruction=instruction),  # type: ignore[arg-type]
        steps=steps,
        final_claim=FinalClaim(
            text=text,
            claims=[Claim(type=t, subject=s) for t, s in (claims or [])],
        ),
    )
