"""The agent-trace/v1 schema: pydantic models, JSON Schema validation and I/O.

Traces are read and written as JSON Lines: one `agent-trace/v1` object per
line. Every line is validated against `schemas/agent-trace-v1.json` (Draft
2020-12) before it becomes a `Trace`, and every trace this project writes is
validated again before the bytes leave the process.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from agent_claimcheck import resources

Domain = Literal["booking", "crm", "coding", "browser", "other"]
StepKind = Literal["message", "tool_call", "tool_result", "state_probe"]
StepRole = Literal["user", "agent", "tool", "environment"]
Outcome = Literal["success", "failure", "unknown"]
CheckedBy = Literal["state_probe", "human", "none"]


class Task(BaseModel):
    """What the agent was asked to do."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    domain: Domain
    instruction: str


class Step(BaseModel):
    """One event in a trace: a message, a tool call, its result, or a probe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    i: int = Field(ge=0)
    ts: str
    kind: StepKind
    role: StepRole
    name: str | None = None
    content: str | None = None
    args: dict[str, Any] | None = None
    ok: bool | None = None
    output: Any = None
    error: str | dict[str, Any] | None = None


class Claim(BaseModel):
    """One thing the agent asserted it did."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    subject: dict[str, Any] = Field(default_factory=dict)


class FinalClaim(BaseModel):
    """The agent's final user-visible message, plus what it claims to have done."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str | None
    claims: list[Claim] = Field(default_factory=list)


class GroundTruth(BaseModel):
    """The label, when one exists. Never seen by detectors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Outcome
    checked_by: CheckedBy
    details: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """One full agent-trace/v1 record."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True, serialize_by_alias=True
    )

    schema_: Literal["agent-trace/v1"] = Field(alias="schema")
    trace_id: str
    source: str
    task: Task
    steps: list[Step]
    final_claim: FinalClaim
    ground_truth: GroundTruth | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class TraceError:
    """One validation error, tied to a line and a JSON path within it."""

    line_no: int
    json_path: str
    message: str


@dataclass(frozen=True)
class LineResult:
    """The outcome of validating one line: either a trace, or errors."""

    line_no: int
    trace: Trace | None
    errors: list[TraceError] = field(default_factory=list)


@dataclass
class LoadReport:
    """Everything `load_traces_report` learned about an input file."""

    traces: list[Trace]
    errors: list[TraceError]
    warnings: list[str]


class TraceValidationError(Exception):
    """Raised in strict mode, or on write, when a trace fails validation."""

    def __init__(self, errors: list[TraceError]) -> None:
        self.errors = errors
        preview = "; ".join(f"line {e.line_no}: {e.json_path}: {e.message}" for e in errors[:5])
        super().__init__(preview or "trace validation failed")


_FORMAT_CHECKER = FormatChecker()


@_FORMAT_CHECKER.checks("date-time", raises=ValueError)
def _check_date_time(value: object) -> bool:
    if not isinstance(value, str):
        return True
    datetime.fromisoformat(value)
    return True


_PROBE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["trace_id", "name", "ok", "ts"],
    "additionalProperties": False,
    "properties": {
        "trace_id": {"type": "string"},
        "name": {"type": "string"},
        "args": {"type": ["object", "null"]},
        "ok": {"type": ["boolean", "null"]},
        "output": {},
        "error": {"anyOf": [{"type": "string"}, {"type": "object"}, {"type": "null"}]},
        "ts": {"type": "string", "format": "date-time"},
    },
}


def _load_schema_dict() -> dict[str, Any]:
    schema_path = resources.path("schemas/agent-trace-v1.json")
    with schema_path.open("r", encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
        return loaded


def _trace_validator() -> Draft202012Validator:
    return Draft202012Validator(_load_schema_dict(), format_checker=_FORMAT_CHECKER)


def _json_path(err: JsonSchemaValidationError) -> str:
    parts = ["$"]
    for segment in err.absolute_path:
        if isinstance(segment, int):
            parts[-1] += f"[{segment}]"
        else:
            parts.append(f".{segment}")
    return "".join(parts)


def _semantic_errors(obj: Any, line_no: int, seen_trace_ids: dict[str, int]) -> list[TraceError]:
    errors: list[TraceError] = []
    if not isinstance(obj, dict):
        return errors

    trace_id = obj.get("trace_id")
    if isinstance(trace_id, str):
        first_line = seen_trace_ids.get(trace_id)
        if first_line is None:
            seen_trace_ids[trace_id] = line_no
        else:
            errors.append(
                TraceError(
                    line_no,
                    "$.trace_id",
                    f"duplicate trace_id {trace_id!r}, first seen on line {first_line}",
                )
            )

    steps = obj.get("steps")
    if isinstance(steps, list):
        previous_i: int | None = None
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            i = step.get("i")
            if not isinstance(i, int):
                continue
            if previous_i is not None and i <= previous_i:
                errors.append(
                    TraceError(line_no, f"$.steps[{idx}].i", "i must be strictly increasing")
                )
            previous_i = i
    return errors


def _resolve_input(path_or_alias: str | Path) -> Path:
    if isinstance(path_or_alias, str) and path_or_alias in resources.ALIASES:
        return resources.path(path_or_alias)
    return Path(path_or_alias)


def iter_trace_lines(path_or_alias: str | Path) -> Iterator[LineResult]:
    """Validate a JSON Lines trace file, one `LineResult` per non-blank line."""
    resolved = _resolve_input(path_or_alias)
    validator = _trace_validator()
    seen_trace_ids: dict[str, int] = {}

    with resolved.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw = raw_line.strip()
            if not raw:
                continue

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                yield LineResult(line_no, None, [TraceError(line_no, "$", f"invalid JSON: {exc}")])
                continue

            schema_errors = sorted(
                validator.iter_errors(obj), key=lambda e: (list(e.absolute_path), e.message)
            )
            errors = [TraceError(line_no, _json_path(e), e.message) for e in schema_errors]
            if not errors:
                errors = _semantic_errors(obj, line_no, seen_trace_ids)

            if errors:
                yield LineResult(line_no, None, errors)
                continue

            try:
                trace = Trace.model_validate(obj)
            except PydanticValidationError as exc:
                yield LineResult(line_no, None, [TraceError(line_no, "$", str(exc))])
                continue

            yield LineResult(line_no, trace, [])


def pair_results(steps: list[Step]) -> dict[int, int | None]:
    """Map each `tool_result` step's `i` to the nearest preceding `tool_call`
    step's `i` with the same `name`. An orphan result (no such call) maps to
    `None`. Steps must already be ordered by `i`.
    """
    pairing: dict[int, int | None] = {}
    last_call_i_by_name: dict[str, int] = {}
    for step in steps:
        if step.kind == "tool_call" and step.name is not None:
            last_call_i_by_name[step.name] = step.i
        elif step.kind == "tool_result":
            pairing[step.i] = last_call_i_by_name.get(step.name) if step.name is not None else None
    return pairing


def merge_probes(traces: list[Trace], probes_path: str | Path) -> tuple[list[Trace], list[str]]:
    """Merge `probes_path`'s lines into the matching traces (by `trace_id`).

    Each probe becomes an appended `state_probe` step, `role: environment`,
    `i` continuing after the trace's last step, in file order. A probe for
    an unknown `trace_id` is skipped and reported as a warning, not an
    error. Public so callers other than `load_traces_report` (the `Checker`)
    can merge probes into traces they already hold.
    """
    return _merge_probes(traces, Path(probes_path))


def _merge_probes(traces: list[Trace], probes_path: Path) -> tuple[list[Trace], list[str]]:
    by_id = {t.trace_id: t for t in traces}
    order = [t.trace_id for t in traces]
    appended: dict[str, list[dict[str, Any]]] = {tid: [] for tid in order}
    warnings: list[str] = []

    validator = Draft202012Validator(_PROBE_SCHEMA, format_checker=_FORMAT_CHECKER)

    with probes_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw = raw_line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            probe_errors = list(validator.iter_errors(obj))
            if probe_errors:
                messages = "; ".join(e.message for e in probe_errors)
                raise TraceValidationError([TraceError(line_no, "$", f"invalid probe: {messages}")])
            trace_id = obj["trace_id"]
            if trace_id not in by_id:
                warnings.append(f"probe on line {line_no} references unknown trace_id {trace_id!r}")
                continue
            appended[trace_id].append(obj)

    merged: list[Trace] = []
    for trace_id in order:
        trace = by_id[trace_id]
        extra = appended[trace_id]
        if not extra:
            merged.append(trace)
            continue
        next_i = trace.steps[-1].i + 1 if trace.steps else 0
        new_steps = list(trace.steps)
        for probe in extra:
            new_steps.append(
                Step(
                    i=next_i,
                    ts=probe["ts"],
                    kind="state_probe",
                    role="environment",
                    name=probe["name"],
                    content=None,
                    args=probe.get("args"),
                    ok=probe.get("ok"),
                    output=probe.get("output"),
                    error=probe.get("error"),
                )
            )
            next_i += 1
        merged.append(trace.model_copy(update={"steps": new_steps}))
    return merged, warnings


def load_traces_report(
    path_or_alias: str | Path,
    *,
    strict: bool = False,
    probes: str | Path | None = None,
) -> LoadReport:
    """Load and validate a trace file, reporting every error and warning."""
    traces: list[Trace] = []
    errors: list[TraceError] = []

    for line_result in iter_trace_lines(path_or_alias):
        if line_result.errors:
            errors.extend(line_result.errors)
            if strict:
                raise TraceValidationError(errors)
            continue
        assert line_result.trace is not None
        traces.append(line_result.trace)

    warnings: list[str] = []
    if probes is not None:
        traces, warnings = _merge_probes(traces, _resolve_input(probes))

    return LoadReport(traces=traces, errors=errors, warnings=warnings)


def load_traces(
    path_or_alias: str | Path,
    *,
    strict: bool = False,
    probes: str | Path | None = None,
) -> list[Trace]:
    """Load a trace file, returning only the valid traces.

    Use `load_traces_report` for the per-line errors and probe warnings.
    """
    return load_traces_report(path_or_alias, strict=strict, probes=probes).traces


def dump_trace(trace: Trace) -> str:
    """Serialize a trace to one canonical JSON line, validating it first."""
    obj = trace.model_dump(mode="json", exclude_none=False)
    # Optional top-level objects are omitted rather than written as null.
    for key in ("ground_truth", "meta"):
        if obj.get(key) is None:
            obj.pop(key, None)
    validator = _trace_validator()
    schema_errors = list(validator.iter_errors(obj))
    if schema_errors:
        raise TraceValidationError([TraceError(0, _json_path(e), e.message) for e in schema_errors])
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
