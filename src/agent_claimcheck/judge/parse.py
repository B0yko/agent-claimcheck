"""Parsing the judge's JSON answer.

Number validation is strict: bare numeric types only, `bool` rejected even
though it is an `int` subclass, and finite-range checks.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

#: `rationale` longer than this is cut, and `rationale_truncated` is set.
RATIONALE_LIMIT = 300

FAILURE_KINDS: tuple[str, ...] = (
    "none",
    "phantom_action",
    "error_ignored",
    "wrong_target",
    "wrong_value",
    "not_persisted",
    "partial_completion",
    "cannot_tell",
)

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.S)


class ParseError(Exception):
    """The judge's content could not be parsed into a valid judgment.

    Callers map this to `DetectorOutput(abstain=True, abstain_reason="parse_error")`.
    """


@dataclass(frozen=True)
class ParsedJudgment:
    """A validated judge answer."""

    p_success: float
    failure_kind: str
    evidence_steps: list[int]
    rationale: str
    rationale_truncated: bool
    invalid_citation: bool
    claims: Any = None


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError("p_success must be a number")
    number = float(value)
    if not math.isfinite(number) or not (0.0 <= number <= 1.0):
        raise ParseError("p_success must be within [0, 1]")
    return number


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        raise ParseError("evidence_steps must be a list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ParseError("evidence_steps must contain only integers")
        result.append(item)
    return result


def parse_judgment(content: str, *, valid_steps: Collection[int]) -> ParsedJudgment:
    """Parse one judge answer.

    `content` may be a bare JSON object or one fenced in ``` or ```json.
    Anything else, a missing key, a wrong type, an out-of-range number or an
    unknown enum value raises `ParseError`. A cited step not in
    `valid_steps` does not raise: it sets `invalid_citation` instead.
    """
    unfenced = _strip_fence(content)
    try:
        obj = json.loads(unfenced)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ParseError("expected a JSON object")

    for key in ("p_success", "failure_kind", "evidence_steps", "rationale"):
        if key not in obj:
            raise ParseError(f"missing key: {key}")

    p_success = _number(obj["p_success"])

    failure_kind = obj["failure_kind"]
    if not isinstance(failure_kind, str) or failure_kind not in FAILURE_KINDS:
        raise ParseError(f"unknown failure_kind: {failure_kind!r}")

    evidence_steps = _int_list(obj["evidence_steps"])

    rationale = obj["rationale"]
    if not isinstance(rationale, str):
        raise ParseError("rationale must be a string")
    rationale_truncated = len(rationale) > RATIONALE_LIMIT
    if rationale_truncated:
        rationale = rationale[:RATIONALE_LIMIT]

    invalid_citation = any(step not in valid_steps for step in evidence_steps)

    return ParsedJudgment(
        p_success=p_success,
        failure_kind=failure_kind,
        evidence_steps=evidence_steps,
        rationale=rationale,
        rationale_truncated=rationale_truncated,
        invalid_citation=invalid_citation,
        claims=obj.get("claims"),
    )
