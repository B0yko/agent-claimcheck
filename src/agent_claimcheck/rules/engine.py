"""The declarative rule engine: safe YAML packs, no code evaluation.

A rule pack maps claim types to a tool-call glob, a set of receipt checks on
that call's result, and an optional state-probe glob with its own checks. It
never embeds a Python expression: the only vocabulary is `exists`, `equals`,
`in`, `contains`, `matches` and the list check `items`, plus a small set of
typed normalisers (see `docs/rules.md` and ADR 0002).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from agent_claimcheck.schema import Step, pair_results

Outcome = Literal["contradicted", "unsupported", "unknown", "receipt_only", "probe_supported"]

#: Worst to best. Used both to pick the trace outcome and to order claims.
OUTCOME_ORDER: tuple[Outcome, ...] = (
    "contradicted",
    "unsupported",
    "unknown",
    "receipt_only",
    "probe_supported",
)

#: Raw `p_success` per outcome (ADR 0004). `unknown` abstains: no fixed score.
RULE_SCORES: dict[Outcome, float | None] = {
    "contradicted": 0.03,
    "unsupported": 0.05,
    "unknown": None,
    "receipt_only": 0.70,
    "probe_supported": 0.97,
}

_CheckOp = Literal["exists", "equals", "in", "contains", "matches", "items"]
_OPS: tuple[_CheckOp, ...] = ("exists", "equals", "in", "contains", "matches", "items")
_NORMALISERS = ("datetime", "email", "number", "string")


class RulePackError(Exception):
    """Raised when a rule pack fails to load: bad YAML, schema or regex."""


@dataclass(frozen=True)
class Check:
    """One field check: exactly one operator, plus an optional normaliser."""

    op: _CheckOp
    value: Any
    as_: str | None = None


@dataclass(frozen=True)
class ItemsCheck:
    """The `items` operator: what must, or must not, be in a list field.

    `some`: at least one element passes every field check. `none`: no element
    does. `covers`: every value of a claimed list equals some element. `when`:
    checks on the same output that say whether this list describes the claim.
    """

    some: dict[str, Check] | None = None
    none: dict[str, Check] | None = None
    covers: str | None = None
    when: dict[str, Check] = field(default_factory=dict)


@dataclass(frozen=True)
class Rule:
    """One claim type's evidence rule: an action, its receipt, an optional probe."""

    action: str
    receipt: dict[str, Check]
    probe: str | None = None
    probe_checks: dict[str, Check] = field(default_factory=dict)


@dataclass(frozen=True)
class RulePack:
    """A loaded, validated rule pack."""

    pack: str
    version: int
    tools: tuple[str, ...]
    claim_patterns: dict[str, tuple[re.Pattern[str], ...]]
    claims: dict[str, Rule]
    path: Path | None = None
    requires_probe: str | None = None


@dataclass(frozen=True)
class CheckResultRow:
    """One evaluated check, for `details.claims[].checks`."""

    path: str
    op: str
    expected: Any
    actual: Any
    result: Literal["pass", "fail", "skipped"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "op": self.op,
            "expected": self.expected,
            "actual": self.actual,
            "result": self.result,
        }


@dataclass(frozen=True)
class ClaimOutcome:
    """One claim's outcome, with its step citation and evaluated checks."""

    claim_type: str
    outcome: Outcome
    step: int | None
    detail: str
    checks: tuple[CheckResultRow, ...] = ()
    missing_subject: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim_type,
            "outcome": self.outcome,
            "step": self.step,
            "detail": self.detail,
            "checks": [c.as_dict() for c in self.checks],
            "missing_subject": self.missing_subject,
        }


_FIELD_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "minProperties": 1,
    "additionalProperties": False,
    "properties": {
        "exists": {"type": "boolean"},
        "equals": {},
        "in": {"type": "array"},
        "contains": {},
        "matches": {"type": "string"},
        "as": {"enum": list(_NORMALISERS)},
    },
}

_FIELD_CHECKS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "minProperties": 1,
    "additionalProperties": _FIELD_CHECK_SCHEMA,
}

#: `items` holds plain field checks only, so it cannot nest.
_CHECK_SCHEMA: dict[str, Any] = {
    **_FIELD_CHECK_SCHEMA,
    "properties": {
        **_FIELD_CHECK_SCHEMA["properties"],
        "items": {
            "type": "object",
            "additionalProperties": False,
            "anyOf": [{"required": ["some"]}, {"required": ["none"]}, {"required": ["covers"]}],
            "properties": {
                "some": _FIELD_CHECKS_SCHEMA,
                "none": _FIELD_CHECKS_SCHEMA,
                "covers": {"type": "string", "pattern": r"^\{(subject|args)\.[^{}]*\}$"},
                "when": _FIELD_CHECKS_SCHEMA,
            },
        },
    },
}

_PACK_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["pack", "version", "claims"],
    "additionalProperties": False,
    "properties": {
        "pack": {"type": "string"},
        "version": {"type": "integer"},
        "tools": {"type": "array", "items": {"type": "string"}},
        "requires_probe": {"type": "string"},
        "claim_patterns": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
        },
        "claims": {"type": "object", "additionalProperties": {"$ref": "#/$defs/rule"}},
    },
    "$defs": {
        "rule": {
            "type": "object",
            "required": ["action", "receipt"],
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string"},
                "receipt": {"type": "object", "additionalProperties": _CHECK_SCHEMA},
                "probe": {"type": "string"},
                "probe_checks": {"type": "object", "additionalProperties": _CHECK_SCHEMA},
            },
        },
    },
}

_PACK_VALIDATOR = Draft202012Validator(_PACK_SCHEMA)


def _json_path(err: JsonSchemaValidationError) -> str:
    parts = ["$"]
    for segment in err.absolute_path:
        if isinstance(segment, int):
            parts[-1] += f"[{segment}]"
        else:
            parts.append(f".{segment}")
    return "".join(parts)


def _build_check(raw: Mapping[str, Any], context: str) -> Check:
    present = [op for op in _OPS if op in raw]
    if len(present) != 1:
        raise RulePackError(
            f"{context}: a check must declare exactly one of {_OPS}, got {sorted(raw)}"
        )
    op = present[0]
    as_ = raw.get("as")
    if op == "matches":
        try:
            pattern = re.compile(raw["matches"])
        except re.error as exc:
            raise RulePackError(f"{context}: bad regex {raw['matches']!r}: {exc}") from exc
        return Check(op="matches", value=pattern, as_=as_)
    if op == "items":
        return Check(op="items", value=_build_items(raw["items"], as_, context), as_=as_)
    value = raw["in"] if op == "in" else raw[op]
    return Check(op=op, value=value, as_=as_)


def _build_items(raw: Mapping[str, Any], as_: str | None, context: str) -> ItemsCheck:
    if as_ is not None and "covers" not in raw:
        raise RulePackError(f"{context}: `as` on `items` applies only to `covers`")

    def clause(name: str) -> dict[str, Check] | None:
        if name not in raw:
            return None
        return {
            path: _build_check(check, f"{context}.items.{name}.{path}")
            for path, check in raw[name].items()
        }

    for name in ("some", "none"):
        for path in raw.get(name, {}):
            if path == "args" or path.startswith("args."):
                raise RulePackError(f"{context}.items.{name}: {path!r} is not an element field")
    return ItemsCheck(
        some=clause("some"),
        none=clause("none"),
        covers=raw.get("covers"),
        when=clause("when") or {},
    )


def load_rule_pack(path: str | Path) -> RulePack:
    """Load and validate one rule pack from a YAML file.

    Raises `RulePackError` on invalid YAML (including a disallowed tag such
    as `!!python/object`), a schema violation, an unknown key, or a bad
    regex in `claim_patterns` or a `matches` check.
    """
    resolved = Path(path)
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RulePackError(f"{resolved}: invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise RulePackError(f"{resolved}: a rule pack must be a YAML mapping")

    errors = sorted(_PACK_VALIDATOR.iter_errors(raw), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        raise RulePackError(f"{resolved}: {_json_path(first)}: {first.message}")

    claim_patterns: dict[str, tuple[re.Pattern[str], ...]] = {}
    for claim_type, patterns in raw.get("claim_patterns", {}).items():
        compiled = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                raise RulePackError(
                    f"{resolved}: bad regex {pattern!r} in claim_patterns.{claim_type}: {exc}"
                ) from exc
        claim_patterns[claim_type] = tuple(compiled)

    claims: dict[str, Rule] = {}
    for claim_type, rule_raw in raw.get("claims", {}).items():
        context = f"{resolved}: claims.{claim_type}"
        receipt = {
            path_: _build_check(check_raw, f"{context}.receipt.{path_}")
            for path_, check_raw in rule_raw.get("receipt", {}).items()
        }
        probe_checks = {
            path_: _build_check(check_raw, f"{context}.probe_checks.{path_}")
            for path_, check_raw in rule_raw.get("probe_checks", {}).items()
        }
        claims[claim_type] = Rule(
            action=rule_raw["action"],
            receipt=receipt,
            probe=rule_raw.get("probe"),
            probe_checks=probe_checks,
        )

    return RulePack(
        pack=raw["pack"],
        version=raw["version"],
        tools=tuple(raw.get("tools", [])),
        claim_patterns=claim_patterns,
        claims=claims,
        path=resolved,
        requires_probe=raw.get("requires_probe"),
    )


_PACKS_DIR = Path(__file__).resolve().parent / "packs"
_BUILTIN_PACK_NAMES = ("booking", "booking-truth", "crm", "coding", "generic")


def builtin_packs() -> dict[str, RulePack]:
    """The built-in packs, keyed by pack name."""
    return {name: load_rule_pack(_PACKS_DIR / f"{name}.yaml") for name in _BUILTIN_PACK_NAMES}


def pack_applies(pack: RulePack, domain: str, steps: Sequence[Step]) -> bool:
    """A pack applies when a tool_call matches one of its globs, or the trace
    makes no tool calls at all and the task domain equals the pack name.
    A pack with `requires_probe` applies only to traces holding a matching
    state probe, so common tool names cannot pull in another system's rules.
    """
    if pack.requires_probe is not None and not any(
        s.kind == "state_probe" and s.name and fnmatchcase(s.name, pack.requires_probe)
        for s in steps
    ):
        return False
    tool_calls = [s for s in steps if s.kind == "tool_call" and s.name]
    if any(fnmatchcase(call.name or "", glob) for call in tool_calls for glob in pack.tools):
        return True
    return not tool_calls and domain == pack.pack


def _get_path(obj: Any, path: str) -> tuple[Any, bool]:
    """Resolve a dotted field path (list indices as digits) into `obj`."""
    if path == "":
        return obj, True
    current = obj
    for part in path.split("."):
        if isinstance(current, list):
            if not part.lstrip("-").isdigit():
                return None, False
            idx = int(part)
            if not (-len(current) <= idx < len(current)):
                return None, False
            current = current[idx]
        elif isinstance(current, dict):
            if part not in current:
                return None, False
            current = current[part]
        else:
            return None, False
    return current, True


def _resolve_field(field_path: str, output: Any, call_args: Mapping[str, Any]) -> tuple[Any, bool]:
    if field_path == "args" or field_path.startswith("args."):
        rest = field_path[len("args.") :] if field_path.startswith("args.") else ""
        return _get_path(call_args, rest)
    return _get_path(output, field_path)


_PLACEHOLDER = re.compile(r"\{(?P<ns>subject|args)\.(?P<path>[^{}]*)\}")


def _interpolate(
    value: Any, subject: Mapping[str, Any], call_args: Mapping[str, Any]
) -> tuple[Any, bool]:
    """Resolve `{subject.*}`/`{args.*}` placeholders. `ok=False` means a
    referenced field was missing (`missing_subject`).
    """
    if not isinstance(value, str):
        return value, True

    full = _PLACEHOLDER.fullmatch(value)
    if full is not None:
        source = subject if full["ns"] == "subject" else call_args
        return _get_path(source, full["path"])

    missing = False

    def _sub(match: re.Match[str]) -> str:
        nonlocal missing
        source = subject if match["ns"] == "subject" else call_args
        resolved, found = _get_path(source, match["path"])
        if not found:
            missing = True
            return ""
        return str(resolved)

    result = _PLACEHOLDER.sub(_sub, value)
    return (None, False) if missing else (result, True)


def _normalise(raw: Any, as_: str) -> tuple[Any, bool]:
    if as_ == "datetime":
        if not isinstance(raw, str):
            return None, False
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None, False
        return (dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)), True
    if as_ == "email":
        if not isinstance(raw, str):
            return None, False
        return raw.strip().casefold(), True
    if as_ == "number":
        if isinstance(raw, bool):
            return None, False
        try:
            return float(raw), True
        except (TypeError, ValueError):
            return None, False
    if as_ == "string":
        return str(raw).strip().casefold(), True
    raise AssertionError(f"unknown normaliser {as_!r}")  # unreachable: schema enum


def _equal(actual: Any, expected: Any, as_: str | None) -> bool:
    if as_ is None:
        return bool(actual == expected)
    a_norm, a_ok = _normalise(actual, as_)
    e_norm, e_ok = _normalise(expected, as_)
    if not (a_ok and e_ok):
        return False
    if as_ == "number":
        return bool(abs(a_norm - e_norm) <= 1e-9)
    return bool(a_norm == e_norm)


def _apply_check(
    field_path: str,
    check: Check,
    subject: Mapping[str, Any],
    call_args: Mapping[str, Any],
    output: Any,
) -> tuple[CheckResultRow, bool]:
    if check.op == "items":
        return _apply_items(field_path, check, subject, call_args, output)

    actual, found = _resolve_field(field_path, output, call_args)

    if check.op == "exists":
        want = bool(check.value)
        passed = found == want
        return CheckResultRow(
            field_path, "exists", want, actual if found else None, _pf(passed)
        ), False

    if not found:
        if check.op != "matches":
            values = check.value if check.op == "in" else [check.value]
            if not all(_interpolate(item, subject, call_args)[1] for item in values):
                return CheckResultRow(field_path, check.op, check.value, None, "skipped"), True
        return CheckResultRow(field_path, check.op, _display(check.value), None, "fail"), False

    if check.op == "matches":
        pattern: re.Pattern[str] = check.value
        text = actual if isinstance(actual, str) else json.dumps(actual, sort_keys=True)
        passed = pattern.search(text) is not None
        return CheckResultRow(field_path, "matches", pattern.pattern, actual, _pf(passed)), False

    if check.op == "in":
        resolved: list[Any] = []
        missing = False
        for item in check.value:
            item_value, ok = _interpolate(item, subject, call_args)
            resolved.append(item_value)
            missing = missing or not ok
        if missing:
            return CheckResultRow(field_path, "in", check.value, actual, "skipped"), True
        passed = any(_equal(actual, item, check.as_) for item in resolved)
        return CheckResultRow(field_path, "in", resolved, actual, _pf(passed)), False

    expected, ok = _interpolate(check.value, subject, call_args)
    if not ok:
        return CheckResultRow(field_path, check.op, check.value, actual, "skipped"), True

    if check.op == "equals":
        passed = _equal(actual, expected, check.as_)
        return CheckResultRow(field_path, "equals", expected, actual, _pf(passed)), False

    # contains
    if isinstance(actual, Mapping):
        # An object contains a mapping when every expected key is present with an equal value.
        passed = isinstance(expected, Mapping) and all(
            key in actual and _equal(actual[key], value, check.as_)
            for key, value in expected.items()
        )
    elif isinstance(actual, str):
        passed = str(expected) in actual
    elif isinstance(actual, list):
        passed = any(_equal(item, expected, check.as_) for item in actual)
    else:
        passed = False
    return CheckResultRow(field_path, "contains", expected, actual, _pf(passed)), False


def _claim_value_missing(
    check: Check, subject: Mapping[str, Any], call_args: Mapping[str, Any]
) -> bool:
    """Whether an element check depends on a claim value that is absent or null."""
    if check.op in ("exists", "matches"):
        return False
    values = check.value if check.op == "in" else [check.value]
    for item in values:
        resolved, ok = _interpolate(item, subject, call_args)
        if not ok or (resolved is None and item is not None):
            return True
    return False


def _element_matches(
    checks: Mapping[str, Check],
    element: Any,
    subject: Mapping[str, Any],
    call_args: Mapping[str, Any],
) -> bool:
    rows, _missing = _evaluate_checks(checks, subject, call_args, element)
    return all(row.result == "pass" for row in rows)


def _items_applies(
    check: Check, subject: Mapping[str, Any], call_args: Mapping[str, Any], output: Any
) -> bool:
    """Whether an `items` check's `when` guard holds on this output."""
    if check.op != "items":
        return True
    guard: ItemsCheck = check.value
    return _element_matches(guard.when, output, subject, call_args)


def _shown(check: Check, subject: Mapping[str, Any], call_args: Mapping[str, Any]) -> Any:
    """An element check's value for the evidence row, with claim values filled in."""
    if check.op == "in":
        return [_interpolate(item, subject, call_args)[0] for item in check.value]
    if check.op in ("equals", "contains"):
        value, ok = _interpolate(check.value, subject, call_args)
        return value if ok else check.value
    return _display(check.value)


def _apply_items(
    field_path: str,
    check: Check,
    subject: Mapping[str, Any],
    call_args: Mapping[str, Any],
    output: Any,
) -> tuple[CheckResultRow, bool]:
    spec: ItemsCheck = check.value
    expected: dict[str, Any] = {
        name: {path: _shown(c, subject, call_args) for path, c in clause.items()}
        for name, clause in (("some", spec.some), ("none", spec.none))
        if clause is not None
    }
    if spec.covers is not None:
        expected["covers"] = spec.covers

    if not _items_applies(check, subject, call_args, output):
        return CheckResultRow(field_path, "items", expected, None, "skipped"), False

    clauses = [c for c in (spec.some, spec.none) if c is not None]
    claimed: Any = None
    missing = any(
        _claim_value_missing(c, subject, call_args) for clause in clauses for c in clause.values()
    )
    if spec.covers is not None:
        claimed, found = _interpolate(spec.covers, subject, call_args)
        missing = missing or not found or claimed is None or claimed == []
    if missing:
        return CheckResultRow(field_path, "items", expected, None, "skipped"), True

    actual, found = _resolve_field(field_path, output, call_args)
    if not found or not isinstance(actual, list):
        return CheckResultRow(field_path, "items", expected, None, "fail"), False

    summary: dict[str, Any] = {"count": len(actual)}
    passed = True
    if spec.some is not None:
        summary["some"] = sum(_element_matches(spec.some, e, subject, call_args) for e in actual)
        passed = passed and summary["some"] > 0
    if spec.none is not None:
        summary["none"] = sum(_element_matches(spec.none, e, subject, call_args) for e in actual)
        passed = passed and summary["none"] == 0
    if spec.covers is not None:
        wanted = claimed if isinstance(claimed, list) else [claimed]
        uncovered = [w for w in wanted if not any(_equal(e, w, check.as_) for e in actual)]
        summary["uncovered"] = uncovered[:5]
        passed = passed and not uncovered
    return CheckResultRow(field_path, "items", expected, summary, _pf(passed)), False


def _pf(passed: bool) -> Literal["pass", "fail"]:
    return "pass" if passed else "fail"


def _display(value: Any) -> Any:
    return value.pattern if isinstance(value, re.Pattern) else value


def _evaluate_checks(
    checks: Mapping[str, Check],
    subject: Mapping[str, Any],
    call_args: Mapping[str, Any],
    output: Any,
) -> tuple[list[CheckResultRow], bool]:
    rows: list[CheckResultRow] = []
    any_missing = False
    for field_path, check in checks.items():
        row, missing = _apply_check(field_path, check, subject, call_args, output)
        rows.append(row)
        any_missing = any_missing or missing
    return rows, any_missing


def _invert_pairing(steps: Sequence[Step]) -> dict[int, Step]:
    """Map each `tool_call` step's `i` to its paired `tool_result` step."""
    by_i = {s.i: s for s in steps}
    pairing = pair_results(list(steps))
    result_for_call: dict[int, Step] = {}
    for result_i, call_i in pairing.items():
        if call_i is not None:
            result_for_call[call_i] = by_i[result_i]
    return result_for_call


def evaluate_claim(
    subject: Mapping[str, Any],
    claim_type: str,
    *,
    domain: str,
    steps: Sequence[Step],
    packs: Sequence[RulePack],
) -> ClaimOutcome:
    """Evaluate one claim's outcome against the trace's steps."""
    applicable = [p for p in packs if pack_applies(p, domain, steps)]
    rule = next((p.claims[claim_type] for p in applicable if claim_type in p.claims), None)
    if rule is None:
        return ClaimOutcome(claim_type, "unknown", None, "no rule pack covers this claim type")

    matching_calls = [
        s for s in steps if s.kind == "tool_call" and s.name and fnmatchcase(s.name, rule.action)
    ]
    if not matching_calls:
        return ClaimOutcome(
            claim_type, "unsupported", None, f"no tool call matches {rule.action!r}"
        )

    result_for_call = _invert_pairing(steps)

    evaluated = []
    for call in matching_calls:
        result_step = result_for_call.get(call.i)
        call_args = call.args or {}
        output = result_step.output if result_step is not None else None
        checks, missing = _evaluate_checks(rule.receipt, subject, call_args, output)
        passes = (
            result_step is not None
            and result_step.ok is True
            and result_step.error is None
            and all(c.result != "fail" for c in checks)
        )
        evaluated.append((call, result_step, passes, checks, missing))

    passing = [e for e in evaluated if e[2]]
    if passing:
        call, result_step, _passed, receipt_checks, receipt_missing = max(
            passing, key=lambda e: e[0].i
        )
        assert result_step is not None
        return _resolve_probe(
            claim_type, subject, call, result_step, receipt_checks, receipt_missing, rule, steps
        )

    contradicting = [
        e
        for e in evaluated
        if e[1] is not None
        and (e[1].ok is False or e[1].error is not None or any(c.result == "fail" for c in e[3]))
    ]
    if contradicting:
        call, result_step, _passed, checks, missing = max(contradicting, key=lambda e: e[0].i)
        assert result_step is not None
        return ClaimOutcome(
            claim_type,
            "contradicted",
            result_step.i,
            "the matching call's result contradicts the claim",
            tuple(checks),
            missing,
        )

    return ClaimOutcome(
        claim_type, "unsupported", None, "no matching call returned a result", missing_subject=False
    )


def _resolve_probe(
    claim_type: str,
    subject: Mapping[str, Any],
    call: Step,
    result_step: Step,
    receipt_checks: list[CheckResultRow],
    receipt_missing: bool,
    rule: Rule,
    steps: Sequence[Step],
) -> ClaimOutcome:
    if rule.probe is None:
        return ClaimOutcome(
            claim_type,
            "receipt_only",
            result_step.i,
            "the call's result matches the claim; no state probe is defined for it",
            tuple(receipt_checks),
            receipt_missing,
        )

    probe_candidates = [
        s for s in steps if s.kind == "state_probe" and s.name and fnmatchcase(s.name, rule.probe)
    ]
    if not probe_candidates:
        return ClaimOutcome(
            claim_type,
            "receipt_only",
            result_step.i,
            "the call's result matches the claim, but no state probe checked it",
            tuple(receipt_checks),
            receipt_missing,
        )

    probe_step = max(probe_candidates, key=lambda s: s.i)
    probe_checks, probe_missing = _evaluate_checks(
        rule.probe_checks, subject, call.args or {}, probe_step.output
    )
    all_checks = receipt_checks + probe_checks
    missing_subject = receipt_missing or probe_missing

    contradicted = (
        probe_step.ok is False
        or probe_step.error is not None
        or any(c.result == "fail" for c in probe_checks)
    )
    if contradicted:
        return ClaimOutcome(
            claim_type,
            "contradicted",
            probe_step.i,
            "the state probe contradicts the claim",
            tuple(all_checks),
            missing_subject,
        )

    if not all(
        _items_applies(c, subject, call.args or {}, probe_step.output)
        for c in rule.probe_checks.values()
    ):
        return ClaimOutcome(
            claim_type,
            "unknown",
            probe_step.i,
            "the state probe does not describe the system this claim is about",
            tuple(all_checks),
            missing_subject,
        )

    if all_checks and all(c.result == "skipped" for c in all_checks):
        return ClaimOutcome(
            claim_type,
            "receipt_only",
            result_step.i,
            "every check was skipped, so nothing confirms the claim",
            tuple(all_checks),
            missing_subject,
        )

    return ClaimOutcome(
        claim_type,
        "probe_supported",
        probe_step.i,
        "the state probe confirms the claim",
        tuple(all_checks),
        missing_subject,
    )
