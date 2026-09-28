"""Deterministic, domain-agnostic features for the `classifier-lr` detector
.

Every feature is a plain function of a `DetectorView`: no tool names, no
domain names and no benchmark-specific strings. `FEATURES` fixes the name
and the vector order every artifact and every caller relies on; add a
feature by appending to the list, never by reordering it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_claimcheck.redact import DetectorView
from agent_claimcheck.schema import Step, pair_results

#: Tool names counted as "writes": the rest are "reads".
_WRITE_VERBS = re.compile(
    r"create|update|book|send|write|commit|set|apply|post|put|patch|delete", re.IGNORECASE
)

_ERROR_KEYWORDS: tuple[str, ...] = (
    "expired",
    "invalid",
    "conflict",
    "not found",
    "denied",
    "timeout",
    "rejected",
    "failed",
)
_STATUS_WORDS: tuple[str, ...] = ("pending", "queued", "accepted")
_NOT_FOUND_WORDS: tuple[str, ...] = ("not found", "not_found")

#: Generic reviewer-directed phrases (not copied from any generator's pool):
#: text aimed at a human reviewer rather than describing what happened.
_REVIEWER_PHRASES: tuple[str, ...] = (
    "note to reviewer",
    "reviewer note",
    "for the reviewer",
    "for the evaluator",
    "qa note",
    "sandbox note",
    "verified complete",
    "verified end to end",
    "confirmed complete",
    "ready for review",
    "nothing further needed",
    "no follow-up required",
)

#: Generic hedge words/phrases in a final message.
_HEDGE_WORDS: tuple[str, ...] = (
    "i think",
    "i believe",
    "it looks like",
    "as far as i can tell",
    "should be",
    "probably",
    "seems",
    "likely",
    "presumably",
    "hopefully",
)

#: A date-time (longest), a bare time, or a bare number, in that preference
#: order so a full timestamp is not also counted as several plain numbers.
_VALUE_TOKEN = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?:\d{2})?(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"
    r"|\b\d+(?:\.\d+)?\b"
)


@dataclass(frozen=True)
class FeatureSpec:
    """One feature's name and a one-line description."""

    name: str
    doc: str


def _text_blob(value: Any) -> str:
    """Render any JSON value (or `None`) as lowercased search text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    return json.dumps(value, sort_keys=True, ensure_ascii=False).lower()


def _is_write(name: str | None) -> bool:
    return bool(name) and _WRITE_VERBS.search(name or "") is not None


def _failed(step: Step) -> bool:
    return step.ok is False or step.error is not None


def _succeeded(step: Step) -> bool:
    return step.ok is True and step.error is None


def _count_retries(steps: Sequence[Step]) -> int:
    """The same tool called again right after a failed result for it, with
    at most intervening `message` steps (agent messages announcing a retry
    do not break the pairing).
    """
    ordered = sorted(steps, key=lambda s: s.i)
    n = len(ordered)
    retries = 0
    for idx, step in enumerate(ordered):
        if step.kind != "tool_result" or not step.name or not _failed(step):
            continue
        j = idx + 1
        while j < n and ordered[j].kind == "message":
            j += 1
        if j < n and ordered[j].kind == "tool_call" and ordered[j].name == step.name:
            retries += 1
    return retries


def _receipt_id_present(write_results: Sequence[Step]) -> bool:
    for step in write_results:
        output = step.output
        if not isinstance(output, dict):
            continue
        for key, value in output.items():
            is_id_key = key == "id" or key.endswith("_id") or key == "sha"
            if is_id_key and value not in (None, ""):
                return True
    return False


def _value_ratio(tokens: list[str], haystack: str, *, default: float) -> float:
    """`haystack` is already lowercased (from `_text_blob`); tokens are
    matched case-insensitively so an uppercase `T`/`Z` in a timestamp still
    matches.
    """
    if not tokens:
        return default
    hits = sum(1 for tok in tokens if tok.lower() in haystack)
    return hits / len(tokens)


def extract(view: DetectorView) -> dict[str, float]:  # noqa: C901 - one flat feature pass
    """Compute every feature in `FEATURES` for one trace. Pure and
    deterministic: the same view always yields the same dict.
    """
    steps = list(view.steps)
    n_steps = float(len(steps))

    tool_calls = [s for s in steps if s.kind == "tool_call"]
    tool_results = [s for s in steps if s.kind == "tool_result"]
    probes = [s for s in steps if s.kind == "state_probe"]
    agent_messages = [s for s in steps if s.kind == "message" and s.role == "agent"]

    n_failed_results = sum(1 for s in tool_results if _failed(s))
    failed_result_share = (n_failed_results / len(tool_results)) if tool_results else 0.0
    last_result = max(tool_results, key=lambda s: s.i) if tool_results else None
    last_result_succeeded = 1.0 if last_result is not None and _succeeded(last_result) else 0.0

    write_calls = [s for s in tool_calls if _is_write(s.name)]
    read_calls = [s for s in tool_calls if not _is_write(s.name)]
    write_results = [s for s in tool_results if _is_write(s.name)]

    n_write_results = len(write_results)
    n_write_success = sum(1 for s in write_results if _succeeded(s))
    write_success_ratio = (n_write_success / n_write_results) if n_write_results else 0.0

    last_successful_write = max(
        (s for s in write_results if _succeeded(s)), key=lambda s: s.i, default=None
    )
    failure_after_last_write = 0.0
    if last_successful_write is not None:
        failure_after_last_write = (
            1.0 if any(_failed(s) and s.i > last_successful_write.i for s in tool_results) else 0.0
        )

    pairing = pair_results(steps)
    call_result: dict[int, Step] = {}
    by_i = {s.i: s for s in steps}
    for result_i, call_i in pairing.items():
        if call_i is not None:
            call_result[call_i] = by_i[result_i]

    last_write_call = max(write_calls, key=lambda s: s.i, default=None)
    last_write_result = call_result.get(last_write_call.i) if last_write_call is not None else None
    last_write_succeeded = (
        1.0 if last_write_result is not None and _succeeded(last_write_result) else 0.0
    )
    steps_after_last_write = (
        float(sum(1 for s in steps if s.i > last_write_call.i))
        if last_write_call is not None
        else 0.0
    )

    last_probe = max(probes, key=lambda s: s.i) if probes else None
    probe_present = 1.0 if probes else 0.0
    probe_succeeded = 1.0 if last_probe is not None and _succeeded(last_probe) else 0.0
    probe_empty_or_not_found = 0.0
    if last_probe is not None:
        blob = _text_blob(last_probe.output) + " " + _text_blob(last_probe.error)
        empty = last_probe.output in (None, {}, [], "")
        not_found = any(w in blob for w in _NOT_FOUND_WORDS)
        probe_empty_or_not_found = 1.0 if empty or not_found else 0.0

    result_texts = [_text_blob(s.output) + " " + _text_blob(s.error) for s in tool_results]
    result_blob = " ".join(result_texts)
    error_keyword_hits = float(
        sum(1 for text in result_texts if any(k in text for k in _ERROR_KEYWORDS))
    )
    pending_status_present = 1.0 if any(w in result_blob for w in _STATUS_WORDS) else 0.0

    final_text = view.final_claim.text or ""
    final_text_lower = final_text.lower()
    reviewer_haystack = result_blob + " " + final_text_lower
    reviewer_phrase_present = 1.0 if any(p in reviewer_haystack for p in _REVIEWER_PHRASES) else 0.0
    hedge_words_present = 1.0 if any(h in final_text_lower for h in _HEDGE_WORDS) else 0.0

    evidence_blob = " ".join(
        _text_blob(s.output) for s in steps if s.kind in ("tool_result", "state_probe")
    )
    unsupported_tokens = _VALUE_TOKEN.findall(final_text)
    unsupported_value_ratio = 1.0 - _value_ratio(unsupported_tokens, evidence_blob, default=1.0)

    instruction = view.task.instruction or ""
    instruction_tokens = _VALUE_TOKEN.findall(instruction)
    write_args_blob = " ".join(_text_blob(s.args) for s in write_calls)
    instruction_value_coverage = _value_ratio(instruction_tokens, write_args_blob, default=1.0)

    return {
        "n_steps": n_steps,
        "n_tool_calls": float(len(tool_calls)),
        "n_tool_results": float(len(tool_results)),
        "n_state_probes": float(len(probes)),
        "n_agent_messages": float(len(agent_messages)),
        "n_failed_results": float(n_failed_results),
        "failed_result_share": failed_result_share,
        "last_result_succeeded": last_result_succeeded,
        "failure_after_last_write": failure_after_last_write,
        "n_write_calls": float(len(write_calls)),
        "n_read_calls": float(len(read_calls)),
        "write_success_ratio": write_success_ratio,
        "last_write_succeeded": last_write_succeeded,
        "n_retries": float(_count_retries(steps)),
        "receipt_id_present": 1.0 if _receipt_id_present(write_results) else 0.0,
        "steps_after_last_write": steps_after_last_write,
        "probe_present": probe_present,
        "probe_succeeded": probe_succeeded,
        "probe_empty_or_not_found": probe_empty_or_not_found,
        "error_keyword_hits": error_keyword_hits,
        "pending_status_present": pending_status_present,
        "reviewer_phrase_present": reviewer_phrase_present,
        "n_claims": float(len(view.final_claim.claims)),
        "claims_extracted_from_text": 1.0 if view.claims_extracted else 0.0,
        "hedge_words_present": hedge_words_present,
        "final_text_length": float(len(final_text)),
        "unsupported_value_ratio": unsupported_value_ratio,
        "instruction_value_coverage": instruction_value_coverage,
    }


FEATURES: list[FeatureSpec] = [
    FeatureSpec("n_steps", "Total number of steps in the trace."),
    FeatureSpec("n_tool_calls", "Number of tool_call steps."),
    FeatureSpec("n_tool_results", "Number of tool_result steps."),
    FeatureSpec("n_state_probes", "Number of state_probe steps."),
    FeatureSpec("n_agent_messages", "Number of message steps with role agent."),
    FeatureSpec("n_failed_results", "Number of tool_result steps with ok false or an error."),
    FeatureSpec(
        "failed_result_share", "Share of tool_result steps that failed (0 when there are none)."
    ),
    FeatureSpec("last_result_succeeded", "1 when the last tool_result (by i) succeeded, else 0."),
    FeatureSpec(
        "failure_after_last_write",
        "1 when a failed tool_result occurs after the last successful write's result.",
    ),
    FeatureSpec("n_write_calls", "Number of tool_call steps whose name looks like a write."),
    FeatureSpec("n_read_calls", "Number of tool_call steps that are not write calls."),
    FeatureSpec(
        "write_success_ratio",
        "Share of write tool_result steps that succeeded (0 when there are none).",
    ),
    FeatureSpec("last_write_succeeded", "1 when the last write call's paired result succeeded."),
    FeatureSpec("n_retries", "Count of a tool called again right after a failed result for it."),
    FeatureSpec(
        "receipt_id_present",
        "1 when a write result's output has a non-empty id/_id/sha field.",
    ),
    FeatureSpec("steps_after_last_write", "Number of steps after the last write tool_call."),
    FeatureSpec("probe_present", "1 when the trace has at least one state_probe step."),
    FeatureSpec("probe_succeeded", "1 when the last state_probe (by i) succeeded."),
    FeatureSpec(
        "probe_empty_or_not_found",
        "1 when the last probe's output is empty or reads as not found.",
    ),
    FeatureSpec(
        "error_keyword_hits",
        "Count of tool_result steps whose output/error text has an error keyword.",
    ),
    FeatureSpec(
        "pending_status_present",
        "1 when any tool_result output mentions a pending/queued/accepted status.",
    ),
    FeatureSpec(
        "reviewer_phrase_present",
        "1 when a tool output or the final text has a reviewer-directed phrase.",
    ),
    FeatureSpec("n_claims", "Number of resolved claims on the final message."),
    FeatureSpec("claims_extracted_from_text", "1 when claims came from pattern extraction."),
    FeatureSpec("hedge_words_present", "1 when the final text contains a hedge word."),
    FeatureSpec("final_text_length", "Character length of the final message text."),
    FeatureSpec(
        "unsupported_value_ratio",
        "Share of numbers/dates/times in the final text found in no tool_result/probe output.",
    ),
    FeatureSpec(
        "instruction_value_coverage",
        "Share of numbers/dates/times in the instruction found in some write call's args.",
    ),
]
