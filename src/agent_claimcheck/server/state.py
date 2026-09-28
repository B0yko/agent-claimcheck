"""The dashboard's in-memory view of traces and their check results.

Built once, at startup, from a `Checker("cascade-offline")` pass (or a
`--results` file) over the traces named on the command line, then updated in
place as the judge panel scores queued traces. `TraceRecord.view` is the same
redacted `DetectorView` a detector would see: no `ground_truth`, no `meta`.
The inspector payload built from it never renders either field, so a human
reviewer sees exactly what the detectors saw, not the answer key.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_claimcheck.checker import CheckResult
from agent_claimcheck.claims import ClaimExtractor
from agent_claimcheck.detectors.base import DetectorOutput
from agent_claimcheck.ledger import now_iso
from agent_claimcheck.redact import DetectorView, detector_view, resolve_claims
from agent_claimcheck.schema import GroundTruth, Trace, dump_trace

__all__ = [
    "TraceRecord",
    "TraceStore",
    "append_review",
    "build_record",
    "build_store",
    "inspector_payload",
]


@dataclass
class TraceRecord:
    """One trace, its redacted view, and its current check result.

    `rules_output`/`classifier_output`/`judge_output` are whichever of those
    components actually ran for this trace (pulled out of the scoring
    detector's `components`, e.g. `cascade-offline` only runs the classifier
    when the rules outcome was not conclusive). `judge_output` starts `None`
    and is filled in once the judge panel scores this trace.
    """

    trace: Trace
    view: DetectorView
    result: CheckResult
    rules_output: DetectorOutput | None = None
    classifier_output: DetectorOutput | None = None
    judge_output: DetectorOutput | None = None


def _find_component(
    result: CheckResult, *, exact: str | None, prefix: str | None
) -> DetectorOutput | None:
    if not result.detectors:
        return None
    top = result.detectors[0]
    candidates: list[DetectorOutput] = list(top.components) or [top]
    for candidate in candidates:
        if exact is not None and candidate.detector == exact:
            return candidate
        if prefix is not None and candidate.detector.startswith(prefix):
            return candidate
    return None


def build_record(trace: Trace, result: CheckResult, extractor: ClaimExtractor) -> TraceRecord:
    """Project `trace` to a `DetectorView` and pair it with its `result`."""
    view = resolve_claims(detector_view(trace), extractor)
    return TraceRecord(
        trace=trace,
        view=view,
        result=result,
        rules_output=_find_component(result, exact="rules", prefix=None),
        classifier_output=_find_component(result, exact="classifier-lr", prefix=None),
        judge_output=_find_component(result, exact=None, prefix="judge:"),
    )


def build_store(
    traces: Sequence[Trace],
    results_by_id: Mapping[str, CheckResult],
    extractor: ClaimExtractor,
) -> TraceStore:
    """Build a `TraceStore` for every trace that has a matching result."""
    records: dict[str, TraceRecord] = {}
    for trace in traces:
        result = results_by_id.get(trace.trace_id)
        if result is None:
            continue
        records[trace.trace_id] = build_record(trace, result, extractor)
    return TraceStore(records)


class TraceStore:
    """Thread-safe map of `trace_id` to `TraceRecord`, mutated as the judge
    panel scores queued traces.
    """

    def __init__(self, records: dict[str, TraceRecord]) -> None:
        self._lock = threading.Lock()
        self._records = records

    def get(self, trace_id: str) -> TraceRecord | None:
        with self._lock:
            return self._records.get(trace_id)

    def all(self) -> list[TraceRecord]:
        with self._lock:
            return list(self._records.values())

    def queue(self) -> list[TraceRecord]:
        """Currently-`unverifiable` traces, closest to the 0.5 fence first."""
        with self._lock:
            records = [r for r in self._records.values() if r.result.verdict == "unverifiable"]

        def distance(record: TraceRecord) -> float:
            p = record.result.p_success
            return abs((p if p is not None else 0.5) - 0.5)

        return sorted(records, key=distance)

    def overview(self) -> dict[str, dict[str, int]]:
        """Counts by verdict, then by domain."""
        with self._lock:
            records = list(self._records.values())
        counts: dict[str, dict[str, int]] = {}
        for record in records:
            by_domain = counts.setdefault(record.result.verdict, {})
            domain = record.trace.task.domain
            by_domain[domain] = by_domain.get(domain, 0) + 1
        return counts

    def apply_judge_result(
        self, trace_id: str, *, result: CheckResult, judge_output: DetectorOutput | None
    ) -> None:
        """Replace `trace_id`'s result (and judge output, if any) in place."""
        with self._lock:
            record = self._records.get(trace_id)
            if record is None:
                return
            record.result = result
            if judge_output is not None:
                record.judge_output = judge_output


def inspector_payload(record: TraceRecord) -> dict[str, Any]:
    """The `/api/trace` response: everything a human reviewer needs, and
    nothing from `ground_truth` or `meta`.
    """
    trace = record.trace
    steps = [
        {
            "i": s.i,
            "kind": s.kind,
            "role": s.role,
            "name": s.name,
            "args": s.args,
            "ok": s.ok,
            "output": s.output,
            "error": s.error,
            "content": s.content,
        }
        for s in trace.steps
    ]
    claims = [c.model_dump(mode="json") for c in record.view.final_claim.claims]

    rule_evidence: list[Any] = []
    if record.rules_output is not None:
        rule_evidence = list(record.rules_output.details.get("claims", []))

    judge: dict[str, Any] | None = None
    if record.judge_output is not None and not record.judge_output.abstain:
        parsed = record.judge_output.details.get("parsed", {})
        judge = {
            "model": record.judge_output.detector,
            "p_success": parsed.get("p_success", record.judge_output.p_success),
            "failure_kind": parsed.get("failure_kind"),
            "rationale": parsed.get("rationale"),
            "evidence_steps": parsed.get("evidence_steps", []),
            "invalid_citation": record.judge_output.details.get("invalid_citation", False),
        }

    contributions: list[Any] = []
    if record.classifier_output is not None:
        contributions = list(record.classifier_output.details.get("contributions", []))

    return {
        "trace_id": trace.trace_id,
        "domain": trace.task.domain,
        "instruction": trace.task.instruction,
        "final_message": trace.final_claim.text,
        "claims": claims,
        "steps": steps,
        "rule_evidence": rule_evidence,
        "judge": judge,
        "classifier_contributions": contributions,
        "verdict": record.result.verdict,
        "p_success": record.result.p_success,
        "reasons": [r.model_dump(mode="json") for r in record.result.reasons],
    }


def append_review(
    reviews_path: str | Path, record: TraceRecord, *, decision: str, note: str
) -> None:
    """Append `record.trace` to `reviews_path` as `agent-trace/v1`, with a
    human `ground_truth` recording this review. Validated (via `dump_trace`)
    before anything reaches disk.
    """
    ground_truth = GroundTruth(
        outcome="success" if decision == "verified" else "failure",
        checked_by="human",
        details={
            "note": note,
            "decision": decision,
            "previous_verdict": record.result.verdict,
            "reviewed_at": now_iso(),
        },
    )
    labelled = record.trace.model_copy(update={"ground_truth": ground_truth})
    line = dump_trace(labelled)

    path = Path(reviews_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
