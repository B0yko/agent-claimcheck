# Python API

```python
from agent_claimcheck import load_traces, Checker

checker = Checker(detector="cascade-offline")  # or Checker.from_config("claimcheck.toml")
for r in checker.check(load_traces("traces.jsonl"), probes="probes.jsonl"):
    print(r.trace_id, r.verdict, r.p_success, r.confidence, r.reasons[0].detail)
```

## Loading traces

`load_traces(path_or_alias, *, strict=False, probes=None) -> list[Trace]` reads an
`agent-trace/v1` JSON Lines file (or a packaged alias: `bench:train`, `bench:test`,
`example:mixed`, `example:browser`), validating every line. A bad line is skipped by
default; `strict=True` raises `TraceValidationError` on the first one instead.
`load_traces_report` (in `agent_claimcheck.schema`) returns the same traces plus every
line's errors and probe-merge warnings, for callers that want to keep going.

## `Checker`

```python
Checker(
    detector="cascade-offline",  # rules | classifier | judge | cascade-offline | cascade | ...
    config=None,  # an agent_claimcheck.config.Config, or Checker.from_config(path)
    rules=(),  # extra rule pack paths, on top of the built-in packs
    prompt=None,  # a judge/cascade prompt: a built-in name or a file path
    calibration=None,  # a CalibratorSet, a path to one, or None for the built-in set
    thresholds=None,  # a gate.Thresholds, or None for the config's/defaults'
    max_usd=None,  # live judge budget cap; None uses the config's
    concurrency=None,  # thread pool size for a concurrent detector; None uses the config's
)
```

`Checker.from_config(path)` reads a `claimcheck.toml` file and builds a `Checker` from
it (see `docs/rules.md`/`docs/detectors.md` for what each config section controls).

`checker.check(traces, probes=None) -> Iterator[CheckResult]` scores `traces` in the
order given. Each trace is redacted to a `DetectorView`, its claims resolved (structured
claims win; otherwise pattern extraction runs), and — when it has no success claim —
yielded as a `skipped` result without ever reaching the detector. A concurrent detector
(the judge, or an ensemble built on one) scores traces on a thread pool sized by
`concurrency`; every other detector scores them one at a time. `probes`, when given, is
merged into `traces` first, exactly as `load_traces(..., probes=...)` would.

## `CheckResult`

One per trace: `trace_id`, `domain`, `detector`, `verdict` (`verified`, `false_success`,
`unverifiable` or `skipped`), `p_success` (the gate's value — calibrated when a
calibrator exists, else raw; `None` only for `skipped`), `p_success_raw`, `calibrated`,
`abstain`, `abstain_reason`, `confidence`, `reasons` (at least one, citing evidence),
`claims` (every resolved claim, not only the success ones), `detectors` (the raw
`DetectorOutput`(s) behind the verdict), `cost_usd`, `latency_ms`, `cached`.
`agent_claimcheck.checker.dump_result(result)` serializes one result to a canonical
JSON line, validated against `schemas/claimcheck-result-v1.json` first; every line a
`check` run writes is validated this way.

## Extending

```python
from agent_claimcheck import Detector, DetectorOutput, register_detector


class MyDetector:
    name = "my-detector"

    def score(self, trace) -> DetectorOutput: ...


register_detector("my-detector", lambda **cfg: MyDetector())
```

`Detector` is a `Protocol`: `name: str` plus `score(self, trace: DetectorView) ->
DetectorOutput`, and it must be safe to call from several threads at once if it sets
`concurrent = True`. `trace` is already redacted (`ground_truth`, `meta` and `source`
never reach a detector) and has resolved claims; see `docs/detectors.md` for the shape
of `DetectorOutput` and the built-in detectors.

`load_rule_pack(path) -> RulePack` loads and validates one YAML rule pack (see
`docs/rules.md` for its DSL), for building a `Checker` with `rules=[...]`  or for
inspecting a pack directly.

## Metrics

```python
from agent_claimcheck import metrics

evaluation = metrics.evaluate(results, traces)
```

`results` is one detector's `CheckResult`s (from one `checker.check(...)` run) and
`traces` are the `Trace`s they were computed from (for their `ground_truth`). `skipped`
results, and results whose trace has no usable ground truth, are excluded. The returned
dict has `n`, `n_success`, `n_failure`, `auroc` (`{value, ci_low, ci_high}`, or `None`
when one class is absent), `ece_raw`/`ece_calibrated`, `brier_raw`/`brier_calibrated`,
`extremes_raw`/`extremes_calibrated`, and `decisions` (coverage, accuracy on decided,
caught, missed, false alarms, sent to review, and a `success`/`failure` ×
`verified`/`false_success`/`unverifiable` confusion table). The module also exports the
individual metric functions (`auroc`, `bootstrap_ci`, `ece`, `brier`, `extremes_share`,
`decision_stats`) for computing any one of them directly.
