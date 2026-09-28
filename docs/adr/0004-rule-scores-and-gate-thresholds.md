# 0004: rule outcome scores and gate thresholds, fixed before evaluation

## Status

Accepted.

## Context

The rule engine turns each claim into one of five outcomes:
`contradicted`, `unsupported`, `unknown`, `receipt_only` or
`probe_supported`. Each outcome needs a raw `p_success` so it can go through
the same gate as every other detector. The gate itself needs threshold
values that turn a probability into `verified`, `false_success` or
`unverifiable`. Both numbers must be picked before any detector is measured
against the benchmark, so the choice cannot be tuned to make the results
look better after the fact.

## Decision

Rule outcome scores (`RULE_SCORES`):

| outcome | p_success |
|---|---|
| contradicted | 0.03 |
| unsupported | 0.05 |
| unknown | abstain (reports the configured base rate) |
| receipt_only | 0.70 |
| probe_supported | 0.97 |

Gate thresholds (`Thresholds`, `gate.py`): `verified = 0.80`,
`false_success = 0.20`.

`receipt_only` is deliberately placed at 0.70, below the 0.80 verified
threshold. A tool accepting a call without independent confirmation (no
state probe) is evidence, not proof: not persisted, not applied to the
right record, or later rolled back are all still possible. A receipt alone
should reach a human queue unless a calibrator, fitted on real outcomes,
raises it.

## Consequences

- Every detector compared in the benchmark is judged by the same 0.80/0.20
  gate, so differences in the results reflect the detectors, not the gate.
- `receipt_only` traces are `unverifiable` by default under the rules
  detector alone; `cascade-offline` and `cascade` exist specifically to
  resolve some of them with a classifier or a judge instead of leaving every
  one of them to a human.
- These numbers are committed now, before the recorded benchmark run
  (ADR 0005), so they cannot be adjusted after seeing the results.
