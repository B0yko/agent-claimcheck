# 0006: Platt scaling as the only v0.1 calibrator

## Status

Accepted.

## Context

Raw detector probabilities are not directly comparable: a rule outcome maps
to one of five fixed scores, a classifier's logistic output is already
somewhat calibrated, and a judge's self-reported probability tends to pin
near 0 or 1. Something has to map each detector's raw score onto an
actual probability of success before it is meaningful to gate on, compare
across detectors, or report calibration metrics (ECE, Brier) for.

## Decision

Use Platt scaling only: a one-dimensional logistic regression fitted on the
clipped logit of each detector's raw `p_success`, `p' = sigmoid(a * x + b)`,
fitted with Newton's method to convergence. Targets are smoothed
(`t+ = (N+ + 1)/(N+ + 2)`, `t- = 1/(N- + 2)`) so a perfectly separable
training split does not drive `a` to infinity. Calibrators are fitted on the
train split's non-abstaining outputs only and stored as a small JSON file
per detector. Abstaining outputs are never calibrated; they keep the
configured base rate.

No isotonic regression, no temperature scaling, no per-domain calibrators,
and no stacking of calibrators in v0.1.

## Consequences

- One simple, well-understood, two-parameter calibrator per detector, cheap
  to fit and to store, and easy to sanity-check by eye.
- Platt scaling assumes a roughly monotonic, sigmoid-shaped relationship
  between the raw score and the true probability; a detector whose errors
  are not monotonic in its raw score will calibrate poorly, and that
  limitation is reported rather than hidden.
- Isotonic calibration, which fits a weaker assumption, is left on the
  roadmap for when there is enough labelled data to justify its extra
  parameters.
