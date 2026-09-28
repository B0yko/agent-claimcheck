# 0003: JSON model artifacts instead of pickle

## Status

Accepted.

## Context

The project ships a trained logistic regression classifier and fitted
calibrators, and lets users retrain both on their own labelled traces. Model
artifacts are read by `agent-claimcheck check` on every run, including
against untrusted or shared files, and are distributed as package data.

## Decision

Model and calibrator artifacts are plain JSON: feature names, means, scales,
coefficients, intercept and metadata for the classifier; `method`, `a`, `b`
and metadata for a Platt calibrator. No artifact is ever loaded with
`pickle`, `joblib.load` on an arbitrary path, or any other format that can
execute code on load. Inference reads the JSON and reconstructs the linear
model with numpy only.

## Consequences

- Loading a model artifact, including one a user downloaded from somewhere
  else, cannot execute arbitrary code; at worst it is malformed data that
  fails validation.
- Artifacts are human-readable and diffable in version control.
- Only model families that reduce to a small set of numeric parameters
  (here, standardised logistic regression and one-dimensional Platt
  scaling) fit this approach. A model that needed its full estimator object
  would need a different decision.
