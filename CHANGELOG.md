# Changelog

All notable changes to this project are documented in this file.

## [0.1.1] - 2026-09-28

Packaging release: the code, benchmark and recorded results are unchanged from 0.1.0.

### Added

- Published on PyPI (`pip install agent-claimcheck`); the package metadata now carries the README, with relative links rewritten to the tagged sources on GitHub, project URLs, keywords and classifiers.

### Changed

- The licence is declared as the SPDX expression `Apache-2.0`.
- CI and the publishing workflow use the current major versions of `actions/checkout`, `actions/setup-node` and `astral-sh/setup-uv`.

## [0.1.0] - 2026-09-28

Initial release.

### Added

- **Trace I/O**: load and validate `agent-trace/v1` JSON Lines files against
  the published `schemas/agent-trace-v1.json`, with per-line error reporting
  (`agent-claimcheck validate`), an optional `--probes` file merged as
  `state_probe` steps, and a redaction projection that keeps `ground_truth`
  and `meta` out of every detector's input.
- **Claim handling**: structured `final_claim.claims`, or pattern-based
  extraction over `final_claim.text` when none are present, with a negation
  guard and a configurable set of non-success claim types.
- **Rules detector**: a declarative, code-free YAML rule DSL (`exists`,
  `equals`, `in`, `contains`, `matches`, with typed normalisers and
  `{subject.*}`/`{args.*}` interpolation), three built-in domain packs
  (`booking`, `crm`, `coding`) plus a generic fallback, and support for
  user-supplied packs (`--rules`).
- **LLM judge detector**: any OpenAI-compatible `/chat/completions`
  endpoint, two built-in prompts (`claim-audit`, `claim-by-claim`) or a
  custom prompt file, strict JSON parsing that abstains rather than
  guesses, an on-disk response cache, and a cost ledger with a pre-call
  budget reservation and a lifetime spend cap.
- **Trained classifier**: a logistic-regression detector (`classifier-lr`)
  on 28 deterministic, domain-agnostic features, shipped as a JSON artifact
  (never a pickle), with out-of-distribution abstention and retraining on
  the user's own labelled traces (`agent-claimcheck train`).
- **Calibration**: Platt scaling fitted on non-abstaining raw scores,
  applied by the shared gate whenever a calibrator exists for a detector.
- **Baselines and ensembles**: `trust-agent`, `any-error`, and the
  `cascade-offline` (rules, then the classifier) and `cascade` (rules, then
  the judge) ensembles.
- **Shared gate**: one pure, configurable-threshold function turning any
  detector's output into `verified` / `false_success` / `unverifiable`, used
  by every detector and every ensemble alike.
- **CLI**: `validate`, `check` (with `--detector`, `--rules`, `--prompt`,
  `--calibration`, `--probes`, `--fail-on`, `--max-usd`, table/json/jsonl
  output), `train`, `dataset generate`/`dataset validate`, and `bench`
  (`--offline`, `--live`, `--from-recorded`, `--check-readme`).
- **Local dashboard** (`agent-claimcheck serve`): an overview, a review
  queue for `unverifiable` traces with a trace inspector, human review
  decisions written as `agent-trace/v1` (feeding straight into `train`), and
  an on-demand judge run over the queue with a cost estimate and a cancel
  button — localhost-only, with `Host`/`Origin`/content-type checks on
  every state-changing request.
- **Python API**: `load_traces`, `Checker`, `CheckResult`, `Detector`,
  `register_detector`, `load_rule_pack` and `metrics.evaluate`, documented
  in `docs/python-api.md`.
- **Benchmark**: a synthetic, seeded, regenerable 300-trace false-success
  benchmark across the three domains, with a leakage audit, and a
  reproducible offline/recorded benchmark report (`bench.json`, `bench.md`,
  reliability and histogram charts) with four pre-registered hypotheses.
- **Packaging**: installable via `uvx`/`pip`, with the schemas, examples and
  benchmark shipped as package data so the packaged aliases
  (`bench:train`, `bench:test`, `example:mixed`, `example:browser`,
  `recorded:v0.1.0`) work without a clone.
