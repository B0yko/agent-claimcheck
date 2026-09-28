# Contributing

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for the Python 3.12
environment and dependency locking.

```sh
uv sync
```

This creates `.venv` and installs the runtime and dev dependency groups from
`uv.lock`. Run everything below through `uv run` so it uses that
environment.

## Checks to run before opening a pull request

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run pytest -q
```

All four must pass locally; CI runs the same commands (plus, on `ubuntu`, a
job that regenerates `benchmark/v1` from its seed and diffs it against the
committed files) and will not merge otherwise. Default CI makes no network
calls: the judge detector's integration test starts a fake
`/chat/completions` server inside the test process. A live judge run against
a real API is a separate, manually invoked path (`agent-claimcheck check
--detector judge`, `bench --live`), never part of CI.

## Adding a rule pack

Rule packs are YAML, loaded with `yaml.safe_load` and validated before use —
see `docs/rules.md` for the full DSL reference and `examples/rules/custom.yaml`
for a worked example that maps an invented tool API. To add one of your own:

1. Write a pack with `pack`, `version`, `tools` (globs matched against your
   `tool_call` names) and `claims` (one rule per claim type: an `action`
   glob, `receipt` checks, and an optional `probe`/`probe_checks`).
2. Load it directly with `load_rule_pack(path)`, or pass its path with
   `Checker(rules=[path])` / the CLI's `--rules path.yaml`, on top of the
   three built-in domain packs.
3. If the pack is meant to ship as a new built-in (rather than a
   `--rules` addition users bring themselves), add the YAML file under
   `src/agent_claimcheck/rules/packs/`, add its name to
   `rules.engine._BUILTIN_PACK_NAMES`, and add it to the rule-engine tests
   (`tests/test_rules_engine.py`).

A pack never embeds a Python expression — there is no code-execution path
through a rule pack, by design (ADR 0002). Keep new packs that way: if a
check you need cannot be expressed with `exists`/`equals`/`in`/`contains`/
`matches`, that is a signal to extend the DSL itself (with its own tests and
an ADR), not to bypass it.

## Adding a detector

A detector is anything implementing the `Detector` protocol: a `name` and a
`score(self, trace: DetectorView) -> DetectorOutput` method (see
`docs/python-api.md` for the exact shapes, and `docs/detectors.md` for how
the built-in detectors use them). Register it with `register_detector(name,
factory)` so `Checker(name)` and `--detector name` can find it. A detector
must only look at the `DetectorView` it is given — never at a `Trace`
directly — since `ground_truth` and `meta` are not on that view at all; the
redaction test (`tests/test_redact.py`) puts a canary string into both and
asserts it reaches no detector input.

## Benchmark regeneration

`agent-claimcheck dataset generate --seed 20260924 --out benchmark/v1`
regenerates the committed benchmark byte for byte from the in-repo
generator (`src/agent_claimcheck/bench/generator/`); `agent-claimcheck
dataset validate` checks the regenerated files against their manifest,
counts and leakage audit. If you change the generator, regenerate and
re-validate, and make sure `benchmark/v1/manifest.json`'s checksums are
updated in the same commit.

**Never tune the generator, a rule pack or the classifier against the test
split** (`benchmark/v1/traces.test.jsonl`). The leakage audit
(`dataset validate`) enforces this mechanically during development: it
scores a TF-IDF-plus-logistic-regression baseline on the final message alone
with 5-fold cross-validation on the *train* split only, and fails when that
CV AUROC is above 0.65. The test-split AUROC for that same baseline is
computed once, after the generator is frozen, and reported as a limitation
in `benchmark/v1/DATASET_CARD.md` — it is not iterated on. If a change to
the generator, a rule pack, the classifier's features or its
hyperparameters was made by looking at test-split results, that result is
no longer honest; regenerate the evaluation from the train split's own
signal instead.

## Commit style

Conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `ci:`,
`refactor:`, ...), in English, describing one coherent change per commit.
