# Examples

Everything here is hand-written, fictional (`example.com`/`example.test`
people, addresses and companies) and released under Apache-2.0, same as the
code. It is meant to be read, not just run: each file demonstrates one
integration point in isolation, before you reach for the much larger
synthetic benchmark (`benchmark/v1/`, `docs/benchmark-tools.md`).

## `traces.jsonl` — the `example:mixed` alias

Twelve traces, four per domain (`booking`, `crm`, `coding`), one of each
verdict `cascade-offline` produces: `verified`, `false_success`,
`unverifiable` and `skipped` per domain. Good for a first look at what a
`check` run actually reports, and for CI smoke-testing that the CLI still
does what it says (`--fail-on false_success` should exit 1 on this file; exit 2
means the input itself was bad, for example a line that fails validation, see
`docs/interop.md`).

```sh
agent-claimcheck check example:mixed
```

## `probes.jsonl` — bringing your own environment checks

One state probe (`calendar.get_event` for `booking-01`), in the shape
`--probes` expects: `trace_id`, `name`, `args`, `ok`, `output`, `error`,
`ts`, one per line. Merged in, it upgrades `booking-01`'s `booked` claim
from `receipt_only` (the write call succeeded, nothing independently
confirms it) to `probe_supported` (the probe reads back a confirmed,
matching event) — compare the two runs below.

```sh
agent-claimcheck check examples/traces.jsonl
agent-claimcheck check examples/traces.jsonl --probes examples/probes.jsonl
```

## `rules/custom.yaml` — mapping your own tool names

A rule pack for an invented `acme.*` scheduling API, written the same way
`src/agent_claimcheck/rules/packs/booking.yaml` maps the built-in
`calendar.*` tools — see `docs/rules.md` for the DSL it uses. It has no
effect on `traces.jsonl` or `browser-demo.jsonl` (neither calls an `acme.*`
tool); it is a template for your own traces:

```sh
agent-claimcheck check your-traces.jsonl --rules examples/rules/custom.yaml
```

## `browser-demo.jsonl` — the `example:browser` alias

Twenty-four traces converted from a browser-automation agent, domain
`browser`. No built-in rule pack covers `browser.*` tool names, so every
claim's rules outcome is `unknown` and the rules detector abstains on all
24 — this is deliberate: it demonstrates a domain with no rule pack
abstaining rather than guessing, and traces with no success claim
(`agent_status: blocked`) being skipped rather than scored.

```sh
agent-claimcheck check example:browser
```

## Trying the dashboard on this data

`agent-claimcheck serve [INPUT ...] [--results F] [--reviews F] [--host
127.0.0.1] [--port 8765] [--max-usd 1.0]` starts a local, localhost-only
review dashboard against any of the inputs above (or the packaged benchmark
splits):

```sh
agent-claimcheck serve example:mixed
```

Open the printed `http://127.0.0.1:8765` URL, and the review queue shows
this file's `unverifiable` trace with its claims, step timeline and rule
evidence. A Verified/False success decision you make there is appended to
`./claimcheck-reviews.jsonl` as an `agent-trace/v1` line
(`ground_truth.checked_by: "human"`), which `agent-claimcheck train` reads
directly — see `docs/interop.md`.

## Validating any of these files directly

```sh
agent-claimcheck validate examples/traces.jsonl --probes examples/probes.jsonl
agent-claimcheck validate examples/browser-demo.jsonl
```
