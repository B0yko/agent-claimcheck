# Interop: agent-trace/v1 and other trace formats

agent-claimcheck reads and writes one shared format, `agent-trace/v1`, so a
trace produced by another tool (including the sibling portfolio projects
that emit this format) can be checked here without any glue code, and so a
trace written here — the benchmark itself, or a human review decision — can
be reused as a labelled example elsewhere.

## `agent-trace/v1` summary

JSON Lines, one trace object per line, published as
`schemas/agent-trace-v1.json`:

```json
{
  "schema": "agent-trace/v1",
  "trace_id": "string, unique",
  "source": "string, the producing tool and version, e.g. booking-truth/0.1.0",
  "task": {"id": "string", "domain": "booking | crm | coding | browser | other", "instruction": "string"},
  "steps": [
    {
      "i": 0,
      "ts": "RFC 3339 timestamp",
      "kind": "message | tool_call | tool_result | state_probe",
      "role": "user | agent | tool | environment",
      "name": "tool or probe name, null for messages",
      "content": "text for messages, null otherwise",
      "args": {},
      "ok": true,
      "output": {},
      "error": null
    }
  ],
  "final_claim": {
    "text": "the agent's final user-visible message",
    "claims": [{"type": "string, e.g. booked | updated | tests_passed | done", "subject": {}}]
  },
  "ground_truth": {
    "outcome": "success | failure | unknown",
    "checked_by": "state_probe | human | none",
    "details": {}
  },
  "meta": {}
}
```

`steps` are ordered by `i`. `ground_truth` may be omitted or
`{"outcome": "unknown", "checked_by": "none"}` for an unlabelled trace —
`agent-claimcheck check` still scores it, it just cannot enter any metric.
Unknown extra fields are read permissively wherever the shared spec allows
them (inside `args`, `output`, `subject`, `details` and `meta`); everywhere
this project *writes* a trace, the result validates against the published
schema.

### The pairing rule

A `tool_result` step refers to the nearest **preceding** `tool_call` step
with the same `name` (`schema.pair_results`). This is what lets the rules
detector and the classifier's write/read/retry features find "the call that
produced this result" without a call ever carrying an explicit result id: a
producer only has to emit `tool_call` and `tool_result` steps for the same
tool name in call-then-result order. A `tool_result` with no preceding call
of that name pairs to nothing (an orphan result) rather than raising.

### Probes

A probe file (`--probes probes.jsonl`, one JSON object per line: `trace_id`,
`name`, `args`, `ok`, `output`, `error`, `ts`) merges into the matching
traces as appended `state_probe` steps with `role: environment`; `i`
continues from each trace's own last step, in file order. A probe for an
unknown `trace_id` is skipped and reported as a warning, not an error — so a
probes file that covers only some of a batch's traces still merges cleanly.
This is the mechanism for bringing your own environment checks (a database
read, a second API call, a UI screenshot's OCR'd text) into a claim
evaluation the rule DSL or the classifier can use as a `state_probe`; see
`examples/probes.jsonl` for one worked example.

### Result schema: `claimcheck-result/v1`

Every result `agent-claimcheck check` writes (`--out results.jsonl`, or the
`jsonl`/`json` `--format`) validates against `schemas/claimcheck-result-v1.json`
before being written — `checker.dump_result` runs that validation itself, so
an invalid result is a bug, never a silent partial write. `docs/python-api.md`
documents `CheckResult`'s fields; the two schemas' `$defs` are the
authoritative shapes for a downstream consumer that wants to parse results
without importing this package.

## Human review as agent-trace/v1

The local dashboard, `agent-claimcheck serve [INPUT ...] [--results F]
[--reviews F] [--host 127.0.0.1] [--port 8765] [--max-usd 1.0]`, is one
producer and one consumer of this same format. Its review queue lets a
person resolve an `unverifiable` trace to Verified or False success; each
decision is appended to the reviews file (default `./claimcheck-reviews.jsonl`)
as a full `agent-trace/v1` line with `ground_truth.checked_by: "human"`. That
file needs no conversion to feed `agent-claimcheck train`: reviewed traces
are labelled agent-trace/v1 like any other.

## Bring your own tool names

Interop with your own agent's tool names goes through the rule DSL
(`docs/rules.md`), not through this format: write a rule pack whose `tools`
globs match your tool names (`examples/rules/custom.yaml` is a worked
example) and pass it with `--rules custom.yaml`. The trace format itself
does not need to know about any particular tool vocabulary.

## Mapping from other tracing platforms

Importing another platform's trace format is out of scope for v0.1 — no
importer ships, and none is planned for a `browser`-like adapter beyond what
`agent-trace/v1` already covers. What follows is a field mapping for
someone converting their own OpenTelemetry GenAI, Langfuse or LangSmith
traces into `agent-trace/v1` by hand (or with a small script of their own).
It names the closest corresponding field in each platform as of late 2026;
none of these platforms is a dependency of this project, and this project
reads and writes nothing from them directly.

| `agent-trace/v1` | OpenTelemetry GenAI semantic conventions | Langfuse | LangSmith |
|---|---|---|---|
| `step.kind: tool_call`, `step.name` | `gen_ai.tool.name` (on a tool span) | a `tool`-type observation's `name` | a `Run` with `run_type: "tool"`, its `name` |
| `step.args` (of a `tool_call`) | `gen_ai.tool.call.arguments` | that observation's `input` | that run's `inputs` |
| `step.output` (of the paired `tool_result`) | `gen_ai.tool.call.result` | that observation's `output` | that run's `outputs` |
| correlating a `tool_call` with its `tool_result` | `gen_ai.tool.call.id` | the observation's own id (one observation covers call and result) | the run's `id` |
| `step.kind: message`, `step.content` | `gen_ai.input.messages` / `gen_ai.output.messages` | a `generation`-type observation's `input`/`output` | a `run_type: "llm"` run's `inputs`/`outputs` |
| `task.instruction` (a system/task framing) | `gen_ai.system_instructions` | — (modelled as the first message, platform-dependent) | — (modelled as the first message, platform-dependent) |
| `trace_id` | the span's trace context | `traceId` | `trace_id` (runs nest via `parent_run_id`) |
| available tool schema (not part of one step) | `gen_ai.tool.definitions` | — | — |

Notes: `output`/`input` on a Langfuse `generation` observation additionally
carries `model` and `usageDetails.{input,output}` for token accounting, which
has no `agent-trace/v1` counterpart (this project's own cost accounting is
per judge call, in the ledger, not on the trace). A LangSmith `Run` also
carries `error`, which maps directly to `step.error`. None of this is a
one-to-one schema match — in particular, a single agent-trace/v1
`state_probe` step (an independent environment read, not a message or a
tool the agent itself called) has no direct counterpart in any of the three;
model it as a tool-type span/observation/run with a name that makes clear it
is an out-of-band check, e.g. `probe.<name>`.
