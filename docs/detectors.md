# Detectors

A detector scores a `DetectorView` (never a full `Trace`: `redact.detector_view` is the only
projection a detector ever sees; `ground_truth`, `meta` and `source` never reach it) and returns a
`DetectorOutput`: a raw `p_success`, an `abstain` flag and reason when it declines to score, a list
of `Reason`s citing evidence, and cost, latency and caching bookkeeping. Each detector has its own
section below; the rule engine's DSL is documented in `docs/rules.md`.

## Classifier

`classifier-lr` (`detectors/classifier.py`) scores a trace with a
standardised logistic regression fitted on the deterministic features below
(`features.py`). Inputs are generic: no tool name, domain name or
benchmark-specific string ever appears in a feature's definition, only step
kinds, roles, and shapes any producer of agent-trace/v1 already has. The
trained model is a plain JSON artifact (`models/lr-v1.json`), never a
pickle — see ADR 0003.

The detector abstains with `out_of_distribution` when the trace's
`task.domain` is not one of the artifact's `training_domains`, or the trace
has no `tool_result` step at all; `guard=False` disables this check (used to
measure leave-one-domain-out transfer). `details.contributions` lists the
ten features with the largest `|coefficient × standardised value|`, so a
reviewer can see what moved the score.

### Classifier features

| feature | description |
|---|---|
| `n_steps` | Total number of steps in the trace. |
| `n_tool_calls` | Number of tool_call steps. |
| `n_tool_results` | Number of tool_result steps. |
| `n_state_probes` | Number of state_probe steps. |
| `n_agent_messages` | Number of message steps with role agent. |
| `n_failed_results` | Number of tool_result steps with ok false or an error. |
| `failed_result_share` | Share of tool_result steps that failed (0 when there are none). |
| `last_result_succeeded` | 1 when the last tool_result (by i) succeeded, else 0. |
| `failure_after_last_write` | 1 when a failed tool_result occurs after the last successful write's result. |
| `n_write_calls` | Number of tool_call steps whose name looks like a write. |
| `n_read_calls` | Number of tool_call steps that are not write calls. |
| `write_success_ratio` | Share of write tool_result steps that succeeded (0 when there are none). |
| `last_write_succeeded` | 1 when the last write call's paired result succeeded. |
| `n_retries` | Count of a tool called again right after a failed result for it. |
| `receipt_id_present` | 1 when a write result's output has a non-empty id/_id/sha field. |
| `steps_after_last_write` | Number of steps after the last write tool_call. |
| `probe_present` | 1 when the trace has at least one state_probe step. |
| `probe_succeeded` | 1 when the last state_probe (by i) succeeded. |
| `probe_empty_or_not_found` | 1 when the last probe's output is empty or reads as not found. |
| `error_keyword_hits` | Count of tool_result steps whose output/error text has an error keyword. |
| `pending_status_present` | 1 when any tool_result output mentions a pending/queued/accepted status. |
| `reviewer_phrase_present` | 1 when a tool output or the final text has a reviewer-directed phrase. |
| `n_claims` | Number of resolved claims on the final message. |
| `claims_extracted_from_text` | 1 when claims came from pattern extraction. |
| `hedge_words_present` | 1 when the final text contains a hedge word. |
| `final_text_length` | Character length of the final message text. |
| `unsupported_value_ratio` | Share of numbers/dates/times in the final text found in no tool_result/probe output. |
| `instruction_value_coverage` | Share of numbers/dates/times in the instruction found in some write call's args. |

### Training

`train_lr(views, labels, domains, seed=0)` fits `StandardScaler` (population
variance; a zero-variance feature gets scale 1.0) followed by
`LogisticRegression(penalty="l2", solver="lbfgs", tol=1e-10, max_iter=10000)`,
with `C` chosen by 5-fold `StratifiedKFold(shuffle=True, random_state=seed)`
mean log-loss over `[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]`.
The target is success (1) versus failure (0). `oof_predictions(...)` reruns
the same C-selection and returns 5-fold out-of-fold `p_success` for the
`classifier-lr` calibrator. `fit_lodo(...)` is the same
fitting logic, meant to be called with only two domains' train traces, for
leave-one-domain-out evaluation with the guard off.

`agent-claimcheck train <labelled.jsonl|alias> --out DIR [--calibrate rules]`
retrains both the classifier artifact and its calibrator(s) on the caller's
own labelled traces (traces without a resolved ground truth, or without a
success claim, are skipped and the counts are reported).

## LLM judge (`detectors/judge.py`)

The judge asks a chat-completions model to audit the trace directly: it
renders the trace and the final claim into one prompt, asks for a
structured verdict, and turns the parsed answer into a `DetectorOutput`.
Registered as `"judge"`; an instance's detector key is `judge:<model>` for
the default prompt, or `judge:<model>:<prompt-name>` for any other prompt.

### Prompts

Two built-in prompts live under `judge/prompts/`, both `string.Template`
bodies with YAML front matter (`name`, `version`):

- **`claim-audit`** (the default) asks directly for a verdict.
- **`claim-by-claim`** first asks the model to list each claim with the
  step numbers that support or contradict it, then decide. Its answer
  carries an extra `claims` key that the parser accepts and ignores for
  scoring purposes; it is kept in `details.parsed.claims` for inspection.

A prompt body is rendered with exactly three placeholders:
`$instruction` (the task instruction), `$steps` (the rendered trace) and
`$claims` (the final message plus resolved claims, as JSON). A literal `$`
is written `$$`. A prompt that references any other `$name` fails to load
with a `PromptError` naming the placeholder.

**Custom prompts**: pass a filesystem path instead of a built-in name. The
file must have the same front matter and only use `$instruction`/`$steps`/
`$claims`; anything else is a `PromptError` at render time, not a silent
partial fill.

### Rendering the trace

Each step renders as one block: a `[i] kind role name` header, then
whichever of `args:`, `ok:`, `output:`, `error:`/`content:` the step
carries. When `output` is an object, each top-level field is serialized
and truncated independently at 2,000 characters (marker:
`…[truncated N chars]`), so one long field (a shell command's stdout, say)
never crowds the rest of the step out of the prompt.

Rendering is a pure function of the `DetectorView`: the same view and the
same prompt always produce byte-identical output. `request_sha256` is the
sha256 of the request body's canonical JSON (`sort_keys`, `ensure_ascii`
false, `(",", ":")` separators); `prompt_sha256` is the sha256 of the raw
prompt file. Both are recorded in `details` so a later `--from-recorded`
replay can detect a prompt or request drift and refuse instead of silently
comparing against stale numbers.

`ground_truth` and `meta` never reach a detector (they are not on
`DetectorView` at all), so they cannot leak into a rendered prompt or the
request body. Everything else in the trace — including any text inside a
message, a tool call's arguments, or a tool result's output or error that
reads like an instruction to a reviewer — is passed to the model labelled
as data to weigh, never as a command; both built-in prompts say so
explicitly.

### Output format and parsing (`judge/parse.py`)

The model is asked for one JSON object, optionally fenced in ``` ``` ``` or
` ```json `, with exactly:

```json
{
  "p_success": 0.9,
  "failure_kind": "none",
  "evidence_steps": [4, 7],
  "rationale": "the state probe confirms the event was created"
}
```

- `p_success`: a number in `[0, 1]`. A JSON boolean is rejected even though
  Python treats `bool` as an `int` subclass.
- `failure_kind`: one of `none`, `phantom_action`, `error_ignored`,
  `wrong_target`, `wrong_value`, `not_persisted`, `partial_completion`,
  `cannot_tell`.
- `evidence_steps`: a list of step indices. An index that does not appear
  in the trace does not fail parsing; it sets `invalid_citation: true` in
  `details` instead, since a judge citing a step that does not exist is
  itself useful signal about the judge's reliability.
- `rationale`: a string, truncated to 300 characters with
  `rationale_truncated: true` recorded when it was longer.

Anything else — unparseable text, a missing key, a wrong type, an
out-of-range number, an unknown `failure_kind`, a top-level JSON value that
is not an object — abstains with `abstain_reason: "parse_error"` rather
than guessing.

### Request (`judge/render.py`)

```json
{
  "model": "...",
  "messages": [{"role": "user", "content": "<rendered prompt>"}],
  "temperature": 0,
  "max_tokens": 400,
  "response_format": {"type": "json_object"}
}
```

No system message: the whole prompt, instructions included, is the one
user message. `response_format` is present only when `json_mode` is on
(the default).

**OpenRouter extras**: when, and only when, the configured `base_url`'s
host is exactly `openrouter.ai`, two extra fields are added:
`provider: {require_parameters: true}` whenever JSON mode is requested
(so OpenRouter only routes to a provider that actually honours
`response_format`), and `reasoning: {enabled: false}` for models whose
`/models` listing reports `reasoning` in `supported_parameters` (so a
reasoning-capable model does not spend budget on hidden reasoning tokens
for a task that needs a short structured answer). This decision is made
once, from whatever `PriceBook` the caller already has (see below), and
carried on `JudgeSpec.extra_body` as a plain dict — a later `--from-recorded`
replay reuses the recorded value and never needs a live `/models` lookup.

### Caching (`judge/cache.py`)

Each request is cached at
`<cache_dir>/judge/<sha256(model + "\n" + prompt_version + "\n" + request_json)>.json`,
written atomically (temp file, then `os.replace`). A cache hit sets
`cached: true`, `cost_usd: 0`, makes no HTTP call, and is excluded from
cost/latency statistics. `read_cache=False` skips reading a hit but still
writes one after a live call, so a forced re-run still leaves a fresh cache
entry for next time. `bench --live` never reads the cache (every trace gets
a fresh call) but still writes to it.

### Budget and the ledger (`ledger.py`)

`PriceBook` resolves a model's price: an explicit `--price-in`/
`--price-out` override first, then a configured price, then OpenRouter's
`/models` listing. `list_price` (the top-level list price) prices actual
settlement; `max_price` (the highest price across that model's live
`/models/{author}/{slug}/endpoints`) sizes the conservative pre-call
reservation. An unknown model raises `UnknownPriceError` — a live run never
silently prices a call at zero.

Before every live call, `Budget.reserve` holds a conservative estimate
(`reservation_usd`: request bytes ÷ 2.5 as a proxy token count for the
input, `max_tokens` for the output, priced at `max_price`, plus a 10%
margin) against the run's `max_usd` and, if a lifetime `CLAIMCHECK_LEDGER_CAP_USD`
cap is configured, against the ledger's running total plus this process's
outstanding reservations. `Budget` is thread-safe: the judge runs
concurrently, and reservations never let the committed total cross either
limit. A call that cannot be reserved abstains `budget_exhausted` or
`ledger_cap` without making an HTTP request. After the call, the
reservation is settled at the actual cost (`usage.cost` when the provider
returns it, else usage tokens priced at `list_price`) or released if the
call itself failed.

Every attempted call — success, failure, or cache hit — appends one line
to the ledger (JSONL, `CLAIMCHECK_LEDGER` or `<cache_dir>/ledger.jsonl`,
flushed and fsynced): timestamp, run id, model, trace id, prompt name,
token counts, reserved and actual cost, whether it was a cache hit, and a
status code. No trace content — no message, claim, or tool output — is
ever written to the ledger.

### Client and error taxonomy (`judge/client.py`)

A synchronous `httpx.Client` (HTTP/1.1, no HTTP/2), one POST to
`<base_url>/chat/completions`, up to two retries on `429` or a `5xx`
status (`4xx` is never retried). The retry delay honours a numeric
`Retry-After` header, clamped to 10 seconds, falling back to
`0.5 * 2^attempt` when the header is absent or not a number. Reported
latency includes any time spent waiting between retries.

Every failure is reported as one of a fixed set of codes —
`http_429`, `http_5xx`, `http_4xx`, `timeout`, `network`,
`invalid_response` — never as the underlying exception's message, the
response body, or its headers, in the `DetectorOutput`, the ledger, or any
log line. The full `abstain_reason` vocabulary a judge can report is:
`no_api_key`, `budget_exhausted`, `ledger_cap`, `parse_error`, and the six
client codes above.

**API key resolution**: `CLAIMCHECK_API_KEY`, else `OPENROUTER_API_KEY`
when, and only when, `base_url`'s host is exactly `openrouter.ai`. The
fallback key is never sent to any other host; with no key resolved at all,
the detector abstains `no_api_key` before attempting any HTTP call.

## Baselines (`detectors/baselines.py`)

Two detectors with no fitted or learned part, for comparison in the bench
report. Neither ever abstains, and neither is ever calibrated (a
`CalibratorSet` simply has no entry for either key).

- **`trust-agent`**: always `p_success = 1.0` — the agent's own claim,
  taken at face value.
- **`any-error`**: `p_success = 0.0` when any `tool_result` step has
  `ok: false` or a non-null `error`, else `1.0`.

## Ensembles (`detectors/ensemble.py`)

Both ensembles run the rules detector first and take its output directly
when the trace's worst claim outcome is conclusive — `contradicted`,
`unsupported` or `probe_supported` (the three raw scores `RULE_SCORES`
never assigns to an abstaining or `receipt_only` outcome). Otherwise a
second component decides. Either way, `details.decided_by` names the
component that won (`"rules"`, `"classifier-lr"` or `"judge"`), and the
raw/calibrated `p_success` reported is that component's own.

Unlike a single detector — which a `Checker` calibrates itself, after
`score()` returns — an ensemble calibrates each component it actually runs
*before* deciding (`is_ensemble = True` tells a `Checker` to skip its own
calibration step for these outputs), because the decision and the
calibrated value it reports both belong to the deciding component, not to
the ensemble as a whole.

- **`cascade-offline`** (the default detector for `check`, and free):
  falls through to `classifier-lr` when rules is inconclusive.
- **`cascade`**: falls through to the configured LLM judge instead.
  `details.sent_to_judge` records whether this trace reached the judge, so
  a bench report can compute the share that did. `concurrent = True`: a
  `Checker` scores traces for it on a thread pool, since the judge
  component may make a network call.

Both report `cost_usd`/`latency_ms` as the sum of every component they
actually ran (just the rules component when rules decided; rules plus the
second component otherwise).
