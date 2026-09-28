# Detectors

Reference material specific to one detector at a time. Each detector gets
its own top-level heading in this file.

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
