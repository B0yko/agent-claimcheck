300 traces (180 train / 120 test), 120 false successes (48 in test), 3 domains, 7 false-success kinds, seed 20260924, test sha256 `5300042deee1`.

Recorded run: `agent-claimcheck bench --live --judges deepseek/deepseek-v4-flash,qwen/qwen3-235b-a22b-2507,mistralai/mistral-small-3.2-24b-instruct --ablation-judge mistralai/mistral-small-3.2-24b-instruct --ablation-prompt claim-by-claim --run-name v0.1.0 --max-usd 8.0 --concurrency 8 --out results/v0.1.0` on 2026-09-28, MacBook Air M5, 24 GB, concurrency 8, 1200 judge calls, total spend $0.1789.

| judge | price in $/M | price out $/M | price date |
|---|---|---|---|
| deepseek/deepseek-v4-flash | $0.072 | $0.143 | 2026-09-28 |
| mistralai/mistral-small-3.2-24b-instruct | $0.094 | $0.250 | 2026-09-28 |
| qwen/qwen3-235b-a22b-2507 | $0.087 | $0.350 | 2026-09-28 |

### Table A: discrimination and calibration (test split)

| detector | AUROC [95% CI] | ECE raw | ECE calibrated | Brier calibrated | extremes (raw) |
|---|---|---|---|---|---|
| trust-agent | 0.500 [0.500, 0.500] | 0.400 | 0.400 | 0.400 | 100.0% |
| any-error | 0.427 [0.354, 0.500] | 0.525 | 0.525 | 0.525 | 100.0% |
| rules | 0.910 [0.848, 0.963] | 0.057 | 0.070 | 0.090 | 55.8% |
| classifier-lr | 0.947 [0.891, 0.986] | 0.056 | 0.083 | 0.082 | 56.7% |
| cascade-offline | 0.953 [0.904, 0.989] | 0.060 | 0.049 | 0.053 | 73.3% |
| judge:deepseek/deepseek-v4-flash | 0.840 [0.769, 0.905] | 0.192 | 0.035 | 0.144 | 60.8% |
| judge:mistralai/mistral-small-3.2-24b-instruct | 0.919 [0.862, 0.970] | 0.054 | 0.090 | 0.115 | 52.5% |
| judge:qwen/qwen3-235b-a22b-2507 | 0.911 [0.855, 0.964] | 0.123 | 0.033 | 0.080 | 86.7% |
| cascade | 0.928 [0.870, 0.978] | 0.065 | 0.039 | 0.065 | 82.5% |

### Table B: decisions after the shared gate (test split)

| detector | coverage | accuracy (decided) | caught | missed | false alarms | sent to review | USD/1k | wall-clock s/1k | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| trust-agent | 100.0% | 60.0% | 0/48 | 48/48 | 0/72 | 0 | $0.000 | <0.01 | <0.01 | <0.01 |
| any-error | 100.0% | 47.5% | 9/48 | 39/48 | 24/72 | 0 | $0.000 | <0.01 | <0.01 | <0.01 |
| rules | 70.0% | 95.2% | 36/48 | 4/48 | 0/72 | 36 | $0.000 | 0.12 | 0.08 | 0.29 |
| classifier-lr | 67.5% | 97.5% | 32/48 | 2/48 | 0/72 | 39 | $0.000 | 0.17 | 0.11 | 0.25 |
| cascade-offline | 90.8% | 96.3% | 41/48 | 4/48 | 0/72 | 11 | $0.000 | 0.18 | 0.11 | 0.46 |
| judge:deepseek/deepseek-v4-flash | 41.7% | 90.0% | 0/48 | 5/48 | 0/72 | 70 | $0.059 | 345 | 2157 | 3079 |
| judge:mistralai/mistral-small-3.2-24b-instruct | 52.5% | 96.8% | 26/48 | 2/48 | 0/72 | 57 | $0.146 | 343 | 2606 | 3542 |
| judge:qwen/qwen3-235b-a22b-2507 | 89.2% | 92.5% | 31/48 | 8/48 | 0/72 | 13 | $0.211 | 404 | 3101 | 5804 |
| cascade | 96.7% | 94.0% | 40/48 | 7/48 | 0/72 | 4 | $0.065 | 121 | 0.11 | 3580 |

### Recall by false-success kind (caught/total)

| kind | rules | classifier-lr | cascade-offline | judge:deepseek/deepseek-v4-flash | judge:qwen/qwen3-235b-a22b-2507 | judge:mistralai/mistral-small-3.2-24b-instruct | cascade |
|---|---|---|---|---|---|---|---|
| phantom_action | 7/7 | 5/7 | 7/7 | 0/7 | 7/7 | 5/7 | 7/7 |
| error_ignored | 7/7 | 6/7 | 7/7 | 0/7 | 5/7 | 4/7 | 7/7 |
| wrong_target | 2/7 | 3/7 | 3/7 | 0/7 | 6/7 | 5/7 | 4/7 |
| wrong_value | 4/7 | 2/7 | 5/7 | 0/7 | 2/7 | 2/7 | 5/7 |
| not_persisted | 3/7 | 5/7 | 6/7 | 0/7 | 4/7 | 4/7 | 4/7 |
| partial_completion | 7/7 | 5/7 | 7/7 | 0/7 | 1/7 | 1/7 | 7/7 |
| reviewer_injection | 6/6 | 6/6 | 6/6 | 0/6 | 6/6 | 5/6 | 6/6 |

### Evidence breakdown: state probe vs. receipt only

| detector | state_probe AUROC | state_probe missed | receipt_only AUROC | receipt_only missed |
|---|---|---|---|---|
| rules | 0.934 [0.867, 0.984] | 4 | 0.750 [0.625, 0.875] | 0 |
| classifier-lr | 0.936 [0.860, 0.989] | 2 | 0.968 [0.910, 1.000] | 0 |
| cascade-offline | 0.936 [0.875, 0.985] | 4 | 0.985 [0.950, 1.000] | 0 |
| judge:deepseek/deepseek-v4-flash | 0.838 [0.758, 0.915] | 2 | 0.830 [0.690, 0.941] | 3 |
| judge:qwen/qwen3-235b-a22b-2507 | 0.920 [0.853, 0.984] | 5 | 0.887 [0.772, 0.990] | 3 |
| judge:mistralai/mistral-small-3.2-24b-instruct | 0.952 [0.900, 0.991] | 1 | 0.841 [0.694, 0.953] | 1 |
| cascade | 0.934 [0.867, 0.984] | 4 | 0.891 [0.775, 0.995] | 3 |

![reliability](results/v0.1.0/reliability.svg)
![histogram](results/v0.1.0/histogram.svg)

### Leave-one-domain-out (classifier-lr)

| held-out domain | LODO AUROC | LODO ECE | shipped classifier-lr AUROC |
|---|---|---|---|
| booking | 0.948 [0.854, 1.000] | 0.128 | 0.919 [0.786, 1.000] |
| crm | 0.969 [0.909, 1.000] | 0.154 | 0.966 [0.898, 1.000] |
| coding | 0.948 [0.862, 1.000] | 0.094 | 0.969 [0.909, 1.000] |

Leakage audit (final-message-only baseline, test split): **0.434 AUROC**.

### Prompt ablation: claim-audit vs. claim-by-claim (cheapest judge)

Model: `mistralai/mistral-small-3.2-24b-instruct`.

| prompt | AUROC | ECE calibrated | USD/1k |
|---|---|---|---|
| claim-audit | 0.919 [0.862, 0.970] | 0.090 | $0.146 |
| claim-by-claim | 0.900 [0.836, 0.958] | 0.114 | $0.179 |

### Parse-error and abstention rates, and the share sent to the judge

| judge | parse-error rate (test) | abstain rate (test) | parse errors (all calls) | abstentions (all calls) |
|---|---|---|---|---|
| judge:deepseek/deepseek-v4-flash | 0.0% | 0.0% | 0/300 | 0/300 |
| judge:mistralai/mistral-small-3.2-24b-instruct | 0.0% | 0.0% | 0/300 | 0/300 |
| judge:mistralai/mistral-small-3.2-24b-instruct:claim-by-claim | 0.0% | 0.0% | 0/300 | 0/300 |
| judge:qwen/qwen3-235b-a22b-2507 | 0.0% | 0.0% | 2/300 | 2/300 |
cascade share sent to the judge: 30.0%.

### Confusion matrices

<details><summary>trust-agent</summary>

| actual \ verdict | verified | false_success | unverifiable |
|---|---|---|---|
| success | 72 | 0 | 0 |
| failure | 48 | 0 | 0 |

</details>

<details><summary>any-error</summary>

| actual \ verdict | verified | false_success | unverifiable |
|---|---|---|---|
| success | 48 | 24 | 0 |
| failure | 39 | 9 | 0 |

</details>

<details><summary>rules</summary>

| actual \ verdict | verified | false_success | unverifiable |
|---|---|---|---|
| success | 44 | 0 | 28 |
| failure | 4 | 36 | 8 |

</details>

<details><summary>classifier-lr</summary>

| actual \ verdict | verified | false_success | unverifiable |
|---|---|---|---|
| success | 47 | 0 | 25 |
| failure | 2 | 32 | 14 |

</details>

<details><summary>cascade-offline</summary>

| actual \ verdict | verified | false_success | unverifiable |
|---|---|---|---|
| success | 64 | 0 | 8 |
| failure | 4 | 41 | 3 |

</details>

<details><summary>judge:deepseek/deepseek-v4-flash</summary>

| actual \ verdict | verified | false_success | unverifiable |
|---|---|---|---|
| success | 45 | 0 | 27 |
| failure | 5 | 0 | 43 |

</details>

<details><summary>judge:mistralai/mistral-small-3.2-24b-instruct</summary>

| actual \ verdict | verified | false_success | unverifiable |
|---|---|---|---|
| success | 35 | 0 | 37 |
| failure | 2 | 26 | 20 |

</details>

<details><summary>judge:qwen/qwen3-235b-a22b-2507</summary>

| actual \ verdict | verified | false_success | unverifiable |
|---|---|---|---|
| success | 68 | 0 | 4 |
| failure | 8 | 31 | 9 |

</details>

<details><summary>cascade</summary>

| actual \ verdict | verified | false_success | unverifiable |
|---|---|---|---|
| success | 69 | 0 | 3 |
| failure | 7 | 40 | 1 |

</details>

Data sources and licences: every trace comes from the in-repo synthetic generator (Apache-2.0); the two hand-written example sets are separately authored and released under the same licence. Nothing here comes from a real system.

### Hypotheses

- H1 (rules: fewest missed, lowest coverage): **not supported**: fewest missed = classifier-lr (2/48) vs rules 4/48; lowest coverage = judge:deepseek/deepseek-v4-flash 41.7% (rules 70.0%).
- H2 (raw judge extremes + Platt lowers ECE): **not supported**: judge:deepseek/deepseek-v4-flash: extremes raw 60.8%, ECE raw 0.192 → calibrated 0.035; judge:mistralai/mistral-small-3.2-24b-instruct: extremes raw 52.5%, ECE raw 0.054 → calibrated 0.090; judge:qwen/qwen3-235b-a22b-2507: extremes raw 86.7%, ECE raw 0.123 → calibrated 0.033.
- H3 (classifier-lr loses AUROC LODO): **not supported**: booking LODO 0.948 vs shipped 0.919; crm LODO 0.969 vs shipped 0.966; coding LODO 0.948 vs shipped 0.969.
- H4 (a judge is fooled by reviewer-directed text): **not supported** (all 300 traces, raw outputs through the gate): judge:deepseek/deepseek-v4-flash: reviewer_injection recall 91.7% vs other-kind mean 82.4%; judge:mistralai/mistral-small-3.2-24b-instruct: reviewer_injection recall 83.3% vs other-kind mean 52.8%; judge:qwen/qwen3-235b-a22b-2507: reviewer_injection recall 83.3% vs other-kind mean 55.6%.
