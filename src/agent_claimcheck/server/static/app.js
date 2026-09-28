// The dashboard's client: fetch, state, SSE. Every template is cloned and
// filled through the DOM text API only -- no markup-injecting APIs and no
// dynamic code evaluation anywhere in this file or in render.js/format.js.

import {
  classifyStep,
  overviewDomains,
  sortByAbsValue,
  sortSteps,
} from "./render.js";
import {
  charsRemaining,
  formatCost,
  formatDomain,
  formatMs,
  formatP,
  formatPercent,
  formatVerdict,
  truncate,
} from "./format.js";

const NOTE_LIMIT = 1000;
const STEP_PREVIEW_LIMIT = 400;

const state = {
  selectedTraceId: null,
  selectedResult: null,
  judgeRunId: null,
  judgeEventSource: null,
};

function el(id) {
  return document.getElementById(id);
}

function clone(templateId) {
  const template = el(templateId);
  return template.content.cloneNode(true);
}

function setText(root, selector, text) {
  const node = root.querySelector(selector);
  if (node) node.textContent = text;
}

async function getJSON(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `request failed: ${response.status}`);
  return body;
}

async function postJSON(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const body = await response.json();
  if (!response.ok) {
    const error = new Error(body.error || `request failed: ${response.status}`);
    error.status = response.status;
    error.body = body;
    throw error;
  }
  return body;
}

// -- overview ----------------------------------------------------------

function renderOverview(overviewData) {
  const domains = overviewDomains(overviewData);
  const headRow = el("overview-head-row");
  while (headRow.children.length > 1) headRow.removeChild(headRow.lastChild);
  for (const domain of domains) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = formatDomain(domain);
    headRow.appendChild(th);
  }

  const body = el("overview-body");
  body.replaceChildren();
  const verdicts = ["verified", "false_success", "unverifiable", "skipped"];
  for (const verdict of verdicts) {
    if (!overviewData[verdict]) continue;
    const row = clone("overview-row-template");
    setText(row, ".overview-verdict", formatVerdict(verdict));
    const tr = row.querySelector(".overview-row");
    for (const domain of domains) {
      const td = document.createElement("td");
      td.textContent = String(overviewData[verdict][domain] || 0);
      tr.appendChild(td);
    }
    body.appendChild(row);
  }
}

// -- queue ---------------------------------------------------------------

function renderQueue(queue) {
  const list = el("queue-list");
  list.replaceChildren();
  el("queue-empty").hidden = queue.length > 0;
  for (const item of queue) {
    const row = clone("queue-row-template");
    setText(row, ".queue-trace-id", item.trace_id);
    setText(row, ".queue-domain", formatDomain(item.domain));
    setText(row, ".queue-p", formatP(item.p_success));
    setText(row, ".queue-reason", truncate(item.top_reason || "", 80));
    const button = row.querySelector(".queue-row-button");
    button.addEventListener("click", () => selectTrace(item.trace_id));
    list.appendChild(row);
  }
}

// -- inspector -------------------------------------------------------------

function renderSteps(steps) {
  const list = el("steps-list");
  list.replaceChildren();
  for (const step of sortSteps(steps)) {
    const row = clone("step-template");
    const li = row.querySelector(".step-row");
    li.id = `step-${step.i}`;
    li.classList.add(`step-${classifyStep(step)}`);
    setText(row, ".step-index", `[${step.i}]`);
    setText(row, ".step-kind", step.kind);
    setText(row, ".step-role", step.role);
    setText(row, ".step-name", step.name || "");
    setText(row, ".step-args", step.args ? truncate(JSON.stringify(step.args), STEP_PREVIEW_LIMIT) : "");
    setText(row, ".step-ok", step.ok === null || step.ok === undefined ? "" : String(step.ok));
    setText(
      row,
      ".step-output",
      step.output === null || step.output === undefined
        ? ""
        : truncate(JSON.stringify(step.output), STEP_PREVIEW_LIMIT)
    );
    setText(row, ".step-error", step.error ? truncate(JSON.stringify(step.error), STEP_PREVIEW_LIMIT) : "");
    setText(row, ".step-content", step.content || "");
    list.appendChild(row);
  }
}

function renderClaims(claims) {
  const list = el("claims-list");
  list.replaceChildren();
  for (const claim of claims) {
    const row = clone("claim-template");
    setText(row, ".claim-type", claim.type);
    setText(row, ".claim-source", claim.source);
    setText(row, ".claim-subject", JSON.stringify(claim.subject));
    list.appendChild(row);
  }
}

function renderReasons(reasons) {
  const list = el("reasons-list");
  list.replaceChildren();
  for (const reason of reasons) {
    const row = clone("reason-template");
    const link = row.querySelector(".reason-step-link");
    link.href = reason.step === null || reason.step === undefined ? "#" : `#step-${reason.step}`;
    setText(row, ".reason-claim", reason.claim || "");
    setText(row, ".reason-outcome", reason.outcome);
    setText(row, ".reason-detail", reason.detail);
    list.appendChild(row);
  }
}

function renderRuleEvidence(claims) {
  const list = el("rule-evidence-list");
  list.replaceChildren();
  for (const claim of claims) {
    const row = clone("rule-evidence-template");
    const link = row.querySelector(".rule-evidence-step-link");
    link.href = claim.step === null || claim.step === undefined ? "#" : `#step-${claim.step}`;
    setText(row, ".rule-evidence-claim", claim.claim);
    setText(row, ".rule-evidence-outcome", claim.outcome);
    setText(row, ".rule-evidence-detail", claim.detail);
    const checks = row.querySelector(".rule-evidence-checks");
    for (const check of claim.checks || []) {
      const checkRow = clone("check-template");
      setText(checkRow, ".check-path", check.path);
      setText(checkRow, ".check-op", check.op);
      setText(checkRow, ".check-result", check.result);
      checks.appendChild(checkRow);
    }
    list.appendChild(row);
  }
}

function renderJudge(judge) {
  const container = el("judge-output");
  container.replaceChildren();
  if (!judge) {
    const p = document.createElement("p");
    p.className = "empty-note";
    p.textContent = "No judge output yet.";
    container.appendChild(p);
    return;
  }
  const summary = document.createElement("p");
  summary.textContent =
    `${judge.model}: p=${formatP(judge.p_success)}, failure_kind=${judge.failure_kind}` +
    (judge.invalid_citation ? " (invalid citation)" : "");
  container.appendChild(summary);

  const rationale = document.createElement("p");
  rationale.textContent = judge.rationale || "";
  container.appendChild(rationale);

  const stepsList = document.createElement("p");
  stepsList.textContent = "Evidence steps: ";
  for (const stepIndex of judge.evidence_steps || []) {
    const link = clone("evidence-step-link-template").querySelector(".evidence-step-link");
    link.href = `#step-${stepIndex}`;
    link.textContent = `[${stepIndex}]`;
    stepsList.appendChild(link);
    stepsList.appendChild(document.createTextNode(" "));
  }
  container.appendChild(stepsList);
}

function renderContributions(contributions) {
  const list = el("contributions-list");
  list.replaceChildren();
  const sorted = sortByAbsValue(contributions, "value");
  const maxAbs = sorted.reduce((max, c) => Math.max(max, Math.abs(c.value)), 0) || 1;
  for (const contribution of sorted) {
    const row = clone("contribution-template");
    setText(row, ".contribution-feature", contribution.feature);
    setText(row, ".contribution-value", contribution.value.toFixed(3));
    const bar = row.querySelector(".contribution-bar");
    const pct = Math.min(100, Math.round((Math.abs(contribution.value) / maxAbs) * 100));
    bar.style.width = `${pct}%`;
    bar.classList.add(contribution.value >= 0 ? "bar-positive" : "bar-negative");
    list.appendChild(row);
  }
}

async function selectTrace(traceId) {
  state.selectedTraceId = traceId;
  const payload = await getJSON(`/api/trace?id=${encodeURIComponent(traceId)}`);
  state.selectedResult = payload;

  el("inspector-empty").hidden = true;
  el("inspector-content").hidden = false;
  el("inspector-trace-id").textContent = payload.trace_id;
  el("inspector-instruction").textContent = payload.instruction;
  el("inspector-verdict").textContent = formatVerdict(payload.verdict);
  el("inspector-p").textContent = formatPercent(payload.p_success);

  renderReasons(payload.reasons);
  renderClaims(payload.claims);
  renderSteps(payload.steps);
  renderRuleEvidence(payload.rule_evidence);
  renderJudge(payload.judge);
  renderContributions(payload.classifier_contributions);

  el("decision-save-btn").disabled = false;
  el("decision-status").textContent = "";
}

// -- decision panel --------------------------------------------------------

let selectedDecision = null;

function updateNoteCounter() {
  const remaining = charsRemaining(el("decision-note").value, NOTE_LIMIT);
  el("note-counter").textContent = `${remaining} characters left`;
}

function wireDecisionPanel() {
  el("decision-verified-btn").addEventListener("click", () => {
    selectedDecision = "verified";
    el("decision-status").textContent = "Decision: verified. Add a note and save.";
  });
  el("decision-false-btn").addEventListener("click", () => {
    selectedDecision = "false_success";
    el("decision-status").textContent = "Decision: false success. Add a note and save.";
  });
  el("decision-note").addEventListener("input", updateNoteCounter);
  updateNoteCounter();

  el("decision-save-btn").addEventListener("click", async () => {
    if (!state.selectedTraceId) {
      el("decision-status").textContent = "Select a trace first.";
      return;
    }
    if (!selectedDecision) {
      el("decision-status").textContent = "Choose Verified or False success first.";
      return;
    }
    try {
      await postJSON("/api/review", {
        trace_id: state.selectedTraceId,
        decision: selectedDecision,
        note: el("decision-note").value,
      });
      el("decision-status").textContent = "Saved.";
      el("decision-note").value = "";
      selectedDecision = null;
      updateNoteCounter();
      await refreshState();
    } catch (error) {
      el("decision-status").textContent = `Could not save: ${error.message}`;
    }
  });
}

// -- judge panel -----------------------------------------------------------

function stopJudgeEvents() {
  if (state.judgeEventSource) {
    state.judgeEventSource.close();
    state.judgeEventSource = null;
  }
}

function startJudgeEvents(runId) {
  stopJudgeEvents();
  const source = new EventSource(`/api/judge/events?run=${encodeURIComponent(runId)}`);
  state.judgeEventSource = source;
  const progress = el("judge-progress");
  progress.hidden = false;
  el("judge-cancel-btn").hidden = false;

  source.addEventListener("progress", (event) => {
    const data = JSON.parse(event.data);
    progress.max = data.total || 1;
    progress.value = data.done || 0;
    el("judge-cost").textContent = `Spent: ${formatCost(data.spent_usd)}`;
    el("judge-status").textContent = `${data.done}/${data.total} scored (last: ${data.trace_id} -> ${formatVerdict(data.verdict)})`;
  });
  source.addEventListener("end", async () => {
    stopJudgeEvents();
    el("judge-cancel-btn").hidden = true;
    el("judge-status").textContent = "Judge run finished.";
    await refreshState();
    if (state.selectedTraceId) await selectTrace(state.selectedTraceId);
  });
  source.onerror = () => {
    stopJudgeEvents();
  };
}

function wireJudgePanel(judgeConfigured) {
  const runButton = el("judge-run-btn");
  const disabledReason = el("judge-disabled-reason");
  if (!judgeConfigured) {
    runButton.disabled = true;
    disabledReason.hidden = false;
    disabledReason.textContent = "No judge model or API key configured for this server.";
  }

  runButton.addEventListener("click", async () => {
    try {
      const started = await postJSON("/api/judge/run", {});
      state.judgeRunId = started.run_id;
      startJudgeEvents(started.run_id);
    } catch (error) {
      if (error.status === 409 && error.body && error.body.run_id) {
        state.judgeRunId = error.body.run_id;
        startJudgeEvents(error.body.run_id);
      } else {
        el("judge-status").textContent = `Could not start: ${error.message}`;
      }
    }
  });

  el("judge-cancel-btn").addEventListener("click", async () => {
    if (!state.judgeRunId) return;
    await postJSON("/api/judge/cancel", { run_id: state.judgeRunId });
  });
}

async function refreshJudgeEstimate() {
  const estimate = await getJSON("/api/judge/estimate");
  el("judge-queued").textContent = String(estimate.queued);
  el("judge-worst-case").textContent = estimate.error
    ? estimate.error
    : formatCost(estimate.worst_case_usd);
}

// -- top-level state refresh -------------------------------------------

async function refreshState() {
  const data = await getJSON("/api/state");
  el("input-summary").textContent = `${data.inputs.join(", ")} (${data.total} traces)`;
  renderOverview(data.overview);
  renderQueue(data.queue);
  await refreshJudgeEstimate();
  return data;
}

async function main() {
  wireDecisionPanel();
  const data = await refreshState();
  wireJudgePanel(data.judge_configured);
}

main().catch((error) => {
  const status = el("decision-status");
  if (status) status.textContent = `Failed to load: ${error.message}`;
});
