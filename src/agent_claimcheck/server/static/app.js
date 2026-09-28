// The dashboard's client: fetch, state, SSE. Every template is cloned and
// filled through the DOM text API only -- no markup-injecting APIs and no
// dynamic code evaluation anywhere in this file or in render.js/format.js.

import {
  checkGlyph,
  checkTone,
  citationStatus,
  divergingBars,
  domainTotals,
  failureKindTone,
  hasStep,
  inspectorSections,
  meterGeometry,
  normalizeThresholds,
  outcomeTone,
  overviewCount,
  overviewDomains,
  sortSteps,
  stepAnchorId,
  stepBase,
  stepStatus,
  subjectPairs,
  verdictOrder,
  verdictTotals,
} from "./render.js";
import {
  charsRemaining,
  collapsePreview,
  expandLabel,
  formatBlock,
  formatCharsLeft,
  formatCost,
  formatDomain,
  formatEstimateError,
  formatInputSummary,
  formatP,
  formatSigned,
  formatSpend,
  formatValue,
  formatVerdict,
  hasContent,
  humanize,
} from "./format.js";

const NOTE_LIMIT = 1000;
const LOW_CHARS_LEFT = 100;
const WIDE_LAYOUT = "(min-width: 1100px)";

const state = {
  selectedTraceId: null,
  thresholds: normalizeThresholds(null),
  worstCaseUsd: null,
  judgeConfigured: false,
  judgeRunId: null,
  judgeEventSource: null,
  selectedDecision: null,
};

// -- small DOM helpers -----------------------------------------------------

function el(id) {
  return document.getElementById(id);
}

function clone(templateId) {
  return el(templateId).content.cloneNode(true);
}

function setText(root, selector, text) {
  const node = root.querySelector(selector);
  if (node) node.textContent = text;
}

// Fills the node with `text`, or removes it when there is nothing to show, so
// a null field never leaves an empty block behind.
function fill(root, selector, text) {
  const node = root.querySelector(selector);
  if (!node) return null;
  if (text === null || text === undefined || text === "") {
    node.remove();
    return null;
  }
  node.textContent = text;
  return node;
}

function setVisible(node, visible) {
  node.hidden = !visible;
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

// -- p_success meter ---------------------------------------------------------

function applyMeter(meter, p) {
  const geometry = meterGeometry(p, state.thresholds);
  meter.dataset.zone = geometry.zone;
  meter.querySelector(".pmeter-fill").style.width = `${geometry.fill}%`;
  meter.querySelector(".pmeter-mark-lo").style.left = `${geometry.lo}%`;
  meter.querySelector(".pmeter-mark-hi").style.left = `${geometry.hi}%`;
}

// -- header chips and overview -------------------------------------------------

function renderChips(overview) {
  const list = el("verdict-chips");
  list.replaceChildren();
  const totals = verdictTotals(overview);
  for (const verdict of verdictOrder(overview)) {
    const chip = clone("verdict-chip-template");
    chip.querySelector(".chip-verdict").dataset.verdict = verdict;
    setText(chip, ".chip-label", formatVerdict(verdict));
    setText(chip, ".chip-count", String(totals[verdict]));
    list.appendChild(chip);
  }
}

function numberCell(count) {
  const cell = document.createElement("td");
  cell.className = count === 0 ? "num is-zero" : "num";
  cell.textContent = String(count);
  return cell;
}

function headerCell(text) {
  const th = document.createElement("th");
  th.scope = "col";
  th.className = "num";
  th.textContent = text;
  return th;
}

function renderOverview(overview) {
  const domains = overviewDomains(overview);

  const headRow = el("overview-head-row");
  while (headRow.children.length > 1) headRow.removeChild(headRow.lastChild);
  for (const domain of domains) headRow.appendChild(headerCell(formatDomain(domain)));
  headRow.appendChild(headerCell("Total"));

  const body = el("overview-body");
  body.replaceChildren();
  const totals = verdictTotals(overview);
  for (const verdict of verdictOrder(overview)) {
    const row = clone("overview-row-template");
    const tr = row.querySelector(".overview-row");
    tr.dataset.verdict = verdict;
    setText(row, ".overview-label", formatVerdict(verdict));
    for (const domain of domains) tr.appendChild(numberCell(overviewCount(overview, verdict, domain)));
    const total = numberCell(totals[verdict]);
    total.classList.add("is-total");
    tr.appendChild(total);
    body.appendChild(row);
  }

  const footRow = el("overview-foot-row");
  while (footRow.children.length > 1) footRow.removeChild(footRow.lastChild);
  const byDomain = domainTotals(overview);
  let all = 0;
  for (const domain of domains) {
    footRow.appendChild(numberCell(byDomain[domain]));
    all += byDomain[domain];
  }
  const grand = numberCell(all);
  grand.classList.add("is-total");
  footRow.appendChild(grand);
}

// -- review queue ------------------------------------------------------------------

function markSelectedRow() {
  for (const button of el("queue-list").querySelectorAll(".queue-row-button")) {
    if (button.dataset.traceId === state.selectedTraceId) {
      button.setAttribute("aria-current", "true");
    } else {
      button.removeAttribute("aria-current");
    }
  }
}

function renderQueue(queue) {
  const list = el("queue-list");
  list.replaceChildren();
  el("queue-empty").hidden = queue.length > 0;
  el("queue-count").textContent = queue.length > 0 ? `${queue.length} waiting` : "";
  for (const item of queue) {
    const row = clone("queue-row-template");
    setText(row, ".queue-trace-id", item.trace_id);
    setText(row, ".queue-domain", formatDomain(item.domain));
    setText(row, ".queue-p", formatP(item.p_success));
    fill(row, ".queue-reason", item.top_reason);
    applyMeter(row.querySelector(".queue-meter"), item.p_success);
    const button = row.querySelector(".queue-row-button");
    button.dataset.traceId = item.trace_id;
    button.title = item.top_reason || item.trace_id;
    button.addEventListener("click", () => selectTrace(item.trace_id, { reveal: true }));
    list.appendChild(row);
  }
  markSelectedRow();
}

// -- step references ---------------------------------------------------------------

// "step N" link that scrolls to and highlights the step; a plain, marked
// label when the trace has no such step.
function stepRef(step, steps, label = `step ${step}`) {
  const exists = hasStep(steps, step);
  const node = clone(exists ? "step-link-template" : "step-missing-template").firstElementChild;
  node.textContent = label;
  if (exists) {
    node.href = `#${stepAnchorId(step)}`;
    node.dataset.step = String(step);
  }
  return node;
}

function appendStepRef(container, step, steps) {
  if (step === null || step === undefined) return;
  container.appendChild(stepRef(step, steps));
}

function focusStep(index) {
  const target = document.getElementById(stepAnchorId(index));
  if (!target) return;
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  target.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "center" });
  target.focus({ preventScroll: true });
  target.classList.remove("is-flashing");
  void target.offsetWidth; // restart the animation when the same step is hit twice
  target.classList.add("is-flashing");
}

function wireStepLinks() {
  document.addEventListener("click", (event) => {
    const link = event.target.closest("a.step-link");
    if (!link || !link.dataset.step) return;
    event.preventDefault();
    focusStep(Number(link.dataset.step));
  });
  document.addEventListener("animationend", (event) => {
    if (event.target.classList.contains("is-flashing")) event.target.classList.remove("is-flashing");
  });
}

// -- inspector sections --------------------------------------------------------------

function renderInstruction(payload) {
  const instruction = el("inspector-instruction");
  instruction.textContent = payload.instruction || "";
  setVisible(instruction, Boolean(payload.instruction));
  el("inspector-final-message").textContent = payload.final_message || "";
  setVisible(el("final-message"), Boolean(payload.final_message));
}

function renderReasons(reasons, steps) {
  const list = el("reasons-list");
  list.replaceChildren();
  for (const reason of reasons) {
    const row = clone("reason-template");
    const badge = row.querySelector(".reason-outcome");
    badge.textContent = humanize(reason.outcome);
    badge.dataset.tone = outcomeTone(reason.outcome);
    fill(row, ".reason-claim", reason.claim);
    fill(row, ".reason-detail", reason.detail);
    appendStepRef(row.querySelector(".reason-row"), reason.step, steps);
    list.appendChild(row);
  }
}

function renderClaims(claims) {
  const list = el("claims-list");
  list.replaceChildren();
  for (const claim of claims) {
    const row = clone("claim-template");
    setText(row, ".claim-type", claim.type);
    const source = fill(row, ".claim-source", claim.source);
    if (source) source.dataset.source = claim.source;
    const holder = row.querySelector(".claim-subject");
    const pairs = subjectPairs(claim.subject);
    if (pairs.length === 0) {
      holder.textContent = "no subject fields";
      holder.classList.add("is-empty");
    }
    for (const { key, value } of pairs) {
      const pair = clone("subject-pair-template");
      setText(pair, ".pair-key", key);
      setText(pair, ".pair-value", formatValue(value));
      holder.appendChild(pair);
    }
    list.appendChild(row);
  }
}

function renderRuleEvidence(items, steps) {
  const list = el("rule-evidence-list");
  list.replaceChildren();
  for (const item of items) {
    const card = clone("evidence-template");
    setText(card, ".evidence-claim", item.claim);
    card.querySelector(".evidence-card").dataset.tone = outcomeTone(item.outcome);
    setText(card, ".evidence-outcome", humanize(item.outcome));
    if (!item.missing_subject) card.querySelector(".evidence-nosubject").remove();
    fill(card, ".evidence-detail", item.detail);
    appendStepRef(card.querySelector(".evidence-head"), item.step, steps);

    const checks = item.checks || [];
    if (checks.length === 0) {
      card.querySelector(".checks-table").remove();
    } else {
      const body = card.querySelector(".checks-body");
      for (const check of checks) {
        const row = clone("check-template");
        row.querySelector(".check-row").dataset.result = check.result;
        setText(row, ".check-path", check.path);
        setText(row, ".check-op", check.op);
        setText(row, ".check-expected", formatValue(check.expected));
        setText(row, ".check-actual", formatValue(check.actual));
        const mark = row.querySelector(".check-mark");
        mark.textContent = checkGlyph(check.result);
        mark.dataset.tone = checkTone(check.result);
        mark.setAttribute("aria-label", check.result);
        body.appendChild(row);
      }
    }
    list.appendChild(card);
  }
}

function renderContributions(contributions) {
  const list = el("contributions-list");
  list.replaceChildren();
  for (const item of divergingBars(contributions)) {
    const row = clone("contribution-template");
    setText(row, ".contribution-feature", item.feature);
    setText(row, ".contribution-value", formatSigned(item.value, 3));
    const bar = row.querySelector(".diverge-bar");
    bar.dataset.direction = item.direction;
    bar.style.left = `${item.left}%`;
    bar.style.width = `${item.width}%`;
    list.appendChild(row);
  }
}

function renderJudge(judge, steps) {
  el("judge-model").textContent = judge.model || "";
  el("judge-p").textContent = formatP(judge.p_success);

  const kind = el("judge-kind");
  setVisible(kind, hasContent(judge.failure_kind));
  kind.textContent = humanize(judge.failure_kind || "");
  kind.dataset.tone = failureKindTone(judge.failure_kind);

  const rationale = el("judge-rationale");
  rationale.textContent = judge.rationale || "";
  setVisible(rationale, Boolean(judge.rationale));

  const links = el("judge-cited-links");
  links.replaceChildren();
  const cited = citationStatus(judge.evidence_steps, steps);
  for (const { step } of cited) links.appendChild(stepRef(step, steps, `step ${step}`));
  setVisible(el("judge-cited"), cited.length > 0);
  setVisible(el("judge-invalid"), Boolean(judge.invalid_citation));
}

// Args and outputs: short text as a plain block, long text collapsed in a
// native <details> that keeps the first few lines visible.
function appendTextBlock(container, text, kind) {
  const info = collapsePreview(text);
  if (!info.collapsible) {
    const block = clone("text-block-template");
    const pre = block.querySelector(".text-block");
    pre.textContent = text;
    pre.dataset.part = kind;
    container.appendChild(block);
    return;
  }
  const block = clone("collapse-block-template");
  block.querySelector(".collapse-block").dataset.part = kind;
  setText(block, ".collapse-preview", info.preview);
  setText(block, ".collapse-full", text);
  setText(block, ".toggle-closed", expandLabel(info));
  container.appendChild(block);
}

function renderSteps(steps) {
  const list = el("steps-list");
  list.replaceChildren();
  for (const step of sortSteps(steps)) {
    const row = clone("step-template");
    const item = row.querySelector(".step-row");
    const base = stepBase(step);
    const status = stepStatus(step);
    item.id = stepAnchorId(step.i);
    item.tabIndex = -1;
    if (status === "failed") item.classList.add("is-failed");

    setText(row, ".step-index", `[${step.i}]`);
    const kind = row.querySelector(".step-kind");
    kind.textContent = base === "unknown" ? step.kind : base;
    kind.dataset.kind = base;
    fill(row, ".step-role", step.role);
    fill(row, ".step-name", step.name);
    const badge = fill(row, ".step-status", status === null ? null : status === "ok" ? "OK" : "FAILED");
    if (badge) badge.dataset.status = status;

    fill(row, ".step-content", step.content);
    const blocks = row.querySelector(".step-blocks");
    if (hasContent(step.args)) appendTextBlock(blocks, formatBlock(step.args), "args");
    if (hasContent(step.output)) appendTextBlock(blocks, formatBlock(step.output), "output");
    if (blocks.childElementCount === 0) blocks.remove();
    fill(row, ".step-error", hasContent(step.error) ? formatBlock(step.error) : null);
    list.appendChild(row);
  }
  el("steps-count").textContent = `${steps.length} ${steps.length === 1 ? "step" : "steps"}`;
}

async function selectTrace(traceId, { reveal = false } = {}) {
  const changed = traceId !== state.selectedTraceId;
  state.selectedTraceId = traceId;
  markSelectedRow();
  let payload;
  try {
    payload = await getJSON(`/api/trace?id=${encodeURIComponent(traceId)}`);
  } catch (error) {
    const empty = el("inspector-empty");
    empty.textContent = `Could not load ${traceId}: ${error.message}`;
    empty.hidden = false;
    el("inspector-content").hidden = true;
    return;
  }
  if (state.selectedTraceId !== traceId) return; // a newer selection won the race

  el("inspector-empty").hidden = true;
  el("inspector-content").hidden = false;
  el("inspector-trace-id").textContent = payload.trace_id;
  el("inspector-domain").textContent = formatDomain(payload.domain);
  const verdict = el("inspector-verdict");
  verdict.textContent = formatVerdict(payload.verdict);
  verdict.dataset.verdict = payload.verdict;
  el("inspector-p").textContent = formatP(payload.p_success);
  applyMeter(el("inspector-meter"), payload.p_success);

  const sections = inspectorSections(payload);
  setVisible(el("sec-instruction"), sections.instruction);
  setVisible(el("sec-reasons"), sections.reasons);
  setVisible(el("sec-claims"), sections.claims);
  setVisible(el("sec-evidence"), sections.evidence);
  setVisible(el("sec-contributions"), sections.contributions);
  setVisible(el("sec-judge"), sections.judge);
  setVisible(el("sec-steps"), sections.steps);

  renderInstruction(payload);
  renderReasons(payload.reasons || [], payload.steps);
  renderClaims(payload.claims || []);
  renderRuleEvidence(payload.rule_evidence || [], payload.steps);
  renderContributions(payload.classifier_contributions || []);
  if (payload.judge) renderJudge(payload.judge, payload.steps);
  renderSteps(payload.steps || []);

  if (changed) resetDecision();
  updateSaveState();

  // Stacked layout: the inspector sits below the queue, so bring it into view.
  if (reveal && !window.matchMedia(WIDE_LAYOUT).matches) {
    el("inspector").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

// -- decision panel ------------------------------------------------------------------

function updateNoteCounter() {
  const remaining = charsRemaining(el("decision-note").value, NOTE_LIMIT);
  const counter = el("note-counter");
  counter.textContent = formatCharsLeft(remaining);
  counter.classList.toggle("is-low", remaining < LOW_CHARS_LEFT);
}

function updateSaveState() {
  el("decision-save-btn").disabled = !(state.selectedTraceId && state.selectedDecision);
  el("decision-verified-btn").setAttribute("aria-pressed", String(state.selectedDecision === "verified"));
  el("decision-false-btn").setAttribute("aria-pressed", String(state.selectedDecision === "false_success"));
}

function resetDecision() {
  state.selectedDecision = null;
  el("decision-note").value = "";
  el("decision-status").textContent = "";
  updateNoteCounter();
  updateSaveState();
}

function chooseDecision(decision) {
  state.selectedDecision = decision;
  el("decision-status").textContent = "";
  updateSaveState();
}

function wireDecisionPanel() {
  el("decision-verified-btn").addEventListener("click", () => chooseDecision("verified"));
  el("decision-false-btn").addEventListener("click", () => chooseDecision("false_success"));
  el("decision-note").addEventListener("input", updateNoteCounter);
  updateNoteCounter();

  el("decision-save-btn").addEventListener("click", async () => {
    if (!state.selectedTraceId || !state.selectedDecision) return;
    const status = el("decision-status");
    try {
      await postJSON("/api/review", {
        trace_id: state.selectedTraceId,
        decision: state.selectedDecision,
        note: el("decision-note").value,
      });
      resetDecision();
      status.textContent = "Saved.";
      await refreshState();
    } catch (error) {
      status.textContent = `Could not save: ${error.message}`;
    }
  });
}

// -- judge panel ---------------------------------------------------------------------

function setJudgeRunning(running) {
  el("judge-cancel-btn").hidden = !running;
  el("judge-run").hidden = !running && el("judge-progress").value === 0;
  el("judge-run-btn").disabled = running || !state.judgeConfigured;
}

function stopJudgeEvents() {
  if (state.judgeEventSource) {
    state.judgeEventSource.close();
    state.judgeEventSource = null;
  }
}

function updateCostMeter(spentUsd) {
  const meter = el("judge-cost-meter");
  const known = typeof state.worstCaseUsd === "number" && state.worstCaseUsd > 0;
  meter.hidden = !known;
  if (known) {
    meter.max = state.worstCaseUsd;
    meter.value = Math.min(spentUsd || 0, state.worstCaseUsd);
  }
  el("judge-cost").textContent = formatSpend(spentUsd, state.worstCaseUsd);
}

function startJudgeEvents(runId) {
  stopJudgeEvents();
  const source = new EventSource(`/api/judge/events?run=${encodeURIComponent(runId)}`);
  state.judgeEventSource = source;
  const progress = el("judge-progress");
  el("judge-run").hidden = false;
  setJudgeRunning(true);
  updateCostMeter(0);

  source.addEventListener("progress", (event) => {
    const data = JSON.parse(event.data);
    progress.max = data.total || 1;
    progress.value = data.done || 0;
    updateCostMeter(data.spent_usd);
    el("judge-status").textContent =
      `${data.done}/${data.total} scored · last ${data.trace_id}: ${formatVerdict(data.verdict)}`;
  });
  source.addEventListener("end", async () => {
    stopJudgeEvents();
    setJudgeRunning(false);
    el("judge-status").textContent = "Judge run finished.";
    await refreshState();
    if (state.selectedTraceId) await selectTrace(state.selectedTraceId);
  });
  source.onerror = () => {
    stopJudgeEvents();
    setJudgeRunning(false);
    el("judge-status").textContent = "Lost the connection to the judge run.";
  };
}

function wireJudgePanel(judgeConfigured) {
  state.judgeConfigured = judgeConfigured;
  const runButton = el("judge-run-btn");
  const disabledReason = el("judge-disabled-reason");
  if (!judgeConfigured) {
    runButton.disabled = true;
    disabledReason.hidden = false;
    disabledReason.textContent = "Needs a judge model and its API key; neither is configured for this server.";
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
    el("judge-status").textContent = "Cancelling...";
    try {
      await postJSON("/api/judge/cancel", { run_id: state.judgeRunId });
    } catch (error) {
      el("judge-status").textContent = `Could not cancel: ${error.message}`;
    }
  });
}

async function refreshJudgeEstimate() {
  const estimate = await getJSON("/api/judge/estimate");
  state.worstCaseUsd = typeof estimate.worst_case_usd === "number" ? estimate.worst_case_usd : null;
  el("judge-queued").textContent = String(estimate.queued);
  el("judge-worst-case").textContent = estimate.error
    ? formatEstimateError(estimate.error)
    : formatCost(estimate.worst_case_usd);
}

// -- top-level state refresh ---------------------------------------------------------

async function refreshState() {
  const data = await getJSON("/api/state");
  state.thresholds = normalizeThresholds(data.thresholds);
  el("input-summary").textContent = formatInputSummary(data.inputs, data.total);
  renderChips(data.overview);
  renderOverview(data.overview);
  renderQueue(data.queue);
  await refreshJudgeEstimate();
  return data;
}

async function main() {
  wireDecisionPanel();
  wireStepLinks();
  const data = await refreshState();
  wireJudgePanel(data.judge_configured);
  if (data.queue.length > 0 && !state.selectedTraceId) await selectTrace(data.queue[0].trace_id);
}

main().catch((error) => {
  const empty = el("inspector-empty");
  empty.textContent = `Failed to load: ${error.message}`;
  empty.hidden = false;
});
