// Pure data helpers for the dashboard's views: no DOM, importable from
// `node --test`. Turning these into markup is app.js's job, with
// `<template>` cloning and `textContent` only.

export function classifyStep(step) {
  if (!step) return "unknown";
  if (step.kind === "tool_call") return "call";
  if (step.kind === "tool_result") return stepFailed(step) ? "result-failed" : "result";
  if (step.kind === "state_probe") return stepFailed(step) ? "probe-failed" : "probe";
  if (step.kind === "message") return "message";
  return "unknown";
}

export function stepFailed(step) {
  if (!step) return false;
  return step.ok === false || (step.error !== null && step.error !== undefined);
}

export function sortSteps(steps) {
  return [...(steps || [])].sort((a, b) => a.i - b.i);
}

export function sortQueueByDistance(items) {
  const distance = (item) => Math.abs((item.p_success ?? 0.5) - 0.5);
  return [...(items || [])].sort((a, b) => distance(a) - distance(b));
}

export function sortByAbsValue(items, key = "value") {
  return [...(items || [])].sort((a, b) => Math.abs(b[key]) - Math.abs(a[key]));
}

export function verdictTotals(overview) {
  const totals = {};
  for (const verdict of Object.keys(overview || {})) {
    totals[verdict] = Object.values(overview[verdict]).reduce((sum, n) => sum + n, 0);
  }
  return totals;
}

export function overviewDomains(overview) {
  const set = new Set();
  for (const verdict of Object.keys(overview || {})) {
    for (const domain of Object.keys(overview[verdict])) set.add(domain);
  }
  return [...set].sort();
}

export function overviewCount(overview, verdict, domain) {
  const byDomain = (overview || {})[verdict];
  if (!byDomain) return 0;
  return byDomain[domain] || 0;
}

// -- verdicts and overview -------------------------------------------------

const VERDICT_ORDER = ["verified", "false_success", "unverifiable", "skipped"];

// Known verdicts in their fixed order, then any unexpected ones alphabetically;
// only those the overview actually has.
export function verdictOrder(overview) {
  const present = Object.keys(overview || {});
  const known = VERDICT_ORDER.filter((verdict) => present.includes(verdict));
  const extra = present.filter((verdict) => !VERDICT_ORDER.includes(verdict)).sort();
  return [...known, ...extra];
}

export function domainTotals(overview) {
  const totals = {};
  for (const domain of overviewDomains(overview)) {
    totals[domain] = Object.keys(overview).reduce(
      (sum, verdict) => sum + overviewCount(overview, verdict, domain),
      0
    );
  }
  return totals;
}

// -- p_success meter -------------------------------------------------------

export const DEFAULT_THRESHOLDS = { false_success: 0.2, verified: 0.8 };

export function normalizeThresholds(thresholds) {
  const lo = thresholds && thresholds.false_success;
  const hi = thresholds && thresholds.verified;
  const valid =
    typeof lo === "number" && typeof hi === "number" && lo >= 0 && lo < hi && hi <= 1;
  return valid ? { false_success: lo, verified: hi } : { ...DEFAULT_THRESHOLDS };
}

// Positions (percent of the track) for the fill and the two gate marks, and
// which side of the gate `p` falls on.
export function meterGeometry(p, thresholds) {
  const gate = normalizeThresholds(thresholds);
  const marks = { lo: gate.false_success * 100, hi: gate.verified * 100 };
  if (typeof p !== "number" || Number.isNaN(p)) return { fill: 0, zone: "none", ...marks };
  const clamped = Math.min(1, Math.max(0, p));
  let zone = "mid";
  if (p >= gate.verified) zone = "high";
  else if (p <= gate.false_success) zone = "low";
  return { fill: clamped * 100, zone, ...marks };
}

// -- steps -----------------------------------------------------------------

export function stepAnchorId(index) {
  return `step-${index}`;
}

export function hasStep(steps, index) {
  return (steps || []).some((step) => step.i === index);
}

// message | call | result | probe | unknown, whether or not the step failed.
export function stepBase(step) {
  return classifyStep(step).replace("-failed", "");
}

// "ok" / "failed" for steps that report an outcome (results and probes),
// null for messages and calls.
export function stepStatus(step) {
  if (!step) return null;
  if (stepFailed(step)) return "failed";
  if (step.ok === true) return "ok";
  return null;
}

// -- tones -----------------------------------------------------------------

const OUTCOME_TONES = {
  probe_supported: "good",
  receipt_only: "warn",
  unsupported: "bad",
  contradicted: "bad",
};

export function outcomeTone(outcome) {
  if (Object.hasOwn(OUTCOME_TONES, outcome || "")) return OUTCOME_TONES[outcome];
  return "neutral";
}

const CHECK_TONES = { pass: "good", fail: "bad", skipped: "neutral" };
const CHECK_GLYPHS = { pass: "✓", fail: "✗", skipped: "–" };

export function checkTone(result) {
  return Object.hasOwn(CHECK_TONES, result || "") ? CHECK_TONES[result] : "neutral";
}

export function checkGlyph(result) {
  return Object.hasOwn(CHECK_GLYPHS, result || "") ? CHECK_GLYPHS[result] : "?";
}

export function failureKindTone(kind) {
  if (kind === "none") return "good";
  if (kind === "cannot_tell" || !kind) return "neutral";
  return "bad";
}

// -- claims, contributions, citations --------------------------------------

export function subjectPairs(subject) {
  if (!subject || typeof subject !== "object") return [];
  return Object.entries(subject).map(([key, value]) => ({ key, value }));
}

// Contributions largest first, each with the geometry of a bar that grows from
// the centre line: positive to the right, negative to the left. The largest
// magnitude fills half the track (`left` and `width` are percentages of it).
export function divergingBars(contributions) {
  const sorted = sortByAbsValue(contributions, "value");
  const maxAbs = sorted.reduce((max, item) => Math.max(max, Math.abs(item.value)), 0);
  return sorted.map((item) => {
    const width = maxAbs === 0 ? 0 : (Math.abs(item.value) / maxAbs) * 50;
    const positive = item.value >= 0;
    return {
      feature: item.feature,
      value: item.value,
      direction: positive ? "positive" : "negative",
      left: positive ? 50 : 50 - width,
      width,
    };
  });
}

// Each cited step with whether the trace actually has it; a judge can cite
// a step that does not exist.
export function citationStatus(cited, steps) {
  return (cited || []).map((step) => ({ step, valid: hasStep(steps, step) }));
}

// Which inspector sections have something to show.
export function inspectorSections(payload) {
  const data = payload || {};
  const nonEmpty = (list) => Array.isArray(list) && list.length > 0;
  return {
    instruction: Boolean(data.instruction) || Boolean(data.final_message),
    reasons: nonEmpty(data.reasons),
    claims: nonEmpty(data.claims),
    evidence: nonEmpty(data.rule_evidence),
    contributions: nonEmpty(data.classifier_contributions),
    judge: Boolean(data.judge),
    steps: nonEmpty(data.steps),
  };
}
