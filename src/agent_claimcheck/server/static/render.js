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
