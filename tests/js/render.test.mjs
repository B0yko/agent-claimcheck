import assert from "node:assert/strict";
import { test } from "node:test";

import {
  classifyStep,
  overviewCount,
  overviewDomains,
  sortByAbsValue,
  sortQueueByDistance,
  sortSteps,
  stepFailed,
  verdictTotals,
} from "../../src/agent_claimcheck/server/static/render.js";

test("classifyStep labels each step kind, flagging failed results/probes", () => {
  assert.equal(classifyStep({ kind: "tool_call" }), "call");
  assert.equal(classifyStep({ kind: "tool_result", ok: true }), "result");
  assert.equal(classifyStep({ kind: "tool_result", ok: false }), "result-failed");
  assert.equal(classifyStep({ kind: "tool_result", error: "not_found: x" }), "result-failed");
  assert.equal(classifyStep({ kind: "state_probe", ok: true }), "probe");
  assert.equal(classifyStep({ kind: "state_probe", ok: false }), "probe-failed");
  assert.equal(classifyStep({ kind: "message" }), "message");
  assert.equal(classifyStep(null), "unknown");
});

test("stepFailed is true only for ok:false or a non-null error", () => {
  assert.equal(stepFailed({ ok: true, error: null }), false);
  assert.equal(stepFailed({ ok: false, error: null }), true);
  assert.equal(stepFailed({ ok: true, error: "boom" }), true);
  assert.equal(stepFailed(null), false);
});

test("sortSteps orders by `i` without mutating the input", () => {
  const steps = [{ i: 2 }, { i: 0 }, { i: 1 }];
  const sorted = sortSteps(steps);
  assert.deepEqual(
    sorted.map((s) => s.i),
    [0, 1, 2]
  );
  assert.deepEqual(
    steps.map((s) => s.i),
    [2, 0, 1]
  );
});

test("sortQueueByDistance orders closest to the 0.5 fence first", () => {
  const items = [{ p_success: 0.9 }, { p_success: 0.5 }, { p_success: 0.6 }, { p_success: null }];
  const sorted = sortQueueByDistance(items);
  assert.deepEqual(
    sorted.map((i) => i.p_success),
    [0.5, null, 0.6, 0.9]
  );
});

test("sortByAbsValue orders by magnitude, largest first", () => {
  const items = [{ value: 0.1 }, { value: -0.9 }, { value: 0.4 }];
  const sorted = sortByAbsValue(items);
  assert.deepEqual(
    sorted.map((i) => i.value),
    [-0.9, 0.4, 0.1]
  );
});

test("verdictTotals sums counts across domains", () => {
  const overview = { verified: { booking: 3, crm: 2 }, false_success: { booking: 1 } };
  assert.deepEqual(verdictTotals(overview), { verified: 5, false_success: 1 });
});

test("overviewDomains and overviewCount read the verdict-by-domain shape", () => {
  const overview = { verified: { booking: 3 }, false_success: { crm: 1 } };
  assert.deepEqual(overviewDomains(overview), ["booking", "crm"]);
  assert.equal(overviewCount(overview, "verified", "booking"), 3);
  assert.equal(overviewCount(overview, "verified", "crm"), 0);
  assert.equal(overviewCount(overview, "unverifiable", "booking"), 0);
});
