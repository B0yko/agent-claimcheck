import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_THRESHOLDS,
  checkGlyph,
  checkTone,
  citationStatus,
  classifyStep,
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
  sortByAbsValue,
  sortQueueByDistance,
  sortSteps,
  stepAnchorId,
  stepBase,
  stepFailed,
  stepStatus,
  subjectPairs,
  verdictOrder,
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

test("verdictOrder lists known verdicts in a fixed order, then unknown ones", () => {
  const overview = { skipped: {}, weird: {}, unverifiable: {}, verified: {}, false_success: {} };
  assert.deepEqual(verdictOrder(overview), [
    "verified",
    "false_success",
    "unverifiable",
    "skipped",
    "weird",
  ]);
  assert.deepEqual(verdictOrder({ false_success: {} }), ["false_success"]);
  assert.deepEqual(verdictOrder(null), []);
});

test("domainTotals sums each domain over the verdicts", () => {
  const overview = { verified: { booking: 3, crm: 2 }, false_success: { booking: 1 } };
  assert.deepEqual(domainTotals(overview), { booking: 4, crm: 2 });
  assert.deepEqual(domainTotals({}), {});
});

test("normalizeThresholds keeps a valid gate and falls back otherwise", () => {
  assert.deepEqual(normalizeThresholds({ verified: 0.9, false_success: 0.1 }), {
    verified: 0.9,
    false_success: 0.1,
  });
  assert.deepEqual(normalizeThresholds(undefined), DEFAULT_THRESHOLDS);
  assert.deepEqual(normalizeThresholds({ verified: 0.1, false_success: 0.9 }), DEFAULT_THRESHOLDS);
  assert.deepEqual(normalizeThresholds({ verified: "0.8", false_success: 0.2 }), DEFAULT_THRESHOLDS);
});

test("meterGeometry places the fill and the gate marks and names the zone", () => {
  const low = meterGeometry(0.1, null);
  assert.equal(low.zone, "low");
  assert.equal(low.fill, 10);
  assert.equal(low.lo, 20);
  assert.equal(low.hi, 80);
  assert.equal(meterGeometry(0.5, null).zone, "mid");
  assert.equal(meterGeometry(0.8, null).zone, "high");
  assert.equal(meterGeometry(0.2, null).zone, "low");
  assert.equal(meterGeometry(1.4, null).fill, 100);
  assert.equal(meterGeometry(-1, null).fill, 0);
  const custom = meterGeometry(0.6, { verified: 0.6, false_success: 0.3 });
  assert.equal(custom.zone, "high");
  assert.equal(custom.hi, 60);
  assert.deepEqual(meterGeometry(null, null), { fill: 0, zone: "none", lo: 20, hi: 80 });
});

test("stepBase strips the failed suffix and stepStatus reports ok/failed/none", () => {
  assert.equal(stepBase({ kind: "tool_result", ok: false }), "result");
  assert.equal(stepBase({ kind: "state_probe", error: "gone" }), "probe");
  assert.equal(stepBase({ kind: "tool_call" }), "call");
  assert.equal(stepBase({ kind: "message" }), "message");
  assert.equal(stepBase({ kind: "mystery" }), "unknown");
  assert.equal(stepStatus({ kind: "tool_result", ok: true, error: null }), "ok");
  assert.equal(stepStatus({ kind: "tool_result", ok: false }), "failed");
  assert.equal(stepStatus({ kind: "state_probe", ok: true, error: "boom" }), "failed");
  assert.equal(stepStatus({ kind: "message", ok: null }), null);
  assert.equal(stepStatus({ kind: "tool_call" }), null);
  assert.equal(stepStatus(null), null);
});

test("stepAnchorId and hasStep agree on step indexes", () => {
  assert.equal(stepAnchorId(5), "step-5");
  assert.equal(hasStep([{ i: 0 }, { i: 5 }], 5), true);
  assert.equal(hasStep([{ i: 0 }], 5), false);
  assert.equal(hasStep(undefined, 0), false);
});

test("tones map outcomes, check results and failure kinds", () => {
  assert.equal(outcomeTone("probe_supported"), "good");
  assert.equal(outcomeTone("receipt_only"), "warn");
  assert.equal(outcomeTone("unsupported"), "bad");
  assert.equal(outcomeTone("contradicted"), "bad");
  assert.equal(outcomeTone("classifier-lr"), "neutral");
  assert.equal(outcomeTone(null), "neutral");
  assert.equal(outcomeTone("toString"), "neutral");
  assert.equal(checkTone("pass"), "good");
  assert.equal(checkTone("fail"), "bad");
  assert.equal(checkTone("skipped"), "neutral");
  assert.equal(checkGlyph("pass"), "✓");
  assert.equal(checkGlyph("fail"), "✗");
  assert.equal(checkGlyph("skipped"), "–");
  assert.equal(checkGlyph("other"), "?");
  assert.equal(failureKindTone("none"), "good");
  assert.equal(failureKindTone("cannot_tell"), "neutral");
  assert.equal(failureKindTone(null), "neutral");
  assert.equal(failureKindTone("phantom_action"), "bad");
});

test("subjectPairs lists key/value pairs and tolerates an empty or missing subject", () => {
  assert.deepEqual(subjectPairs({ record_id: "rec_1", fields: { a: 1 } }), [
    { key: "record_id", value: "rec_1" },
    { key: "fields", value: { a: 1 } },
  ]);
  assert.deepEqual(subjectPairs({}), []);
  assert.deepEqual(subjectPairs(null), []);
});

test("divergingBars sorts by magnitude and grows bars from the centre line", () => {
  const bars = divergingBars([
    { feature: "small", value: 0.5 },
    { feature: "big_negative", value: -2 },
    { feature: "big_positive", value: 1 },
  ]);
  assert.deepEqual(
    bars.map((bar) => bar.feature),
    ["big_negative", "big_positive", "small"]
  );
  assert.deepEqual(bars[0], {
    feature: "big_negative",
    value: -2,
    direction: "negative",
    left: 0,
    width: 50,
  });
  assert.equal(bars[1].direction, "positive");
  assert.equal(bars[1].left, 50);
  assert.equal(bars[1].width, 25);
  assert.equal(bars[2].width, 12.5);
});

test("divergingBars handles no contributions and all-zero contributions", () => {
  assert.deepEqual(divergingBars(undefined), []);
  const zero = divergingBars([{ feature: "flat", value: 0 }]);
  assert.equal(zero[0].width, 0);
  assert.equal(zero[0].left, 50);
});

test("citationStatus flags cited steps the trace does not have", () => {
  const steps = [{ i: 0 }, { i: 1 }, { i: 2 }];
  assert.deepEqual(citationStatus([2, 9], steps), [
    { step: 2, valid: true },
    { step: 9, valid: false },
  ]);
  assert.deepEqual(citationStatus(undefined, steps), []);
});

test("inspectorSections reports which sections have content", () => {
  const empty = inspectorSections({
    instruction: "",
    final_message: "",
    reasons: [],
    claims: [],
    rule_evidence: [],
    classifier_contributions: [],
    judge: null,
    steps: [],
  });
  assert.deepEqual(empty, {
    instruction: false,
    reasons: false,
    claims: false,
    evidence: false,
    contributions: false,
    judge: false,
    steps: false,
  });
  const full = inspectorSections({
    instruction: "Book it",
    reasons: [{}],
    claims: [{}],
    rule_evidence: [{}],
    classifier_contributions: [{}],
    judge: { model: "m" },
    steps: [{}],
  });
  assert.equal(Object.values(full).every(Boolean), true);
  assert.equal(inspectorSections({ final_message: "Done" }).instruction, true);
  assert.equal(inspectorSections(null).steps, false);
});
