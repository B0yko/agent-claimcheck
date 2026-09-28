import assert from "node:assert/strict";
import { test } from "node:test";

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
  formatJson,
  formatMs,
  formatP,
  formatPercent,
  formatSigned,
  formatSpend,
  formatValue,
  formatVerdict,
  hasContent,
  humanize,
  pluralize,
  spacedJson,
  truncate,
} from "../../src/agent_claimcheck/server/static/format.js";

test("formatP renders two decimals, and '-' for null/undefined", () => {
  assert.equal(formatP(0.8), "0.80");
  assert.equal(formatP(0), "0.00");
  assert.equal(formatP(null), "-");
  assert.equal(formatP(undefined), "-");
});

test("formatPercent rounds to a whole percent", () => {
  assert.equal(formatPercent(0.821), "82%");
  assert.equal(formatPercent(0), "0%");
  assert.equal(formatPercent(null), "-");
});

test("formatCost shows more precision for sub-cent amounts", () => {
  assert.equal(formatCost(0), "$0.00");
  assert.equal(formatCost(0.0031), "$0.0031");
  assert.equal(formatCost(1.5), "$1.50");
  assert.equal(formatCost(null), "-");
});

test("formatVerdict maps known verdicts and passes through unknown ones", () => {
  assert.equal(formatVerdict("verified"), "Verified");
  assert.equal(formatVerdict("false_success"), "False success");
  assert.equal(formatVerdict("unverifiable"), "Unverifiable");
  assert.equal(formatVerdict("skipped"), "Skipped");
  assert.equal(formatVerdict("something_else"), "something_else");
  assert.equal(formatVerdict(null), "-");
});

test("formatMs switches from milliseconds to seconds at 1000ms", () => {
  assert.equal(formatMs(250), "250 ms");
  assert.equal(formatMs(999), "999 ms");
  assert.equal(formatMs(1500), "1.5 s");
  assert.equal(formatMs(null), "-");
});

test("truncate only cuts strings longer than the limit", () => {
  assert.equal(truncate("hello", 10), "hello");
  assert.equal(truncate("hello world", 5), "hello…");
  assert.equal(truncate(null, 5), "");
});

test("formatDomain capitalises the first letter", () => {
  assert.equal(formatDomain("booking"), "Booking");
  assert.equal(formatDomain("crm"), "CRM");
  assert.equal(formatDomain(""), "-");
  assert.equal(formatDomain(null), "-");
});

test("charsRemaining counts down from the limit", () => {
  assert.equal(charsRemaining("", 1000), 1000);
  assert.equal(charsRemaining("abc", 1000), 997);
  assert.equal(charsRemaining(null, 1000), 1000);
});

test("formatSigned always shows the sign of a non-zero number", () => {
  assert.equal(formatSigned(2.1716, 3), "+2.172");
  assert.equal(formatSigned(-0.9633, 3), "-0.963");
  assert.equal(formatSigned(0, 2), "0.00");
  assert.equal(formatSigned(null), "-");
});

test("pluralize and humanize", () => {
  assert.equal(pluralize(1, "trace"), "trace");
  assert.equal(pluralize(0, "trace"), "traces");
  assert.equal(pluralize(2, "match", "matches"), "matches");
  assert.equal(humanize("probe_supported"), "probe supported");
  assert.equal(humanize("classifier-lr"), "classifier-lr");
  assert.equal(humanize(null), "");
});

test("formatInputSummary joins the inputs and counts the traces", () => {
  assert.equal(formatInputSummary(["bench:test"], 120), "bench:test · 120 traces");
  assert.equal(formatInputSummary(["a.jsonl", "b.jsonl"], 1), "a.jsonl, b.jsonl · 1 trace");
  assert.equal(formatInputSummary([], 0), "0 traces");
});

test("formatCharsLeft counts down, singular at one", () => {
  assert.equal(formatCharsLeft(1000), "1000 characters left");
  assert.equal(formatCharsLeft(1), "1 character left");
  assert.equal(formatCharsLeft(0), "0 characters left");
});

test("formatEstimateError maps the server's codes to short labels", () => {
  assert.equal(formatEstimateError("not_configured"), "-");
  assert.equal(formatEstimateError("unknown_price"), "unknown price");
  assert.equal(formatEstimateError("weird_code"), "weird code");
  assert.equal(formatEstimateError(undefined), "-");
});

test("spacedJson puts a space after every colon and comma, nested too", () => {
  assert.equal(spacedJson({ a: 1, b: [1, 2], c: { d: "x" } }), '{"a": 1, "b": [1, 2], "c": {"d": "x"}}');
  assert.equal(spacedJson([]), "[]");
  assert.equal(spacedJson({}), "{}");
  assert.equal(spacedJson(null), "null");
  assert.equal(spacedJson(undefined), "null");
  assert.equal(spacedJson("a, b: c"), '"a, b: c"');
});

test("formatJson keeps short values on one line and indents wide ones", () => {
  assert.equal(formatJson({ query: "Seto Butu" }), '{"query": "Seto Butu"}');
  const wide = { text: "x".repeat(200) };
  assert.equal(formatJson(wide), JSON.stringify(wide, null, 2));
  assert.equal(formatJson({ a: "b".repeat(40) }, 20), JSON.stringify({ a: "b".repeat(40) }, null, 2));
  assert.equal(formatJson(undefined), "null");
});

test("formatValue writes scalars plainly and structures as one-line JSON", () => {
  assert.equal(formatValue("rec_1"), "rec_1");
  assert.equal(formatValue(30), "30");
  assert.equal(formatValue(true), "true");
  assert.equal(formatValue(null), "null");
  assert.equal(formatValue(["first", "second"]), '["first", "second"]');
  assert.equal(formatValue({ k: 1 }), '{"k": 1}');
});

test("formatBlock keeps strings verbatim and formats everything else as JSON", () => {
  assert.equal(formatBlock("line one\nline two"), "line one\nline two");
  assert.equal(formatBlock({ ok: true }), '{"ok": true}');
  assert.equal(formatBlock([]), "[]");
});

test("hasContent is false only for null, undefined and the empty string", () => {
  assert.equal(hasContent(null), false);
  assert.equal(hasContent(undefined), false);
  assert.equal(hasContent(""), false);
  assert.equal(hasContent({}), true);
  assert.equal(hasContent([]), true);
  assert.equal(hasContent(0), true);
  assert.equal(hasContent(false), true);
});

test("collapsePreview leaves short text alone", () => {
  const info = collapsePreview("one\ntwo\nthree");
  assert.equal(info.collapsible, false);
  assert.equal(info.preview, "one\ntwo\nthree");
  assert.equal(info.lineCount, 3);
});

test("collapsePreview keeps the first three lines of a many-line text", () => {
  const text = ["a", "b", "c", "d", "e"].join("\n");
  const info = collapsePreview(text);
  assert.equal(info.collapsible, true);
  assert.equal(info.preview, "a\nb\nc");
  assert.equal(info.lineCount, 5);
  assert.equal(expandLabel(info), "Show all 5 lines");
});

test("collapsePreview also collapses one very long line, capped at previewChars", () => {
  const text = "x".repeat(700);
  const info = collapsePreview(text);
  assert.equal(info.collapsible, true);
  assert.equal(info.preview.length, 600);
  assert.equal(info.lineCount, 1);
  assert.equal(expandLabel(info), "Show all 700 characters");
  assert.equal(collapsePreview("x".repeat(240)).collapsible, false);
  assert.equal(collapsePreview("x".repeat(241)).collapsible, true);
});

test("collapsePreview treats missing text as an empty, uncollapsed block", () => {
  const info = collapsePreview(null);
  assert.equal(info.collapsible, false);
  assert.equal(info.preview, "");
  assert.equal(info.lineCount, 0);
});

test("formatSpend shows the budget only when it is known", () => {
  assert.equal(formatSpend(0.0031, 0.12), "Spent $0.0031 of $0.12");
  assert.equal(formatSpend(0.5, null), "Spent $0.50");
  assert.equal(formatSpend(0, undefined), "Spent $0.00");
});
