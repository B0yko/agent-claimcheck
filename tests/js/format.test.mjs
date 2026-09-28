import assert from "node:assert/strict";
import { test } from "node:test";

import {
  charsRemaining,
  formatCost,
  formatDomain,
  formatMs,
  formatP,
  formatPercent,
  formatVerdict,
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
