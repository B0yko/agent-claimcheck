// Pure formatting helpers: no DOM, importable from `node --test`.

export function formatP(p) {
  if (p === null || p === undefined || Number.isNaN(p)) return "-";
  return p.toFixed(2);
}

export function formatPercent(p) {
  if (p === null || p === undefined || Number.isNaN(p)) return "-";
  return `${Math.round(p * 100)}%`;
}

export function formatCost(usd) {
  if (usd === null || usd === undefined || Number.isNaN(usd)) return "-";
  if (usd === 0) return "$0.00";
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

const VERDICT_LABELS = {
  verified: "Verified",
  false_success: "False success",
  unverifiable: "Unverifiable",
  skipped: "Skipped",
};

export function formatVerdict(verdict) {
  if (!verdict) return "-";
  return VERDICT_LABELS[verdict] || verdict;
}

export function formatMs(ms) {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return "-";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function truncate(text, limit) {
  if (typeof text !== "string") return "";
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}…`;
}

const DOMAIN_LABELS = { crm: "CRM" };

export function formatDomain(domain) {
  if (typeof domain !== "string" || domain.length === 0) return "-";
  if (Object.hasOwn(DOMAIN_LABELS, domain)) return DOMAIN_LABELS[domain];
  return domain[0].toUpperCase() + domain.slice(1);
}

export function charsRemaining(text, limit) {
  const used = typeof text === "string" ? text.length : 0;
  return limit - used;
}

export function formatSigned(value, digits = 2) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  const text = value.toFixed(digits);
  return value > 0 ? `+${text}` : text;
}

export function pluralize(count, singular, plural = `${singular}s`) {
  return count === 1 ? singular : plural;
}

// "probe_supported" -> "probe supported"; ids such as "classifier-lr" keep their dash.
export function humanize(text) {
  if (typeof text !== "string") return "";
  return text.replaceAll("_", " ");
}

export function formatInputSummary(inputs, total) {
  const names = (inputs || []).join(", ");
  const count = `${total} ${pluralize(total, "trace")}`;
  return names ? `${names} · ${count}` : count;
}

export function formatCharsLeft(remaining) {
  return `${remaining} ${pluralize(remaining, "character")} left`;
}

const ESTIMATE_ERRORS = {
  not_configured: "-",
  unknown_price: "unknown price",
};

export function formatEstimateError(code) {
  if (typeof code !== "string" || code.length === 0) return "-";
  if (Object.hasOwn(ESTIMATE_ERRORS, code)) return ESTIMATE_ERRORS[code];
  return humanize(code);
}

// One-line JSON with a space after every ':' and ',' (JSON.stringify's compact
// form is hard to read once it wraps).
export function spacedJson(value) {
  if (value === undefined) return "null";
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map(spacedJson).join(", ")}]`;
  const parts = Object.entries(value).map(([key, item]) => `${JSON.stringify(key)}: ${spacedJson(item)}`);
  return `{${parts.join(", ")}}`;
}

// Short values stay on one line; anything wider than `width` is indented so
// the reader can scan it.
export function formatJson(value, width = 160) {
  const flat = spacedJson(value);
  if (flat.length <= width) return flat;
  return JSON.stringify(value, null, 2) ?? "null";
}

// A scalar as a person would write it: strings without quotes, structures as
// one-line JSON.
export function formatValue(value) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "object") return spacedJson(value);
  return String(value);
}

// Decides whether a block of text collapses, and what the collapsed form shows:
// the first `maxLines` lines, capped at `previewChars`. CSS clamps the preview
// to the same number of visual lines, so a single long line collapses too.
export function collapsePreview(text, { maxLines = 3, maxChars = 240, previewChars = 600 } = {}) {
  const source = typeof text === "string" ? text : "";
  const lineCount = source === "" ? 0 : source.split("\n").length;
  const collapsible = lineCount > maxLines || source.length > maxChars;
  if (!collapsible) return { collapsible: false, preview: source, lineCount, chars: source.length };
  let preview = source.split("\n").slice(0, maxLines).join("\n");
  if (preview.length > previewChars) preview = preview.slice(0, previewChars);
  return { collapsible: true, preview, lineCount, chars: source.length };
}

export function expandLabel(info) {
  if (info.lineCount > 1) return `Show all ${info.lineCount} lines`;
  return `Show all ${info.chars} characters`;
}

// What a step's args/output look like on screen: strings verbatim, everything
// else as (indented when wide) JSON.
export function formatBlock(value) {
  if (typeof value === "string") return value;
  return formatJson(value);
}

// True when there is something to draw. Empty objects and arrays count (an
// empty search result is information); null, undefined and "" do not.
export function hasContent(value) {
  return value !== null && value !== undefined && value !== "";
}

export function formatSpend(spentUsd, worstCaseUsd) {
  const spent = `Spent ${formatCost(spentUsd)}`;
  if (typeof worstCaseUsd !== "number" || Number.isNaN(worstCaseUsd)) return spent;
  return `${spent} of ${formatCost(worstCaseUsd)}`;
}
