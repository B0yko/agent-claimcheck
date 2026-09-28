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

export function formatDomain(domain) {
  if (typeof domain !== "string" || domain.length === 0) return "-";
  return domain[0].toUpperCase() + domain.slice(1);
}

export function charsRemaining(text, limit) {
  const used = typeof text === "string" ? text.length : 0;
  return limit - used;
}
