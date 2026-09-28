"""A dependency-free, deterministic SVG writer for `reliability.svg` and
`histogram.svg`.

No external plotting library (matplotlib is explicitly excluded from this
project's dependencies): every element is a hand-built `<path>`/`<rect>`/
`<circle>`/`<line>`/`<text>`. Every number is formatted with a fixed
precision, the only arithmetic beyond `+ - * /` is `math.sqrt` (correctly
rounded under IEEE 754, so identical on every platform), the output is pure
ASCII, and nothing here embeds a timestamp: two runs over the same data
produce byte-identical files. The chart sits on an opaque white card with a
rounded border, so it reads the same on a light and a dark host page.

Layout: a title row (title left, legend right), a caption row, then one
square panel per detector with gridlines and ticks at 0, 0.25, 0.5, 0.75
and 1 on both axes, a shared y-axis title on the left and a shared x-axis
title under the panels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

_BINS = 10
_TICKS: tuple[tuple[float, str], ...] = (
    (0.0, "0"),
    (0.25, "0.25"),
    (0.5, "0.5"),
    (0.75, "0.75"),
    (1.0, "1"),
)

# Layout, in viewBox units (about one pixel each at README width).
_MIN_WIDTH = 960.0
_OUTER = 24.0
_PLOT = 240.0
_PLOT_LEFT = 80.0
_GAP = 66.0
_RIGHT = 28.0
_TITLE_Y = 36.0
_CAPTION_Y = 58.0
_PANEL_TITLE_Y = 96.0
_PLOT_TOP = 108.0
_PLOT_BOTTOM = _PLOT_TOP + _PLOT
_HEIGHT = _PLOT_BOTTOM + 66.0
_TITLE_MAX_CHARS = 32

_FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
_INK = "#0f172a"
_MUTED = "#475569"
_GRID = "#e2e8f0"
_AXIS = "#94a3b8"
_BORDER = "#e2e8f0"
_RAW_COLOR = "#d97706"
_CAL_COLOR = "#2563eb"
_WHITE = "#ffffff"

# Marker radius: area grows linearly with the bin's share of the panel's
# traces, from `_R_MIN` (one trace) towards `_R_MAX` (every trace).
_R_MIN = 3.0
_R_MAX = 8.5


def _esc(text: str) -> str:
    escaped = (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )
    return "".join(c if ord(c) < 128 else f"&#{ord(c)};" for c in escaped)


def _f(x: float) -> str:
    """Two decimals, trailing zeros dropped (`80`, `80.5`, `80.25`)."""
    text = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


@dataclass(frozen=True)
class Panel:
    """One detector's raw and calibrated `(p_success, y_success)` pairs."""

    name: str
    raw: Sequence[tuple[float, float]]
    calibrated: Sequence[tuple[float, float]]


def short_title(name: str, max_chars: int = _TITLE_MAX_CHARS) -> str:
    """A panel title: `judge:qwen/qwen3-235b` -> `judge · qwen3-235b`, a
    prompt suffix in parentheses, anything else unchanged; titles longer than
    `max_chars` are cut with an ellipsis.
    """
    title = name
    if name.startswith("judge:"):
        model, _, prompt = name[len("judge:") :].partition(":")
        title = f"judge \u00b7 {model.rsplit('/', 1)[-1]}"
        if prompt:
            title += f" ({prompt})"
    if len(title) > max_chars:
        title = title[: max_chars - 1] + "\u2026"
    return title


def _bin_index(p: float) -> int:
    return min(max(int(p * _BINS), 0), _BINS - 1)


def _bin_reliability(points: Sequence[tuple[float, float]]) -> list[tuple[float, float, int]]:
    """`(mean predicted, mean actual, count)` per equal-width bin, empty bins
    skipped.
    """
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(_BINS)]
    for p, y in points:
        buckets[_bin_index(p)].append((p, y))
    result: list[tuple[float, float, int]] = []
    for bucket in buckets:
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        mean_y = sum(y for _, y in bucket) / len(bucket)
        result.append((mean_p, mean_y, len(bucket)))
    return result


def _bin_counts(values: Sequence[float]) -> list[int]:
    counts = [0] * _BINS
    for v in values:
        counts[_bin_index(v)] += 1
    return counts


def _layout(n: int) -> tuple[float, list[float]]:
    """Canvas width and each panel's plot-area left edge; the panel group is
    centred when it is narrower than the minimum width.
    """
    slots = max(n, 1)
    content = _PLOT_LEFT + slots * _PLOT + (slots - 1) * _GAP + _RIGHT
    width = max(content, _MIN_WIDTH)
    shift = (width - content) / 2
    return width, [shift + _PLOT_LEFT + i * (_PLOT + _GAP) for i in range(n)]


def _px(left: float, p: float) -> float:
    return left + p * _PLOT


def _py(y: float) -> float:
    return _PLOT_TOP + (1.0 - y) * _PLOT


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: float,
    fill: str = _INK,
    anchor: str = "start",
    weight: int | None = None,
    extra: str = "",
) -> str:
    anchor_attr = "" if anchor == "start" else f' text-anchor="{anchor}"'
    weight_attr = "" if weight is None else f' font-weight="{weight}"'
    return (
        f'<text x="{_f(x)}" y="{_f(y)}" font-size="{_f(size)}" fill="{fill}"'
        f"{anchor_attr}{weight_attr}{extra}>{_esc(content)}</text>"
    )


def _estimated_width(text: str, size: float) -> float:
    """A generous width estimate (no font metrics here), used only to lay the
    legend out from the right edge.
    """
    return len(text) * size * 0.58


def _legend_swatch(kind: str, x: float, y: float) -> list[str]:
    """A 24-wide legend sample whose vertical centre is `y`."""
    if kind in ("raw-bar", "cal-bar"):
        color = _RAW_COLOR if kind == "raw-bar" else _CAL_COLOR
        return [
            f'<rect x="{_f(x + 6)}" y="{_f(y - 6)}" width="12" height="12" rx="2" fill="{color}"/>'
        ]
    if kind == "raw-line":
        return [
            f'<line x1="{_f(x)}" y1="{_f(y)}" x2="{_f(x + 24)}" y2="{_f(y)}" '
            f'stroke="{_RAW_COLOR}" stroke-width="2" stroke-dasharray="5 3"/>',
            f'<circle cx="{_f(x + 12)}" cy="{_f(y)}" r="3.75" fill="{_WHITE}" '
            f'stroke="{_RAW_COLOR}" stroke-width="1.75"/>',
        ]
    return [
        f'<line x1="{_f(x)}" y1="{_f(y)}" x2="{_f(x + 24)}" y2="{_f(y)}" '
        f'stroke="{_CAL_COLOR}" stroke-width="2.25"/>',
        f'<circle cx="{_f(x + 12)}" cy="{_f(y)}" r="4" fill="{_CAL_COLOR}" '
        f'stroke="{_WHITE}" stroke-width="1.25"/>',
    ]


def _open(width: float, title: str, caption: str, legend: Sequence[tuple[str, str]]) -> list[str]:
    """The root element, card background, title row with the legend at its
    right end, and the caption row.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_f(width)}" height="{_f(_HEIGHT)}" '
        f'viewBox="0 0 {_f(width)} {_f(_HEIGHT)}" font-family="{_FONT}" role="img">',
        f"<title>{_esc(title)}</title>",
        f'<rect x="0.5" y="0.5" width="{_f(width - 1)}" height="{_f(_HEIGHT - 1)}" rx="12" '
        f'fill="{_WHITE}" stroke="{_BORDER}"/>',
        _text(_OUTER, _TITLE_Y, title, size=16, weight=600),
        _text(_OUTER, _CAPTION_Y, caption, size=12, fill=_MUTED),
    ]
    x_end = width - _OUTER
    for label, kind in reversed(legend):
        text_x = x_end - _estimated_width(label, 12)
        swatch_x = text_x - 6 - 24
        parts.extend(_legend_swatch(kind, swatch_x, _TITLE_Y - 4.5))
        parts.append(_text(text_x, _TITLE_Y, label, size=12))
        x_end = swatch_x - 18
    return parts


def _axes(lefts: Sequence[float], x_title: str, y_title: str) -> list[str]:
    """Gridlines, axis lines and tick labels for every panel, plus the shared
    axis titles.
    """
    parts: list[str] = []
    for left in lefts:
        grid = []
        for t, _label in _TICKS:
            x = _px(left, t)
            y = _py(t)
            grid.append(f"M{_f(x)},{_f(_PLOT_TOP)}V{_f(_PLOT_BOTTOM)}")
            grid.append(f"M{_f(left)},{_f(y)}H{_f(left + _PLOT)}")
        parts.append(f'<path d="{"".join(grid)}" fill="none" stroke="{_GRID}"/>')
        parts.append(
            f'<path d="M{_f(left)},{_f(_PLOT_TOP)}V{_f(_PLOT_BOTTOM)}H{_f(left + _PLOT)}" '
            f'fill="none" stroke="{_AXIS}"/>'
        )
        for t, label in _TICKS:
            parts.append(
                _text(_px(left, t), _PLOT_BOTTOM + 20, label, size=11, fill=_MUTED, anchor="middle")
            )
            parts.append(_text(left - 8, _py(t) + 3.75, label, size=11, fill=_MUTED, anchor="end"))
    if lefts:
        mid_x = (lefts[0] + lefts[-1] + _PLOT) / 2
        parts.append(_text(mid_x, _PLOT_BOTTOM + 46, x_title, size=12, fill=_INK, anchor="middle"))
        y_x = lefts[0] - 44
        y_mid = _PLOT_TOP + _PLOT / 2
        parts.append(
            _text(
                y_x,
                y_mid,
                y_title,
                size=12,
                fill=_INK,
                anchor="middle",
                extra=f' transform="rotate(-90 {_f(y_x)} {_f(y_mid)})"',
            )
        )
    return parts


def _panel_heading(left: float, panel: Panel, n_title: int) -> str:
    """The panel title; a muted `· n = k` follows it when this panel covers
    fewer traces than the chart title's `n`.
    """
    note = "" if len(panel.raw) == n_title else f" \u00b7 n = {len(panel.raw)}"
    title = _esc(short_title(panel.name, _TITLE_MAX_CHARS - len(note)))
    if note:
        title += f'<tspan fill="{_MUTED}" font-weight="400">{_esc(note)}</tspan>'
    return (
        f'<text x="{_f(left)}" y="{_f(_PANEL_TITLE_Y)}" font-size="13" fill="{_INK}" '
        f'font-weight="600">{title}</text>'
    )


def _title_n(panels: Sequence[Panel]) -> int:
    return max((len(p.raw) for p in panels), default=0)


def _marker_radius(count: int, total: int) -> float:
    share = count / total if total else 0.0
    return sqrt(_R_MIN * _R_MIN + (_R_MAX * _R_MAX - _R_MIN * _R_MIN) * share)


def _series(left: float, points: Sequence[tuple[float, float]], *, raw: bool) -> list[str]:
    """One reliability curve: the connecting line, then its markers."""
    bins = _bin_reliability(points)
    if not bins:
        return []
    color = _RAW_COLOR if raw else _CAL_COLOR
    d = "".join(
        f"{'M' if i == 0 else 'L'}{_f(_px(left, p))},{_f(_py(y))}"
        for i, (p, y, _n) in enumerate(bins)
    )
    style = ' stroke-width="2" stroke-dasharray="6 4"' if raw else ' stroke-width="2.25"'
    parts = [f'<path d="{d}" fill="none" stroke="{color}"{style} stroke-linejoin="round"/>']
    total = len(points)
    for p, y, count in bins:
        r = _marker_radius(count, total)
        if raw:
            marker = f'fill="{_WHITE}" stroke="{color}" stroke-width="1.75"'
        else:
            marker = f'fill="{color}" stroke="{_WHITE}" stroke-width="1.25"'
        parts.append(f'<circle cx="{_f(_px(left, p))}" cy="{_f(_py(y))}" r="{_f(r)}" {marker}/>')
    return parts


def reliability_svg(panels: Sequence[Panel]) -> str:
    """One reliability panel per detector: raw (dashed amber, hollow markers)
    vs. calibrated (solid blue, filled markers) against the diagonal, on the
    test split. Marker area grows with the bin's trace count.
    """
    width, lefts = _layout(len(panels))
    n_title = _title_n(panels)
    parts = _open(
        width,
        f"Reliability on the test split (n = {n_title}) \u2014 raw vs Platt-calibrated",
        "10 equal-width bins \u00b7 marker area grows with the bin's trace count "
        "\u00b7 dashed diagonal = perfect calibration",
        [("raw", "raw-line"), ("calibrated", "cal-line")],
    )
    parts.extend(_axes(lefts, "predicted p_success", "observed success rate"))
    for left, panel in zip(lefts, panels, strict=True):
        parts.append(_panel_heading(left, panel, n_title))
        parts.append(
            f'<line x1="{_f(left)}" y1="{_f(_PLOT_BOTTOM)}" x2="{_f(left + _PLOT)}" '
            f'y2="{_f(_PLOT_TOP)}" stroke="{_AXIS}" stroke-width="1.25" stroke-dasharray="4 4"/>'
        )
        parts.extend(_series(left, panel.raw, raw=True))
        parts.extend(_series(left, panel.calibrated, raw=False))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _bar(x: float, w: float, h: float, color: str) -> str:
    """A bar standing on the x axis with rounded top corners."""
    r = min(2.0, h, w / 2)
    top = _PLOT_BOTTOM - h
    d = (
        f"M{_f(x)},{_f(_PLOT_BOTTOM)}V{_f(top + r)}Q{_f(x)},{_f(top)} {_f(x + r)},{_f(top)}"
        f"H{_f(x + w - r)}Q{_f(x + w)},{_f(top)} {_f(x + w)},{_f(top + r)}V{_f(_PLOT_BOTTOM)}Z"
    )
    return f'<path d="{d}" fill="{color}"/>'


def histogram_svg(panels: Sequence[Panel]) -> str:
    """One histogram panel per detector: the share of traces in each of 10
    equal-width `p_success` bins, raw (amber) and calibrated (blue) bars side
    by side.
    """
    width, lefts = _layout(len(panels))
    n_title = _title_n(panels)
    parts = _open(
        width,
        f"p_success distribution on the test split (n = {n_title}) \u2014 raw vs Platt-calibrated",
        "Share of test traces in each of 10 equal-width p_success bins",
        [("raw", "raw-bar"), ("calibrated", "cal-bar")],
    )
    parts.extend(_axes(lefts, "p_success", "share of traces"))
    bin_w = _PLOT / _BINS
    pad = 3.0
    inner_gap = 2.0
    bar_w = (bin_w - 2 * pad - inner_gap) / 2
    for left, panel in zip(lefts, panels, strict=True):
        parts.append(_panel_heading(left, panel, n_title))
        for points, color, offset in (
            (panel.raw, _RAW_COLOR, pad),
            (panel.calibrated, _CAL_COLOR, pad + bar_w + inner_gap),
        ):
            counts = _bin_counts([p for p, _ in points])
            total = sum(counts)
            for i, count in enumerate(counts):
                if count == 0:
                    continue
                h = count / total * _PLOT
                parts.append(_bar(left + i * bin_w + offset, bar_w, h, color))
        parts.append(
            f'<path d="M{_f(left)},{_f(_PLOT_BOTTOM)}H{_f(left + _PLOT)}" fill="none" '
            f'stroke="{_AXIS}"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
