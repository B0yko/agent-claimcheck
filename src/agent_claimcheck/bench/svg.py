"""A dependency-free, deterministic SVG writer for `reliability.svg` and
`histogram.svg`.

No external plotting library (matplotlib is explicitly excluded from this
project's dependencies): every element is a hand-built `<path>`/`<rect>`/
`<circle>`/`<text>`. Every number is formatted with a fixed precision so two
runs over the same data produce byte-identical output, and nothing here
embeds a timestamp. Readable on a plain white background in both a light and
a dark host page, since the background itself is opaque.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

_BINS = 10
_PLOT = 200.0
_PAD_LEFT = 46.0
_PAD_RIGHT = 14.0
_PAD_TOP = 34.0
_PAD_BOTTOM = 40.0
_PANEL_W = _PLOT + _PAD_LEFT + _PAD_RIGHT
_PANEL_H = _PLOT + _PAD_TOP + _PAD_BOTTOM

_AXIS_COLOR = "#94a3b8"
_DIAGONAL_COLOR = "#cbd5e1"
_RAW_COLOR = "#f97316"
_CAL_COLOR = "#2563eb"
_TEXT_COLOR = "#0f172a"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _f(x: float) -> str:
    return f"{x:.2f}"


@dataclass(frozen=True)
class Panel:
    """One detector's raw and calibrated `(p_success, y_success)` pairs."""

    name: str
    raw: Sequence[tuple[float, float]]
    calibrated: Sequence[tuple[float, float]]


def _bin_reliability(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Mean predicted vs. mean actual per equal-width bin, empty bins skipped."""
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(_BINS)]
    for p, y in points:
        idx = min(int(p * _BINS), _BINS - 1)
        buckets[idx].append((p, y))
    result: list[tuple[float, float]] = []
    for bucket in buckets:
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        mean_y = sum(y for _, y in bucket) / len(bucket)
        result.append((mean_p, mean_y))
    return result


def _bin_counts(values: Sequence[float]) -> list[int]:
    counts = [0] * _BINS
    for v in values:
        idx = min(int(v * _BINS), _BINS - 1)
        counts[idx] += 1
    return counts


def _plot_x(p: float) -> float:
    return _PAD_LEFT + p * _PLOT


def _plot_y(y: float) -> float:
    return _PAD_TOP + (1.0 - y) * _PLOT


def reliability_svg(panels: Sequence[Panel]) -> str:
    """One reliability panel per detector: raw (orange) vs. calibrated (blue)
    curves against the diagonal, on the test split.
    """
    n = len(panels)
    width = _PANEL_W * n
    height = _PANEL_H
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(width)} {_f(height)}" '
        f'font-family="sans-serif">',
        f'<rect x="0" y="0" width="{_f(width)}" height="{_f(height)}" fill="#ffffff"/>',
    ]
    for panel_idx, panel in enumerate(panels):
        ox = _PANEL_W * panel_idx
        parts.append(
            f'<text x="{_f(ox + _PANEL_W / 2)}" y="18" text-anchor="middle" '
            f'font-size="13" fill="{_TEXT_COLOR}">{_esc(panel.name)}</text>'
        )
        parts.append(
            f'<rect x="{_f(ox + _PAD_LEFT)}" y="{_f(_PAD_TOP)}" width="{_f(_PLOT)}" '
            f'height="{_f(_PLOT)}" fill="none" stroke="{_AXIS_COLOR}"/>'
        )
        parts.append(
            f'<line x1="{_f(ox + _PAD_LEFT)}" y1="{_f(_PAD_TOP + _PLOT)}" '
            f'x2="{_f(ox + _PAD_LEFT + _PLOT)}" y2="{_f(_PAD_TOP)}" '
            f'stroke="{_DIAGONAL_COLOR}" stroke-dasharray="4,3"/>'
        )
        for series, color, dash in (
            (_bin_reliability(panel.raw), _RAW_COLOR, ' stroke-dasharray="5,3"'),
            (_bin_reliability(panel.calibrated), _CAL_COLOR, ""),
        ):
            if series:
                path_d = " ".join(
                    f"{'M' if i == 0 else 'L'}{_f(ox + _plot_x(p))},{_f(_plot_y(y))}"
                    for i, (p, y) in enumerate(series)
                )
                parts.append(
                    f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="1.6"{dash}/>'
                )
            for p, y in series:
                parts.append(
                    f'<circle cx="{_f(ox + _plot_x(p))}" cy="{_f(_plot_y(y))}" r="2.2" '
                    f'fill="{color}"/>'
                )
        parts.append(
            f'<text x="{_f(ox + _PAD_LEFT)}" y="{_f(_PAD_TOP + _PLOT + 16)}" font-size="9" '
            f'fill="{_TEXT_COLOR}">0.0</text>'
        )
        parts.append(
            f'<text x="{_f(ox + _PAD_LEFT + _PLOT - 12)}" y="{_f(_PAD_TOP + _PLOT + 16)}" '
            f'font-size="9" fill="{_TEXT_COLOR}">1.0</text>'
        )
        parts.append(
            f'<text x="{_f(ox + _PAD_LEFT - 4)}" y="{_f(_PAD_TOP + _PLOT + 30)}" font-size="9" '
            f'fill="{_TEXT_COLOR}">predicted p_success, actual success rate by bin</text>'
        )
    legend_y = height - 8.0
    parts.append(
        f'<line x1="8" y1="{_f(legend_y - 4)}" x2="24" y2="{_f(legend_y - 4)}" '
        f'stroke="{_RAW_COLOR}" stroke-width="1.6" stroke-dasharray="5,3"/>'
        f'<text x="28" y="{_f(legend_y)}" font-size="9" fill="{_TEXT_COLOR}">raw</text>'
        f'<line x1="60" y1="{_f(legend_y - 4)}" x2="76" y2="{_f(legend_y - 4)}" '
        f'stroke="{_CAL_COLOR}" stroke-width="1.6"/>'
        f'<text x="80" y="{_f(legend_y)}" font-size="9" fill="{_TEXT_COLOR}">calibrated</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def histogram_svg(panels: Sequence[Panel]) -> str:
    """One histogram panel per detector: raw vs. calibrated `p_success`
    counts over 10 equal-width bins, as paired bars.
    """
    n = len(panels)
    width = _PANEL_W * n
    height = _PANEL_H
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(width)} {_f(height)}" '
        f'font-family="sans-serif">',
        f'<rect x="0" y="0" width="{_f(width)}" height="{_f(height)}" fill="#ffffff"/>',
    ]
    for panel_idx, panel in enumerate(panels):
        ox = _PANEL_W * panel_idx
        raw_counts = _bin_counts([p for p, _ in panel.raw])
        cal_counts = _bin_counts([p for p, _ in panel.calibrated])
        raw_total = max(sum(raw_counts), 1)
        cal_total = max(sum(cal_counts), 1)
        max_share = max(
            [c / raw_total for c in raw_counts] + [c / cal_total for c in cal_counts] + [0.0001]
        )
        parts.append(
            f'<text x="{_f(ox + _PANEL_W / 2)}" y="18" text-anchor="middle" '
            f'font-size="13" fill="{_TEXT_COLOR}">{_esc(panel.name)}</text>'
        )
        parts.append(
            f'<rect x="{_f(ox + _PAD_LEFT)}" y="{_f(_PAD_TOP)}" width="{_f(_PLOT)}" '
            f'height="{_f(_PLOT)}" fill="none" stroke="{_AXIS_COLOR}"/>'
        )
        bin_w = _PLOT / _BINS
        for i in range(_BINS):
            bx = ox + _PAD_LEFT + i * bin_w
            for counts, total, color, offset, bar_w in (
                (raw_counts, raw_total, _RAW_COLOR, 0.0, bin_w * 0.42),
                (cal_counts, cal_total, _CAL_COLOR, bin_w * 0.46, bin_w * 0.42),
            ):
                share = counts[i] / total
                bar_h = (share / max_share) * (_PLOT - 4.0) if max_share else 0.0
                parts.append(
                    f'<rect x="{_f(bx + offset + 1)}" y="{_f(_PAD_TOP + _PLOT - bar_h)}" '
                    f'width="{_f(bar_w)}" height="{_f(bar_h)}" fill="{color}"/>'
                )
        parts.append(
            f'<text x="{_f(ox + _PAD_LEFT)}" y="{_f(_PAD_TOP + _PLOT + 16)}" font-size="9" '
            f'fill="{_TEXT_COLOR}">0.0</text>'
        )
        parts.append(
            f'<text x="{_f(ox + _PAD_LEFT + _PLOT - 12)}" y="{_f(_PAD_TOP + _PLOT + 16)}" '
            f'font-size="9" fill="{_TEXT_COLOR}">1.0</text>'
        )
        parts.append(
            f'<text x="{_f(ox + _PAD_LEFT - 4)}" y="{_f(_PAD_TOP + _PLOT + 30)}" font-size="9" '
            f'fill="{_TEXT_COLOR}">p_success histogram (share of trace count)</text>'
        )
    legend_y = height - 8.0
    parts.append(
        f'<rect x="8" y="{_f(legend_y - 10)}" width="12" height="8" fill="{_RAW_COLOR}"/>'
        f'<text x="24" y="{_f(legend_y)}" font-size="9" fill="{_TEXT_COLOR}">raw</text>'
        f'<rect x="60" y="{_f(legend_y - 10)}" width="12" height="8" fill="{_CAL_COLOR}"/>'
        f'<text x="76" y="{_f(legend_y)}" font-size="9" fill="{_TEXT_COLOR}">calibrated</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)
