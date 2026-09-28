"""Tests for `bench/svg.py`'s dependency-free SVG writer."""

from __future__ import annotations

import re
import xml.dom.minidom as minidom

import pytest

from agent_claimcheck.bench.svg import Panel, histogram_svg, reliability_svg, short_title

_FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def _panel(name: str) -> Panel:
    raw = [(0.9, 1.0), (0.1, 0.0), (0.5, 1.0), (0.95, 0.0), (0.02, 0.0)]
    calibrated = [(0.8, 1.0), (0.2, 0.0), (0.55, 1.0), (0.85, 0.0), (0.05, 0.0)]
    return Panel(name, raw, calibrated)


def _content(node: minidom.Node) -> str:
    if node.nodeType == node.TEXT_NODE:
        return str(node.data)  # type: ignore[attr-defined]
    return "".join(_content(child) for child in node.childNodes)


def _texts(svg: str) -> list[tuple[float, float, str]]:
    """`(x, y, content)` of every `<text>`, nested `<tspan>` text included."""
    doc = minidom.parseString(svg)
    return [
        (float(node.getAttribute("x")), float(node.getAttribute("y")), _content(node))
        for node in doc.getElementsByTagName("text")
    ]


def test_reliability_svg_is_valid_and_deterministic() -> None:
    svg = reliability_svg([_panel("rules"), _panel("classifier-lr")])
    minidom.parseString(svg)
    assert svg == reliability_svg([_panel("rules"), _panel("classifier-lr")])
    assert svg.startswith("<svg")
    assert 'fill="#ffffff"' in svg
    assert "rules" in svg and "classifier-lr" in svg


def test_histogram_svg_is_valid_and_deterministic() -> None:
    svg = histogram_svg([_panel("rules")])
    minidom.parseString(svg)
    assert svg == histogram_svg([_panel("rules")])
    assert svg.startswith("<svg")


@pytest.mark.parametrize("render", [reliability_svg, histogram_svg])
def test_output_is_ascii_with_the_system_font_stack_and_a_card(render: object) -> None:
    svg = render([_panel("rules"), _panel("judge:vendor/model")])  # type: ignore[operator]
    svg.encode("ascii")  # non-ASCII characters are written as character references
    root = minidom.parseString(svg).documentElement
    assert root.getAttribute("font-family") == _FONT
    card = minidom.parseString(svg).getElementsByTagName("rect")[0]
    assert card.getAttribute("fill") == "#ffffff"
    assert card.getAttribute("rx")  # rounded border
    assert card.getAttribute("stroke") == "#e2e8f0"


def test_reliability_has_title_ticks_axis_titles_and_legend_on_top() -> None:
    svg = reliability_svg([_panel("rules"), _panel("classifier-lr"), _panel("judge:a/b")])
    texts = _texts(svg)
    contents = [c for _, _, c in texts]
    assert "Reliability on the test split (n = 5) \u2014 raw vs Platt-calibrated" in contents
    assert "predicted p_success" in contents
    assert "observed success rate" in contents
    for label in ("0", "0.25", "0.5", "0.75", "1"):
        # One x and one y tick label per panel.
        assert contents.count(label) == 6
    title_y = next(y for _, y, c in texts if c.startswith("Reliability"))
    legend = [(x, y) for x, y, c in texts if c in ("raw", "calibrated")]
    assert len(legend) == 2
    assert all(y == title_y for _, y in legend)  # the legend shares the title row
    panel_title_y = next(y for _, y, c in texts if c == "rules")
    assert title_y < panel_title_y  # and so sits above every panel
    assert 'stroke-dasharray="4 4"' in svg  # the diagonal reference


def test_reliability_series_styles_and_marker_sizes() -> None:
    # 9 traces in the top bin, 1 in the bottom one.
    points = [(0.95, 1.0)] * 9 + [(0.05, 0.0)]
    svg = reliability_svg([Panel("rules", points, points)])
    doc = minidom.parseString(svg)
    raw_paths = [
        p
        for p in doc.getElementsByTagName("path")
        if p.getAttribute("stroke") == "#d97706" and p.getAttribute("stroke-dasharray")
    ]
    cal_paths = [
        p
        for p in doc.getElementsByTagName("path")
        if p.getAttribute("stroke") == "#2563eb" and not p.getAttribute("stroke-dasharray")
    ]
    assert len(raw_paths) == 1 and len(cal_paths) == 1
    circles = doc.getElementsByTagName("circle")
    hollow = [c for c in circles if c.getAttribute("stroke") == "#d97706"]
    filled = [c for c in circles if c.getAttribute("fill") == "#2563eb"]
    # Plus one legend marker each.
    assert len(hollow) == 3 and len(filled) == 3
    assert all(c.getAttribute("fill") == "#ffffff" for c in hollow)
    small, big = sorted(float(c.getAttribute("r")) for c in hollow[1:])
    assert small < big


def test_histogram_bars_are_shares_side_by_side() -> None:
    raw = [(0.05, 0.0)] * 3 + [(0.95, 1.0)]
    calibrated = [(0.15, 0.0)] * 3 + [(0.85, 1.0)]
    svg = histogram_svg([Panel("rules", raw, calibrated)])
    texts = [c for _, _, c in _texts(svg)]
    assert "p_success" in texts and "share of traces" in texts
    doc = minidom.parseString(svg)
    bars = [p for p in doc.getElementsByTagName("path") if p.getAttribute("fill") != "none"]
    # Two non-empty bins per series; empty bins draw nothing.
    assert [b.getAttribute("fill") for b in bars] == ["#d97706"] * 2 + ["#2563eb"] * 2
    heights = []
    for bar in bars:
        ys = [float(v) for v in re.findall(r"[\d.]+,([\d.]+)", bar.getAttribute("d"))]
        heights.append(round(max(ys) - min(ys), 1))
    assert heights == [180.0, 60.0, 180.0, 60.0]  # 3/4 and 1/4 of the 240-unit plot


def test_short_panel_titles() -> None:
    assert short_title("rules") == "rules"
    assert short_title("classifier-lr") == "classifier-lr"
    assert short_title("judge:qwen/qwen3-235b-a22b-2507") == "judge \u00b7 qwen3-235b-a22b-2507"
    assert short_title("judge:m/small:claim-by-claim") == "judge \u00b7 small (claim-by-claim)"
    long = short_title("judge:v/" + "x" * 60)
    assert len(long) == 32 and long.endswith("\u2026")
    svg = reliability_svg([Panel("judge:qwen/qwen3-235b-a22b-2507", [(0.5, 1.0)], [(0.5, 1.0)])])
    assert "judge &#183; qwen3-235b-a22b-2507" in svg
    assert "qwen/" not in svg


def test_panel_with_fewer_traces_notes_its_n() -> None:
    full = _panel("rules")
    partial = Panel("judge:a/b", full.raw[:3], full.calibrated[:3])
    texts = [c for _, _, c in _texts(reliability_svg([full, partial]))]
    assert "Reliability on the test split (n = 5) \u2014 raw vs Platt-calibrated" in texts
    assert "judge \u00b7 b \u00b7 n = 3" in texts
    assert "rules" in texts  # the full panel carries no note


def test_no_timestamps_embedded() -> None:
    svg = reliability_svg([_panel("rules")]) + histogram_svg([_panel("rules")])
    # No ISO date/time patterns and no year-like tokens sneak in.
    assert "T00:" not in svg
    assert "2026" not in svg


def test_panel_name_is_escaped() -> None:
    svg = reliability_svg([Panel("judge:<a&b>", [(0.5, 1.0)], [(0.5, 1.0)])])
    minidom.parseString(svg)  # would fail to parse if the name broke the XML
    assert "&lt;a&amp;b&gt;" in svg


def test_empty_panel_list_still_renders_a_document() -> None:
    for svg in (reliability_svg([]), histogram_svg([])):
        minidom.parseString(svg)
        assert svg.startswith("<svg")
