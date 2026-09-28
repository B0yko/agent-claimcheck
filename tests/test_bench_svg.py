"""Tests for `bench/svg.py`'s dependency-free SVG writer."""

from __future__ import annotations

import xml.dom.minidom as minidom

from agent_claimcheck.bench.svg import Panel, histogram_svg, reliability_svg


def _panel(name: str) -> Panel:
    raw = [(0.9, 1.0), (0.1, 0.0), (0.5, 1.0), (0.95, 0.0), (0.02, 0.0)]
    calibrated = [(0.8, 1.0), (0.2, 0.0), (0.55, 1.0), (0.85, 0.0), (0.05, 0.0)]
    return Panel(name, raw, calibrated)


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
    svg = reliability_svg([])
    minidom.parseString(svg)
    assert svg.startswith("<svg")
