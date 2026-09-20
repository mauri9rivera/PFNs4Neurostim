"""Anchor-and-derive palette rules (sprint 2026-09-20, Task 2)."""
from __future__ import annotations

import colorsys

import pytest

from pfns4neurostim.visualization import style


def _hue(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[0]


def test_anchors_are_byte_exact() -> None:
    assert style.model_color("tabpfn_v2_5") == "#0072B2"
    assert style.model_color("gp_mll") == "#D55E00"


def test_derived_colours_distinct_within_family() -> None:
    for fam in ("pfn", "gp"):
        colors = [s.color for s in style.MODEL_STYLES.values() if s.family == fam]
        assert len(colors) == len(set(colors))


def test_family_hue_preserved() -> None:
    for st in style.MODEL_STYLES.values():
        if st.family == "pfn":
            assert _hue(st.color) == pytest.approx(_hue(style.ANCHOR_PFN), abs=0.01)
        if st.family == "gp":
            assert _hue(st.color) == pytest.approx(_hue(style.ANCHOR_GP), abs=0.01)


def test_pfn_cool_gp_warm() -> None:
    assert all(0.45 < _hue(s.color) < 0.70 for s in style.MODEL_STYLES.values() if s.family == "pfn")
    assert all(_hue(s.color) < 0.15 for s in style.MODEL_STYLES.values() if s.family == "gp")


def test_only_anchors_and_grey_are_literals() -> None:
    literal = {style.ANCHOR_PFN, style.ANCHOR_GP, style.NEUTRAL_GREY}
    non_derived = {k for k, s in style.MODEL_STYLES.items() if s.color in literal}
    assert non_derived == {"tabpfn_v2_5", "gp_mll", "random"}


def test_derive_shade_deterministic_and_monotone() -> None:
    a = [style.derive_shade(style.ANCHOR_PFN, i, 4) for i in range(4)]
    assert a == [style.derive_shade(style.ANCHOR_PFN, i, 4) for i in range(4)]
    lights = [colorsys.rgb_to_hls(*(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)))[1] for c in a]
    assert lights == sorted(lights)
    with pytest.raises(ValueError):
        style.derive_shade(style.ANCHOR_PFN, 4, 4)


def test_acquisition_style_families() -> None:
    first = style.acquisition_style("gp_mll", "ei")
    assert first.color == style.model_color("gp_mll")
    others = {style.acquisition_style("gp_mll", a).color for a in style.ACQUISITION_ORDER}
    assert len(others) == len(style.ACQUISITION_ORDER)
    assert style.acquisition_style("gp_mll", "ucb").linestyle != first.linestyle
    with pytest.raises(KeyError):
        style.acquisition_style("gp_mll", "nope")
