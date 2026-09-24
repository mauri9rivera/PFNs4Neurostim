"""Anchor-and-derive palette rules (sprint 2026-09-20, Task 2)."""
from __future__ import annotations

import colorsys
import itertools

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


#: Smallest CIE76 Delta E between two models of one family; ~20 is where colours read as clearly different.
MIN_FAMILY_DELTA_E: float = 25.0
#: Smallest Delta E between an anchor and any model derived from it (the headline pair must be unmistakable).
MIN_ANCHOR_DELTA_E: float = 40.0


def test_family_hue_inside_band() -> None:
    """Derived colours stay inside their family's hue band, so cool-PFN / warm-GP is a rule, not a list."""
    for st in style.MODEL_STYLES.values():
        if st.family in style.FAMILY_HUE_BAND:
            lo, hi = style.FAMILY_HUE_BAND[st.family]
            assert lo - 0.01 <= _hue(st.color) <= hi + 0.01, (st.label, st.color)


@pytest.mark.parametrize("family", ["pfn", "gp"])
def test_models_of_a_family_are_perceptually_distinct(family: str) -> None:
    members = [(k, s.color) for k, s in style.MODEL_STYLES.items() if s.family == family]
    for (ka, ca), (kb, cb) in itertools.combinations(members, 2):
        assert style.perceptual_distance(ca, cb) >= MIN_FAMILY_DELTA_E, (ka, kb, ca, cb)


def test_gp_mll_and_gp_fixed_are_clearly_different() -> None:
    """Regression (2026-09-23): GP-fixed used to be a near-copy of GP-MLL in trajectories.svg."""
    d = style.perceptual_distance(style.model_color("gp_mll"), style.model_color("gp_naive"))
    assert d >= MIN_ANCHOR_DELTA_E


def test_no_model_is_close_to_an_anchor_of_another_family_or_the_baseline_grey() -> None:
    for key, st in style.MODEL_STYLES.items():
        if key in ("tabpfn_v2_5", "gp_mll", "random"):
            continue
        for other in (style.ANCHOR_PFN, style.ANCHOR_GP, style.NEUTRAL_GREY):
            if other == style._FAMILY_ANCHOR.get(st.family):
                continue
            assert style.perceptual_distance(st.color, other) >= MIN_FAMILY_DELTA_E, (key, other)


def test_derived_colours_stay_visible_on_white() -> None:
    lo, hi = style.DERIVED_LSTAR_RANGE
    for key, st in style.MODEL_STYLES.items():
        if key in ("tabpfn_v2_5", "gp_mll", "random"):
            continue
        assert lo <= style._srgb_to_lab(st.color)[0] <= hi, (key, st.color)


def test_derive_family_is_deterministic_and_validates() -> None:
    a = style.derive_family(style.ANCHOR_GP, 3, style.FAMILY_HUE_BAND["gp"])
    assert a == style.derive_family(style.ANCHOR_GP, 3, style.FAMILY_HUE_BAND["gp"])
    assert style.ANCHOR_GP not in a and len(set(a)) == 3
    with pytest.raises(ValueError):
        style.derive_family(style.ANCHOR_GP, -1, style.FAMILY_HUE_BAND["gp"])
    with pytest.raises(ValueError):
        style.derive_family(style.ANCHOR_GP, 2, (0.5, 0.2))


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


def test_knob_wording() -> None:
    """K5 reads as a contamination fraction and K6 as a failed-electrode fraction (restructure 2026-09-23)."""
    assert style.knob_x_label("k5_outliers", "level").startswith("Contamination")
    assert "trial slots" in style.knob_x_label("k5_outliers", "level")
    assert style.knob_x_label("k6_failure", "level").startswith("Failed electrodes")
    assert style.KNOB_X_AXIS["k2_global"] == style.KNOB_X_AXIS["k2_channel"] == "achieved_snr_db"
    assert style.knob_x_label("k2_channel", "achieved_snr_db") == style.axis_label("achieved_snr_db")
