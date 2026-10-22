"""Canonical figure style for every PFNs4Neurostim deliverable.

Single source of truth for figure geometry, typography, model colours/labels
and axis wording. **Every** figure-producing function (and every agent writing
one) must import from this module instead of hardcoding colours, labels,
figure sizes or axis strings.

Decisions settled with the user on 2026-09-18 (JNE / IOP submission target):

* Geometry: single-column default (8.3 cm), with named width tokens
  (``single`` / ``onehalf`` / ``double``) selectable per figure.
* Typography: Arial/Helvetica, base 9 pt (JNE floor is ~7 pt at final size;
  9 pt keeps single-column panels legible after scaling).
* Output: SVG only by default (vector, fonts kept as text). PNG/PDF are
  opt-in via ``save_figure(..., formats=("svg", "png"))`` at 600 dpi.
* Palette: Okabe-Ito, colour *family* carries model family - PFNs cool,
  GPs warm, non-learning baselines grey - so the PFN-vs-GP contrast survives
  grayscale printing and the common colour-vision deficiencies.
* Model labels: compact code-like (``TabPFN-2.5``, ``GP-MLL``, ``GP-fixed``,
  ``GP-oracle``, ``Random``).
* Stress vocabulary: roadmap jargon is canonical in code, figures and text
  (``knob``, ``level``, ``K2``, ``synthetic``/``draws``, ``breakdown point``).
"""
from __future__ import annotations

import colorsys
import os
from dataclasses import dataclass
from typing import Iterable, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "derive_family",
    "perceptual_distance",
    "knob_x_label",
    "KNOB_X_LABELS",
    "R2_AXIS_FLOOR",
    "PANEL_ASPECT",
    "DECORATION_HEIGHT",
    "SEQUENTIAL_CMAP",
    "DIVERGING_CMAP",
    "OUTLINE_COLOR",
    "FIG_WIDTHS",
    "FONT_SIZES",
    "ANCHOR_PFN",
    "ANCHOR_GP",
    "NEUTRAL_GREY",
    "REFERENCE_FLOOR_COLOR",
    "REFERENCE_CEILING_COLOR",
    "REFERENCE_BAND_ALPHA",
    "RANK_PLACEMENT_TICKS",
    "RANK_PLACEMENT_MIDLINE_LABEL",
    "RATIO_PLACEMENT_TICKS",
    "LEVEL_CMAP",
    "ACQUISITION_ORDER",
    "ACQUISITION_LINESTYLES",
    "READOUT_LINESTYLES",
    "READOUT_LABELS",
    "BELIEF_TARGET_LABELS",
    "SITE_POOL_LINESTYLES",
    "SURROGATE_BAND_LABEL",
    "SUBJECT_COLOR_RANGE",
    "SUBJECT_MARKERS",
    "SUBJECT_LABEL",
    "LINK_REFERENCE_TITLES",
    "LINK_SLOPE_NOTE",
    "LINK_TITLE",
    "SURPRISE_REFERENCE_TITLE",
    "ENGINE_LABELS",
    "IMPLIED_NOISE_LEGEND",
    "HYPC_LINE_ONLY_ENGINES",
    "HYPC_BAND_N_BOOT",
    "HYPC_BAND_CI",
    "HYPC_BAND_SEED",
    "subject_style",
    "set_integer_xaxis",
    "derive_shade",
    "acquisition_style",
    "ModelStyle",
    "MODEL_STYLES",
    "MODEL_ORDER",
    "AXIS_LABELS",
    "KNOB_LABELS",
    "KNOB_X_AXIS",
    "KNOB_SYMBOLS",
    "KNOB_PARAM_LABELS",
    "NOMINAL_MARKER_COLOR",
    "NOMINAL_MARKER_LABEL",
    "CALIBRATION_TARGET_COLOR",
    "CALIBRATION_TARGET_LABEL",
    "NO_BREAKDOWN_LABEL",
    "BREAKDOWN_RULE_NOTE",
    "ONLINE_Y_SCALER_LABELS",
    "CALIBRATION_TARGET_LABELS",
    "mark_nominal",
    "snr_series_label",
    "DEMO_LABELS",
    "DATASET_LABELS",
    "GT_MODE_LABELS",
    "apply_style",
    "figure",
    "axis_label",
    "model_style",
    "model_label",
    "model_color",
    "model_palette",
    "plot_kwargs",
    "panel_letters",
    "save_figure",
]

# ---------------------------------------------------------------------------
# Geometry (inches). JNE column widths: 8.3 cm single, 17.6 cm double.
# ---------------------------------------------------------------------------
FIG_WIDTHS: dict[str, float] = {
    "single": 3.27,    # 8.3 cm  - default
    "onehalf": 4.90,   # 12.4 cm
    "double": 6.93,    # 17.6 cm
    # Beyond the JNE print grid, for multi-panel *analysis* figures only (Hyp C mechanism panels, the
    # placement strip, the regime heatmaps). Added 2026-09-27: an 8-panel row at 'double' leaves 1.7 in
    # per panel, which tangles tick labels and panel titles. These are diagnostic figures that are
    # cropped or split before submission, so they are allowed to exceed the column width.
    "wide": 10.00,     # 25.4 cm
    "full": 13.50,     # 34.3 cm
}

DEFAULT_WIDTH: str = "single"
#: Height / width of EVERY panel, whatever the panel count (settled 2026-09-25): a row of many panels no
#: longer flattens into a strip, so trajectories stay separable and panel titles have room.
PANEL_ASPECT: float = 1.5
#: Extra figure height (inches) for the suptitle and a legend placed outside the axes, so the constrained
#: layout takes that room from here rather than from the panels.
DECORATION_HEIGHT: float = 0.6
#: Lower display limit of every R^2 axis. The data are never clipped, only the view: a poorly fitted
#: surrogate reaches R^2 far below it (GP-fixed reaches -55), which would flatten every other curve.
R2_AXIS_FLOOR: float = -1.0
#: Layout engine that keeps axis labels, titles and shared legends apart.
LAYOUT_ENGINE: str = "constrained"
#: Colormaps of the regime surfaces (S5/S10): sequential for a metric, diverging (centred on 0) for a
#: model difference. Both are perceptually uniform and colour-blind safe.
SEQUENTIAL_CMAP: str = "viridis"
DIVERGING_CMAP: str = "RdBu_r"
#: Colour of the outline marking non-inferior cells on a difference surface.
OUTLINE_COLOR: str = "#000000"

FONT_SIZES: dict[str, float] = {
    "base": 9.0,
    "axis_label": 9.0,
    "tick": 8.0,
    "legend": 8.0,
    "title": 9.0,
    "panel_letter": 10.0,
    "annotation": 7.5,
}

LINE_WIDTH: float = 0.9
MARKER_SIZE: float = 3.0
BAND_ALPHA: float = 0.18   # shaded CI bands
FAINT_ALPHA: float = 0.20  # per-channel spaghetti lines
SAVE_DPI: int = 600
FONT_FAMILY: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
#: A Matplotlib linestyle: a named style (``'-'``, ``'--'``, ...) or an ``(offset, (on, off, ...))`` dash
#: pattern, which the GP family needs once it has more members than named dashed styles.
Linestyle = Union[str, tuple[float, tuple[float, ...]]]


@dataclass(frozen=True)
class ModelStyle:
    """Canonical plotting identity of one surrogate model.

    Attributes:
        label: Display string used in legends, tables and captions.
        color: Hex colour (Okabe-Ito family).
        linestyle: Matplotlib linestyle: a named style or an ``(offset, dashes)`` pattern.
        marker: Matplotlib marker.
        family: ``'pfn'``, ``'gp'`` or ``'baseline'``.
        zorder: Draw order; headline models sit on top.
    """

    label: str
    color: str
    linestyle: Linestyle = "-"
    marker: str = "o"
    family: str = "pfn"
    zorder: int = 2


# ---------------------------------------------------------------------------
# Palette: two anchors, everything else derived
# ---------------------------------------------------------------------------
#: The only two literal model colours in the package (Okabe-Ito blue / vermillion).
#: Every other model and acquisition colour is derived from them by
#: :func:`derive_shade`, so the palette is a rule rather than a list.
ANCHOR_PFN: str = "#0072B2"   # TabPFN-2.5 (cool = PFN family)
ANCHOR_GP: str = "#D55E00"    # GP-MLL (warm = GP family)
NEUTRAL_GREY: str = "#999999"  # non-learning baselines

#: Colours of the *reference* bands in the Hyp C placement figures (prior floor, noise ceiling).
#: A **crimson/pink family**, chosen 2026-09-30, and deliberately NOT the model anchors: until 2026-09-27 the
#: floor was ANCHOR_PFN and the ceiling ANCHOR_GP, which mean TabPFN-2.5 and GP-MLL in every other figure, so
#: a reader trained on the rest of the paper read the noise ceiling as the GP. Greys fixed that but competed
#: with NEUTRAL_GREY (the non-learning baselines) and with every unlabelled line in a busy panel, so the
#: references are now chromatic in the one hue region no other token occupies: ANCHOR_PFN is blue, ANCHOR_GP
#: vermillion, and LEVEL_CMAP (viridis) runs blue-purple -> green -> yellow, leaving crimson/pink free. The
#: two are told apart by lightness, an edge linestyle and a legend entry. (They were also the two ends of the
#: placement-surface colormap until 2026-10-07; that heatmap now uses SEQUENTIAL_CMAP, whose lightness range the
#: crimson -> pink ramp lacked.)
REFERENCE_FLOOR_COLOR: str = "#C2185B"
REFERENCE_CEILING_COLOR: str = "#F48FB1"
#: Fill alpha and edge style of those bands (a dark band at BAND_ALPHA is indistinguishable from a light one).
REFERENCE_BAND_ALPHA: float = 0.25
REFERENCE_FLOOR_LINESTYLE: str = "-"
REFERENCE_CEILING_LINESTYLE: str = "--"
#: Named tick positions of a **rank**-placement colourbar or axis, as ``(value, label)`` pairs. The rank's
#: endpoints are not the ratio's: 0 means the channel sits below *every* reference map and 1 above every one,
#: while 0.5 means it falls between the two reference clouds -- so the ratio's "prior floor" / "noise ceiling"
#: wording cannot be reused here. Lives in style.py because it is an axis string (G4).
RANK_PLACEMENT_TICKS: tuple[tuple[float, str], ...] = (
    (0.0, "0 (below all refs.)"),
    (0.5, "0.5 (between)"),
    (1.0, "1 (above all refs.)"),
)
#: Legend label of the between-clouds guide line at 0.5 on a rank-placement axis.
RANK_PLACEMENT_MIDLINE_LABEL: str = "Between references (rank = 0.5)"
#: Named tick positions of the frozen **ratio** placement's colourbar, kept distinct from the rank's above.
RATIO_PLACEMENT_TICKS: tuple[tuple[float, str], ...] = (
    (0.0, "0 (prior floor)"),
    (0.5, "0.5"),
    (1.0, "1 (noise ceiling)"),
)
#: Sequential colormap for an ordered *stress level* series (the K2 ladder in the placement trajectory).
#: Replaces grey shades, which collided with the achromatic reference bands above.
LEVEL_CMAP: str = "viridis"

#: Colour of the **default-regime marker**: the vertical slice naming the unstressed level of the dataset in
#: every stress figure (user, 2026-10-12). A deep violet, the last hue region no other token occupies --
#: ANCHOR_PFN is blue, ANCHOR_GP vermillion, REFERENCE_* crimson/pink, LEVEL_CMAP viridis. It is chromatically
#: closest to viridis's dark end, so a level series drawn with LEVEL_CMAP must start above its floor (the
#: mechanism figures sample ``0.1 + 0.8 * i / (n - 1)``, which is still blue-violet: the marker is told apart
#: from it by being a dashed *vertical* line with an arrow and a label, never a data series).
NOMINAL_MARKER_COLOR: str = "#6A1B9A"
#: Linestyle and width of that marker, so every figure draws the same line.
NOMINAL_MARKER_LINESTYLE: str = (0, (4, 2))
NOMINAL_MARKER_LINEWIDTH: float = 1.0

#: Horizontal guide at the value a perfectly calibrated predictive scores (coverage 0.5 / 0.9, ECE 0) in the
#: E12 calibration-under-stress figure (2026-10-21). A dark grey, thin and solid, drawn under the data: darker
#: than NEUTRAL_GREY so it is never read as the Random baseline, and horizontal and labelled so it is never
#: read as a PFN (solid) model line.
CALIBRATION_TARGET_COLOR: str = "#4D4D4D"
CALIBRATION_TARGET_LINESTYLE: str = "-"
CALIBRATION_TARGET_LINEWIDTH: float = 0.6
CALIBRATION_TARGET_LABEL: str = "Perfect calibration"
#: Legend entry of the open marker that the E10 breakdown panel draws at the harshest level of the ladder when a
#: model never breaks down at that budget (a gap would read as "missing", not as "robust").
NO_BREAKDOWN_LABEL: str = "Open marker: no breakdown on the ladder"
#: Panel title of the E10 breakdown panel. The breakdown rule is *relative to the reference model* (the reason
#: ``breakdown_vs_budget.svg`` was dropped), so the panel states it where it is read.
BREAKDOWN_RULE_NOTE: str = "Rule: non-inferior to {reference} (margin {margin:g})"
#: Wording of each ``experiment.online_y_scaler`` setting, for figure titles: calibration is only comparable
#: within one setting, so every calibration figure states the one it was computed under.
#: What the E12 calibration metrics are scored against, for figure titles. ``bo_runner`` scores the predictive of
#: the *noisy observations* against the *noise-free* ground-truth mean map, so every figure names that target
#: (user, 2026-10-21) instead of letting "coverage" read as coverage of new trials.
CALIBRATION_TARGET_LABELS: dict[str, str] = {
    "ground_truth_mean": "coverage of the ground-truth mean map",
}
ONLINE_Y_SCALER_LABELS: dict[str, str] = {
    "none": "offline y-scaling",
    "minmax": "online y min-max",
    "zscore": "online y z-score",
}

#: Lightness band (HLS) that derived shades walk; bounded so shades stay mutually
#: distinguishable, lighter than either anchor, and never wash out on white.
SHADE_LIGHTNESS_BAND: tuple[float, float] = (0.48, 0.78)


def derive_shade(
    anchor_hex: str,
    rank: int,
    n: int,
    band: tuple[float, float] = SHADE_LIGHTNESS_BAND,
) -> str:
    """Derive the ``rank``-th of ``n`` shades of an anchor colour.

    The hue and saturation of the anchor are held; lightness walks monotonically
    over ``band``. Deterministic and order-stable (stdlib ``colorsys`` only).

    Args:
        anchor_hex: Anchor colour, ``'#RRGGBB'``.
        rank: Zero-based position among the ``n`` shades.
        n: Number of shades in the family (>= 1).
        band: ``(low, high)`` HLS lightness range to walk.

    Returns:
        Hex colour ``'#RRGGBB'``.

    Raises:
        ValueError: If ``rank`` is outside ``[0, n)`` or the band is invalid.
    """
    if not 0 <= rank < n:
        raise ValueError(f"derive_shade: rank {rank} outside [0, {n}).")
    lo, hi = band
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"derive_shade: invalid lightness band {band}.")
    r, g, b = (int(anchor_hex[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    h, _, sat = colorsys.rgb_to_hls(r, g, b)
    frac = 0.0 if n == 1 else rank / (n - 1)
    r2, g2, b2 = colorsys.hls_to_rgb(h, lo + frac * (hi - lo), sat)
    return "#{:02X}{:02X}{:02X}".format(*(round(c * 255) for c in (r2, g2, b2)))


def _srgb_to_lab(hex_color: str) -> np.ndarray:
    """Convert ``'#RRGGBB'`` to CIE L*a*b* (D65).

    Args:
        hex_color: Colour, ``'#RRGGBB'``.

    Returns:
        Array ``[3]`` holding ``L*, a*, b*``.
    """
    rgb = np.array([int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])          # [3]
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)       # [3]
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ lin
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])                                     # [3] relative to D65 white
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)      # [3]
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def perceptual_distance(hex_a: str, hex_b: str) -> float:
    """CIE76 colour difference (Delta E) between two colours; above ~20 they read as clearly different.

    Args:
        hex_a: First colour, ``'#RRGGBB'``.
        hex_b: Second colour, ``'#RRGGBB'``.

    Returns:
        The Euclidean distance in L*a*b*.
    """
    return float(np.linalg.norm(_srgb_to_lab(hex_a) - _srgb_to_lab(hex_b)))


#: Hue band (HLS hue, 0-1) each family's derived colours may occupy. The families never overlap, so the
#: PFN-vs-GP reading always survives, and each family's *anchor* keeps the canonical cool/warm identity.
#:
#: Widened 2026-09-27. The previous PFN band (0.47, 0.68) was only 0.21 wide, so six non-anchor PFNs were
#: all some shade of blue-violet: TabICL and TabFM sat 44 dE apart and read as the same colour in a line
#: plot. Widening the bands (PFN green -> cyan -> blue -> violet -> magenta, GP red -> orange -> gold)
#: lifts the worst within-family separation from 26.6 to 40.9 dE, with TabICL-vs-TabFM at 190 dE. The
#: allocation is still over every *registered* model, not the models in one figure, because a model must
#: keep one colour across every figure in the paper.
FAMILY_HUE_BAND: dict[str, tuple[float, float]] = {"pfn": (0.33, 0.92), "gp": (0.0, 0.20)}
#: Candidate lightness / saturation / hue-step grids searched by :func:`derive_family`. The lightness cap
#: keeps every derived colour visible on white.
DERIVED_LIGHTNESS: tuple[float, ...] = (0.25, 0.32, 0.4, 0.48, 0.56, 0.64)
DERIVED_SATURATION: tuple[float, ...] = (0.45, 0.65, 0.85)
DERIVED_HUE_STEPS: int = 15
#: CIE L* window a derived colour must fall in: dark enough to read on white, light enough to stay off black.
DERIVED_LSTAR_RANGE: tuple[float, float] = (32.0, 68.0)


def derive_family(
    anchor_hex: str,
    n: int,
    hue_band: tuple[float, float],
    *,
    avoid: Sequence[str] = (),
) -> tuple[str, ...]:
    """Derive ``n`` mutually distinguishable colours of one family from its anchor.

    Greedy farthest-point selection in CIE L*a*b*: each pick is the candidate (hue inside ``hue_band``,
    lightness and saturation from fixed grids) whose *smallest* distance to everything already chosen is
    largest. The first picks are therefore maximally different from the anchor and from each other, and
    later ones fill the gaps. Deterministic: it depends only on the arguments.

    Args:
        anchor_hex: The family anchor, ``'#RRGGBB'``; never returned, always kept at a distance.
        n: Number of colours to derive (>= 0).
        hue_band: ``(low, high)`` HLS hue range the picks must stay inside.
        avoid: Further colours to stay away from (the other family's anchor, the baseline grey).

    Returns:
        ``n`` hex colours, in pick order (pick 0 is the most distinct from the anchor).

    Raises:
        ValueError: If ``n`` is negative or the band is invalid.
    """
    if n < 0:
        raise ValueError(f"derive_family: n={n} must be >= 0.")
    lo, hi = hue_band
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"derive_family: invalid hue band {hue_band}.")
    candidates: list[str] = []
    for step in range(DERIVED_HUE_STEPS):
        hue = lo + (hi - lo) * step / (DERIVED_HUE_STEPS - 1)
        for light in DERIVED_LIGHTNESS:
            for sat in DERIVED_SATURATION:
                r, g, b = colorsys.hls_to_rgb(hue, light, sat)
                hex_color = "#{:02X}{:02X}{:02X}".format(*(round(c * 255) for c in (r, g, b)))
                if DERIVED_LSTAR_RANGE[0] <= _srgb_to_lab(hex_color)[0] <= DERIVED_LSTAR_RANGE[1]:
                    candidates.append(hex_color)
    chosen: list[str] = []
    fixed = [anchor_hex, *avoid]
    for _ in range(n):
        taken = fixed + chosen
        best = max(
            (c for c in dict.fromkeys(candidates) if c not in taken),
            key=lambda c: (min(perceptual_distance(c, t) for t in taken), c),
        )
        chosen.append(best)
    return tuple(chosen)


#: Legend / table ordering: headline pair first, then the GP ladder, then baselines.
#: It also fixes each derived model's pick rank within its family (:func:`derive_family`), so the models that
#: are plotted together most often (TabICL next to TabPFN, GP-fixed next to GP-MLL) come first and get the
#: most distinct colours.
MODEL_ORDER: tuple[str, ...] = (
    "tabpfn_v2_5",
    "gp_mll",
    "gp_naive",
    "gp_oracle",
    "gp_deep_kernel",
    "gp_mll_lbfgs",
    "gp_bayes",  # added 2026-10-15 without moving any existing GP colour
    "gp_mll_frozen",  # last GP member: added 2026-10-19 (Hypothesis C1 reference) without moving any GP colour
    "tabicl",
    "tabpfn_v3_5",
    "causilo",
    "tabfm",
    "pfns4bo",
    "tabflex",
    "mitra",
    "random",
)

#: ``key -> (label, linestyle, marker, family, zorder)``; colours come from the anchors.
_MODEL_SPECS: dict[str, tuple[str, Linestyle, str, str, int]] = {
    "tabpfn_v2_5": ("TabPFN-2.5", "-", "o", "pfn", 5),
    "pfns4bo": ("PFNs4BO", "-", "^", "pfn", 3),
    "tabflex": ("TabFlex", "-", "<", "pfn", 3),
    "tabfm": ("TabFM", "-", ">", "pfn", 3),
    "mitra": ("Mitra", "-", "P", "pfn", 3),
    "tabicl": ("TabICL", "-", "d", "pfn", 3),
    "tabpfn_v3_5": ("TabPFN-3.5", "-", "h", "pfn", 4),
    "causilo": ("Causilo", "-", "H", "pfn", 3),
    "gp_mll": ("GP-MLL", "--", "s", "gp", 4),
    "gp_mll_lbfgs": ("GP-MLL (L-BFGS)", "--", "p", "gp", 4),
    "gp_naive": ("GP-fixed", "-.", "D", "gp", 3),
    # Hyperparameters marginalised on a grid (OMEGA, 2026-10-15): the exact-Bayes reference with learning.
    "gp_bayes": ("GP-Bayes", (0, (5, 1.5, 1, 1.5, 1, 1.5)), "v", "gp", 4),
    # Type-II ML hyperparameters fitted on D, frozen through the update: the GP-like reference of Hypothesis C1 only.
    "gp_mll_frozen": ("GP-MLL frozen", (0, (1, 1)), "<", "gp", 4),
    "gp_oracle": ("GP-oracle", ":", "*", "gp", 3),
    "gp_deep_kernel": ("GP-deep", "--", "X", "gp", 2),
    "random": ("Random", ":", "x", "baseline", 1),
}

_FAMILY_ANCHOR: dict[str, str] = {"pfn": ANCHOR_PFN, "gp": ANCHOR_GP}
_ANCHOR_KEYS: frozenset[str] = frozenset({"tabpfn_v2_5", "gp_mll"})


def _build_model_styles() -> dict[str, ModelStyle]:
    """Build :data:`MODEL_STYLES`: anchors verbatim, every other model derived."""
    derived: dict[str, list[str]] = {
        fam: [k for k in MODEL_ORDER if _MODEL_SPECS[k][3] == fam and k not in _ANCHOR_KEYS]
        for fam in _FAMILY_ANCHOR
    }
    palette: dict[str, tuple[str, ...]] = {
        fam: derive_family(
            anchor,
            len(derived[fam]),
            FAMILY_HUE_BAND[fam],
            avoid=(NEUTRAL_GREY, *(a for f, a in _FAMILY_ANCHOR.items() if f != fam)),
        )
        for fam, anchor in _FAMILY_ANCHOR.items()
    }
    styles: dict[str, ModelStyle] = {}
    for key, (label, ls, marker, fam, z) in _MODEL_SPECS.items():
        if fam == "baseline":
            color = NEUTRAL_GREY
        elif key in _ANCHOR_KEYS:
            color = _FAMILY_ANCHOR[fam]
        else:
            color = palette[fam][derived[fam].index(key)]
        styles[key] = ModelStyle(label, color, ls, marker, fam, zorder=z)
    return styles


MODEL_STYLES: dict[str, ModelStyle] = _build_model_styles()

#: Aliases from legacy result-dict ``model_type`` strings to canonical keys.
MODEL_ALIASES: dict[str, str] = {
    "gp": "gp_mll",
    "exact_gp": "gp_mll",
    "mll_gp": "gp_mll",
    "naive_gp": "gp_naive",
    "oracle_gp": "gp_oracle",
    "deep_kernel_gp": "gp_deep_kernel",
    "vanilla_tabpfn": "tabpfn_v2_5",
    "tabpfn": "tabpfn_v2_5",
    "tabpfn_v2": "tabpfn_v2_5",
    "pfn": "tabpfn_v2_5",
    "random_search": "random",
}

def _canonical_key(model: str) -> str:
    """Resolve a raw model identifier to a key of :data:`MODEL_STYLES`.

    Args:
        model: Raw identifier (canonical key, legacy ``model_type`` or label).

    Returns:
        A key of :data:`MODEL_STYLES`.

    Raises:
        KeyError: If the identifier is unknown (fail fast rather than plot a
            silently miscoloured series).
    """
    key = str(model).strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")
    key = MODEL_ALIASES.get(key, key)
    if key not in MODEL_STYLES:
        label_map = {
            st.label.lower().replace("-", "_").replace(" ", "_").replace(".", "_"): k
            for k, st in MODEL_STYLES.items()
        }
        if key in label_map:
            return label_map[key]
        raise KeyError(
            f"Unknown model identifier {model!r} (resolved to {key!r}). "
            f"Add it to MODEL_STYLES in visualization/style.py; "
            f"known keys: {sorted(MODEL_STYLES)}"
        )
    return key


def model_style(model: str) -> ModelStyle:
    """Return the canonical :class:`ModelStyle` for ``model`` (alias-tolerant)."""
    return MODEL_STYLES[_canonical_key(model)]


def model_label(model: str) -> str:
    """Return the canonical display label for ``model``."""
    return model_style(model).label


def model_color(model: str) -> str:
    """Return the canonical hex colour for ``model``."""
    return model_style(model).color


#: Acquisition order fixes its shade rank; the first is drawn in the model's own colour.
ACQUISITION_ORDER: tuple[str, ...] = (
    "ei", "ucb", "ts_marginal", "ts_joint", "pi", "greedy", "random",
)

#: Linestyle carries acquisition identity, so a model's rows read as one colour family.
ACQUISITION_LINESTYLES: dict[str, str] = {
    "ei": "-",
    "ucb": "--",
    "ts_marginal": "-.",
    "ts_joint": ":",
    "pi": (0, (5, 1, 1, 1, 1, 1)),
    "greedy": (0, (1, 1)),
    "random": (0, (3, 1, 1, 1)),
}


def acquisition_style(model: str, acq_type: str) -> ModelStyle:
    """Return the style of one (model, acquisition) series.

    Colour: the model's colour, shaded by acquisition rank (the first acquisition,
    ``ei``, keeps the model colour exactly). Linestyle carries the acquisition.

    Args:
        model: Model identifier (alias-tolerant).
        acq_type: Acquisition type; must be in :data:`ACQUISITION_ORDER`.

    Returns:
        A :class:`ModelStyle` whose label reads ``'<model> (<acq>)'``.

    Raises:
        KeyError: If ``acq_type`` is unknown (fail fast, never a silent default).
    """
    if acq_type not in ACQUISITION_ORDER:
        raise KeyError(
            f"Unknown acquisition {acq_type!r}; add it to ACQUISITION_ORDER in "
            f"visualization/style.py (known: {list(ACQUISITION_ORDER)})."
        )
    base = model_style(model)
    rank = ACQUISITION_ORDER.index(acq_type)
    color = (
        base.color
        if rank == 0
        else derive_shade(base.color, rank - 1, len(ACQUISITION_ORDER) - 1)
    )
    return ModelStyle(
        f"{base.label} ({acq_type})",
        color,
        ACQUISITION_LINESTYLES[acq_type],
        base.marker,
        base.family,
        base.zorder,
    )


#: Linestyles cycled when one model carries several acquisition configs that are not registered
#: acquisition types (e.g. the fixed-kappa UCB grid).
SERIES_LINESTYLES: tuple[object, ...] = ("-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)))


def series_style(model: str, label: str, labels: Sequence[str]) -> ModelStyle:
    """Style of one (model, acquisition-config) series in a multi-series figure.

    A model with a single series keeps its own style; a model with several is shaded
    and dashed by acquisition (registered types via :func:`acquisition_style`, other
    config names such as ``ucb_k2`` by their rank within the model).

    Args:
        model: Model identifier.
        label: This series' acquisition config name.
        labels: Every acquisition config name that this model carries in the figure.

    Returns:
        The :class:`ModelStyle` for the series.
    """
    if len(labels) <= 1:
        return model_style(model)
    if label in ACQUISITION_ORDER:
        return acquisition_style(model, label)
    ordered = sorted(labels)
    idx = ordered.index(label)
    base = model_style(model)
    color = base.color if idx == 0 else derive_shade(base.color, idx - 1, len(ordered) - 1)
    return ModelStyle(
        f"{base.label} ({label})", color, SERIES_LINESTYLES[idx % len(SERIES_LINESTYLES)],
        base.marker, base.family, base.zorder,
    )


def model_palette(
    models: Iterable[str] | None = None,
    *,
    by_label: bool = True,
) -> dict[str, str]:
    """Build a seaborn-compatible ``{hue_value: colour}`` palette.

    Args:
        models: Model identifiers to include; defaults to :data:`MODEL_ORDER`.
        by_label: If True the keys are display labels (matching a ``hue`` column
            that already holds labels); otherwise canonical keys.

    Returns:
        Mapping usable as ``palette=`` in seaborn calls.
    """
    keys = [_canonical_key(m) for m in (models if models is not None else MODEL_ORDER)]
    return {
        (MODEL_STYLES[k].label if by_label else k): MODEL_STYLES[k].color for k in keys
    }


def plot_kwargs(model: str, **overrides: object) -> dict[str, object]:
    """Return ``ax.plot`` kwargs for ``model`` (colour, linestyle, marker, label, zorder).

    Args:
        model: Model identifier.
        **overrides: Keyword arguments overriding the canonical defaults.

    Returns:
        Keyword-argument dict for ``ax.plot`` / ``ax.errorbar``.
    """
    st = model_style(model)
    kw: dict[str, object] = {
        "color": st.color,
        "linestyle": st.linestyle,
        "marker": st.marker,
        "markersize": MARKER_SIZE,
        "linewidth": LINE_WIDTH,
        "label": st.label,
        "zorder": st.zorder,
    }
    kw.update(overrides)
    return kw


# ---------------------------------------------------------------------------
# Wording (axes, knobs, demos, datasets) - keep identical across all figures.
# ---------------------------------------------------------------------------
AXIS_LABELS: dict[str, str] = {
    "r2": r"$R^2$ of surrogate",
    # Heavy-tailed metrics are summarized by the median with an interquartile band over channels
    # (a single badly fitted channel would otherwise dominate the mean); the aggregation is in the label.
    "r2_median": "$R^2$ of surrogate\n(median, IQR over channels)",
    "mean_query_latency_s_median": "Per-query latency (s)\n(median, IQR over channels)",
    "spearman": r"Spearman $\rho$",
    "simple_regret": "Simple regret (0-1)",
    "exploration_score": "Exploration score",
    "final_regret": "Final regret (norm. GT range)",
    "recommended_regret": "Recommended-site regret (norm. GT range)",
    "best_queried_regret": "Best-queried regret (norm. GT range)",
    "cumulative_regret": "Cumulative regret (norm. GT range)",
    "budget": r"BO iterations (incl. $n_\mathrm{init}$)",
    # A8 / A9 (2026-09-30).
    "reached_fraction": "Runs reaching the target",
    "elapsed_s": "Cumulative wall clock (s)",
    "queries_to_target": "Queries to target",
    "iteration": "BO iteration",
    "achieved_snr_db": "Achieved SNR (dB)",
    "target_delta_snr_db": "SNR degradation vs channel floor (dB)",
    "achieved_contamination": r"Achieved contamination (fraction of trials)",
    "achieved_failure": "Failed electrodes (fraction of array)",
    "n_survivors": "Surviving electrodes",
    "alpha": r"Residual amplification $\alpha$",
    "coverage_50": "50% interval coverage",
    "coverage_90": "90% interval coverage",
    "ece": "Expected calibration error",
    "nll": "Negative log-likelihood (nats)",
    "crps": "CRPS (norm. GT range)",
    "cvar10_regret": r"CVaR$_{10\%}$ of final regret",
    "mean_query_latency_s": "Per-query latency (s)",
    "top1_hit": "Top-1 hit rate",
    "top3_hit": "Top-3 hit rate",
    "opt_distance": "Recommendation-to-optimum distance (electrode pitch)",
    "regret_difference": "Regret difference vs GP-MLL (norm. GT range)",
    "morans_i": "Moran's I",
    "skewness": "Skewness",
    "cv": "Coefficient of variation",
    "mean_sd_corr": "Mean-SD correlation",
    "degradation_auc": "Degradation AUC (norm.)",
    "reliability": "Split-half reliability (Spearman-Brown)",
    # Hyp C mechanism figures (tasks #5, #6, #9).
    "context_t": "Context size $t$ (observations)",
    "psd_neg_mass": "Implied-covariance negative-eigenvalue mass",
    "placement": "Placement $p$ (0 = floor, 1 = ceiling)",
    # Rank placement (2026-10-14). A DIFFERENT reading from ``placement`` above, and the two must never share a
    # label: the ratio is 0 at the prior floor and 1 at the noise ceiling, whereas the rank is the fraction of the
    # reference maps the channel exceeds -- 0 below every reference, 1 above every one, and 0.5 when the channel
    # falls BETWEEN the two reference clouds. Bounded in [0, 1] by construction, so no clipping and no dropped
    # layers; the ratio's denominator (median gap 0.049 on NHP) is what put 51 % of its values outside [0, 1].
    "p_rank": "Rank placement (fraction of references exceeded)",
    "rank_vs_floor": "Rank vs prior floor (fraction exceeded)",
    "rank_vs_ceiling": "Rank vs noise ceiling (fraction exceeded)",
    # BETA representation analysis (2026-10-21). CKA and Procrustes share their invariance group (translation,
    # isotropic scale, orthogonal maps) and differ in how they weight the cross-covariance spectrum: CKA by a
    # Frobenius norm (leading directions dominate), NBS by a nuclear norm (every direction counts;
    # analysis/procrustes.py). Every statistic here is read against the smooth-field surrogate (BETA D3d); the
    # deconfounded CKA removes 4-bandwidth geometry (and GP-fixed's covariance for covariance targets).
    "cka_debiased": "Debiased CKA",
    "cka_deconfounded": "Deconfounded debiased CKA\n(geometry removed)",
    "headline_deconfounded_vs_surrogate": "Deconfounded debiased CKA\n(geometry removed)",
    "headline_floor_subtracted_raw": "Debiased CKA minus\nsurrogate median",
    "nbs": "Normalised Bures similarity (NBS)",
    "shape_distance": r"Procrustes shape distance $\arccos$(NBS) (rad)",
    "bures": "Bures distance (z-units)",
    "bures_rel": r"Bures distance / $\sqrt{\mathrm{tr}\,\Sigma_\mathrm{ref}}$",
    "trace_ratio": r"Trace ratio $\mathrm{tr}\,C_\mathrm{PFN}$ / $\mathrm{tr}\,\Sigma_\mathrm{ref}$",
    "mantel_r": r"Mantel Spearman $r$",
    "w2_site": "Per-site W2 (z-units)",
    "w2_over_sd": "Per-site W2 / reference SD",
    "w2_location": "Location term (z-units$^2$)",
    "w2_spread": "Spread term (z-units$^2$)",
    "w2_shape": "Shape term (z-units$^2$)",
    "lens_w2_over_sd": "Lens W2 / reference SD\n(median over sites)",
    "layer": "Transformer layer",
    # OMEGA bayes_update (2026-10-15). Part A: every coherent Bayesian model scores 0 drift, a shrink / move
    # ratio of 1 and 0 asymmetry, whatever its prior. Part B: TabPFN's update against a GP reference, matched per
    # site before any aggregate. ``cov_asymmetry`` is the tower-property covariance -- never the retired
    # ``sym_err``, which divided by (1 - gain) and so assumed homoscedastic noise.
    # Hyp C labels sit in grids of four narrow panels, so the long ones break over two lines (user, 2026-10-22).
    "drift_abs": "Mean drift $|\\delta|$\n(predictive SD)",
    "drift_w1": "Distributional drift W1 (predictive SD)",
    "shrink_move_ratio": "Expected shrink / mean movement (1 = Bayes)",
    "total_var_ratio": "Total-variance ratio\n(1 = Bayes)",
    "cov_asymmetry": "Implied-covariance\nasymmetry (0 = Bayes)",
    "latent_frac": "Latent share of predictive\nvariance at $x'$",
    "implied_noise": r"Implied noise variance at $x'$ (z-units$^2$)",
    "trial_var": r"Trial variance at $x'$ (z-units$^2$)",
    "sigma_recovery_rho": r"Spearman $\rho$, implied noise vs trial variance",
    "gain": r"Gain $T(x')$",
    "transfer_slope": "Transfer slope\nvs reference",
    "footprint_r": "Footprint correlation\n(Pearson $r$)",
    "gap_change": "Change of W1 gap to\nreference (ref. SD)",
    "delta_crps": "$\\Delta$CRPS, model $-$\nreference (z-units)",
    "saturation": "Saturation\n(1 = linear response)",
    # The ladder unit is the predictive SD of the run's ladder reference (bayes_update.shared.reference, user
    # 2026-10-21); the surprise figures name that engine in their title (SURPRISE_REFERENCE_TITLE).
    "ladder_c": "Injected surprise $c$\n(ladder-reference SD at $x'$)",
    "update_anchor_sd": "Update at $x'$\n(ladder-reference SD at $x'$)",
    "surprise_slope": "Slope of the update at $x'$\nvs surprise (1 = full update)",
    "implied_cov": r"Implied covariance $C(x', x)$",
    "far_field_share": "Far-field share\nof update energy",
    "abs_transfer": "Mean $|T(x)|$\n(update per unit surprise)",
    "distance_pitch": r"Distance from $x'$ (electrode pitches)",
    "dmu_exemplar": r"Change of mean $\Delta\mu(x)$ (z-units)",
    # M8 predictive link (2026-10-21): the per-channel breakdown outcome is a count on the stress ladder (levels
    # passed before the first non-inferiority failure; the ladder length = never breaks down).
    "breakdown": "Breakdown point\n(stress levels passed)",
    # E12 calibration under stress (2026-10-21): the predictive is scored against the noise-free ground-truth mean
    # map, not against held-out noisy trials (bo_runner), and the label says so; repetitions reduced within a
    # channel, then the median and the interquartile range over channels.
    "coverage_50_gt_median": "50% interval coverage of GT mean\n(median, IQR over channels)",
    "coverage_90_gt_median": "90% interval coverage of GT mean\n(median, IQR over channels)",
    "ece_gt_median": "Expected calibration error vs GT mean\n(median, IQR over channels)",
    "nll_gt_median": "NLL of GT mean (nats)\n(median, IQR over channels)",
    # E10 robustness vs BO budget (2026-10-21). The breakdown point is in knob-axis units, so its label is this
    # prefix plus the knob's own x-axis wording; the two regret summaries average the per-level values over the
    # ladder (each level weighted equally).
    "breakdown_point": "Pooled breakdown point (95% channel-bootstrap CI)",
    "recommended_regret_median_level_mean": "Recommended-site regret (norm. GT range)\n(median over channels, mean over levels)",
    "cvar10_level_mean": "CVaR$_{10\\%}$ of recommended regret\n(mean over levels)",
}

#: Linestyle of each embedding readout (B1 Step 3; BETA 2026-10-21). Readouts are told apart by linestyle so that
#: colour stays free for the context size or stress level of the series; never a new hex colour. The label token is
#: the reported readout (solid); the feature-token mean is a diagnostic line (G6).
READOUT_LINESTYLES: dict[str, str] = {
    "label": "-",
    "feature_mean": ":",
    "lens": "--",
    "feature_tokens": "-.",
}

#: Legend text of each readout.
READOUT_LABELS: dict[str, str] = {
    "label": "Label token",
    "feature_mean": "Feature-token mean (diagnostic)",
    "lens": "Lens predictive (quantile embedding)",
    "feature_tokens": "Per feature token",
}

#: Panel titles of the BETA belief targets (D3c / D3e). Location and spread are separate targets, never averaged.
BELIEF_TARGET_LABELS: dict[str, str] = {
    "mean_own": "TabPFN-2.5 mean map",
    "sd_own": "TabPFN-2.5 SD map",
    "quant_own": "TabPFN-2.5 quantile embedding",
    "mean_mll": "GP-MLL refit mean map",
    "sd_mll": "GP-MLL refit SD map",
    "quant_mll": "GP-MLL refit quantile embedding",
    "cov_mll": r"GP-MLL refit covariance $\Sigma_\mathrm{MLL}$",
    "mean_fixed": "GP-fixed mean map",
    "sd_fixed": "GP-fixed SD map",
    "quant_fixed": "GP-fixed quantile embedding",
    "cov_fixed": r"GP-fixed covariance $\Sigma_\mathrm{fixed}$",
    "mean_bayes": "GP-Bayes mean map",
    "sd_bayes": "GP-Bayes SD map",
    "quant_bayes": "GP-Bayes quantile embedding",
    "cov_bayes": r"GP-Bayes covariance $\Sigma_\mathrm{B}$",
    "K_X": r"Electrode geometry $K_X$",
    "K_GT": r"Ground-truth map $K_\mathrm{GT}$",
}

#: Linestyle of the two site pools of a site-matched readout (β1a; never pooled).
SITE_POOL_LINESTYLES: dict[str, str] = {"unobserved": "-", "observed": "--"}

#: Legend text of the smooth-field surrogate chance level (BETA D3d), drawn in :data:`NEUTRAL_GREY`: a chance level
#: is a non-learning baseline, never a model anchor nor the prior / noise reference colours of the placement figures.
SURROGATE_BAND_LABEL: str = "Smooth-field surrogate (95th pct.)"

#: Subject identity in per-channel scatters (the M8 predictive link, 2026-10-21). Subjects are an ordered handful
#: (NHP 0-3), so their colours are evenly spaced samples of :data:`SEQUENTIAL_CMAP` over this range (the top of
#: viridis is too light on white), and the marker repeats the identity for greyscale print. No model is drawn in
#: those figures, so no subject colour can be read as a model anchor.
SUBJECT_COLOR_RANGE: tuple[float, float] = (0.0, 0.85)
SUBJECT_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "h")
SUBJECT_LABEL: str = "Subject {subject}"

#: Panel titles of the predictive link: which reference a predictor is read against. ``none`` = the engine's own
#: readout; ``contrast`` = a site-matched contrast with the reference (Part B); ``difference`` = the matched
#: difference engine - reference of an engine readout.
LINK_REFERENCE_TITLES: dict[str, str] = {
    "none": "{model}",
    "contrast": "vs {reference}",
    "difference": "{model} $-$ {reference}",
}
#: Slope annotation of a predictive-link panel (random-intercept slope, channel-bootstrap CI).
LINK_SLOPE_NOTE: str = "slope {slope:.3g} [{lo:.3g}, {hi:.3g}]"
#: Figure title of a predictive-link figure; the link is exploratory and the title says so where it is read.
LINK_TITLE: str = ("Exploratory link to {outcome}: {model}, {pool} anchors, level {level:g}; "
                   "random-intercept slope, 95% channel-bootstrap CI")
#: Figure title of the surprise-response figures (Omega-4, C1): the engine whose predictive SD is the ladder unit.
SURPRISE_REFERENCE_TITLE: str = "Surprise in {reference} predictive SD at $x'$"
#: Hypothesis C engine labels that differ from the model's palette label: GP-MLL refit and GP-MLL frozen share the
#: GP-MLL identity, so the refit one is named in full wherever both can appear (user, 2026-10-22).
ENGINE_LABELS: dict[str, str] = {"gp_mll_refit": "GP-MLL refit"}
#: Legend entry of the implied-noise scatter (Omega-2): median sigma-recovery rho and the share of anchors whose
#: implied noise is <= 0 (off a log axis).
IMPLIED_NOISE_LEGEND: str = r"{model}: median $\rho$ {rho:.2f}; {neg:.0f}% $\leq 0$"
#: Line bands of the Hypothesis C figures: median over channels with a within-subject channel bootstrap interval
#: (BETA D4), replicates and central mass. Replaces the inter-quartile range over cells (user, 2026-10-22).
#: Engines drawn as the median line alone: GP-MLL's channel-to-channel spread is wide even as an interval of the
#: median (its refits often collapse the noise to the floor), and the band hid the other engines (user, 2026-10-22).
HYPC_LINE_ONLY_ENGINES: tuple[str, ...] = ("gp_mll_refit", "gp_mll_frozen")
HYPC_BAND_N_BOOT: int = 2000
HYPC_BAND_CI: float = 0.95
HYPC_BAND_SEED: int = 42


def subject_style(subject: object, subjects: Sequence[object]) -> ModelStyle:
    """Colour, marker and label of one subject among the subjects of a figure.

    Args:
        subject: The subject.
        subjects: Every subject of the figure, in display order.

    Returns:
        A :class:`ModelStyle` (family ``'subject'``, solid line).

    Raises:
        ValueError: If ``subject`` is not in ``subjects``.
    """
    order = list(subjects)
    if subject not in order:
        raise ValueError(f"subject_style: {subject!r} not in {order}.")
    i = order.index(subject)
    lo, hi = SUBJECT_COLOR_RANGE
    frac = 0.0 if len(order) == 1 else i / (len(order) - 1)
    color = mpl.colors.to_hex(mpl.colormaps[SEQUENTIAL_CMAP](lo + frac * (hi - lo)))
    return ModelStyle(SUBJECT_LABEL.format(subject=subject), color, "-", SUBJECT_MARKERS[i % len(SUBJECT_MARKERS)],
                      "subject", 3)

#: Canonical **stress parameter** of each knob, in mathtext (settled with the user 2026-10-12; the catalogue
#: lives in ``research_design.md`` §2 Hyp B). One letter per knob, so a symbol never means two things across
#: the manuscript: the two K2 settings were both called ``alpha`` and the two fractions both ``epsilon``,
#: while ``alpha``/``sigma``/``kappa``/``lambda``/``rho``/``delta`` are already spoken for by the GP, the
#: acquisition functions and the Hyp C update-rule statistics. Mirrors ``StressKnob.symbol``, which is the
#: definition; this map exists so a figure never has to import the knob registry.
KNOB_SYMBOLS: dict[str, str] = {
    "k2_channel": r"$\gamma$",
    "k2_global": r"$\eta$",
    "k5_outliers": r"$\varepsilon$",
    "k6_failure": r"$\varphi$",
}

#: One-line meaning of each stress parameter, for a caption or a legend title.
KNOB_PARAM_LABELS: dict[str, str] = {
    "k2_channel": r"residual gain $\gamma$",
    "k2_global": r"added noise SD $\eta$ (z-units)",
    "k5_outliers": r"contaminated trial slots $\varepsilon$",
    "k6_failure": r"failed electrodes $\varphi$",
}

#: Canonical stress-knob names (roadmap jargon) and their axis strings.
KNOB_LABELS: dict[str, str] = {
    "k2_channel": r"K2 SNR (channel-relative noise, $\gamma$)",
    "k2_global": r"K2 SNR (global noise, $\eta$)",
    "k5_outliers": r"K5 outlier contamination ($\varepsilon$)",
    "k6_failure": r"K6 electrode failure ($\varphi$)",
    "nominal": "Nominal anchor",
}

#: The knob level itself is never the primary x-axis for K2 - achieved SNR is.
KNOB_X_AXIS: dict[str, str] = {
    "k2_channel": "achieved_snr_db",
    "k2_global": "achieved_snr_db",
    "k5_outliers": "level",
    "k6_failure": "level",
}

#: X-axis wording per knob, where the knob's vocabulary beats the column's (two lines, unit in parentheses,
#: so it fits a panel of a three-panel row). K5 and K6 read as levels of the perturbation, not as SNR.
KNOB_X_LABELS: dict[str, str] = {
    "k5_outliers": "Contamination $\\varepsilon$\n(fraction of trial slots)",
    # varphi, not varepsilon: K5's contamination fraction owns epsilon (2026-10-12 notation settlement).
    "k6_failure": "Failed electrodes $\\varphi$\n(fraction of array)",
}


def knob_x_label(knob: str, x_col: str) -> str:
    """X-axis string of a stress figure.

    Args:
        knob: Knob name.
        x_col: Plotted x column (used when the knob has no dedicated wording).

    Returns:
        The knob's own label when :data:`KNOB_X_LABELS` has one, else :func:`axis_label` of the column.
    """
    return KNOB_X_LABELS.get(knob) or axis_label(x_col)


#: Legend/annotation wording of the default regime. Every figure that marks it uses this exact string, so a
#: reader meets one name for it across Hyp B and Hyp C.
NOMINAL_MARKER_LABEL: str = "Default (no stress)"


def mark_nominal(
    ax: plt.Axes,
    x: float,
    *,
    label: str | None = NOMINAL_MARKER_LABEL,
    annotate: bool = True,
    y_frac: float = 0.0,
    axis: str = "x",
) -> None:
    """Mark the dataset's default (unstressed) regime as a violet slice with an arrow.

    The default regime is the one level of a ladder at which nothing has been applied; it is the same
    channel, and therefore the same achieved SNR, for every knob of a dataset (NHP 3.98 dB, spinal 6.30 dB,
    ``5d_rat`` 2.06 dB measured 2026-10-12), so marking it is what lets a reader read two stress figures
    against each other.

    Args:
        ax: Axes to mark.
        x: Position of the default regime, in the marked axis's own units.
        label: Legend entry; ``None`` draws the line without one (for the 2nd..nth panel of a row, so the
            legend holds a single entry).
        annotate: Draw the arrow pointing at the slice from outside the axes.
        y_frac: Offset of the arrow's tail along the *other* axis, in axes fraction.
        axis: ``'x'`` for a vertical slice (the degradation curves), ``'y'`` for a horizontal one (the
            regime heatmap, whose stress ladder runs up the y-axis).

    Raises:
        ValueError: If ``x`` is not finite — a silently skipped marker would read as "no default exists" —
            or if ``axis`` is neither ``'x'`` nor ``'y'``.
    """
    if not np.isfinite(x):
        raise ValueError(f"mark_nominal: x must be finite, got {x!r}.")
    if axis not in ("x", "y"):
        raise ValueError(f"mark_nominal: axis must be 'x' or 'y', got {axis!r}.")
    line = ax.axvline if axis == "x" else ax.axhline
    line(
        x,
        color=NOMINAL_MARKER_COLOR,
        linestyle=NOMINAL_MARKER_LINESTYLE,
        linewidth=NOMINAL_MARKER_LINEWIDTH,
        zorder=2,
        label=label,
    )
    if annotate:
        transform = ax.get_xaxis_transform() if axis == "x" else ax.get_yaxis_transform()
        tail = (x, y_frac) if axis == "x" else (y_frac, x)
        head = (x, y_frac + 0.08) if axis == "x" else (y_frac + 0.08, x)
        ax.annotate(
            "",
            xy=head,
            xytext=tail,
            xycoords=transform,
            textcoords=transform,
            arrowprops={
                "arrowstyle": "-|>",
                "color": NOMINAL_MARKER_COLOR,
                "linewidth": NOMINAL_MARKER_LINEWIDTH,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
        )


def snr_series_label(snr_db: float, *, is_default: bool = False) -> str:
    """Legend entry of one stress level, stated as achieved SNR rather than a raw knob level.

    The Hyp C mechanism figures used to label their stress series by the raw K2 level (``level 1 / 2 / 4``),
    which no reader could place against the Hyp B x-axis. Both are now absolute achieved SNR in dB (user,
    2026-10-12), and the default regime is named as such.

    Args:
        snr_db: Mean achieved SNR of the level, in dB.
        is_default: Whether this level is the dataset's unstressed regime.

    Returns:
        E.g. ``'4.0 dB (default)'`` or ``'-8.1 dB'``.
    """
    return f"{snr_db:.1f} dB" + (" (default)" if is_default else "")


DEMO_LABELS: dict[str, str] = {
    "synthetic": "Synthetic",
    "draws": "Draws (in vivo)",
}

DATASET_LABELS: dict[str, str] = {
    "nhp": "NHP cortex",
    "rat": "Rat cortex",
    "spinal": "Spinal cord",
    "5d_rat": "Rat 5D motor cortex",
    "synthetic_neurostim": "Synthetic neurostim",
    "synthetic_nhp": "Synthetic (NHP-fitted)",
    "synthetic_5d_rat": "Synthetic (5D-rat-fitted)",
    "synthetic_spinal": "Synthetic (spinal-fitted)",
}

GT_MODE_LABELS: dict[str, str] = {
    "full_mean": "Full-mean GT",
    "split_half": "Split-half GT",
}


def axis_label(key: str) -> str:
    """Return the canonical axis string for a metric/column name.

    Falls back to a de-underscored version of ``key`` so a missing entry is
    visible in the figure rather than silently wrong.

    Args:
        key: Metric or column name (e.g. ``'final_regret'``).

    Returns:
        Axis label string (may contain mathtext).
    """
    if key in AXIS_LABELS:
        return AXIS_LABELS[key]
    if key in KNOB_LABELS:
        return KNOB_LABELS[key]
    return key.replace("_", " ").capitalize()


def set_integer_xaxis(ax: plt.Axes) -> None:
    """Force integer-only major ticks on the x-axis (discrete layer / block index).

    Args:
        ax: Axes whose x-axis indexes discrete blocks.
    """
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))


# ---------------------------------------------------------------------------
# rcParams + helpers
# ---------------------------------------------------------------------------
def apply_style() -> None:
    """Install the canonical rcParams.

    Call once at the top of any plotting entry point (``figure`` does it for you).
    """
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(FONT_FAMILY),
            "font.size": FONT_SIZES["base"],
            "axes.labelsize": FONT_SIZES["axis_label"],
            "axes.titlesize": FONT_SIZES["title"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "legend.fontsize": FONT_SIZES["legend"],
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.dpi": 150,
            "savefig.dpi": SAVE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": False,
            # TrueType (Type 42) fonts - required by IOP for vector submissions.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
        }
    )


def figure(
    width: str | float = DEFAULT_WIDTH,
    *,
    nrows: int = 1,
    ncols: int = 1,
    panel_aspect: float | None = None,
    **subplots_kwargs: object,
):
    """Create a correctly sized figure for the JNE column grid.

    Every panel is ``panel_aspect`` (default :data:`PANEL_ASPECT`) times taller than it is wide, whatever ``nrows`` and ``ncols``:
    the width is fixed by the journal column, so the height follows the panel width
    (``width / ncols * PANEL_ASPECT`` per row) plus :data:`DECORATION_HEIGHT` for titles and legends.

    Args:
        width: Width token (``'single'``/``'onehalf'``/``'double'``) or inches.
        nrows: Subplot rows.
        ncols: Subplot columns.
        panel_aspect: Panel height / width. Square image grids (heatmaps) pass 1.0, otherwise the default
            tall panel leaves a wide empty band between rows.
        **subplots_kwargs: Forwarded to :func:`matplotlib.pyplot.subplots`.

    Returns:
        ``(fig, axes)`` as returned by ``plt.subplots``.
    """
    apply_style()
    w = FIG_WIDTHS[width] if isinstance(width, str) else float(width)
    aspect = PANEL_ASPECT if panel_aspect is None else float(panel_aspect)
    height = w / max(ncols, 1) * aspect * nrows + DECORATION_HEIGHT
    return plt.subplots(nrows, ncols, figsize=(w, height), **subplots_kwargs)


def panel_letters(
    axes: Sequence,
    *,
    start: int = 0,
    x: float = -0.18,
    y: float = 1.04,
) -> None:
    """Annotate axes with bold panel letters ``(a)``, ``(b)``, ... in reading order.

    Args:
        axes: Axes in reading order.
        start: Index offset into the alphabet (0 -> ``(a)``).
        x: X position in axes coordinates.
        y: Y position in axes coordinates.
    """
    for i, ax in enumerate(list(axes)):
        ax.text(
            x,
            y,
            f"({chr(ord('a') + start + i)})",
            transform=ax.transAxes,
            fontsize=FONT_SIZES["panel_letter"],
            fontweight="bold",
            va="bottom",
            ha="left",
        )


def save_figure(
    fig,
    out_dir: str,
    name: str,
    *,
    formats: Sequence[str] = ("svg",),
    close: bool = True,
) -> list[str]:
    """Save ``fig`` as ``out_dir/name.<ext>`` for each requested format.

    SVG is the project default (user decision 2026-09-18); PDF/PNG are opt-in
    (PNG is written at :data:`SAVE_DPI`).

    Args:
        fig: Figure to save.
        out_dir: Destination directory (created if absent).
        name: Basename without extension.
        formats: Extensions to write, e.g. ``("svg", "pdf")``.
        close: Close the figure afterwards.

    Returns:
        The list of written paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    for ext in formats:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, format=ext, dpi=SAVE_DPI if ext == "png" else None)
        written.append(path)
    if close:
        plt.close(fig)
    return written
