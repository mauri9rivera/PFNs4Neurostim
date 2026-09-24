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
  (``knob``, ``level``, ``K2``, ``Demo 1``/``Demo 2``, ``breakdown point``).
"""
from __future__ import annotations

import colorsys
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "derive_family",
    "perceptual_distance",
    "knob_x_label",
    "KNOB_X_LABELS",
    "R2_AXIS_FLOOR",
    "MULTIPANEL_ASPECT",
    "STACKED_ASPECT",
    "SINGLE_PANEL_ASPECT",
    "SEQUENTIAL_CMAP",
    "DIVERGING_CMAP",
    "OUTLINE_COLOR",
    "FIG_WIDTHS",
    "FONT_SIZES",
    "ANCHOR_PFN",
    "ANCHOR_GP",
    "NEUTRAL_GREY",
    "ACQUISITION_ORDER",
    "ACQUISITION_LINESTYLES",
    "derive_shade",
    "acquisition_style",
    "ModelStyle",
    "MODEL_STYLES",
    "MODEL_ORDER",
    "AXIS_LABELS",
    "KNOB_LABELS",
    "KNOB_X_AXIS",
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
}

DEFAULT_WIDTH: str = "single"
DEFAULT_ASPECT: float = 0.72  # height / width for a single un-faceted panel
#: Height / width of ONE panel in a row of side-by-side panels (``figure`` divides by ``ncols``). Taller than
#: square so tick labels, two-line axis labels and a shared legend do not collide (2026-09-23).
MULTIPANEL_ASPECT: float = 1.45
#: Height / width of one panel in a vertical stack sharing an x-axis.
STACKED_ASPECT: float = 0.6
#: Height / width of a lone panel that carries a legend and a two-line axis label (taller than the 0.72 default).
SINGLE_PANEL_ASPECT: float = 0.85
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
@dataclass(frozen=True)
class ModelStyle:
    """Canonical plotting identity of one surrogate model.

    Attributes:
        label: Display string used in legends, tables and captions.
        color: Hex colour (Okabe-Ito family).
        linestyle: Matplotlib linestyle.
        marker: Matplotlib marker.
        family: ``'pfn'``, ``'gp'`` or ``'baseline'``.
        zorder: Draw order; headline models sit on top.
    """

    label: str
    color: str
    linestyle: str = "-"
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


#: Hue band (HLS hue, 0-1) each family's derived colours may occupy: cool blues/teals for PFNs, warm
#: reds/oranges/golds for GPs. The families never overlap, so the cool-PFN / warm-GP reading always survives.
FAMILY_HUE_BAND: dict[str, tuple[float, float]] = {"pfn": (0.47, 0.68), "gp": (0.0, 0.14)}
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
    "tabicl",
    "tabfm",
    "pfns4bo",
    "tabflex",
    "tabpfn_v1",
    "mitra",
    "random",
)

#: ``key -> (label, linestyle, marker, family, zorder)``; colours come from the anchors.
_MODEL_SPECS: dict[str, tuple[str, str, str, str, int]] = {
    "tabpfn_v2_5": ("TabPFN-2.5", "-", "o", "pfn", 5),
    "pfns4bo": ("PFNs4BO", "-", "^", "pfn", 3),
    "tabpfn_v1": ("TabPFN-1", "-", "v", "pfn", 3),
    "tabflex": ("TabFlex", "-", "<", "pfn", 3),
    "tabfm": ("TabFM", "-", ">", "pfn", 3),
    "mitra": ("Mitra", "-", "P", "pfn", 3),
    "tabicl": ("TabICL", "-", "d", "pfn", 3),
    "gp_mll": ("GP-MLL", "--", "s", "gp", 4),
    "gp_naive": ("GP-fixed", "-.", "D", "gp", 3),
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
    "iteration": "BO iteration",
    "achieved_snr_db": "Achieved SNR (dB)",
    "target_delta_snr_db": "SNR degradation vs channel floor (dB)",
    "achieved_contamination": r"Achieved contamination (fraction of trials)",
    "achieved_failure": "Failed electrodes (fraction of array)",
    "achieved_budget": r"BO iterations (incl. $n_\mathrm{init}$)",
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
    "decoy_capture": "Decoy-capture rate",
    "amplitude_ratio": r"Decoy amplitude ratio $a_2/a_1$",
    "regret_difference": "Regret difference vs GP-MLL (norm. GT range)",
    "morans_i": "Moran's I",
    "skewness": "Skewness",
    "cv": "Coefficient of variation",
    "mean_sd_corr": "Mean-SD correlation",
    "degradation_auc": "Degradation AUC (norm.)",
    "reliability": "Split-half reliability (Spearman-Brown)",
    # Hyp C mechanism figures (tasks #5, #6, #9).
    "context_t": "Context size $t$ (observations)",
    "rho_shape": r"Update-shape alignment with GP (Pearson $r$)",
    "surprise_c": r"Surprise $c$ (predictive SD at anchor)",
    "update_at_anchor": "Update at anchor (predictive SD units)",
    "update_normalized": "Update (normalized to max)",
    "ell_hat": "Implicit-kernel lengthscale (norm. coords)",
    "sym_err": "Implied-kernel asymmetry",
    "psd_neg_mass": "Implied-kernel negative-eigenvalue mass",
    "ell_cv_anchor": "Lengthscale CV across anchors",
    "placement": "Placement $p$ (0 = floor, 1 = ceiling)",
    "cka_minus_null": "Debiased CKA minus permutation null",
    "layer": "Transformer layer",
    "layer_corr": r"Corr. of $\|\Delta Z_\ell\|$ with GP update",
    "layer_probe_r2": r"Ridge-probe $R^2$ ($\Delta Z_\ell \to \Delta_\mathrm{PFN}$)",
}

#: Canonical stress-knob names (roadmap jargon) and their axis strings.
KNOB_LABELS: dict[str, str] = {
    "k1_decoy": r"K1 decoy peak (amplitude ratio $a_2/a_1$)",
    "k2_channel": "K2 SNR (channel-relative noise)",
    "k2_global": "K2 SNR (global noise)",
    "k5_outliers": "K5 outlier contamination",
    "k6_budget": "K6 sparsity (BO iterations)",
    "k6_failure": "K6 sparsity (electrode failure fraction)",
    "k7_shuffle": r"K7 spatial shuffle fraction $f$",
    "nominal": "Nominal anchor",
}

#: The knob level itself is never the primary x-axis for K2 - achieved SNR is.
KNOB_X_AXIS: dict[str, str] = {
    "k1_decoy": "level",
    "k2_channel": "achieved_snr_db",
    "k2_global": "achieved_snr_db",
    "k5_outliers": "level",
    "k6_budget": "budget",
    "k6_failure": "level",
    "k7_shuffle": "level",
}

#: X-axis wording per knob, where the knob's vocabulary beats the column's (two lines, unit in parentheses,
#: so it fits a panel of a three-panel row). K5 and K6 read as levels of the perturbation, not as SNR.
KNOB_X_LABELS: dict[str, str] = {
    "k1_decoy": "Decoy amplitude ratio\n($a_2/a_1$)",
    "k5_outliers": "Contamination $\\varepsilon$\n(fraction of trial slots)",
    "k6_failure": "Failed electrodes $\\varepsilon$\n(fraction of array)",
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


DEMO_LABELS: dict[str, str] = {
    "demo1": "Demo 1 (synthetic)",
    "demo2": "Demo 2 (in vivo)",
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
    aspect: float = DEFAULT_ASPECT,
    height: float | None = None,
    **subplots_kwargs: object,
):
    """Create a correctly sized figure for the JNE column grid.

    Args:
        width: Width token (``'single'``/``'onehalf'``/``'double'``) or inches.
        nrows: Subplot rows.
        ncols: Subplot columns.
        aspect: Height/width ratio of a *single panel*; total height scales
            with ``nrows`` and shrinks with ``ncols``.
        height: Explicit total height in inches, overriding ``aspect``.
        **subplots_kwargs: Forwarded to :func:`matplotlib.pyplot.subplots`.

    Returns:
        ``(fig, axes)`` as returned by ``plt.subplots``.
    """
    apply_style()
    w = FIG_WIDTHS[width] if isinstance(width, str) else float(width)
    if height is None:
        height = w * aspect * nrows / max(ncols, 1)
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
