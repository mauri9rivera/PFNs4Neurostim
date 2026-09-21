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

__all__ = [
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


#: Legend / table ordering: headline pair first, then the GP ladder, then baselines.
#: It also fixes each model's shade rank within its family.
MODEL_ORDER: tuple[str, ...] = (
    "tabpfn_v2_5",
    "gp_mll",
    "gp_naive",
    "gp_oracle",
    "gp_deep_kernel",
    "pfns4bo",
    "tabpfn_v1",
    "tabflex",
    "tabfm",
    "mitra",
    "tabicl",
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
    styles: dict[str, ModelStyle] = {}
    for key, (label, ls, marker, fam, z) in _MODEL_SPECS.items():
        if fam == "baseline":
            color = NEUTRAL_GREY
        elif key in _ANCHOR_KEYS:
            color = _FAMILY_ANCHOR[fam]
        else:
            members = derived[fam]
            color = derive_shade(_FAMILY_ANCHOR[fam], members.index(key), len(members))
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
    "achieved_contamination": r"Achieved contamination (fraction of trials)",
    "achieved_dropout": "Achieved electrode dropout (fraction)",
    "achieved_budget": r"BO iterations (incl. $n_\mathrm{init}$)",
    "n_queryable": "Queryable electrodes",
    "n_donor_trials": "Distinct artefact donors",
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
    "degradation_auc": "Degradation AUC (norm.)",
    "reliability": "Split-half reliability (Spearman-Brown)",
}

#: Canonical stress-knob names (roadmap jargon) and their axis strings.
KNOB_LABELS: dict[str, str] = {
    "k1_decoy": r"K1 decoy peak (amplitude ratio $a_2/a_1$)",
    "k2_snr": "K2 SNR",
    "k5_outliers": r"K5 outlier contamination (fraction $\epsilon$)",
    "k6_budget": "K6 sparsity (BO iterations)",
    "k6_dropout": "K6 sparsity (electrode dropout fraction)",
    "k7_shuffle": r"K7 spatial shuffle fraction $f$",
    "nominal": "Nominal anchor",
}

#: The knob level itself is never the primary x-axis for K2 - achieved SNR is.
KNOB_X_AXIS: dict[str, str] = {
    "k1_decoy": "level",
    "k2_snr": "achieved_snr_db",
    "k5_outliers": "level",
    "k6_budget": "budget",
    "k6_dropout": "level",
    "k7_shuffle": "level",
}

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
