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

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "FIG_WIDTHS",
    "FONT_SIZES",
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


# Okabe-Ito reference: blue #0072B2, sky #56B4E9, green #009E73, yellow #F0E442,
# orange #E69F00, vermillion #D55E00, purple #CC79A7, grey #999999.
MODEL_STYLES: dict[str, ModelStyle] = {
    # --- PFN family (cool) ---
    "tabpfn_v2_5": ModelStyle("TabPFN-2.5", "#0072B2", "-", "o", "pfn", zorder=5),
    "pfns4bo": ModelStyle("PFNs4BO", "#56B4E9", "-", "^", "pfn", zorder=3),
    "tabpfn_v1": ModelStyle("TabPFN-1", "#CC79A7", "-", "v", "pfn", zorder=3),
    "tabflex": ModelStyle("TabFlex", "#7E4E9B", "-", "<", "pfn", zorder=3),
    "tabfm": ModelStyle("TabFM", "#00688B", "-", ">", "pfn", zorder=3),
    "mitra": ModelStyle("Mitra", "#4C9BE8", "-", "P", "pfn", zorder=3),
    "tabicl": ModelStyle("TabICL", "#3A7CA5", "-", "d", "pfn", zorder=3),
    # --- GP family (warm) ---
    "gp_mll": ModelStyle("GP-MLL", "#D55E00", "--", "s", "gp", zorder=4),
    "gp_naive": ModelStyle("GP-fixed", "#E69F00", "-.", "D", "gp", zorder=3),
    "gp_oracle": ModelStyle("GP-oracle", "#B8860B", ":", "*", "gp", zorder=3),
    "gp_deep_kernel": ModelStyle("GP-deep", "#A64B00", "--", "X", "gp", zorder=2),
    # --- Non-learning baselines (grey) ---
    "random": ModelStyle("Random", "#999999", ":", "x", "baseline", zorder=1),
}

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

#: Legend / table ordering: headline pair first, then the GP ladder, then baselines.
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
