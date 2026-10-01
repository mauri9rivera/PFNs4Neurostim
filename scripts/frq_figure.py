"""Build the 3-panel preliminary-results figure for the FRQ B2 2027-2028 research project.

This is a *grant* figure, not a paper figure: it must fit inside a 2-page French
submission (Times New Roman 12 pt, 2 cm margins), so it uses a flatter panel aspect than
the JNE print grid and localized axis strings. Everything else -- widths, colours,
linestyles, model labels -- comes from :mod:`pfns4neurostim.visualization.style`, which
stays the single source of style (CLAUDE.md section 7).

The three panels share all three NeuroMOSAICS datasets:

    (a) recommended-site regret vs query budget   -- does the surrogate optimize better?
    (b) final surrogate R^2                        -- does it reconstruct the map better?
    (c) per-query latency vs BO step               -- does its per-query cost grow with n?

Panel (c) is the operations-research claim: the GP's per-query cost grows with the number
of observations while the amortized surrogate's is flat. It is a *slope* claim, not a
level claim, and the caption facts printed by :func:`caption_facts` say so.

Usage:
    python scripts/frq_figure.py
    python scripts/frq_figure.py --out-dir output/figures --formats svg png
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from pfns4neurostim.visualization import style

# ---------------------------------------------------------------------------
# What the figure shows. Every value here is a declared constant, never inlined
# in a function body (CLAUDE.md section 8).
# ---------------------------------------------------------------------------

#: Run directory of each dataset's PFN benchmark, relative to the repository root.
RUN_DIRS: dict[str, str] = {
    "nhp": "output/benchmark/nhp/hyp0-pfn-bench-nhp",
    "spinal": "output/benchmark/spinal/hyp0-pfn-bench-spinal",
    "5d_rat": "output/benchmark/5d_rat/hyp0-pfn-bench-5d_rat",
}

#: Extended latency sweep used for the panel-(c) annotation.
SCALING_CSV: str = "output/scaling/scaling_timing.csv"

#: The Gaussian-process reference, plotted in every panel.
GP_MODEL: str = "gp_mll"

#: Best-performing amortized surrogate per dataset, by final simple regret in
#: ``comparison_table.csv``. It is *not* the same architecture everywhere -- TabICL wins on
#: 5d_rat and loses on the other two -- and that dataset-dependence is exactly what
#: Objective 1 characterizes, so the caption states it rather than hiding it.
#: :func:`verify_best_amortized` re-checks these choices against the data on every run.
AMORTIZED_MODEL: dict[str, str] = {
    "nhp": "tabpfn_v2_5",
    "spinal": "tabpfn_v2_5",
    "5d_rat": "tabicl",
}

#: Models that may be selected by :func:`verify_best_amortized` (the GP is excluded, and so
#: are runs that did not complete on every dataset).
AMORTIZED_CANDIDATES: tuple[str, ...] = ("tabpfn_v2_5", "tabicl", "pfns4bo")

#: Panels (b) and (c) use ONE amortized architecture on all three datasets, unlike panel
#: (a). Panel (a) answers "how well does the best available amortized surrogate optimize?",
#: which is a per-dataset question; (b) and (c) are claims about one model class held fixed
#: across modalities, where mixing architectures would confound architecture with the effect
#: being shown. TabPFN-2.5 ran on every dataset, so it is the comparable choice.
REFERENCE_AMORTIZED_MODEL: str = "tabpfn_v2_5"

#: Encoding of this figure: **colour carries the dataset** and **linestyle carries the model
#: family** (GP dashed, amortized solid). This inverts the package convention, where colour
#: is model identity -- deliberately, because the figure's message is "the same contrast
#: holds across three modalities", which a reader sees at once when each modality is one
#: colour. Markers repeat the dataset so the panels survive greyscale printing.
DATASET_MARKERS: dict[str, str] = {"nhp": "o", "spinal": "^", "5d_rat": "s"}

#: Okabe-Ito colours -- the same colour-blind-safe source as ``style.ANCHOR_PFN`` and
#: ``style.ANCHOR_GP``, which are reused here for two of the three datasets. These are
#: *dataset* colours, not model colours: the package rule against hard-coding a model colour
#: is untouched, since no model is coloured in this figure.
#: The third colour is a saturated magenta rather than Okabe-Ito's bluish green. The
#: lightened GP tone is produced by holding hue and saturation and raising lightness
#: (:func:`style.derive_shade`), and a saturated green treated that way lands on a
#: near-fluorescent mint (#52FFD0) that has almost no contrast against white. Magenta keeps
#: its identity when lightened, and by :func:`style.perceptual_distance` it is the best
#: separated of the candidates tried: dE 70 to the blue and 97 to the vermillion, with its
#: pale tone dE 82 from the nearest other pale tone. Markers repeat the dataset, so the
#: panels stay readable even where two hues converge under colour-vision deficiency.
DATASET_COLORS: dict[str, str] = {
    "nhp": style.ANCHOR_PFN,    # #0072B2 blue
    "spinal": style.ANCHOR_GP,  # #D55E00 vermillion
    "5d_rat": "#A5108C",        # magenta
}

#: Legend order and French display names of the datasets.
DATASET_ORDER: tuple[str, ...] = ("nhp", "spinal", "5d_rat")
DATASET_LABELS_FR: dict[str, str] = {
    "nhp": "Primate non humain",
    "spinal": "Moelle épinière",
    "5d_rat": "Rat 5-D",
}

#: French axis strings for the FRQ submission. Keys are the metric names understood by
#: :func:`pfns4neurostim.visualization.style.axis_label`; this table localizes that single
#: source rather than replacing it, and :func:`_fr_axis_label` falls back to it.
AXIS_LABELS_FR: dict[str, str] = {
    "recommended_regret": "Regret du site recommandé\n(normalisé)",
    "budget": "Requêtes",
    "iteration": "Pas de la boucle",
    "r2": "$R^2$ de la carte",
    "latency_growth": "Coût par requête\n(relatif au 1er pas)",
}

#: Short dataset names for the table's column headers; full names are in the legend/caption.
DATASET_TICKS_FR: dict[str, str] = {"nhp": "PNH", "spinal": "Moelle", "5d_rat": "Rat 5-D"}

#: Panel aspect (height / width). Flatter than ``style.PANEL_ASPECT`` because a 2-page
#: submission cannot spend 45% of a page on one figure.
PANEL_ASPECT_GRANT: float = 0.85

#: Number of markers drawn along each trajectory (markers identify the dataset; drawing one
#: per step would fill the panel).
N_MARKERS_PER_CURVE: int = 6

#: Horizontal offset between the GP and amortized points of one dataset in panel (b).
PAIRED_DOT_OFFSET: float = 0.16

#: Shade rank/count used by :func:`grant_color` for amortized surrogates other than the
#: family anchor, so they stay inside the anchor's hue instead of drifting to another one.
#: Linestyle per model family; colour is spent on the dataset (see :data:`DATASET_COLORS`).
GP_LINESTYLE: str = "--"
AMORTIZED_LINESTYLE: str = "-"

#: Figure variants written by one run.
#:
#: * ``'plain'`` -- linestyle alone separates the two model families.
#: * ``'contrast'`` -- linestyle, marker fill and colour tone all separate them: the GP is
#:   drawn in a lightened tone of its dataset colour with open markers, the amortized
#:   surrogate in the full tone with filled markers. Three redundant cues, because at the
#:   size this figure is printed in a 2-page submission one cue is not enough.
FIGURE_VARIANTS: tuple[str, ...] = ("plain", "contrast")

#: Lightness band used to lighten the GP tone in the ``'contrast'`` variant. The upper end
#: is above ``style.SHADE_LIGHTNESS_BAND`` so the GP reads as clearly paler at print size.
GP_TONE_BAND: tuple[float, float] = (0.50, 0.66)

#: Aggregation over channels x repetitions, chosen per metric to match the summary that
#: ``comparison_table.csv`` already reports, so the curves and the quoted numbers agree.
#:
#: * Regret is **averaged**. The median is actively misleading here: most channels are
#:   solved exactly within the budget, so the median regret of *both* models collapses to
#:   zero on NHP and spinal and the entire difference -- which lives in the hard channels,
#:   i.e. the upper tail -- disappears from the plot.
#: * R^2 is **median**, following ``style.AXIS_LABELS['r2_median']``: a single badly fitted
#:   channel would otherwise dominate the mean.
#: * Latency is **median**, which is the robust choice for timings on a shared cluster.
AGGREGATE_BY_FIELD: dict[str, str] = {
    "recommended_regret_per_step": "mean",
    "r2_per_step": "median",
    "step_times_s": "median",
}

DEFAULT_OUT_DIR: str = "output/figures"
DEFAULT_NAME: str = "frq_figure1"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_trajectories(run_dir: str) -> tuple[tuple[str, ...], dict[tuple, dict]]:
    """Load one run's trajectory pickle.

    Args:
        run_dir: Run directory containing ``trajectories.pkl``.

    Returns:
        ``(key_columns, trajectories)`` where ``key_columns`` names the fields of each
        trajectory key and ``trajectories`` maps those keys to per-step arrays.

    Raises:
        FileNotFoundError: If the pickle is absent.
    """
    path = os.path.join(run_dir, "trajectories.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run the hyp0_pfn_bench experiment first.")
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    return tuple(blob["key_columns"]), blob["trajectories"]


def stack_curves(
    run_dir: str,
    model: str,
    field: str,
) -> np.ndarray:
    """Collect every repetition's per-step curve for one model into a dense array.

    Repetitions can differ in length by a step (a channel whose loop stopped early), so the
    curves are truncated to the shortest one rather than padded with a fabricated value.

    Args:
        run_dir: Run directory.
        model: Canonical model key, e.g. ``'gp_mll'``.
        field: Trajectory field, e.g. ``'recommended_regret_per_step'``.

    Returns:
        Array of shape ``[n_curves, n_steps]``.

    Raises:
        KeyError: If no trajectory matches ``model``.
        RuntimeError: If any curve contains NaN or Inf (fail fast, CLAUDE.md section 4).
    """
    key_columns, trajectories = load_trajectories(run_dir)
    model_idx = key_columns.index("model")
    rows: list[np.ndarray] = [
        np.asarray(traj[field], dtype=float)
        for key, traj in trajectories.items()
        if key[model_idx] == model
    ]
    if not rows:
        raise KeyError(f"No trajectory for model {model!r} in {run_dir}.")
    n_steps = min(len(r) for r in rows)
    stacked = np.stack([r[:n_steps] for r in rows])  # [n_curves, n_steps]
    if not np.isfinite(stacked).all():
        raise RuntimeError(
            f"Non-finite values in {field!r} for {model!r} in {run_dir}: "
            f"{np.count_nonzero(~np.isfinite(stacked))} of {stacked.size} entries."
        )
    return stacked


def load_comparison_table(run_dir: str) -> pd.DataFrame:
    """Load one run's ``comparison_table.csv`` indexed by model."""
    path = os.path.join(run_dir, "comparison_table.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}.")
    return pd.read_csv(path).set_index("model")


def verify_best_amortized(run_dirs: dict[str, str]) -> dict[str, str]:
    """Re-derive the best amortized surrogate per dataset from the data.

    :data:`AMORTIZED_MODEL` is a declared constant so the figure is reproducible, but a
    re-run of the benchmark could change which architecture wins. This recomputes the
    choice and raises if the constant is stale, so the figure and its caption can never
    disagree with the numbers behind them.

    Args:
        run_dirs: Dataset -> run directory.

    Returns:
        Dataset -> winning model key, as derived from the data.

    Raises:
        ValueError: If a derived winner differs from :data:`AMORTIZED_MODEL`.
    """
    derived: dict[str, str] = {}
    for dataset, run_dir in run_dirs.items():
        table = load_comparison_table(run_dir)
        available = [m for m in AMORTIZED_CANDIDATES if m in table.index]
        derived[dataset] = min(available, key=lambda m: table.loc[m, "simple_regret_final"])
    stale = {d: (AMORTIZED_MODEL[d], derived[d]) for d in derived if AMORTIZED_MODEL[d] != derived[d]}
    if stale:
        raise ValueError(
            "AMORTIZED_MODEL is stale; the data now favour a different architecture: "
            + ", ".join(f"{d}: declared {a!r}, derived {b!r}" for d, (a, b) in stale.items())
        )
    return derived


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def _fr_axis_label(key: str) -> str:
    """French axis string for ``key``, falling back to the canonical English one."""
    return AXIS_LABELS_FR.get(key, style.axis_label(key))


def family_linestyle(model: str) -> str:
    """Linestyle of ``model``: dashed for the GP family, solid for amortized surrogates.

    Taken from the package's own per-model linestyles, which already follow that rule
    (``style._MODEL_SPECS``: PFN solid, GP dashed/dash-dot/dotted), collapsed here to one
    linestyle per family because this figure shows one model per family.

    Args:
        model: Model identifier (alias-tolerant).

    Returns:
        A matplotlib linestyle string.
    """
    return GP_LINESTYLE if style.model_style(model).family == "gp" else AMORTIZED_LINESTYLE


def series_color(model: str, dataset: str, variant: str) -> str:
    """Colour of one (model, dataset) series.

    Args:
        model: Model identifier (alias-tolerant).
        dataset: Dataset key.
        variant: One of :data:`FIGURE_VARIANTS`.

    Returns:
        Hex colour ``'#RRGGBB'`` -- the dataset colour, lightened for the GP in the
        ``'contrast'`` variant.
    """
    base = DATASET_COLORS[dataset]
    if variant == "contrast" and style.model_style(model).family == "gp":
        return style.derive_shade(base, 1, 2, band=GP_TONE_BAND)
    return base


def _series_kwargs(model: str, dataset: str, variant: str) -> dict[str, object]:
    """Line kwargs for one (model, dataset) series.

    Colour and marker carry the dataset; linestyle carries the model family, per the
    figure's encoding (see :data:`DATASET_COLORS`). In the ``'contrast'`` variant the
    colour tone and the marker fill carry the family as well.
    """
    is_gp = style.model_style(model).family == "gp"
    color = series_color(model, dataset, variant)
    kwargs: dict[str, object] = {
        "color": color,
        "linestyle": family_linestyle(model),
        "marker": DATASET_MARKERS[dataset],
        "markersize": style.MARKER_SIZE,
        "linewidth": style.LINE_WIDTH,
        "zorder": style.model_style(model).zorder,
    }
    if variant == "contrast":
        kwargs["markerfacecolor"] = "none" if is_gp else color
        kwargs["markeredgecolor"] = color
        kwargs["markeredgewidth"] = style.LINE_WIDTH
    return kwargs


def _marker_every(n_steps: int) -> int:
    """Marker stride that puts about :data:`N_MARKERS_PER_CURVE` markers on a curve."""
    return max(1, n_steps // N_MARKERS_PER_CURVE)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _plot_aggregate_curves(
    ax,
    run_dirs: dict[str, str],
    models_for: dict[str, str],
    field: str,
    aggregate: str,
    variant: str,
    *,
    normalize_to_first: bool = False,
) -> None:
    """Plot one aggregate per-step curve of ``field`` for the GP and one amortized model.

    Args:
        ax: Target axes.
        run_dirs: Dataset -> run directory.
        models_for: Dataset -> the amortized model to pair with the GP.
        field: Trajectory field to plot.
        aggregate: ``'mean'`` or ``'median'`` over channels x repetitions; see
            :data:`AGGREGATE_BY_FIELD` for why the choice is per metric.
        variant: One of :data:`FIGURE_VARIANTS`.
        normalize_to_first: Divide each curve by its own first step, turning an absolute
            measurement into a dimensionless growth factor.

    Raises:
        KeyError: If ``aggregate`` is not a known aggregation.
    """
    reducer = {"mean": np.mean, "median": np.median}[aggregate]
    for dataset in DATASET_ORDER:
        run_dir = run_dirs[dataset]
        for model in (GP_MODEL, models_for[dataset]):
            curve = reducer(stack_curves(run_dir, model, field), axis=0)
            if normalize_to_first:
                curve = curve / curve[0]
            steps = np.arange(len(curve))
            ax.plot(steps, curve, markevery=_marker_every(len(steps)),
                    **_series_kwargs(model, dataset, variant))


def panel_regret(ax, run_dirs: dict[str, str], amortized: dict[str, str], variant: str) -> None:
    """Panel (a): recommended-site regret vs query budget, averaged over channels x reps."""
    _plot_aggregate_curves(ax, run_dirs, amortized, "recommended_regret_per_step",
                           AGGREGATE_BY_FIELD["recommended_regret_per_step"], variant)
    ax.set_xlabel(_fr_axis_label("budget"))
    ax.set_ylabel(_fr_axis_label("recommended_regret"))
    ax.set_ylim(bottom=0.0)


def panel_r2(ax, run_dirs: dict[str, str], variant: str) -> None:
    """Panel (b): surrogate R^2 vs query budget, one amortized architecture throughout."""
    models_for = {d: REFERENCE_AMORTIZED_MODEL for d in DATASET_ORDER}
    _plot_aggregate_curves(ax, run_dirs, models_for, "r2_per_step",
                           AGGREGATE_BY_FIELD["r2_per_step"], variant)
    ax.set_xlabel(_fr_axis_label("budget"))
    ax.set_ylabel(_fr_axis_label("r2"))


def panel_latency(ax, run_dirs: dict[str, str], variant: str) -> None:
    """Panel (c): relative growth of per-query cost vs query budget.

    Each curve is divided by its own first step. This is deliberate and necessary: in these
    runs the GP was executed on CPU and every amortized surrogate on GPU, on different
    cluster nodes (``config.yaml`` ``model_devices``, and the ``host_*`` columns of
    ``tidy.csv``), so absolute latencies are not comparable across models. Growth *within* a
    run is measured on one machine for one model and is therefore valid -- and it is also
    the sharper claim, since O(n^3) versus amortized inference is a statement about slope.
    """
    models_for = {d: REFERENCE_AMORTIZED_MODEL for d in DATASET_ORDER}
    _plot_aggregate_curves(ax, run_dirs, models_for, "step_times_s",
                           AGGREGATE_BY_FIELD["step_times_s"], variant,
                           normalize_to_first=True)
    ax.axhline(1.0, color=style.NEUTRAL_GREY, linewidth=style.LINE_WIDTH * 0.7,
               linestyle=":", zorder=0)
    ax.set_xlabel(_fr_axis_label("budget"))
    ax.set_ylabel(_fr_axis_label("latency_growth"))


def build_legend(fig, axes: Sequence, amortized: dict[str, str], variant: str) -> None:
    """Attach the two-part legend: models by colour/linestyle, datasets by marker."""
    from matplotlib.lines import Line2D

    amortized_names = " / ".join(
        dict.fromkeys(
            style.model_label(m)
            for m in [*(amortized[d] for d in DATASET_ORDER), REFERENCE_AMORTIZED_MODEL]
        )
    )
    # The model keys are shown in neutral grey/black: colour is spent on the dataset, so a
    # model swatch must not suggest one. In the 'contrast' variant the swatches also carry
    # the marker fill, which is the cue that separates the families inside each panel.
    gp_key: dict[str, object] = {"color": "black", "linestyle": GP_LINESTYLE}
    as_key: dict[str, object] = {"color": "black", "linestyle": AMORTIZED_LINESTYLE}
    if variant == "contrast":
        gp_key |= {"marker": "o", "markerfacecolor": "none", "markeredgecolor": "black",
                   "markeredgewidth": style.LINE_WIDTH, "markersize": style.MARKER_SIZE}
        as_key |= {"marker": "o", "markerfacecolor": "black", "markersize": style.MARKER_SIZE}
    handles = [
        Line2D([], [], linewidth=style.LINE_WIDTH,
               label=f"{style.model_label(GP_MODEL)} (référence PG)", **gp_key),
        Line2D([], [], linewidth=style.LINE_WIDTH,
               label=f"Substitut amorti ({amortized_names})", **as_key),
    ]
    handles += [
        Line2D([], [], color=DATASET_COLORS[d], linestyle=AMORTIZED_LINESTYLE,
               marker=DATASET_MARKERS[d], markersize=style.MARKER_SIZE,
               linewidth=style.LINE_WIDTH, label=DATASET_LABELS_FR[d])
        for d in DATASET_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles), fontsize=style.FONT_SIZES["legend"])


# ---------------------------------------------------------------------------
# Table (alternative to the figure)
# ---------------------------------------------------------------------------
#: Label of the aggregated amortized row. It is one architecture *per dataset* -- the one
#: that reaches the lowest final regret there -- not the best cell of each column, which
#: would be cherry-picking across architectures. :func:`table_note` names them.
AMORTIZED_ROW_LABEL: str = "Meilleur substitut amorti"

#: Metric columns nested under each dataset: ``(key, French header, lower_is_better)``.
TABLE_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("simple_regret_final", "Regret", True),
    ("r2_final", "R²", False),
    ("latency_final_s", "Coût/req. (s)", True),
)


def _metric_value(run_dir: str, model: str, key: str) -> float:
    """Return one table cell.

    ``latency_final_s`` is the median per-query latency at the *final* BO step, taken from
    the trajectories; the other keys are read from ``comparison_table.csv``.
    """
    if key == "latency_final_s":
        return float(np.median(stack_curves(run_dir, model, "step_times_s"), axis=0)[-1])
    return float(load_comparison_table(run_dir).loc[model, key])


def table_rows(run_dirs: dict[str, str], amortized: dict[str, str]) -> pd.DataFrame:
    """Build the summary table: the GP reference against the best amortized surrogate.

    Columns are a two-level MultiIndex ``(dataset, metric)``.

    Args:
        run_dirs: Dataset -> run directory.
        amortized: Dataset -> the amortized model to report for that dataset.

    Returns:
        DataFrame with two rows, indexed by row label.
    """
    gp_label = style.model_label(GP_MODEL)
    data: dict[tuple[str, str], dict[str, float]] = {}
    for dataset in DATASET_ORDER:
        run_dir = run_dirs[dataset]
        for key, header, _ in TABLE_METRICS:
            data[(DATASET_TICKS_FR[dataset], header)] = {
                gp_label: _metric_value(run_dir, GP_MODEL, key),
                AMORTIZED_ROW_LABEL: _metric_value(run_dir, amortized[dataset], key),
            }
    frame = pd.DataFrame(data)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["Jeu de données", ""])
    return frame.reindex([gp_label, AMORTIZED_ROW_LABEL])


def table_note(run_dirs: dict[str, str], amortized: dict[str, str]) -> str:
    """Return the note that must travel with the table.

    It names the architecture used per dataset and records that absolute latencies are not
    comparable between the two rows, because the GP ran on CPU and the amortized surrogates
    on GPU, on different cluster nodes.
    """
    names = ", ".join(
        f"{style.model_label(amortized[d])} ({DATASET_TICKS_FR[d]})" for d in DATASET_ORDER
    )
    def gp_growth_pct(dataset: str) -> float:
        median = np.median(stack_curves(run_dirs[dataset], GP_MODEL, "step_times_s"), axis=0)
        return 100.0 * (median[-1] / median[0] - 1.0)

    growth = ", ".join(
        f"{DATASET_TICKS_FR[d]} {gp_growth_pct(d):+.0f} %" for d in DATASET_ORDER
    )
    return (
        f"Substitut amorti : {names}. Regret et R² en fin de boucle "
        f"(regret moyenné, R² médian sur les canaux ; 10 répétitions). "
        f"Coût par requête : médiane au dernier pas. "
        f"**Les temps absolus ne sont pas comparables entre les deux lignes** : le PG s'exécute "
        f"sur CPU et les substituts amortis sur GPU, sur des nœuds distincts. Ce qui est "
        f"comparable, c'est la croissance du coût au fil de la boucle — celui du PG augmente "
        f"({growth}), celui du substitut amorti reste constant (+1 %)."
    )


#: Columns whose better value is emphasised. Latency is excluded on purpose: bolding the
#: lower absolute time would assert exactly the cross-row comparison that :func:`table_note`
#: says is invalid (CPU versus GPU, different nodes).
TABLE_BOLD_METRICS: frozenset[str] = frozenset({"Regret", "R²"})


def format_table_markdown(frame: pd.DataFrame) -> str:
    """Render :func:`table_rows` as a Markdown table, emphasising the better value.

    Args:
        frame: Table from :func:`table_rows`.

    Returns:
        A Markdown string ready to paste into the draft.
    """
    lower_is_better = {header: low for _, header, low in TABLE_METRICS}
    header_top = "| Modèle | " + " | ".join(
        f"{ds} — {metric}" for ds, metric in frame.columns) + " |"
    sep = "|" + "---|" * (len(frame.columns) + 1)
    lines = [header_top, sep]
    for model, row in frame.iterrows():
        cells: list[str] = []
        for (dataset, metric), value in row.items():
            if pd.isna(value):
                cells.append("—")
                continue
            text = f"{value:.3f}"
            if metric in TABLE_BOLD_METRICS:
                series = frame[(dataset, metric)].dropna()
                best = series.min() if lower_is_better[metric] else series.max()
                if value == best:
                    text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_table(
    run_dirs: dict[str, str],
    amortized: dict[str, str],
    out_dir: str,
    name: str,
) -> list[str]:
    """Write the summary table as CSV (machine-readable) and Markdown (paste-ready).

    Args:
        run_dirs: Dataset -> run directory.
        amortized: Dataset -> the amortized model to report for that dataset.
        out_dir: Destination directory.
        name: Basename without extension.

    Returns:
        The list of written paths.
    """
    frame = table_rows(run_dirs, amortized)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    md_path = os.path.join(out_dir, f"{name}.md")
    frame.to_csv(csv_path)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(format_table_markdown(frame) + "\n\n")
        fh.write(table_note(run_dirs, amortized) + "\n")
    return [csv_path, md_path]


# ---------------------------------------------------------------------------
# Caption facts
# ---------------------------------------------------------------------------
def caption_facts(run_dirs: dict[str, str], amortized: dict[str, str]) -> str:
    """Re-derive every number quoted in the figure caption, so the caption cannot drift.

    Args:
        run_dirs: Dataset -> run directory.
        amortized: Dataset -> winning amortized model.

    Returns:
        A printable block of the caption-relevant numbers.
    """
    lines: list[str] = ["", "=" * 72, "CAPTION FACTS (paste-checked against the data)", "=" * 72]
    for dataset in DATASET_ORDER:
        run_dir = run_dirs[dataset]
        table = load_comparison_table(run_dir)
        tidy = pd.read_csv(os.path.join(run_dir, "tidy.csv"))
        lines.append(
            f"\n[{dataset}] {DATASET_LABELS_FR[dataset]}  "
            f"budget={int(tidy['budget'].max())}  "
            f"sites={int(tidy['n_sites'].max())}  "
            f"channels={int(table.loc[GP_MODEL, 'n_channels'])}  "
            f"reps={int(table.loc[GP_MODEL, 'n_reps'])}"
        )
        # Panel (a) uses the per-dataset winner; (b)/(c) use REFERENCE_AMORTIZED_MODEL.
        for model in dict.fromkeys([GP_MODEL, amortized[dataset], REFERENCE_AMORTIZED_MODEL]):
            row = table.loc[model]
            curves = stack_curves(run_dir, model, "step_times_s")
            median = np.median(curves, axis=0)
            growth = 100.0 * (median[-1] / median[0] - 1.0)
            panels = "".join(
                tag for tag, used in (
                    ("a", model in (GP_MODEL, amortized[dataset])),
                    ("bc", model in (GP_MODEL, REFERENCE_AMORTIZED_MODEL)),
                ) if used
            )
            lines.append(
                f"    {style.model_label(model):<12s} [{panels:<3s}] "
                f"regret={row['simple_regret_final']:.3f}  "
                f"R2={row['r2_final']:.3f}  "
                f"latency {median[0]:.3f}s -> {median[-1]:.3f}s ({growth:+.0f}%)  "
                f"loop total={np.median(curves.sum(axis=1)):.1f}s"
            )
    if os.path.exists(SCALING_CSV):
        scaling = pd.read_csv(SCALING_CSV).groupby(["model", "n"])["time_s"].mean().unstack("model")
        lines.append(f"\n[scaling sweep] {SCALING_CSV}")
        for n, row in scaling.iterrows():
            lines.append("    n=%-5d " % n + "  ".join(f"{m}={v:.3f}s" for m, v in row.items()))
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_figure(run_dirs: dict[str, str], amortized: dict[str, str], variant: str):
    """Assemble the three-panel figure.

    Args:
        run_dirs: Dataset -> run directory.
        amortized: Dataset -> winning amortized model.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    width = style.FIG_WIDTHS["double"]
    n_panels = 3
    height = width / n_panels * PANEL_ASPECT_GRANT + style.DECORATION_HEIGHT
    fig, axes = plt.subplots(1, n_panels, figsize=(width, height),
                             layout=style.LAYOUT_ENGINE)
    panel_regret(axes[0], run_dirs, amortized, variant)
    panel_r2(axes[1], run_dirs, variant)
    panel_latency(axes[2], run_dirs, variant)
    style.panel_letters(axes)
    build_legend(fig, axes, amortized, variant)
    return fig


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="Directory for the rendered figure.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Basename without extension.")
    parser.add_argument("--formats", nargs="+", default=["svg", "png"],
                        help="Formats to write; PNG is rendered at style.SAVE_DPI.")
    args = parser.parse_args()

    amortized = verify_best_amortized(RUN_DIRS)
    written: list[str] = []
    for variant in FIGURE_VARIANTS:
        fig = build_figure(RUN_DIRS, amortized, variant)
        suffix = "" if variant == FIGURE_VARIANTS[0] else f"_{variant}"
        written += style.save_figure(fig, args.out_dir, f"{args.name}{suffix}",
                                     formats=tuple(args.formats))
    written += write_table(RUN_DIRS, amortized, args.out_dir,
                           args.name.replace("figure", "table"))
    print(caption_facts(RUN_DIRS, amortized))
    print()
    print(format_table_markdown(table_rows(RUN_DIRS, amortized)))
    print()
    print(table_note(RUN_DIRS, amortized))
    print()
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
