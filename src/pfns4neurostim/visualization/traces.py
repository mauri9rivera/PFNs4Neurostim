"""Curves along the BO run, read from a run's ``trajectories.pkl`` (2026-09-21).

Every BO repetition stores, per step, the recommended-site simple regret, the exploration score
(true response at the recommended site over the true maximum, raw units), the surrogate's R^2 over
the pool and the step latency. This module turns those into figures without re-running anything.

Averaging: repetitions are reduced within a channel first, then channels are summarized, so a subject with
many EMGs cannot dominate. Bounded traces (regret, exploration) give the mean with a 95% CI over channels;
heavy-tailed traces (R^2, latency) give the median with the interquartile range (:data:`HEAVY_TAILED_FIELDS`).

x-axis convention: the recommendation ``i`` is made after ``n_init + i`` observations, and the step
latency ``i`` is the time of the fit and acquisition on ``n_init + i`` observations.
"""
from __future__ import annotations

import math
import os
import textwrap
from typing import Any

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ..evaluation import results as _results
from ..models.registry import MODEL_REGISTRY
from . import style as S

__all__ = [
    "load_trace_frame",
    "comparison_table",
    "plot_trace_panels",
    "plot_latency_curves",
    "plot_queries_to_target",
    "plot_anytime_regret",
    "queries_to_target_table",
]

TRAJECTORY_FILE: str = "trajectories.pkl"

#: (trace field, axis-label key) for the three outcome panels, in reading order.
TRACE_PANELS: tuple[tuple[str, str], ...] = (
    ("recommended_regret_per_step", "simple_regret"),
    ("exploration_per_step", "exploration_score"),
    ("r2_per_step", "r2"),
)

#: Extra trajectory fields the A8 / A9 figures need on top of :data:`TRACE_PANELS`: the true value of each
#: query (A8's basis) and the end-of-run ground truth with its survivor mask (to state A8's target on the
#: same range the regrets use). Loaded as object columns because their lengths differ from the panels'.
EXTRA_TRACE_FIELDS: tuple[str, ...] = ("real_values", "y_gt", "survivors")

#: Wall-clock grid of the A9 figure: log-spaced between the fastest single step and the slowest full run in
#: the frame, so both ends of a two-orders-of-magnitude cost difference are visible.
ANYTIME_GRID_POINTS: int = 120

#: Trace fields that are heavy-tailed (early GP predictions reach R^2 far below -1; latency has weight-load
#: outliers), so they are summarized by the median over channels with an interquartile band.
HEAVY_TAILED_FIELDS: frozenset[str] = frozenset({"r2_per_step", "latency"})

#: Columns per row of the shared legend.
LEGEND_COLUMNS: int = 4

#: Models left out of the latency figure: random search is trivially the fastest, and GP-fixed (no
#: hyperparameter fitting) is not a like-for-like cost comparison with the fitted models.
#: The latency figure compares the PFN family with the GP it is meant to replace (GP-MLL) and nothing else:
#: an include rule, so a new baseline or GP variant never slips in (2026-09-25).
LATENCY_GP_REFERENCE: str = "gp_mll"


def _in_latency_figure(model: str) -> bool:
    """Whether a model belongs in the latency figure: any PFN, plus the GP-MLL reference."""
    spec = MODEL_REGISTRY.get(model)
    return model == LATENCY_GP_REFERENCE or (spec is not None and spec.family == "pfn")

LATENCY_NOTE: str = "Latency comparison pending the converged-GP fix (P0.10); not evidence of a speed advantage."
#: Characters per line of the latency figure's title and caveat (single-column width).
LATENCY_WRAP: int = 55


def load_trace_frame(run_dir: str) -> pd.DataFrame | None:
    """Load a run's trajectories into a frame with one row per BO repetition.

    Args:
        run_dir: Run directory containing ``trajectories.pkl``.

    Returns:
        Frame with the tidy key columns, ``n_init``, ``latency`` and one array column per trace
        field (``None`` when a repetition did not record it); ``None`` if there is no pickle.
    """
    path = os.path.join(run_dir, TRAJECTORY_FILE)
    if not os.path.exists(path):
        return None
    records: list[dict[str, Any]] = []
    for key, traj in _results.read_trajectories(path).items():
        rec = dict(zip(_results.KEY_COLUMNS, key))
        recs = traj["best_rec_indices"]
        rec["n_init"] = len(traj["observed_indices"]) - (len(recs) - 1)
        rec["latency"] = np.asarray(traj["step_times_s"], dtype=float)
        for field, _ in TRACE_PANELS:
            value = traj.get(field)
            rec[field] = None if value is None else np.asarray(value, dtype=float)
        for field in EXTRA_TRACE_FIELDS:
            value = traj.get(field)
            rec[field] = None if value is None else np.asarray(value)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _channel_band(
    sub: pd.DataFrame, field: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Curve over channels with its band (repetitions reduced within a channel first).

    Bounded fields give the mean over channels with a 95% CI; heavy-tailed fields
    (:data:`HEAVY_TAILED_FIELDS`) give the median of per-channel medians with the interquartile range over
    channels.

    Args:
        sub: Rows of one series.
        field: Array column.

    Returns:
        ``(centre, lower, upper, n_channels)``, each curve ``[L]``, or ``None`` if no row has the field.

    Raises:
        ValueError: If the rows of one series have curves of different lengths.
    """
    robust = field in HEAVY_TAILED_FIELDS
    reduce_reps = np.nanmedian if robust else np.nanmean
    curves: list[np.ndarray] = []
    for _, grp in sub.groupby(["subject", "emg"], dropna=False):
        arrays = [a for a in grp[field] if a is not None]
        if not arrays:
            continue
        if len({a.shape[0] for a in arrays}) != 1:
            raise ValueError(f"Curves of {field!r} in one series have different lengths.")
        curves.append(reduce_reps(np.vstack(arrays), axis=0))            # [L]
    if not curves:
        return None
    stack = np.vstack(curves)                                            # [n_channels, L]
    n = stack.shape[0]
    if robust:
        return (np.nanmedian(stack, axis=0), np.nanpercentile(stack, 25, axis=0),
                np.nanpercentile(stack, 75, axis=0), n)
    mean = np.nanmean(stack, axis=0)                                     # [L]
    ci = 1.96 * np.nanstd(stack, axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros_like(mean)
    ci = np.nan_to_num(ci, nan=0.0)
    return mean, mean - ci, mean + ci, n


def _series(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame, S.ModelStyle]]:
    """Split a frame into (model, acquisition-config) series with their styles.

    Args:
        frame: Trace frame.

    Returns:
        ``(model, acq_label, rows, style)`` in model order then acquisition name.
    """
    labels_of = {m: sorted(set(g["acq_label"])) for m, g in frame.groupby("model")}
    order = {m: i for i, m in enumerate(S.MODEL_ORDER)}
    out = []
    for (model, label), rows in sorted(
        frame.groupby(["model", "acq_label"]), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1])
    ):
        out.append((model, label, rows, S.series_style(model, label, labels_of[model])))
    return out


def _title(frame: pd.DataFrame, dataset: str, what: str) -> str:
    """Figure title stating the dataset and how many repetitions and channels back it."""
    n_chan = int(frame[["subject", "emg"]].drop_duplicates().shape[0])
    return f"{S.DATASET_LABELS.get(dataset, dataset)} - {what} (n={int(frame['rep'].nunique())} reps, {n_chan} channels)"


def _legend_below(fig, axes, n_series: int) -> None:
    """Place one shared legend under the axes (the constrained layout reserves the room for it).

    Args:
        fig: Figure created with ``layout=S.LAYOUT_ENGINE``.
        axes: Axes whose handles are used.
        n_series: Number of legend entries.
    """
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(LEGEND_COLUMNS, n_series), frameon=False,
               fontsize=S.FONT_SIZES["legend"])


def plot_trace_panels(frame: pd.DataFrame, out_dir: str, *, dataset: str, name: str = "trajectories") -> list[str]:
    """(a) simple regret, (b) exploration score, (c) R^2 against BO iteration.

    Regret and exploration show the mean over channels with a 95% CI; R^2 (heavy-tailed) the median with an
    interquartile band over channels, its axis clipped at :data:`style.R2_AXIS_FLOOR` for display only.

    Args:
        frame: Trace frame from :func:`load_trace_frame`.
        out_dir: Destination directory.
        dataset: Dataset name.
        name: Output basename.

    Returns:
        Paths written (empty when no repetition recorded any trace).
    """
    panels = [(f, k) for f, k in TRACE_PANELS if frame[f].notna().any()]
    if not panels:
        return []
    series = _series(frame)
    fig, axes = S.figure("double", nrows=1, ncols=len(panels), layout=S.LAYOUT_ENGINE)
    axes = np.atleast_1d(axes)
    for ax, (field, key) in zip(axes, panels):
        for model, label, rows, st in series:
            band = _channel_band(rows, field)
            if band is None:
                continue
            centre, lower, upper, _ = band
            x = int(rows["n_init"].iloc[0]) + np.arange(centre.shape[0])
            ax.plot(x, centre, color=st.color, linestyle=st.linestyle, linewidth=S.LINE_WIDTH, label=st.label,
                    zorder=st.zorder)
            ax.fill_between(x, lower, upper, color=st.color, alpha=S.BAND_ALPHA, linewidth=0)
        ax.set_xlabel(S.axis_label("budget"))
        ax.set_ylabel(S.axis_label(f"{key}_median" if field in HEAVY_TAILED_FIELDS else key))
        if key in ("simple_regret", "exploration_score"):
            ax.set_ylim(0.0, 1.0)
        if key == "r2":
            ax.set_ylim(bottom=S.R2_AXIS_FLOOR, top=1.0)
            ax.text(0.98, 0.02, f"axis clipped at {S.R2_AXIS_FLOOR:g}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=S.FONT_SIZES["annotation"], alpha=0.7)
    fig.suptitle(_title(frame, dataset, "BO trajectories"), x=0.01, ha="left", fontsize=S.FONT_SIZES["title"])
    S.panel_letters(axes, x=0.0)
    _legend_below(fig, axes, len(series))
    return S.save_figure(fig, out_dir, name)


def plot_latency_curves(frame: pd.DataFrame, out_dir: str, *, dataset: str, name: str = "latency") -> list[str]:
    """Per-step latency along the BO run for the PFNs and GP-MLL only (median over channels, IQR band), log scale.

    Args:
        frame: Trace frame.
        out_dir: Destination directory.
        dataset: Dataset name.
        name: Output basename.

    Returns:
        Paths written (empty when no latency was recorded).
    """
    frame = frame[(frame["latency"].map(len) > 0) & frame["model"].map(_in_latency_figure)]
    if frame.empty:
        return []
    frame = frame.assign(latency=frame["latency"].map(lambda a: a if len(a) else None))
    series = _series(frame)
    # A lone panel takes the single-column width, so PANEL_ASPECT does not turn it into a tall strip.
    fig, ax = S.figure("single", layout=S.LAYOUT_ENGINE)
    for model, label, rows, st in series:
        band = _channel_band(rows, "latency")
        if band is None:
            continue
        centre, lower, upper, _ = band
        x = int(rows["n_init"].iloc[0]) + np.arange(centre.shape[0])
        ax.plot(x, centre, color=st.color, linestyle=st.linestyle, linewidth=S.LINE_WIDTH, label=st.label,
                zorder=st.zorder)
        ax.fill_between(x, np.maximum(lower, np.finfo(float).tiny), upper, color=st.color,
                        alpha=S.BAND_ALPHA, linewidth=0)
    ax.set_yscale("log")
    # Label 1-2-5 steps in plain seconds: a log axis spanning less than a decade otherwise shows one tick label.
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(S.axis_label("budget"))
    ax.set_ylabel(S.axis_label("mean_query_latency_s_median"))
    # Single-column figure: wrap the title, and carry the G1 caveat as the legend's title so the constrained
    # layout reserves room for both (a separate supxlabel collided with the legend below the axes).
    fig.suptitle(textwrap.fill(_title(frame, dataset, "per-step cost"), LATENCY_WRAP), x=0.01, ha="left",
                 fontsize=S.FONT_SIZES["title"])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(2, len(series)), frameon=False,
               fontsize=S.FONT_SIZES["legend"], title=textwrap.fill(LATENCY_NOTE, LATENCY_WRAP),
               title_fontsize=S.FONT_SIZES["annotation"])
    return S.save_figure(fig, out_dir, name)


def _queries_to_target(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Per-repetition queries-to-target, derived from the trajectories (roadmap A8).

    Computed here rather than read from ``tidy.csv`` so a run recorded before the columns were populated
    (every run before 2026-09-30) still produces the figure from its pickle alone. The arithmetic is the
    canonical :func:`evaluation.metrics.queries_to_target`, not a copy of it.

    Args:
        frame: Trace frame from :func:`load_trace_frame`.
        fraction: Target fraction of the ground-truth range.

    Returns:
        ``frame`` with a ``qtt`` column: the 1-based query index at which the target was first reached, or
        NaN for a right-censored repetition.
    """
    from ..evaluation.metrics import queries_to_target  # noqa: PLC0415 - avoids a viz -> eval import at load

    key = f"queries_to_target_{int(round(fraction * 100))}"
    values: list[float] = []
    for _, row in frame.iterrows():
        if row["real_values"] is None or row["y_gt"] is None:
            values.append(float("nan"))
            continue
        survivors = row["survivors"]
        values.append(queries_to_target(
            np.asarray(row["y_gt"], dtype=float),
            np.asarray(row["real_values"], dtype=float),
            reference=None if survivors is None else np.asarray(survivors, dtype=bool),
            fractions=(fraction,),
        )[key])
    return frame.assign(qtt=values)


def plot_queries_to_target(
    frame: pd.DataFrame,
    out_dir: str,
    *,
    dataset: str,
    fractions: tuple[float, ...] = (0.90, 0.95),
    name: str = "queries_to_target",
) -> list[str]:
    """A8: fraction of runs that have reached a target response, against query index.

    One panel per target fraction, one line per model: the reach-by-k curve ``mean(qtt <= k)``. Runs that
    never reach the target are **right-censored**, so each curve simply plateaus below 1 -- the plateau height
    is the share of (channel, repetition) runs that got there at all, which is as much the result as the
    speed. Repetitions are reduced within a channel first, so the curve weights channels equally.

    Args:
        frame: Trace frame from :func:`load_trace_frame`.
        out_dir: Destination directory.
        dataset: Dataset name.
        fractions: Target fractions of the ground-truth range.
        name: Output basename.

    Returns:
        Paths written (empty when no repetition recorded the queried values).
    """
    if "real_values" not in frame or frame["real_values"].isna().all():
        return []
    series = _series(frame)
    budget = int(max(len(v) for v in frame["real_values"] if v is not None))
    steps = np.arange(1, budget + 1)
    fig, axes = S.figure("double", ncols=len(fractions), layout=S.LAYOUT_ENGINE, squeeze=False)
    for ax, fraction in zip(axes[0], fractions):
        for model, label, rows, st in series:
            sub = _queries_to_target(rows, fraction)
            per_channel = []
            for _keys, chan in sub.groupby(["subject", "emg"], dropna=False):
                qtt = chan["qtt"].to_numpy(dtype=float)
                # NaN (never reached) never satisfies <=, so a censored run lowers the curve at every k,
                # which is the intended reading.
                per_channel.append([(np.nan_to_num(qtt, nan=np.inf) <= k).mean() for k in steps])
            if not per_channel:
                continue
            curve = np.mean(np.asarray(per_channel), axis=0)
            ax.plot(steps, curve, color=st.color, linestyle=st.linestyle, linewidth=S.LINE_WIDTH,
                    label=st.label, zorder=st.zorder)
        ax.set_xlabel(S.axis_label("budget"))
        ax.set_ylabel(S.axis_label("reached_fraction"))
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{int(round(fraction * 100))}% of the optimum")
    fig.suptitle(_title(frame, dataset, "queries to target"), x=0.01, ha="left",
                 fontsize=S.FONT_SIZES["title"])
    S.panel_letters(axes[0], x=0.0)
    _legend_below(fig, axes[0], len(series))
    return S.save_figure(fig, out_dir, name)


def queries_to_target_table(
    frame: pd.DataFrame,
    out_dir: str,
    *,
    fractions: tuple[float, ...] = (0.90, 0.95),
    name: str = "queries_to_target.csv",
) -> tuple[pd.DataFrame, str] | None:
    """A8's companion table: median [IQR] queries to target per model, with the censored count.

    The median is reported as ``> budget`` when more than half the runs never reached the target, because a
    median over the reaching runs alone would describe a subset selected on the outcome.

    Args:
        frame: Trace frame.
        out_dir: Destination directory.
        fractions: Target fractions.
        name: Output filename.

    Returns:
        ``(table, path)``, or ``None`` when the frame carries no queried values.
    """
    if "real_values" not in frame or frame["real_values"].isna().all():
        return None
    records: list[dict[str, Any]] = []
    for fraction in fractions:
        for model, label, rows, _st in _series(frame):
            sub = _queries_to_target(rows, fraction)
            qtt = sub["qtt"].to_numpy(dtype=float)
            reached = qtt[np.isfinite(qtt)]
            censored = int(qtt.size - reached.size)
            budget = int(max(len(v) for v in sub["real_values"] if v is not None))
            enough = reached.size > qtt.size / 2
            records.append({
                "model": model,
                "acq_label": label,
                "target": fraction,
                "n_runs": int(qtt.size),
                "n_censored": censored,
                "reached_fraction": float(reached.size / qtt.size) if qtt.size else float("nan"),
                "median": float(np.median(reached)) if enough else float("nan"),
                "median_display": f"{np.median(reached):.0f}" if enough else f"> {budget}",
                "q25": float(np.quantile(reached, 0.25)) if enough else float("nan"),
                "q75": float(np.quantile(reached, 0.75)) if enough else float("nan"),
            })
    table = pd.DataFrame.from_records(records)
    path = os.path.join(out_dir, name)
    table.to_csv(path, index=False)
    return table, path


def _nanmean(block: np.ndarray) -> np.ndarray:
    """Column-wise mean of finite entries; NaN where a column has none, without a warning.

    The A9 grid deliberately extends below a model's first completed step, so whole columns are legitimately
    undefined. ``np.nanmean`` returns the right answer there but warns on every all-NaN slice, and a warning
    on expected behaviour trains the reader to ignore warnings.

    Args:
        block: Values with runs on axis 0, shape [R, G].

    Returns:
        Mean per column, shape [G].
    """
    finite = np.isfinite(block)
    count = finite.sum(axis=0)                                                      # [G]
    total = np.where(finite, block, 0.0).sum(axis=0)                                # [G]
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _nanstd(block: np.ndarray) -> np.ndarray:
    """Column-wise sample standard deviation of finite entries; 0 where fewer than two exist.

    Args:
        block: Values with runs on axis 0, shape [R, G].

    Returns:
        Standard deviation per column, shape [G].
    """
    finite = np.isfinite(block)
    count = finite.sum(axis=0)                                                      # [G]
    mean = _nanmean(block)                                                          # [G]
    dev = np.where(finite, (block - mean) ** 2, 0.0).sum(axis=0)                    # [G]
    return np.where(count > 1, np.sqrt(dev / np.maximum(count - 1, 1)), 0.0)


def plot_anytime_regret(
    frame: pd.DataFrame,
    out_dir: str,
    *,
    dataset: str,
    name: str = "anytime_regret",
) -> list[str]:
    """A9: best-so-far simple regret against cumulative wall-clock seconds (log x).

    The comparison Hyp A's cost claim actually needs: at equal elapsed compute, not at equal query count.
    Each repetition's regret is resampled onto one shared log-spaced time grid by
    :func:`evaluation.metrics.anytime_regret_curve`, repetitions are reduced within a channel, and the band is
    a 95% CI over channels. A curve starts only once that model's first step has finished, so a slow first
    step shows as a late start rather than as free progress.

    Carries the same caveat as the latency figure: until the converged-GP fix (P0.10) lands, the GP arm's
    per-step cost is not a converged GP's cost, so this is descriptive (guardrail G1).

    Args:
        frame: Trace frame.
        out_dir: Destination directory.
        dataset: Dataset name.
        name: Output basename.

    Returns:
        Paths written (empty when no repetition recorded both latency and regret).
    """
    from ..evaluation.metrics import anytime_regret_curve  # noqa: PLC0415

    field = "recommended_regret_per_step"
    usable = frame[frame["latency"].notna() & frame[field].notna()]
    if usable.empty:
        return []
    totals = [float(np.sum(v)) for v in usable["latency"] if v is not None and len(v)]
    firsts = [float(v[0]) for v in usable["latency"] if v is not None and len(v)]
    if not totals:
        return []
    grid = np.geomspace(max(min(firsts), 1e-4), max(totals), ANYTIME_GRID_POINTS)   # [G]
    fig, ax = S.figure("onehalf", layout=S.LAYOUT_ENGINE)
    series = _series(usable)
    for model, label, rows, st in series:
        per_channel = []
        for _keys, chan in rows.groupby(["subject", "emg"], dropna=False):
            curves = np.asarray([anytime_regret_curve(r["latency"], r[field], grid)
                                 for _, r in chan.iterrows()])                      # [R, G]
            per_channel.append(_nanmean(curves))
        block = np.asarray(per_channel)                                             # [C, G]
        n = np.sum(np.isfinite(block), axis=0)                                      # [G]
        centre = _nanmean(block)
        sd = _nanstd(block) if block.shape[0] > 1 else np.zeros_like(centre)
        half = 1.96 * sd / np.sqrt(np.maximum(n, 1))
        ax.plot(grid, centre, color=st.color, linestyle=st.linestyle, linewidth=S.LINE_WIDTH,
                label=st.label, zorder=st.zorder)
        ax.fill_between(grid, centre - half, centre + half, color=st.color, alpha=S.BAND_ALPHA, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel(S.axis_label("elapsed_s"))
    ax.set_ylabel(S.axis_label("simple_regret"))
    ax.set_ylim(0.0, 1.0)
    ax.text(0.98, 0.98, textwrap.fill(LATENCY_NOTE, LATENCY_WRAP), transform=ax.transAxes, ha="right",
            va="top", fontsize=S.FONT_SIZES["annotation"], alpha=0.8)
    fig.suptitle(_title(usable, dataset, "anytime regret"), x=0.01, ha="left", fontsize=S.FONT_SIZES["title"])
    _legend_below(fig, np.atleast_1d(ax), len(series))
    return S.save_figure(fig, out_dir, name)

def comparison_table(frame: pd.DataFrame, out_dir: str) -> tuple[pd.DataFrame, str]:
    """Final values per model x acquisition, for choosing a headline acquisition.

    Repetitions are reduced within a channel first. Bounded quantities (final simple regret, regret AUC,
    final exploration score) are the mean over channels with a 95% CI half-width (``*_ci``); heavy-tailed ones
    (final R^2, per-step latency averaged over the run) are the median over channels with the interquartile
    range (``*_q25``, ``*_q75``), so one badly fitted channel cannot dominate.

    Args:
        frame: Trace frame.
        out_dir: Destination directory.

    Returns:
        ``(table, csv_path)``.
    """
    records: list[dict[str, Any]] = []
    for model, label, rows, _ in _series(frame):
        rec: dict[str, Any] = {
            "model": model,
            "acq_label": label,
            "n_reps": int(rows["rep"].nunique()),
            "n_channels": int(rows[["subject", "emg"]].drop_duplicates().shape[0]),
        }
        for field, key in TRACE_PANELS + (("latency", "latency_s"),):
            band = _channel_band(rows, field) if rows[field].notna().any() else None
            if band is None:
                continue
            centre, lower, upper, _ = band
            summary = np.nanmean if field == "latency" else (lambda v: v[-1])  # latency: mean over the run
            if field in HEAVY_TAILED_FIELDS:
                stem = "latency_s_median" if field == "latency" else f"{key}_final"
                rec[stem] = float(summary(centre))
                rec[f"{stem}_q25"] = float(summary(lower))
                rec[f"{stem}_q75"] = float(summary(upper))
            else:
                rec[f"{key}_final"] = float(summary(centre))
                rec[f"{key}_final_ci"] = float(summary(upper) - summary(centre))
            if field == "recommended_regret_per_step":
                rec["simple_regret_auc"] = float(centre.mean())
        records.append(rec)
    table = pd.DataFrame.from_records(records)
    path = os.path.join(out_dir, "comparison_table.csv")
    table.to_csv(path, index=False)
    return table, path
