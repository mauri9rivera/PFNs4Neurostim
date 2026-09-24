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
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation import results as _results
from . import style as S

__all__ = [
    "load_trace_frame",
    "comparison_table",
    "plot_trace_panels",
    "plot_latency_curves",
]

TRAJECTORY_FILE: str = "trajectories.pkl"

#: (trace field, axis-label key) for the three outcome panels, in reading order.
TRACE_PANELS: tuple[tuple[str, str], ...] = (
    ("recommended_regret_per_step", "simple_regret"),
    ("exploration_per_step", "exploration_score"),
    ("r2_per_step", "r2"),
)

#: Trace fields that are heavy-tailed (early GP predictions reach R^2 far below -1; latency has weight-load
#: outliers), so they are summarized by the median over channels with an interquartile band.
HEAVY_TAILED_FIELDS: frozenset[str] = frozenset({"r2_per_step", "latency"})

#: Columns per row of the shared legend.
LEGEND_COLUMNS: int = 4

#: Models left out of the latency figure: random search is trivially the fastest, and GP-fixed (no
#: hyperparameter fitting) is not a like-for-like cost comparison with the fitted models.
LATENCY_EXCLUDED_MODELS: tuple[str, ...] = ("random", "gp_naive")

LATENCY_NOTE: str = "Latency comparison pending the converged-GP fix (P0.10); not evidence of a speed advantage."


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
    fig, axes = S.figure("double", nrows=1, ncols=len(panels), aspect=S.MULTIPANEL_ASPECT, layout=S.LAYOUT_ENGINE)
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
    """Per-step latency along the BO run (median over channels, interquartile band), log scale.

    Args:
        frame: Trace frame.
        out_dir: Destination directory.
        dataset: Dataset name.
        name: Output basename.

    Returns:
        Paths written (empty when no latency was recorded).
    """
    frame = frame[(frame["latency"].map(len) > 0) & ~frame["model"].isin(LATENCY_EXCLUDED_MODELS)]
    if frame.empty:
        return []
    frame = frame.assign(latency=frame["latency"].map(lambda a: a if len(a) else None))
    series = _series(frame)
    fig, ax = S.figure("onehalf", aspect=S.SINGLE_PANEL_ASPECT, layout=S.LAYOUT_ENGINE)
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
    ax.set_xlabel(S.axis_label("budget"))
    ax.set_ylabel(S.axis_label("mean_query_latency_s_median"))
    fig.suptitle(_title(frame, dataset, "per-step cost"), x=0.01, ha="left", fontsize=S.FONT_SIZES["title"])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False, fontsize=S.FONT_SIZES["legend"])
    fig.supxlabel(LATENCY_NOTE, x=0.01, ha="left", fontsize=S.FONT_SIZES["annotation"], alpha=0.75)
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
