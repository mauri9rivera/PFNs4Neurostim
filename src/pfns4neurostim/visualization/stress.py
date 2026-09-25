"""Hypothesis B stress-regime figures and tables (roadmap S2, S9).

Everything here reads the sweep's ``tidy.csv`` and nothing else, so every figure
can be regenerated without re-running an experiment (``--replot``). All geometry,
colours, labels and axis wording come from :mod:`pfns4neurostim.visualization.style`.

Deliverables:
    ``degradation_{demo}.svg``  regret and R-squared vs the knob axis, per model,
                               with 95% CI bands, faint per-channel lines and
                               breakdown points marked.
    ``outcomes_{demo}.svg``     simple regret, exploration score and R-squared vs the knob axis.
    ``robustness.csv``          breakdown point, degradation AUC, CVaR-10%,
                                relative robustness, per model.
    ``breakdown_vs_budget.svg`` breakdown point (TOST vs GP-MLL) against the BO budget, one line per model.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..evaluation.robustness import (
    noninferiority,
    bootstrap_ci,
    breakdown_point,
    cvar,
    degradation_auc,
    relative_robustness,
)
from . import style as S
from . import traces as T

__all__ = [
    "clip_low_snr",
    "plot_degradation_curves",
    "plot_outcome_panels",
    "build_robustness_table",
    "breakdown_vs_budget",
    "plot_breakdown_vs_budget",
    "render_all",
]

#: Rule used to declare a breakdown against the reference. Non-inferiority (2026-09-21, user decision) replaces
#: the two-sided TOST equivalence, which fails whenever a model is clearly better than the reference too.
BREAKDOWN_TEST: str = "noninferiority"

#: Reference model for equivalence testing and relative robustness (roadmap S8).
REFERENCE_MODEL: str = "gp_mll"

#: Metrics with heavy tails (measured 2026-09-23: R^2 excess kurtosis up to ~490, latency ~100, NLL ~780), so
#: they are summarized by the median over channels with an interquartile band. Bounded metrics (regret,
#: exploration score) keep the mean with a 95% CI.
HEAVY_TAILED_METRICS: frozenset[str] = frozenset({"r2", "mean_query_latency_s", "nll", "crps"})

#: Optional lower bound on achieved SNR (dB) below which a (channel, level) cell is left out of the figures
#: and the robustness tables. **Off by default** since the stress restructure of 2026-09-23: K2-global and
#: K2-channel ladders are meant to reach negative SNR, so those levels are plotted (this reverses the earlier
#: same-day 0 dB clip). Pass a number to re-enable it. ``tidy.csv`` is never modified.
DEFAULT_MIN_SNR_DB: float | None = None

#: A level backed by fewer channels than this is dropped, since a mean over so few is not a curve point.
MIN_CHANNELS_PER_LEVEL: int = 3

#: Written next to the figures: how many channels back every level before and after the SNR clip.
CHANNEL_COUNTS_FILE: str = "channel_counts.csv"


def _knob_axis(df: pd.DataFrame, knob: str) -> str:
    """Return the column to use as the x-axis for this knob.

    K2 is plotted against achieved SNR in dB, never against the raw level
    (roadmap S2); other knobs fall back to the level itself.

    Args:
        df: Tidy sweep frame.
        knob: Knob name.

    Returns:
        A column name present in ``df``.
    """
    preferred = S.KNOB_X_AXIS.get(knob, "level")
    if preferred in df.columns and df[preferred].notna().any():
        return preferred
    return "level"


#: Knobs whose *level* gets more severe as it decreases (the level is a budget). Every other knob (K2 alpha,
#: K5 contamination, K6 dropout) gets more severe as the level increases, whatever the plotted x-axis does.
LEVEL_SEVERITY_DESCENDING: frozenset[str] = frozenset({"k6_budget"})


#: Axes on which a *lower* value means more stress (SNR, or SNR degradation relative to a channel's floor).
X_DESCENDS_WITH_STRESS: frozenset[str] = frozenset({"achieved_snr_db", "target_delta_snr_db"})

#: Column holding a calibrated ladder shared by every channel (dB of SNR degradation vs each channel's floor).
CALIBRATED_TARGET_COLUMN: str = "target_delta_snr_db"


def _is_calibrated(df: pd.DataFrame) -> bool:
    """Whether the sweep used per-channel calibrated levels (a shared dB ladder)."""
    return CALIBRATED_TARGET_COLUMN in df.columns and bool(df[CALIBRATED_TARGET_COLUMN].notna().all())


def _shared_ladder(df: pd.DataFrame) -> pd.DataFrame:
    """Use the shared dB ladder as the level when the sweep was calibrated per channel.

    Calibrated sweeps (K5, and K2 in dB form) solve a different raw level per channel to hit the same SNR
    degradation, so the raw level (a contamination fraction) is different in every channel and there is no
    shared ladder to test or average over. The target dB is the shared ladder, so it becomes ``level``;
    the raw value is kept as ``level_raw``.

    Args:
        df: Tidy sweep frame.

    Returns:
        The frame with ``level`` replaced by the target dB when calibrated, otherwise unchanged.
    """
    if not _is_calibrated(df) or "level_raw" in df.columns:
        return df
    return df.assign(level_raw=df["level"], level=df[CALIBRATED_TARGET_COLUMN].astype(float))


def _trace_frame_on_ladder(frame: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Re-key a trace frame's raw levels to the shared calibrated ladder of ``df`` (no-op otherwise).

    Args:
        frame: Trace frame (its ``level`` is the raw per-channel level).
        df: Tidy frame after :func:`_shared_ladder`.

    Returns:
        The trace frame with ``level`` on the shared ladder.
    """
    if "level_raw" not in df.columns:
        return frame
    mapping = df[["subject", "emg", "level_raw", "level"]].drop_duplicates().rename(columns={"level": "ladder"})
    mapping = mapping.assign(key=mapping["level_raw"].astype(float).round(12))
    keyed = frame.assign(key=frame["level"].astype(float).round(12))
    merged = keyed.merge(mapping[["subject", "emg", "key", "ladder"]], on=["subject", "emg", "key"], how="inner")
    return merged.assign(level=merged["ladder"]).drop(columns=["key", "ladder"])


def _restrict_to_cells(frame: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Keep the trace-frame rows whose (channel, level) cell survives in ``df`` (e.g. after the SNR clip).

    Args:
        frame: Trace frame with ``level`` already on the shared ladder.
        df: Tidy frame after :func:`clip_low_snr`.

    Returns:
        The rows of ``frame`` whose (subject, emg, level) also occurs in ``df``.
    """
    cells = df[["subject", "emg", "level"]].drop_duplicates()
    cells = cells.assign(_key=cells["level"].astype(float).round(12)).drop(columns="level")
    keyed = frame.assign(_key=frame["level"].astype(float).round(12))
    return keyed.merge(cells, on=["subject", "emg", "_key"], how="inner").drop(columns="_key")


def _level_severity_ascending(knob: str, df: pd.DataFrame | None = None) -> bool:
    """Whether a larger knob *level* means more stress (the order in which breakdown walks the ladder).

    This is about the level, not the plotted axis: K2 is plotted against achieved SNR, which falls with
    stress, but its level (alpha) rises with stress. Passing the axis direction here made the walk start at
    the harshest level and report a breakdown there for every model. A calibrated ladder is in dB of SNR
    degradation, where a more negative level is more severe.

    Args:
        knob: Knob name.
        df: The sweep frame after :func:`_shared_ladder`, to detect a calibrated ladder.

    Returns:
        True when larger levels are more severe.
    """
    if df is not None and _is_calibrated(df):
        return False
    return knob not in LEVEL_SEVERITY_DESCENDING


def _severity_ascending(x_col: str) -> bool:
    """Whether a larger x means more stress.

    Achieved SNR *decreases* with stress, so severity is descending in x.

    Args:
        x_col: The x-axis column name.

    Returns:
        True when larger x means more severe stress.
    """
    return x_col not in X_DESCENDS_WITH_STRESS


def _model_curve(sub: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate one model's metric into a mean curve with a 95% CI half-width.

    Channels are averaged within a level first, so a subject with many EMGs does
    not dominate the mean.

    Args:
        sub: Rows for one model.
        x_col: X-axis column.
        y_col: Metric column.

    Returns:
        ``(x, mean, ci_half)`` sorted by increasing x.
    """
    per_channel = (
        sub.groupby(["level", "subject", "emg"], dropna=False)
        .agg(x=(x_col, "mean"), y=(y_col, "mean"))
        .reset_index()
    )
    grouped = per_channel.groupby("level", dropna=False)
    x = grouped["x"].mean().to_numpy(dtype=float)
    mean = grouped["y"].mean().to_numpy(dtype=float)
    counts = grouped["y"].count().to_numpy(dtype=float)
    sd = grouped["y"].std(ddof=1).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ci = 1.96 * sd / np.sqrt(counts)
    ci = np.nan_to_num(ci, nan=0.0)
    order = np.argsort(x)
    return x[order], mean[order], ci[order]


def _level_to_x(df: pd.DataFrame, x_col: str) -> dict[float, float]:
    """Map each knob level to its mean x-axis value.

    Breakdown points are *found* on the level ladder but must be *reported* on the
    plotted axis (achieved SNR for K2), so every level needs its x value.

    Args:
        df: Tidy sweep frame.
        x_col: X-axis column.

    Returns:
        Mapping level -> mean x.
    """
    grouped = df.groupby("level", dropna=False)[x_col].mean()
    return {float(level): float(value) for level, value in grouped.items()}


def _per_level_values(sub: pd.DataFrame, y_col: str) -> dict[float, np.ndarray]:
    """Group a metric by knob level, one array of per-(channel, rep) values.

    Args:
        sub: Rows for one model.
        y_col: Metric column.

    Returns:
        Mapping level -> values.
    """
    ordered = sub.sort_values(["subject", "emg", "rep"])
    return {
        float(level): grp[y_col].to_numpy(dtype=float)
        for level, grp in ordered.groupby("level", dropna=False)
    }


def _is_heavy_tailed(metric: str) -> bool:
    """Whether ``metric`` is summarized by the median and an interquartile band."""
    return metric in HEAVY_TAILED_METRICS


def _model_band(sub: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarize one model's metric per level as ``(x, centre, lower, upper)``.

    Repetitions are reduced within a channel first, then channels are summarized, so a subject with many
    EMGs cannot dominate. Bounded metrics give the mean over channels with a 95% CI; heavy-tailed metrics
    (:data:`HEAVY_TAILED_METRICS`) give the median of per-channel medians with the interquartile range over
    channels, so one badly fitted channel cannot drag the curve.

    Args:
        sub: Rows for one model.
        x_col: X-axis column.
        y_col: Metric column.

    Returns:
        Four arrays sorted by increasing x, each ``[n_levels]``.
    """
    if not _is_heavy_tailed(y_col):
        x, mean, ci = _model_curve(sub, x_col, y_col)
        return x, mean, mean - ci, mean + ci
    per_channel = (
        sub.groupby(["level", "subject", "emg"], dropna=False)
        .agg(x=(x_col, "mean"), y=(y_col, "median"))
        .reset_index()
    )
    grouped = per_channel.groupby("level", dropna=False)
    x = grouped["x"].mean().to_numpy(dtype=float)                                    # [n_levels]
    centre = grouped["y"].median().to_numpy(dtype=float)                             # [n_levels]
    lower = grouped["y"].quantile(0.25).to_numpy(dtype=float)                        # [n_levels]
    upper = grouped["y"].quantile(0.75).to_numpy(dtype=float)                        # [n_levels]
    order = np.argsort(x)
    return x[order], centre[order], lower[order], upper[order]


def clip_low_snr(
    df: pd.DataFrame,
    min_snr_db: float | None = DEFAULT_MIN_SNR_DB,
    *,
    min_channels: int = MIN_CHANNELS_PER_LEVEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop the (channel, level) cells whose achieved SNR is below ``min_snr_db`` from a sweep frame.

    A cell is one channel at one level (every model and repetition share the same stressed channel, so its
    achieved SNR is one number). Cells with an unknown SNR are kept. A level left with fewer than
    ``min_channels`` channels is dropped altogether, with a warning naming it (``min_snr_db=None`` disables
    the clip and this rule). ``tidy.csv`` on disk is never touched: this only decides what the figures and
    robustness tables are built from.

    Because different channels drop out at different levels, the surviving channel set can change from level
    to level; the returned counts let the figures state how many channels back every level.

    Args:
        df: Tidy sweep frame (after :func:`_shared_ladder`, so ``level`` is the shared ladder).
        min_snr_db: Lower bound in dB; ``None`` keeps everything.
        min_channels: Smallest number of channels a level needs to be kept.

    Returns:
        ``(kept_frame, counts)`` where ``counts`` has one row per level with ``level``,
        ``n_channels_total``, ``n_channels_kept`` and ``kept`` (whether the level survives).
    """
    if "achieved_snr_db" not in df.columns:
        min_snr_db = None
    cell = (
        df.groupby(["subject", "emg", "level"], dropna=False)["achieved_snr_db"].mean().rename("snr").reset_index()
        if "achieved_snr_db" in df.columns
        else df[["subject", "emg", "level"]].drop_duplicates().assign(snr=np.nan)
    )
    ok = cell["snr"].isna() | (cell["snr"] >= (-np.inf if min_snr_db is None else min_snr_db))
    total = cell.groupby("level").size()
    kept_n = cell[ok].groupby("level").size().reindex(total.index, fill_value=0)
    counts = pd.DataFrame(
        {"level": total.index.astype(float), "n_channels_total": total.to_numpy(), "n_channels_kept": kept_n.to_numpy()}
    )
    # With the clip disabled every level stays, however few channels back it (a tiny fixture, a one-channel run).
    counts["kept"] = True if min_snr_db is None else counts["n_channels_kept"] >= min_channels
    for _, row in counts[~counts["kept"]].iterrows():
        warnings.warn(
            f"SNR clip (>= {min_snr_db} dB): level {row['level']:g} keeps {int(row['n_channels_kept'])} of "
            f"{int(row['n_channels_total'])} channels (< {min_channels}); the level is dropped from the figures.",
            stacklevel=2,
        )
    keep_levels = set(counts.loc[counts["kept"], "level"])
    keys = cell[ok & cell["level"].astype(float).isin(keep_levels)][["subject", "emg", "level"]]
    kept = df.merge(keys, on=["subject", "emg", "level"], how="inner")
    return kept, counts


def _annotate_counts(ax, df: pd.DataFrame, x_col: str, counts: pd.DataFrame | None) -> None:
    """Write the number of channels kept above each level's x position (top edge of ``ax``).

    Args:
        ax: Axes to annotate.
        df: Frame the curves were built from (supplies each level's mean x).
        x_col: X-axis column.
        counts: Output of :func:`clip_low_snr`; ``None`` draws nothing.
    """
    if counts is None or counts.empty:
        return
    level_x = _level_to_x(df, x_col)
    for _, row in counts[counts["kept"]].iterrows():
        x = level_x.get(float(row["level"]))
        if x is None or not np.isfinite(x):
            continue
        ax.text(x, 1.0, f"{int(row['n_channels_kept'])}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=S.FONT_SIZES["annotation"], alpha=0.8)


def plot_degradation_curves(
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
    metrics: Sequence[str] = ("recommended_regret", "r2"),
    breakdowns: dict[str, float] | None = None,
    show_channels: bool = True,
    counts: pd.DataFrame | None = None,
) -> list[str]:
    """Degradation curves: one panel per metric, one line per model.

    Bounded metrics show the mean over channels with a 95% CI; heavy-tailed ones (R^2) the median with an
    interquartile band (see :func:`_model_band`), and the R^2 axis is clipped for display only.

    Args:
        df: Tidy sweep frame.
        out_dir: Destination directory.
        knob: Knob name (selects the x-axis and its label).
        dataset: Dataset name, for the title.
        metrics: Metric columns, one panel each.
        breakdowns: Optional model -> breakdown point **in x-axis units**
            (``breakdown_x`` of the robustness table), marked on the first (regret) panel.
        show_channels: Draw faint per-channel traces behind the model means.
        counts: Output of :func:`clip_low_snr`; the channels kept per level are written above the first panel.

    Returns:
        Paths written.
    """
    x_col = _knob_axis(df, knob)
    demo = str(df["demo"].iloc[0]) if "demo" in df.columns else "demo2"
    models = [m for m in S.MODEL_ORDER if m in set(df["model"])]

    # One panel per metric side by side (a vertical stack of 1.5-aspect panels would be too tall).
    fig, axes = S.figure("double", nrows=1, ncols=len(metrics), sharex=True, layout=S.LAYOUT_ENGINE)
    axes = np.atleast_1d(axes)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model in models:
            sub = df[df["model"] == model]
            if sub.empty or metric not in sub.columns:
                continue

            if show_channels:
                for (subj, emg), grp in sub.groupby(["subject", "emg"], dropna=False):
                    chan = grp.groupby("level", dropna=False).agg(
                        x=(x_col, "mean"), y=(metric, "median" if _is_heavy_tailed(metric) else "mean")
                    )
                    chan = chan.sort_values("x")
                    ax.plot(
                        chan["x"].to_numpy(),
                        chan["y"].to_numpy(),
                        color=S.model_color(model),
                        alpha=S.FAINT_ALPHA,
                        linewidth=0.5,
                        marker="",
                        zorder=1,
                    )

            x, centre, lower, upper = _model_band(sub, x_col, metric)
            ax.plot(x, centre, **S.plot_kwargs(model))
            ax.fill_between(
                x,
                lower,
                upper,
                color=S.model_color(model),
                alpha=S.BAND_ALPHA,
                linewidth=0,
            )

            if breakdowns and model in breakdowns and np.isfinite(breakdowns[model]):
                if metric == metrics[0]:
                    ax.axvline(
                        breakdowns[model],
                        color=S.model_color(model),
                        linestyle=":",
                        linewidth=0.8,
                        alpha=0.9,
                    )

        ax.set_ylabel(S.axis_label(f"{metric}_median" if _is_heavy_tailed(metric) else metric))
        if metric == "r2":
            ax.set_ylim(bottom=S.R2_AXIS_FLOOR, top=1.0)
            ax.text(0.98, 0.02, f"axis clipped at {S.R2_AXIS_FLOOR:g}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=S.FONT_SIZES["annotation"], alpha=0.7)
        if idx == 0:
            _annotate_counts(ax, df, x_col, counts)

    # Achieved SNR decreases with stress: plot it decreasing left to right so the
    # x-axis always reads "more stress to the right" (roadmap S2).
    if x_col in X_DESCENDS_WITH_STRESS:
        axes[0].invert_xaxis()
    for ax in axes:
        ax.set_xlabel(S.knob_x_label(knob, x_col))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False,
               fontsize=S.FONT_SIZES["legend"])
    note = "; top: channels kept per level" if counts is not None else ""
    fig.suptitle(
        f"{S.DATASET_LABELS.get(dataset, dataset)} - {S.KNOB_LABELS.get(knob, knob)} - "
        f"{S.DEMO_LABELS.get(demo, demo)}{note}",
        x=0.01, ha="left", fontsize=S.FONT_SIZES["title"],
    )
    S.panel_letters(axes, x=0.0, y=1.06 if counts is not None else 1.02)
    return S.save_figure(fig, out_dir, f"degradation_{'invivo' if demo == 'demo2' else 'synthetic'}")


def plot_outcome_panels(
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
    counts: pd.DataFrame | None = None,
) -> list[str]:
    """Final-run outcomes vs the knob axis: (a) simple regret, (b) exploration score, (c) R^2.

    Replaces the 90%-coverage calibration figure (2026-09-21). Lines and bands over channels, one line per
    model; no bars. Regret and exploration show the mean with a 95% CI; R^2 (heavy-tailed) the median with
    an interquartile band, its axis clipped for display only.

    Args:
        df: Tidy sweep frame.
        out_dir: Destination directory.
        knob: Knob name.
        dataset: Dataset name.
        counts: Output of :func:`clip_low_snr`; the channels kept per level are written above panel (a).

    Returns:
        Paths written (empty when the frame has none of the outcome columns).
    """
    panels = [
        (col, key)
        for col, key in (
            ("recommended_regret", "simple_regret"),
            ("exploration_score", "exploration_score"),
            ("r2", "r2"),
        )
        if col in df.columns and df[col].notna().any()
    ]
    if not panels:
        return []

    x_col = _knob_axis(df, knob)
    demo = str(df["demo"].iloc[0]) if "demo" in df.columns else "demo2"
    models = [m for m in S.MODEL_ORDER if m in set(df["model"])]

    fig, axes = S.figure("double", nrows=1, ncols=len(panels), layout=S.LAYOUT_ENGINE)
    axes = np.atleast_1d(axes)
    for ax, (col, key) in zip(axes, panels):
        for model in models:
            sub = df[df["model"] == model]
            if sub.empty or sub[col].isna().all():
                continue
            x, centre, lower, upper = _model_band(sub, x_col, col)
            ax.plot(x, centre, **S.plot_kwargs(model))
            ax.fill_between(x, lower, upper, color=S.model_color(model), alpha=S.BAND_ALPHA, linewidth=0)
        ax.set_ylabel(S.axis_label(f"{key}_median" if _is_heavy_tailed(col) else key))
        ax.set_xlabel(S.knob_x_label(knob, x_col))
        if col == "r2":
            ax.set_ylim(bottom=S.R2_AXIS_FLOOR, top=1.0)
            ax.text(0.98, 0.02, f"axis clipped at {S.R2_AXIS_FLOOR:g}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=S.FONT_SIZES["annotation"], alpha=0.7)
        if x_col in X_DESCENDS_WITH_STRESS:
            ax.invert_xaxis()
    _annotate_counts(axes[0], df, x_col, counts)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False,
               fontsize=S.FONT_SIZES["legend"])
    n_reps = int(df["rep"].nunique()) if "rep" in df.columns else 0
    n_chan = int(df[["subject", "emg"]].drop_duplicates().shape[0])
    note = "; top of (a): channels kept per level" if counts is not None else ""
    fig.suptitle(
        f"{S.DATASET_LABELS.get(dataset, dataset)} - {S.KNOB_LABELS.get(knob, knob)} "
        f"(n={n_reps} reps, {n_chan} channels{note})",
        x=0.01, ha="left", fontsize=S.FONT_SIZES["title"],
    )
    S.panel_letters(axes, x=0.0, y=1.06 if counts is not None else 1.02)
    return S.save_figure(fig, out_dir, f"outcomes_{'invivo' if demo == 'demo2' else 'synthetic'}")


def build_robustness_table(
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    margin: float = 0.05,
    metric: str = "recommended_regret",
    reference: str = REFERENCE_MODEL,
) -> tuple[pd.DataFrame, str]:
    """Compute the S9 robustness metrics per model and write them as CSV.

    Args:
        df: Tidy sweep frame.
        out_dir: Destination directory.
        knob: Knob name (selects the x-axis and severity direction).
        margin: Pre-registered TOST equivalence margin, in the metric's units.
        metric: Metric the robustness statements are about.
        reference: Reference model for TOST and relative robustness.

    Returns:
        ``(table, csv_path)``.
    """
    x_col = _knob_axis(df, knob)
    ascending = _severity_ascending(x_col)
    level_ascending = _level_severity_ascending(knob, df)
    models = [m for m in S.MODEL_ORDER if m in set(df["model"])]
    ref_sub = df[df["model"] == reference]
    level_x = _level_to_x(df, x_col)

    records: list[dict[str, Any]] = []
    for model in models:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        x, mean, _ = _model_curve(sub, x_col, metric)
        rec: dict[str, Any] = {
            "model": model,
            "model_label": S.model_label(model),
            "knob": knob,
            "metric": metric,
            "x_axis": x_col,
            "n_levels": int(len(x)),
            "n_rows": int(len(sub)),
        }
        try:
            rec["degradation_auc"] = degradation_auc(x, mean)
        except ValueError as exc:
            rec["degradation_auc"] = float("nan")
            rec["degradation_auc_note"] = str(exc)
        rec["cvar10_regret"] = cvar(sub[metric].to_numpy(dtype=float), alpha=0.1)
        rec["mean_at_nominal"] = float(mean[-1] if not ascending else mean[0])

        if model != reference and not ref_sub.empty:
            bp = breakdown_point(
                _per_level_values(sub, metric),
                _per_level_values(ref_sub, metric),
                margin=margin,
                severity_ascending=level_ascending,
                test=BREAKDOWN_TEST,
            )
            rec["breakdown_level"] = bp["breakdown_level"]
            rec["breakdown_x"] = level_x.get(bp["breakdown_level"], float("nan"))
            rec["breakdown_reason"] = bp["breakdown_reason"]
            rec["equivalent_levels"] = int(np.sum(bp["equivalent"]))
            # Per-channel breakdown levels give the CI a sample to bootstrap over.
            per_channel_bd: list[float] = []
            for (subj, emg), grp in sub.groupby(["subject", "emg"], dropna=False):
                ref_grp = ref_sub[(ref_sub["subject"] == subj) & (ref_sub["emg"] == emg)]
                if ref_grp.empty:
                    continue
                try:
                    chan_bp = breakdown_point(
                        _per_level_values(grp, metric),
                        _per_level_values(ref_grp, metric),
                        margin=margin,
                        test=BREAKDOWN_TEST,
                        severity_ascending=level_ascending,
                    )
                except ValueError:
                    continue
                per_channel_bd.append(chan_bp["breakdown_level"])
            # The CI is bootstrapped in plotted-axis units, matching breakdown_x.
            per_channel_x = [
                level_x.get(v, float("nan")) for v in per_channel_bd if np.isfinite(v)
            ]
            lo, hi = bootstrap_ci([v for v in per_channel_x if np.isfinite(v)])
            rec["breakdown_ci_lo"], rec["breakdown_ci_hi"] = lo, hi
            rec["n_channels_broken"] = int(np.sum(np.isfinite(per_channel_bd)))
            rec["n_channels"] = int(len(per_channel_bd))

            _, ref_mean, _ = _model_curve(ref_sub, x_col, metric)
            rec.update(relative_robustness(x, mean, ref_mean))
        else:
            rec.update(
                {
                    "breakdown_level": float("nan"),
                    "breakdown_x": float("nan"),
                    "breakdown_reason": "reference",
                    "gap_mean": 0.0,
                    "gap_max": 0.0,
                    "crossing_x": float("nan"),
                }
            )
        records.append(rec)

    table = pd.DataFrame.from_records(records)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "robustness.csv")
    table.to_csv(path, index=False)
    return table, path


#: Fractions of the run's budget at which the breakdown point is re-evaluated.
BREAKDOWN_BUDGET_FRACTIONS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)


def breakdown_vs_budget(
    frame: pd.DataFrame,
    df: pd.DataFrame,
    *,
    knob: str,
    margin: float,
    reference: str = REFERENCE_MODEL,
    fractions: Sequence[float] = BREAKDOWN_BUDGET_FRACTIONS,
) -> pd.DataFrame:
    """Breakdown point of every model as a function of the BO budget.

    At each budget t the recommended-site regret after t observations (read from the stored
    per-step trace, so no re-run) is fed to the same paired-TOST breakdown rule as the
    robustness table.

    Args:
        frame: Trace frame from :func:`traces.load_trace_frame`.
        df: Tidy sweep frame (supplies the knob level -> plotted-axis mapping).
        knob: Knob name.
        margin: Pre-registered TOST margin.
        reference: Reference model.
        fractions: Budget fractions at which to evaluate.

    Returns:
        Long frame with ``model``, ``budget``, ``breakdown_level``, ``breakdown_x``, ``reason``.
    """
    x_col = _knob_axis(df, knob)
    level_ascending = _level_severity_ascending(knob, df)
    level_x = _level_to_x(df, x_col)
    field = "recommended_regret_per_step"
    frame = frame[frame[field].notna()].sort_values(["level", "subject", "emg", "rep"])
    if frame.empty or reference not in set(frame["model"]):
        # A shard that holds only some models (e.g. the GPU job without the GP reference) has no breakdown to report.
        return pd.DataFrame(columns=["model", "budget", "breakdown_level", "breakdown_x", "reason"])
    n_init = int(frame["n_init"].iloc[0])
    total = int(frame[field].iloc[0].shape[0]) - 1 + n_init
    budgets = sorted({int(round(f * total)) for f in fractions if n_init < round(f * total) <= total})

    def per_level(model: str, t: int) -> dict[float, np.ndarray]:
        sub = frame[frame["model"] == model]
        return {
            float(level): np.array([trace[t - n_init] for trace in grp[field]], dtype=float)
            for level, grp in sub.groupby("level", dropna=False)
        }

    records: list[dict[str, Any]] = []
    for model in [m for m in S.MODEL_ORDER if m in set(frame["model"]) and m != reference]:
        for t in budgets:
            bp = breakdown_point(
                per_level(model, t), per_level(reference, t), margin=margin, severity_ascending=level_ascending,
                test=BREAKDOWN_TEST,
            )
            level = bp["breakdown_level"]
            records.append(
                {
                    "model": model,
                    "budget": t,
                    "breakdown_level": level,
                    "breakdown_x": level_x.get(level, float("nan")) if np.isfinite(level) else float("nan"),
                    "reason": bp["breakdown_reason"],
                }
            )
    return pd.DataFrame.from_records(records)


def plot_breakdown_vs_budget(
    table: pd.DataFrame,
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
) -> list[str]:
    """Breakdown point (y) vs BO budget (x), coloured by model.

    A filled marker is a breakdown at that budget; an open marker at the most severe tested level
    means the model never broke down within the ladder.

    Args:
        table: Output of :func:`breakdown_vs_budget`.
        df: Tidy sweep frame.
        out_dir: Destination directory.
        knob: Knob name.
        dataset: Dataset name.

    Returns:
        Paths written (empty when the table is empty).
    """
    if table.empty:
        return []
    x_col = _knob_axis(df, knob)
    ascending = _severity_ascending(x_col)
    level_x = _level_to_x(df, x_col)
    xs = sorted(level_x.values())
    harshest = max(xs) if ascending else min(xs)

    table.to_csv(os.path.join(out_dir, "breakdown_vs_budget.csv"), index=False)
    fig, ax = S.figure("single", layout=S.LAYOUT_ENGINE)   # lone panel: single column (PANEL_ASPECT)
    for model, grp in table.groupby("model"):
        grp = grp.sort_values("budget")
        y = grp["breakdown_x"].fillna(harshest).to_numpy(dtype=float)
        broken = grp["breakdown_x"].notna().to_numpy()
        color = S.model_color(str(model))
        ax.plot(grp["budget"], y, color=color, linewidth=S.LINE_WIDTH, label=S.model_label(str(model)))
        ax.plot(grp["budget"][broken], y[broken], linestyle="none", marker=S.model_style(str(model)).marker,
                color=color, markersize=S.MARKER_SIZE + 1)
        ax.plot(grp["budget"][~broken], y[~broken], linestyle="none", marker=S.model_style(str(model)).marker,
                markerfacecolor="white", color=color, markersize=S.MARKER_SIZE + 1)
    ax.set_xlabel(S.axis_label("budget"))
    ax.set_ylabel(f"Breakdown point\n{S.knob_x_label(knob, x_col)}")
    fig.suptitle(
        f"{S.DATASET_LABELS.get(dataset, dataset)} - {S.KNOB_LABELS.get(knob, knob)} breakdown",
        x=0.01, ha="left", fontsize=S.FONT_SIZES["title"],
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False,
               fontsize=S.FONT_SIZES["legend"])
    return S.save_figure(fig, out_dir, "breakdown_vs_budget")


#: Knobs whose sweep also yields the S5 interaction surface with the K6 budget (read from the per-step traces).
REGIME_KNOBS: frozenset[str] = frozenset({"k2_channel", "k2_global"})

#: Number of BO-budget columns of a regime surface (S5/S10), spread evenly from the first acquisition to the full budget.
REGIME_BUDGET_COLUMNS: int = 8


def regime_surface(
    frame: pd.DataFrame,
    df: pd.DataFrame,
    *,
    knob: str,
    model: str,
    reference: str | None = None,
    margin: float | None = None,
    n_budgets: int = REGIME_BUDGET_COLUMNS,
) -> dict[str, np.ndarray]:
    """Regret over the (knob level x BO budget) plane, read from the stored per-step traces.

    A budget-``B`` run's final recommendation is the recommendation after ``B`` observations,
    which every longer run already records (``recommended_regret_per_step[B - n_init]``), so
    the K2 x K6-budget surface (roadmap S5) costs no extra compute. Each cell is the median
    over channels of the repetition-mean regret. With ``reference`` the cell is instead the
    mean paired difference ``model - reference`` and ``noninferior`` marks the cells where
    the one-sided paired test at ``margin`` passes (the robustness table's rule).

    Args:
        frame: Trace frame (:func:`traces.load_trace_frame`), restricted to the sweep's cells.
        df: Tidy sweep frame (level -> plotted-axis mapping).
        knob: Knob name.
        model: Model whose surface to compute.
        reference: Reference model for a difference surface, or ``None``.
        margin: Non-inferiority margin (required with ``reference``).
        n_budgets: Number of budget columns.

    Returns:
        ``x`` (plotted level values, sorted) [L], ``budgets`` [B], ``value`` [L, B] and, for a
        difference surface, ``noninferior`` [L, B] (bool).

    Raises:
        ValueError: If the model (or reference) has no traces, or ``margin`` is missing.
    """
    field = "recommended_regret_per_step"
    frame = frame[frame[field].notna()]
    needed = {model} | ({reference} if reference else set())
    missing = needed - set(frame["model"])
    if missing:
        raise ValueError(f"regime_surface: no traces for {sorted(missing)}.")
    if reference is not None and margin is None:
        raise ValueError("regime_surface: a difference surface needs the non-inferiority margin.")
    n_init = int(frame["n_init"].iloc[0])
    total = int(frame[field].iloc[0].shape[0]) - 1 + n_init
    budgets = np.unique(np.linspace(n_init + 1, total, n_budgets).round().astype(int))   # [B]
    level_x = _level_to_x(df, _knob_axis(df, knob))
    levels = sorted(set(frame["level"]), key=lambda lv: level_x.get(float(lv), float(lv)))

    def regrets(m: str, level: float, b: int) -> pd.Series:
        """Per-(channel, rep) regret of model ``m`` at a level and budget."""
        sub = frame[(frame["model"] == m) & (frame["level"] == level)]
        values = [trace[b - n_init] for trace in sub[field]]
        return pd.Series(values, index=pd.MultiIndex.from_frame(sub[["subject", "emg", "rep"]]))

    value = np.full((len(levels), budgets.size), np.nan)                                   # [L, B]
    ok = np.zeros_like(value, dtype=bool)                                                  # [L, B]
    for i, level in enumerate(levels):
        for j, b in enumerate(budgets):
            a = regrets(model, level, int(b))
            if reference is None:
                value[i, j] = float(a.groupby(level=["subject", "emg"]).mean().median())
                continue
            r = regrets(reference, level, int(b))
            a, r = a.align(r, join="inner")
            value[i, j] = float(np.mean(a - r))
            ok[i, j] = bool(noninferiority(a.to_numpy(), r.to_numpy(), float(margin))["ok"])
    out = {"x": np.array([level_x.get(float(lv), float(lv)) for lv in levels]), "budgets": budgets, "value": value}
    if reference is not None:
        out["noninferior"] = ok
    return out


def _outline_cells(ax, mask: np.ndarray) -> None:
    """Draw a box around each ``True`` cell of an ``imshow`` grid.

    Args:
        ax: Axes holding the image (cell ``(i, j)`` centred at ``(j, i)``).
        mask: Boolean grid, shape [rows, cols].
    """
    from matplotlib.patches import Rectangle  # noqa: PLC0415 - figure-time import

    for i, j in zip(*np.nonzero(mask)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=S.OUTLINE_COLOR,
                               linewidth=S.LINE_WIDTH))


def _draw_surface(ax, surf: dict[str, np.ndarray], *, cmap: str, vmin: float, vmax: float):
    """Draw one regime surface as an image with labelled budget / level ticks.

    Args:
        ax: Target axes.
        surf: Output of :func:`regime_surface`.
        cmap: Colormap token from :mod:`style`.
        vmin: Lower colour limit.
        vmax: Upper colour limit.

    Returns:
        The ``AxesImage``.
    """
    im = ax.imshow(surf["value"], origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(surf["budgets"].size), [str(int(b)) for b in surf["budgets"]])
    ax.set_yticks(range(surf["x"].size), [f"{x:.3g}" for x in surf["x"]])
    if "noninferior" in surf:
        _outline_cells(ax, surf["noninferior"])
    return im


def plot_regime_heatmap(
    frame: pd.DataFrame,
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
    margin: float,
    reference: str = REFERENCE_MODEL,
) -> list[str]:
    """K2 x K6-budget interaction surface (roadmap S5), one panel per model plus differences.

    Top row: median recommended regret per (level, budget) for every model. Bottom row:
    paired difference of each non-reference model against ``reference``, diverging around
    zero, with the non-inferior cells outlined.

    Args:
        frame: Trace frame restricted to the sweep's cells.
        df: Tidy sweep frame.
        out_dir: Destination directory.
        knob: Knob name (normally a K2 setting).
        dataset: Dataset name.
        margin: Pre-registered non-inferiority margin.
        reference: Reference model of the difference row.

    Returns:
        Paths written (the figure and ``regime_surface.csv``).
    """
    models = [m for m in S.MODEL_ORDER if m in set(frame["model"])]
    others = [m for m in models if m != reference] if reference in models else []
    surfaces = {m: regime_surface(frame, df, knob=knob, model=m) for m in models}
    diffs = {m: regime_surface(frame, df, knob=knob, model=m, reference=reference, margin=margin) for m in others}

    records = []
    for kind, table in (("regret", surfaces), ("difference", diffs)):
        for m, surf in table.items():
            for i, x in enumerate(surf["x"]):
                for j, b in enumerate(surf["budgets"]):
                    rec = {"kind": kind, "model": m, "x": float(x), "budget": int(b), "value": float(surf["value"][i, j])}
                    if kind == "difference":
                        rec["noninferior"] = bool(surf["noninferior"][i, j])
                    records.append(rec)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "regime_surface.csv")
    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)

    nrows = 2 if diffs else 1
    fig, axes = S.figure("double", nrows=nrows, ncols=len(models),
                         layout=S.LAYOUT_ENGINE, squeeze=False)
    y_label = S.knob_x_label(knob, _knob_axis(df, knob))
    top = max(float(np.nanmax(s["value"])) for s in surfaces.values())
    for ax, m in zip(axes[0], models):
        im = _draw_surface(ax, surfaces[m], cmap=S.SEQUENTIAL_CMAP, vmin=0.0, vmax=top)
        ax.set_title(S.model_label(m), fontsize=S.FONT_SIZES["title"])
        ax.set_xlabel(S.axis_label("budget"))
        ax.set_ylabel(y_label)
    fig.colorbar(im, ax=list(axes[0]), label=S.axis_label("recommended_regret"))
    if diffs:
        span = max(float(np.nanmax(np.abs(s["value"]))) for s in diffs.values()) or 1.0
        for ax in axes[1]:
            ax.set_visible(False)
        for ax, m in zip(axes[1], others):
            ax.set_visible(True)
            im = _draw_surface(ax, diffs[m], cmap=S.DIVERGING_CMAP, vmin=-span, vmax=span)
            ax.set_title(f"{S.model_label(m)} - {S.model_label(reference)}", fontsize=S.FONT_SIZES["title"])
            ax.set_xlabel(S.axis_label("budget"))
            ax.set_ylabel(y_label)
        fig.colorbar(im, ax=list(axes[1]), label=S.axis_label("regret_difference"))
    fig.suptitle(
        f"{S.DATASET_LABELS.get(dataset, dataset)} - {S.KNOB_LABELS.get(knob, knob)} x K6 budget",
        x=0.01, ha="left", fontsize=S.FONT_SIZES["title"],
    )
    return [csv_path, *S.save_figure(fig, out_dir, "regime_heatmap")]


def plot_bridge(
    synthetic_frame: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    margin: float,
    model: str,
    reference: str = REFERENCE_MODEL,
) -> list[str]:
    """Real channels placed on the synthetic regime surface (roadmap S10).

    The background is the Demo 1 difference surface ``model - reference`` (non-inferior
    cells outlined); each real channel is a point at its nominal achieved SNR and the
    budget its Demo 2 run used, coloured by dataset. It answers where real
    neurostimulation sits relative to the synthetic breakdown boundary.

    Args:
        synthetic_frame: Trace frame of the Demo 1 sweep.
        synthetic_df: Tidy frame of the Demo 1 sweep (x-axis must be achieved SNR).
        real_df: Tidy frame of the matching Demo 2 sweep; its nominal-level rows supply
            each channel's achieved SNR and budget.
        out_dir: Destination directory.
        knob: Knob name of the Demo 1 sweep.
        margin: Non-inferiority margin.
        model: Model of the difference surface.
        reference: Reference model.

    Returns:
        Paths written.
    """
    surf = regime_surface(synthetic_frame, synthetic_df, knob=knob, model=model, reference=reference, margin=margin)
    from ..data.stress import KNOB_REGISTRY  # noqa: PLC0415

    nominal = real_df[np.isclose(real_df["level"].astype(float), KNOB_REGISTRY[knob].nominal_level)]
    points = nominal.groupby(["dataset", "subject", "emg"], as_index=False).agg(
        snr=("achieved_snr_db", "mean"), budget=("budget", "first")
    )
    # Map real coordinates onto the image grid (cells are indexed, not metric, on both axes).
    xs = np.interp(points["budget"], surf["budgets"], np.arange(surf["budgets"].size))
    order = np.argsort(surf["x"])
    ys = np.interp(points["snr"], surf["x"][order], np.arange(surf["x"].size)[order])

    fig, ax = S.figure("onehalf", layout=S.LAYOUT_ENGINE)
    span = float(np.nanmax(np.abs(surf["value"]))) or 1.0
    im = _draw_surface(ax, surf, cmap=S.DIVERGING_CMAP, vmin=-span, vmax=span)
    for ds in points["dataset"].unique():
        on = (points["dataset"] == ds).to_numpy()
        ax.plot(xs[on], ys[on], linestyle="none", marker="o", markersize=S.MARKER_SIZE + 1, markerfacecolor="white",
                color=S.NEUTRAL_GREY, label=S.DATASET_LABELS.get(str(ds), str(ds)))
    ax.set_xlabel(S.axis_label("budget"))
    ax.set_ylabel(S.axis_label("achieved_snr_db"))
    fig.colorbar(im, ax=ax, label=S.axis_label("regret_difference"))
    ax.legend(frameon=False, fontsize=S.FONT_SIZES["legend"])
    fig.suptitle(f"Real channels on the synthetic surface: {S.model_label(model)} - {S.model_label(reference)}",
                 x=0.01, ha="left", fontsize=S.FONT_SIZES["title"])
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bridge_points.csv")
    points.to_csv(csv_path, index=False)
    return [csv_path, *S.save_figure(fig, out_dir, "real_on_synthetic_surface")]


def plot_generator_validation(features: pd.DataFrame, out_dir: str) -> list[str]:
    """Real vs nominal-synthetic map meta-features (roadmap S0 validation).

    Args:
        features: One row per channel and demo, columns ``demo`` plus the keys of
            :func:`~pfns4neurostim.data.synthetic_neurostim.meta_features`.
        out_dir: Destination directory.

    Returns:
        Paths written (figure and ``generator_validation.csv``).
    """
    names = ["morans_i", "skewness", "cv", "mean_sd_corr"]
    fig, axes = S.figure("double", ncols=len(names), layout=S.LAYOUT_ENGINE)
    demos = [d for d in ("demo2", "demo1") if d in set(features["demo"])]
    for ax, name in zip(axes, names):
        data = [features.loc[features["demo"] == d, name].dropna().to_numpy() for d in demos]
        parts = ax.violinplot(data, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(S.NEUTRAL_GREY)
        ax.set_xticks(range(1, len(demos) + 1), [S.DEMO_LABELS[d] for d in demos])
        ax.set_ylabel(S.axis_label(name))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "generator_validation.csv")
    features.to_csv(csv_path, index=False)
    return [csv_path, *S.save_figure(fig, out_dir, "generator_validation")]


def render_all(
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
    margin: float = 0.05,
    min_snr_db: float | None = DEFAULT_MIN_SNR_DB,
) -> list[str]:
    """Produce every stress deliverable from a tidy sweep frame.

    Args:
        df: Tidy sweep frame (from ``tidy.csv``).
        out_dir: Destination directory.
        knob: Knob name.
        dataset: Dataset name.
        margin: Pre-registered TOST margin for the robustness table.
        min_snr_db: Cells (channel x level) with an achieved SNR below this many dB are left out of every
            figure and table; the channels kept per level are annotated and written to
            ``channel_counts.csv``. ``None`` keeps every cell. ``tidy.csv`` is never modified.

    Returns:
        Every path written, figures and tables.
    """
    df = _shared_ladder(df)
    df, counts = clip_low_snr(df, min_snr_db)
    os.makedirs(out_dir, exist_ok=True)
    counts_path = os.path.join(out_dir, CHANNEL_COUNTS_FILE)
    counts.to_csv(counts_path, index=False)
    print(f"[stress figures] channels kept per level (min SNR {min_snr_db} dB):\n{counts.to_string(index=False)}")
    if df.empty:
        warnings.warn(f"SNR clip (>= {min_snr_db} dB) left no level with enough channels; no figure was drawn.",
                      stacklevel=2)
        return [counts_path]
    table, table_path = build_robustness_table(df, out_dir, knob=knob, margin=margin)
    breakdowns = {
        str(r["model"]): float(r["breakdown_x"])
        for _, r in table.iterrows()
        if np.isfinite(r.get("breakdown_x", np.nan))
    }
    written = [table_path, counts_path]
    written += plot_degradation_curves(
        df, out_dir, knob=knob, dataset=dataset, breakdowns=breakdowns, counts=counts
    )
    written += plot_outcome_panels(df, out_dir, knob=knob, dataset=dataset, counts=counts)
    frame = T.load_trace_frame(out_dir)
    if frame is not None and not frame.empty and knob != "k6_budget":
        frame = _restrict_to_cells(_trace_frame_on_ladder(frame, df), df)
        bd = breakdown_vs_budget(frame, df, knob=knob, margin=margin)
        written += plot_breakdown_vs_budget(bd, df, out_dir, knob=knob, dataset=dataset)
        if knob in REGIME_KNOBS:
            written += plot_regime_heatmap(frame, df, out_dir, knob=knob, dataset=dataset, margin=margin)
    return written
