"""Hypothesis B stress-regime figures and tables (roadmap S2, S9).

Everything here reads the sweep's ``tidy.csv`` and nothing else, so every figure
can be regenerated without re-running an experiment (``--replot``). All geometry,
colours, labels and axis wording come from :mod:`pfns4neurostim.visualization.style`.

Deliverables:
    ``degradation_{demo}.svg``  recommended regret, exploration score and R-squared vs the knob axis,
                               per model, with 95% CI bands, faint per-channel lines and
                               breakdown points marked.
    ``robustness.csv``          breakdown point, degradation AUC, CVaR-10%,
                                relative robustness, per model.
    ``robustness_vs_budget.{csv,svg}``  E10 (2026-10-21): breakdown point (with its bootstrap CI), the model's
                                own median recommended regret and CVaR-10% as functions of the BO budget B,
                                read off ``recommended_regret_per_step`` at the S5 budget grid (no extra
                                compute). Breakdown is relative to GP-MLL, so the figure always draws the
                                model's own regret next to it -- the confound that retired
                                ``breakdown_vs_budget.svg`` (below) is shown, not hidden.
    ``calibration_under_stress.{csv,svg}``  E12 (2026-10-21): coverage-50 / coverage-90, ECE and NLL vs the
                                knob axis, median over channels with an IQR band. Refuses a frame that mixes
                                ``online_y_scaler`` (or ``normalization``) settings: calibration is never
                                compared across them. The target is the noise-free ground-truth mean map
                                (``bo_runner`` scores against ``y_end``) while the predictive is of noisy
                                observations, so its intervals carry the noise variance and cover the GT
                                mean *more* often than nominal -- increasingly so as the SNR falls.

Retired 2026-09-27:
    ``outcomes_{demo}.svg``     merged into ``degradation_{demo}.svg``. Both drew recommended regret and
                                R-squared against the same axis for the same run; ``outcomes`` added only
                                the exploration-score panel, which is now the middle panel of
                                ``degradation``, and ``degradation`` keeps the breakdown markers and the
                                per-channel traces that ``outcomes`` lacked. No information was lost.
    ``breakdown_vs_budget.svg`` dropped. It re-evaluated the breakdown rule at budget fractions, but the
                                rule is defined *relative to GP-MLL*, so the curve mixed "robust under
                                stress" with "ahead of the reference at this budget" and could not be read
                                as either. The budget axis is covered by ``regime_heatmap.svg``, which
                                reports regret directly.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..evaluation.results import DEMO_NAMES, canonical_demo, canonical_demo_column
from ..evaluation.robustness import (
    noninferiority,
    bootstrap_breakdown,
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
    "build_robustness_table",
    "budget_grid",
    "regret_at_budget",
    "robustness_vs_budget_table",
    "plot_robustness_vs_budget",
    "calibration_under_stress_table",
    "plot_calibration_under_stress",
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

#: Tail fraction of every CVaR in this module (CVaR-10%, roadmap S9).
CVAR_ALPHA: float = 0.1


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


def _level_severity_ascending(df: pd.DataFrame | None = None) -> bool:
    """Whether a larger knob *level* means more stress (the order in which breakdown walks the ladder).

    This is about the level, not the plotted axis: K2 is plotted against achieved SNR, which falls with
    stress, but its level (alpha) rises with stress. Passing the axis direction here made the walk start at
    the harshest level and report a breakdown there for every model. Every knob's level gets more severe as
    it increases, except a calibrated ladder, which is in dB of SNR degradation, where a more negative level
    is more severe.

    Args:
        df: The sweep frame after :func:`_shared_ladder`, to detect a calibrated ladder.

    Returns:
        True when larger levels are more severe.
    """
    if df is not None and _is_calibrated(df):
        return False
    return True


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


def _nominal_x(df: pd.DataFrame, knob: str, x_col: str) -> float | None:
    """Return the x position of the dataset's default (unstressed) regime, or ``None``.

    Resolved from the tidy frame alone, so ``--replot`` draws the marker without rebuilding the knob. The
    level is the knob's own nominal level (see
    :meth:`~pfns4neurostim.data.stress.StressKnob.nominal_in`); its x is that level's mean x, which for the
    K2 arms is the channel-averaged floor SNR of the dataset and is therefore the *same number* in every
    figure of that dataset.

    A **calibrated** ladder (``knob.targets_db``) is the one case where the nominal level is not a constant
    across channels: each channel solves its own level for the same dB target, so the unstressed cell is the
    0 dB target rather than any particular level value.

    Args:
        df: Tidy sweep frame (already on the shared ladder).
        knob: Knob name.
        x_col: Plotted x column.

    Returns:
        The x position, or ``None`` when the run never visits the unstressed regime or the knob is unknown.
    """
    from ..data.stress import KNOB_REGISTRY  # noqa: PLC0415 - avoids a data -> visualization import cycle

    if x_col == CALIBRATED_TARGET_COLUMN:
        targets = df[x_col].astype(float)
        return 0.0 if bool(np.isclose(targets, 0.0).any()) else None
    cls = KNOB_REGISTRY.get(knob)
    if cls is None:
        return None
    level = cls.nominal_in(sorted(df["level"].astype(float).unique()))
    if level is None:
        return None
    x = _level_to_x(df, x_col).get(float(level))
    return x if x is not None and np.isfinite(x) else None


def _nominal_snr_db(df: pd.DataFrame, knob: str) -> float | None:
    """Return the channel-averaged achieved SNR of the default regime, in dB.

    This is the number that has to agree across every figure of a dataset: the unstressed channel is the same
    object whichever knob is being swept, so its achieved SNR is a property of the dataset (NHP 3.98 dB,
    spinal 6.30 dB, ``5d_rat`` 2.06 dB; the fitted synthetic twins match to 0.15 dB by construction).

    Args:
        df: Tidy sweep frame.
        knob: Knob name.

    Returns:
        Mean achieved SNR over the channels of the nominal level, or ``None`` when the frame carries no SNR
        column or never visits the unstressed regime.
    """
    if "achieved_snr_db" not in df.columns:
        return None
    from ..data.stress import KNOB_REGISTRY  # noqa: PLC0415 - avoids a data -> visualization import cycle

    cls = KNOB_REGISTRY.get(knob)
    if cls is None:
        return None
    level = cls.nominal_in(sorted(df["level"].astype(float).unique()))
    if level is None:
        return None
    cells = (
        df[np.isclose(df["level"].astype(float), level)]
        .groupby(["subject", "emg"], dropna=False)["achieved_snr_db"]
        .mean()
    )
    return float(cells.mean()) if not cells.empty and np.isfinite(cells.mean()) else None


def _nominal_label(df: pd.DataFrame, knob: str, x_col: str, nominal_x: float | None) -> str:
    """Legend entry of the default-regime marker on a knob x-axis.

    On an SNR x-axis the marker's position already states the default SNR; on a level axis (K5, K6) it does
    not, so the legend entry carries it, which is what makes the default regime comparable between a K2
    figure and a K5 one.

    Args:
        df: Tidy sweep frame.
        knob: Knob name.
        x_col: Plotted x column.
        nominal_x: Output of :func:`_nominal_x`.

    Returns:
        :data:`style.NOMINAL_MARKER_LABEL`, with the default SNR appended on a level axis.
    """
    nominal_snr = _nominal_snr_db(df, knob)
    if nominal_x is not None and nominal_snr is not None and x_col not in X_DESCENDS_WITH_STRESS:
        return f"{S.NOMINAL_MARKER_LABEL}, {nominal_snr:.1f} dB"
    return S.NOMINAL_MARKER_LABEL


def _demo_of(df: pd.DataFrame) -> str:
    """The run's arm (``draws`` / ``synthetic``) in its current spelling, ``draws`` when unrecorded."""
    return canonical_demo(str(df["demo"].iloc[0])) if "demo" in df.columns else "draws"


def _figure_title(df: pd.DataFrame, dataset: str, knob: str, suffix: str = "") -> str:
    """Suptitle of a per-run stress figure: dataset - knob - arm, plus an optional suffix.

    Args:
        df: Tidy sweep frame (supplies the arm).
        dataset: Dataset name.
        knob: Knob name.
        suffix: Appended after a ``' - '`` separator when non-empty.

    Returns:
        The title string, every part taken from :mod:`style`'s label tables.
    """
    demo = _demo_of(df)
    title = (f"{S.DATASET_LABELS.get(dataset, dataset)} - {S.KNOB_LABELS.get(knob, knob)} - "
             f"{S.DEMO_LABELS.get(demo, demo)}")
    return f"{title} - {suffix}" if suffix else title


def _shared_legend(fig, axes: Sequence, extra: Sequence = ()) -> None:
    """One legend below a row of panels, each label once: models first (in :data:`style.MODEL_ORDER`), then
    the other entries in the order they were first drawn, then ``extra`` proxy handles.

    Args:
        fig: Figure to put the legend on.
        axes: Panels whose labelled artists are collected.
        extra: Additional proxy artists (with labels) appended at the end.
    """
    by_label: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            by_label.setdefault(label, handle)
    for handle in extra:
        by_label.setdefault(handle.get_label(), handle)
    model_labels = [S.model_label(m) for m in S.MODEL_ORDER]
    ordered = [lb for lb in model_labels if lb in by_label] + [lb for lb in by_label if lb not in model_labels]
    fig.legend([by_label[lb] for lb in ordered], ordered, loc="outside lower center", ncol=len(ordered),
               frameon=False, fontsize=S.FONT_SIZES["legend"])


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
    metrics: Sequence[str] = ("recommended_regret", "exploration_score", "r2"),
    breakdowns: dict[str, float] | None = None,
    show_channels: bool = True,
    counts: pd.DataFrame | None = None,
) -> list[str]:
    """Degradation curves: one panel per metric, one line per model.

    Bounded metrics show the mean over channels with a 95% CI; heavy-tailed ones (R^2) the median with an
    interquartile band (see :func:`_model_band`), and the R^2 axis is clipped for display only.

    Absorbed the retired ``outcomes_{demo}.svg`` on 2026-09-27: ``exploration_score`` is now a default
    panel, so this is the single per-knob outcome figure. A metric absent from ``df`` is skipped.

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
    nominal_x = _nominal_x(df, knob, x_col)
    nominal_label = _nominal_label(df, knob, x_col, nominal_x)
    # ``demo`` is resolved to its current spelling here, not in the cell: a run assembled from cells cached
    # before the 2026-09-27 rename carries ``demo2``/``demo1``, and titling such a figure by the raw value made
    # every pre-rename in-vivo run call itself "Synthetic" (fixed 2026-10-13).
    demo = canonical_demo(str(df["demo"].iloc[0])) if "demo" in df.columns else "draws"
    models = [m for m in S.MODEL_ORDER if m in set(df["model"])]
    # A knob that does not record one of the default metrics simply loses that panel.
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return []

    # One panel per metric side by side (a vertical stack of 1.5-aspect panels would be too tall).
    fig, axes = S.figure("wide", nrows=1, ncols=len(metrics), sharex=True, layout=S.LAYOUT_ENGINE)
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

        # The default regime of the dataset, on every panel; only the first panel carries the legend entry.
        if nominal_x is not None:
            S.mark_nominal(ax, nominal_x, label=nominal_label if idx == 0 else None)

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
    if demo not in DEMO_NAMES:
        raise ValueError(
            f"plot_degradation_curves: unknown demo {demo!r}; expected one of {DEMO_NAMES}. A new arm needs a "
            f"filename of its own -- the pre-2026-10-13 code folded anything unrecognised into 'synthetic', "
            f"which is how legacy 'demo2' in-vivo runs came to be labelled synthetic."
        )
    return S.save_figure(fig, out_dir, f"degradation_{demo}")


def _breakdown_with_ci(
    sub: pd.DataFrame,
    ref_sub: pd.DataFrame,
    *,
    metric: str,
    margin: float,
    level_ascending: bool,
    level_x: dict[float, float],
) -> dict[str, Any]:
    """Pooled breakdown point of one model against the reference, with a bootstrap CI over channels.

    The pooled breakdown pairs every (channel, repetition) of the model with the reference's at each level
    (:func:`breakdown_point` with :data:`BREAKDOWN_TEST`). The CI is a CI **of that pooled point**
    (2026-10-21; :func:`~pfns4neurostim.evaluation.robustness.bootstrap_breakdown`): channels are resampled
    with replacement, the pooled breakdown is recomputed with the same rule on each resample, and the
    percentile interval of the resampled ladder positions -- "no breakdown" counted as one step beyond the
    harshest level -- is extended if needed to bracket the point. It replaced the CI of the *mean of
    per-channel breakdowns*, which estimated a different quantity and often excluded the plotted point
    (NHP K2 GP-fixed: point 24.0 dB, old CI 12.4-21.6 dB). Shared by the robustness table (S9) and the
    robustness-vs-budget table (E10), so both apply one rule.

    Args:
        sub: Rows of the model (one per channel x level x repetition).
        ref_sub: Rows of the reference model, same layout.
        metric: Lower-is-better metric column.
        margin: Pre-registered non-inferiority margin, in the metric's units.
        level_ascending: Whether a larger level is more severe (:func:`_level_severity_ascending`).
        level_x: Level -> plotted-axis value (:func:`_level_to_x`).

    Returns:
        ``breakdown_level``, ``breakdown_x``, ``breakdown_reason``, ``equivalent_levels``;
        ``breakdown_ci_lo`` / ``breakdown_ci_hi`` (numeric min / max in plotted-axis units; an end at "no
        breakdown" is written as the harshest level's x), ``breakdown_ci_mild_level`` /
        ``breakdown_ci_harsh_level`` (the two ends on the level ladder, NaN = no breakdown),
        ``breakdown_ci_beyond_ladder`` (the harsh end is "no breakdown"), ``breakdown_boot_none_frac``
        (share of resamples with no breakdown), ``breakdown_ci_extended`` (the percentile interval had to be
        widened to contain the point), and the per-channel counts ``n_channels_broken`` / ``n_channels``.

    Raises:
        ValueError: If the bootstrap's own pooled point disagrees with :func:`breakdown_point`'s (the model
            and reference rows are not paired channel for channel), since the CI would then not be of the
            plotted point.
    """
    bp = breakdown_point(
        _per_level_values(sub, metric),
        _per_level_values(ref_sub, metric),
        margin=margin,
        severity_ascending=level_ascending,
        test=BREAKDOWN_TEST,
    )
    out: dict[str, Any] = {
        "breakdown_level": bp["breakdown_level"],
        "breakdown_x": level_x.get(bp["breakdown_level"], float("nan")),
        "breakdown_reason": bp["breakdown_reason"],
        "equivalent_levels": int(np.sum(bp["equivalent"])),
    }
    ref_by_channel = dict(tuple(ref_sub.groupby(["subject", "emg"], dropna=False)))
    model_channels: list[dict[float, np.ndarray]] = []
    ref_channels: list[dict[float, np.ndarray]] = []
    per_channel_bd: list[float] = []
    for key, grp in sub.groupby(["subject", "emg"], dropna=False):
        ref_grp = ref_by_channel.get(key)
        if ref_grp is None or ref_grp.empty:
            continue
        model_channels.append(_per_level_values(grp, metric))
        ref_channels.append(_per_level_values(ref_grp, metric))
        # Per-channel breakdowns are kept as a descriptive count (how many channels break down on their own).
        try:
            chan_bp = breakdown_point(model_channels[-1], ref_channels[-1], margin=margin, test=BREAKDOWN_TEST,
                                      severity_ascending=level_ascending)
        except ValueError:
            continue
        per_channel_bd.append(chan_bp["breakdown_level"])

    nan = float("nan")
    out.update({"breakdown_ci_lo": nan, "breakdown_ci_hi": nan, "breakdown_ci_mild_level": nan,
                "breakdown_ci_harsh_level": nan, "breakdown_ci_beyond_ladder": False,
                "breakdown_boot_none_frac": nan, "breakdown_ci_extended": False})
    if model_channels and BREAKDOWN_TEST == "noninferiority":
        boot = bootstrap_breakdown(model_channels, ref_channels, margin=margin, severity_ascending=level_ascending)
        ladder = boot["levels"]                                                    # severity order, [L]
        point = ladder[boot["point_rank"]] if boot["point_rank"] < len(ladder) else nan
        if not (point == bp["breakdown_level"] or (np.isnan(point) and np.isnan(bp["breakdown_level"]))):
            raise ValueError(
                f"_breakdown_with_ci: the channel-paired pooled breakdown ({point}) differs from the "
                f"position-paired one ({bp['breakdown_level']}); the model and reference rows are not paired "
                "channel for channel, so a channel bootstrap would not be a CI of the plotted point."
            )
        mild_r, harsh_r = boot["ci_rank"]
        mild = ladder[mild_r] if mild_r < len(ladder) else nan
        harsh = ladder[harsh_r] if harsh_r < len(ladder) else nan
        # "No breakdown" sits beyond the harshest level; in x units it is drawn at the harshest level's x.
        ends_x = [level_x.get(ladder[min(r, len(ladder) - 1)], nan) for r in (mild_r, harsh_r)]
        out.update({
            "breakdown_ci_lo": float(np.nanmin(ends_x)) if np.isfinite(ends_x).any() else nan,
            "breakdown_ci_hi": float(np.nanmax(ends_x)) if np.isfinite(ends_x).any() else nan,
            "breakdown_ci_mild_level": mild,
            "breakdown_ci_harsh_level": harsh,
            "breakdown_ci_beyond_ladder": bool(harsh_r >= len(ladder)),
            "breakdown_boot_none_frac": boot["none_fraction"],
            "breakdown_ci_extended": bool(boot["extended"]),
        })
    out["n_channels_broken"] = int(np.sum(np.isfinite(per_channel_bd)))
    out["n_channels"] = int(len(per_channel_bd))
    return out


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
    level_ascending = _level_severity_ascending(df)
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
        rec["cvar10_regret"] = cvar(sub[metric].to_numpy(dtype=float), alpha=CVAR_ALPHA)
        rec["mean_at_nominal"] = float(mean[-1] if not ascending else mean[0])

        if model != reference and not ref_sub.empty:
            rec.update(_breakdown_with_ci(sub, ref_sub, metric=metric, margin=margin,
                                          level_ascending=level_ascending, level_x=level_x))
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


#: Knobs that get the S5 knob x BO-budget regime heatmap.
REGIME_KNOBS: frozenset[str] = frozenset({"k2_channel", "k2_global"})

#: Number of BO-budget columns of a regime surface (S5/S10), spread evenly from the first acquisition to the full budget.
#: E10's robustness-vs-budget readout uses the same grid, so its budgets are the heatmap's columns.
REGIME_BUDGET_COLUMNS: int = 8

#: Trace field a budget-B readout is taken from: ``[B - n_init]`` is the recommendation after B observations.
BUDGET_TRACE_FIELD: str = "recommended_regret_per_step"

#: Tidy column a budget read-off stands in for (the final recommendation of a budget-B run).
BUDGET_METRIC: str = "recommended_regret"


def budget_grid(n_init: int, total_budget: int, n_budgets: int = REGIME_BUDGET_COLUMNS) -> np.ndarray:
    """BO budgets at which the per-step traces are read (S5 regime surface, E10 robustness vs budget).

    ``n_budgets`` values spread evenly from the first acquisition (``n_init + 1``) to the full budget, rounded
    to whole queries; duplicates after rounding collapse, so a short run can return fewer.

    Args:
        n_init: Initial-design size of the run.
        total_budget: Total queries of the run, including ``n_init``.
        n_budgets: Number of budgets requested.

    Returns:
        Sorted unique integer budgets, shape [B] with B <= ``n_budgets``.

    Raises:
        ValueError: If ``n_budgets < 1`` or the run has no acquisition after its initial design.
    """
    if n_budgets < 1:
        raise ValueError(f"budget_grid: n_budgets must be >= 1, got {n_budgets}.")
    if total_budget <= n_init:
        raise ValueError(f"budget_grid: total budget {total_budget} leaves no acquisition after n_init={n_init}.")
    return np.unique(np.linspace(n_init + 1, total_budget, n_budgets).round().astype(int))   # [B]


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
    the K2 x BO-budget surface (roadmap S5) costs no extra compute. Each cell is the median
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
    field = BUDGET_TRACE_FIELD
    frame = frame[frame[field].notna()]
    needed = {model} | ({reference} if reference else set())
    missing = needed - set(frame["model"])
    if missing:
        raise ValueError(f"regime_surface: no traces for {sorted(missing)}.")
    if reference is not None and margin is None:
        raise ValueError("regime_surface: a difference surface needs the non-inferiority margin.")
    n_init = int(frame["n_init"].iloc[0])
    total = int(frame[field].iloc[0].shape[0]) - 1 + n_init
    budgets = budget_grid(n_init, total, n_budgets)                                        # [B]
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


def _nominal_row(surf: dict[str, np.ndarray], nominal_x: float | None) -> float | None:
    """Return the ``imshow`` row index of the default regime on a regime surface.

    The heatmap's rows are categorical (cell ``i`` is centred at ``y = i``), so the marker needs the row
    index rather than the x value the row stands for.

    Args:
        surf: Output of :func:`regime_surface`; ``surf['x']`` is the x value of each row, in row order.
        nominal_x: X value of the default regime, or ``None``.

    Returns:
        The row index as a float, or ``None`` when the surface has no row at that x.
    """
    if nominal_x is None:
        return None
    hits = np.flatnonzero(np.isclose(np.asarray(surf["x"], dtype=float), float(nominal_x)))
    return float(hits[0]) if hits.size else None


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
    """K2 x BO-budget interaction surface (roadmap S5), one panel per model plus one difference.

    Top row: median recommended regret per (level, budget) for every model. Bottom row: a **single**
    centred panel, the paired difference of the primary PFN against ``reference``, diverging around zero,
    with the non-inferior cells outlined. Negative (blue) means the PFN has less regret than the reference.

    Changed 2026-09-27: the bottom row used to carry one difference panel per non-reference model, which
    added a GP-fixed-minus-GP-MLL surface. That compared two GP variants to each other and answered no
    question the hypothesis asks, so only the PFN-vs-reference contrast is drawn; it keeps the same panel
    geometry as the top row and sits under the middle column.

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
    # Exactly one difference surface: the primary PFN against the reference. A GP-fixed-vs-GP-MLL panel
    # contrasts two GP variants, which is not a question any hypothesis asks (changed 2026-09-27).
    contrast = [m for m in models if m != reference and S.MODEL_STYLES[m].family == "pfn"][:1] \
        if reference in models else []
    surfaces = {m: regime_surface(frame, df, knob=knob, model=m) for m in models}
    diffs = {m: regime_surface(frame, df, knob=knob, model=m, reference=reference, margin=margin)
             for m in contrast}

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
    fig, axes = S.figure("wide", nrows=nrows, ncols=len(models),
                         layout=S.LAYOUT_ENGINE, squeeze=False)
    x_col = _knob_axis(df, knob)
    y_label = S.knob_x_label(knob, x_col)
    # The stress ladder runs up the y-axis here, so the default regime is a horizontal slice at its row.
    nominal_row = _nominal_row(next(iter(surfaces.values())), _nominal_x(df, knob, x_col))
    top = max(float(np.nanmax(s["value"])) for s in surfaces.values())
    for idx, (ax, m) in enumerate(zip(axes[0], models)):
        im = _draw_surface(ax, surfaces[m], cmap=S.SEQUENTIAL_CMAP, vmin=0.0, vmax=top)
        if nominal_row is not None:
            S.mark_nominal(ax, nominal_row, axis="y", label=S.NOMINAL_MARKER_LABEL if idx == 0 else None)
        ax.set_title(S.model_label(m), fontsize=S.FONT_SIZES["title"])
        ax.set_xlabel(S.axis_label("budget"))
        ax.set_ylabel(y_label)
    fig.colorbar(im, ax=list(axes[0]), label=S.axis_label("recommended_regret"))
    if diffs:
        span = max(float(np.nanmax(np.abs(s["value"]))) for s in diffs.values()) or 1.0
        for ax in axes[1]:
            ax.set_visible(False)
        # Centre the single difference panel under the middle column, keeping the top row's geometry.
        centre = len(models) // 2
        for ax, m in zip([axes[1][centre]], contrast):
            ax.set_visible(True)
            im = _draw_surface(ax, diffs[m], cmap=S.DIVERGING_CMAP, vmin=-span, vmax=span)
            if nominal_row is not None:
                S.mark_nominal(ax, nominal_row, axis="y", label=None)
            ax.set_title(f"{S.model_label(m)} - {S.model_label(reference)}", fontsize=S.FONT_SIZES["title"])
            ax.set_xlabel(S.axis_label("budget"))
            ax.set_ylabel(y_label)
        fig.colorbar(im, ax=[axes[1][centre]], label=S.axis_label("regret_difference"))
    fig.suptitle(
        f"{S.DATASET_LABELS.get(dataset, dataset)} - {S.KNOB_LABELS.get(knob, knob)} x BO budget",
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

    The background is the synthetic difference surface ``model - reference`` (non-inferior
    cells outlined); each real channel is a point at its nominal achieved SNR and the
    budget its draws run used, coloured by dataset. It answers where real
    neurostimulation sits relative to the synthetic breakdown boundary.

    Args:
        synthetic_frame: Trace frame of the synthetic sweep.
        synthetic_df: Tidy frame of the synthetic sweep (x-axis must be achieved SNR).
        real_df: Tidy frame of the matching draws sweep; its nominal-level rows supply
            each channel's achieved SNR and budget.
        out_dir: Destination directory.
        knob: Knob name of the synthetic sweep.
        margin: Non-inferiority margin.
        model: Model of the difference surface.
        reference: Reference model.

    Returns:
        Paths written.
    """
    surf = regime_surface(synthetic_frame, synthetic_df, knob=knob, model=model, reference=reference, margin=margin)
    from ..data.stress import KNOB_REGISTRY  # noqa: PLC0415

    # ``nominal_in`` rather than the class attribute, so a knob without a constant nominal level resolves too.
    nominal_level = KNOB_REGISTRY[knob].nominal_in(sorted(real_df["level"].astype(float).unique()))
    if nominal_level is None:
        raise ValueError(
            f"plot_bridge: the draws sweep has no unstressed level for knob {knob!r}, so its channels have "
            "no nominal achieved SNR to be placed at."
        )
    nominal = real_df[np.isclose(real_df["level"].astype(float), nominal_level)]
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
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "generator_validation.csv")
    features.to_csv(csv_path, index=False)
    demos = [d for d in ("draws", "synthetic") if d in set(features["demo"])]
    if not demos:   # a run with neither a draws nor a synthetic demo: no real-vs-synthetic comparison to draw
        return [csv_path]
    fig, axes = S.figure("double", ncols=len(names), layout=S.LAYOUT_ENGINE)
    for ax, name in zip(axes, names):
        data = {d: features.loc[features["demo"] == d, name].dropna().to_numpy() for d in demos}
        shown = [d for d in demos if data[d].size]      # a demo with no finite value has no violin
        if shown:
            parts = ax.violinplot([data[d] for d in shown], showmedians=True)
            for body in parts["bodies"]:
                body.set_facecolor(S.NEUTRAL_GREY)
        ax.set_xticks(range(1, len(shown) + 1), [S.DEMO_LABELS[d] for d in shown])
        missing = [S.DEMO_LABELS[d] for d in demos if d not in shown]
        if missing:
            ax.text(0.5, 0.98, f"no finite values: {', '.join(missing)}", transform=ax.transAxes, ha="center",
                    va="top", fontsize=S.FONT_SIZES["annotation"])
        ax.set_ylabel(S.axis_label(name))
    return [csv_path, *S.save_figure(fig, out_dir, "generator_validation")]


# ---------------------------------------------------------------------------
# E10 - robustness vs BO budget (2026-10-21)
# ---------------------------------------------------------------------------
#: Key columns a budget read-off keeps from the trace frame (those present).
BUDGET_KEY_COLUMNS: tuple[str, ...] = ("demo", "knob", "dataset", "model", "level", "subject", "emg", "rep")

#: Horizontal offset between models' markers in the E10 breakdown panel, as a fraction of the budget span, so
#: two CIs drawn at the same budget do not hide one another.
BUDGET_DODGE_FRACTION: float = 0.012

#: Written next to the figure.
ROBUSTNESS_VS_BUDGET_FILE: str = "robustness_vs_budget"

#: The two own-regret panels of the E10 figure: (summary column, axis-label key).
BUDGET_REGRET_PANELS: tuple[str, ...] = ("recommended_regret_median_level_mean", "cvar10_level_mean")


def _trace_budget_axis(frame: pd.DataFrame, field: str = BUDGET_TRACE_FIELD) -> tuple[int, int]:
    """Initial-design size and total budget shared by every repetition of a trace frame.

    Args:
        frame: Trace frame (:func:`traces.load_trace_frame`).
        field: Per-step trace column.

    Returns:
        ``(n_init, total_budget)``; trace index ``B - n_init`` is the recommendation after ``B`` observations.

    Raises:
        ValueError: If no repetition recorded ``field``, or the repetitions disagree on ``n_init`` or on the
            trace length (a mixed-budget run has no shared budget axis).
    """
    rows = frame[frame[field].notna()]
    if rows.empty:
        raise ValueError(f"no repetition recorded {field!r}.")
    n_inits = set(rows["n_init"].astype(int))
    lengths = {int(np.asarray(a).shape[0]) for a in rows[field]}
    if len(n_inits) != 1 or len(lengths) != 1:
        raise ValueError(
            f"the repetitions disagree on the budget axis (n_init {sorted(n_inits)}, {field!r} lengths "
            f"{sorted(lengths)}); a budget read-off needs one shared run length."
        )
    n_init = n_inits.pop()
    return n_init, lengths.pop() - 1 + n_init


def regret_at_budget(frame: pd.DataFrame, budget: int, *, field: str = BUDGET_TRACE_FIELD) -> pd.DataFrame:
    """Recommended regret every repetition would have reported as a budget-``budget`` run.

    The S5 read-off: ``trace[budget - n_init]`` is the recommendation after ``budget`` observations, which a
    shorter run with the same seed reports as its final ``recommended_regret``
    (``tests/integration/test_stress_synthetic_splithalf.py::TestBudgetFromTraces``).

    Args:
        frame: Trace frame, one row per BO repetition.
        budget: Total queries including ``n_init``.
        field: Per-step trace column.

    Returns:
        One row per repetition: the key columns of :data:`BUDGET_KEY_COLUMNS` present in ``frame``, the
        read-off value as :data:`BUDGET_METRIC` and ``budget``.

    Raises:
        ValueError: If ``budget`` lies outside ``[n_init, total_budget]`` or a read-off value is not finite.
    """
    rows = frame[frame[field].notna()]
    n_init, total = _trace_budget_axis(rows, field)
    if not n_init <= int(budget) <= total:
        raise ValueError(f"regret_at_budget: budget {budget} outside the recorded run [{n_init}, {total}].")
    stack = np.vstack([np.asarray(a, dtype=np.float64) for a in rows[field]])   # [R, total - n_init + 1]
    values = stack[:, int(budget) - n_init]                                     # [R]
    bad = ~np.isfinite(values)
    if bad.any():
        raise ValueError(
            f"regret_at_budget: {int(bad.sum())} of {values.size} repetitions have a non-finite {field!r} at "
            f"budget {budget}."
        )
    keys = [c for c in BUDGET_KEY_COLUMNS if c in rows.columns]
    return rows[keys].reset_index(drop=True).assign(**{BUDGET_METRIC: values, "budget": int(budget)})


def robustness_vs_budget_table(
    frame: pd.DataFrame,
    df: pd.DataFrame,
    *,
    knob: str,
    margin: float,
    reference: str = REFERENCE_MODEL,
    n_budgets: int = REGIME_BUDGET_COLUMNS,
    budgets: Sequence[int] | None = None,
) -> pd.DataFrame:
    """E10: breakdown point, own regret and CVaR-10% of each model as functions of the BO budget.

    At every budget ``B`` of the grid the traces are read off as if each run had stopped after ``B``
    observations (:func:`regret_at_budget`), and the robustness table's rules are applied to that frame:

    * **breakdown point** -- :func:`_breakdown_with_ci` against ``reference`` (the S9 rule, non-inferiority
      at ``margin``), with the bootstrap CI over channels. It is *relative to the reference at the same
      budget*, so it is reported next to the model's own regret and never alone;
    * **own regret** -- per level, the median over channels of the repetition-mean recommended regret (the
      value of the S5 regime-surface cell);
    * **CVaR-10%** -- per level, :func:`cvar` over every (channel, repetition) value at that level.

    Each per-level value is also summarised per (model, B) by the mean over levels (every level weighted
    equally) and, for CVaR, the maximum over levels.

    Args:
        frame: Trace frame on the sweep's shared ladder, restricted to its cells.
        df: Tidy sweep frame (level -> plotted-axis mapping, severity direction).
        knob: Knob name.
        margin: Pre-registered non-inferiority margin (range-normalized regret).
        reference: Reference model of the breakdown rule.
        n_budgets: Size of the budget grid (:func:`budget_grid`, the S5 heatmap's columns).
        budgets: Explicit budgets, overriding the grid.

    Returns:
        One row per (budget, model, level) with the per-level values, the per-(model, budget) summaries and
        the breakdown fields repeated on every level row of that (model, budget). ``breakdown_reason`` is
        ``'reference'`` for the reference itself and ``'no_reference'`` when the run lacks it.
    """
    x_col = _knob_axis(df, knob)
    level_x = _level_to_x(df, x_col)
    level_ascending = _level_severity_ascending(df)
    n_init, total = _trace_budget_axis(frame)
    grid = budget_grid(n_init, total, n_budgets) if budgets is None else np.unique(np.asarray(budgets, dtype=int))
    models = [m for m in S.MODEL_ORDER if m in set(frame["model"])]
    nan = float("nan")

    records: list[dict[str, Any]] = []
    for b in grid:
        at_b = regret_at_budget(frame, int(b))
        ref_sub = at_b[at_b["model"] == reference]
        for model in models:
            sub = at_b[at_b["model"] == model]
            if sub.empty:
                continue
            if model != reference and not ref_sub.empty:
                bd = _breakdown_with_ci(sub, ref_sub, metric=BUDGET_METRIC, margin=margin,
                                        level_ascending=level_ascending, level_x=level_x)
            else:
                bd = {"breakdown_level": nan, "breakdown_x": nan,
                      "breakdown_reason": "reference" if model == reference else "no_reference",
                      "equivalent_levels": nan, "breakdown_ci_lo": nan, "breakdown_ci_hi": nan,
                      "breakdown_ci_mild_level": nan, "breakdown_ci_harsh_level": nan,
                      "breakdown_ci_beyond_ladder": False, "breakdown_boot_none_frac": nan,
                      "breakdown_ci_extended": False, "n_channels_broken": nan, "n_channels": nan}
            bd["n_channels_tested"] = bd.pop("n_channels")

            per_level: list[dict[str, Any]] = []
            for level, grp in sub.groupby("level", dropna=False):
                chan = grp.groupby(["subject", "emg"], dropna=False)[BUDGET_METRIC].mean()     # [n_channels]
                per_level.append({
                    "level": float(level),
                    "x": level_x.get(float(level), nan),
                    "recommended_regret_median": float(chan.median()),
                    "cvar10_recommended_regret": cvar(grp[BUDGET_METRIC].to_numpy(dtype=float), alpha=CVAR_ALPHA),
                    "n_channels_at_level": int(chan.size),
                })
            medians = np.array([r["recommended_regret_median"] for r in per_level])           # [L]
            tails = np.array([r["cvar10_recommended_regret"] for r in per_level])             # [L]
            summary = {
                "recommended_regret_median_level_mean": float(np.mean(medians)),
                "cvar10_level_mean": float(np.mean(tails)),
                "cvar10_level_max": float(np.max(tails)),
            }
            for rec in sorted(per_level, key=lambda r: r["x"]):
                records.append({"knob": knob, "x_axis": x_col, "model": model, "model_label": S.model_label(model),
                                "budget": int(b), "n_init": n_init, **rec, **summary, **bd,
                                "reference": reference, "margin": float(margin)})
    return pd.DataFrame.from_records(records)


def plot_robustness_vs_budget(
    frame: pd.DataFrame,
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
    margin: float,
    reference: str = REFERENCE_MODEL,
    n_budgets: int = REGIME_BUDGET_COLUMNS,
    budgets: Sequence[int] | None = None,
    formats: Sequence[str] = ("svg",),
) -> list[str]:
    """E10 figure and table: robustness vs BO budget, one line per model.

    Panels: (a) breakdown point vs B with its 95% CI over channels, an open marker at the harshest level
    when the model never breaks down at that budget, and the default regime as a horizontal slice; (b) the
    model's own median recommended regret vs B; (c) CVaR-10% vs B. (b) and (c) are each the mean over the
    ladder's levels and include the reference, so a breakdown that moves with B can be read as "the model
    got better / worse" or "the reference did" -- the confound that retired ``breakdown_vs_budget.svg``.

    Args:
        frame: Trace frame on the sweep's shared ladder, restricted to its cells.
        df: Tidy sweep frame.
        out_dir: Destination directory.
        knob: Knob name.
        dataset: Dataset name, for the title.
        margin: Pre-registered non-inferiority margin.
        reference: Reference model of the breakdown rule.
        n_budgets: Size of the budget grid.
        budgets: Explicit budgets, overriding the grid.
        formats: Figure formats (SVG by default; PNG for previews).

    Returns:
        Paths written (``robustness_vs_budget.csv`` and the figure).
    """
    from matplotlib.lines import Line2D  # noqa: PLC0415 - figure-time import

    table = robustness_vs_budget_table(frame, df, knob=knob, margin=margin, reference=reference,
                                       n_budgets=n_budgets, budgets=budgets)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{ROBUSTNESS_VS_BUDGET_FILE}.csv")
    table.to_csv(csv_path, index=False)
    if table.empty:
        return [csv_path]

    summary = table.drop_duplicates(["model", "budget"]).sort_values("budget")
    x_col = _knob_axis(df, knob)
    descending = x_col in X_DESCENDS_WITH_STRESS
    ladder_x = [v for v in _level_to_x(df, x_col).values() if np.isfinite(v)]
    harshest = min(ladder_x) if descending else max(ladder_x)
    grid = np.sort(summary["budget"].unique())                                               # [B]
    dodge = BUDGET_DODGE_FRACTION * float(grid[-1] - grid[0]) if grid.size > 1 else 0.0
    models = [m for m in S.MODEL_ORDER if m in set(summary["model"])]
    tested = [m for m in models
              if not summary.loc[summary["model"] == m, "breakdown_reason"].isin(["reference", "no_reference"]).all()]

    fig, axes = S.figure("wide", nrows=1, ncols=1 + len(BUDGET_REGRET_PANELS), sharex=True, layout=S.LAYOUT_ENGINE)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    any_open = False
    for i, model in enumerate(tested):
        s = summary[summary["model"] == model]
        bx = s["budget"].to_numpy(dtype=float) + dodge * (i - (len(tested) - 1) / 2)          # [B]
        ax.plot(bx, s["breakdown_x"].to_numpy(dtype=float), **S.plot_kwargs(model))
        lo = s["breakdown_ci_lo"].to_numpy(dtype=float)                                        # [B]
        hi = s["breakdown_ci_hi"].to_numpy(dtype=float)                                        # [B]
        has_ci = np.isfinite(lo) & np.isfinite(hi)
        if has_ci.any():
            ax.vlines(bx[has_ci], lo[has_ci], hi[has_ci], color=S.model_color(model), linewidth=S.LINE_WIDTH,
                      zorder=S.model_style(model).zorder)
        never = (s["breakdown_reason"] == "none").to_numpy()
        if never.any():
            any_open = True
            ax.plot(bx[never], np.full(int(never.sum()), harshest),
                    **S.plot_kwargs(model, linestyle="none", markerfacecolor="none", label=None))
    nominal_x = _nominal_x(df, knob, x_col)
    if nominal_x is not None:
        S.mark_nominal(ax, nominal_x, axis="y", label=_nominal_label(df, knob, x_col, nominal_x))
    if descending:
        # Achieved SNR falls with stress: invert so a later breakdown (more robust) sits higher, as on a level axis.
        ax.invert_yaxis()
    ax.set_ylabel(f"{S.axis_label('breakdown_point')}\n{S.knob_x_label(knob, x_col).replace(chr(10), ' ')}")
    ax.set_title(S.BREAKDOWN_RULE_NOTE.format(reference=S.model_label(reference), margin=margin),
                 fontsize=S.FONT_SIZES["annotation"])

    for ax, column in zip(axes[1:], BUDGET_REGRET_PANELS):
        for model in models:
            s = summary[summary["model"] == model]
            ax.plot(s["budget"].to_numpy(dtype=float), s[column].to_numpy(dtype=float), **S.plot_kwargs(model))
        ax.set_ylabel(S.axis_label(column))

    for ax in axes:
        ax.set_xlabel(S.axis_label("budget"))
        ax.set_xticks(grid, [str(int(b)) for b in grid])
    extra = [Line2D([], [], linestyle="none", marker="o", markerfacecolor="none", color=S.OUTLINE_COLOR,
                    label=S.NO_BREAKDOWN_LABEL)] if any_open else []
    _shared_legend(fig, axes, extra)
    fig.suptitle(_figure_title(df, dataset, knob, "robustness vs BO budget"), x=0.01, ha="left",
                 fontsize=S.FONT_SIZES["title"])
    S.panel_letters(axes, x=0.0, y=1.02)
    return [csv_path, *S.save_figure(fig, out_dir, ROBUSTNESS_VS_BUDGET_FILE, formats=formats)]


# ---------------------------------------------------------------------------
# E12 - calibration under stress (2026-10-21)
# ---------------------------------------------------------------------------
#: Calibration metrics of the E12 figure, one panel each, in reading order.
CALIBRATION_METRICS: tuple[str, ...] = ("coverage_50", "coverage_90", "ece", "nll")

#: Value a perfectly calibrated predictive scores. NLL has none: its optimum depends on the channel's noise.
CALIBRATION_TARGETS: dict[str, float] = {"coverage_50": 0.5, "coverage_90": 0.9, "ece": 0.0}

#: Columns that fix the units / scaling of the calibration metrics; a calibration frame holds one value of each.
CALIBRATION_SCALING_COLUMNS: tuple[str, ...] = ("online_y_scaler", "normalization")

#: Quantiles of the interquartile band over channels.
IQR_QUANTILES: tuple[float, float] = (0.25, 0.75)

#: Written next to the figure.
CALIBRATION_UNDER_STRESS_FILE: str = "calibration_under_stress"

#: What the tidy calibration columns are scored against: the noise-free ground-truth mean map (``bo_runner``
#: calls ``calibration_metrics(y_end[alive], ...)``), not held-out noisy trials. Written into the CSV and named
#: in the figure, since a noisy-observation predictive over-covers that target by construction.
CALIBRATION_TARGET: str = "ground_truth_mean"


def _single_scaling(df: pd.DataFrame) -> dict[str, str]:
    """The one ``online_y_scaler`` / ``normalization`` setting of a frame, refusing a mixture.

    Calibration (NLL, coverage, ECE, CRPS) is never compared across ``online_y_scaler`` settings, for any
    model (CLAUDE.md section 4): the scaler moves the predictive SD and the NLL's units. Pooling two settings
    into one curve would do exactly that comparison silently.

    Args:
        df: Tidy sweep frame.

    Returns:
        Column -> its single value, for the columns of :data:`CALIBRATION_SCALING_COLUMNS` present.

    Raises:
        ValueError: If any of those columns holds more than one value (NaN counts as a value).
    """
    settings: dict[str, str] = {}
    for column in CALIBRATION_SCALING_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.unique(df[column])
        if len(values) > 1:
            raise ValueError(
                f"calibration under stress: the frame mixes {column} settings {sorted(map(str, values))}. "
                f"Calibration (coverage, ECE, NLL, CRPS) is never compared across {column} settings; split the "
                f"frame by {column} and plot each setting on its own."
            )
        settings[column] = str(values[0])
    return settings


def calibration_under_stress_table(
    df: pd.DataFrame,
    *,
    knob: str,
    metrics: Sequence[str] = CALIBRATION_METRICS,
) -> pd.DataFrame:
    """E12: calibration metrics vs the knob axis, median and IQR over channels, per model.

    Repetitions are reduced within a channel first, then the median and the interquartile range are taken
    over channels, so a subject with many EMGs cannot dominate and one badly calibrated channel cannot drag
    the curve. Bounded metrics (coverage, ECE) average the repetitions; NLL, which is in
    :data:`HEAVY_TAILED_METRICS`, takes their median, as every heavy-tailed metric of this module does: a
    single degenerate GP-MLL repetition reaches NLL ~ 2e8 on NHP at the mildest K2 level (measured
    2026-10-21), which a mean carries into the channel value and the IQR. A model whose metric is NaN on
    every row (no predictive distribution) is left out of that metric.

    Args:
        df: Tidy sweep frame, one ``online_y_scaler`` / ``normalization`` setting.
        knob: Knob name (selects the x-axis: achieved SNR for K2, the level for K5 / K6).
        metrics: Calibration columns; those absent from ``df`` are skipped.

    Returns:
        One row per (metric, model, level): ``x``, ``median``, ``q25``, ``q75``, ``n_channels``,
        ``rep_reduction`` (how repetitions were reduced within a channel), ``calibration_target``
        (:data:`CALIBRATION_TARGET`) and the scaling setting the values were computed under.

    Raises:
        ValueError: If the frame mixes scaling settings, or a model has some but not all values non-finite.
    """
    settings = _single_scaling(df)
    x_col = _knob_axis(df, knob)
    models = [m for m in S.MODEL_ORDER if m in set(df["model"])]
    records: list[dict[str, Any]] = []
    for metric in [m for m in metrics if m in df.columns]:
        for model in models:
            sub = df[df["model"] == model]
            values = sub[metric].to_numpy(dtype=float)                                         # [n_rows]
            if np.isnan(values).all():
                continue
            if not np.isfinite(values).all():
                raise ValueError(
                    f"calibration under stress: {model} has {int((~np.isfinite(values)).sum())} non-finite "
                    f"{metric!r} values of {values.size}."
                )
            reduction = "median" if _is_heavy_tailed(metric) else "mean"
            per_channel = (
                sub.groupby(["level", "subject", "emg"], dropna=False)
                .agg(x=(x_col, "mean"), y=(metric, reduction))
                .reset_index()
            )
            rows = [{
                "knob": knob, "x_axis": x_col, "metric": metric, "model": model,
                "model_label": S.model_label(model), "level": float(level), "x": float(grp["x"].mean()),
                "median": float(grp["y"].median()), "q25": float(grp["y"].quantile(IQR_QUANTILES[0])),
                "q75": float(grp["y"].quantile(IQR_QUANTILES[1])), "n_channels": int(len(grp)),
                "rep_reduction": reduction, "calibration_target": CALIBRATION_TARGET, **settings,
            } for level, grp in per_channel.groupby("level", dropna=False)]
            records += sorted(rows, key=lambda r: r["x"])
    return pd.DataFrame.from_records(records)


def plot_calibration_under_stress(
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
    metrics: Sequence[str] = CALIBRATION_METRICS,
    formats: Sequence[str] = ("svg",),
) -> list[str]:
    """E12 figure and table: coverage-50 / coverage-90, ECE and NLL vs the knob axis, one line per model.

    Each panel shows the median over channels with an IQR band, a thin guide at the perfectly calibrated
    value (:data:`CALIBRATION_TARGETS`; none for NLL) and the default-regime marker. The title states the
    ``online_y_scaler`` setting, the only one the figure may be compared within.

    Args:
        df: Tidy sweep frame, one scaling setting.
        out_dir: Destination directory.
        knob: Knob name.
        dataset: Dataset name, for the title.
        metrics: Calibration columns, one panel each.
        formats: Figure formats (SVG by default; PNG for previews).

    Returns:
        Paths written (``calibration_under_stress.csv`` and the figure).

    Raises:
        ValueError: If the frame mixes ``online_y_scaler`` or ``normalization`` settings.
    """
    settings = _single_scaling(df)
    table = calibration_under_stress_table(df, knob=knob, metrics=metrics)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{CALIBRATION_UNDER_STRESS_FILE}.csv")
    table.to_csv(csv_path, index=False)
    shown = [m for m in metrics if not table.empty and m in set(table["metric"])]
    if not shown:
        return [csv_path]

    x_col = _knob_axis(df, knob)
    nominal_x = _nominal_x(df, knob, x_col)
    models = [m for m in S.MODEL_ORDER if m in set(table["model"])]
    fig, axes = S.figure("wide", nrows=1, ncols=len(shown), sharex=True, layout=S.LAYOUT_ENGINE)
    axes = np.atleast_1d(axes)
    target_labelled = False
    for idx, (ax, metric) in enumerate(zip(axes, shown)):
        if metric in CALIBRATION_TARGETS:
            ax.axhline(CALIBRATION_TARGETS[metric], color=S.CALIBRATION_TARGET_COLOR,
                       linestyle=S.CALIBRATION_TARGET_LINESTYLE, linewidth=S.CALIBRATION_TARGET_LINEWIDTH,
                       zorder=0, label=None if target_labelled else S.CALIBRATION_TARGET_LABEL)
            target_labelled = True
        for model in models:
            s = table[(table["metric"] == metric) & (table["model"] == model)].sort_values("x")
            if s.empty:
                continue
            x = s["x"].to_numpy(dtype=float)                                                    # [L]
            ax.plot(x, s["median"].to_numpy(dtype=float), **S.plot_kwargs(model))
            ax.fill_between(x, s["q25"].to_numpy(dtype=float), s["q75"].to_numpy(dtype=float),
                            color=S.model_color(model), alpha=S.BAND_ALPHA, linewidth=0)
        if nominal_x is not None:
            S.mark_nominal(ax, nominal_x, label=_nominal_label(df, knob, x_col, nominal_x) if idx == 0 else None)
        ax.set_ylabel(S.axis_label(f"{metric}_gt_median"))
        ax.set_xlabel(S.knob_x_label(knob, x_col))
    if x_col in X_DESCENDS_WITH_STRESS:
        axes[0].invert_xaxis()   # shared x: "more stress to the right" on every panel (roadmap S2)
    _shared_legend(fig, axes)
    scaler = settings.get("online_y_scaler")
    suffix = (f"calibration under stress, {S.CALIBRATION_TARGET_LABELS[CALIBRATION_TARGET]}"
              + (f" ({S.ONLINE_Y_SCALER_LABELS.get(scaler, scaler)})" if scaler else ""))
    fig.suptitle(_figure_title(df, dataset, knob, suffix), x=0.01, ha="left", fontsize=S.FONT_SIZES["title"])
    S.panel_letters(axes, x=0.0, y=1.02)
    return [csv_path, *S.save_figure(fig, out_dir, CALIBRATION_UNDER_STRESS_FILE, formats=formats)]


def render_all(
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
    margin: float = 0.05,
    min_snr_db: float | None = DEFAULT_MIN_SNR_DB,
    n_budgets: int = REGIME_BUDGET_COLUMNS,
) -> list[str]:
    """Produce every stress deliverable from a tidy sweep frame.

    The trace-based deliverables (S5 regime heatmap for K2, E10 robustness vs budget for every knob) are
    drawn when the run directory holds ``trajectories.pkl``; E12 calibration under stress when the frame
    carries the calibration columns.

    Args:
        df: Tidy sweep frame (from ``tidy.csv``).
        out_dir: Destination directory.
        knob: Knob name.
        dataset: Dataset name.
        margin: Pre-registered TOST margin for the robustness table.
        min_snr_db: Cells (channel x level) with an achieved SNR below this many dB are left out of every
            figure and table; the channels kept per level are annotated and written to
            ``channel_counts.csv``. ``None`` keeps every cell. ``tidy.csv`` is never modified.
        n_budgets: Size of the E10 budget grid (:func:`budget_grid`).

    Returns:
        Every path written, figures and tables.

    Raises:
        ValueError: If the frame mixes ``online_y_scaler`` / ``normalization`` settings (E12 refuses to pool
            calibration across them); every other deliverable is written first.
    """
    df = _shared_ladder(df)
    if "demo" in df.columns:
        # Once, for every deliverable below: legacy cells keep their recorded spelling on disk, readers see the
        # current one (:func:`pfns4neurostim.evaluation.results.canonical_demo`).
        df = df.assign(demo=canonical_demo_column(df["demo"]))
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
    frame = T.load_trace_frame(out_dir)
    if frame is not None and not frame.empty:
        frame = _restrict_to_cells(_trace_frame_on_ladder(frame, df), df)
    has_traces = frame is not None and not frame.empty
    if has_traces and knob in REGIME_KNOBS:
        written += plot_regime_heatmap(frame, df, out_dir, knob=knob, dataset=dataset, margin=margin)
    if has_traces and BUDGET_TRACE_FIELD in frame.columns and frame[BUDGET_TRACE_FIELD].notna().any():
        written += plot_robustness_vs_budget(frame, df, out_dir, knob=knob, dataset=dataset, margin=margin,
                                             n_budgets=n_budgets)
    if any(m in df.columns for m in CALIBRATION_METRICS):
        written += plot_calibration_under_stress(df, out_dir, knob=knob, dataset=dataset)
    return written
