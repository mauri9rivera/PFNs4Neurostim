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
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..evaluation.robustness import (
    bootstrap_ci,
    breakdown_point,
    cvar,
    degradation_auc,
    relative_robustness,
)
from . import style as S
from . import traces as T

__all__ = [
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


def plot_degradation_curves(
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
    metrics: Sequence[str] = ("recommended_regret", "r2"),
    breakdowns: dict[str, float] | None = None,
    show_channels: bool = True,
) -> list[str]:
    """Degradation curves: one row per metric, one line per model.

    Args:
        df: Tidy sweep frame.
        out_dir: Destination directory.
        knob: Knob name (selects the x-axis and its label).
        dataset: Dataset name, for the title.
        metrics: Metric columns, one panel row each.
        breakdowns: Optional model -> breakdown point **in x-axis units**
            (``breakdown_x`` of the robustness table), marked on the regret row.
        show_channels: Draw faint per-channel traces behind the model means.

    Returns:
        Paths written.
    """
    x_col = _knob_axis(df, knob)
    demo = str(df["demo"].iloc[0]) if "demo" in df.columns else "demo2"
    models = [m for m in S.MODEL_ORDER if m in set(df["model"])]

    fig, axes = S.figure("onehalf", nrows=len(metrics), ncols=1, aspect=0.62, sharex=True)
    axes = np.atleast_1d(axes)

    for row_idx, metric in enumerate(metrics):
        ax = axes[row_idx]
        for model in models:
            sub = df[df["model"] == model]
            if sub.empty or metric not in sub.columns:
                continue

            if show_channels:
                for (subj, emg), grp in sub.groupby(["subject", "emg"], dropna=False):
                    chan = grp.groupby("level", dropna=False).agg(
                        x=(x_col, "mean"), y=(metric, "mean")
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

            x, mean, ci = _model_curve(sub, x_col, metric)
            ax.plot(x, mean, **S.plot_kwargs(model))
            ax.fill_between(
                x,
                mean - ci,
                mean + ci,
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

        ax.set_ylabel(S.axis_label(metric))
        if row_idx == 0:
            ax.set_title(
                f"{S.DATASET_LABELS.get(dataset, dataset)} - "
                f"{S.KNOB_LABELS.get(knob, knob)} - {S.DEMO_LABELS.get(demo, demo)}"
            )
            ax.legend(ncol=1, loc="best")

    # Achieved SNR decreases with stress: plot it decreasing left to right so the
    # x-axis always reads "more stress to the right" (roadmap S2).
    if x_col in X_DESCENDS_WITH_STRESS:
        axes[0].invert_xaxis()
    axes[-1].set_xlabel(S.axis_label(x_col))
    S.panel_letters(axes, x=-0.13)
    fig.tight_layout()
    return S.save_figure(fig, out_dir, f"degradation_{'invivo' if demo == 'demo2' else 'synthetic'}")


def plot_outcome_panels(
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
) -> list[str]:
    """Final-run outcomes vs the knob axis: (a) simple regret, (b) exploration score, (c) R^2.

    Replaces the 90%-coverage calibration figure (2026-09-21). Lines and 95% CI bands over
    channels, one line per model; no bars.

    Args:
        df: Tidy sweep frame.
        out_dir: Destination directory.
        knob: Knob name.
        dataset: Dataset name.

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

    fig, axes = S.figure("double", nrows=1, ncols=len(panels), aspect=1.0)
    axes = np.atleast_1d(axes)
    for ax, (col, key) in zip(axes, panels):
        for model in models:
            sub = df[df["model"] == model]
            if sub.empty or sub[col].isna().all():
                continue
            x, mean, ci = _model_curve(sub, x_col, col)
            ax.plot(x, mean, **S.plot_kwargs(model))
            ax.fill_between(x, mean - ci, mean + ci, color=S.model_color(model), alpha=S.BAND_ALPHA, linewidth=0)
        ax.set_ylabel(S.axis_label(key))
        ax.set_xlabel(S.axis_label(x_col))
        if x_col in X_DESCENDS_WITH_STRESS:
            ax.invert_xaxis()
    axes[0].legend(loc="best")
    n_reps = int(df["rep"].nunique()) if "rep" in df.columns else 0
    n_chan = int(df[["subject", "emg"]].drop_duplicates().shape[0])
    fig.suptitle(
        f"{S.DATASET_LABELS.get(dataset, dataset)} - {S.KNOB_LABELS.get(knob, knob)} "
        f"(n={n_reps} reps, {n_chan} channels)",
        x=0.01, ha="left", fontsize=S.FONT_SIZES["title"],
    )
    S.panel_letters(axes, x=-0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
    fig, ax = S.figure("single", aspect=0.75)
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
    ax.set_ylabel(f"Breakdown point
({S.axis_label(x_col)})")
    ax.set_title(f"{S.DATASET_LABELS.get(dataset, dataset)} - {S.KNOB_LABELS.get(knob, knob)} breakdown", loc="left")
    ax.legend(loc="best")
    fig.tight_layout()
    return S.save_figure(fig, out_dir, "breakdown_vs_budget")


def plot_regime_heatmap(*args: Any, **kwargs: Any) -> list[str]:
    """K2 x K6 interaction surface (roadmap S5). **Not implemented.**

    Raises:
        NotImplementedError: Lands with task #10 Step 9, after the K6 knobs.
    """
    raise NotImplementedError("plot_regime_heatmap lands with task #10 Step 9 (roadmap S5).")


def render_all(
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
    margin: float = 0.05,
) -> list[str]:
    """Produce every stress deliverable from a tidy sweep frame.

    Args:
        df: Tidy sweep frame (from ``tidy.csv``).
        out_dir: Destination directory.
        knob: Knob name.
        dataset: Dataset name.
        margin: Pre-registered TOST margin for the robustness table.

    Returns:
        Every path written, figures and tables.
    """
    df = _shared_ladder(df)
    table, table_path = build_robustness_table(df, out_dir, knob=knob, margin=margin)
    breakdowns = {
        str(r["model"]): float(r["breakdown_x"])
        for _, r in table.iterrows()
        if np.isfinite(r.get("breakdown_x", np.nan))
    }
    written = [table_path]
    written += plot_degradation_curves(
        df, out_dir, knob=knob, dataset=dataset, breakdowns=breakdowns
    )
    written += plot_outcome_panels(df, out_dir, knob=knob, dataset=dataset)
    frame = T.load_trace_frame(out_dir)
    if frame is not None and not frame.empty and knob != "k6_budget":
        bd = breakdown_vs_budget(_trace_frame_on_ladder(frame, df), df, knob=knob, margin=margin)
        written += plot_breakdown_vs_budget(bd, df, out_dir, knob=knob, dataset=dataset)
    return written
