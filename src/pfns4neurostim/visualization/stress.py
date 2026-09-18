"""Hypothesis B stress-regime figures and tables (roadmap S2, S9).

Everything here reads the sweep's ``tidy.csv`` and nothing else, so every figure
can be regenerated without re-running an experiment (``--replot``). All geometry,
colours, labels and axis wording come from :mod:`pfns4neurostim.visualization.style`.

Deliverables:
    ``degradation_{demo}.svg``  regret and R-squared vs the knob axis, per model,
                               with 95% CI bands, faint per-channel lines and
                               breakdown points marked.
    ``calibration_{demo}.svg``  90% coverage and ECE vs the knob axis.
    ``robustness.csv``          breakdown point, degradation AUC, CVaR-10%,
                                relative robustness, per model.
    ``robustness_forest.svg``   breakdown points with bootstrap CIs.
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

__all__ = [
    "plot_degradation_curves",
    "plot_calibration_row",
    "build_robustness_table",
    "plot_robustness_forest",
    "render_all",
]

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


def _severity_ascending(x_col: str) -> bool:
    """Whether a larger x means more stress.

    Achieved SNR *decreases* with stress, so severity is descending in x.

    Args:
        x_col: The x-axis column name.

    Returns:
        True when larger x means more severe stress.
    """
    return x_col != "achieved_snr_db"


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
    return {
        float(level): grp[y_col].to_numpy(dtype=float)
        for level, grp in sub.groupby("level", dropna=False)
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
    if x_col == "achieved_snr_db":
        axes[0].invert_xaxis()
    axes[-1].set_xlabel(S.axis_label(x_col))
    S.panel_letters(axes, x=-0.13)
    fig.tight_layout()
    return S.save_figure(fig, out_dir, f"degradation_{'invivo' if demo == 'demo2' else 'synthetic'}")


def plot_calibration_row(
    df: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
    dataset: str,
) -> list[str]:
    """Calibration panels: 90% interval coverage and ECE vs the knob axis.

    A horizontal reference line marks nominal 90% coverage; a well-calibrated
    model tracks it as stress rises, an overconfident one falls below it.

    Args:
        df: Tidy sweep frame.
        out_dir: Destination directory.
        knob: Knob name.
        dataset: Dataset name.

    Returns:
        Paths written (empty when the sweep carries no calibration columns).
    """
    if not {"coverage_90", "ece"} <= set(df.columns) or df["coverage_90"].isna().all():
        return []

    x_col = _knob_axis(df, knob)
    demo = str(df["demo"].iloc[0]) if "demo" in df.columns else "demo2"
    models = [m for m in S.MODEL_ORDER if m in set(df["model"])]

    fig, axes = S.figure("onehalf", nrows=1, ncols=2, aspect=1.0)
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, ("coverage_90", "ece")):
        for model in models:
            sub = df[df["model"] == model]
            if sub.empty:
                continue
            x, mean, ci = _model_curve(sub, x_col, metric)
            ax.plot(x, mean, **S.plot_kwargs(model))
            ax.fill_between(
                x, mean - ci, mean + ci, color=S.model_color(model), alpha=S.BAND_ALPHA, linewidth=0
            )
        if metric == "coverage_90":
            ax.axhline(0.9, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
            ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(S.axis_label(metric))
        ax.set_xlabel(S.axis_label(x_col))
        if x_col == "achieved_snr_db":
            ax.invert_xaxis()

    axes[0].legend(loc="best")
    axes[0].set_title(f"{S.DATASET_LABELS.get(dataset, dataset)} - calibration", loc="left")
    S.panel_letters(axes, x=-0.22)
    fig.tight_layout()
    return S.save_figure(fig, out_dir, f"calibration_{'invivo' if demo == 'demo2' else 'synthetic'}")


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
                severity_ascending=ascending,
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
                        severity_ascending=ascending,
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


def plot_robustness_forest(
    table: pd.DataFrame,
    out_dir: str,
    *,
    knob: str,
) -> list[str]:
    """Forest plot of breakdown points with bootstrap CIs, one row per model.

    Args:
        table: Output of :func:`build_robustness_table`.
        out_dir: Destination directory.
        knob: Knob name, for the axis label.

    Returns:
        Paths written (empty when no model has a finite breakdown point).
    """
    plottable = table[np.isfinite(table.get("breakdown_x", pd.Series(dtype=float)))]
    if plottable.empty:
        return []

    x_col = str(table["x_axis"].iloc[0])
    fig, ax = S.figure("single", aspect=0.55)
    ys = np.arange(len(plottable))
    for y, (_, row) in zip(ys, plottable.iterrows()):
        color = S.model_color(str(row["model"]))
        lo = row.get("breakdown_ci_lo", np.nan)
        hi = row.get("breakdown_ci_hi", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.2, solid_capstyle="butt")
        ax.plot(
            [row["breakdown_x"]],
            [y],
            marker=S.model_style(str(row["model"])).marker,
            color=color,
            markersize=S.MARKER_SIZE + 1,
            linestyle="none",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([S.model_label(m) for m in plottable["model"]])
    ax.set_xlabel(f"Breakdown point ({S.axis_label(x_col)})")
    ax.set_title(f"{S.KNOB_LABELS.get(knob, knob)} breakdown", loc="left")
    ax.invert_yaxis()
    fig.tight_layout()
    return S.save_figure(fig, out_dir, "robustness_forest")


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
    written += plot_calibration_row(df, out_dir, knob=knob, dataset=dataset)
    written += plot_robustness_forest(table, out_dir, knob=knob)
    return written
