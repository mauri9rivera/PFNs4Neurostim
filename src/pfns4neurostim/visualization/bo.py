"""Figures for the BO benchmark (Hyp 0 / Hyp A).

Reads the benchmark's ``tidy.csv`` and nothing else, so everything regenerates
with ``--replot``. Style comes entirely from
:mod:`pfns4neurostim.visualization.style`.

Deliverables:
    ``acquisition_table.svg``  the three co-primary regrets per model x acquisition
                               (Hyp 0: does acquisition choice matter less for PFNs?)
    ``latency.svg``            per-query latency per model x acquisition
                               (Hyp A's "cheaper" claim, same acquisition on both sides)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from . import style as S

__all__ = ["plot_acquisition_table", "plot_latency", "render_all"]

#: The three co-primary regret outcomes (user decision 2026-09-16).
CO_PRIMARY: tuple[str, ...] = (
    "recommended_regret",
    "best_queried_regret",
    "cumulative_regret",
)


def _channel_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average repetitions within a channel, then summarize across channels.

    Args:
        df: Tidy benchmark frame.
        metric: Metric column.

    Returns:
        Frame with ``model``, ``acq_label``, ``mean`` and ``ci`` columns.
    """
    per_channel = (
        df.groupby(["model", "acq_label", "subject", "emg"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    grouped = per_channel.groupby(["model", "acq_label"], dropna=False)[metric]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ci"] = 1.96 * out["std"] / np.sqrt(out["count"])
    out["ci"] = out["ci"].fillna(0.0)
    return out


def plot_acquisition_table(
    df: pd.DataFrame,
    out_dir: str,
    *,
    dataset: str,
    metrics: Sequence[str] = CO_PRIMARY,
) -> list[str]:
    """Grouped bars: one panel per co-primary regret, x = acquisition, colour = model.

    Reading the figure: a PFN whose bars are flat across acquisitions supports the
    Hyp 0 claim that it depends less on acquisition choice than the GP does.

    Args:
        df: Tidy benchmark frame.
        out_dir: Destination directory.
        dataset: Dataset name, for the title.
        metrics: Metric columns, one panel each.

    Returns:
        Paths written.
    """
    metrics = [m for m in metrics if m in df.columns]
    models = [m for m in S.MODEL_ORDER if m in set(df["model"])]
    acqs = sorted(set(df["acq_label"]))

    fig, axes = S.figure("double", nrows=1, ncols=len(metrics), aspect=1.25)
    axes = np.atleast_1d(axes)

    width = 0.8 / max(len(models), 1)
    for ax, metric in zip(axes, metrics):
        summary = _channel_means(df, metric)
        for m_idx, model in enumerate(models):
            sub = summary[summary["model"] == model].set_index("acq_label")
            xs, heights, errs = [], [], []
            for a_idx, acq in enumerate(acqs):
                if acq not in sub.index:
                    continue
                xs.append(a_idx + (m_idx - (len(models) - 1) / 2) * width)
                heights.append(float(sub.loc[acq, "mean"]))
                errs.append(float(sub.loc[acq, "ci"]))
            if not xs:
                continue
            ax.bar(
                xs,
                heights,
                width=width * 0.92,
                yerr=errs,
                color=S.model_color(model),
                label=S.model_label(model),
                error_kw={"linewidth": 0.7, "capsize": 1.5},
            )
        ax.set_xticks(range(len(acqs)))
        ax.set_xticklabels(acqs, rotation=30, ha="right")
        ax.set_ylabel(S.axis_label(metric))

    axes[0].legend(loc="best")
    axes[0].set_title(f"{S.DATASET_LABELS.get(dataset, dataset)} - acquisition comparison", loc="left")
    S.panel_letters(axes, x=-0.22)
    fig.tight_layout()
    return S.save_figure(fig, out_dir, "acquisition_table")


def plot_latency(df: pd.DataFrame, out_dir: str, *, dataset: str) -> list[str]:
    """Per-query latency per model and acquisition (the Hyp A cost axis).

    The comparison is only meaningful within an acquisition type — a GP running
    EI against a PFN running EI — which is why acquisition is the x-axis here too.

    Args:
        df: Tidy benchmark frame.
        out_dir: Destination directory.
        dataset: Dataset name.

    Returns:
        Paths written (empty when no latency was recorded).
    """
    if "mean_query_latency_s" not in df.columns or df["mean_query_latency_s"].isna().all():
        return []

    models = [m for m in S.MODEL_ORDER if m in set(df["model"])]
    acqs = sorted(set(df["acq_label"]))
    summary = _channel_means(df, "mean_query_latency_s")

    fig, ax = S.figure("single", aspect=0.75)
    width = 0.8 / max(len(models), 1)
    for m_idx, model in enumerate(models):
        sub = summary[summary["model"] == model].set_index("acq_label")
        xs, heights = [], []
        for a_idx, acq in enumerate(acqs):
            if acq not in sub.index:
                continue
            xs.append(a_idx + (m_idx - (len(models) - 1) / 2) * width)
            heights.append(float(sub.loc[acq, "mean"]))
        if xs:
            ax.bar(xs, heights, width=width * 0.92, color=S.model_color(model), label=S.model_label(model))
    ax.set_xticks(range(len(acqs)))
    ax.set_xticklabels(acqs, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(S.axis_label("mean_query_latency_s"))
    ax.set_title(f"{S.DATASET_LABELS.get(dataset, dataset)} - per-query cost", loc="left")
    ax.legend(loc="best")
    fig.tight_layout()
    return S.save_figure(fig, out_dir, "latency")


def render_all(df: pd.DataFrame, out_dir: str, *, dataset: str) -> list[str]:
    """Produce every benchmark figure from a tidy frame.

    Args:
        df: Tidy benchmark frame.
        out_dir: Destination directory.
        dataset: Dataset name.

    Returns:
        Every path written.
    """
    written = plot_acquisition_table(df, out_dir, dataset=dataset)
    written += plot_latency(df, out_dir, dataset=dataset)
    return written
