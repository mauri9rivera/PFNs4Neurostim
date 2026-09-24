"""Figures for the split-half ground-truth sensitivity check (task #7 Step 6).

One identity scatter per metric: x = the channel's value under the full-mean GT,
y = the same channel under the split-half GT, one point per (channel, model). Points
above the diagonal for regret (below for R²) mean the full-mean GT flattered the model,
i.e. the leakage bias between observed trials and the GT mean. All style comes from
:mod:`pfns4neurostim.visualization.style`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from . import style

__all__ = ["plot_gt_identity"]


def plot_gt_identity(
    paired: pd.DataFrame,
    out_dir: str,
    *,
    metrics: Sequence[str] = ("recommended_regret", "r2"),
    name: str = "gt_identity",
) -> list[str]:
    """Identity scatter of full-mean vs split-half per-channel metrics.

    Args:
        paired: Frame with ``model`` and, for each metric, ``<metric>_full_mean`` and
            ``<metric>_split_half`` columns (one row per channel x model x acquisition).
        out_dir: Destination directory.
        metrics: Metrics to draw, one panel each.
        name: Output basename.

    Returns:
        Written paths.
    """
    metrics = [m for m in metrics if f"{m}_full_mean" in paired and f"{m}_split_half" in paired]
    if not metrics:
        return []
    fig, axes = style.figure("onehalf", ncols=len(metrics), aspect=style.SINGLE_PANEL_ASPECT)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        x_col, y_col = f"{metric}_full_mean", f"{metric}_split_half"
        for model in sorted(paired["model"].unique()):
            sub = paired[paired["model"] == model]
            st = style.model_style(model)
            ax.scatter(
                sub[x_col], sub[y_col], s=style.MARKER_SIZE ** 2, color=st.color,
                marker=st.marker, label=st.label, linewidths=0, zorder=st.zorder,
            )
        both = pd.concat([paired[x_col], paired[y_col]]).dropna()
        if not both.empty:
            lo, hi = float(both.min()), float(both.max())
            ax.plot([lo, hi], [lo, hi], color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH, ls="--", zorder=0)
        ax.set_xlabel(f"{style.axis_label(metric)}\n{style.GT_MODE_LABELS['full_mean']}")
        ax.set_ylabel(f"{style.axis_label(metric)}\n{style.GT_MODE_LABELS['split_half']}")
    axes[0].legend(loc="best")
    style.panel_letters(axes)
    return style.save_figure(fig, out_dir, name)
