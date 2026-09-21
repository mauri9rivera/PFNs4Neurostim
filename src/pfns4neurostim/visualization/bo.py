"""Figures for the BO benchmark (Hyp 0 / Hyp A / Hyp 0 acquisition table / PFN benchmark).

Reads the run's ``tidy.csv`` and ``trajectories.pkl`` and nothing else, so everything regenerates
with ``--replot``. Style comes entirely from :mod:`pfns4neurostim.visualization.style`.

Deliverables (2026-09-21: curves along the BO run, no bar plots):
    ``trajectories.svg``       (a) simple regret, (b) exploration score, (c) R^2 vs BO iteration,
                               one line per model (x acquisition), mean over repetitions and channels
    ``latency.svg``            per-step latency vs BO iteration, same averaging
    ``comparison_table.csv``   final values, regret AUC and latency per model x acquisition
"""
from __future__ import annotations

import os

import pandas as pd

from . import traces as T

__all__ = ["render_all"]


def render_all(df: pd.DataFrame, out_dir: str, *, dataset: str) -> list[str]:
    """Produce every benchmark figure from a run directory.

    Args:
        df: Tidy benchmark frame (kept for the interface shared with the stress figures).
        out_dir: Run directory holding ``trajectories.pkl``; figures are written here.
        dataset: Dataset name.

    Returns:
        Every path written (empty when the run has no trajectories).
    """
    frame = T.load_trace_frame(out_dir)
    if frame is None or frame.empty:
        return []
    table, table_path = T.comparison_table(frame, out_dir)
    written = [table_path]
    written += T.plot_trace_panels(frame, out_dir, dataset=dataset, name="trajectories")
    written += T.plot_latency_curves(frame, out_dir, dataset=dataset, name="latency")
    return written
