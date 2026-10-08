"""Cross-run pooling: one long table from every run's ``tidy.csv`` (task plan A8).

Every assembled run directory holds a ``tidy.csv`` (one row per cell, :mod:`.results` schema) and the
``config.yaml`` that produced it. Pooling concatenates those tables and stamps each row with where it came
from -- ``run_dir`` (relative to the output root, POSIX separators), ``family`` and ``tag`` from the config,
and ``online_y_scaler`` -- so cross-dataset analyses (the A6 TOST, the summary tables) read one file and
can never mix runs silently: calibration metrics must not be compared across ``online_y_scaler`` settings.

The channel summary mirrors :func:`pfns4neurostim.visualization.traces.comparison_table`: repetitions are
reduced within a channel first, then bounded metrics give the mean over channels with a 95% CI half-width
and heavy-tailed ones the median over channels with the interquartile range.
"""
from __future__ import annotations

import os
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .results import read_config

__all__ = [
    "PROVENANCE_COLUMNS",
    "SKIPPED_DIR_NAMES",
    "discover_tidy_runs",
    "load_run",
    "pool_runs",
    "channel_summary",
]

#: Columns pooling adds in front of each run's tidy columns.
PROVENANCE_COLUMNS: tuple[str, ...] = ("run_dir", "family", "tag", "online_y_scaler")

#: Directory names never descended into: archives, superseded-cohort tables, per-job shard records.
SKIPPED_DIR_NAMES: frozenset[str] = frozenset({"archive", "_stale_cohort", "shards"})

#: ``online_y_scaler`` of a run written before the key existed (2026-09-27): every such run scaled y offline.
PRE_ONLINE_SCALER_VALUE: str = "none"

#: Two-sided normal quantile of the 95% CI half-width.
CI_Z: float = 1.96

#: Quantiles of the interquartile range reported for heavy-tailed metrics.
IQR_QUANTILES: tuple[float, float] = (0.25, 0.75)


def discover_tidy_runs(roots: Iterable[str]) -> list[str]:
    """Find every run directory under ``roots`` that has both ``tidy.csv`` and ``config.yaml``.

    Args:
        roots: Directories to walk (e.g. ``output/benchmark``, ``output/stress``).

    Returns:
        Sorted run-directory paths. Directories named in :data:`SKIPPED_DIR_NAMES` are not descended into.

    Raises:
        FileNotFoundError: If a root does not exist (a typo must not yield an empty pool).
    """
    found: list[str] = []
    for root in roots:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Pooling root {root!r} does not exist.")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIPPED_DIR_NAMES]
            if "tidy.csv" in filenames and "config.yaml" in filenames:
                found.append(dirpath)
    return sorted(found)


def load_run(run_dir: str, output_root: str) -> pd.DataFrame:
    """Read one run's ``tidy.csv`` and prepend its provenance columns.

    Args:
        run_dir: Run directory holding ``tidy.csv`` and ``config.yaml``.
        output_root: Root the stored ``run_dir`` is made relative to.

    Returns:
        The tidy frame with :data:`PROVENANCE_COLUMNS` first.

    Note:
        ``online_y_scaler`` is taken from the config, not from the tidy column. The cell identity holds the
        scaler whenever it is active, so every cell a run serves was computed under the run's setting; the row
        field, however, was added to cached rows after some cells existed, and those rows read back the
        ``TidyRow`` default ``'none'`` (2026-10-08: 270 local NHP cells keyed ``minmax``). Disagreeing rows
        are counted in a warning.

    Raises:
        ValueError: If the tidy file is empty.
    """
    tidy = pd.read_csv(os.path.join(run_dir, "tidy.csv"), low_memory=False)
    if tidy.empty:
        raise ValueError(f"{run_dir}: tidy.csv has no rows.")
    config = read_config(run_dir)
    scaler = str(config.get("online_y_scaler", PRE_ONLINE_SCALER_VALUE))
    if "online_y_scaler" in tidy.columns:
        n_stale = int((tidy["online_y_scaler"].astype(str) != scaler).sum())
        if n_stale:
            warnings.warn(f"{run_dir}: {n_stale} rows carry a stale online_y_scaler label; using the config's "
                          f"{scaler!r} (the cell identity guarantees it).", stacklevel=2)
        tidy = tidy.drop(columns="online_y_scaler")
    provenance = pd.DataFrame({
        "run_dir": os.path.relpath(run_dir, output_root).replace(os.sep, "/"),
        "family": config.get("family"),
        "tag": config.get("tag"),
        "online_y_scaler": scaler,
    }, index=tidy.index)
    return pd.concat([provenance, tidy], axis=1)


def pool_runs(run_dirs: Sequence[str], output_root: str) -> pd.DataFrame:
    """Concatenate the provenance-stamped tidy tables of ``run_dirs``.

    Args:
        run_dirs: Run directories, e.g. from :func:`discover_tidy_runs`.
        output_root: Root the stored ``run_dir`` values are relative to.

    Returns:
        One long frame; a column absent from a run (e.g. ``knob`` in a benchmark) is NaN for its rows.

    Raises:
        ValueError: If ``run_dirs`` is empty.
    """
    if not run_dirs:
        raise ValueError("No run directories to pool.")
    return pd.concat([load_run(d, output_root) for d in run_dirs], ignore_index=True, sort=False)


def channel_summary(
    frame: pd.DataFrame,
    by: Sequence[str],
    metric: str,
    robust: bool,
) -> pd.DataFrame:
    """Summarise ``metric`` over channels within each ``by`` group.

    Repetitions are averaged (``robust``: median) within a channel ``(subject, emg)`` first. Then
    ``robust=False`` gives ``<metric>`` = mean over channels and ``<metric>_ci`` = 1.96 SD / sqrt(C);
    ``robust=True`` gives the median with ``<metric>_q25`` / ``<metric>_q75`` over channels.

    Args:
        frame: Pooled (or single-run) tidy rows.
        by: Grouping columns, e.g. ``["run_dir", "model", "acq_label"]``.
        metric: Metric column.
        robust: Median/IQR (heavy-tailed metrics such as R^2) instead of mean/CI.

    Returns:
        One row per group with ``by``, ``n_channels``, ``n_reps`` and the summary columns.

    Raises:
        KeyError: If ``metric`` or a ``by`` column is missing.
        ValueError: If a group has a non-finite metric value (fail fast, never averaged away).
    """
    missing = [c for c in (*by, metric, "subject", "emg", "rep") if c not in frame.columns]
    if missing:
        raise KeyError(f"channel_summary: missing column(s) {missing}.")
    values = frame[metric].to_numpy(dtype=float)                                   # [N]
    if not np.all(np.isfinite(values)):
        bad = frame.loc[~np.isfinite(values), [*by, "subject", "emg", "rep"]].head()
        raise ValueError(f"channel_summary: non-finite {metric!r} in rows like\n{bad}")
    reduce_reps = "median" if robust else "mean"
    per_channel = (frame.groupby([*by, "subject", "emg"], dropna=False)
                   .agg(value=(metric, reduce_reps), n_reps=("rep", "nunique"))
                   .reset_index())
    records: list[dict[str, object]] = []
    for keys, grp in per_channel.groupby(list(by), dropna=False):
        v = grp["value"].to_numpy()                                                # [C]
        rec: dict[str, object] = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        rec["n_channels"] = int(v.size)
        rec["n_reps"] = int(grp["n_reps"].max())
        if robust:
            rec[metric] = float(np.median(v))
            rec[f"{metric}_q25"] = float(np.quantile(v, IQR_QUANTILES[0]))
            rec[f"{metric}_q75"] = float(np.quantile(v, IQR_QUANTILES[1]))
        else:
            rec[metric] = float(v.mean())
            sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
            rec[f"{metric}_ci"] = CI_Z * sd / float(np.sqrt(v.size))
        records.append(rec)
    return pd.DataFrame.from_records(records)
