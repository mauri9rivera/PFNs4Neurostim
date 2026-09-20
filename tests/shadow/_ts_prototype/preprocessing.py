"""Shared preprocessing (P0.9): identical X / y units for every surrogate.

Phase-3 target: ``src/pfns4neurostim/data/preprocessing.py`` (shared by all models).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


@dataclass
class SharedData:
    """One (subject, channel) problem in shared units.

    Attributes:
        X: Inputs MinMax-scaled per dimension to [0, 1], shape [M, D].
        y_trials: Single-trial responses z-scored with valid-trial mean/SD, NaN = invalid,
            shape [M, R].
        y_gt: Per-site ground-truth mean response in the same units, shape [M].
        name: Identifier string.
    """

    X: np.ndarray
    y_trials: np.ndarray
    y_gt: np.ndarray
    name: str


def shared_preprocess(
    coords: np.ndarray, trials: np.ndarray, gt_mean: Optional[np.ndarray] = None, name: str = ""
) -> SharedData:
    """Apply the shared transform used by ALL surrogates.

    Args:
        coords: Raw inputs, shape [M, D].
        trials: Raw single-trial responses with NaN for invalid trials, shape [M, R].
        gt_mean: Raw per-site ground truth [M]; defaults to the nan-mean of ``trials``.
        name: Identifier.

    Returns:
        SharedData in shared units.

    Raises:
        RuntimeError: If any site has no valid trial or data are non-finite.
    """
    coords = np.asarray(coords, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    valid = ~np.isnan(trials)
    if not valid.any(axis=1).all():
        raise RuntimeError("shared_preprocess: every site needs at least one valid trial.")
    lo, hi = coords.min(0), coords.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    X = (coords - lo) / span                                    # [M, D]
    mu, sd = np.nanmean(trials), np.nanstd(trials)
    if not (np.isfinite(mu) and sd > 0):
        raise RuntimeError("shared_preprocess: degenerate trial distribution.")
    y_trials = (trials - mu) / sd                               # [M, R]
    gt = np.nanmean(trials, axis=1) if gt_mean is None else np.asarray(gt_mean, dtype=np.float64)
    y_gt = (gt - mu) / sd                                       # [M]
    if not (np.isfinite(X).all() and np.isfinite(y_gt).all()):
        raise RuntimeError("shared_preprocess: non-finite output.")
    return SharedData(X=X, y_trials=y_trials, y_gt=y_gt, name=name)


def load_neurostim(dataset: str, subject: int, emg: int) -> SharedData:
    """Load a real neurostim problem via the (read-only) ``src`` loader, in shared units.

    Args:
        dataset: ``'nhp'``, ``'rat'``, ``'spinal'`` or ``'5d_rat'``.
        subject: Subject index.
        emg: Channel index.

    Returns:
        SharedData.
    """
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    from pfns4neurostim.data.legacy_io import load_data  # noqa: WPS433  (read-only import)

    cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)  # src loader uses './data'
    try:
        data = load_data(dataset, subject)
    finally:
        os.chdir(cwd)
    trials = data["sorted_resp"][:, emg, :].astype(np.float64)
    if "sorted_isvalid" in data:
        trials[data["sorted_isvalid"][:, emg, :] == 0] = np.nan
    mask = ~np.isnan(trials).all(axis=1)  # mirrors src/evaluation._valid_site_mask
    return shared_preprocess(
        data["ch2xy"][mask], trials[mask], data["sorted_respMean"][mask, emg],
        name=f"{dataset}-s{subject}-e{emg}",
    )


def draw_trial(data: SharedData, idx: int, rng: np.random.Generator) -> float:
    """One valid single trial at site ``idx`` (the agent's noisy observation).

    Args:
        data: Problem.
        idx: Site index.
        rng: Explicit Generator.

    Returns:
        Observed value in shared units.
    """
    row = data.y_trials[idx]
    valid = np.flatnonzero(~np.isnan(row))
    return float(row[rng.choice(valid)])
