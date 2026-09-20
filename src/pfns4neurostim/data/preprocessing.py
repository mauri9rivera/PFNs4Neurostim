"""Native per-channel preprocessing (the project's preprocessing contract).

**Contract (sprint 2026-09-20, Task 1).** Every surrogate sees the same inputs:

* **X — MinMax to [0, 1]^D.** GPyTorch's RBF lengthscale prior assumes O(1) input
  scales; raw electrode indices push the true lengthscale into the prior's tail, and
  on ``5d_rat`` the five axes carry different physical units.
* **y — z-scored (zero mean, unit variance)** with the scaler fitted on *valid*
  trials only. GP mean/outputscale/noise priors are calibrated for standardized
  targets, and TabPFN standardizes internally anyway.
* **Reported regret does not depend on the y scaler**: it is divided by the
  ground-truth range and is affine-invariant.

The alternative modes in :data:`NORMALIZATIONS` exist so the contract can be
*measured* (the Task 1 A/B) and so a result always states which one produced it.

This module is pure NumPy/scikit-learn: it takes the raw subject dictionary and
returns arrays. It does not touch :mod:`pfns4neurostim.data.legacy_io`, which stays
frozen as the pre-restructure reference (``tests/data/test_preprocessing.py`` pins
the default mode to it numerically).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

__all__ = [
    "DEFAULT_NORMALIZATION",
    "NORMALIZATIONS",
    "Normalization",
    "PreprocessedChannel",
    "get_normalization",
    "valid_site_mask",
    "preprocess_channel",
]

XScaling = Literal["minmax", "raw"]
YScaling = Literal["zscore", "minmax"]


@dataclass(frozen=True)
class Normalization:
    """One (X scaling, y scaling) preprocessing mode.

    Attributes:
        x: ``'minmax'`` maps each coordinate axis to [0, 1]; ``'raw'`` passes
            electrode coordinates through unchanged.
        y: ``'zscore'`` standardizes responses; ``'minmax'`` maps them to [0, 1].
    """

    x: XScaling
    y: YScaling


#: Name of the project-wide default mode (the contract above).
DEFAULT_NORMALIZATION: str = "pfn"

#: Registered modes. ``'pfn'`` is the contract; the rest are A/B arms.
NORMALIZATIONS: dict[str, Normalization] = {
    "pfn": Normalization(x="minmax", y="zscore"),
    "minmax_x_minmax_y": Normalization(x="minmax", y="minmax"),
    "raw_x_zscore_y": Normalization(x="raw", y="zscore"),
    "raw_x_minmax_y": Normalization(x="raw", y="minmax"),
}


def get_normalization(name: str) -> Normalization:
    """Look up a registered normalization mode.

    Args:
        name: Key of :data:`NORMALIZATIONS`.

    Returns:
        The :class:`Normalization`.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization {name!r}; registered: {sorted(NORMALIZATIONS)}."
        )
    return NORMALIZATIONS[name]


@dataclass(frozen=True)
class PreprocessedChannel:
    """Arrays for one (subject, EMG) channel after preprocessing.

    Attributes:
        X_pool: Candidate coordinates after X scaling. Shape [N, D].
        Y_trials: Standardized single-trial responses, NaN at invalid trials. Shape [N, R].
        y_gt: Per-site ground-truth mean in the same space as ``Y_trials``. Shape [N].
        Y_invalid: Lab-flagged invalid trials in the same space, NaN where the trial
            was valid; ``None`` when the dataset has no validity flags. Shape [N, R].
        site_mask: Which raw sites survived (>= 1 valid trial). Shape [N_raw].
        scaler_y: Fitted y scaler, for inverse transforms.
        normalization: Name of the mode that produced these arrays.
    """

    X_pool: np.ndarray
    Y_trials: np.ndarray
    y_gt: np.ndarray
    Y_invalid: np.ndarray | None
    site_mask: np.ndarray
    scaler_y: Any
    normalization: str


def valid_site_mask(subject_data: Mapping[str, Any], emg: int) -> np.ndarray:
    """Return the sites with at least one valid trial at this EMG.

    Sites without one carry the loader's fill value (0.0), never a measurement, so
    they must not enter the pool.

    Args:
        subject_data: Raw subject dictionary (``ch2xy``, ``sorted_resp``, optionally
            ``sorted_isvalid``).
        emg: EMG index.

    Returns:
        Boolean mask, shape [N_raw].
    """
    n_raw = int(np.asarray(subject_data["ch2xy"]).shape[0])
    if "sorted_isvalid" not in subject_data:
        return np.ones(n_raw, dtype=bool)
    valid = np.asarray(subject_data["sorted_isvalid"])[:, emg, :]  # [N_raw, R]
    return (valid != 0).any(axis=-1)


def preprocess_channel(
    subject_data: Mapping[str, Any],
    emg: int,
    normalization: str = DEFAULT_NORMALIZATION,
) -> PreprocessedChannel:
    """Build the per-EMG arrays every experiment consumes.

    Args:
        subject_data: Raw subject dictionary.
        emg: EMG index.
        normalization: Key of :data:`NORMALIZATIONS`.

    Returns:
        A :class:`PreprocessedChannel`.

    Raises:
        ValueError: For an unknown ``normalization``.
        RuntimeError: If the channel has no valid trial, or a scaled array is
            non-finite (fail fast rather than propagate NaN into a BO run).
    """
    mode = get_normalization(normalization)
    coords_raw = np.asarray(subject_data["ch2xy"])                                     # [N_raw, D]
    # float32 matches the pre-restructure pipeline, so the default mode reproduces
    # its numbers exactly.
    resp_raw = np.asarray(subject_data["sorted_resp"])[:, emg, :].astype(np.float32)   # [N_raw, R]
    resp_mean_raw = np.asarray(subject_data["sorted_respMean"])[:, emg]                # [N_raw]

    site_mask = valid_site_mask(subject_data, emg)                                     # [N_raw]
    invalid_raw: np.ndarray | None = None
    if "sorted_isvalid" in subject_data:
        flagged = np.asarray(subject_data["sorted_isvalid"])[:, emg, :] == 0           # [N_raw, R]
        invalid_raw = np.asarray(subject_data["sorted_resp"])[:, emg, :].astype(np.float64)
        invalid_raw = np.where(flagged, invalid_raw, np.nan)                           # [N_raw, R]
        resp_raw = np.where(flagged, np.nan, resp_raw)

    coords = coords_raw[site_mask]                                                     # [N, D]
    resp = resp_raw[site_mask]                                                         # [N, R]
    resp_mean = resp_mean_raw[site_mask]                                               # [N]

    valid_flat = resp[~np.isnan(resp)].reshape(-1, 1)                                  # [n_valid, 1]
    if valid_flat.size == 0:
        raise RuntimeError(f"preprocess_channel: no valid trials for emg={emg}.")

    X_pool = (
        MinMaxScaler().fit_transform(coords) if mode.x == "minmax" else coords.astype(np.float64)
    )                                                                                  # [N, D]
    scaler_y = StandardScaler() if mode.y == "zscore" else MinMaxScaler()
    scaler_y.fit(valid_flat)

    n, r = resp.shape
    Y_trials = scaler_y.transform(resp.reshape(-1, 1)).reshape(n, r)                   # [N, R]
    y_gt = scaler_y.transform(resp_mean.reshape(-1, 1)).ravel()                        # [N]

    Y_invalid: np.ndarray | None = None
    if invalid_raw is not None:
        inv = invalid_raw[site_mask]                                                   # [N, R]
        Y_invalid = np.full(inv.shape, np.nan, dtype=np.float64)
        has = np.isfinite(inv)
        if has.any():
            Y_invalid[has] = scaler_y.transform(inv[has].reshape(-1, 1)).ravel()

    for name, arr in (("X_pool", X_pool), ("y_gt", y_gt)):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"preprocess_channel: {name} is non-finite for emg={emg}.")

    return PreprocessedChannel(
        X_pool=np.asarray(X_pool, dtype=np.float64),
        Y_trials=np.asarray(Y_trials, dtype=np.float64),
        y_gt=np.asarray(y_gt, dtype=np.float64),
        Y_invalid=Y_invalid,
        site_mask=site_mask,
        scaler_y=scaler_y,
        normalization=normalization,
    )
