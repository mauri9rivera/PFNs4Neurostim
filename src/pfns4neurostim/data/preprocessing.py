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
        site_mask: Which raw sites survived (>= 1 valid trial). Shape [N_raw].
        scaler_y: Fitted y scaler, for inverse transforms.
        normalization: Name of the mode that produced these arrays.
    """

    X_pool: np.ndarray
    Y_trials: np.ndarray
    y_gt: np.ndarray
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
    if "sorted_isvalid" in subject_data:
        flagged = np.asarray(subject_data["sorted_isvalid"])[:, emg, :] == 0           # [N_raw, R]
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

    for name, arr in (("X_pool", X_pool), ("y_gt", y_gt)):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"preprocess_channel: {name} is non-finite for emg={emg}.")

    return PreprocessedChannel(
        X_pool=np.asarray(X_pool, dtype=np.float64),
        Y_trials=np.asarray(Y_trials, dtype=np.float64),
        y_gt=np.asarray(y_gt, dtype=np.float64),
        site_mask=site_mask,
        scaler_y=scaler_y,
        normalization=normalization,
    )


# ---------------------------------------------------------------------------
# Online (causal) y scaling — deployment-aligned, refit inside the BO loop
# ---------------------------------------------------------------------------
#: Smallest y spread an online scaler will divide by. Below it the scale is set to 1.0 (offset only):
#: at ``n_init`` a handful of initial draws can share almost the same value, and a range of ~0 would
#: otherwise produce inf/NaN targets, which the fail-fast contract forbids.
ONLINE_SCALE_FLOOR: float = 1e-8

#: Online y-scaling modes. ``'none'`` keeps the dataset-level scaler of :data:`NORMALIZATIONS` (the
#: current behaviour); the others refit an affine y transform from the observations collected SO FAR.
ONLINE_Y_MODES: tuple[str, ...] = ("none", "minmax", "zscore")


@dataclass
class OnlineYScaler:
    """Affine y transform refit from the observations collected so far.

    Why this exists (2026-09-27). The dataset-level scaler of :data:`NORMALIZATIONS` is fitted on *all*
    trials of a channel, which no live experiment can do: at query *t* you have seen *t* responses and
    nothing else. This scaler is the deployment-aligned alternative -- it is *causal*, fitted only on the
    observations already collected -- so the gap between the two is measurable rather than argued about.

    It is applied inside the BO loop, not at load time: the transform changes at every step, and the
    surrogate's predictions must be mapped back before they are compared with the ground truth. ``minmax``
    is the mode the legacy GP arm used offline (``raw_x_minmax_y``); ``zscore`` is the online counterpart of
    the current default.

    This is **not** a neutral reparameterization, and the effect differs per model family. Measured
    2026-09-27 on 3 NHP channels x 2 reps, ``ts_marginal``, budget 40 (mean recommended regret / R^2 /
    90% coverage):

    ==============  ===============  ===============  ===============
    model           none             minmax           zscore
    ==============  ===============  ===============  ===============
    TabPFN-2.5      0.491/0.37/0.94  0.491/0.37/0.94  0.491/0.37/0.94
    GP-MLL          0.322/0.46/0.97  0.563/0.31/0.93  0.493/0.25/0.93
    GP-fixed        0.255/-0.86/0.60 0.224/-0.18/0.95 0.380/-2.08/0.68
    ==============  ===============  ===============  ===============

    * **TabPFN is exactly invariant** -- identical to four decimals in every mode, which confirms directly
      that it undoes any affine y transform internally (it z-scores ``y`` and rescales its
      bar-distribution borders to match).
    * **GP-MLL is hurt**, by ~75% in regret under ``minmax``. Its priors assume standardized targets, and
      refitting the scaler every step makes the marginal-likelihood fit chase a moving target that is
      itself estimated from very few points early in the run.
    * **GP-fixed is helped, contrary to the expectation first written here.** Under ``minmax`` its regret
      improves slightly and its calibration improves a lot (90% coverage 0.60 -> 0.95, R^2 -0.86 -> -0.18):
      its pinned ``noise=0.01`` is far too small next to z-scored targets but close to reasonable next to
      ``[0, 1]`` targets. So online ``minmax`` is not a uniform cost -- it redistributes accuracy between
      the two GP arms.
    * Under ``minmax`` the running maximum *is* the incumbent, and the max of noisy draws is upward
      biased, so the scale is itself a noisy, monotonically growing quantity early in a run.

    Range-normalized regret is unaffected either way, because it divides by the ground-truth range and is
    invariant to any affine transform of ``y``; R-squared is scale-invariant too. NLL is **not** (it carries
    a ``log(scale)`` term), so NLL must not be compared across modes.

    Attributes:
        mode: One of :data:`ONLINE_Y_MODES`.
        offset: Subtracted before scaling.
        scale: Divided by after the offset.
    """

    mode: str
    offset: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the mode.

        Raises:
            ValueError: If ``mode`` is not in :data:`ONLINE_Y_MODES`.
        """
        if self.mode not in ONLINE_Y_MODES:
            raise ValueError(
                f"OnlineYScaler mode must be one of {list(ONLINE_Y_MODES)}, got {self.mode!r}."
            )

    @property
    def active(self) -> bool:
        """Whether this scaler does anything (``mode != 'none'``)."""
        return self.mode != "none"

    def fit(self, y: np.ndarray) -> "OnlineYScaler":
        """Refit the transform from the observations seen so far.

        Args:
            y: Observed responses, shape [t].

        Returns:
            ``self``, so a fit can be chained into a transform.

        Raises:
            RuntimeError: If ``y`` holds a non-finite value.
        """
        if self.mode == "none":
            return self
        y = np.asarray(y, dtype=np.float64).ravel()
        if not np.isfinite(y).all():
            raise RuntimeError(
                f"OnlineYScaler.fit received {int((~np.isfinite(y)).sum())} non-finite observations."
            )
        if self.mode == "minmax":
            lo, hi = float(np.min(y)), float(np.max(y))
            self.offset, spread = lo, hi - lo
        else:
            self.offset, spread = float(np.mean(y)), float(np.std(y))
        self.scale = spread if spread > ONLINE_SCALE_FLOOR else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Map observations into the scaled space the surrogate is fitted in.

        Args:
            y: Responses, any shape.

        Returns:
            The transformed array (a copy).
        """
        if not self.active:
            return np.asarray(y, dtype=np.float64)
        return (np.asarray(y, dtype=np.float64) - self.offset) / self.scale

    def inverse_mean(self, mean: np.ndarray) -> np.ndarray:
        """Map a predicted mean back to the channel's own y space.

        Args:
            mean: Predictive means in the scaled space.

        Returns:
            The means in channel space.
        """
        if not self.active:
            return np.asarray(mean, dtype=np.float64)
        return np.asarray(mean, dtype=np.float64) * self.scale + self.offset

    def inverse_std(self, std: np.ndarray) -> np.ndarray:
        """Map a predictive standard deviation back to the channel's own y space.

        A standard deviation carries the scale but not the offset.

        Args:
            std: Predictive standard deviations in the scaled space.

        Returns:
            The standard deviations in channel space.
        """
        if not self.active:
            return np.asarray(std, dtype=np.float64)
        return np.asarray(std, dtype=np.float64) * self.scale
