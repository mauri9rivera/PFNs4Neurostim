"""Per-channel neurostimulation data access.

A *channel* is one (dataset, subject, EMG) triple: a discrete pool of electrode
sites with a bank of noisy single-trial responses and a ground-truth response
per site. Every experiment in the package consumes :class:`ChannelData` and
nothing else, so a stressed channel, a synthetic channel and a real channel are
interchangeable.

**Migration seam.** ``_load_legacy_subject`` is the only caller of
:mod:`pfns4neurostim.data.legacy_io` (raw ``.mat`` loading); preprocessing is native
(:mod:`pfns4neurostim.data.preprocessing`).

**Preprocessing contract (P0.9).** Every model sees the *same* preprocessing —
MinMax-scaled coordinates in [0, 1]^D and per-channel z-scored responses
(``normalization='pfn'``, the default). The mode is an explicit argument
(``dataset.normalization`` in configs), recorded on every channel and in every
tidy row. The legacy code fed GPs raw coordinates and MinMax responses, which made
GP and PFN regrets live in different units. Regret is additionally reported in
units of the ground-truth range (:attr:`ChannelData.gt_range`), invariant to any
affine rescaling of the responses; see the preprocessing module for the rationale.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Sequence

import numpy as np

from .preprocessing import DEFAULT_NORMALIZATION, preprocess_channel

__all__ = [
    "ChannelData",
    "load_channel",
    "iter_channels",
    "count_emgs",
    "DEFAULT_DATA_ROOT",
]

#: Default raw-data root. Overridable per call and via the ``data_root`` config
#: key, so a SLURM job can point at ``$SLURM_TMPDIR/data`` (node-local SSD).
DEFAULT_DATA_ROOT: str = os.environ.get("PFNS4NEUROSTIM_DATA_ROOT", "./data")


@dataclass(frozen=True)
class ChannelData:
    """One (dataset, subject, EMG) search problem.

    Attributes:
        dataset: Dataset name (``'nhp'``, ``'rat'``, ``'spinal'``, ``'5d_rat'``,
            ``'synthetic_neurostim'``).
        subject: Subject index within the dataset.
        emg: EMG (response channel) index.
        X_pool: Candidate coordinates, MinMax-scaled to [0, 1]^D. Shape [N, D].
        Y_trials: Noisy single-trial responses, standardized; NaN where the lab
            flagged the trial invalid. Shape [N, R].
        y_gt: Ground-truth response per site, standardized in the same space as
            ``Y_trials``. Shape [N].
        ch2xy: Integer grid position of each site. Shape [N, D].
        grid_shape: Shape of the electrode grid the sites live on.
        gt_mode: How ``y_gt`` was built (``'full_mean'`` or ``'split_half'``).
        demo: ``'demo2'`` for in-vivo channels, ``'demo1'`` for synthetic ones.
        normalization: Preprocessing mode that produced the arrays (a key of
            :data:`pfns4neurostim.data.preprocessing.NORMALIZATIONS`).
        Y_invalid: The trials the lab flagged invalid, standardized in the same
            space as ``Y_trials``, NaN where the trial *was* valid. Shape [N, R].
            Preprocessing drops these, but the K5 knob needs them: contaminating
            with real artefacts is more faithful than synthesising heavy tails.
            ``None`` when the dataset carries no validity flags.
        queryable: Boolean mask over sites the optimizer is allowed to query,
            shape [N]. ``None`` means every site. The K6 dropout knob sets this
            rather than deleting rows, so regret stays measured against the full
            ground-truth map: losing electrodes must not make the task look
            easier by shrinking the optimum out of the comparison.
        stress: Provenance of any applied stress knob
            (``{'knob': ..., 'level': ...}``); empty for a nominal channel.
        meta: Free-form provenance (scaler, subject file, generator params).
    """

    dataset: str
    subject: int
    emg: int
    X_pool: np.ndarray       # [N, D]
    Y_trials: np.ndarray     # [N, R], NaN at invalid trials
    y_gt: np.ndarray         # [N]
    ch2xy: np.ndarray        # [N, D]
    grid_shape: tuple[int, ...]
    gt_mode: str = "full_mean"
    demo: str = "demo2"
    normalization: str = DEFAULT_NORMALIZATION
    Y_invalid: np.ndarray | None = None     # [N, R], NaN where the trial was valid
    queryable: np.ndarray | None = None     # [N] bool
    stress: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shapes and reject non-finite ground truth (fail fast)."""
        n, d = self.X_pool.shape
        if self.Y_trials.shape[0] != n:
            raise ValueError(
                f"ChannelData: Y_trials has {self.Y_trials.shape[0]} rows but "
                f"X_pool has {n}."
            )
        if self.y_gt.shape != (n,):
            raise ValueError(
                f"ChannelData: y_gt has shape {self.y_gt.shape}, expected ({n},)."
            )
        if self.ch2xy.shape[0] != n:
            raise ValueError(
                f"ChannelData: ch2xy has {self.ch2xy.shape[0]} rows but X_pool has {n}."
            )
        if not np.isfinite(self.y_gt).all():
            raise RuntimeError(
                f"ChannelData({self.label}): y_gt contains "
                f"{int((~np.isfinite(self.y_gt)).sum())} non-finite values."
            )
        if not np.isfinite(self.X_pool).all():
            raise RuntimeError(f"ChannelData({self.label}): X_pool contains non-finite values.")
        if self.Y_invalid is not None and self.Y_invalid.shape != self.Y_trials.shape:
            raise ValueError(
                f"ChannelData: Y_invalid has shape {self.Y_invalid.shape}, expected "
                f"{self.Y_trials.shape}."
            )
        if self.queryable is not None:
            if self.queryable.shape != (n,):
                raise ValueError(
                    f"ChannelData: queryable has shape {self.queryable.shape}, expected ({n},)."
                )
            if not self.queryable.any():
                raise ValueError("ChannelData: queryable mask excludes every site.")
        if np.isnan(self.Y_trials).all(axis=1).any():
            bad = int(np.isnan(self.Y_trials).all(axis=1).sum())
            raise RuntimeError(
                f"ChannelData({self.label}): {bad} site(s) have no valid trial; "
                "they must be dropped during preprocessing."
            )

    # --- derived quantities -------------------------------------------------
    @property
    def n_sites(self) -> int:
        """Number of candidate electrode sites N."""
        return int(self.X_pool.shape[0])

    @property
    def n_dims(self) -> int:
        """Search-space dimensionality D."""
        return int(self.X_pool.shape[1])

    @property
    def n_trials(self) -> int:
        """Number of trial columns R in the response bank."""
        return int(self.Y_trials.shape[1])

    @property
    def gt_range(self) -> float:
        """Ground-truth response range, the unit of every reported regret (P0.9)."""
        rng = float(np.max(self.y_gt) - np.min(self.y_gt))
        if not np.isfinite(rng) or rng <= 0.0:
            raise RuntimeError(
                f"ChannelData({self.label}): degenerate ground-truth range {rng}; "
                "regret would be undefined."
            )
        return rng

    @property
    def best_site(self) -> int:
        """Index of the true optimum site over the full ground-truth map."""
        return int(np.argmax(self.y_gt))

    @property
    def queryable_indices(self) -> np.ndarray:
        """Indices the optimizer may query, shape [n_queryable]."""
        if self.queryable is None:
            return np.arange(self.n_sites)
        return np.flatnonzero(self.queryable)

    @property
    def n_queryable(self) -> int:
        """How many sites the optimizer may query."""
        return int(self.queryable_indices.size)

    @property
    def n_invalid_trials(self) -> int:
        """Number of lab-flagged invalid trials available for K5 contamination."""
        if self.Y_invalid is None:
            return 0
        return int(np.isfinite(self.Y_invalid).sum())

    @property
    def label(self) -> str:
        """Short identifier, e.g. ``'nhp-s1-e0'``."""
        return f"{self.dataset}-s{self.subject}-e{self.emg}"

    def with_trials(self, Y_trials: np.ndarray, **updates: Any) -> "ChannelData":
        """Return a copy carrying a new trial bank (the knob-application path).

        Args:
            Y_trials: Replacement response bank, shape [N, R].
            **updates: Other fields to override (e.g. ``stress=...``).

        Returns:
            A new :class:`ChannelData`; the original is never mutated.
        """
        return replace(self, Y_trials=Y_trials, **updates)


# ---------------------------------------------------------------------------
# Legacy seam: pfns4neurostim.data.legacy_io (native split pending, task #1 Step 2)
# ---------------------------------------------------------------------------
def _load_legacy_subject(dataset: str, subject: int, data_root: str) -> dict[str, Any]:
    """Load one subject's raw data dict through the legacy loader.

    Args:
        dataset: Dataset name.
        subject: Subject index.
        data_root: Directory holding the raw ``.mat`` trees.

    Returns:
        The legacy data dictionary (keys ``sorted_resp``, ``sorted_respMean``,
        ``sorted_isvalid``, ``ch2xy``, ``grid_shape``, ...).
    """
    from .legacy_io import load_data  # noqa: PLC0415 - seam, intentional

    return load_data(dataset, subject, data_root=data_root)


def count_emgs(dataset: str, subject: int, data_root: str = DEFAULT_DATA_ROOT) -> int:
    """Return the number of EMG channels recorded for one subject.

    Args:
        dataset: Dataset name.
        subject: Subject index.
        data_root: Raw-data root.

    Returns:
        Number of EMG channels.
    """
    data = _load_legacy_subject(dataset, subject, data_root)
    return int(np.asarray(data["sorted_resp"]).shape[1])


def load_channel(
    dataset: str,
    subject: int,
    emg: int,
    *,
    data_root: str = DEFAULT_DATA_ROOT,
    gt_mode: str = "full_mean",
    normalization: str = DEFAULT_NORMALIZATION,
    subject_data: dict[str, Any] | None = None,
) -> ChannelData:
    """Load one (dataset, subject, EMG) channel as a :class:`ChannelData`.

    Args:
        dataset: Dataset name (``'nhp'``, ``'rat'``, ``'spinal'``, ``'5d_rat'``).
        subject: Subject index.
        emg: EMG index.
        data_root: Directory holding the raw ``.mat`` trees. Point this at
            ``$SLURM_TMPDIR/data`` inside a cluster job.
        gt_mode: ``'full_mean'`` (primary). ``'split_half'`` arrives with
            task #7 and raises until then.
        normalization: Preprocessing mode, a key of
            :data:`pfns4neurostim.data.preprocessing.NORMALIZATIONS`.
        subject_data: Pre-loaded legacy dict, to avoid re-reading the ``.mat``
            file once per EMG.

    Returns:
        The channel, with shared-unit preprocessing applied.

    Raises:
        NotImplementedError: For ``gt_mode='split_half'`` (task #7 Step 0).
        ValueError: For an unknown ``normalization``.
    """
    if gt_mode not in ("full_mean", "split_half"):
        raise ValueError(f"load_channel: unknown gt_mode {gt_mode!r}.")
    if gt_mode == "split_half":
        raise NotImplementedError(
            "gt_mode='split_half' lands with task #7 (data/ground_truth.py). "
            "Use gt_mode='full_mean' until then."
        )

    data = subject_data if subject_data is not None else _load_legacy_subject(
        dataset, subject, data_root
    )
    pre = preprocess_channel(data, emg, normalization)
    ch2xy = np.asarray(data["ch2xy"])[pre.site_mask]  # [N, D]

    # 5d_rat carries the key with value None, so `.get(key, default)` is not enough.
    raw_grid = data.get("grid_shape")
    if raw_grid is None:
        raw_grid = ch2xy.max(axis=0) + 1
    grid_shape = tuple(int(v) for v in np.ravel(raw_grid))

    return ChannelData(
        dataset=dataset,
        subject=int(subject),
        emg=int(emg),
        X_pool=pre.X_pool,
        Y_trials=pre.Y_trials,
        y_gt=pre.y_gt,
        ch2xy=np.asarray(ch2xy, dtype=int),
        grid_shape=grid_shape,
        gt_mode=gt_mode,
        demo="demo2",
        normalization=pre.normalization,
        Y_invalid=pre.Y_invalid,
        meta={"scaler_y": pre.scaler_y, "data_root": data_root},
    )


def iter_channels(
    dataset: str,
    subjects: Sequence[int],
    emgs: Sequence[int] | None = None,
    *,
    data_root: str = DEFAULT_DATA_ROOT,
    gt_mode: str = "full_mean",
    normalization: str = DEFAULT_NORMALIZATION,
) -> Iterator[ChannelData]:
    """Yield every (subject, EMG) channel of a dataset, loading each subject once.

    Args:
        dataset: Dataset name.
        subjects: Subject indices to iterate.
        emgs: EMG indices; ``None`` means every EMG the subject has.
        data_root: Raw-data root.
        gt_mode: Ground-truth mode, see :func:`load_channel`.
        normalization: Preprocessing mode, see :func:`load_channel`.

    Yields:
        One :class:`ChannelData` per (subject, EMG) pair, skipping EMGs whose
        preprocessing fails (e.g. no valid trials), with a warning to stderr.
    """
    for subject in subjects:
        data = _load_legacy_subject(dataset, subject, data_root)
        emg_indices = (
            list(emgs) if emgs is not None else list(range(int(np.asarray(data["sorted_resp"]).shape[1])))
        )
        for emg in emg_indices:
            try:
                yield load_channel(
                    dataset,
                    subject,
                    emg,
                    data_root=data_root,
                    gt_mode=gt_mode,
                    normalization=normalization,
                    subject_data=data,
                )
            except (RuntimeError, ValueError) as exc:
                print(
                    f"[iter_channels] skipping {dataset}-s{subject}-e{emg}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
