"""Per-channel neurostimulation data access.

A *channel* is one (dataset, subject, EMG) triple: a discrete pool of electrode
sites with a bank of noisy single-trial responses and a ground-truth response
per site. Every experiment in the package consumes :class:`ChannelData` and
nothing else, so a stressed channel, a synthetic channel and a real channel are
interchangeable.

**Loading seam.** Raw ``.mat`` access goes through
:func:`pfns4neurostim.data.loaders.load_subject`, which serves ``5d_rat`` from its own
native module and the remaining datasets from the frozen
:mod:`pfns4neurostim.data.legacy_io`. Preprocessing is native
(:mod:`pfns4neurostim.data.preprocessing`). A loader that versions its raw files reports
a **data cohort**, which this module puts in ``ChannelData.meta['data_cohort']``; the
runners fold it into the cell-cache identity, so replacing a dataset's raw files
invalidates that dataset's cached results and nothing else.

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
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .preprocessing import DEFAULT_NORMALIZATION, preprocess_channel

__all__ = [
    "ChannelData",
    "load_channel",
    "iter_channels",
    "count_emgs",
    "parse_shard",
    "shard_size",
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
        failure_time: When each electrode fails, as a fraction of the run in
            [0, 1); ``inf`` for electrodes that never fail. Shape [N]. ``None``
            means no electrode fails. Set by the K6 failure knob: from that moment
            a query of the electrode returns ``failure_value`` instead of a trial,
            and metrics are scored on the surviving electrodes.
        failure_value: Response of a failed electrode, in the standardized space
            (0.0 = the channel's mean response).
        stress: Provenance of any applied stress knob
            (``{'knob': ..., 'level': ...}``); empty for a nominal channel.
        meta: Free-form provenance (scaler, subject file, generator params).
        split_id: Split-half instance index in ``[0, 2R)`` when
            ``gt_mode='split_half'`` (see :mod:`pfns4neurostim.data.ground_truth`);
            ``None`` for a full-mean channel.
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
    failure_time: np.ndarray | None = None  # [N], run fraction; inf = never fails
    failure_value: float = 0.0
    stress: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    split_id: int | None = None

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
        if self.failure_time is not None:
            if self.failure_time.shape != (n,):
                raise ValueError(
                    f"ChannelData: failure_time has shape {self.failure_time.shape}, expected ({n},)."
                )
            if not np.isinf(self.failure_time).any():
                raise ValueError("ChannelData: every electrode fails; nothing would survive.")
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
    def survivors(self) -> np.ndarray:
        """Electrodes still alive at the end of the run, bool shape [N]."""
        if self.failure_time is None:
            return np.ones(self.n_sites, dtype=bool)
        return np.isinf(self.failure_time)

    def is_failed(self, index: int, progress: float) -> bool:
        """Whether an electrode has failed at a given point of the run.

        Args:
            index: Site index.
            progress: Fraction of the run elapsed, in [0, 1).

        Returns:
            True once ``progress`` reaches the electrode's failure time.
        """
        return self.failure_time is not None and bool(progress >= self.failure_time[index])

    @property
    def label(self) -> str:
        """Short identifier, e.g. ``'nhp-s1-e0'``."""
        return f"{self.dataset}-s{self.subject}-e{self.emg}"

    def to_raw(self, y: np.ndarray) -> np.ndarray | None:
        """Map standardized responses back to raw units, or ``None`` without a scaler.

        The exploration score is a ratio and needs raw (non-negative) units, so it
        is undefined for a channel that carries no ``scaler_y`` (only hand-built
        test channels; every loaded channel records one).

        Args:
            y: Responses in the standardized space, shape [N].

        Returns:
            The same responses in raw units, shape [N], or ``None``.
        """
        scaler = self.meta.get("scaler_y")
        if scaler is None:
            return None
        raw = scaler.inverse_transform(np.asarray(y, dtype=np.float64).reshape(-1, 1))  # [N, 1]
        return np.asarray(raw, dtype=np.float64).reshape(-1)                         # [N]

    @property
    def y_end(self) -> np.ndarray:
        """Ground truth as it stands at the end of the run, shape [N].

        Failed electrodes read ``failure_value``; everything else is ``y_gt``.
        Equal to ``y_gt`` when no electrode fails.
        """
        return np.where(self.survivors, self.y_gt, self.failure_value)

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
def _load_subject(dataset: str, subject: int, data_root: str) -> dict[str, Any]:
    """Load one subject's raw data dict through the dataset's loader.

    Args:
        dataset: Dataset name.
        subject: Subject index.
        data_root: Directory holding the raw ``.mat`` trees.

    Returns:
        The raw data dictionary (keys ``sorted_resp``, ``sorted_respMean``,
        ``sorted_isvalid``, ``ch2xy``, ``grid_shape``, ...).
    """
    from .loaders import load_subject  # noqa: PLC0415 - seam, intentional

    return load_subject(dataset, subject, data_root=data_root)


def count_emgs(dataset: str, subject: int, data_root: str = DEFAULT_DATA_ROOT) -> int:
    """Return the number of EMG channels recorded for one subject.

    Args:
        dataset: Dataset name.
        subject: Subject index.
        data_root: Raw-data root.

    Returns:
        Number of EMG channels.
    """
    data = _load_subject(dataset, subject, data_root)
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
        gt_mode: ``'full_mean'`` only. Split-half channels are several instances
            per channel, built per repetition by
            :func:`pfns4neurostim.data.ground_truth.ground_truth_instances` from the
            full-mean channel, so ``'split_half'`` raises here with that pointer.
        normalization: Preprocessing mode, a key of
            :data:`pfns4neurostim.data.preprocessing.NORMALIZATIONS`.
        subject_data: Pre-loaded raw dict, to avoid re-reading the ``.mat``
            file once per EMG.

    Returns:
        The channel, with shared-unit preprocessing applied.

    Raises:
        NotImplementedError: For ``gt_mode='split_half'`` (expand instead, see above).
        ValueError: For an unknown ``normalization``.
    """
    if gt_mode not in ("full_mean", "split_half"):
        raise ValueError(f"load_channel: unknown gt_mode {gt_mode!r}.")
    if gt_mode == "split_half":
        raise NotImplementedError(
            "gt_mode='split_half' is not a load-time mode: load with gt_mode='full_mean' and "
            "expand with pfns4neurostim.data.ground_truth.ground_truth_instances (one instance "
            "per repetition, rep i -> instance i mod 2R). bo_benchmark does this; a runner that "
            "passes split_half here has not been wired for it yet."
        )

    data = subject_data if subject_data is not None else _load_subject(
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
        meta=_channel_meta(data, pre.scaler_y, data_root),
    )


def _channel_meta(data: Mapping[str, Any], scaler_y: Any, data_root: str) -> dict[str, Any]:
    """Assemble a channel's provenance, including the raw files' cohort stamp.

    Args:
        data: The raw subject dictionary.
        scaler_y: Fitted response scaler.
        data_root: Directory the raw files were read from.

    Returns:
        The ``ChannelData.meta`` mapping. ``data_cohort`` and ``subject_key`` appear
        only for datasets whose loader versions its raw files, so a channel of any
        other dataset keeps exactly the provenance it had before.
    """
    meta: dict[str, Any] = {"scaler_y": scaler_y, "data_root": data_root}
    for key in ("data_cohort", "subject_key"):
        if key in data:
            meta[key] = data[key]
    return meta


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse a ``--shard`` spec ``'i/n'`` into ``(i, n)``.

    Args:
        spec: Text such as ``'2/4'`` (the third of four shards).

    Returns:
        ``(index, count)`` with ``0 <= index < count``.

    Raises:
        ValueError: If the spec is malformed or out of range.
    """
    try:
        i_txt, n_txt = spec.split("/")
        i, n = int(i_txt), int(n_txt)
    except ValueError as exc:
        raise ValueError(f"--shard must look like 'i/n' (e.g. 0/4), got {spec!r}.") from exc
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"--shard {spec!r}: need n >= 1 and 0 <= i < n.")
    return i, n


def shard_size(total: int, shard: tuple[int, int] | None) -> int:
    """Number of channels a shard owns out of ``total`` (round-robin partition).

    Args:
        total: Total channel count.
        shard: ``(i, n)`` or ``None`` for everything.

    Returns:
        Count of channel positions ``p`` in ``range(total)`` with ``p % n == i``.
    """
    if shard is None:
        return total
    i, n = shard
    return max(0, (total - i + n - 1) // n)


def iter_channels(
    dataset: str,
    subjects: Sequence[int],
    emgs: Sequence[int] | None = None,
    *,
    data_root: str = DEFAULT_DATA_ROOT,
    gt_mode: str = "full_mean",
    normalization: str = DEFAULT_NORMALIZATION,
    shard: tuple[int, int] | None = None,
) -> Iterator[ChannelData]:
    """Yield every (subject, EMG) channel of a dataset, loading each subject once.

    Args:
        dataset: Dataset name.
        subjects: Subject indices to iterate.
        emgs: EMG indices; ``None`` means every EMG the subject has.
        data_root: Raw-data root.
        gt_mode: Ground-truth mode, see :func:`load_channel`.
        normalization: Preprocessing mode, see :func:`load_channel`.
        shard: ``(i, n)`` keeps every n-th channel starting at ``i`` in iteration order
            (subject-major), so ``n`` processes cover all channels exactly once with
            balanced load. The position counts every candidate channel, even ones that
            later fail preprocessing, so shards stay disjoint and complete.

    Yields:
        One :class:`ChannelData` per (subject, EMG) pair, skipping EMGs whose
        preprocessing fails (e.g. no valid trials), with a warning to stderr.
    """
    position = 0
    for subject in subjects:
        data = _load_subject(dataset, subject, data_root)
        emg_indices = (
            list(emgs) if emgs is not None else list(range(int(np.asarray(data["sorted_resp"]).shape[1])))
        )
        for emg in emg_indices:
            mine = shard is None or position % shard[1] == shard[0]
            position += 1
            if not mine:
                continue
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
