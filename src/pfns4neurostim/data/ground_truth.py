"""Ground-truth construction for in-vivo channels (task #7, P0.7).

Two modes:

* ``full_mean`` (primary): the ground truth of a site is the mean of *all* its valid
  trials, and the BO loop draws its observations from the *same* trials. This is the
  pre-restructure behaviour and is reproduced bit for bit (the channel is returned
  unchanged).
* ``split_half`` (sensitivity check, audit F8): the valid trials of each (site, EMG)
  are randomly split into a ground-truth half **A** and an observation half **B**.
  The loop only ever sees half B and is scored against the mean of half A, so an
  observed trial can never also be part of the target it is scored against.

Split protocol (decisions of the 2026-09-16 interview):

* per split draw ``r = 1..R`` and per site the valid trials are permuted;
  half A receives ``floor(n/2)`` trials and half B the remaining ones, so an odd
  count gives the extra trial to the **observation** half;
* each draw yields two instances, (A = GT, B = observations) and the cross-fitted
  swap (B = GT, A = observations), i.e. ``2R`` instances per channel;
* sites with fewer than 2 valid trials cannot be split and are dropped (logged,
  counted in ``meta['n_dropped_sites']``);
* the y-scaler is refitted on the **observation half only** (no GT information in
  the scaler) and the GT half is transformed with that same scaler;
* BO repetition ``i`` uses instance ``i mod 2R`` (:func:`instance_for_rep`).

Reliability: Spearman–Brown split-half reliability per channel,
``r_half = corr_sites(mean_A, mean_B)``, ``rel = 2 r_half / (1 + r_half)``, averaged
over split draws. ``r_half`` is the reliability of a *half*-trial mean and therefore
the R² noise ceiling of a split-half-scored run; ``rel`` is the reliability of the
full-trial mean, the ceiling of a full-mean-scored run.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .channels import ChannelData

__all__ = [
    "GT_MODES",
    "DEFAULT_N_SPLITS",
    "SplitMasks",
    "draw_split_masks",
    "build_ground_truth",
    "split_half_instances",
    "ground_truth_instances",
    "instance_for_rep",
    "split_half_reliability",
]

#: Registered ground-truth modes.
GT_MODES: tuple[str, ...] = ("full_mean", "split_half")

#: Default number of split draws R (so 2R = 20 instances per channel).
DEFAULT_N_SPLITS: int = 10

#: Minimum valid trials a site needs to be split into two non-empty halves.
_MIN_TRIALS_FOR_SPLIT: int = 2


@dataclass(frozen=True)
class SplitMasks:
    """One random split of a trial bank into two disjoint halves.

    Attributes:
        half_a: Trials assigned to half A (``floor(n/2)`` per site). Shape [N, R].
        half_b: Trials assigned to half B (the rest; the extra trial when n is odd).
            Shape [N, R].
        kept_sites: Sites with at least two valid trials. Shape [N].
    """

    half_a: np.ndarray    # [N, R] bool
    half_b: np.ndarray    # [N, R] bool
    kept_sites: np.ndarray  # [N] bool


def draw_split_masks(Y_trials: np.ndarray, rng: np.random.Generator) -> SplitMasks:
    """Randomly split each site's valid trials into halves A (floor(n/2)) and B (rest).

    Args:
        Y_trials: Trial bank, NaN at invalid trials. Shape [N, R].
        rng: Generator driving the per-site permutations.

    Returns:
        The two disjoint masks and the sites that could be split. Dropped sites have
        both masks all-False.
    """
    valid = np.isfinite(Y_trials)                        # [N, R]
    n_sites = Y_trials.shape[0]
    half_a = np.zeros_like(valid)
    half_b = np.zeros_like(valid)
    kept = valid.sum(axis=1) >= _MIN_TRIALS_FOR_SPLIT    # [N]
    for site in range(n_sites):
        if not kept[site]:
            continue
        cols = rng.permutation(np.flatnonzero(valid[site]))  # [n_valid]
        n_a = cols.size // 2
        half_a[site, cols[:n_a]] = True
        half_b[site, cols[n_a:]] = True
    return SplitMasks(half_a=half_a, half_b=half_b, kept_sites=kept)


def _masked_mean(Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-row mean of the entries selected by ``mask``.

    Args:
        Y: Values, shape [N, R].
        mask: Selection, shape [N, R]; every row must select at least one entry.

    Returns:
        Row means, shape [N].

    Raises:
        RuntimeError: If a row selects nothing (would silently produce NaN).
    """
    counts = mask.sum(axis=1)                            # [N]
    if (counts == 0).any():
        raise RuntimeError(
            f"_masked_mean: {int((counts == 0).sum())} row(s) have an empty selection."
        )
    return np.where(mask, Y, 0.0).sum(axis=1) / counts   # [N]


def build_ground_truth(
    Y_trials: np.ndarray,
    mode: str,
    rng: np.random.Generator | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Array-level ground truth: list of ``(y_gt, Y_obs_pool)`` instances.

    No rescaling happens here; the channel-level :func:`ground_truth_instances`
    adds the observation-half scaler refit. Dropped sites (split mode) are removed
    from both arrays.

    Args:
        Y_trials: Trial bank in any fixed units, NaN at invalid trials. Shape [N, R].
        mode: ``'full_mean'`` or ``'split_half'``.
        rng: Generator for the split draws (required for ``split_half``).
        n_splits: Number of split draws R; split mode returns ``2R`` instances.

    Returns:
        ``full_mean``: one instance ``(nanmean over trials, Y_trials)``.
        ``split_half``: ``2R`` instances ordered (A→GT, B→obs), (B→GT, A→obs) per draw;
        observation arrays keep shape [N_kept, R] with NaN outside the half.

    Raises:
        ValueError: On an unknown mode, a missing ``rng`` or ``n_splits < 1``.
    """
    if mode not in GT_MODES:
        raise ValueError(f"build_ground_truth: unknown mode {mode!r}; expected one of {GT_MODES}.")
    if mode == "full_mean":
        valid = np.isfinite(Y_trials)
        return [(_masked_mean(np.nan_to_num(Y_trials), valid), Y_trials)]
    if rng is None:
        raise ValueError("build_ground_truth: split_half needs an explicit rng.")
    if n_splits < 1:
        raise ValueError(f"build_ground_truth: n_splits must be >= 1, got {n_splits}.")
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_splits):
        masks = draw_split_masks(Y_trials, rng)
        keep = masks.kept_sites
        Y = Y_trials[keep]                                          # [N_kept, R]
        a, b = masks.half_a[keep], masks.half_b[keep]               # [N_kept, R]
        Y0 = np.nan_to_num(Y)
        out.append((_masked_mean(Y0, a), np.where(b, Y, np.nan)))
        out.append((_masked_mean(Y0, b), np.where(a, Y, np.nan)))
    return out


def _raw_units(channel: ChannelData, values: np.ndarray) -> np.ndarray:
    """Map standardized values back to raw response units with the channel's scaler.

    Args:
        channel: Channel whose ``meta['scaler_y']`` defines the current units.
        values: Values in the channel's units, any shape (NaN preserved).

    Returns:
        Values in raw units (unchanged when the channel carries no scaler).
    """
    scaler = channel.meta.get("scaler_y")
    if scaler is None:
        return np.asarray(values, dtype=np.float64)
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    out = np.full(flat.shape, np.nan)
    finite = np.isfinite(flat[:, 0])
    if finite.any():
        out[finite] = scaler.inverse_transform(flat[finite])
    return out.reshape(np.shape(values))


def _fresh_scaler(channel: ChannelData) -> Any:
    """A new, unfitted scaler of the same kind the channel was preprocessed with.

    Args:
        channel: Source channel.

    Returns:
        ``MinMaxScaler`` when the channel used one, else ``StandardScaler`` (the
        project contract).
    """
    scaler = channel.meta.get("scaler_y")
    return MinMaxScaler() if isinstance(scaler, MinMaxScaler) else StandardScaler()


def _transform(scaler: Any, values: np.ndarray) -> np.ndarray:
    """Apply a fitted scaler elementwise, preserving NaN and shape.

    Args:
        scaler: Fitted scikit-learn scaler (one feature).
        values: Array of any shape.

    Returns:
        Transformed array, same shape.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    out = np.full(flat.shape, np.nan)
    finite = np.isfinite(flat[:, 0])
    if finite.any():
        out[finite] = scaler.transform(flat[finite])
    return out.reshape(np.shape(values))


def split_half_instances(
    channel: ChannelData,
    rng: np.random.Generator,
    n_splits: int = DEFAULT_N_SPLITS,
) -> list[ChannelData]:
    """Build the ``2R`` split-half instances of one channel.

    Each instance is an ordinary :class:`ChannelData` (``gt_mode='split_half'``,
    ``split_id`` in ``[0, 2R)``) whose ``Y_trials`` holds only the observation half
    and whose ``y_gt`` is the mean of the GT half, both standardized with a scaler
    fitted on the observation half's valid trials.

    Args:
        channel: A full-mean channel (as returned by ``load_channel``).
        rng: Generator for the split draws.
        n_splits: Number of split draws R.

    Returns:
        ``2R`` channels, ordered ``[draw0-A, draw0-swap, draw1-A, ...]``.

    Raises:
        ValueError: If the channel is already a split instance or ``n_splits < 1``.
        RuntimeError: If fewer than 2 sites survive, or the GT is degenerate.
    """
    if channel.gt_mode != "full_mean":
        raise ValueError(
            f"split_half_instances: expected a full_mean channel, got gt_mode={channel.gt_mode!r}."
        )
    if n_splits < 1:
        raise ValueError(f"split_half_instances: n_splits must be >= 1, got {n_splits}.")
    raw = _raw_units(channel, channel.Y_trials)                          # [N, R]

    instances: list[ChannelData] = []
    for draw in range(n_splits):
        masks = draw_split_masks(raw, rng)
        keep = masks.kept_sites                                          # [N]
        n_dropped = int((~keep).sum())
        if keep.sum() < 2:
            raise RuntimeError(
                f"split_half_instances({channel.label}): only {int(keep.sum())} site(s) have "
                ">= 2 valid trials; the channel cannot be split."
            )
        if n_dropped and draw == 0:
            print(
                f"[ground_truth] {channel.label}: dropping {n_dropped} site(s) with < 2 valid "
                "trials in split_half mode.",
                file=sys.stderr,
            )
        Y = raw[keep]                                                    # [N_kept, R]
        Y0 = np.nan_to_num(Y)
        halves = (masks.half_a[keep], masks.half_b[keep])
        for swap in (0, 1):
            gt_mask, obs_mask = (halves[0], halves[1]) if swap == 0 else (halves[1], halves[0])
            obs_raw = np.where(obs_mask, Y, np.nan)                      # [N_kept, R]
            gt_raw = _masked_mean(Y0, gt_mask)                           # [N_kept]
            scaler = _fresh_scaler(channel)
            scaler.fit(obs_raw[np.isfinite(obs_raw)].reshape(-1, 1))
            y_gt = _transform(scaler, gt_raw)                            # [N_kept]
            Y_obs = _transform(scaler, obs_raw)                          # [N_kept, R]
            # Per-site stress fields (if a knob was applied first) follow the kept sites.
            per_site: dict[str, Any] = {}
            failure_time = getattr(channel, "failure_time", None)
            if failure_time is not None:
                per_site["failure_time"] = failure_time[keep]
            meta = copy.copy(channel.meta)
            meta.update(
                scaler_y=scaler,
                n_dropped_sites=n_dropped,
                split_draw=draw,
                split_swap=swap,
                site_mask_split=keep.copy(),
            )
            instances.append(
                replace(
                    channel,
                    X_pool=channel.X_pool[keep],
                    Y_trials=Y_obs,
                    y_gt=y_gt,
                    ch2xy=channel.ch2xy[keep],
                    gt_mode="split_half",
                    split_id=2 * draw + swap,
                    meta=meta,
                    **per_site,
                )
            )
    return instances


def ground_truth_instances(
    channel: ChannelData,
    mode: str,
    *,
    rng: np.random.Generator | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
) -> list[ChannelData]:
    """Channel-level dispatcher used by the experiment runners.

    Args:
        channel: A full-mean channel.
        mode: ``'full_mean'`` (returns ``[channel]`` unchanged) or ``'split_half'``.
        rng: Generator for the split draws (required for ``split_half``).
        n_splits: Number of split draws R.

    Returns:
        The instances; repetition ``i`` uses ``instances[instance_for_rep(i, len)]``.

    Raises:
        ValueError: On an unknown mode or a missing ``rng``.
    """
    if mode not in GT_MODES:
        raise ValueError(f"ground_truth_instances: unknown mode {mode!r}; expected one of {GT_MODES}.")
    if mode == "full_mean":
        return [channel]
    if rng is None:
        raise ValueError("ground_truth_instances: split_half needs an explicit rng.")
    return split_half_instances(channel, rng, n_splits)


def instance_for_rep(rep: int, n_instances: int) -> int:
    """Instance index used by BO repetition ``rep`` (``rep mod 2R``).

    Args:
        rep: Repetition index (>= 0).
        n_instances: Number of available instances (>= 1).

    Returns:
        ``rep % n_instances``.
    """
    if n_instances < 1:
        raise ValueError(f"instance_for_rep: n_instances must be >= 1, got {n_instances}.")
    return int(rep) % int(n_instances)


def split_half_reliability(
    Y_trials: np.ndarray,
    rng: np.random.Generator,
    n_splits: int = DEFAULT_N_SPLITS,
) -> dict[str, float]:
    """Spearman–Brown split-half reliability of a channel's site means.

    Implements the P0.7 reliability: per draw ``r = corr_sites(mean_A, mean_B)`` over
    sites with >= 2 valid trials, ``rel = 2r / (1 + r)``; both averaged over draws.
    Pearson correlation is invariant to affine rescaling, so the units of
    ``Y_trials`` do not matter.

    Args:
        Y_trials: Trial bank, NaN at invalid trials. Shape [N, R].
        rng: Generator for the split draws.
        n_splits: Number of split draws R.

    Returns:
        ``{'r_half': ..., 'spearman_brown': ...}``. ``r_half`` is the R² noise
        ceiling of a split-half-scored run, ``spearman_brown`` that of a full-mean one.

    Raises:
        RuntimeError: If fewer than 3 sites can be split or a half-mean vector is
            constant (correlation undefined).
    """
    rs: list[float] = []
    for _ in range(n_splits):
        masks = draw_split_masks(Y_trials, rng)
        keep = masks.kept_sites
        if keep.sum() < 3:
            raise RuntimeError(
                f"split_half_reliability: only {int(keep.sum())} splittable site(s); need >= 3."
            )
        Y0 = np.nan_to_num(Y_trials[keep])
        m_a = _masked_mean(Y0, masks.half_a[keep])      # [N_kept]
        m_b = _masked_mean(Y0, masks.half_b[keep])      # [N_kept]
        if np.std(m_a) == 0 or np.std(m_b) == 0:
            raise RuntimeError("split_half_reliability: constant half-mean vector; correlation undefined.")
        rs.append(float(np.corrcoef(m_a, m_b)[0, 1]))
    r = np.asarray(rs)
    # Spearman-Brown is undefined at r = -1 (fail fast rather than return -inf).
    if (r <= -1.0).any():
        raise RuntimeError(f"split_half_reliability: r_half = -1 in some draw ({rs}).")
    sb = 2.0 * r / (1.0 + r)
    return {"r_half": float(r.mean()), "spearman_brown": float(sb.mean())}
