"""One context draw, shared by every Hyp C analysis (task plan #19 Step 1).

A "context" is the evidence a surrogate is conditioned on: ``t`` sites of a channel and one observed trial at
each. All three mechanism analyses -- the update-rule probe (M10), CKA (a) and (b), and the MMD / sliced-W2
placement -- need one, and until 2026-09-30 each drew its own from its own seed stream
(``'update_rule'``, ``'cka'``, ``'cka_placement'``, ``'placement_c'``). Two things followed from that, both
bad:

* **Nothing could be shared.** Everything that depends on the *sites alone* -- the electrode-geometry kernel,
  the embeddings of the prior/noise reference maps, and hence the placement floor and ceiling -- had to be
  recomputed for every channel, because no two channels (let alone two analyses) saw the same sites.
* **The analyses did not describe the same operating points**, so M8's predictive link joined per-channel
  metrics that came from different contexts and called the result a relationship.

The contract here splits the draw in two, which is what makes sharing sound rather than convenient:

``sites``   keyed by **(grid, t, draw)** -- every channel on the same electrode grid, at every stress level,
            sees the *same* sites. The design becomes paired: a difference between two channels, or between
            two levels, can no longer come from having been shown different electrodes.
``trials``  keyed by **(channel, level, t, draw)** -- each channel still draws its own noisy observation at
            each of those sites, which is the part that genuinely differs.

A grid is identified by a hash of its coordinate array rather than by a dataset name, so two datasets that
happen to share a layout share their site sets, and one dataset whose subjects have different condition sets
(``5d_rat``) correctly gets one stream per layout.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..data.channels import ChannelData
from ..seeding import rng_for
from .update_rule import Context, draw_context

__all__ = ["CONTEXT_SITE_STREAM", "CONTEXT_TRIAL_STREAM", "context_for", "context_sites", "grid_id"]

#: Seed-stream names. Distinct constants, so a future change to one cannot silently move the other.
CONTEXT_SITE_STREAM: str = "mechanism_context_sites"
CONTEXT_TRIAL_STREAM: str = "mechanism_context_trials"

#: Characters of the coordinate hash kept as a grid identifier. 12 hex characters is 48 bits: collision-free
#: for the handful of layouts a study holds, and short enough to read in a log line or a CSV column.
GRID_ID_CHARS: int = 12


def grid_id(X_pool: np.ndarray) -> str:
    """Stable identifier of an electrode layout.

    Args:
        X_pool: Candidate coordinates, shape [N, D].

    Returns:
        The first :data:`GRID_ID_CHARS` hex characters of a SHA-1 over the array's bytes.
    """
    array = np.ascontiguousarray(np.asarray(X_pool, dtype=np.float64))
    return hashlib.sha1(array.tobytes()).hexdigest()[:GRID_ID_CHARS]


def context_sites(
    X_pool: np.ndarray,
    t: int,
    draw: int,
    *,
    base_seed: int,
) -> np.ndarray:
    """Site indices of one context draw, shared by every channel on this grid.

    Deliberately independent of the channel and of the stress level: see the module docstring.

    Args:
        X_pool: Candidate coordinates of the grid, shape [N, D].
        t: Context size; clamped by the caller, not here, so an out-of-range request fails loudly.
        draw: Context-draw index.
        base_seed: Experiment-wide base seed.

    Returns:
        Sorted, distinct site indices, shape [t].

    Raises:
        ValueError: If ``t`` is not in ``[2, N]``.
    """
    n_sites = int(np.asarray(X_pool).shape[0])
    if not 2 <= t <= n_sites:
        raise ValueError(f"context_sites: t={t} must be in [2, {n_sites}] for this grid.")
    rng = rng_for(grid_id(X_pool), CONTEXT_SITE_STREAM, int(t), int(draw), base_seed=base_seed)
    return np.sort(rng.choice(n_sites, size=int(t), replace=False))          # [t]


def context_for(
    channel: ChannelData,
    t: int,
    draw: int,
    *,
    base_seed: int,
    level: float | None = None,
) -> Context:
    """The context one analysis cell is conditioned on: shared sites, this channel's own trials.

    Args:
        channel: Channel (already stressed, when a knob is applied).
        t: Context size.
        draw: Context-draw index.
        base_seed: Experiment-wide base seed.
        level: Stress level of the cell, part of the trial stream's key so that two levels of the same
            channel draw different trials at the same sites. ``None`` for an unstressed cell.

    Returns:
        The context, with ``sites`` from :func:`context_sites` and one valid trial drawn per site.
    """
    sites = context_sites(channel.X_pool, t, draw, base_seed=base_seed)      # [t]
    rng = rng_for(
        channel.label, CONTEXT_TRIAL_STREAM, level, int(t), int(draw), base_seed=base_seed
    )
    return draw_context(channel, int(t), rng, sites=sites)
