"""Model- and acquisition-agnostic Bayesian-optimization loop (task #1 Step 5).

Replaces the legacy ``utils.bo_loops.run_bo_loop`` seam. The loop knows nothing
about which surrogate or acquisition it is running: it refits the surrogate on
the observations so far, asks the acquisition registry to score the pool, queries
the selected site by drawing **one noisy trial** from that site's trial bank (the
real experimental protocol — the optimizer never sees the ground truth), and
records the pure-exploitation recommendation at every step.

Budget semantics (**P0.3**): ``budget`` is the total number of queries including
the ``n_init`` random initial ones, and may be below the pool size.

Differences from the legacy loop, all deliberate:

* Selection uses a **random tie-break** (D6) instead of the lowest index.
* Every RNG draw comes from an explicit :class:`numpy.random.Generator` (D8);
  nothing touches the global NumPy or torch RNG state mid-loop.
* Acquisition parameters actually used at each step are recorded (P0.2), so a
  schedule is visible in the output rather than inferred from the config.
* A site whose trials are all NaN raises instead of silently contributing a zero.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..acquisition.base import BOState, acquire
from ..acquisition.registry import AcqParams, AcquisitionSpec
from ..data.channels import ChannelData
from ..models.protocol import marginals

__all__ = ["BOTrajectory", "run_bo_loop", "draw_trial"]


@dataclass
class BOTrajectory:
    """Per-step record of one BO run.

    Attributes:
        observed_indices: Queried site indices, in order, length ``budget``.
        observed_values: The noisy value observed at each query.
        real_values: The ground-truth value of each queried site (for regret).
        recommendations: Pure-exploitation recommendation after each acquisition
            step (argmax of the predictive mean over the whole pool).
        step_times_s: Wall-clock seconds per acquisition step (fit + score + readout).
        acq_params: Resolved acquisition parameters at each step.
        y_pred: Final predictive mean over the pool, shape [N].
        y_std: Final predictive standard deviation over the pool, shape [N].
    """

    observed_indices: list[int] = field(default_factory=list)
    observed_values: list[float] = field(default_factory=list)
    real_values: list[float] = field(default_factory=list)
    recommendations: list[int] = field(default_factory=list)
    step_times_s: list[float] = field(default_factory=list)
    acq_params: list[dict[str, float]] = field(default_factory=list)
    y_pred: np.ndarray | None = None
    y_std: np.ndarray | None = None

    @property
    def recommended_index(self) -> int:
        """The final recommendation, or the argmax of the final prediction."""
        if self.recommendations:
            return int(self.recommendations[-1])
        if self.y_pred is None:
            raise RuntimeError("BOTrajectory has neither recommendations nor a final prediction.")
        return int(np.argmax(self.y_pred))


def draw_trial(Y_trials: np.ndarray, index: int, rng: np.random.Generator) -> float:
    """Draw one valid noisy trial from a site's trial bank.

    Args:
        Y_trials: Trial bank, shape [N, R], NaN where the trial was flagged invalid.
        index: Site to query.
        rng: Seeded generator.

    Returns:
        One observed response.

    Raises:
        RuntimeError: If the site has no valid trial — that site should have been
            dropped during preprocessing, so reaching here is a data bug, not a
            case to paper over with a zero.
    """
    row = np.asarray(Y_trials[index], dtype=np.float64)   # [R]
    valid = np.flatnonzero(np.isfinite(row))
    if valid.size == 0:
        raise RuntimeError(
            f"draw_trial: site {index} has no valid trial; such sites must be dropped "
            "during preprocessing rather than observed as zero."
        )
    return float(row[rng.choice(valid)])


def run_bo_loop(
    surrogate: Any,
    channel: ChannelData,
    spec: AcquisitionSpec,
    params: AcqParams,
    *,
    budget: int,
    n_init: int,
    rng: np.random.Generator,
) -> BOTrajectory:
    """Run one Bayesian-optimization repetition over a channel's candidate pool.

    Args:
        surrogate: Surrogate adapter (see ``models.protocol.SurrogateAdapter``).
        channel: The channel to optimize over, already stressed if applicable.
        spec: Acquisition registry entry.
        params: That type's parameter instance.
        budget: Total queries including ``n_init`` (P0.3).
        n_init: Random initial queries.
        rng: Seeded generator; the only source of randomness in the loop.

    Returns:
        The :class:`BOTrajectory`.

    Raises:
        ValueError: If the budget is inconsistent with ``n_init`` or the pool size.
        NotImplementedError: If the acquisition needs a joint posterior the
            surrogate does not have (``ts_joint`` with a PFN).
    """
    n_sites = channel.n_sites
    if budget <= n_init:
        raise ValueError(
            f"budget ({budget}) must exceed n_init ({n_init}); budget counts total "
            "queries including the initial design (P0.3)."
        )
    if budget > n_sites:
        raise ValueError(
            f"budget ({budget}) exceeds the {n_sites} sites of {channel.label}; "
            "queries are drawn without replacement."
        )
    if spec.needs_joint and not getattr(surrogate, "supports_joint", False):
        raise NotImplementedError(
            f"Acquisition {spec.name!r} needs a joint posterior, which "
            f"{getattr(surrogate, 'key', surrogate)!r} does not provide."
        )

    traj = BOTrajectory()
    X_pool = channel.X_pool                                  # [N, D]

    # --- initial design: uniform random sites without replacement ------------
    initial = rng.choice(n_sites, size=n_init, replace=False)
    for index in initial:
        traj.observed_indices.append(int(index))
        traj.observed_values.append(draw_trial(channel.Y_trials, int(index), rng))
        traj.real_values.append(float(channel.y_gt[int(index)]))

    n_steps = budget - n_init
    for step in range(n_steps):
        t0 = time.time()
        state = BOState(
            observed_indices=tuple(traj.observed_indices),
            observed_values=tuple(traj.observed_values),
            step=step,
            n_steps=n_steps,
            n_dims=channel.n_dims,
        )
        surrogate.fit(X_pool[np.asarray(traj.observed_indices, dtype=int)], np.asarray(traj.observed_values))
        result = acquire(spec.score_fn, surrogate, X_pool, state, rng, params)

        # Pure-exploitation recommendation: what the model would advise right now.
        pool_mean, _ = marginals(surrogate, X_pool)           # [N]
        traj.recommendations.append(int(np.argmax(pool_mean)))

        index = result.index
        traj.observed_indices.append(index)
        traj.observed_values.append(draw_trial(channel.Y_trials, index, rng))
        traj.real_values.append(float(channel.y_gt[index]))
        traj.acq_params.append(result.params)
        traj.step_times_s.append(time.time() - t0)

    # --- final refit and prediction on the full pool -------------------------
    surrogate.fit(X_pool[np.asarray(traj.observed_indices, dtype=int)], np.asarray(traj.observed_values))
    mean, std = marginals(surrogate, X_pool)                  # [N], [N]
    traj.y_pred = np.asarray(mean, dtype=np.float64)
    traj.y_std = np.asarray(std, dtype=np.float64)
    traj.recommendations.append(int(np.argmax(traj.y_pred)))
    return traj
