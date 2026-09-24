"""Model- and acquisition-agnostic Bayesian-optimization loop (task #1 Step 5).

Replaces the legacy ``utils.bo_loops.run_bo_loop`` seam. The loop knows nothing
about which surrogate or acquisition it is running: it refits the surrogate on
the observations so far, asks the acquisition registry to score the pool, queries
the selected site by drawing **one noisy trial** from that site's trial bank (the
real experimental protocol — the optimizer never sees the ground truth), and
records the pure-exploitation recommendation at every step.

Budget semantics (**P0.3**): ``budget`` is the total number of queries including
the ``n_init`` random initial ones. It may be below **or above** the pool size:
sites can be re-queried (each query is a fresh noisy trial), so the optimizer can
concentrate its budget on promising configurations and narrow its posterior there.

Differences from the legacy loop, all deliberate:

* Selection uses a **random tie-break** (D6) instead of the lowest index.
* Re-querying an observed site is allowed, exactly as in the legacy loop. An
  intermediate version (2026-09-17 to 2026-09-21) masked observed sites; that
  turned budget = pool size into an exhaustive scan and removed the exploitation
  that noise averaging makes possible, so it was removed.
* Every RNG draw comes from an explicit :class:`numpy.random.Generator` (D8);
  nothing touches the global NumPy or torch RNG state mid-loop.
* Acquisition parameters actually used at each step are recorded (P0.2), so a
  schedule is visible in the output rather than inferred from the config.
* A site whose trials are all NaN raises instead of silently contributing a zero.
* ``channel.failure_time`` (the K6 failure knob) makes an electrode return
  ``channel.failure_value`` once the run has progressed past its failure time.
  The electrode stays queryable and recommendable: the optimizer is not told and
  must notice from the data. Per-step R-squared is scored on survivors.
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
from .metrics import r2_score

__all__ = ["BOTrajectory", "run_bo_loop", "draw_trial"]


@dataclass
class BOTrajectory:
    """Per-step record of one BO run.

    Attributes:
        observed_indices: Queried site indices, in order, length ``budget``.
        observed_values: The noisy value observed at each query.
        real_values: The true value of each query at the time it was made (the
            ground truth, or the failure value once the electrode had failed).
        recommendations: Pure-exploitation recommendation after each acquisition
            step (argmax of the predictive mean over the whole pool).
        r2_per_step: R-squared of the surrogate's predictive mean over the whole
            pool after each acquisition step (fit on the observations so far), plus
            a final entry after the last observation; same length as
            ``recommendations``.
        step_times_s: Wall-clock seconds per acquisition step (fit + score + readout).
        acq_params: Resolved acquisition parameters at each step.
        y_pred: Final predictive mean over the pool, shape [N].
        y_std: Final predictive standard deviation over the pool, shape [N].
    """

    observed_indices: list[int] = field(default_factory=list)
    observed_values: list[float] = field(default_factory=list)
    real_values: list[float] = field(default_factory=list)
    recommendations: list[int] = field(default_factory=list)
    r2_per_step: list[float] = field(default_factory=list)
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
        ValueError: If the budget is inconsistent with ``n_init`` or ``n_init`` exceeds the pool.
        NotImplementedError: If the acquisition needs a capability the model does
            not have: a joint posterior (``ts_joint`` with a PFN) or a native policy
            (``native`` with a plain surrogate).
    """
    n_sites = channel.n_sites
    if budget <= n_init:
        raise ValueError(
            f"budget ({budget}) must exceed n_init ({n_init}); budget counts total "
            "queries including the initial design (P0.3)."
        )
    if n_init > n_sites:
        raise ValueError(
            f"n_init ({n_init}) exceeds the {n_sites} site(s) of {channel.label}; "
            "the initial design is drawn without replacement."
        )
    if spec.needs_joint and not getattr(surrogate, "supports_joint", False):
        raise NotImplementedError(
            f"Acquisition {spec.name!r} needs a joint posterior, which "
            f"{getattr(surrogate, 'key', surrogate)!r} does not provide."
        )
    if spec.needs_native_policy and not getattr(surrogate, "has_native_policy", False):
        raise NotImplementedError(
            f"Acquisition {spec.name!r} needs a model that owns its query decision, which "
            f"{getattr(surrogate, 'key', surrogate)!r} does not."
        )

    traj = BOTrajectory()
    X_pool = channel.X_pool                                  # [N, D]
    survivors = channel.survivors                            # [N] bool

    def observe(index: int) -> None:
        """Query ``index`` as the next of the ``budget`` queries and record it."""
        progress = len(traj.observed_indices) / budget
        if channel.is_failed(index, progress):
            observed = real = channel.failure_value
        else:
            observed = draw_trial(channel.Y_trials, index, rng)
            real = float(channel.y_gt[index])
        traj.observed_indices.append(index)
        traj.observed_values.append(observed)
        traj.real_values.append(real)

    # --- initial design: uniform random sites without replacement ------------
    for index in rng.choice(n_sites, size=n_init, replace=False):
        observe(int(index))

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

        # Pure-exploitation recommendation over the whole pool.
        pool_mean, _ = marginals(surrogate, X_pool)           # [N]
        traj.recommendations.append(int(np.argmax(pool_mean)))
        traj.r2_per_step.append(r2_score(channel.y_gt[survivors], pool_mean[survivors]))

        observe(result.index)
        traj.acq_params.append(result.params)
        traj.step_times_s.append(time.time() - t0)

    # --- final refit and prediction on the full pool -------------------------
    surrogate.fit(X_pool[np.asarray(traj.observed_indices, dtype=int)], np.asarray(traj.observed_values))
    mean, std = marginals(surrogate, X_pool)                  # [N], [N]
    traj.y_pred = np.asarray(mean, dtype=np.float64)
    traj.y_std = np.asarray(std, dtype=np.float64)
    traj.recommendations.append(int(np.argmax(traj.y_pred)))
    traj.r2_per_step.append(r2_score(channel.y_gt[survivors], traj.y_pred[survivors]))
    return traj
