"""One Bayesian-optimization run on one channel, fully instrumented.

Drives the native :mod:`pfns4neurostim.evaluation.bo_loop` with a registered
acquisition type (P0.2) and turns its trajectory into the tidy metrics of
:mod:`pfns4neurostim.evaluation.metrics`.

Budget semantics (**P0.3**): ``budget`` is the total number of queries including
``n_init``, stated as an explicit iteration count and allowed to be below the
grid size.

As of task #1 Steps 4-5 the legacy BO-loop seam is gone: the loop, the
acquisition functions and the surrogate interface are all package-native.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..acquisition.registry import build_acquisition
from ..data.channels import ChannelData
from ..models.registry import MODEL_REGISTRY, build_surrogate, model_version
from . import metrics as _metrics
from .bo_loop import run_bo_loop

__all__ = ["BOResult", "run_channel_bo"]


@dataclass
class BOResult:
    """Outcome of one BO run on one channel.

    Attributes:
        row: Scalar metrics and the keys the tidy schema needs from this level.
        trajectory: Per-step data for the companion pickle.
    """

    row: dict[str, Any]
    trajectory: dict[str, Any] = field(default_factory=dict)


def run_channel_bo(
    model: str,
    channel: ChannelData,
    *,
    acq_fn: str = "ei",
    acq_params: dict[str, Any] | None = None,
    acq_schedules: dict[str, Any] | None = None,
    budget: int = 50,
    n_init: int = 5,
    seed: int = 42,
    device: str = "cpu",
    model_params: dict[str, Any] | None = None,
    with_calibration: bool = True,
) -> BOResult:
    """Run one BO repetition and compute every tidy metric for it.

    Args:
        model: Registered model key (e.g. ``'tabpfn_v2_5'``).
        channel: The channel to optimize over, already stressed if applicable.
        acq_fn: Registered acquisition type (``ei``, ``ucb``, ``pi``,
            ``ts_marginal``, ``ts_joint``, ``greedy``, ``random``, ``native``).
        acq_params: Parameters for that type; unknown keys raise (P0.2).
        acq_schedules: Optional per-parameter schedules.
        budget: Total queries including ``n_init`` (P0.3).
        n_init: Number of random initial queries.
        seed: Seed for this repetition.
        device: Default torch device string.
        model_params: Extra constructor parameters for the surrogate. A ``device`` key
            overrides ``device`` for this model only (e.g. GP on CPU, TabPFN on CUDA).
        with_calibration: Compute coverage/ECE/NLL/CRPS from the final posterior.
            Ignored for models without a predictive distribution (random search).

    Returns:
        A :class:`BOResult`.

    Raises:
        KeyError: On an unknown model or acquisition type.
        ValueError: On invalid budget or acquisition parameters.
        NotImplementedError: For ``ts_joint`` with a surrogate that has no joint
            posterior (every PFN).
    """
    from ..seeding import set_seed  # local import keeps torch out of module import

    set_seed(seed)
    rng = np.random.default_rng(seed)

    spec, params = build_acquisition(acq_fn, acq_params, acq_schedules)
    extra = dict(model_params or {})
    device = extra.pop("device", None) or device  # per-model override (P0.12)
    surrogate = build_surrogate(model, device=device, **extra)

    t0 = time.time()
    traj = run_bo_loop(
        surrogate, channel, spec, params, budget=budget, n_init=n_init, rng=rng
    )
    total_time = time.time() - t0

    recommended = traj.recommended_index
    step_times = traj.step_times_s

    row: dict[str, Any] = {
        "model": model,
        "model_version": model_version(model),
        "acq_type": acq_fn,
        "budget": int(budget),
        "n_init": int(n_init),
        "seed": int(seed),
        "device": device,
        "n_sites": channel.n_sites,
        "n_dims": channel.n_dims,
        "total_time_s": float(total_time),
        "mean_query_latency_s": float(np.mean(step_times)) if step_times else float("nan"),
        "median_query_latency_s": float(np.median(step_times)) if step_times else float("nan"),
    }
    # Every score is taken over the electrodes alive at the end of the run (all of
    # them unless the K6 failure knob is active); a failed electrode reads
    # ``failure_value``, so recommending it costs regret like any poor site.
    alive = channel.survivors                                                       # [N] bool
    y_end = channel.y_end                                                           # [N]
    row.update(
        _metrics.regret_metrics(
            y_end, traj.observed_indices, recommended,
            reference=alive, queried_values=traj.real_values,
        )
    )
    row.update(_metrics.surrogate_accuracy(y_end[alive], traj.y_pred[alive]))
    row.update(
        _metrics.identification_metrics(np.where(alive, y_end, -np.inf), recommended, channel.ch2xy)
    )
    if "decoy_basin" in channel.meta:
        # K1: did the final recommendation land on the decoy's side of the map?
        row["decoy_capture"] = float(channel.meta["decoy_basin"][recommended])

    # Per-step curves along the BO run (post-processing reads these, never re-runs).
    y_star = float(np.max(y_end[alive]))
    y_range = float(np.ptp(y_end[alive]))
    recs = np.asarray(traj.recommendations, dtype=int)                              # [T+1]
    regret_per_step = (y_star - y_end[recs]) / y_range                              # [T+1]
    y_raw = channel.to_raw(y_end)
    exploration_per_step = None
    if y_raw is not None:
        exploration_per_step = [
            _metrics.exploration_score(y_raw, int(i), reference=alive) for i in recs
        ]                                                                           # [T+1]
        row["exploration_score"] = float(exploration_per_step[-1])

    # Random search has no predictive distribution, so coverage/ECE/NLL/CRPS are
    # undefined for it and are reported as "not computed" rather than invented.
    if with_calibration and MODEL_REGISTRY[model].has_predictive_distribution:
        row.update(
            _metrics.calibration_metrics(
                y_end[alive], traj.y_pred[alive], traj.y_std[alive], gt_range=y_range
            )
        )

    trajectory = {
        "observed_indices": list(traj.observed_indices),
        "observed_values": list(traj.observed_values),
        "real_values": list(traj.real_values),
        "best_rec_indices": list(traj.recommendations),
        "step_times_s": list(step_times),
        "r2_per_step": list(traj.r2_per_step),
        "recommended_regret_per_step": regret_per_step.tolist(),
        "exploration_per_step": exploration_per_step,
        "acq_params": list(traj.acq_params),
        "y_pred": np.asarray(traj.y_pred, dtype=np.float64),
        "y_std": np.asarray(traj.y_std, dtype=np.float64),
        "y_gt": np.asarray(y_end, dtype=np.float64),
        "survivors": alive,
        "gt_range": y_range,
    }
    return BOResult(row=row, trajectory=trajectory)
