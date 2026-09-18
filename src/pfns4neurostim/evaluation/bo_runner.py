"""One Bayesian-optimization run on one channel, fully instrumented.

Wraps the model- and acquisition-agnostic loop and turns its trajectory into the
tidy metrics of :mod:`pfns4neurostim.evaluation.metrics`. Budget semantics follow
**P0.3**: ``budget`` is the total number of queries *including* ``n_init``, stated
as an explicit iteration count and allowed to be below the grid size.

**Migration seam.** The loop itself is still ``utils.bo_loops.run_bo_loop``;
task #1 Step 5 replaces it with ``evaluation/bo_loop.py`` and only
:func:`_run_loop` changes here.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.channels import ChannelData
from ..models.registry import build_surrogate, model_version
from . import metrics as _metrics

__all__ = ["BOResult", "run_channel_bo"]

#: Acquisition functions the legacy loop understands.
_SUPPORTED_ACQ: frozenset[str] = frozenset({"ei", "ucb", "ts_marginal"})


@dataclass
class BOResult:
    """Outcome of one BO run on one channel.

    Attributes:
        row: Scalar metrics and keys for the tidy CSV.
        trajectory: Per-step arrays (queried indices, observed values,
            ground-truth values, recommendations, step times) for the companion
            pickle.
    """

    row: dict[str, Any]
    trajectory: dict[str, Any] = field(default_factory=dict)


def _ensure_legacy_on_path() -> None:
    """Put the flat ``src/`` tree on ``sys.path`` so ``utils.*`` imports resolve."""
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _run_loop(
    surrogate: Any,
    channel: ChannelData,
    *,
    acq_fn: str,
    budget: int,
    n_init: int,
    kappa_schedule: float,
    ts_temperature: float,
) -> dict[str, Any]:
    """Execute the BO loop through the legacy implementation.

    Args:
        surrogate: Fitted-on-demand surrogate.
        channel: The (possibly stressed) channel.
        acq_fn: ``'ei'``, ``'ucb'`` or ``'ts_marginal'``.
        budget: Total queries including ``n_init`` (P0.3).
        n_init: Random initial queries.
        kappa_schedule: UCB kappa (0.0 selects the auto-annealed schedule);
            inert unless ``acq_fn='ucb'``.
        ts_temperature: Thompson-sampling temperature; inert unless TS.

    Returns:
        The raw loop dictionary.
    """
    _ensure_legacy_on_path()
    from utils.bo_loops import run_bo_loop  # noqa: PLC0415 - seam, intentional

    # The legacy loop names marginal Thompson sampling 'ts'; the package uses the
    # unambiguous 'ts_marginal' everywhere (task #4 decision, 2026-09-17).
    legacy_acq = "ts" if acq_fn == "ts_marginal" else acq_fn
    return run_bo_loop(
        surrogate,
        channel.X_pool,
        channel.Y_trials,
        channel.X_pool,
        channel.y_gt,
        n_init=n_init,
        budget=budget,
        kappa_schedule=kappa_schedule,
        acq_fn=legacy_acq,
        ts_temperature=ts_temperature,
    )


def run_channel_bo(
    model: str,
    channel: ChannelData,
    *,
    acq_fn: str = "ei",
    budget: int = 50,
    n_init: int = 5,
    seed: int = 42,
    device: str = "cpu",
    model_params: dict[str, Any] | None = None,
    kappa_schedule: float = 0.0,
    ts_temperature: float = 1.0,
    with_calibration: bool = True,
) -> BOResult:
    """Run one BO repetition and compute every tidy metric for it.

    Args:
        model: Registered model key (e.g. ``'tabpfn_v2_5'``).
        channel: The channel to optimize over, already stressed if applicable.
        acq_fn: Acquisition function; one of ``ei``, ``ucb``, ``ts_marginal``.
        budget: Total queries including ``n_init`` (P0.3).
        n_init: Number of random initial queries.
        seed: Seed for this repetition; set on all three RNGs.
        device: Torch device string.
        model_params: Extra constructor parameters for the surrogate.
        kappa_schedule: UCB kappa; inert unless ``acq_fn='ucb'``.
        ts_temperature: TS temperature; inert unless ``acq_fn='ts_marginal'``.
        with_calibration: Compute coverage/ECE/NLL/CRPS from a final refit.

    Returns:
        A :class:`BOResult` whose ``row`` carries the metrics and the keys the
        tidy schema needs from this level of the stack.

    Raises:
        ValueError: On an unsupported acquisition function or an invalid budget.
    """
    if acq_fn not in _SUPPORTED_ACQ:
        raise ValueError(
            f"run_channel_bo: unsupported acq_fn {acq_fn!r}; supported: {sorted(_SUPPORTED_ACQ)}."
        )
    if budget <= n_init:
        raise ValueError(
            f"run_channel_bo: budget ({budget}) must exceed n_init ({n_init}); "
            "budget counts total queries including the initial design (P0.3)."
        )
    if budget > channel.n_sites:
        raise ValueError(
            f"run_channel_bo: budget ({budget}) exceeds the {channel.n_sites} sites of "
            f"{channel.label}; queries are without replacement."
        )

    from ..seeding import set_seed  # local import keeps torch out of module import

    set_seed(seed)
    surrogate = build_surrogate(model, device=device, **(model_params or {}))

    t0 = time.time()
    loop = _run_loop(
        surrogate,
        channel,
        acq_fn=acq_fn,
        budget=budget,
        n_init=n_init,
        kappa_schedule=kappa_schedule,
        ts_temperature=ts_temperature,
    )
    total_time = time.time() - t0

    observed_indices = [int(i) for i in loop["observed_indices"]]
    best_rec = [int(i) for i in loop["best_rec_indices"]]
    recommended = best_rec[-1] if best_rec else int(np.argmax(np.asarray(loop["y_pred"])))
    step_times = [float(t) for t in loop["times"]]

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
    row.update(_metrics.regret_metrics(channel.y_gt, observed_indices, recommended))
    row.update(_metrics.surrogate_accuracy(channel.y_gt, np.asarray(loop["y_pred"])))
    row.update(_metrics.identification_metrics(channel.y_gt, recommended, channel.ch2xy))

    if with_calibration:
        # The loop leaves the surrogate fitted on all observations; predict once
        # more to obtain the predictive spread the loop does not return.
        mu, sigma = surrogate.predict(channel.X_pool)      # [N], [N]
        row.update(
            _metrics.calibration_metrics(
                channel.y_gt,
                np.asarray(mu, dtype=np.float64),
                np.asarray(sigma, dtype=np.float64),
                gt_range=channel.gt_range,
            )
        )

    trajectory = {
        "observed_indices": observed_indices,
        "observed_values": [float(v) for v in loop["observed_values"]],
        "real_values": [float(v) for v in loop["real_values"]],
        "best_rec_indices": best_rec,
        "step_times_s": step_times,
        "y_pred": np.asarray(loop["y_pred"], dtype=np.float64),
        "y_gt": np.asarray(channel.y_gt, dtype=np.float64),
        "gt_range": channel.gt_range,
    }
    return BOResult(row=row, trajectory=trajectory)
