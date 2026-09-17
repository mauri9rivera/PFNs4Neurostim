"""Minimal BO loop recording acquisition metadata and all three regrets.

Phase-3 target: ``src/pfns4neurostim/evaluation/bo_loop.py``.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .metrics import regret_curves
from .preprocessing import SharedData, draw_trial
from .protocol import Acquisition, BOState, Surrogate


def run_bo(
    surrogate: Surrogate,
    acquisition: Acquisition,
    data: SharedData,
    n_init: int,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    """Run one BO trajectory on a discrete pool with single-trial noisy observations.

    Args:
        surrogate: Fresh surrogate.
        acquisition: Acquisition callable.
        data: Problem in shared units.
        n_init: Random initial queries (without replacement).
        budget: Total queries including ``n_init`` (explicit iteration count).
        seed: Seed of the run's ``np.random.Generator``.

    Returns:
        Dict with ``queried_idx`` [budget], ``observed_y`` [budget], ``recommended_idx``
        [budget - n_init + 1], ``step_time_s`` [budget - n_init], ``regret`` (three curves)
        and ``metadata`` (acq_fn + all params, surrogate, n_init, budget, seed).
    """
    if budget <= n_init:
        raise ValueError(f"budget ({budget}) must exceed n_init ({n_init}).")
    rng = np.random.default_rng(seed)
    M = data.X.shape[0]
    q = list(rng.choice(M, size=n_init, replace=False))
    y = [draw_trial(data, i, rng) for i in q]
    rec: list[int] = []
    times: list[float] = []
    n_steps = budget - n_init
    for t in range(n_steps + 1):
        idx = np.asarray(q)
        surrogate.fit(data.X[idx], np.asarray(y))
        rec.append(int(np.argmax(surrogate.predict_marginals(data.X, "latent" if surrogate.has_exact_joint else "predictive").mean)))
        if t == n_steps:
            break
        t0 = time.perf_counter()
        state = BOState(X_obs=data.X[idx], y_obs=np.asarray(y), obs_idx=idx, t=t, n_steps=n_steps)
        res = acquisition(surrogate, data.X, state, rng)
        times.append(time.perf_counter() - t0)
        q.append(res.index)
        y.append(draw_trial(data, res.index, rng))
    queried = np.asarray(q)
    return {
        "queried_idx": queried,
        "observed_y": np.asarray(y),
        "recommended_idx": np.asarray(rec),
        "step_time_s": np.asarray(times),
        "regret": regret_curves(data.y_gt, queried, np.asarray(rec)),
        "metadata": {
            "acq_fn": acquisition.name,
            "acq_params": acquisition.params(),
            "surrogate": surrogate.name,
            "n_init": n_init,
            "budget": budget,
            "seed": seed,
            "problem": data.name,
            "units": "shared: X minmax[0,1], y z-score (valid trials); regrets / GT range",
        },
    }
