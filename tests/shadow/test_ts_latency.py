"""Runtime sanity (#10) and PFN-vs-GP per-step latency per acquisition (report-only, decision 3)."""
from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np
import pytest
import torch

from _shadow_utils import report
from _ts_prototype.acquisitions import EI, UCB
from _ts_prototype.gp import ExactGPSurrogate
from _ts_prototype.preprocessing import SharedData, draw_trial
from _ts_prototype.protocol import BOState
from _ts_prototype.tabpfn_surrogate import TabPFNJointSurrogate
from _ts_prototype.thompson import JointTSConfig, TSJoint, TSMarginal

MAX_STEP_S = float(os.environ.get("SHADOW_TS_MAX_STEP_S", "10.0"))

SETTINGS = (("nhp", 1, 0, 50), ("5d_rat", 1, 0, 150))


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _step_time(make_fit: Callable[[], object], acq: object, d: SharedData, idx: np.ndarray,
               y: np.ndarray, n_rep: int = 3) -> float:
    """Median wall-clock of fit(n obs, warm from n-1) + acquisition, in seconds."""
    times = []
    sur = make_fit()
    sur.fit(d.X[idx[:-1]], y[:-1])
    state = BOState(X_obs=d.X[idx], y_obs=y, obs_idx=idx, t=0, n_steps=1)
    rng = np.random.default_rng(0)
    for rep in range(n_rep + 1):
        sur.fit(d.X[idx[:-1]], y[:-1])
        _sync()
        t0 = time.perf_counter()
        sur.fit(d.X[idx], y)
        acq(sur, d.X, state, rng)
        _sync()
        if rep > 0:  # first iteration is warm-up
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _problem(loader: Callable[[str, int, int], SharedData], ds: str, s: int, e: int,
             n: int) -> tuple[SharedData, np.ndarray, np.ndarray]:
    d = loader(ds, s, e)
    rng = np.random.default_rng(0)
    idx = rng.choice(d.X.shape[0], n)
    return d, idx, np.array([draw_trial(d, i, rng) for i in idx])


@pytest.mark.gpu
def test_runtime_budget(tabpfn_device: str, neurostim_loader: Callable[[str, int, int], SharedData]) -> None:
    """#10: PFN TS-joint (default config) BO step below SHADOW_TS_MAX_STEP_S at M=96 and M~2048."""
    out = {}
    for ds, s, e, n in SETTINGS:
        d, idx, y = _problem(neurostim_loader, ds, s, e, n)
        out[f"{ds}_M{d.X.shape[0]}_pfn_ts_s"] = _step_time(
            lambda: TabPFNJointSurrogate(tabpfn_device), TSJoint(), d, idx, y, n_rep=2)
    report("runtime_budget", limit_s=MAX_STEP_S, **out)
    assert all(v < MAX_STEP_S for v in out.values())


@pytest.mark.slow
@pytest.mark.gpu
def test_latency_pfn_vs_gp_report(tabpfn_device: str,
                                  neurostim_loader: Callable[[str, int, int], SharedData]) -> None:
    """Report-only: per-step (fit + acquisition) time, PFN vs converged GP, same data/context/machine.

    GP time is the faster of CPU and CUDA (strictest comparison). Design goal: ratio < 1.
    """
    acqs = {
        "ts": (lambda: TSJoint(), lambda: TSJoint()),
        "ts_marginal": (lambda: TSMarginal("predictive"), lambda: TSMarginal("predictive")),
        "ei": (lambda: EI("predictive"), lambda: EI("predictive")),
        "ucb": (lambda: UCB(2.0, "predictive"), lambda: UCB(2.0, "predictive")),
    }
    for ds, s, e, n in SETTINGS:
        d, idx, y = _problem(neurostim_loader, ds, s, e, n)
        M = d.X.shape[0]
        for name, (pfn_acq, gp_acq) in acqs.items():
            gp_cpu = _step_time(lambda: ExactGPSurrogate("cpu"), gp_acq(), d, idx, y)
            gp_gpu = (_step_time(lambda: ExactGPSurrogate("cuda"), gp_acq(), d, idx, y)
                      if torch.cuda.is_available() else float("inf"))
            pfn = _step_time(lambda: TabPFNJointSurrogate(tabpfn_device), pfn_acq(), d, idx, y)
            gp = min(gp_cpu, gp_gpu)
            report(f"latency_{ds}_M{M}_n{n}_{name}", pfn_s=pfn, gp_cpu_s=gp_cpu, gp_cuda_s=gp_gpu,
                   ratio_pfn_over_best_gp=pfn / gp, goal_met=pfn < gp)
        if M <= 128:
            full = _step_time(lambda: TabPFNJointSurrogate(tabpfn_device),
                              TSJoint(JointTSConfig(k_max=0, q_hi=1.0 - 1e-9)), d, idx, y, n_rep=2)
            report(f"latency_{ds}_M{M}_n{n}_ts_full_pool", pfn_s=full)
        else:
            k128 = _step_time(lambda: TabPFNJointSurrogate(tabpfn_device),
                              TSJoint(JointTSConfig(k_max=128)), d, idx, y, n_rep=1)
            report(f"latency_{ds}_M{M}_n{n}_ts_k128", pfn_s=k128)
