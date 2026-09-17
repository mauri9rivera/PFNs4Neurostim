"""#8 (report-only): BO regret, TS-joint vs TS-marginal, on synthetic 8x8 GP-prior grids.

Budget is an explicit iteration count (24 total incl. 4 initial) on a 64-site grid.
All three co-primary regrets are reported in range-normalised GT units.
"""
from __future__ import annotations

import numpy as np
import pytest

from _shadow_utils import report
from _ts_prototype.bo import run_bo
from _ts_prototype.gp import ExactGPSurrogate
from _ts_prototype.synthetic import gp_problem, grid, sample_gp_functions
from _ts_prototype.tabpfn_surrogate import TabPFNJointSurrogate
from _ts_prototype.thompson import JointTSConfig, TSJoint, TSMarginal

BUDGET = 24
N_INIT = 4
N_SEEDS = 6


@pytest.mark.slow
@pytest.mark.gpu
def test_regret_ts_joint_vs_marginal_synthetic(tabpfn_device: str) -> None:
    """Report mean final regrets (recommended, best-queried, cumulative) per arm."""
    X = grid(8, 2)
    arms = {
        "pfn_ts_joint": (lambda: TabPFNJointSurrogate(tabpfn_device), lambda: TSJoint()),
        "pfn_ts_joint_full": (lambda: TabPFNJointSurrogate(tabpfn_device),
                              lambda: TSJoint(JointTSConfig(k_max=0, q_hi=1.0 - 1e-9))),
        "pfn_ts_joint_empirical": (lambda: TabPFNJointSurrogate(tabpfn_device),
                                   lambda: TSJoint(JointTSConfig(noise_source="empirical"))),
        "pfn_ts_marginal": (lambda: TabPFNJointSurrogate(tabpfn_device), lambda: TSMarginal("predictive")),
        "gp_ts_joint": (lambda: ExactGPSurrogate(), lambda: TSJoint()),
        "gp_ts_marginal": (lambda: ExactGPSurrogate(), lambda: TSMarginal("predictive")),
    }
    rng = np.random.default_rng(99)
    problems = [gp_problem(X, sample_gp_functions(X, 1, 0.2, 1.0, rng)[0], 0.5, 8, rng, name=f"gp{i}")
                for i in range(N_SEEDS)]
    summary = {}
    for arm, (make_s, make_a) in arms.items():
        rec, bq, cum, uniq, meta = [], [], [], [], None
        for i, d in enumerate(problems):
            res = run_bo(make_s(), make_a(), d, n_init=N_INIT, budget=BUDGET, seed=1000 + i)
            rec.append(res["regret"]["recommended"][-1])
            bq.append(res["regret"]["best_queried"][-1])
            cum.append(res["regret"]["cumulative"][-1])
            uniq.append(len(np.unique(res["queried_idx"])))
            meta = res["metadata"]
        assert meta is not None and meta["acq_fn"] in ("ts", "ts_marginal")
        summary[arm] = (float(np.mean(rec)), float(np.mean(bq)), float(np.mean(cum)), float(np.mean(uniq)))
        report(f"regret_{arm}", recommended=summary[arm][0], best_queried=summary[arm][1],
               cumulative=summary[arm][2], unique_sites=summary[arm][3], n_problems=N_SEEDS,
               budget=BUDGET, acq_params=meta["acq_params"])
    j, m = summary["pfn_ts_joint"], summary["pfn_ts_marginal"]
    report("regret_joint_le_marginal", recommended=j[0] <= m[0], best_queried=j[1] <= m[1],
           cumulative=j[2] <= m[2])
    assert all(np.isfinite(v).all() for v in summary.values())
