"""Sampler correctness (#1, #2), latent deflation (#5), determinism (#9)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from _shadow_utils import report
from _ts_prototype.acquisitions import EI, UCB
from _ts_prototype.bo import run_bo
from _ts_prototype.gp import ExactGPSurrogate
from _ts_prototype.preprocessing import draw_trial
from _ts_prototype.protocol import Surrogate
from _ts_prototype.synthetic import gp_problem, grid, sample_gp_functions
from _ts_prototype.tabpfn_surrogate import TabPFNJointSurrogate
from _ts_prototype.thompson import (
    JointTSConfig,
    TSEnsemble,
    TSJoint,
    TSMarginal,
    sequential_joint_sample,
)


def _energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Energy distance between two samples, shapes [n, d] and [m, d]."""
    d = lambda u, v: np.sqrt(((u[:, None, :] - v[None, :, :]) ** 2).sum(-1)).mean()
    return 2 * d(a, b) - d(a, a) - d(b, b)


def _perm_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    obs = _energy_distance(a, b)
    pooled = np.concatenate([a, b])
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled.shape[0])
        cnt += _energy_distance(pooled[p[: a.shape[0]]], pooled[p[a.shape[0]:]]) >= obs
    return (cnt + 1) / (n_perm + 1)


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def _argmax_dist(samples: np.ndarray, cand: np.ndarray, M: int) -> np.ndarray:
    """Empirical argmax distribution over the pool from samples [S, k] on candidates [k]."""
    am = cand[np.argmax(samples, axis=1)]
    return np.bincount(am, minlength=M) / samples.shape[0]


def test_sequential_sampler_matches_exact_gp_joint() -> None:
    """#1: generic sequential sampler (chunk=1, predictive) == exact GP predictive MVN."""
    rng = np.random.default_rng(0)
    X = np.linspace(0, 1, 20)[:, None]                                # [20, 1]
    f = sample_gp_functions(X, 1, 0.2, 1.0, rng)[0]
    d = gp_problem(X, f, 0.3, 8, rng)
    idx = rng.choice(20, 8)
    y = np.array([draw_trial(d, i, rng) for i in idx])
    gp = ExactGPSurrogate()
    gp.fit(d.X[idx], y)
    S = 4000
    seq = sequential_joint_sample(gp, d.X, S, rng, JointTSConfig(chunk_size=1, latent_mode="none"))["y"]
    ex = gp.sample_joint(d.X, S, rng, space="predictive")             # [S, 20]
    ex2 = gp.sample_joint(d.X, S, rng, space="predictive")
    se = ex.std(0) / np.sqrt(S)
    z_mean = float(np.max(np.abs(seq.mean(0) - ex.mean(0)) / (np.sqrt(2) * se)))
    cov_ex = np.cov(ex.T)
    rel = float(np.linalg.norm(np.cov(seq.T) - cov_ex) / np.linalg.norm(cov_ex))
    rel_floor = float(np.linalg.norm(np.cov(ex2.T) - cov_ex) / np.linalg.norm(cov_ex))
    p = _perm_pvalue(seq[:600], ex[:600], 100, rng)
    chunked = sequential_joint_sample(gp, d.X, S, rng, JointTSConfig(chunk_size=4, latent_mode="none"))["y"]
    rel_chunk = float(np.linalg.norm(np.cov(chunked.T) - cov_ex) / np.linalg.norm(cov_ex))
    report("seq_vs_exact_gp", max_mean_z=z_mean, cov_rel_err=rel, cov_rel_err_exact_vs_exact=rel_floor,
           energy_perm_p=p, cov_rel_err_chunk4=rel_chunk)
    assert z_mean < 4.0
    assert rel < 0.1
    assert p > 0.01


@pytest.mark.slow
def test_ts_joint_argmax_matches_prob_optimal_gp() -> None:
    """#2: latent TS-joint argmax distribution ~= P(optimal) from exact joint latent samples (GP).

    Ground truth uses a GP with known hyperparameters; the generic sequential sampler with
    each latent correction is compared by total variation. Headline mode must be < 0.05.
    """
    rng = np.random.default_rng(1)
    X = grid(6, 2)                                                     # [36, 2]
    M = X.shape[0]
    res: dict[str, list[float]] = {}
    for rep in range(3):
        f = sample_gp_functions(X, 1, 0.3, 1.0, rng)[0]
        idx = rng.choice(M, 10)
        y = f[idx] + 0.5 * rng.standard_normal(10)
        gp = ExactGPSurrogate()
        gp.set_fixed(X[idx], y, lengthscale=0.3, outputscale=1.0, noise_var=0.25)
        p_opt = np.bincount(np.argmax(gp.sample_joint(X, 40000, rng, "latent"), 1), minlength=M) / 40000
        p_ref = np.bincount(np.argmax(gp.sample_joint(X, 4000, rng, "latent"), 1), minlength=M) / 4000
        res.setdefault("exact_4000_noise_floor", []).append(_tv(p_ref, p_opt))
        for mode, chunk in (("matheron_diag", 1), ("deflate", 1), ("none", 1),
                            ("matheron_diag", 2), ("matheron_diag", 4)):
            acq = TSJoint(JointTSConfig(k_max=0, chunk_size=chunk, latent_mode=mode),
                          space="latent" if mode != "none" else "predictive", use_exact_if_available=False)
            s, cand, _ = acq.draw(gp, X, None, rng, n_samples=4000)
            key = mode if chunk == 1 else f"{mode}_chunk{chunk}"
            res.setdefault(key, []).append(_tv(_argmax_dist(s, cand, M), p_opt))
        for space in ("latent", "predictive"):
            sm = TSMarginal(space).draw(gp, X, rng, 4000)
            res.setdefault(f"marginal_{space}", []).append(_tv(_argmax_dist(sm, np.arange(M), M), p_opt))
    means = {k: float(np.mean(v)) for k, v in res.items()}
    report("tv_vs_prob_optimal_gp", **means)
    assert means["matheron_diag"] < 0.05
    assert means["matheron_diag"] < means["marginal_predictive"]


@pytest.mark.gpu
def test_latent_deflation_shrinks_with_replicates(tabpfn_device: str) -> None:
    """#5: TabPFN latent SD at x* decreases with replicates and stays below predictive SD."""
    X = np.linspace(0, 1, 25)[:, None]
    j = 12
    lat = {0: [], 5: [], 20: []}
    pred = {0: [], 5: [], 20: []}
    noise_est = {0: [], 5: [], 20: []}
    for seed in range(4):
        rng = np.random.default_rng(seed)
        f = np.sin(6 * X[:, 0]) + 0.5 * X[:, 0]
        others = rng.choice(np.setdiff1d(np.arange(25), [j]), 10, replace=False)
        for R in lat:
            idx = np.concatenate([others, np.full(R, j)]).astype(int)
            y = f[idx] + 0.4 * rng.standard_normal(idx.size)
            sur = TabPFNJointSurrogate(device=tabpfn_device)
            sur.fit(X[idx], y)
            pm = sur.predict_marginals(X[j:j + 1], "predictive")
            lm = sur.predict_marginals(X[j:j + 1], "latent")
            lat[R].append(float(lm.std[0]))
            pred[R].append(float(pm.std[0]))
            noise_est[R].append(float(np.sqrt(max(pm.std[0] ** 2 - lm.std[0] ** 2, 0.0))))
    L = {R: float(np.mean(v)) for R, v in lat.items()}
    P = {R: float(np.mean(v)) for R, v in pred.items()}
    N = {R: float(np.mean(v)) for R, v in noise_est.items()}
    report("deflation", latent_sd=L, predictive_sd=P, implied_noise_sd=N, true_noise_sd=0.4,
           ideal_latent_sd_R20=0.4 / np.sqrt(20))
    assert L[20] < L[5] < L[0]
    assert L[5] < P[5] and L[20] < P[20]


def _det_cases(device: str) -> list[tuple[str, Callable[[], Surrogate], object]]:
    small = JointTSConfig(k_max=8, chunk_size=4)
    return [
        ("gp_ts_exact", lambda: ExactGPSurrogate(), TSJoint()),
        ("gp_ts_sequential", lambda: ExactGPSurrogate(), TSJoint(small, use_exact_if_available=False)),
        ("gp_ei", lambda: ExactGPSurrogate(), EI()),
        ("pfn_ts_joint", lambda: TabPFNJointSurrogate(device), TSJoint(small)),
        ("pfn_ts_marginal", lambda: TabPFNJointSurrogate(device), TSMarginal()),
        ("pfn_ts_ensemble", lambda: TabPFNJointSurrogate(device), TSEnsemble()),
        ("pfn_ucb", lambda: TabPFNJointSurrogate(device), UCB()),
    ]


@pytest.mark.gpu
def test_determinism_fixed_rng(tabpfn_device: str) -> None:
    """#9: identical Generator seeds give identical query sequences for every acquisition."""
    rng = np.random.default_rng(7)
    X = grid(6, 2)
    d = gp_problem(X, sample_gp_functions(X, 1, 0.3, 1.0, rng)[0], 0.3, 6, rng)
    ok = {}
    for name, make, acq in _det_cases(tabpfn_device):
        a = run_bo(make(), acq, d, n_init=4, budget=9, seed=11)["queried_idx"]
        b = run_bo(make(), acq, d, n_init=4, budget=9, seed=11)["queried_idx"]
        c = run_bo(make(), acq, d, n_init=4, budget=9, seed=12)["queried_idx"]
        ok[name] = bool(np.array_equal(a, b))
        ok[name + "_seed_changes"] = bool(not np.array_equal(a, c))
    report("determinism", **ok)
    assert all(v for k, v in ok.items() if not k.endswith("_seed_changes"))
