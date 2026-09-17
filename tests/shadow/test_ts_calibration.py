"""Calibration of P(argmax) (#3), heteroscedastic over-exploration (#4), candidate-set
restriction mass (#6) and order sensitivity (#7, report-only)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from _shadow_utils import report
from _ts_prototype.gp import ExactGPSurrogate
from _ts_prototype.preprocessing import SharedData, draw_trial
from _ts_prototype.protocol import BOState
from _ts_prototype.synthetic import grid, sample_gp_functions
from _ts_prototype.tabpfn_surrogate import TabPFNJointSurrogate
from _ts_prototype.thompson import (
    JointTSConfig,
    TSEnsemble,
    TSJoint,
    TSMarginal,
    select_candidates,
    sequential_joint_sample,
)

ECE_BOUND_GP_TRUE = 0.08   # harness sanity (exact model = data-generating prior)
ECE_BOUND_PFN = 0.15       # design A (and B, to be retained) must meet this


def _argmax_probs(samples: np.ndarray, cand: np.ndarray, M: int) -> np.ndarray:
    """Empirical argmax distribution over the pool, samples [S, k] on candidates [k] -> [M]."""
    am = cand[np.argmax(samples, axis=1)]
    return np.bincount(am, minlength=M) / samples.shape[0]


def _top1_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 5) -> float:
    """Expected calibration error of top-1 confidence (equal-width bins)."""
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


@pytest.mark.slow
@pytest.mark.gpu
def test_prob_optimal_calibration_synthetic(tabpfn_device: str) -> None:
    """#3: calibration of P(argmax = true argmax) over functions drawn from a known GP prior.

    8x8 grid (M=64), lengthscale 0.2, signal var 1, noise SD 0.5, 12 noisy random observations.
    Per method: top-1 confidence ECE (5 bins) and mean P(true argmax) (a proper score; top-1
    ECE alone rewards the very diffuse TS-marginal). A one-off extended run (2026-09-16, 120
    functions, 66 min) additionally measured ECE / mean P_true: PFN joint k_max=32
    0.086 / 0.038, full pool 0.093 / 0.033, fantasy_mean 0.135 / 0.039, empirical-noise
    ablation 0.044 / 0.039, marginal-latent 0.045 / 0.035, GP fitted 0.128 / 0.070.
    """
    rng = np.random.default_rng(2024)
    X = grid(8, 2)
    M = X.shape[0]
    n_fun, n_ctx = 120, 12
    methods = ("gp_true", "gp_seq_matheron", "pfn_ts_joint", "pfn_ts_marginal", "pfn_ts_ensemble")
    conf: dict[str, list[float]] = {m: [] for m in methods}
    correct: dict[str, list[float]] = {m: [] for m in methods}
    p_true: dict[str, list[float]] = {m: [] for m in methods}
    joint = TSJoint(JointTSConfig())
    gp_seq = TSJoint(JointTSConfig(), use_exact_if_available=False)
    fs = sample_gp_functions(X, n_fun, 0.2, 1.0, rng)                  # [n_fun, M]
    for i in range(n_fun):
        f = fs[i]
        star = int(np.argmax(f))
        idx = rng.choice(M, n_ctx)
        y = f[idx] + 0.5 * rng.standard_normal(n_ctx)
        probs: dict[str, np.ndarray] = {}
        gp_t = ExactGPSurrogate()
        gp_t.set_fixed(X[idx], y, 0.2, 1.0, 0.25)
        probs["gp_true"] = _argmax_probs(gp_t.sample_joint(X, 512, rng, "latent"), np.arange(M), M)
        s, cand, _ = gp_seq.draw(gp_t, X, None, rng, n_samples=256)
        probs["gp_seq_matheron"] = _argmax_probs(s, cand, M)
        pfn = TabPFNJointSurrogate(device=tabpfn_device)
        pfn.fit(X[idx], y)
        s, cand, _ = joint.draw(pfn, X, None, rng, n_samples=64)
        probs["pfn_ts_joint"] = _argmax_probs(s, cand, M)
        probs["pfn_ts_marginal"] = _argmax_probs(TSMarginal("predictive").draw(pfn, X, rng, 512), np.arange(M), M)
        state = BOState(X_obs=X[idx], y_obs=y, obs_idx=idx, t=0, n_steps=1)
        probs["pfn_ts_ensemble"] = _argmax_probs(TSEnsemble().draw(pfn, X, state, rng, 64), np.arange(M), M)
        for m, p in probs.items():
            conf[m].append(float(p.max()))
            correct[m].append(float(np.argmax(p) == star))
            p_true[m].append(float(p[star]))
    out: dict[str, object] = {}
    for m in methods:
        out[f"{m}_ece"] = _top1_ece(np.array(conf[m]), np.array(correct[m]))
        out[f"{m}_meanP_true"] = float(np.mean(p_true[m]))
        out[f"{m}_top1_acc"] = float(np.mean(correct[m]))
        out[f"{m}_mean_conf"] = float(np.mean(conf[m]))
    out["design_B_retained"] = out["pfn_ts_ensemble_ece"] < ECE_BOUND_PFN
    report("calibration", **out)
    assert out["gp_true_ece"] < ECE_BOUND_GP_TRUE
    assert out["pfn_ts_joint_ece"] < ECE_BOUND_PFN


@pytest.mark.gpu
def test_ts_marginal_overexplores_heteroscedastic_toy(tabpfn_device: str) -> None:
    """#4: latent joint TS concentrates on the true best more than marginal predictive TS.

    Pre-registered second claim (marginal TS picks the high-noise decoy > 2x as often as
    joint TS) was NOT reproduced for the PFN (measured ratio ~1.6 with a f~0 decoy; with a
    f=0.5 decoy the latent joint picks it *more*, which is correct behaviour since 3 trials
    of SD 1.5 leave its mean genuinely uncertain). It is kept report-only.
    """
    X = np.linspace(0, 1, 30)[:, None]
    f = 0.8 * np.exp(-((X[:, 0] - 0.7) / 0.08) ** 2)          # true best at x~0.7 (f=0.8)
    decoy = 7                                                   # x~0.24, f~0, trial SD 1.5
    best = int(np.argmax(f))
    sd = np.full(30, 0.1)
    sd[decoy] = 1.5
    res: dict[str, list[float]] = {"marg_decoy": [], "joint_decoy": [], "marg_best": [], "joint_best": []}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        sites = np.unique(np.concatenate([np.arange(0, 30, 3), [decoy, best]]))
        idx = np.repeat(sites, 3)
        y = f[idx] + sd[idx] * rng.standard_normal(idx.size)
        pfn = TabPFNJointSurrogate(device=tabpfn_device)
        pfn.fit(X[idx], y)
        pm = _argmax_probs(TSMarginal("predictive").draw(pfn, X, rng, 1024), np.arange(30), 30)
        s, cand, _ = TSJoint(JointTSConfig()).draw(pfn, X, None, rng, n_samples=256)
        pj = _argmax_probs(s, cand, 30)
        res["marg_decoy"].append(float(pm[decoy]))
        res["joint_decoy"].append(float(pj[decoy]))
        res["marg_best"].append(float(pm[best]))
        res["joint_best"].append(float(pj[best]))
    r = {k: float(np.mean(v)) for k, v in res.items()}
    r["decoy_ratio_marg_over_joint"] = r["marg_decoy"] / max(r["joint_decoy"], 1e-9)
    r["decoy_claim_reproduced"] = r["decoy_ratio_marg_over_joint"] > 2.0
    report("heteroscedastic_toy", **r)
    assert r["joint_best"] > r["marg_best"]


@pytest.mark.slow
@pytest.mark.gpu
def test_candidate_restriction_mass(
    tabpfn_device: str, neurostim_loader: Callable[[str, int, int], SharedData]
) -> None:
    """#6: argmax mass lost by candidate restriction.

    Asserted: P(full-pool sequential argmax falls outside C_default) < 0.02 on NHP contexts
    (default k_max=256 keeps the whole optimism set on 96-site grids). Reported: the
    originally proposed k_max=32 (fails for the PFN) and 5d_rat truncation mass against a
    k=512, chunk-4 reference (itself truncated; report-only).
    """
    outside_default, outside_32, sizes = [], [], []
    cfg = JointTSConfig()
    for subject, emg, seed in ((1, 0, 0), (0, 2, 1), (3, 1, 2)):
        d = neurostim_loader("nhp", subject, emg)
        rng = np.random.default_rng(seed)
        M = d.X.shape[0]
        idx = rng.choice(M, 20)
        y = np.array([draw_trial(d, i, rng) for i in idx])
        pfn = TabPFNJointSurrogate(device=tabpfn_device)
        pfn.fit(d.X[idx], y)
        pred = pfn.predict_marginals(d.X, "predictive")
        C_def = select_candidates(pred, cfg.k_max, cfg.q_hi)
        C_32 = select_candidates(pred, 32, cfg.q_hi)
        full = TSJoint(JointTSConfig(k_max=0, q_hi=1.0 - 1e-9))
        s, cand, _ = full.draw(pfn, d.X, None, rng, n_samples=128)
        am = cand[np.argmax(s, 1)]
        outside_default.append(float(np.mean(~np.isin(am, C_def))))
        outside_32.append(float(np.mean(~np.isin(am, C_32))))
        sizes.append((int(C_def.size), int(C_32.size), M))
    d = neurostim_loader("5d_rat", 1, 0)
    rng = np.random.default_rng(100)
    M = d.X.shape[0]
    idx = rng.choice(M, 100)
    y = np.array([draw_trial(d, i, rng) for i in idx])
    pfn = TabPFNJointSurrogate(device=tabpfn_device)
    pfn.fit(d.X[idx], y)
    pred = pfn.predict_marginals(d.X, "predictive")
    rank = np.argsort(-pred.icdf(np.full(M, 0.99)))
    s, cand, diag = TSJoint(JointTSConfig(k_max=512, chunk_size=4, max_chain_passes=512)).draw(
        pfn, d.X, None, rng, n_samples=64)
    ref = _argmax_probs(s, cand, M)
    lost_5d = {k: round(1 - float(ref[rank[:k]].sum()), 3) for k in (32, 64, 128, 256)}
    report("candidate_restriction", nhp_p_outside_default=outside_default, nhp_p_outside_k32=outside_32,
           nhp_sizes_default_k32_M=sizes, d5_mass_outside_topk_vs_k512_ref=lost_5d, d5_ref_k=diag["k"])
    assert max(outside_default) < 0.02


@pytest.mark.gpu
def test_order_sensitivity_pfn(
    tabpfn_device: str, neurostim_loader: Callable[[str, int, int], SharedData]
) -> None:
    """#7 (report-only): TV of argmax distributions under two chain orders vs same-order floor.

    Uses the top-32 candidates (a subset suffices to probe order dependence of the PFN).
    """
    d = neurostim_loader("nhp", 1, 0)
    rng = np.random.default_rng(5)
    M = d.X.shape[0]
    idx = rng.choice(M, 20)
    y = np.array([draw_trial(d, i, rng) for i in idx])
    pfn = TabPFNJointSurrogate(device=tabpfn_device)
    pfn.fit(d.X[idx], y)
    pred = pfn.predict_marginals(d.X, "predictive")
    C = select_candidates(pred, 32, 0.99)
    k = C.size
    order_a = np.random.default_rng(1).permutation(k)
    order_b = np.random.default_rng(2).permutation(k)

    def _run(order: np.ndarray, seed: int) -> np.ndarray:
        gen = np.random.default_rng(seed)
        Xc = d.X[C][order]
        pc = pfn.predict_marginals(Xc, "predictive")
        sens = pfn.sensitivity(Xc, pc)
        nv = (1 - sens) * np.asarray(pc.std) ** 2
        out = sequential_joint_sample(pfn, Xc, 512, _FixedOrderRng(gen), JointTSConfig(), nv, sens)
        return _argmax_probs(out["f"], C[order], M)

    pa1, pa2, pb = _run(order_a, 10), _run(order_a, 11), _run(order_b, 12)
    tv = lambda p, q: 0.5 * float(np.abs(p - q).sum())
    report("order_sensitivity", tv_different_orders=tv(pa1, pb), tv_same_order_noise_floor=tv(pa1, pa2), k=k)


class _FixedOrderRng:
    """Generator proxy whose ``permutation`` is the identity (order fixed by the caller)."""

    def __init__(self, gen: np.random.Generator) -> None:
        self._gen = gen

    def permutation(self, k: int) -> np.ndarray:
        return np.arange(k)

    def __getattr__(self, name: str) -> object:
        return getattr(self._gen, name)
