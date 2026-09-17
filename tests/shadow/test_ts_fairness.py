"""P0.9 / P0.10 / P0.11 fairness fixes, engine pinning, temperature, units, metadata.

Covers planned tests #11 (temperature semantics), #12 (regret units invariant, extended to
the three co-primary regrets), the tabpfn==6.3.2 pin, and metadata recording.
"""
from __future__ import annotations

import importlib.metadata
import os
import sys
from typing import Callable

import numpy as np
import pytest
import torch
from scipy.stats import norm

from _shadow_utils import report
from _ts_prototype.bo import run_bo
from _ts_prototype.gp import ExactGPSurrogate
from _ts_prototype.metrics import regret_curves
from _ts_prototype.preprocessing import PROJECT_ROOT, SharedData, draw_trial, shared_preprocess
from _ts_prototype.protocol import BarMarginals
from _ts_prototype.summaries import PAIR_DIVISORS, QUANTILE_LEVELS, std_from_quantiles
from _ts_prototype.synthetic import gp_problem, grid, sample_gp_functions
from _ts_prototype.tabpfn_engine import PINNED_TABPFN_VERSION, FrozenTabPFN
from _ts_prototype.tabpfn_surrogate import TabPFNJointSurrogate
from _ts_prototype.thompson import JointTSConfig, TSJoint


def _toy(seed: int, n_ctx: int = 20) -> tuple[SharedData, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = grid(8, 2)                                                   # [64, 2]
    f = sample_gp_functions(X, 1, 0.25, 1.0, rng)[0]                 # [64]
    d = gp_problem(X, f, 0.3, 8, rng)
    idx = rng.choice(64, n_ctx)
    y = np.array([draw_trial(d, i, rng) for i in idx])
    return d, idx, y


def test_tabpfn_engine_pinned(tabpfn_device: str) -> None:
    """Low-level batched forward reproduces TabPFNRegressor.predict logits exactly (fail loudly)."""
    assert importlib.metadata.version("tabpfn") == PINNED_TABPFN_VERSION, (
        "tabpfn version changed: FrozenTabPFN internals must be re-validated")
    d, idx, y = _toy(0)
    eng = FrozenTabPFN(device=tabpfn_device)
    eng.fit(d.X[idx], y)
    ref = eng.wrapper.predict(d.X, output_type="full")["logits"]     # [M, bars]
    low = eng.forward(None, None, eng.transform_x(d.X))[0]           # [M, bars]
    dp = float((ref.softmax(-1) - low.softmax(-1)).abs().max())
    # batched copies must agree with the single-context pass
    Xt = eng.transform_x(d.X)
    batched = eng.forward(Xt[:0].unsqueeze(0).expand(3, -1, -1), torch.zeros(3, 0, device=eng.device), Xt)
    db = float((batched.softmax(-1) - low.softmax(-1).unsqueeze(0)).abs().max())
    report("engine_pinned", max_abs_prob_diff_vs_wrapper=dp, max_abs_prob_diff_batched=db)
    assert dp < 1e-5 and db < 1e-4


def test_temperature_semantics(tabpfn_device: str) -> None:
    """#11: TS uses T=1 bar distributions (TabPFN's default 0.9 is NOT silently applied)."""
    d, idx, y = _toy(1)
    sur = TabPFNJointSurrogate(device=tabpfn_device)
    sur.fit(d.X[idx], y)
    assert sur.engine.wrapper.softmax_temperature == 1.0
    marg = sur.predict_marginals(d.X, "predictive")
    crit_mean = sur.engine.criterion.mean(marg.logprobs).cpu().numpy()
    report("temperature", wrapper_T=sur.engine.wrapper.softmax_temperature,
           max_mean_diff_vs_criterion=float(np.abs(marg.mean - crit_mean).max()))
    assert np.allclose(marg.mean, crit_mean, atol=1e-3)


def test_predictive_summaries_corrected() -> None:
    """P0.11: mean (not median), divisor 3.290 for 5/95, fail fast on NaN."""
    divs = {(lo, hi): div for lo, hi, div in PAIR_DIVISORS}
    assert divs[(0, 6)] == pytest.approx(3.290, abs=1e-3)
    assert divs[(1, 5)] == pytest.approx(2.563, abs=1e-3)
    assert divs[(2, 4)] == pytest.approx(1.349, abs=1e-3)
    sigma = np.array([0.5, 1.0, 3.0])
    q = np.stack([norm.ppf(l, loc=2.0, scale=sigma) for l in QUANTILE_LEVELS])  # [7, 3]
    assert np.allclose(std_from_quantiles(q), sigma, rtol=1e-3)
    with pytest.raises(RuntimeError):
        std_from_quantiles(np.full((7, 3), np.nan))
    # Bar mean vs median on a skewed distribution: must be the mean.
    borders = torch.linspace(0.0, 10.0, 101)                                   # [101]
    p = torch.exp(-torch.linspace(0.0, 10.0, 100))                             # skewed
    p = p / p.sum()
    marg = BarMarginals(p.log().unsqueeze(0), borders, "predictive")
    centres = (borders[:-1] + borders[1:]) / 2
    true_mean = float((p * centres).sum())
    median = float(marg.icdf(np.array([0.5]))[0])
    report("summaries", bar_mean=float(marg.mean[0]), true_mean=true_mean, median=median)
    assert marg.mean[0] == pytest.approx(true_mean, rel=1e-6)
    assert abs(median - true_mean) > 0.1  # the old median-as-mean was materially biased


def test_gp_converged_noise_matches_trial_sd(neurostim_loader: Callable[[str, int, int], SharedData]) -> None:
    """P0.10: converged GP noise matches the trial SD.

    Strict on homoscedastic synthetic data (ratio in [0.67, 1.5]). Real NHP channels are
    strongly heteroscedastic (hotspot trial SD >> median), so a homoscedastic noise SD must
    fall inside the band [median within-site SD / 2, RMS within-site SD * 1.5]; the legacy
    ``src`` GP (50 Adam steps, lr 0.01) is reported for contrast.
    """
    ratios = []
    for seed in range(4):
        d, idx, y = _toy(seed, n_ctx=60)
        gp = ExactGPSurrogate()
        gp.fit(d.X[idx], y)
        emp = float(np.sqrt(np.nanmean(np.nanvar(d.y_trials, axis=1))))
        ratios.append(float(np.sqrt(gp.hyperparameters["noise_var"]) / emp))
    in_band, rows = [], []
    for subject in (0, 1, 3):
        for emg in (0, 1, 2):
            d = neurostim_loader("nhp", subject, emg)
            rng = np.random.default_rng(subject)
            idx = rng.choice(d.X.shape[0], 60)
            y = np.array([draw_trial(d, i, rng) for i in idx])
            gp = ExactGPSurrogate()
            gp.fit(d.X[idx], y)
            sd_site = np.sqrt(np.nanvar(d.y_trials, axis=1))
            lo, hi = float(np.median(sd_site)) / 2, float(np.sqrt(np.mean(sd_site ** 2))) * 1.5
            s = float(np.sqrt(gp.hyperparameters["noise_var"]))
            in_band.append(lo <= s <= hi)
            rows.append(round(s / float(np.mean(sd_site)), 2))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    from models.regressors import GPSurrogate  # read-only import

    d = neurostim_loader("nhp", 1, 0)
    rng = np.random.default_rng(1)
    idx = rng.choice(d.X.shape[0], 60)
    y = np.array([draw_trial(d, i, rng) for i in idx])
    legacy = GPSurrogate(device="cpu")
    legacy.fit(d.X[idx], y)
    new = ExactGPSurrogate()
    new.fit(d.X[idx], y)
    mean_sd = float(np.mean(np.sqrt(np.nanvar(d.y_trials, axis=1))))
    report("gp_noise", synthetic_ratio_to_trial_sd=[round(r, 3) for r in ratios],
           nhp_ratio_to_mean_site_sd=rows, nhp_in_band=f"{sum(in_band)}/{len(in_band)}",
           legacy_src_gp_ratio_s1e0=float(np.sqrt(legacy._likelihood.noise.item())) / mean_sd,
           converged_ratio_s1e0=float(np.sqrt(new.hyperparameters["noise_var"])) / mean_sd)
    assert all(0.67 <= r <= 1.5 for r in ratios)
    assert all(in_band)


def test_regret_units_invariant() -> None:
    """#12: all three regrets are identical under MinMax-y vs z-score-y (any affine map)."""
    rng = np.random.default_rng(3)
    trials = rng.gamma(2.0, 1.0, size=(40, 6))                         # skewed raw EMG-like
    gt = trials.mean(1)
    order = np.argsort(trials.mean(1))
    q = rng.choice(order[:30], 25)            # never queries the optimum -> non-zero regret
    rec = rng.choice(order[:35], 22)
    z = shared_preprocess(rng.random((40, 2)), trials, gt).y_gt       # z-score units
    mm = (gt - trials.min()) / (trials.max() - trials.min())          # legacy 'gp' MinMax units
    a = regret_curves(z, q, rec)
    b = regret_curves(mm, q, rec)
    c = regret_curves(gt * 37.0 - 5.0, q, rec)
    for key in ("recommended", "best_queried", "cumulative"):
        assert np.allclose(a[key], b[key]) and np.allclose(a[key], c[key]), key
    raw_a, raw_b = z.max() - z[q].max(), mm.max() - mm[q].max()
    report("units", raw_regret_z=raw_a, raw_regret_minmax=raw_b, normalised=float(a["best_queried"][-1]))


def test_bo_records_acquisition_metadata() -> None:
    """acq_fn + all acquisition params + units are recorded in every result."""
    d, _, _ = _toy(5)
    acq = TSJoint(JointTSConfig(k_max=16, chunk_size=2))
    res = run_bo(ExactGPSurrogate(), acq, d, n_init=4, budget=10, seed=0)
    md = res["metadata"]
    assert md["acq_fn"] == "ts"
    for k in ("k_max", "chunk_size", "q_hi", "latent_mode", "noise_source", "probe_delta", "space"):
        assert k in md["acq_params"], k
    assert md["budget"] == 10 and md["n_init"] == 4 and "units" in md
    assert res["queried_idx"].shape == (10,) and res["recommended_idx"].shape == (7,)
    for key in ("recommended", "best_queried", "cumulative"):
        assert np.isfinite(res["regret"][key]).all()
