"""OBJECTIVE BETA: the statistics layer (:mod:`analysis.representation`), the additive belief helpers of
:mod:`analysis.bayes_update`, and the ``representation`` runner end to end on a tiny synthetic case.

The runner test replaces the TabPFN forward pass (``_representation_beliefs``) by a deterministic stand-in built from
GP-fixed's predictive, so it is fast and checkpoint-free; the GP beliefs, every statistic, the cache, the summaries
and the figures are the real code. The real TabPFN path is covered by the slow test at the end and by the smoke run.
"""
from __future__ import annotations

import dataclasses
import json
import os
import textwrap
import types

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis import cka as C
from pfns4neurostim.analysis import representation as R
from pfns4neurostim.data.synthetic_neurostim import gp_channel
from pfns4neurostim.experiments import mechanism


# ---------------------------------------------------------------------------
# bayes_update additions (purely additive)
# ---------------------------------------------------------------------------
def test_belief_levels_are_midpoints() -> None:
    np.testing.assert_allclose(B.belief_levels(4), [0.125, 0.375, 0.625, 0.875])
    with pytest.raises(ValueError):
        B.belief_levels(1)


def test_belief_embedding_of_a_gaussian_is_its_quantile_function() -> None:
    mean, sd = np.array([[0.0, 1.0, -2.0]]), np.array([[1.0, 0.5, 2.0]])
    pred = B.GPRefitEngine._gaussian(mean, sd)
    u = B.belief_levels(8)
    E = B.belief_embedding(pred, u)                                                 # [3, 8]
    np.testing.assert_allclose(E, mean[0][:, None] + sd[0][:, None] * norm.ppf(u)[None], atol=1e-6)


def test_site_w2_split_is_an_identity_and_matches_gaussian_w2() -> None:
    rng = np.random.default_rng(0)
    A, Bm = rng.normal(size=(20, 16)).cumsum(1), rng.normal(size=(20, 16)).cumsum(1)
    split = B.site_w2_split(np.sort(A, 1), np.sort(Bm, 1))
    np.testing.assert_allclose(split["w2"] ** 2, split["location"] + split["spread"] + split["shape"], rtol=1e-10)
    assert (split["shape"] >= 0).all()
    # Two Gaussians: W2^2 = dmu^2 + dsigma^2 exactly, so the shape term vanishes (equal quantile shapes)
    u = B.belief_levels(64)
    Ea = (0.3 + 1.5 * norm.ppf(u))[None]
    Eb = (-0.2 + 0.7 * norm.ppf(u))[None]
    s = B.site_w2_split(Ea, Eb)
    assert s["shape"][0] == pytest.approx(0.0, abs=1e-12)
    assert s["location"][0] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        B.site_w2_split(Ea, Eb[:, :10])


def test_implied_covariance_of_gp_fixed_is_its_posterior_covariance() -> None:
    """With every site an anchor, Part A's moment-matched rows reproduce GP-fixed's k_C exactly (β1b's floor)."""
    ch = gp_channel(5, 0.3, 0.2, 4, np.random.default_rng(1))
    rng = np.random.default_rng(2)
    ctx = rng.choice(ch.n_sites, 8, replace=False)
    eng = B.GPFixedEngine(0.6931471805599453, 1.0, 0.05)
    eng.fit(ch.X_pool[ctx], ch.Y_trials[ctx, 0])
    _, _, Cc = B.coherence(eng, ch.X_pool, np.arange(ch.n_sites), 6, 0.01, method="gauss_hermite", u_min=1e-12,
                           u_tail=1e-4, grid_points=64)
    imp = B.implied_posterior_covariance(Cc)
    np.testing.assert_allclose(imp["cov"], eng.posterior_covariance(ch.X_pool), atol=1e-8)
    assert imp["asymmetry"] < 1e-8 and imp["clipped_mass"] < 1e-8
    np.testing.assert_allclose(B.latent_covariance_of(eng, ch.X_pool), eng.posterior_covariance(ch.X_pool))
    with pytest.raises(TypeError):
        B.latent_covariance_of(object(), ch.X_pool)


# ---------------------------------------------------------------------------
# analysis.representation
# ---------------------------------------------------------------------------
def test_parse_target() -> None:
    assert R.parse_target("mean_mll") == ("mean", "mll")
    assert R.parse_target("K_X") == ("K_X", None)
    for bad in ("cov_own", "median_mll", "mean_gp"):
        with pytest.raises(ValueError):
            R.parse_target(bad)


def test_target_confounds_add_layout_for_covariance_targets_only() -> None:
    geom = [np.eye(5), 2 * np.eye(5)]
    lay = 3 * np.eye(5)
    assert len(R.target_confounds("cov_mll", geom, lay, "fixed")) == 3
    assert len(R.target_confounds("cov_fixed", geom, lay, "fixed")) == 2                 # not removed from itself
    assert len(R.target_confounds("mean_mll", geom, lay, "fixed")) == 2


def test_add_headline_selects_the_statistic() -> None:
    df = pd.DataFrame({"cka": [0.6], "cka_deconf": [0.3], "surrogate_q_cka_deconf": [0.1],
                       "surrogate_median_cka": [0.2], "surrogate_q_cka": [0.4]})
    a = R.add_headline(df, "deconfounded_vs_surrogate")
    assert a["headline_value"][0] == 0.3 and a["headline_floor"][0] == 0.1 and a["headline_above_floor"][0]
    b = R.add_headline(df, "floor_subtracted_raw")
    assert b["headline_value"][0] == pytest.approx(0.4) and b["headline_floor"][0] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        R.add_headline(df, "bogus")


def test_surrogate_floor_is_the_most_demanding_lengthscale() -> None:
    rows = [{"target": "t", "lengthscale": ell, "statistic": "cka_deconf", "median": m, "q": q, "n": 3}
            for ell, m, q in ((0.1, 0.0, 0.2), (0.3, 0.05, 0.1))]
    f = R.surrogate_floors(rows)["t"]
    assert f["surrogate_q_cka_deconf"] == 0.2 and f["surrogate_median_cka_deconf"] == 0.05


def test_channel_summary_reduces_draws_then_bootstraps_within_subject() -> None:
    rows = []
    for subject, emgs in ((0, range(4)), (1, range(3))):
        for e in emgs:
            for draw, offset in ((0, -1.0), (1, 1.0)):                   # draws differ, their median is the channel
                rows.append({"subject": subject, "emg": e, "draw": draw, "g": "a", "v": 10 * subject + e + offset})
    out = R.channel_summary(pd.DataFrame(rows), ["v"], ["g"], n_boot=200, ci=0.9, seed=0)
    (row,) = out.to_dict("records")
    channel_values = [0, 1, 2, 3, 10, 11, 12]
    assert row["median"] == pytest.approx(np.median(channel_values))
    assert row["n_channels"] == 7 and row["n_subjects"] == 2
    assert row["ci_low"] <= row["median"] <= row["ci_high"]
    # within-subject resampling: every replicate keeps 4 subject-0 and 3 subject-1 channels, so no replicate median
    # can fall below subject 0's maximum (3) -- the median of 7 values is the 4th.
    assert row["ci_low"] >= 0 and row["ci_high"] <= 11


def test_alignment_rows_flag_a_degenerate_representation() -> None:
    rng = np.random.default_rng(0)
    X = np.stack(np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6), indexing="ij"), -1).reshape(-1, 2)
    y = np.sin(3 * X[:, 0])
    targets = {"mean_own": C.centered_linear_gram(y), "K_X": C.rbf_gram(X, C.median_distance(X))}
    dec = C.Deconfounder(C.geometry_confounds(X, (0.5, 1.0)))
    reps = {("label", 0): C.linear_gram(np.ones((36, 3))),                          # constant across sites
            ("label", 1): C.linear_gram(np.c_[y, rng.normal(size=(36, 2)) * 0.1])}
    rows = R.alignment_rows(reps, targets, {k: dec for k in targets}, n_perm=10, rng=rng, min_residual_frac=1e-10)
    by = {(r["layer"], r["target"]): r for r in rows}
    assert by[(0, "mean_own")]["rep_degenerate"] and by[(0, "mean_own")]["cka"] is None
    assert by[(1, "mean_own")]["cka"] > 0.9 and by[(1, "mean_own")]["perm_null_mean"] is not None
    assert by[(1, "K_X")]["cka_deconf"] == 0.0 and by[(1, "K_X")]["deconf_degenerate"]


# ---------------------------------------------------------------------------
# The runner on a tiny synthetic case
# ---------------------------------------------------------------------------
_N_BLOCKS = 3


def _fake_beliefs(ctx, Q, p, tabpfn, u):
    """Stand-in for the TabPFN pass: GP beliefs are real; 'own' is a distorted GP-fixed; blocks are built from it."""
    from pfns4neurostim.analysis.embeddings import BlockStates

    codes = {e: c for c, e in R.BELIEF_CODES.items()}
    beliefs = {}
    for name in p["belief_engines"]:
        g = mechanism._representation_gp(name, p)
        g.fit(ctx.X, ctx.y)
        base = g.base(Q)
        beliefs[codes[name]] = R.Belief(base.mean[0], base.sd[0], B.belief_embedding(base, u),
                                        B.latent_covariance_of(g, Q))
    f = beliefs["fixed"]
    mean, sd = 0.9 * f.mean + 0.05, 1.1 * f.sd
    own = R.Belief(mean, sd, mean[:, None] + sd[:, None] * norm.ppf(u)[None], None)
    beliefs["own"] = own
    M = len(Q)
    rng = np.random.default_rng(int(ctx.sites.sum()))
    label = [np.ones((M, 6))] + [np.c_[(b + 1) * own.mean, own.sd, 0.1 * rng.normal(size=(M, 4))]
                                 for b in range(1, _N_BLOCKS)]
    feats = [np.stack([Q @ rng.normal(size=(2, 6)), np.sin(3 * Q) @ rng.normal(size=(2, 6))], axis=1)
             for _ in range(_N_BLOCKS)]                                              # [M, 2, 6] geometry only
    w = np.linspace(0.2, 1.0, _N_BLOCKS)
    lens = {"quantiles": np.stack([wi * own.quantiles for wi in w]), "means": np.stack([wi * own.mean for wi in w])}
    states = BlockStates(layers=tuple(range(_N_BLOCKS)), feature_tokens=tuple(feats), label_token=tuple(label))
    return beliefs, states, lens


@pytest.fixture
def tiny_channels(monkeypatch: pytest.MonkeyPatch):
    base = gp_channel(7, 0.25, 0.3, 6, np.random.default_rng(0))
    chans = []
    for subject, emg, seed in ((0, 0, 1), (0, 1, 2), (1, 0, 3)):
        r = np.random.default_rng(seed)
        chans.append(dataclasses.replace(base, subject=subject, emg=emg,
                                         Y_trials=base.Y_trials + 0.1 * r.normal(size=base.Y_trials.shape)))
    monkeypatch.setattr(mechanism, "_channels", lambda cfg: chans)
    monkeypatch.setattr(mechanism, "_representation_beliefs", _fake_beliefs)
    from pfns4neurostim.models.pfn import tabpfn as T

    fake = types.SimpleNamespace(_model=types.SimpleNamespace(
        transformer_encoder=types.SimpleNamespace(layers=list(range(_N_BLOCKS)))))
    monkeypatch.setattr(T, "FrozenTabPFN", lambda **kw: fake)
    return chans


def _representation_config(tmp_path) -> str:
    path = tmp_path / "repr.yaml"
    path.write_text(textwrap.dedent(f"""
        defaults:
          dataset: nhp
        context: {{sizes: [10, 20]}}
        analysis: representation
        tag: tiny
        device: cpu
        seed: 0
        output_root: {tmp_path.as_posix()}
        representation:
          stress: {{knob: k2_channel, levels: [1.0, 2.0]}}
          n_context_draws: 2
          targets: [mean_own, sd_own, mean_mll, sd_mll, cov_mll, quant_mll, K_X, K_GT]
          n_quantiles: 8
          site_sets: [all, unobserved]
          gp_params: {{n_opt_steps: 10, lr: 1.0, optimizer: lbfgs, dtype: float64, noise_floor: 1.0e-3, n_restarts: 1,
                      lengthscale_floor: 0.05}}
          surrogate: {{lengthscales: [0.15, 0.3], n_fields: 16, n_replicates: 3, jitter: 1.0e-8, quantile: 0.95}}
          permutation: {{n_perm: 20}}
          inference: {{n_boot: 50, ci: 0.9}}
          posterior_cov:
            enabled: true
            engine: gp_fixed
            references: [gp_fixed, gp_mll_refit]
            coverage: {{levels: [1.0], context_ts: [10], draws: [0], channels_per_subject: 1}}
            quadrature: {{method: gauss_hermite, K: 6, u_min: 1.0e-12, central_mass: 1.0, n_scrambles: 1}}
            y_grid: {{points: 64, u_tail: 1.0e-4}}
    """))
    return str(path)


def test_representation_runner_end_to_end_cache_and_replot(tmp_path, tiny_channels, monkeypatch) -> None:
    cfg = _representation_config(tmp_path)
    out = mechanism.run_mechanism(cfg)
    assert out.replace("\\", "/").endswith("mechanism/representation/nhp/tiny")
    names = ("representation", "surrogate", "ceiling", "target_relations", "site_match", "lens", "posterior_cov",
             "representation_summary", "site_match_summary", "lens_summary", "posterior_cov_summary")
    tables = {n: pd.read_csv(os.path.join(out, f"{n}.csv")) for n in names}
    rep = tables["representation"]
    assert set(rep["readout"]) == {"label", "feature_mean", "lens"}
    assert set(rep["site_set"]) == {"all", "unobserved"}
    assert set(rep["layer"]) == set(range(_N_BLOCKS))
    # 3 channels x 2 levels x 2 t x 2 draws x 2 site sets x 3 readouts x 3 blocks x 8 targets
    assert len(rep) == 3 * 2 * 2 * 2 * 2 * 3 * _N_BLOCKS * 8
    kx = rep[(rep["target"] == "K_X") & ~rep["rep_degenerate"].astype(bool)]
    assert (kx["cka_deconf"] == 0.0).all() and kx["deconf_degenerate"].all()     # geometry is the confound itself
    block0 = rep[(rep["readout"] == "label") & (rep["layer"] == 0)]
    assert block0["rep_degenerate"].all() and block0["cka"].isna().all()         # "pre-information" block
    late = rep[(rep["readout"] == "label") & (rep["layer"] == _N_BLOCKS - 1) & (rep["target"] == "mean_own")]
    assert (late["cka"] > 0.8).all()
    assert (rep["headline_statistic"] == "deconfounded_vs_surrogate").all()
    assert {"headline_value", "headline_floor", "headline_margin", "surrogate_q_cka_deconf"} <= set(rep.columns)
    sm = tables["site_match"]
    assert set(sm["reference"]) == {"gp_mll_refit", "gp_fixed"}
    np.testing.assert_allclose(sm["w2"] ** 2, sm["location"] + sm["spread"] + sm["shape"], rtol=1e-8, atol=1e-12)
    lens = tables["lens"]
    assert set(lens["comparison"]) == {"tabpfn_v2_5", "gp_mll_refit", "gp_fixed"}
    final = lens[(lens["comparison"] == "tabpfn_v2_5") & (lens["layer"] == _N_BLOCKS - 1)]
    assert np.allclose(final["w2"], 0.0, atol=1e-10)                              # last-block lens = final predictive
    assert set(tables["ceiling"]["relation"]) == {"trial_ceiling"}
    assert "K_X" not in set(tables["ceiling"]["target_a"])                        # beliefs only
    pc = tables["posterior_cov"]
    assert len(pc) == 2 * 2                                                         # 2 channels (1 per subject) x 2 refs
    self_ref = pc[pc["reference"] == "gp_fixed"]
    assert np.allclose(self_ref["nbs"], 1.0, atol=1e-6) and np.allclose(self_ref["cka"], 1.0, atol=1e-6)
    assert (self_ref["bures_rel"] < 1e-4).all() and np.allclose(self_ref["trace_ratio"], 1.0, atol=1e-6)
    assert (pc["asymmetry"] < 1e-8).all()
    head = json.load(open(os.path.join(out, "headline.json"), encoding="utf-8"))
    assert head["status"].startswith("OPEN") and head["computable_alternatives"] == ["floor_subtracted_raw"]
    figures = ("representation_vs_layer", "representation_vs_layer_secondary", "site_match", "lens_distance",
               "posterior_cov")
    for name in figures:
        assert os.path.exists(os.path.join(out, f"{name}.svg")), name

    # --only-cached: every cell is served, nothing is computed, the tables are reproduced
    def boom(*_a, **_k):
        raise AssertionError("an --only-cached run must not compute")

    monkeypatch.setattr(mechanism, "_representation_cell", boom)
    monkeypatch.setattr(mechanism, "_posterior_cov_cell", boom)
    for n in names:
        os.remove(os.path.join(out, f"{n}.csv"))
    mechanism.run_mechanism(cfg, only_cached=True)
    for n in names:
        pd.testing.assert_frame_equal(tables[n], pd.read_csv(os.path.join(out, f"{n}.csv")), check_like=True)

    # --replot rebuilds every figure from the CSVs alone
    for name in figures:
        os.remove(os.path.join(out, f"{name}.svg"))
    mechanism.run_mechanism(cfg, replot=True)
    for name in figures:
        assert os.path.exists(os.path.join(out, f"{name}.svg")), name


def test_representation_validation_refuses_what_it_cannot_honour() -> None:
    ladder = [10, 25, 50, 80]
    v = mechanism._validate_representation
    with pytest.raises(ValueError, match="not available"):
        v({"targets": ["cov_own"]}, ladder)
    with pytest.raises(ValueError, match="unknown representation target"):
        v({"targets": ["median_mll"]}, ladder)
    with pytest.raises(ValueError, match="needs 'gp_bayes'"):
        v({"targets": ["mean_bayes"]}, ladder)
    with pytest.raises(ValueError, match="site by site"):
        v({"target_relations": [["mean_own", "mean_mll"]]}, ladder)
    with pytest.raises(ValueError, match="layout_confound"):
        v({"belief_engines": ["gp_mll_refit"], "targets": ["mean_mll"], "site_match": {"references": ["gp_mll_refit"]},
           "target_relations": []}, ladder)
    with pytest.raises(ValueError, match="headline.statistic"):
        v({"headline": {"statistic": "bogus"}}, ladder)
    with pytest.raises(ValueError, match="Unknown representation key"):
        v({"bogus": 1}, ladder)
    with pytest.raises(ValueError, match="readouts"):
        v({"readouts": ["decoder_hidden"]}, ladder)
    # GP-Bayes is optional, re-enabled by config (D3e)
    p = v({"belief_engines": ["gp_mll_refit", "gp_fixed", "gp_bayes"], "targets": ["mean_bayes", "cov_bayes"],
           "target_relations": []}, ladder)
    assert "gp_bayes" in p["belief_engines"]


def test_the_pre_registered_config_validates() -> None:
    cfg = mechanism.load_mechanism_config("configs/experiment/mechanism_representation_nhp.yaml")
    p = mechanism._validate_representation(cfg.params, cfg.context_sizes)
    assert cfg.analysis == "representation" and list(cfg.dataset.subjects) == [0, 1, 2, 3]
    assert p["headline"]["statistic"] == "deconfounded_vs_surrogate"
    assert p["belief_engines"] == ["gp_mll_refit", "gp_fixed"] and "gp_bayes" not in p["belief_engines"]
    assert p["deconfound"]["bandwidth_multipliers"] == [0.25, 0.5, 1.0, 2.0]
    assert p["surrogate"]["lengthscales"] == [0.10, 0.15, 0.30]
    assert p["posterior_cov"]["coverage"]["context_ts"] == [10, 25, 50, 80]


@pytest.mark.slow
def test_real_tabpfn_pass_lens_reproduces_the_final_predictive() -> None:
    """The real capture: states and predictive of one pass; the last block's lens equals the final predictive."""
    from pfns4neurostim.analysis.context import draw_context
    from pfns4neurostim.models.pfn.tabpfn import FrozenTabPFN

    ch = gp_channel(6, 0.3, 0.2, 4, np.random.default_rng(0))
    ctx = draw_context(ch, 12, np.random.default_rng(0))
    p = mechanism._validate_representation({"n_quantiles": 8}, [10])
    eng = FrozenTabPFN(device="cpu", random_state=0, precision="fp32")
    beliefs, states, lens = mechanism._representation_beliefs(ctx, ch.X_pool, p, eng, B.belief_levels(8))
    assert len(states.layers) == 18 and states.label_token[0].shape == (ch.n_sites, 192)
    np.testing.assert_allclose(lens["means"][-1], beliefs["own"].mean, atol=1e-5)
    np.testing.assert_allclose(lens["quantiles"][-1], beliefs["own"].quantiles, atol=1e-5)
