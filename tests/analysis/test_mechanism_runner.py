"""End-to-end mechanism runner on a tiny known-GP channel (GP engines only; fast)."""
from __future__ import annotations

import json
import os
import textwrap

import numpy as np
import pandas as pd
import pytest

from pfns4neurostim.analysis.update_rule import gp_channel
from pfns4neurostim.experiments import mechanism


@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch):
    ch = gp_channel(7, 0.25, 0.3, 6, np.random.default_rng(0))
    monkeypatch.setattr(mechanism, "_channels", lambda cfg: [ch])
    return ch


def _config(tmp_path, body: str) -> str:
    path = tmp_path / "mech.yaml"
    path.write_text(textwrap.dedent(f"""
        defaults:
          dataset: nhp
        tag: tiny
        device: cpu
        seed: 0
        output_root: {tmp_path.as_posix()}
    """) + textwrap.dedent(body))
    return str(path)


def test_update_rule_runner_writes_tables_gates_and_figures(tmp_path, tiny) -> None:
    cfg = _config(tmp_path, """
        analysis: update_rule
        update_rule:
          engines: [gp_mll_frozen, gp_mll_refit]
          context_sizes: [10]
          stress: {knob: k2_channel, levels: [1.0, 2.0]}
          n_context_draws: 1
          n_anchors: 4
          refit_surprises: [-0.5, 0.5]
          gp_params: {n_opt_steps: 20, lr: 0.1}
          positive_control: {grid_side: 7, lengthscale: 0.25, noise_sd: 0.3, n_trials: 6}
    """)
    out = mechanism.run_mechanism(cfg)
    anchor = pd.read_csv(os.path.join(out, "update_rule.csv"))
    assert set(anchor["engine"]) == {"gp_mll_frozen", "gp_mll_refit"}
    assert set(anchor["level"]) == {1.0, 2.0}
    gates = json.load(open(os.path.join(out, "gates.json")))
    probe = [g for g in gates if g.get("role") == "probe_validity"]
    assert probe and probe[0]["passed"]
    for name in ("shape_vs_context", "surprise_response", "lengthscale_vs_snr", "kernel_properties"):
        assert os.path.exists(os.path.join(out, f"{name}.svg"))
    # --replot rebuilds from CSVs alone
    os.remove(os.path.join(out, "kernel_properties.svg"))
    mechanism.run_mechanism(cfg, replot=True)
    assert os.path.exists(os.path.join(out, "kernel_properties.svg"))


def test_fixed_reference_runs_end_to_end_and_is_recorded(tmp_path, tiny) -> None:
    """reference_gp: fixed measures every arm against the GP-fixed kernel (no fitting); the GP-fixed arm is then identical to
    the reference by construction, and the choice is written into the rows and the exemplar so figures label it."""
    cfg = _config(tmp_path, """
        analysis: update_rule
        update_rule:
          engines: [gp_fixed_frozen, gp_mll_frozen]
          reference_gp: fixed
          context_sizes: [10]
          stress: {knob: k2_channel, levels: [1.0]}
          n_context_draws: 1
          n_anchors: 4
          gp_params: {n_opt_steps: 20, lr: 0.1}
          positive_control: {grid_side: 7, lengthscale: 0.25, noise_sd: 0.3, n_trials: 6}
          exemplar: {context_ts: [10], level: 1.0, draw: 0, anchor_stratum: centre, select: first, subject: null, emg: null}
    """)
    out = mechanism.run_mechanism(cfg)
    cell = pd.read_csv(os.path.join(out, "update_rule_cell.csv"))
    assert set(cell["reference_gp"]) == {"fixed"}
    fixed_arm = cell[cell["engine"] == "gp_fixed_frozen"]
    assert (fixed_arm["rho_shape_median"] > 0.99).all()           # the reference measured against itself
    assert np.load(os.path.join(out, "update_rule_exemplar.npz"))["reference_gp"] == "fixed"
    assert os.path.exists(os.path.join(out, "shape_vs_context.svg"))


def test_make_reference_and_fixed_engine_use_the_fixed_hyperparameters() -> None:
    from pfns4neurostim.models.gp.surrogates import GPSurrogate, NaiveGPSurrogate

    p = {**mechanism.UPDATE_RULE_DEFAULTS, "reference_gp": "fixed"}
    ref = mechanism._make_reference(p)
    assert isinstance(ref.gp, NaiveGPSurrogate)
    assert (ref.gp._lengthscale, ref.gp._outputscale, ref.gp._noise) == (0.2, 1.0, 0.01)
    arm = mechanism._make_engine("gp_fixed_frozen", p, "cpu", 0)
    assert isinstance(arm.gp, NaiveGPSurrogate)
    assert type(mechanism._make_reference({**p, "reference_gp": "mll"}).gp) is GPSurrogate
    with pytest.raises(ValueError, match="reference_gp"):
        mechanism._make_reference({**p, "reference_gp": "bogus"})


def test_unknown_block_key_raises(tmp_path, tiny) -> None:
    cfg = _config(tmp_path, """
        analysis: update_rule
        update_rule:
          bogus: 1
    """)
    with pytest.raises(ValueError, match="bogus"):
        mechanism.run_mechanism(cfg)


def test_placement_runner(tmp_path, tiny, monkeypatch: pytest.MonkeyPatch) -> None:
    """Formulations B, C, A with an injected smooth prior sampler (no libs/ prior needed)."""
    from pfns4neurostim.data.references import grid as G

    real = G.prior_grid_bank

    def smooth_sampler(d: int, n: int, s: int):
        r = np.random.default_rng(s)
        X = r.uniform(size=(n, d))
        w = r.normal(size=d) * 4
        return X, np.sin(X @ w + r.uniform(0, 6))

    monkeypatch.setattr(G, "prior_grid_bank", lambda *a, **k: real(*a, **{**k, "sampler": smooth_sampler}))
    cfg = _config(tmp_path, """
        analysis: placement
        placement:
          n_prior: 12
          n_prior_holdout: 4
          n_noise: 4
          n_dense: 256
          k_nearest: 3
          n_splits: 2
          n_shuffles: 2
          n_boot: 50
          context: {sizes: [10], n_draws: 2, stress: {knob: k2_channel, levels: [1.0, 3.0]}}
    """)
    out = mechanism.run_mechanism(cfg)
    df = pd.read_csv(os.path.join(out, "placement.csv"))
    assert set(df["metric"]) == {"mmd", "w2"}
    for col in ("p_f1c1", "p_f2c2", "floor2", "ceiling2", "gap_f1c1"):
        assert col in df
    ctx = pd.read_csv(os.path.join(out, "placement_context.csv"))
    assert set(ctx["level"]) == {1.0, 3.0}
    summary = json.load(open(os.path.join(out, "placement_summary.json")))
    assert summary["banks"][0]["n_attempted"] == 16
    assert {g["metric"] for g in summary["gates"]} == {"mmd", "w2"}
    # The formulation-B strip figure was dropped on 2026-09-30 (its pair columns live on as CSV columns
    # only); the trajectory panel is the placement deliverable.
    assert not os.path.exists(os.path.join(out, "placement_strip.svg"))
    assert os.path.exists(os.path.join(out, "placement_context.svg"))


def test_cka_runner_rejects_unresolvable_permutation_count(tmp_path, tiny) -> None:
    cfg = _config(tmp_path, """
        analysis: cka
        cka:
          n_perm: 100
    """)
    with pytest.raises(ValueError, match="Bonferroni"):
        mechanism.run_mechanism(cfg)


@pytest.mark.slow
@pytest.mark.gpu
def test_cka_runner_end_to_end(tmp_path, tiny, monkeypatch: pytest.MonkeyPatch) -> None:
    from pfns4neurostim.data.references import grid as G

    real = G.prior_grid_bank

    def smooth_sampler(d: int, n: int, s: int):
        r = np.random.default_rng(s)
        X = r.uniform(size=(n, d))
        return X, np.sin(X @ (r.normal(size=d) * 4))

    monkeypatch.setattr(G, "prior_grid_bank", lambda *a, **k: real(*a, **{**k, "sampler": smooth_sampler}))
    cfg = _config(tmp_path, """
        analysis: cka
        cka:
          layers: [0, 17]
          context_sizes: [15]
          stress: {knob: k2_channel, levels: [1.0]}
          n_context_draws: 1
          n_perm: 60
          gp_params: {n_opt_steps: 20, lr: 0.1}
          controls: {grid_side: 7, lengthscale: 0.25, noise_sd: 0.3, n_trials: 6, t: 15, alpha: 0.05, n_seeds: 2, icc_min: 0.75, n_seed_channels: 1}
          placement: {enabled: true, ts: [10, 15], n_prior: 6, n_prior_holdout: 2, n_noise: 2, n_draws: 1, min_gap: 0.0, bank_quantile: 0.5, n_dense: 256, holdout_frac: 0.1, rmse_threshold: 0.5, prior_type: prior_bag}
    """.replace("device: cuda\n", ""))
    out = mechanism.run_mechanism(cfg, ["device=cuda"])
    df = pd.read_csv(os.path.join(out, "cka.csv"))
    assert set(df["layer"]) == {0, 1} and {"K_X", "K_GT", "K_GP"} <= set(df["target"])
    assert os.path.exists(os.path.join(out, "cka_vs_layer.svg"))
    gates = json.load(open(os.path.join(out, "gates.json")))
    assert {g["gate"] for g in gates} == {"positive_control", "negative_control", "seed_stability"}
    # CKA (b): the context ladder is swept, and the work that does not depend on the channel is shared
    # (2026-09-30). Channels on one grid see the SAME context sites at a given (t, draw), so their floor and
    # ceiling must come out as identical numbers -- that identity is what the restructure bought.
    pl = pd.read_csv(os.path.join(out, "cka_placement.csv"))
    assert set(pl["context_t"]) == {10, 15}
    assert pl["grid_id"].nunique() == 1
    for _keys, part in pl.groupby(["context_t", "draw", "layer"]):
        assert part["floor1"].nunique() == 1 and part["ceiling1"].nunique() == 1
    assert os.path.exists(os.path.join(out, "cka_placement.svg"))
