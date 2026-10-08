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


def _config(tmp_path, body: str, sizes: str = "[10]") -> str:
    path = tmp_path / "mech.yaml"
    path.write_text(textwrap.dedent(f"""
        defaults:
          dataset: nhp
        context: {{sizes: {sizes}}}
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
    for name in ("shape_vs_context", "surprise_response", "kernel_properties"):
        assert os.path.exists(os.path.join(out, f"{name}.svg"))
    assert not os.path.exists(os.path.join(out, "lengthscale_vs_snr.svg"))      # F4 dropped 2026-10-07
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
    ex = np.load(os.path.join(out, "update_rule_exemplar.npz"))
    assert ex["reference_gp"] == "fixed"
    # anchor: common (2026-10-07) -- every column is probed at one site, unobserved in every column's context.
    assert str(ex["anchor_rule"]).startswith("common") and len(set(ex["anchors"].tolist())) == 1
    assert os.path.exists(os.path.join(out, "shape_vs_context.svg"))


def test_make_reference_and_fixed_engine_use_the_fixed_hyperparameters() -> None:
    from pfns4neurostim.models.gp.surrogates import GP_FIXED_NOISE, GPYTORCH_DEFAULT_HYPERPARAMETER, GPSurrogate, NaiveGPSurrogate

    p = {**mechanism.UPDATE_RULE_DEFAULTS, "reference_gp": "fixed"}
    ref = mechanism._make_reference(p)
    assert isinstance(ref.gp, NaiveGPSurrogate)
    assert (ref.gp._lengthscale, ref.gp._outputscale, ref.gp._noise) == (GPYTORCH_DEFAULT_HYPERPARAMETER, 1.0, GP_FIXED_NOISE)
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
          context: {n_draws: 2, stress: {knob: k2_channel, levels: [1.0, 3.0]}}
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
          readouts: [feature_mean, feature_tokens]
          stress: {knob: k2_channel, levels: [1.0]}
          n_context_draws: 1
          n_perm: 120
          gp_params: {n_opt_steps: 20, lr: 0.1}
          controls: {grid_side: 7, lengthscale: 0.25, noise_sd: 0.3, n_trials: 6, t: 15, alpha: 0.05, n_seeds: 2, icc_min: 0.75, n_seed_channels: 1}
          placement: {enabled: true, n_prior: 6, n_prior_holdout: 2, n_noise: 2, n_draws: 1, min_gap: 0.0, bank_quantile: 0.5, n_dense: 256, holdout_frac: 0.1, rmse_threshold: 0.5, prior_type: prior_bag}
    """, sizes="[10, 15]")
    out = mechanism.run_mechanism(cfg, ["device=cuda"])
    df = pd.read_csv(os.path.join(out, "cka.csv"))
    # Layers keep their block index (0 and 17, not 0 and 1); only feature-token readouts exist.
    assert set(df["layer"]) == {0, 17} and "stage" not in df
    assert set(df["readout"]) == {"feature_mean", "feature_tokens"}
    assert {"K_X", "K_GT", "K_GP"} <= set(df["target"]) and set(df["context_t"]) == {10, 15}
    for name in ("cka_vs_layer_feature_mean", "cka_vs_layer_feature_tokens"):
        assert os.path.exists(os.path.join(out, f"{name}.svg"))
    assert not os.path.exists(os.path.join(out, "cka_readouts.svg"))
    gates = json.load(open(os.path.join(out, "gates.json")))
    assert {g["gate"] for g in gates} == {"positive_control", "negative_control", "seed_stability"}
    assert {g["readout"] for g in gates} == {"feature_mean", "feature_tokens"}
    # CKA (b): the context ladder is swept, and the work that does not depend on the channel is shared
    # (2026-09-30). Channels on one grid see the SAME context sites at a given (t, draw), so their floor and
    # ceiling must come out as identical numbers -- that identity is what the restructure bought.
    pl = pd.read_csv(os.path.join(out, "cka_placement.csv"))
    assert set(pl["context_t"]) == {10, 15}
    assert pl["grid_id"].nunique() == 1
    for _keys, part in pl.groupby(["context_t", "draw", "layer"]):
        assert part["floor1"].nunique() == 1 and part["ceiling1"].nunique() == 1
    assert "readout" not in pl                                                 # CKA (b) = feature-token mean
    for name in ("cka_placement", "cka_placement_surface"):
        assert os.path.exists(os.path.join(out, f"{name}.svg"))
    # Resume (B3 Step 3): a second run is served entirely from the cell cache and reproduces the tables.
    keys = ["readout", "subject", "emg", "level", "context_t", "rep", "layer", "target"]   # cka.csv columns
    before = df.sort_values(keys).reset_index(drop=True)
    mechanism.run_mechanism(cfg, ["device=cuda"], only_cached=True)
    after = pd.read_csv(os.path.join(out, "cka.csv")).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_shared_ladder_is_required_and_validated(tmp_path, tiny) -> None:
    """B3 Step 4: one ladder, composed or inline; a missing ladder or an analysis subset outside it fails."""
    body = """
        analysis: update_rule
        update_rule:
          engines: [gp_mll_frozen]
          layer_arm: {enabled: true, context_ts: [25], level: 1.0, ridge: 1.0}
    """
    with pytest.raises(ValueError, match="not in the shared ladder"):
        mechanism.run_mechanism(_config(tmp_path, body))
    path = tmp_path / "noladder.yaml"
    path.write_text("defaults:\n  dataset: nhp\nanalysis: update_rule\n")
    with pytest.raises(ValueError, match="no context ladder"):
        mechanism.load_mechanism_config(str(path))
    bogus = tmp_path / "bogus.yaml"
    bogus.write_text("defaults:\n  dataset: nhp\ncontext: {sizes: [10], bogus: 1}\nanalysis: update_rule\n")
    with pytest.raises(ValueError, match="Unknown context key"):
        mechanism.load_mechanism_config(str(bogus))
    assert mechanism.load_mechanism_config("configs/experiment/mechanism_cka_nhp.yaml").context_sizes == (10, 25, 50, 80)
    for name in ("mechanism_update_rule_nhp", "mechanism_placement_nhp", "mechanism_placement_5d_rat"):
        assert mechanism.load_mechanism_config(f"configs/experiment/{name}.yaml").context_sizes
    assert mechanism.load_mechanism_config(
        "configs/experiment/mechanism_cka_tokens_nhp.yaml").context_sizes == (25,)


def _small_update_rule(tmp_path) -> str:
    return _config(tmp_path, """
        analysis: update_rule
        update_rule:
          engines: [gp_fixed_frozen, gp_mll_frozen]
          stress: {knob: k2_channel, levels: [1.0]}
          n_context_draws: 2
          n_anchors: 4
          gp_params: {n_opt_steps: 20, lr: 0.1}
          positive_control: {grid_side: 7, lengthscale: 0.25, noise_sd: 0.3, n_trials: 6}
          profile: {enabled: true, bin_edges: [0.0, 0.5, 1.5, 2.5, 3.5], far_pitch: 3.0, level: 1.0, panel_ts: [10],
                    headline_gp: gp_fixed_frozen, offmap: {enabled: true, distance: 2.0},
                    controls: {known_gp: true, shuffled: true, context_t: 10, n_channels: 1}}
    """)


def test_cells_resume_and_only_cached_reassemble_identically(tmp_path, tiny, monkeypatch) -> None:
    """B3 Steps 2-3: every cell is persisted; --only-cached recomputes nothing and reproduces the CSVs."""
    from pfns4neurostim.analysis import update_rule as U

    cfg = _small_update_rule(tmp_path)
    out = mechanism.run_mechanism(cfg)
    tables = {n: pd.read_csv(os.path.join(out, n)) for n in (
        "update_rule.csv", "update_rule_cell.csv", "update_rule_profile.csv", "update_rule_offmap.csv",
        "update_rule_profile_controls.csv")}
    cell_dir = tmp_path / "cells" / "nhp" / "mechanism_update_rule"
    assert len(list(cell_dir.glob("*.json"))) > 0
    for n in tables:
        os.remove(os.path.join(out, n))

    def boom(*_a, **_k):
        raise AssertionError("an --only-cached run must not compute")

    monkeypatch.setattr(U, "run_probes", boom)
    monkeypatch.setattr(U, "_probe_cell", boom)
    mechanism.run_mechanism(cfg, only_cached=True)
    for n, before in tables.items():
        pd.testing.assert_frame_equal(before, pd.read_csv(os.path.join(out, n)), check_like=True)


def test_failed_cell_is_reported_and_only_it_is_recomputed(tmp_path, tiny, monkeypatch) -> None:
    """One failing cell no longer kills the grid: the run finishes, raises a listing, and a re-run retries it."""
    from pfns4neurostim.analysis import update_rule as U

    real = U.run_probes
    calls = {"n": 0, "failed": False}

    def flaky(engine, reference, ctx, Q, anchors, strata, surprises=U.DEFAULT_SURPRISES):
        calls["n"] += 1
        if not calls["failed"] and len(anchors) > 1:
            calls["failed"] = True
            raise RuntimeError("injected")
        return real(engine, reference, ctx, Q, anchors, strata, surprises)

    monkeypatch.setattr(U, "run_probes", flaky)
    cfg = _small_update_rule(tmp_path)
    with pytest.raises(RuntimeError, match=r"1 cell\(s\) failed"):
        mechanism.run_mechanism(cfg)
    first = calls["n"]
    mechanism.run_mechanism(cfg)
    # Only the failed probe cell and the off-map cell that hangs off it (skipped while its probe was missing)
    # are computed again; every other cell is served from the cache.
    assert calls["n"] - first == 2


def test_f7_profile_columns_and_figures(tmp_path, tiny) -> None:
    """F7: cell metrics, binned profile, off-map and control tables, main and supplement figures."""
    out = mechanism.run_mechanism(_small_update_rule(tmp_path))
    cell = pd.read_csv(os.path.join(out, "update_rule_cell.csv"))
    for col in ("far_field_share", "locality_index", "offset_share", "offset_fit_ell"):
        assert col in cell
    assert cell["far_field_share"].between(0, 1).all() and cell["offset_share"].between(0, 1).all()
    assert cell["locality_index"].le(1.0 + 1e-9).all()
    prof = pd.read_csv(os.path.join(out, "update_rule_profile.csv"))
    un = prof[~prof["observed"].astype(bool)]
    assert un["cum_energy_share"].between(0, 1 + 1e-9).all()
    ctrl = pd.read_csv(os.path.join(out, "update_rule_profile_controls.csv"))
    assert {"known_gp", "original", "shuffled"} <= set(ctrl["control"]) and "truth" in set(ctrl["engine"])
    off = pd.read_csv(os.path.join(out, "update_rule_offmap.csv"))
    # A GP's update on the grid from an observation two grid widths away is ~0 (relative to the full surprise).
    assert (off["offmap_transfer_median"] < 0.05).all()
    for name in ("update_vs_distance", "update_vs_distance_controls"):
        assert os.path.exists(os.path.join(out, f"{name}.svg"))


def test_empty_csv_is_treated_as_absent(tmp_path) -> None:
    """An empty frame still writes a newline (CRLF on Windows); the figure code must skip it, not crash (2026-10-06)."""
    from pfns4neurostim.visualization import mechanism as figs

    path = tmp_path / "cka_placement.csv"
    pd.DataFrame([]).to_csv(path, index=False)
    path.write_bytes(path.read_bytes() + b"\r\n")
    assert not figs._has_rows(str(path)) and not figs._has_rows(str(tmp_path / "missing.csv"))
    assert figs.render_cka(str(tmp_path)) == []
    pd.DataFrame([{"a": 1}]).to_csv(path, index=False)
    assert figs._has_rows(str(path))


def test_exemplar_gp_row_can_differ_from_the_metric_reference(tmp_path, tiny) -> None:
    """F1's GP row follows exemplar.reference_gp (2026-10-07), while every metric keeps update_rule.reference_gp."""
    body = """
        analysis: update_rule
        update_rule:
          engines: [gp_mll_frozen, gp_fixed_frozen]
          reference_gp: mll
          stress: {knob: k2_channel, levels: [1.0]}
          n_context_draws: 1
          n_anchors: 4
          gp_params: {n_opt_steps: 20, lr: 0.1}
          positive_control: {grid_side: 7, lengthscale: 0.25, noise_sd: 0.3, n_trials: 6}
          exemplar: {context_ts: [10], level: 1.0, draw: 0, anchor_stratum: centre, select: first, subject: null,
                     emg: null, anchor: common, reference_gp: fixed}
    """
    out = mechanism.run_mechanism(_config(tmp_path, body))
    ex = np.load(os.path.join(out, "update_rule_exemplar.npz"))
    assert str(ex["reference_gp"]) == "fixed"
    assert set(pd.read_csv(os.path.join(out, "update_rule_cell.csv"))["reference_gp"]) == {"mll"}
    with pytest.raises(ValueError, match="anchor: common"):
        mechanism.run_mechanism(_config(tmp_path, body.replace("anchor: common", "anchor: per_cell")))
