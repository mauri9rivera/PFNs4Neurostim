"""End-to-end mechanism runner on a tiny known-GP channel (GP engines only; fast)."""
from __future__ import annotations

import json
import os
import textwrap

import numpy as np
import pandas as pd
import pytest

from pfns4neurostim.analysis.predictive_link import READOUTS
from pfns4neurostim.data.synthetic_neurostim import gp_channel
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
    # The rank placement is the REPORTED statistic (2026-10-12) and the ratio ``p`` stays on disk as the frozen
    # reference it is compared against. The rank is bounded in [0, 1] by construction -- the whole reason it
    # replaced the ratio, whose denominator ``gap`` has a median of 0.049 on the real NHP run.
    assert {"rank_vs_floor", "rank_vs_ceiling", "p_rank", "p"} <= set(pl.columns)
    for col in ("rank_vs_floor", "rank_vs_ceiling", "p_rank"):
        assert pl[col].between(0.0, 1.0).all(), f"{col} left [0, 1]"
    assert np.allclose(pl["p_rank"], 0.5 * (pl["rank_vs_floor"] + pl["rank_vs_ceiling"]))
    assert (pl["n_floor_refs"] > 0).all() and (pl["n_ceiling_refs"] > 0).all()
    for name in ("cka_placement", "cka_placement_surface",
                 "cka_placement_ratio", "cka_placement_ratio_surface"):
        assert os.path.exists(os.path.join(out, f"{name}.svg"))
    # Resume (B3 Step 3): a second run is served entirely from the cell cache and reproduces the tables.
    keys = ["readout", "subject", "emg", "level", "context_t", "rep", "layer", "target"]   # cka.csv columns
    before = df.sort_values(keys).reset_index(drop=True)
    mechanism.run_mechanism(cfg, ["device=cuda"], only_cached=True)
    after = pd.read_csv(os.path.join(out, "cka.csv")).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_shared_ladder_is_required_and_validated(tmp_path, tiny) -> None:
    """B3 Step 4: one ladder, composed or inline; a missing ladder or an unknown context key fails."""
    path = tmp_path / "noladder.yaml"
    path.write_text("defaults:\n  dataset: nhp\nanalysis: cka\n")
    with pytest.raises(ValueError, match="no context ladder"):
        mechanism.load_mechanism_config(str(path))
    bogus = tmp_path / "bogus.yaml"
    bogus.write_text("defaults:\n  dataset: nhp\ncontext: {sizes: [10], bogus: 1}\nanalysis: cka\n")
    with pytest.raises(ValueError, match="Unknown context key"):
        mechanism.load_mechanism_config(str(bogus))
    assert mechanism.load_mechanism_config("configs/experiment/mechanism_cka_nhp.yaml").context_sizes == (10, 25, 50, 80)
    for name in ("mechanism_bayes_update_nhp", "mechanism_placement_nhp", "mechanism_placement_5d_rat"):
        assert mechanism.load_mechanism_config(f"configs/experiment/{name}.yaml").context_sizes
    assert mechanism.load_mechanism_config(
        "configs/experiment/mechanism_cka_tokens_nhp.yaml").context_sizes == (25,)


def _small_bayes_update(tmp_path, **extra: str) -> str:
    body = """
        analysis: bayes_update
        bayes_update:
          engines: [gp_bayes, gp_fixed]
          references: [gp_bayes]
          stress: {knob: k2_channel, levels: [1.0]}
          n_context_draws: 1
          anchors: {n_unobserved: 2, n_observed: 1}
          quadrature: {K: 4, u_min: 1.0e-12}
          y_grid: {points: 128, u_tail: 1.0e-4}
          shared: {n_trials: 3, ladder: [-3.0, -1.0, 1.0, 3.0], saturation_c: [1.0, 3.0]}
          far_pitch: 3.0
          gp_bayes: {lengthscale: [0.1, 0.6, 3], outputscale: [0.5, 2.0, 2], noise: [0.05, 0.3, 2], b2: 1.0,
                     prune_mass: 1.0e-10, chunk: 64, exact_max_components: 64, sd_bins: 8}
          exemplar: {enabled: true, subject: null, emg: null, level: 1.0, draw: 0, engines: [gp_bayes, gp_fixed]}
          panel_ts: [10]
          run: {gates: false, grid: true, discrepancy: false}
    """
    return _config(tmp_path, body)


def test_bayes_update_runner_writes_every_table_and_figure_and_reassembles_from_cache(
        tmp_path, tiny, monkeypatch) -> None:
    """OMEGA Step 5: GP engines on a tiny known-GP channel; --only-cached recomputes 0 cells."""
    out = mechanism.run_mechanism(_small_bayes_update(tmp_path))
    tables = {n: pd.read_csv(os.path.join(out, f"{n}.csv")) for n in (
        "coherence", "coherence_cell", "headtohead", "surprise")}
    assert set(tables["coherence"]["engine"]) == {"gp_bayes", "gp_fixed"}
    assert set(tables["coherence"]["pool"]) == {"unobserved", "observed"}
    assert {"host", "seconds"} <= set(tables["coherence_cell"].columns)
    fixed = tables["coherence_cell"][tables["coherence_cell"]["engine"] == "gp_fixed"]
    assert (fixed["drift_abs"].abs() < 1e-8).all() and (fixed["cov_asymmetry"] < 1e-8).all()
    assert tables["coherence_cell"].loc[tables["coherence_cell"]["engine"] == "gp_bayes", "ess"].notna().all()
    h = tables["headtohead"]
    assert set(h.loc[h["reference"] == "gp_bayes", "engine"]) == {"gp_fixed"}
    for name in ("coherence", "implied_noise", "headtohead", "surprise_response", "implied_cov_exemplar"):
        assert os.path.exists(os.path.join(out, f"{name}.svg")), name
    assert os.path.exists(os.path.join(out, "exemplar.npz"))
    for n in tables:
        os.remove(os.path.join(out, f"{n}.csv"))

    def boom(*_a, **_k):
        raise AssertionError("an --only-cached run must not compute")

    monkeypatch.setattr(mechanism, "_bayes_cell", boom)
    mechanism.run_mechanism(_small_bayes_update(tmp_path), only_cached=True)
    for n, before in tables.items():
        pd.testing.assert_frame_equal(before, pd.read_csv(os.path.join(out, f"{n}.csv")), check_like=True)


def test_bayes_update_refuses_unknown_engines_and_ladders_without_saturation_points(tmp_path, tiny) -> None:
    with pytest.raises(ValueError, match="unknown"):
        mechanism.run_mechanism(_config(tmp_path, """
            analysis: bayes_update
            bayes_update: {engines: [gp_bayes, gp_unknown]}
        """))
    with pytest.raises(ValueError, match="Part B reference only"):
        mechanism.run_mechanism(_config(tmp_path, """
            analysis: bayes_update
            bayes_update: {engines: [gp_bayes, gp_mll_frozen], references: [gp_bayes, gp_mll_frozen],
                           coverage: {gp_mll_frozen: {A: {levels: [1.0]}}}}
        """))
    with pytest.raises(ValueError, match="saturation_c"):
        mechanism.run_mechanism(_config(tmp_path, """
            analysis: bayes_update
            bayes_update: {shared: {n_trials: 3, ladder: [-1.0, 1.0], saturation_c: [1.0, 3.0]}}
        """))
    with pytest.raises(ValueError, match="not in the shared ladder"):
        mechanism.run_mechanism(_config(tmp_path, """
            analysis: bayes_update
            bayes_update: {panel_ts: [25]}
        """))


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


# ---------------------------------------------------------------------------
# M8 predictive link (2026-10-21): synthetic OMEGA tables, no engine runs
# ---------------------------------------------------------------------------
_LADDER = [10, 25, 50, 80]


def _omega_tables(run_dir, n_subjects: int = 3, n_emg: int = 5, seed: int = 0) -> pd.DataFrame:
    """Write coherence_cell.csv / headtohead.csv for n_subjects x n_emg channels; return the per-channel drift."""
    rng = np.random.default_rng(seed)
    coh, h2h, truth = [], [], []
    for s in range(n_subjects):
        for e in range(n_emg):
            drift = rng.uniform(0.01, 0.1)
            truth.append({"dataset": "nhp", "subject": s, "emg": e, "drift": drift})
            for level in (1.0, 2.0):
                for t in (10, 25):
                    cell = {"dataset": "nhp", "subject": s, "emg": e, "level": level, "context_t": t, "draw": 0}
                    for pool in ("unobserved", "observed"):
                        for eng, d in (("tabpfn_v2_5", drift), ("gp_bayes", 0.0), ("gp_mll_refit", drift / 2)):
                            coh.append({**cell, "engine": eng, "pool": pool, "drift_abs": d,
                                        "total_var_ratio": 1 + rng.normal(scale=0.05),
                                        "cov_asymmetry": rng.uniform(), "latent_frac": rng.uniform(),
                                        "sigma_recovery_rho": rng.uniform()})
                        for a in range(3):
                            base = {**cell, "pool": pool, "anchor": a}
                            for ref in ("gp_bayes", "gp_mll_refit", "gp_mll_frozen"):
                                h2h.append({**base, "source": "trial", "engine": "tabpfn_v2_5", "reference": ref,
                                            "transfer_slope": rng.normal(1), "footprint_r": rng.uniform(),
                                            "gap_change": rng.normal(), "delta_crps": rng.normal(scale=0.01)})
                            for eng in ("tabpfn_v2_5", "gp_bayes", "gp_mll_refit", "gp_mll_frozen"):
                                h2h.append({**base, "source": "trial", "engine": eng, "reference": "none",
                                            "gain": rng.uniform(), "far_field_share": rng.uniform()})
                                h2h.append({**base, "source": "ladder", "engine": eng, "reference": "none",
                                            "saturation": rng.normal(1, 0.1)})
    pd.DataFrame(coh).to_csv(os.path.join(run_dir, "coherence_cell.csv"), index=False)
    pd.DataFrame(h2h).to_csv(os.path.join(run_dir, "headtohead.csv"), index=False)
    return pd.DataFrame(truth)


def _link_tidies(run_dir, truth: pd.DataFrame, slope: float = 4.0, seed: int = 1) -> tuple[str, str]:
    """A BO tidy whose recommended regret is planted on drift_abs, and a stress tidy (+ config.yaml margin)."""
    rng = np.random.default_rng(seed)
    bo, st = [], []
    for row in truth.itertuples():
        key = {"dataset": "nhp", "subject": row.subject, "emg": row.emg, "acq_label": "ts_marginal"}
        for rep in range(6):
            for model in ("tabpfn_v2_5", "gp_mll"):
                bo.append({**key, "model": model, "rep": rep, "r2": rng.uniform(0.5, 0.9),
                           "recommended_regret": 0.1 * row.subject + slope * row.drift + 0.01 * rng.normal(),
                           "cumulative_regret": rng.uniform(40, 60)})
            for level in (1.0, 2.0, 4.0):
                ref = 0.2 + 0.01 * rng.normal()
                broke = level >= (2.0 if row.drift > 0.05 else 99.0)
                for model, val in (("tabpfn_v2_5", ref + (0.5 if broke else 0.0) + 0.01 * rng.normal()),
                                   ("gp_mll", ref)):
                    st.append({**key, "model": model, "rep": rep, "level": level, "knob": "k2_channel",
                               "demo": "draws", "recommended_regret": val})
    bo_dir, st_dir = os.path.join(run_dir, "bo"), os.path.join(run_dir, "stress")
    os.makedirs(bo_dir)
    os.makedirs(st_dir)
    pd.DataFrame(bo).to_csv(os.path.join(bo_dir, "tidy.csv"), index=False)
    pd.DataFrame(st).to_csv(os.path.join(st_dir, "tidy.csv"), index=False)
    with open(os.path.join(st_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write("equivalence_margin: 0.05\n")
    return os.path.join(bo_dir, "tidy.csv"), os.path.join(st_dir, "tidy.csv")


def _resolved_link(**link: object) -> dict:
    return mechanism._validate_bayes_update({"link": link, "stress": {"knob": "k2_channel", "levels": [1.0, 2.0]}},
                                            _LADDER)["link"]


def test_link_defaults_are_the_pre_extension_block() -> None:
    link = _resolved_link()
    assert link["tidy_csv"] is None and link["stress_tidy_csv"] is None
    assert link["references"] == ["gp_bayes"] and link["engine_contrasts"] is False
    assert link["predictors"] == ["drift_abs", "total_var_ratio", "transfer_slope", "delta_crps"]
    assert link["outcomes"] == ["recommended_regret", "cumulative_regret"] and link["pool"] == "unobserved"
    assert link["breakdown"] == {"reference": "gp_mll", "metric": "recommended_regret", "margin": None}


def test_link_validation_refuses_what_it_cannot_honour() -> None:
    with pytest.raises(ValueError, match="link.predictors: unknown"):
        _resolved_link(predictors=["drift_abs", "nope"])
    with pytest.raises(ValueError, match="not Part B references"):
        _resolved_link(references=["gp_mll_frozen"])                 # not in this run's bayes_update.references
    with pytest.raises(ValueError, match="outcomes: unknown"):
        _resolved_link(outcomes=["regret"])
    with pytest.raises(ValueError, match="needs link.stress_tidy_csv"):
        _resolved_link(tidy_csv="x.csv", outcomes=["r2", "breakdown"])
    with pytest.raises(ValueError, match="need link.tidy_csv"):
        _resolved_link(stress_tidy_csv="x.csv", outcomes=["breakdown", "r2"])
    with pytest.raises(ValueError, match="not a stress level"):
        _resolved_link(tidy_csv="x.csv", level=4.0)
    with pytest.raises(ValueError, match="not in the shared ladder"):
        _resolved_link(context_ts=[11])
    with pytest.raises(ValueError, match="Unknown bayes_update.link.breakdown"):
        _resolved_link(breakdown={"margin": 0.05, "test": "tost"})
    assert _resolved_link(predictors="all")["predictors"] == list(READOUTS)


def test_predictive_link_default_block_recovers_a_planted_slope(tmp_path) -> None:
    truth = _omega_tables(str(tmp_path))
    bo, _ = _link_tidies(str(tmp_path), truth)
    paths = mechanism._predictive_link(str(tmp_path), _resolved_link(tidy_csv=bo, n_boot=200), seed=0)
    res = pd.read_csv(paths[0])
    assert set(zip(res["predictor"], res["reference"])) == {
        ("drift_abs", "none"), ("total_var_ratio", "none"), ("transfer_slope", "gp_bayes"), ("delta_crps", "gp_bayes")}
    row = res[(res["predictor"] == "drift_abs") & (res["outcome"] == "recommended_regret")].iloc[0]
    assert row["slope"] == pytest.approx(4.0, abs=0.2) and row["ci_low"] < 4.0 < row["ci_high"]
    assert (row["n_channels"], row["n_subjects"], row["pool"], row["level"]) == (15, 3, "unobserved", 1.0)
    assert os.path.exists(paths[1])


def test_predictive_link_every_reference_and_outcome_then_figures(tmp_path) -> None:
    from pfns4neurostim.visualization import mechanism as figs

    truth = _omega_tables(str(tmp_path))
    bo, stress = _link_tidies(str(tmp_path), truth)
    link = _resolved_link(tidy_csv=bo, stress_tidy_csv=stress, predictors="all", references="all",
                          engine_contrasts=True, outcomes=["recommended_regret", "r2", "breakdown"], n_boot=50)
    res = pd.read_csv(mechanism._predictive_link(str(tmp_path), link, seed=0)[0])
    assert set(res["outcome"]) == {"recommended_regret", "r2", "breakdown"}
    frozen = res[res["reference"] == "gp_mll_frozen"]
    assert set(frozen["predictor"]) == {"transfer_slope", "footprint_r", "gain", "far_field_share", "saturation"}
    assert not ((res["reference"] == "gp_mll_frozen") & (res["part"] == "A")).any()
    assert set(res.loc[res["predictor"] == "drift_abs", "reference"]) == {"none", "gp_bayes", "gp_mll_refit"}
    # the planted breakdown: channels with drift > 0.05 break at level 2 (1 level passed), the others never (3)
    pts = pd.read_csv(os.path.join(tmp_path, "predictive_link_channels.csv"))
    one = pts[(pts["predictor"] == "drift_abs") & (pts["reference"] == "none")].merge(truth, on=["dataset", "subject",
                                                                                             "emg"])
    assert (one["breakdown"] == np.where(one["drift"] > 0.05, 1.0, 3.0)).all()
    bd = res[(res["predictor"] == "drift_abs") & (res["reference"] == "none") & (res["outcome"] == "breakdown")]
    assert bd["slope"].iloc[0] < 0 and bd["ci_high"].iloc[0] < 0
    written = figs.render_predictive_link(str(tmp_path))
    assert sorted(os.path.basename(p) for p in written) == [
        "predictive_link_breakdown.svg", "predictive_link_r2.svg", "predictive_link_recommended_regret.svg"]


def test_predictive_link_single_subject_and_unknown_reference(tmp_path) -> None:
    truth = _omega_tables(str(tmp_path), n_subjects=1, n_emg=8)
    bo, _ = _link_tidies(str(tmp_path), truth)
    res = pd.read_csv(mechanism._predictive_link(str(tmp_path), _resolved_link(tidy_csv=bo, n_boot=100), seed=0)[0])
    assert (res["n_subjects"] == 1).all() and (res["re_ratio"] == 0.0).all()
    with pytest.raises(ValueError, match="has no tabpfn_v2_5 contrast"):
        mechanism._predictive_link(str(tmp_path), _resolved_link(tidy_csv=bo, references=["gp_fixed"]), seed=0)


class TestStressSeriesAreLabelledInSnr:
    """Hyp C stress series read as achieved SNR, not as raw knob levels (user, 2026-10-12).

    The mechanism panels used to label their K2 ladder ``level 1 / 2 / 4``, which no reader could place
    against the Hyp B x-axis; both are now the same quantity in the same units, and the unstressed level is
    named "default" so the per-dataset anchor is visible in either hypothesis.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"knob": "k2_channel", "level": level, "achieved_snr_db": snr}
                for level, snr in ((1.0, 3.98), (2.0, -2.05), (4.0, -8.07))
            ]
        )

    def test_labels_are_snr_with_the_default_named(self) -> None:
        from pfns4neurostim.visualization import mechanism as figs

        labels = figs._level_snr_labels(self._frame())
        assert labels == {1.0: "4.0 dB (default)", 2.0: "-2.0 dB", 4.0: "-8.1 dB"}

    def test_falls_back_to_the_knob_symbol_without_an_snr_column(self) -> None:
        """A missing column must be visible as the raw parameter, never silently mislabelled as dB."""
        from pfns4neurostim.visualization import mechanism as figs

        labels = figs._level_snr_labels(self._frame().drop(columns=["achieved_snr_db"]))
        assert labels == {1.0: r"$\gamma$ = 1", 2.0: r"$\gamma$ = 2", 4.0: r"$\gamma$ = 4"}

    def test_the_placement_legend_title_is_the_snr_axis(self) -> None:
        from pfns4neurostim.visualization import mechanism as figs

        assert "SNR" in figs.PLACEMENT_LEVEL_LEGEND_TITLE and "alpha" not in figs.PLACEMENT_LEVEL_LEGEND_TITLE
