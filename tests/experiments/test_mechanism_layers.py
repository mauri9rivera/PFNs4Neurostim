"""Layer-dynamics mechanism runner (F6-E Step 1/4, sprint #3).

Config-validation tests are fast (no TabPFN checkpoint touched: the runner validates
``layer_dynamics.engine``, ``.gates``, ``.probe`` and ``.context_ts`` before any channel is
loaded or any model is fitted). The end-to-end smoke test and the ``_fact_sheet`` test load the
real ``tabpfn==6.3.2`` checkpoint on CPU and are marked ``slow``, following the convention of
``tests/analysis/test_mechanism_runner.py`` (a tiny known-GP channel via ``gp_channel``,
monkeypatched into the module's own ``_channels`` name).
"""
from __future__ import annotations

import json
import os
import textwrap

import numpy as np
import pandas as pd
import pytest

from pfns4neurostim.analysis.update_rule import gp_channel
from pfns4neurostim.experiments import mechanism
from pfns4neurostim.experiments import mechanism_layers as ML
from pfns4neurostim.experiments.mechanism import _check_keys


def _config(tmp_path, body: str, sizes: str = "[8]") -> str:
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


# ---------------------------------------------------------------------------
# LAYER_DYNAMICS_DEFAULTS / LAYER_DYNAMICS_GATE_DEFAULTS: pure dict logic, fast
# ---------------------------------------------------------------------------
class TestDefaultsAndCheckKeys:
    def test_empty_block_merges_to_the_defaults_unchanged(self) -> None:
        merged = _check_keys({}, ML.LAYER_DYNAMICS_DEFAULTS, "layer_dynamics")
        assert merged == ML.LAYER_DYNAMICS_DEFAULTS

    def test_unknown_top_level_key_raises(self) -> None:
        with pytest.raises(ValueError, match="bogus"):
            _check_keys({"bogus": 1}, ML.LAYER_DYNAMICS_DEFAULTS, "layer_dynamics")

    def test_gate_defaults_cover_every_conjecture_id_in_the_task_plan(self) -> None:
        # Mirrors task_plan.md's "Conjectures and decision rules" table (frozen pre-registration);
        # a change here without a matching task_plan.md update is a sign the two have drifted.
        expected = {
            "c0a_alignment_delta", "c0a_cosine_min", "c0b_margin", "c0c_spearman_max",
            "c0d_rho_shape", "c0d_ell_hat", "c0d_far_field_share", "c0d_locality",
            "c1_partial_r_max", "c1_gp_fixed_delta_max", "c1_gp_specific_min",
            "c1p_segment_gap", "c2_label_removed_min", "c2_feature_removed_max",
            "c3_axis_delta_r2", "c3_icc_max", "c4c_block5_lift", "c4c_far_field_cut_t10",
            "c5_relative_dip_max", "c5_norm_plateau_dip", "c6a_block0_delta",
            "c6a_mid_block_delta_max",
        }
        assert set(ML.LAYER_DYNAMICS_GATE_DEFAULTS) == expected

    def test_unknown_gate_key_raises(self) -> None:
        with pytest.raises(ValueError, match="bogus"):
            _check_keys({"bogus": 1}, ML.LAYER_DYNAMICS_GATE_DEFAULTS, "layer_dynamics.gates")

    def test_probe_defaults_hold_folds_repeats_and_lambda_ladder(self) -> None:
        probe = ML.LAYER_DYNAMICS_DEFAULTS["probe"]
        assert set(probe) == {
            "n_folds", "n_repeats", "lambda_rels", "spatial", "spatial_n_repeats", "spatial_lambda_rel",
        }
        assert probe["n_folds"] > 0 and probe["n_repeats"] > 0
        assert len(probe["lambda_rels"]) >= 1
        assert probe["spatial_n_repeats"] > 0 and probe["spatial_lambda_rel"] > 0


# ---------------------------------------------------------------------------
# Config validation through the full dispatch (mechanism.run_mechanism), fast:
# every check below fires before any channel is loaded or model fitted.
# ---------------------------------------------------------------------------
class TestConfigValidationFailsBeforeAnyComputation:
    def test_engine_must_be_tabpfn_v2_5(self, tmp_path) -> None:
        cfg = _config(tmp_path, """
            analysis: layer_dynamics
            layer_dynamics:
              engine: gp_mll
        """)
        with pytest.raises(ValueError, match="tabpfn_v2_5"):
            mechanism.run_mechanism(cfg)

    def test_unknown_layer_dynamics_key_raises(self, tmp_path) -> None:
        cfg = _config(tmp_path, """
            analysis: layer_dynamics
            layer_dynamics:
              bogus: 1
        """)
        with pytest.raises(ValueError, match="bogus"):
            mechanism.run_mechanism(cfg)

    def test_unknown_gates_key_raises(self, tmp_path) -> None:
        cfg = _config(tmp_path, """
            analysis: layer_dynamics
            layer_dynamics:
              gates: {bogus: 1}
        """)
        with pytest.raises(ValueError, match="bogus"):
            mechanism.run_mechanism(cfg)

    def test_unknown_probe_key_raises(self, tmp_path) -> None:
        cfg = _config(tmp_path, """
            analysis: layer_dynamics
            layer_dynamics:
              probe: {bogus: 1}
        """)
        with pytest.raises(ValueError, match="bogus"):
            mechanism.run_mechanism(cfg)

    def test_context_ts_outside_the_shared_ladder_raises(self, tmp_path) -> None:
        cfg = _config(tmp_path, """
            analysis: layer_dynamics
            layer_dynamics:
              context_ts: [999]
        """, sizes="[8]")
        with pytest.raises(ValueError, match="context_ts"):
            mechanism.run_mechanism(cfg)

    def test_valid_partial_override_is_accepted_by_check_keys(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sanity check that a well-formed override does NOT raise at the validation stage.
        Channels are monkeypatched to an empty list so the test is deterministic regardless of
        whether real NHP data is present on this machine; past validation, the next thing the
        runner checks is that there is at least one channel."""
        monkeypatch.setattr(ML, "_channels", lambda cfg: [])
        cfg = _config(tmp_path, """
            analysis: layer_dynamics
            layer_dynamics:
              n_anchors: 4
              probe: {n_folds: 3, n_repeats: 2, lambda_rels: [0.5]}
        """, sizes="[8]")
        with pytest.raises(RuntimeError, match="no channels selected"):
            mechanism.run_mechanism(cfg)


# ---------------------------------------------------------------------------
# End-to-end smoke test and _fact_sheet: real tabpfn==6.3.2 on CPU, slow
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch):
    """A 6x6 synthetic GP channel (36 sites), monkeypatched into ``mechanism_layers._channels``.

    ``_channels`` is imported BY NAME into ``mechanism_layers`` (``from .mechanism import
    ..., _channels, ...``), so the patch target is ``mechanism_layers._channels``, not
    ``mechanism._channels`` (the latter is what ``tests/analysis/test_mechanism_runner.py``
    patches for the ``update_rule``/``cka``/``placement`` runners, which call it directly
    inside ``mechanism.py``).
    """
    ch = gp_channel(6, 0.3, 0.2, 5, np.random.default_rng(0))
    monkeypatch.setattr(ML, "_channels", lambda cfg: [ch])
    return ch


@pytest.mark.slow
def test_fact_sheet_reports_architecture_facts_and_the_fingerprint_tie(tmp_path, tiny) -> None:
    cfg = mechanism.load_mechanism_config(_config(tmp_path, """
        analysis: layer_dynamics
        layer_dynamics: {}
    """, sizes="[8]"))
    p = {**ML.LAYER_DYNAMICS_DEFAULTS, "context_ts": [8]}

    fact = ML._fact_sheet(cfg, p, tiny)  # noqa: SLF001 - testing the private helper directly

    assert fact["n_blocks"] == 18
    assert fact["emsize"] == 192
    assert fact["n_heads_feature_attn"] >= 1
    assert fact["n_heads_item_attn"] >= 1
    assert fact["decoder_shapes"] == [[192, 384], [384, fact["n_bars"]]]
    assert fact["n_bars"] > 0
    assert fact["context_t_used"] == 8
    # A0: exactly one column (the fingerprint) differs between test-mode and train-mode transform.
    assert fact["n_test_mode_vs_train_mode_columns_differing"] == 1
    assert len(fact["test_mode_vs_train_mode_fingerprint_columns_differ"]) == 1


@pytest.mark.slow
def test_run_layer_dynamics_end_to_end_writes_tables_fact_sheet_and_figures(tmp_path, tiny) -> None:
    cfg_path = _config(tmp_path, """
        analysis: layer_dynamics
        layer_dynamics:
          n_context_draws: 1
          n_anchors: 2
          surprises: [-1.0, 1.0]
          probe: {n_folds: 2, n_repeats: 1, lambda_rels: [1.0]}
          gp_params: {n_opt_steps: 10, lr: 0.1}
          stress: {knob: k2_channel, levels: [1.0]}
          context_ts: [8]
    """, sizes="[8]")

    out = mechanism.run_mechanism(cfg_path)

    fact_sheet = json.load(open(os.path.join(out, "fact_sheet.json")))
    assert fact_sheet["n_blocks"] == 18

    align = pd.read_csv(os.path.join(out, "layer_dynamics.csv"))
    probe = pd.read_csv(os.path.join(out, "layer_probe.csv"))

    # 1 channel x 1 level x 1 t x 1 draw x 2 anchors x (18 blocks + the H1/H2/H3 head stages).
    assert len(align) == 2 * (18 + 3)
    blocks = align[align["stage"].str.startswith("block")]
    assert len(blocks) == 2 * 18
    assert set(blocks["layer"]) == set(range(18))
    assert set(align["stage"]) == {f"block{k}" for k in range(18)} | {"H1", "H2", "H3"}
    assert align.loc[align["stage"].isin(["H1", "H2", "H3"]), "layer"].isna().all()
    assert {"pearson", "spearman", "common_mode_share", "anchor", "stratum"} <= set(align.columns)

    # Same grid x (1 random-fold lambda_rel + 1 spatial-fold lambda_rel).
    assert len(probe) == 2 * 18 * (1 + 1)
    assert set(probe["fold_scheme"]) == {"random", "spatial"}
    assert {"ridge_r2", "lambda_rel"} <= set(probe.columns)
    assert np.isfinite(probe["ridge_r2"]).all()

    assert os.path.exists(os.path.join(out, "layer_dynamics_alignment.svg"))
    assert os.path.exists(os.path.join(out, "layer_dynamics_probe.svg"))

    # --replot rebuilds purely from the CSVs.
    os.remove(os.path.join(out, "layer_dynamics_alignment.svg"))
    mechanism.run_mechanism(cfg_path, replot=True)
    assert os.path.exists(os.path.join(out, "layer_dynamics_alignment.svg"))


@pytest.mark.slow
def test_run_layer_dynamics_raises_when_no_channels_are_selected(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ML, "_channels", lambda cfg: [])
    cfg_path = _config(tmp_path, """
        analysis: layer_dynamics
        layer_dynamics: {}
    """, sizes="[8]")
    with pytest.raises(RuntimeError, match="no channels selected"):
        mechanism.run_mechanism(cfg_path)
