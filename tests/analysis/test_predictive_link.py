"""Predictive link (roadmap M8): random-intercept slope, pairing rules, per-channel predictors and outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfns4neurostim.analysis import predictive_link as L
from pfns4neurostim.analysis.predictive_link import predictive_link

C1_REFS = ("gp_mll_frozen",)


def _frame(slope: float, seed: int = 0, n_subjects: int = 6, n_emg: int = 5, offset_sd: float = 5.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subj in range(n_subjects):
        offset = rng.normal(scale=offset_sd)            # large subject intercepts
        for emg in range(n_emg):
            x = rng.normal()
            rows.append({"dataset": "nhp", "subject": subj, "emg": emg, "x": x,
                         "regret": offset + slope * x + 0.5 * rng.normal()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the link model
# ---------------------------------------------------------------------------
def test_slope_recovers_a_planted_relation_despite_subject_intercepts() -> None:
    res = predictive_link(_frame(2.0), "x", "regret", n_boot=300)
    assert res["slope"] == pytest.approx(2.0, abs=0.3)
    assert res["ci_low"] < 2.0 < res["ci_high"] and res["ci_low"] > 1.0
    assert res["within_slope"] == pytest.approx(res["slope"], abs=0.1)
    assert res["re_ratio"] > 1.0                        # the subject intercepts are found, not pooled away
    assert (res["n_channels"], res["n_subjects"], res["n_boot_valid"]) == (30, 6, 300)


def test_null_slope_ci_covers_zero() -> None:
    res = predictive_link(_frame(0.0, seed=3), "x", "regret", n_boot=300)
    assert res["ci_low"] < 0.0 < res["ci_high"]


def test_one_subject_is_plain_ols_with_a_channel_bootstrap() -> None:
    df = _frame(1.5, seed=1, n_subjects=1, n_emg=12)
    res = predictive_link(df, "x", "regret", n_boot=300)
    ols_slope, ols_intercept = np.polyfit(df["x"], df["regret"], 1)
    assert res["slope"] == pytest.approx(ols_slope, rel=1e-9)
    assert res["intercept"] == pytest.approx(ols_intercept, rel=1e-9)
    assert res["within_slope"] == pytest.approx(ols_slope, rel=1e-9)
    assert res["n_subjects"] == 1 and res["re_ratio"] == 0.0
    assert res["ci_low"] < res["slope"] < res["ci_high"]


def test_no_subject_variance_pools_to_ols() -> None:
    df = _frame(1.0, seed=2, offset_sd=0.0)
    res = predictive_link(df, "x", "regret", n_boot=50)
    assert res["re_ratio"] < 0.05
    assert res["slope"] == pytest.approx(np.polyfit(df["x"], df["regret"], 1)[0], abs=0.05)


def test_bootstrap_is_seeded() -> None:
    a = predictive_link(_frame(1.0), "x", "regret", n_boot=100, seed=7)
    b = predictive_link(_frame(1.0), "x", "regret", n_boot=100, seed=7)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])


def test_too_few_channels_constant_columns_and_inf_raise() -> None:
    with pytest.raises(ValueError, match=">= 3"):
        predictive_link(_frame(1.0).head(2), "x", "regret")
    with pytest.raises(ValueError, match="constant"):
        predictive_link(_frame(1.0).assign(x=1.0), "x", "regret")
    with pytest.raises(ValueError, match="infinite"):
        predictive_link(_frame(1.0).assign(x=np.inf), "x", "regret")


# ---------------------------------------------------------------------------
# pairing rules
# ---------------------------------------------------------------------------
REFS = ["gp_bayes", "gp_mll_refit", "gp_fixed", "gp_mll_frozen"]


def test_default_pairs_are_the_pre_extension_link() -> None:
    pairs = L.reference_pairs(["drift_abs", "total_var_ratio", "transfer_slope", "delta_crps"], ["gp_bayes"],
                              c1_references=C1_REFS, engine_contrasts=False)
    assert pairs == [("drift_abs", "none"), ("total_var_ratio", "none"), ("transfer_slope", "gp_bayes"),
                     ("delta_crps", "gp_bayes")]


def test_c1_reference_is_never_paired_with_part_a_readouts() -> None:
    pairs = L.reference_pairs(list(L.READOUTS), REFS, c1_references=C1_REFS, engine_contrasts=True)
    with_frozen = {p for p, r in pairs if r == "gp_mll_frozen"}
    assert with_frozen == {p for p, rd in L.READOUTS.items() if rd.c1}
    assert not with_frozen & {p for p, rd in L.READOUTS.items() if rd.part == "A"}
    # C1-only readouts meet the C1 reference only (and their own value)
    assert {r for p, r in pairs if p == "gain"} == {"none", "gp_mll_frozen"}
    # Part A gets its own value and every other reference
    assert [r for p, r in pairs if p == "drift_abs"] == ["none", "gp_bayes", "gp_mll_refit", "gp_fixed"]
    # contrast readouts have no 'none'
    assert [r for p, r in pairs if p == "gap_change"] == ["gp_bayes", "gp_mll_refit", "gp_fixed"]


def test_unknown_predictor_raises() -> None:
    with pytest.raises(ValueError, match="unknown predictor"):
        L.reference_pairs(["drift_abs", "nope"], ["gp_bayes"], c1_references=C1_REFS, engine_contrasts=False)


# ---------------------------------------------------------------------------
# per-cell / per-channel predictors
# ---------------------------------------------------------------------------
CELL = {"dataset": "nhp", "subject": 0, "emg": 0, "context_t": 10, "draw": 0}


def _h2h() -> pd.DataFrame:
    rows = []
    for level in (1.0, 2.0):
        for anchor, (own, ref) in enumerate([(1.0, 0.0), (2.0, 3.0), (6.0, 1.0)]):
            base = {**CELL, "level": level, "anchor": anchor, "pool": "unobserved", "source": "trial"}
            rows.append({**base, "engine": "tabpfn_v2_5", "reference": "none", "gain": own})
            rows.append({**base, "engine": "gp_mll_frozen", "reference": "none", "gain": ref})
            rows.append({**base, "engine": "tabpfn_v2_5", "reference": "gp_bayes", "transfer_slope": own * level})
            rows.append({**base, "pool": "observed", "engine": "tabpfn_v2_5", "reference": "gp_bayes",
                         "transfer_slope": 99.0})
    return pd.DataFrame(rows)


def _coh() -> pd.DataFrame:
    return pd.DataFrame([{**CELL, "level": 1.0, "pool": pool, "engine": eng, "drift_abs": v}
                         for pool in ("unobserved", "observed")
                         for eng, v in (("tabpfn_v2_5", 0.3), ("gp_bayes", 0.1), ("gp_mll_refit", 0.2))])


def test_cell_readouts_match_per_anchor_then_take_the_median() -> None:
    pairs = [("gain", "none"), ("gain", "gp_mll_frozen"), ("transfer_slope", "gp_bayes"), ("drift_abs", "gp_bayes")]
    cells = L.cell_readouts(_coh(), _h2h(), pairs, model="tabpfn_v2_5", pool="unobserved", c1_references=C1_REFS)
    at1 = cells[np.isclose(cells["level"], 1.0)].set_index(["predictor", "reference"])["value"]
    assert at1[("gain", "none")] == pytest.approx(2.0)
    assert at1[("gain", "gp_mll_frozen")] == pytest.approx(1.0)               # median of (1, -1, 5)
    assert at1[("transfer_slope", "gp_bayes")] == pytest.approx(2.0)          # the observed pool's 99 is not read
    assert at1[("drift_abs", "gp_bayes")] == pytest.approx(0.2)


def test_matched_difference_is_aggregated_after_the_difference() -> None:
    h = _h2h()
    h.loc[(h["engine"] == "gp_mll_frozen"), "gain"] = [0.0, 3.0, -10.0] * 2   # per-anchor diffs 1, -1, 16
    cells = L.cell_readouts(_coh(), h, [("gain", "gp_mll_frozen")], model="tabpfn_v2_5", pool="unobserved",
                            c1_references=C1_REFS)
    # median of differences = 1; difference of medians would be 2 - 0 = 2
    assert cells["value"].iloc[0] == pytest.approx(1.0)


def test_part_a_with_the_c1_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="never paired with the Part A"):
        L.cell_readouts(_coh(), _h2h(), [("drift_abs", "gp_mll_frozen")], model="tabpfn_v2_5", pool="unobserved",
                        c1_references=C1_REFS)


def test_missing_reference_rows_raise() -> None:
    with pytest.raises(ValueError, match="no trial rows"):
        L.cell_readouts(_coh(), _h2h(), [("transfer_slope", "gp_fixed")], model="tabpfn_v2_5", pool="unobserved",
                        c1_references=C1_REFS)
    with pytest.raises(ValueError, match="no matched rows"):
        L.cell_readouts(_coh(), _h2h(), [("drift_abs", "gp_fixed")], model="tabpfn_v2_5", pool="unobserved",
                        c1_references=C1_REFS)


def test_channel_readouts_keep_one_level_and_the_requested_context_sizes() -> None:
    cells = pd.DataFrame([{**CELL, "level": lv, "context_t": t, "predictor": "drift_abs", "reference": "none",
                           "value": v}
                          for lv, t, v in ((1.0, 10, 0.1), (1.0, 25, 0.3), (1.0, 50, 0.2), (2.0, 10, 5.0))])
    out = L.channel_readouts(cells, level=1.0)
    assert out["value"].iloc[0] == pytest.approx(0.2) and out["n_cells"].iloc[0] == 3
    out = L.channel_readouts(cells, level=1.0, context_ts=[10, 25])
    assert out["value"].iloc[0] == pytest.approx(0.2) and out["n_cells"].iloc[0] == 2
    assert L.channel_readouts(cells, level=2.0)["value"].iloc[0] == pytest.approx(5.0)


def test_references_in_run_are_in_display_order() -> None:
    h = pd.DataFrame({"engine": ["tabpfn_v2_5"] * 4 + ["gp_fixed"],
                      "reference": ["gp_mll_frozen", "none", "gp_fixed", "gp_bayes", "gp_mll_refit"]})
    assert L.references_in_run(h, "tabpfn_v2_5") == ["gp_bayes", "gp_fixed", "gp_mll_frozen"]


# ---------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------
def _bo() -> pd.DataFrame:
    return pd.DataFrame([{"dataset": "nhp", "subject": 0, "emg": e, "model": m, "acq_label": "ts_marginal", "rep": r,
                          "recommended_regret": 0.1 * e + 0.01 * r, "r2": [0.9, 0.8, -30.0][r]}
                         for e in range(2) for m in ("tabpfn_v2_5", "gp_mll") for r in range(3)])


def test_channel_outcomes_mean_regret_and_median_r2() -> None:
    out = L.channel_outcomes(_bo(), ["recommended_regret", "r2"], model="tabpfn_v2_5")
    assert len(out) == 2
    assert out.loc[1, "recommended_regret"] == pytest.approx(0.11)
    assert out.loc[0, "r2"] == pytest.approx(0.8)                              # the -30 outlier does not drive it


def test_missing_outcome_column_and_ambiguous_tidy_raise() -> None:
    with pytest.raises(ValueError, match="no outcome column"):
        L.channel_outcomes(_bo(), ["cumulative_regret"], model="tabpfn_v2_5")
    two = pd.concat([_bo(), _bo().assign(acq_label="ei")])
    with pytest.raises(ValueError, match="several acquisitions"):
        L.channel_outcomes(two, ["r2"], model="tabpfn_v2_5")
    assert len(L.channel_outcomes(two, ["r2"], model="tabpfn_v2_5", acq_label="ei")) == 2
    with pytest.raises(ValueError, match="no rows of model"):
        L.channel_outcomes(_bo(), ["r2"], model="gp_naive")


def _stress(levels: tuple[float, ...] = (1.0, 2.0, 4.0)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for emg in range(2):
        for lv in levels:
            for rep in range(6):
                ref = 0.2 + 0.01 * rng.normal()
                # channel 0 never breaks; channel 1 is far worse than the reference from level 2 on
                own = ref + (0.5 if (emg == 1 and lv >= 2.0) else 0.0) + 0.01 * rng.normal()
                for model, val in (("tabpfn_v2_5", own), ("gp_mll", ref)):
                    rows.append({"dataset": "nhp", "subject": 0, "emg": emg, "model": model, "level": lv,
                                 "rep": rep, "knob": "k2_channel", "demo": "draws", "recommended_regret": val})
    return pd.DataFrame(rows)


def test_channel_breakdown_counts_levels_passed() -> None:
    out = L.channel_breakdown(_stress(), model="tabpfn_v2_5", reference="gp_mll", metric="recommended_regret",
                              margin=0.05).set_index("emg")
    assert out.loc[0, "breakdown"] == 3.0 and np.isnan(out.loc[0, "breakdown_level"])   # never: ladder length
    assert out.loc[1, "breakdown"] == 1.0 and out.loc[1, "breakdown_level"] == 2.0
    assert (out["n_levels"] == 3).all()


def test_channel_breakdown_resolves_legacy_demo_spellings() -> None:
    """Cells cached before the 2026-09-27 rename say ``demo2``; a backfilled channel says ``draws`` (same ladder)."""
    stress = _stress().assign(demo=lambda d: np.where(d["emg"] == 0, "demo2", "draws"))
    out = L.channel_breakdown(stress, model="tabpfn_v2_5", reference="gp_mll", metric="recommended_regret",
                              margin=0.05).set_index("emg")
    pd.testing.assert_frame_equal(out, L.channel_breakdown(_stress().assign(demo="draws"), model="tabpfn_v2_5",
                                                           reference="gp_mll", metric="recommended_regret",
                                                           margin=0.05).set_index("emg"))
    with pytest.raises(ValueError, match="several demos"):
        L.channel_breakdown(_stress().assign(demo=lambda d: np.where(d["emg"] == 0, "synthetic", "draws")),
                            model="tabpfn_v2_5", reference="gp_mll", metric="recommended_regret", margin=0.05)


def test_channel_breakdown_refuses_missing_columns_and_mixed_ladders() -> None:
    with pytest.raises(ValueError, match="no column"):
        L.channel_breakdown(_stress().drop(columns="rep"), model="tabpfn_v2_5", reference="gp_mll",
                            metric="recommended_regret", margin=0.05)
    ragged = _stress()
    ragged = ragged[~((ragged["emg"] == 1) & (ragged["level"] == 4.0))]
    with pytest.raises(ValueError, match="different level ladders"):
        L.channel_breakdown(ragged, model="tabpfn_v2_5", reference="gp_mll", metric="recommended_regret",
                            margin=0.05)
