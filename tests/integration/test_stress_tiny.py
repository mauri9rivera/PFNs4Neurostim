"""End-to-end stress-sweep test on a tiny synthetic channel (task #10 Step 7).

Runs the real runner path — knob, models, BO loop, tidy schema, figures — on a
20-site problem so the whole pipeline is exercised in seconds without touching
the raw ``.mat`` data. The GP models are used; TabPFN is covered by the
``slow``/``gpu`` suites since loading its weights dominates the runtime.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pfns4neurostim.data.channels import ChannelData
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.evaluation import results as _results
from pfns4neurostim.evaluation.bo_runner import run_channel_bo
from pfns4neurostim.evaluation.results import TidyRow
from pfns4neurostim.seeding import seed_for
from pfns4neurostim.visualization import stress as stress_figs

MODELS = ("gp_mll", "gp_naive")
LEVELS = (1.0, 2.0)
BUDGET = 8
N_INIT = 3
N_REPS = 2


@pytest.fixture(scope="module")
def channel() -> ChannelData:
    """A 20-site, 6-trial synthetic channel with one smooth hotspot."""
    rng = np.random.default_rng(0)
    coords = np.stack(np.meshgrid(np.arange(5), np.arange(4)), axis=-1).reshape(-1, 2)  # [20, 2]
    x = coords / np.array([4.0, 3.0])                                                    # [20, 2]
    y_gt = np.exp(-((x[:, 0] - 0.7) ** 2 + (x[:, 1] - 0.3) ** 2) / 0.1)                  # [20]
    y_gt = (y_gt - y_gt.mean()) / y_gt.std()
    Y = y_gt[:, None] + 0.25 * rng.normal(size=(20, 6))                                  # [20, 6]
    return ChannelData("nhp", 1, 0, x, Y, y_gt, coords, (5, 4))


@pytest.fixture(scope="module")
def sweep(channel: ChannelData) -> pd.DataFrame:
    """Run the tiny sweep grid and return the tidy frame."""
    knob = build_knob("k2_snr", LEVELS)
    rows: list[TidyRow] = []
    extras: list[dict[str, float]] = []
    for level in knob.levels:
        stressed = knob.apply(channel, level, np.random.default_rng(0))
        achieved = knob.achieved(stressed)
        for model in MODELS:
            for rep in range(N_REPS):
                seed = seed_for(channel.label, "k2_snr", level, model, rep)
                res = run_channel_bo(
                    model,
                    stressed,
                    acq_fn="ei",
                    budget=BUDGET,
                    n_init=N_INIT,
                    seed=seed,
                    device="cpu",
                )
                rows.append(
                    TidyRow(
                        run_tag="tiny",
                        experiment="stress_sweep",
                        dataset="nhp",
                        demo="demo2",
                        subject=1,
                        emg=0,
                        model=model,
                        model_version=res.row["model_version"],
                        acq_type="ei",
                        knob="k2_snr",
                        level=float(level),
                        gt_mode="full_mean",
                        rep=rep,
                        r2=res.row["r2"],
                        recommended_regret=res.row["recommended_regret"],
                        best_queried_regret=res.row["best_queried_regret"],
                        cumulative_regret=res.row["cumulative_regret"],
                        final_regret=res.row["recommended_regret"],
                        top1_hit=res.row["top1_hit"],
                        top3_hit=res.row["top3_hit"],
                        opt_distance=res.row["opt_distance"],
                        coverage_50=res.row["coverage_50"],
                        coverage_90=res.row["coverage_90"],
                        ece=res.row["ece"],
                        nll=res.row["nll"],
                        crps=res.row["crps"],
                        mean_query_latency_s=res.row["mean_query_latency_s"],
                        total_time_s=res.row["total_time_s"],
                        achieved_snr_db=achieved["achieved_snr_db"],
                        n_sites=stressed.n_sites,
                        budget=BUDGET,
                        n_init=N_INIT,
                        seed=seed,
                    )
                )
                extras.append(achieved)
    df = _results.rows_to_dataframe(rows, acquisition={"type": "ei", "params": {"xi": 0.0}})
    return df


class TestTinySweep:
    """The sweep produces a schema-correct, self-consistent tidy table."""

    def test_row_count_matches_the_grid(self, sweep: pd.DataFrame) -> None:
        assert len(sweep) == len(LEVELS) * len(MODELS) * N_REPS

    def test_schema_columns_present(self, sweep: pd.DataFrame) -> None:
        required = {
            "run_tag", "experiment", "dataset", "demo", "subject", "emg", "model",
            "model_version", "acq_type", "knob", "level", "gt_mode", "rep",
            "recommended_regret", "best_queried_regret", "cumulative_regret",
            "coverage_90", "ece", "achieved_snr_db", "budget", "n_init", "seed",
        }
        assert required <= set(sweep.columns)

    def test_model_version_is_logged(self, sweep: pd.DataFrame) -> None:
        """P0.1: a result must carry the version that produced it."""
        assert sweep["model_version"].notna().all()
        assert (sweep["model_version"].str.len() > 0).all()

    def test_acquisition_params_are_flattened(self, sweep: pd.DataFrame) -> None:
        """P0.2: acquisition parameters must be recoverable from the CSV alone."""
        assert "acq_param_xi" in sweep.columns
        assert (sweep["acq_type"] == "ei").all()

    def test_regrets_are_in_unit_range(self, sweep: pd.DataFrame) -> None:
        for col in ("recommended_regret", "best_queried_regret"):
            assert sweep[col].between(0.0, 1.0).all()

    def test_stress_lowers_achieved_snr(self, sweep: pd.DataFrame) -> None:
        by_level = sweep.groupby("level")["achieved_snr_db"].mean()
        assert by_level.loc[2.0] < by_level.loc[1.0]
        assert by_level.loc[2.0] - by_level.loc[1.0] == pytest.approx(-6.02, abs=0.05)

    def test_every_metric_is_finite(self, sweep: pd.DataFrame) -> None:
        for col in ("recommended_regret", "cumulative_regret", "coverage_90", "ece", "nll"):
            assert np.isfinite(sweep[col].to_numpy(dtype=float)).all()


class TestDeterminism:
    """Identical seeds reproduce identical trajectories."""

    def test_same_seed_same_queries(self, channel: ChannelData) -> None:
        kwargs = dict(acq_fn="ei", budget=BUDGET, n_init=N_INIT, seed=7, device="cpu")
        a = run_channel_bo("gp_mll", channel, **kwargs)
        b = run_channel_bo("gp_mll", channel, **kwargs)
        assert a.trajectory["observed_indices"] == b.trajectory["observed_indices"]
        assert a.row["recommended_regret"] == pytest.approx(b.row["recommended_regret"])

    def test_seed_derivation_is_stable_and_distinct(self) -> None:
        assert seed_for("nhp-s1-e0", "k2_snr", 2.0, "gp_mll", 0) == seed_for(
            "nhp-s1-e0", "k2_snr", 2.0, "gp_mll", 0
        )
        assert seed_for("nhp-s1-e0", "k2_snr", 2.0, "gp_mll", 0) != seed_for(
            "nhp-s1-e0", "k2_snr", 2.0, "gp_mll", 1
        )


class TestBudgetSemantics:
    """P0.3: budget counts total queries including n_init."""

    def test_queries_equal_budget(self, channel: ChannelData) -> None:
        res = run_channel_bo(
            "gp_naive", channel, acq_fn="ei", budget=BUDGET, n_init=N_INIT, seed=1, device="cpu"
        )
        assert len(res.trajectory["observed_indices"]) == BUDGET

    def test_budget_below_n_init_raises(self, channel: ChannelData) -> None:
        with pytest.raises(ValueError, match="must exceed n_init"):
            run_channel_bo("gp_naive", channel, budget=3, n_init=3, device="cpu")

    def test_budget_above_pool_size_raises(self, channel: ChannelData) -> None:
        with pytest.raises(ValueError, match="exceeds the 20 sites"):
            run_channel_bo("gp_naive", channel, budget=25, n_init=3, device="cpu")


class TestDeliverables:
    """Figures and tables regenerate from the tidy frame alone."""

    def test_render_all_writes_every_deliverable(self, sweep: pd.DataFrame, tmp_path) -> None:
        written = stress_figs.render_all(sweep, str(tmp_path), knob="k2_snr", dataset="nhp")
        names = {os.path.basename(p) for p in written}
        assert "robustness.csv" in names
        assert "degradation_invivo.svg" in names
        assert "calibration_invivo.svg" in names
        for path in written:
            assert os.path.getsize(path) > 0

    def test_robustness_table_has_one_row_per_model(self, sweep: pd.DataFrame, tmp_path) -> None:
        table, path = stress_figs.build_robustness_table(sweep, str(tmp_path), knob="k2_snr")
        assert set(table["model"]) == set(MODELS)
        assert {"degradation_auc", "cvar10_regret", "breakdown_level", "breakdown_x"} <= set(
            table.columns
        )
        assert os.path.exists(path)

    def test_reference_model_has_no_breakdown(self, sweep: pd.DataFrame, tmp_path) -> None:
        table, _ = stress_figs.build_robustness_table(sweep, str(tmp_path), knob="k2_snr")
        ref = table[table["model"] == stress_figs.REFERENCE_MODEL].iloc[0]
        assert ref["breakdown_reason"] == "reference"
