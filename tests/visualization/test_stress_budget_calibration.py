"""E10 robustness vs BO budget and E12 calibration under stress (2026-10-21).

Both are rebuilt from a run's ``tidy.csv`` and ``trajectories.pkl`` alone, so every test here works on small
synthetic frames: the budget read-off must be the trace value, the CVaR the robustness module's, the breakdown
at the full budget the robustness table's, and calibration must refuse to pool ``online_y_scaler`` settings.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pfns4neurostim.evaluation import results as _results
from pfns4neurostim.evaluation.robustness import breakdown_point, cvar
from pfns4neurostim.visualization import stress as stress_figs
from pfns4neurostim.visualization import style

MODELS: tuple[str, ...] = ("gp_mll", "gp_naive", "tabpfn_v2_5")
LEVELS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
N_CHANNELS: int = 5
N_REPS: int = 4
N_INIT: int = 3
BUDGET: int = 20


def _snr(level: float) -> float:
    """Achieved SNR of a K2 level: falls as the level (residual gain) rises; 1.0 is the default regime."""
    return 4.0 - 6.0 * np.log2(level)


def _trace(model: str, level: float, ch: int, rep: int, rng: np.random.Generator) -> np.ndarray:
    """A decreasing regret trace of length ``BUDGET - N_INIT + 1``; GP-fixed degrades faster with the level."""
    steps = np.arange(BUDGET - N_INIT + 1, dtype=float)                                  # [L]
    floor = 0.05 * level * (3.0 if model == "gp_naive" else 1.0)
    return floor + (1.0 - floor) * np.exp(-steps / 5.0) + 0.01 * rng.random(steps.size)


def _sweep(knob: str = "k2_channel") -> tuple[pd.DataFrame, pd.DataFrame]:
    """A tidy frame and its matching trace frame (``recommended_regret`` is the trace's last value)."""
    rng = np.random.default_rng(0)
    rows, traces = [], []
    for ch in range(N_CHANNELS):
        for level in LEVELS:
            for model in MODELS:
                for rep in range(N_REPS):
                    trace = _trace(model, level, ch, rep, rng)
                    key = {"model": model, "level": level, "subject": ch, "emg": 0, "rep": rep}
                    rows.append({
                        **key, "demo": "draws", "knob": knob, "dataset": "nhp", "n_init": N_INIT, "budget": BUDGET,
                        "recommended_regret": float(trace[-1]), "exploration_score": 0.8, "r2": 0.5,
                        "achieved_snr_db": _snr(level), "online_y_scaler": "minmax", "normalization": "pfn",
                        "coverage_50": 0.5 + 0.02 * level + 0.01 * rng.normal(),
                        "coverage_90": 0.9 + 0.01 * level * rng.random(),
                        "ece": 0.05 * level + 0.01 * rng.random(), "nll": 1.0 + 0.2 * level + 0.05 * rng.normal(),
                    })
                    traces.append({**key, "n_init": N_INIT, "recommended_regret_per_step": trace})
    return pd.DataFrame(rows), pd.DataFrame(traces)


def _write_pickle(df: pd.DataFrame, frame: pd.DataFrame, run_dir: str) -> None:
    """Store ``frame``'s traces as ``trajectories.pkl`` the way the sweep runner does."""
    trajs = {}
    for _, r in frame.iterrows():
        key = tuple({"dataset": "nhp", "demo": "draws", "knob": df["knob"].iloc[0], "subject": r["subject"],
                     "emg": r["emg"], "model": r["model"], "level": r["level"], "rep": r["rep"]}.get(c)
                    for c in _results.KEY_COLUMNS)
        n_rec = len(r["recommended_regret_per_step"])
        trajs[key] = {"observed_indices": list(range(N_INIT + n_rec - 1)), "best_rec_indices": [0] * n_rec,
                      "step_times_s": [0.01] * n_rec, "recommended_regret_per_step": list(r["recommended_regret_per_step"])}
    os.makedirs(run_dir, exist_ok=True)
    _results.write_trajectories(run_dir, trajs)


class TestBudgetGrid:
    def test_is_the_s5_heatmap_grid(self) -> None:
        assert list(stress_figs.budget_grid(5, 96, 8)) == [6, 19, 32, 45, 57, 70, 83, 96]

    def test_rejects_a_run_without_acquisitions(self) -> None:
        with pytest.raises(ValueError, match="no acquisition"):
            stress_figs.budget_grid(5, 5)


class TestRegretAtBudget:
    def test_read_off_equals_the_trace_value(self) -> None:
        _, frame = _sweep()
        at = stress_figs.regret_at_budget(frame, 9)
        expected = np.array([t[9 - N_INIT] for t in frame["recommended_regret_per_step"]])
        np.testing.assert_array_equal(at["recommended_regret"].to_numpy(), expected)
        assert (at["budget"] == 9).all() and len(at) == len(frame)

    def test_full_budget_equals_the_tidy_column(self) -> None:
        df, frame = _sweep()
        at = stress_figs.regret_at_budget(frame, BUDGET)
        merged = at.merge(df, on=["model", "level", "subject", "emg", "rep"], suffixes=("_trace", ""))
        np.testing.assert_allclose(merged["recommended_regret_trace"], merged["recommended_regret"])

    def test_budget_outside_the_run_raises(self) -> None:
        _, frame = _sweep()
        with pytest.raises(ValueError, match="outside the recorded run"):
            stress_figs.regret_at_budget(frame, BUDGET + 1)

    def test_non_finite_trace_value_fails_fast(self) -> None:
        _, frame = _sweep()
        bad = frame["recommended_regret_per_step"].iloc[0].copy()
        bad[2] = np.nan
        frame.at[0, "recommended_regret_per_step"] = bad
        with pytest.raises(ValueError, match="non-finite"):
            stress_figs.regret_at_budget(frame, N_INIT + 2)


class TestRobustnessVsBudgetTable:
    def test_shape_one_row_per_budget_model_level(self) -> None:
        df, frame = _sweep()
        table = stress_figs.robustness_vs_budget_table(frame, df, knob="k2_channel", margin=0.05, n_budgets=4)
        grid = stress_figs.budget_grid(N_INIT, BUDGET, 4)
        assert len(table) == grid.size * len(MODELS) * len(LEVELS)
        assert sorted(table["budget"].unique()) == list(grid)

    def test_cvar_is_the_robustness_module_cvar(self) -> None:
        df, frame = _sweep()
        table = stress_figs.robustness_vs_budget_table(frame, df, knob="k2_channel", margin=0.05, budgets=[8])
        at = stress_figs.regret_at_budget(frame, 8)
        row = table[(table["model"] == "tabpfn_v2_5") & (table["level"] == 2.0)].iloc[0]
        values = at[(at["model"] == "tabpfn_v2_5") & (at["level"] == 2.0)]["recommended_regret"]
        assert row["cvar10_recommended_regret"] == pytest.approx(cvar(values.to_numpy(), alpha=0.1))
        per_model = table[table["model"] == "tabpfn_v2_5"]
        assert row["cvar10_level_mean"] == pytest.approx(per_model["cvar10_recommended_regret"].mean())

    def test_median_regret_is_the_regime_surface_cell(self) -> None:
        df, frame = _sweep()
        table = stress_figs.robustness_vs_budget_table(frame, df, knob="k2_channel", margin=0.05, n_budgets=4)
        surf = stress_figs.regime_surface(frame, df, knob="k2_channel", model="gp_mll", n_budgets=4)
        ours = table[table["model"] == "gp_mll"].pivot(index="x", columns="budget", values="recommended_regret_median")
        np.testing.assert_allclose(ours.loc[surf["x"], surf["budgets"]].to_numpy(), surf["value"])

    def test_full_budget_breakdown_matches_the_robustness_table(self, tmp_path) -> None:
        df, frame = _sweep()
        table = stress_figs.robustness_vs_budget_table(frame, df, knob="k2_channel", margin=0.05, budgets=[BUDGET])
        rob, _ = stress_figs.build_robustness_table(df, str(tmp_path), knob="k2_channel", margin=0.05)
        for model in ("gp_naive", "tabpfn_v2_5"):
            ours = table[table["model"] == model].iloc[0]
            theirs = rob[rob["model"] == model].iloc[0]
            assert ours["breakdown_reason"] == theirs["breakdown_reason"]
            np.testing.assert_equal(ours["breakdown_x"], theirs["breakdown_x"])
        assert table.loc[table["model"] == "gp_naive", "breakdown_reason"].iloc[0] == "noninferiority"

    def test_reference_and_missing_reference_have_no_breakdown(self) -> None:
        df, frame = _sweep()
        table = stress_figs.robustness_vs_budget_table(frame, df, knob="k2_channel", margin=0.05, budgets=[10])
        assert (table.loc[table["model"] == "gp_mll", "breakdown_reason"] == "reference").all()
        shard = frame[frame["model"] != "gp_mll"]
        table = stress_figs.robustness_vs_budget_table(shard, df, knob="k2_channel", margin=0.05, budgets=[10])
        assert (table["breakdown_reason"] == "no_reference").all() and table["breakdown_x"].isna().all()


class TestPooledBreakdownCI:
    """The breakdown CI is a channel bootstrap of the plotted pooled point (2026-10-21)."""

    def test_robustness_table_point_is_the_pooled_breakdown_point(self, tmp_path) -> None:
        """Only the CI changed: the point is still breakdown_point on the pooled per-level values."""
        df, _ = _sweep()
        rob, _ = stress_figs.build_robustness_table(df, str(tmp_path), knob="k2_channel", margin=0.05)
        level_x = stress_figs._level_to_x(df, "achieved_snr_db")
        for model in ("gp_naive", "tabpfn_v2_5"):
            bp = breakdown_point(stress_figs._per_level_values(df[df["model"] == model], "recommended_regret"),
                                 stress_figs._per_level_values(df[df["model"] == "gp_mll"], "recommended_regret"),
                                 margin=0.05, test=stress_figs.BREAKDOWN_TEST)
            row = rob[rob["model"] == model].iloc[0]
            np.testing.assert_equal(row["breakdown_level"], bp["breakdown_level"])
            np.testing.assert_equal(row["breakdown_x"], level_x.get(bp["breakdown_level"], np.nan))

    @pytest.mark.parametrize("budget", [6, 12, BUDGET])
    def test_ci_brackets_the_point_in_both_tables(self, tmp_path, budget: int) -> None:
        df, frame = _sweep()
        table = stress_figs.robustness_vs_budget_table(frame, df, knob="k2_channel", margin=0.05, budgets=[budget])
        rob, _ = stress_figs.build_robustness_table(df, str(tmp_path), knob="k2_channel", margin=0.05)
        harshest = min(stress_figs._level_to_x(df, "achieved_snr_db").values())
        for frame_ in (table.drop_duplicates("model"), rob):
            for _, row in frame_[frame_["model"] != "gp_mll"].iterrows():
                point = row["breakdown_x"] if np.isfinite(row["breakdown_x"]) else harshest
                assert row["breakdown_ci_lo"] - 1e-9 <= point <= row["breakdown_ci_hi"] + 1e-9
                if not np.isfinite(row["breakdown_x"]):
                    assert bool(row["breakdown_ci_beyond_ladder"])

    def test_full_budget_ci_matches_the_robustness_table(self, tmp_path) -> None:
        df, frame = _sweep()
        table = stress_figs.robustness_vs_budget_table(frame, df, knob="k2_channel", margin=0.05, budgets=[BUDGET])
        rob, _ = stress_figs.build_robustness_table(df, str(tmp_path), knob="k2_channel", margin=0.05)
        cols = ["breakdown_ci_lo", "breakdown_ci_hi", "breakdown_boot_none_frac"]
        ours = table.drop_duplicates("model").set_index("model").loc[["gp_naive", "tabpfn_v2_5"], cols]
        theirs = rob.set_index("model").loc[["gp_naive", "tabpfn_v2_5"], cols]
        np.testing.assert_allclose(ours.to_numpy(dtype=float), theirs.to_numpy(dtype=float), equal_nan=True)


class TestCalibrationTable:
    def test_median_over_channels_of_rep_means(self) -> None:
        df, _ = _sweep()
        table = stress_figs.calibration_under_stress_table(df, knob="k2_channel")
        assert set(table["metric"]) == set(stress_figs.CALIBRATION_METRICS)
        assert len(table) == len(stress_figs.CALIBRATION_METRICS) * len(MODELS) * len(LEVELS)
        sub = df[(df["model"] == "tabpfn_v2_5") & (df["level"] == 2.0)]
        expected = sub.groupby(["subject", "emg"])["ece"].mean()
        row = table[(table["metric"] == "ece") & (table["model"] == "tabpfn_v2_5") & (table["level"] == 2.0)].iloc[0]
        assert row["median"] == pytest.approx(expected.median())
        assert row["q25"] <= row["median"] <= row["q75"] and row["n_channels"] == N_CHANNELS
        assert row["x"] == pytest.approx(_snr(2.0))

    def test_nll_reduces_repetitions_by_the_median(self) -> None:
        """One degenerate repetition must not set a channel's NLL (heavy-tailed, as on NHP GP-MLL)."""
        df, _ = _sweep()
        df.loc[(df["model"] == "gp_mll") & (df["rep"] == 0), "nll"] = 1e8
        table = stress_figs.calibration_under_stress_table(df, knob="k2_channel", metrics=("nll",))
        assert (table.loc[table["model"] == "gp_mll", "q75"] < 10.0).all()
        assert set(table["rep_reduction"]) == {"median"}

    @pytest.mark.parametrize("column, values", [("online_y_scaler", ("minmax", "none")),
                                                ("normalization", ("pfn", "raw"))])
    def test_mixed_scaling_settings_raise(self, column: str, values: tuple[str, str]) -> None:
        df, _ = _sweep()
        df[column] = np.where(df["rep"] % 2 == 0, values[0], values[1])
        with pytest.raises(ValueError, match=f"never compared across {column}"):
            stress_figs.calibration_under_stress_table(df, knob="k2_channel")

    def test_model_without_predictive_is_left_out_but_partial_nan_fails(self) -> None:
        df, _ = _sweep()
        df.loc[df["model"] == "gp_naive", "ece"] = np.nan
        table = stress_figs.calibration_under_stress_table(df, knob="k2_channel", metrics=("ece",))
        assert "gp_naive" not in set(table["model"])
        df.loc[(df["model"] == "gp_mll") & (df["rep"] == 0), "ece"] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            stress_figs.calibration_under_stress_table(df, knob="k2_channel", metrics=("ece",))


class TestFigures:
    def test_robustness_vs_budget_written(self, tmp_path) -> None:
        df, frame = _sweep()
        written = stress_figs.plot_robustness_vs_budget(frame, df, str(tmp_path), knob="k2_channel", dataset="nhp",
                                                        margin=0.05)
        assert {os.path.basename(p) for p in written} == {"robustness_vs_budget.csv", "robustness_vs_budget.svg"}
        svg = (tmp_path / "robustness_vs_budget.svg").read_text(encoding="utf-8")
        assert style.BREAKDOWN_RULE_NOTE.split("{")[0] in svg

    def test_calibration_figure_written_with_reference_lines(self, tmp_path) -> None:
        df, _ = _sweep()
        written = stress_figs.plot_calibration_under_stress(df, str(tmp_path), knob="k2_channel", dataset="nhp")
        assert {os.path.basename(p) for p in written} == {"calibration_under_stress.csv",
                                                          "calibration_under_stress.svg"}
        svg = (tmp_path / "calibration_under_stress.svg").read_text(encoding="utf-8")
        assert style.CALIBRATION_TARGET_LABEL in svg
        assert style.ONLINE_Y_SCALER_LABELS["minmax"] in svg
        assert style.CALIBRATION_TARGET_LABELS[stress_figs.CALIBRATION_TARGET] in svg
        table = pd.read_csv(tmp_path / "calibration_under_stress.csv")
        assert set(table["calibration_target"]) == {"ground_truth_mean"}

    def test_png_preview_format(self, tmp_path) -> None:
        df, _ = _sweep()
        written = stress_figs.plot_calibration_under_stress(df, str(tmp_path), knob="k2_channel", dataset="nhp",
                                                            formats=("png",))
        assert any(p.endswith(".png") for p in written)

    @pytest.mark.parametrize("knob", ["k2_channel", "k2_global", "k5_outliers", "k6_failure"])
    def test_render_all_wires_both_deliverables_for_every_live_knob(self, tmp_path, knob: str) -> None:
        df, frame = _sweep(knob)
        if knob in ("k5_outliers", "k6_failure"):
            df = df.assign(level=df["level"] - 0.5)       # a fraction ladder starting at the unstressed 0
            frame = frame.assign(level=frame["level"] - 0.5)
        out = tmp_path / knob
        _write_pickle(df, frame, str(out))
        written = stress_figs.render_all(df, str(out), knob=knob, dataset="nhp", min_snr_db=None)
        names = {os.path.basename(p) for p in written}
        assert {"robustness_vs_budget.csv", "robustness_vs_budget.svg", "calibration_under_stress.csv",
                "calibration_under_stress.svg"} <= names

    def test_render_all_refuses_mixed_scalers(self, tmp_path) -> None:
        df, _ = _sweep()
        df["online_y_scaler"] = np.where(df["subject"] == 0, "none", "minmax")
        with pytest.raises(ValueError, match="online_y_scaler"):
            stress_figs.render_all(df, str(tmp_path), knob="k2_channel", dataset="nhp", min_snr_db=None)
