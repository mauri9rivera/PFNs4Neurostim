"""Per-step traces (R², simple regret, exploration) recorded along the BO run (2026-09-21)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from pfns4neurostim.evaluation import metrics
from pfns4neurostim.evaluation.bo_runner import run_channel_bo

BUDGET = 12
N_INIT = 4


class TestMetrics:
    def test_r2_perfect_and_mean_predictor(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert metrics.r2_score(y, y) == pytest.approx(1.0)
        assert metrics.r2_score(y, np.full(4, y.mean())) == pytest.approx(0.0)

    def test_exploration_is_ratio_in_raw_units(self) -> None:
        raw = np.array([2.0, 8.0, 4.0])
        assert metrics.exploration_score(raw, 1) == pytest.approx(1.0)
        assert metrics.exploration_score(raw, 2) == pytest.approx(0.5)

    def test_exploration_rejects_standardized_units(self) -> None:
        with pytest.raises(RuntimeError, match="raw"):
            metrics.exploration_score(np.array([-2.0, -1.0, -3.0]), 0)


class TestTraces:
    def test_traces_have_one_entry_per_recommendation(self, tiny_channel) -> None:
        scaler = StandardScaler().fit(np.array([[1.0], [3.0], [5.0]]))
        ch = dataclasses.replace(tiny_channel, meta={"scaler_y": scaler})
        res = run_channel_bo("gp_naive", ch, acq_fn="ei", budget=BUDGET, n_init=N_INIT, seed=1, device="cpu")
        t = res.trajectory
        n = len(t["best_rec_indices"])
        assert n == BUDGET - N_INIT + 1
        assert len(t["r2_per_step"]) == n
        assert len(t["recommended_regret_per_step"]) == n
        assert len(t["exploration_per_step"]) == n
        assert t["r2_per_step"][-1] == pytest.approx(res.row["r2"])
        assert res.row["exploration_score"] == pytest.approx(t["exploration_per_step"][-1])
        assert res.row["recommended_regret"] == pytest.approx(t["recommended_regret_per_step"][-1])
        assert all(0.0 <= r <= 1.0 for r in t["recommended_regret_per_step"])

    def test_no_scaler_means_exploration_not_computed(self, tiny_channel) -> None:
        res = run_channel_bo("gp_naive", tiny_channel, acq_fn="ei", budget=BUDGET, n_init=N_INIT, seed=1, device="cpu")
        assert res.trajectory["exploration_per_step"] is None
        assert "exploration_score" not in res.row
