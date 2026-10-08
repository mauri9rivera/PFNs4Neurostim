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

    @pytest.mark.parametrize("model", ["gp_naive", "gp_mll"])
    def test_fit_trace_aligns_with_recommendations(self, tiny_channel, model: str) -> None:
        """Fit trace: one entry per fit (each step + the final refit); the last equals the tidy-row diagnostic."""
        from pfns4neurostim.evaluation.bo_loop import FIT_TRACE_KEYS

        res = run_channel_bo(model, tiny_channel, acq_fn="ei", budget=BUDGET, n_init=N_INIT, seed=1, device="cpu")
        trace = res.trajectory["fit_trace"]
        assert set(trace) == set(FIT_TRACE_KEYS)
        n = len(res.trajectory["best_rec_indices"])
        for key, values in trace.items():
            assert values.shape == (n,)
            assert values.dtype == np.float32
            assert float(values[-1]) == pytest.approx(res.diagnostics[key], rel=1e-6, nan_ok=True)

    def test_fixed_gp_trace_is_constant(self, tiny_channel) -> None:
        res = run_channel_bo("gp_naive", tiny_channel, acq_fn="ei", budget=BUDGET, n_init=N_INIT, seed=1, device="cpu")
        ls = res.trajectory["fit_trace"]["gp_lengthscale"]
        assert np.allclose(ls, ls[0])

    def test_no_fit_diagnostics_means_no_trace(self, tiny_channel) -> None:
        res = run_channel_bo("random", tiny_channel, acq_fn="random", budget=BUDGET, n_init=N_INIT, seed=1)
        assert res.trajectory["fit_trace"] is None

    def test_no_scaler_means_exploration_not_computed(self, tiny_channel) -> None:
        res = run_channel_bo("gp_naive", tiny_channel, acq_fn="ei", budget=BUDGET, n_init=N_INIT, seed=1, device="cpu")
        assert res.trajectory["exploration_per_step"] is None
        assert "exploration_score" not in res.row
