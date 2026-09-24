"""Update-rule alignment (task #9): metrics and validation gates.

The fast tests validate the *probe* with GP engines, whose answers are known exactly; the
``slow``/``gpu`` tests run the same gates on TabPFN. Per research_design §Hyp C, if a gate
fails with the true GP as the probed model, the probe is at fault, not the model.
"""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.analysis import update_rule as U
from pfns4neurostim.models.gp.surrogates import GPSurrogate, NaiveGPSurrogate

TRUE_ELL = 0.2
NOISE_SD = 0.3


def _truth() -> U.GPFrozenEngine:
    return U.GPFrozenEngine(
        NaiveGPSurrogate(lengthscale=TRUE_ELL, outputscale=1.0, noise=NOISE_SD ** 2), "truth"
    )


class _LocalOnlyEngine:
    """Non-spatial control model: an observation changes the prediction at its own site only."""

    name = "local_only"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean = float(np.mean(y))

    def base(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full(len(Q), self.mean), np.ones(len(Q))

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.full((len(xs), len(Q)), self.mean)
        for p, x in enumerate(xs):
            hit = np.all(np.isclose(Q, x), axis=1)
            m[p, hit] += 0.5 * (ys[p] - self.mean)
        return m, np.ones_like(m)


@pytest.fixture(scope="module")
def channel():
    return U.gp_channel(10, TRUE_ELL, NOISE_SD, 10, np.random.default_rng(1))


class TestProbeMachinery:
    def test_anchor_selection_is_stratified_and_distinct(self, channel) -> None:
        idx, strata = U.select_anchors(channel, 12, np.random.default_rng(0))
        assert len(set(idx.tolist())) == 12
        assert set(strata) == set(U.ANCHOR_STRATA)

    def test_context_draws_valid_trials(self, channel) -> None:
        ctx = U.draw_context(channel, 25, np.random.default_rng(0))
        assert len(np.unique(ctx.sites)) == 25 and np.isfinite(ctx.y).all()

    def test_gp_vs_itself_is_perfectly_aligned_and_linear(self, channel) -> None:
        rows, cell, _ = U._probe_cell(channel, _truth(), _truth(), 25, 8, U.DEFAULT_SURPRISES,
                                      np.random.default_rng(0))
        for r in rows:
            assert r["rho_shape"] == pytest.approx(1.0)
            assert r["gain_ratio"] == pytest.approx(1.0)
            assert r["lin_r2"] == pytest.approx(1.0)
            assert r["saturation_index"] == pytest.approx(0.0, abs=1e-9)
            # Value-independent SD change: constant (undefined) or zero up to float noise.
            assert r["dsd_surprise_corr"] is None or abs(r["dsd_surprise_corr"]) < 1e-6
        assert cell["sym_err"] == pytest.approx(0.0, abs=1e-8)
        assert cell["psd_neg_mass"] == pytest.approx(0.0, abs=1e-8)

    def test_null_is_far_below_alignment(self, channel) -> None:
        rows, _, _ = U._probe_cell(channel, _truth(), _truth(), 25, 12, U.DEFAULT_SURPRISES,
                                   np.random.default_rng(0))
        null = np.median([r["rho_shape_null_offanchor"] for r in rows])
        assert null < 0.5

    def test_icc(self) -> None:
        rng = np.random.default_rng(0)
        target = rng.normal(size=(20, 1))
        assert U.icc_oneway(np.repeat(target, 5, axis=1)) == pytest.approx(1.0)
        assert U.icc_oneway(target + 0.01 * rng.normal(size=(20, 5))) > 0.99
        assert abs(U.icc_oneway(rng.normal(size=(200, 5)))) < 0.1


class TestGates:
    """Step 2 validation gates, run with GP engines (known answers)."""

    def test_positive_control_true_gp_passes(self) -> None:
        gate = U.positive_control(_truth(), _truth(), lengthscale=TRUE_ELL, noise_sd=NOISE_SD)
        assert gate["passed"], gate
        assert gate["ell_rel_error"] < 0.01          # exact model -> exact lengthscale

    def test_positive_control_mll_gp_passes(self) -> None:
        gate = U.positive_control(U.GPFrozenEngine(GPSurrogate()), _truth(),
                                  lengthscale=TRUE_ELL, noise_sd=NOISE_SD)
        assert gate["passed"], gate

    def test_half_decay_estimator_would_fail_the_positive_control(self, channel) -> None:
        """Why ell_hat conditions on the context: the plain half-decay fit is biased low."""
        rows, _, _ = U._probe_cell(channel, _truth(), _truth(), 25, 12, U.DEFAULT_SURPRISES,
                                   np.random.default_rng(0))
        half = np.median([r["ell_half_decay"] for r in rows])
        cond = np.median([r["ell_hat"] for r in rows])
        assert abs(cond - TRUE_ELL) / TRUE_ELL < 0.01
        assert abs(half - TRUE_ELL) / TRUE_ELL > 0.25

    def test_negative_control_spatial_model_passes(self, channel) -> None:
        gate = U.negative_control(U.GPFrozenEngine(GPSurrogate()), U.GPFrozenEngine(GPSurrogate()), channel)
        assert gate["passed"], gate
        assert gate["rho_offanchor_intact"] > 0.9

    def test_non_spatial_model_sits_at_the_null(self, channel) -> None:
        rows, cell, _ = U._probe_cell(channel, _LocalOnlyEngine(), _truth(), 25, 12,
                                      U.DEFAULT_SURPRISES, np.random.default_rng(0))
        # Off the anchor it does not update at all -> alignment undefined, not spuriously high.
        assert cell["rho_shape_offanchor_median"] is None

    def test_seed_floor_deterministic_engine(self, channel) -> None:
        gate = U.seed_floor(lambda s: _truth(), _truth(), channel, n_seeds=3, n_anchors=6)
        assert gate["passed"] and gate["icc_ell_hat"] == pytest.approx(1.0)

    def test_linear_regime_gp(self, channel) -> None:
        rows, _, _ = U._probe_cell(channel, U.GPFrozenEngine(GPSurrogate()), _truth(), 25, 8,
                                   U.DEFAULT_SURPRISES, np.random.default_rng(0))
        assert U.linear_regime_check(rows)["passed"]

    def test_refit_arm_reports_refit_lengthscale(self, channel) -> None:
        rows, _, _ = U._probe_cell(channel, U.GPRefitEngine(GPSurrogate(n_opt_steps=30)),
                                   _truth(), 25, 2, (-0.5, 0.5), np.random.default_rng(0))
        assert all(r["ell_gp_refit"] > 0 for r in rows)

    def test_determinism(self, channel) -> None:
        a, _, _ = U._probe_cell(channel, _truth(), _truth(), 25, 6, U.DEFAULT_SURPRISES, np.random.default_rng(3))
        b, _, _ = U._probe_cell(channel, _truth(), _truth(), 25, 6, U.DEFAULT_SURPRISES, np.random.default_rng(3))
        assert a == b


@pytest.mark.slow
@pytest.mark.gpu
class TestTabPFN:
    """TabPFN through the gates. Records the result; only the machinery is asserted."""

    def test_frozen_forward_matches_public_api(self, channel) -> None:
        eng = U.PFNEngine(device="cuda")
        ctx = U.draw_context(channel, 25, np.random.default_rng(0))
        eng.fit(ctx.X, ctx.y)
        m, _ = eng.base(channel.X_pool)
        public = eng.engine.wrapper.predict(channel.X_pool, output_type="mean")
        np.testing.assert_allclose(m, public, atol=1e-4)

    def test_gates_run_and_negative_control_passes(self, channel) -> None:
        eng = U.PFNEngine(device="cuda")
        pos = U.positive_control(eng, _truth(), lengthscale=TRUE_ELL, noise_sd=NOISE_SD)
        neg = U.negative_control(eng, U.GPFrozenEngine(GPSurrogate()), channel)
        print(pos, neg)
        assert pos["rho_shape_median"] is not None and pos["ell_hat_median"] is not None
        assert neg["passed"], neg
