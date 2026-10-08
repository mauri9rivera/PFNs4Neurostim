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


# ---------------------------------------------------------------------------
# F7 update_vs_distance (B2 Step 11)
# ---------------------------------------------------------------------------
class TestDistanceProfile:
    def test_locality_index_bounds(self) -> None:
        d = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        assert U.locality_index(d, np.ones_like(d)) == pytest.approx(0.0, abs=1e-12)      # uniform field
        assert U.locality_index(d, np.eye(1, len(d)).ravel()) == pytest.approx(1.0)        # all at the anchor
        assert U.locality_index(d, (d == 3.0).astype(float)) < 0                           # all at the far edge
        assert U.locality_index(d, np.zeros_like(d)) is None

    def test_offset_fit_recovers_a_shift_plus_bump(self) -> None:
        d = np.linspace(0, 8, 60)
        b, a, ell = U.offset_fit(d, 0.15 + 0.6 * np.exp(-0.5 * d ** 2 / 1.7 ** 2))
        assert (b, a, ell) == pytest.approx((0.15, 0.6, 1.7), rel=1e-4)

    def test_transfer_of_the_true_gp_is_the_analytic_kernel_slice(self, channel) -> None:
        """T(x | x*) = k_C(x, x*) / (k_C(x*, x*) + s2) exactly for the GP that generated the probe."""
        truth = _truth()
        _, _, res = U._probe_cell(channel, truth, _truth(), 25, 6, U.DEFAULT_SURPRISES, np.random.default_rng(0))
        T, c0 = U.transfer(res)
        S = truth.gp.posterior_covariance(channel.X_pool)                                 # latent k_C
        expected = S[:, res.anchors].T / (np.diag(S)[res.anchors] + NOISE_SD ** 2)[:, None]
        np.testing.assert_allclose(T, expected, atol=1e-6)
        assert c0 == 0.5

    def test_profile_invariants_and_observed_split(self, channel) -> None:
        _, _, res = U._probe_cell(channel, _truth(), _truth(), 25, 8, U.DEFAULT_SURPRISES, np.random.default_rng(0))
        rows, cell = U.distance_profile(res, channel.ch2xy, bin_edges=[0, 0.5, 1.5, 2.5, 3.5, 5], far_pitch=4.0)
        un = [r for r in rows if not r["observed"]]
        cum = [r["cum_energy_share"] for r in sorted(un, key=lambda r: r["bin"])]
        assert all(0 <= v <= 1 + 1e-12 for v in cum) and np.all(np.diff(cum) >= -1e-12)
        assert cum[-1] == pytest.approx(1.0)
        assert {r["observed"] for r in rows} == {False, True}
        assert un[0]["bin"] == 0 and un[0]["distance_mean"] == 0.0                         # the anchor bin
        assert 0 <= cell["far_field_share"] <= 1 and 0 < cell["locality_index"] <= 1
        # A GP's update is a local bump: the anchor carries the largest transfer, the far field almost none.
        assert cell["transfer_anchor"] > 10 * cell["transfer_far_median"]

    def test_non_spatial_model_has_no_locality(self, channel) -> None:
        """The local-only control moves its own site only: all energy at the anchor (locality 1, far field 0)."""
        _, _, res = U._probe_cell(channel, _LocalOnlyEngine(), _truth(), 25, 6, U.DEFAULT_SURPRISES,
                                  np.random.default_rng(0))
        _, cell = U.distance_profile(res, channel.ch2xy, bin_edges=[0, 0.5, 1.5], far_pitch=3.0)
        assert cell["locality_index"] == pytest.approx(1.0) and cell["far_field_share"] == pytest.approx(0.0)

    def test_shuffled_coordinates_flatten_the_profile(self, channel) -> None:
        """Control (4): in the true geometry, a GP shown shuffled coordinates loses its distance decay."""
        cells = {}
        for name, shown in (("original", channel),
                            ("shuffled", U.shuffle_coordinates(channel, np.random.default_rng(3)))):
            _, _, res = U._probe_cell(shown, _truth(), _truth(), 25, 8, U.DEFAULT_SURPRISES,
                                      np.random.default_rng(0), readout_X=channel.X_pool)
            cells[name] = U.distance_profile(res, channel.ch2xy, bin_edges=[0, 0.5, 1.5, 2.5], far_pitch=3.0)[1]
        assert cells["shuffled"]["locality_index"] < 0.5 * cells["original"]["locality_index"]

    def test_surprise_profile_of_a_gp_is_flat_in_c(self, channel) -> None:
        """Control (5): a GP's update is linear in the surprise, so its far/anchor ratio does not move with |c|."""
        _, _, res = U._probe_cell(channel, _truth(), _truth(), 25, 6, U.DEFAULT_SURPRISES, np.random.default_rng(0))
        ratios = [r["far_peak_ratio"] for r in U.surprise_profile(res, channel.ch2xy, far_pitch=3.0)]
        assert len(ratios) == 4 and np.ptp(ratios) < 1e-9

    def test_offmap_point_lies_outside_the_grid(self, channel) -> None:
        x = U.offmap_point(channel.X_pool, 2.0)
        assert x[0] == pytest.approx(channel.X_pool[:, 0].max() + 2.0 * np.ptp(channel.X_pool[:, 0]))
        assert x[1] == pytest.approx(channel.X_pool[:, 1].mean())


@pytest.mark.slow
@pytest.mark.gpu
def test_layer_alignment_rows(channel) -> None:
    """The layer arm reads the feature-token mean: one row per (anchor, layer), one peak per anchor."""
    eng = U.PFNEngine(device="cuda")
    ctx = U.draw_context(channel, 25, np.random.default_rng(0))
    anchors, strata = U.select_anchors(channel, 3, np.random.default_rng(0), exclude=ctx.sites)
    res = U.run_probes(eng, _fit(ctx), ctx, channel.X_pool, anchors, strata)
    rows = U.layer_alignment(eng, res, channel.X_pool)
    assert len(rows) == 3 * 18 and "readout" not in rows[0]
    assert sum(bool(r["is_peak"]) for r in rows) == 3


def _fit(ctx: U.Context) -> U.GPFrozenEngine:
    ref = _truth()
    ref.fit(ctx.X, ctx.y)
    return ref
