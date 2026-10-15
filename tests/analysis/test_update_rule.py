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


# ---------------------------------------------------------------------------
# conditioned_profile / fit_conditioned_lengthscale (F6-E: conditioned_profile factored out
# of fit_conditioned_lengthscale's local closure into a top-level function, now in __all__,
# so analysis.layer_dynamics's template bank (B1) can evaluate it at a ladder of ell values).
# ---------------------------------------------------------------------------
class TestConditionedProfile:
    def test_fit_conditioned_lengthscale_unchanged_on_a_synthetic_grid(self) -> None:
        """The refactor must not change what the fit recovers: a profile generated from a
        known-ell RBF GP (via conditioned_profile itself) is fit back to the same ell --
        the same positive-control shape as TestGates.test_positive_control_true_gp_passes,
        but exercised directly against the extracted function rather than through the probe.
        """
        Q = np.linspace(0, 10, 11)[:, None]
        context_X = np.array([[2.0], [5.0], [8.0]])
        anchor = 5
        true_ell = 1.3
        noise_ratio = 0.1
        g = U.conditioned_profile(true_ell, Q, anchor, context_X, noise_ratio)
        fitted = U.fit_conditioned_lengthscale(g, Q, anchor, context_X, noise_ratio)
        assert fitted == pytest.approx(true_ell, rel=1e-3)

    def test_conditioned_profile_matches_a_brute_force_gp_posterior_covariance(self) -> None:
        """Cross-check against a direct matrix-inverse GP posterior-covariance computation
        (independent of conditioned_profile's own Cholesky-solve implementation) on a tiny
        2D context, so a bug in the vectorized solve path cannot hide behind a self-consistent
        rewrite of the same math.
        """
        rng = np.random.default_rng(1)
        Q = rng.uniform(0, 10, size=(9, 2))
        context_X = rng.uniform(0, 10, size=(4, 2))
        anchor = 3
        ell = 2.1
        noise_ratio = 0.25

        def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
            return np.exp(-0.5 * d2 / ell ** 2)

        K_CC = rbf(context_X, context_X) + noise_ratio * np.eye(len(context_X))
        K_CC_inv = np.linalg.inv(K_CC)
        x_star = Q[anchor][None, :]
        k_C_Q = rbf(Q, context_X)                        # [M, t]
        k_C_star = rbf(context_X, x_star)                 # [t, 1]
        k_Q_star = rbf(Q, x_star)[:, 0]                    # [M]
        post_cov = k_Q_star - (k_C_Q @ K_CC_inv @ k_C_star)[:, 0]
        post_var_star = 1.0 - (k_C_star.T @ K_CC_inv @ k_C_star)[0, 0]
        expected = post_cov / post_var_star

        got = U.conditioned_profile(ell, Q, anchor, context_X, noise_ratio)
        np.testing.assert_allclose(expected, got, atol=1e-8)

    def test_conditioned_profile_is_symmetric_and_monotone_on_an_isotropic_grid(self) -> None:
        """On a symmetric 1D grid with a single, centred context point, the profile must be
        symmetric about the anchor and decay monotonically with distance from it -- an
        isotropic-kernel sanity check independent of the brute-force cross-check above.
        """
        Q = np.linspace(-6, 6, 13)[:, None]
        context_X = np.array([[0.0]])
        anchor = 6  # Q[6] == 0.0, coincides with the sole context point
        g = U.conditioned_profile(1.5, Q, anchor, context_X, noise_ratio=0.2)
        np.testing.assert_allclose(g, g[::-1], atol=1e-10)
        half = g[6:]
        assert np.all(np.diff(half) <= 1e-12)


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


class TestStratumMasksAndUniformAnchors:
    """``stratum_masks`` labels and ``select_anchors_uniform`` samples; both on the 10 x 10 fixture."""

    def test_stratum_masks_names_match_anchor_strata(self, channel) -> None:
        masks = U.stratum_masks(channel)
        assert set(masks) == set(U.ANCHOR_STRATA)

    def test_stratum_masks_are_boolean_per_site(self, channel) -> None:
        for mask in U.stratum_masks(channel).values():
            assert mask.dtype == bool and mask.shape == (channel.n_sites,)

    def test_stratum_masks_sizes(self, channel) -> None:
        masks = U.stratum_masks(channel)
        q = channel.n_sites // 4
        assert masks["high"].sum() == masks["low"].sum() == masks["centre"].sum() == q
        assert masks["edge"].sum() == 4 * 10 - 4                      # perimeter of a 10 x 10 grid
        assert not (masks["high"] & masks["low"]).any()

    def test_strata_are_overlapping_labels_not_a_partition(self, channel) -> None:
        """Make the response peak sit at a grid corner: that site is both high and edge."""
        import dataclasses

        X = channel.X_pool
        tilted = dataclasses.replace(channel, y_gt=X[:, 0] + X[:, 1])
        masks = U.stratum_masks(tilted)
        assert (masks["high"] & masks["edge"]).any()

    def test_uniform_anchors_respect_n_anchors_and_are_distinct_and_sorted(self, channel) -> None:
        anchors, labels = U.select_anchors_uniform(channel, 12, np.random.default_rng(0))
        assert anchors.shape == (12,) and len(labels) == 12
        assert len(set(anchors.tolist())) == 12
        assert np.array_equal(anchors, np.sort(anchors))

    def test_uniform_anchors_avoid_the_excluded_context_sites(self, channel) -> None:
        exclude = np.random.default_rng(3).choice(channel.n_sites, size=40, replace=False)
        for seed in range(5):
            anchors, _ = U.select_anchors_uniform(channel, 12, np.random.default_rng(seed), exclude)
            assert set(anchors.tolist()).isdisjoint(set(exclude.tolist()))

    def test_uniform_anchors_are_reproducible_per_seed(self, channel) -> None:
        a1, l1 = U.select_anchors_uniform(channel, 12, np.random.default_rng(7))
        a2, l2 = U.select_anchors_uniform(channel, 12, np.random.default_rng(7))
        a3, _ = U.select_anchors_uniform(channel, 12, np.random.default_rng(8))
        assert np.array_equal(a1, a2) and l1 == l2
        assert not np.array_equal(a1, a3)

    def test_labels_are_the_plus_joined_membership_of_stratum_masks(self, channel) -> None:
        masks = U.stratum_masks(channel)
        anchors, labels = U.select_anchors_uniform(channel, 30, np.random.default_rng(1))
        for a, label in zip(anchors, labels):
            expected = "+".join(n for n in U.ANCHOR_STRATA if masks[n][a]) or "none"
            assert label == expected
            assert set(label.split("+")) <= set(U.ANCHOR_STRATA) | {"none"}

    def test_falls_back_to_the_whole_grid_when_too_few_unobserved_sites_remain(self, channel) -> None:
        exclude = np.arange(channel.n_sites - 5)                      # only 5 selectable sites
        anchors, labels = U.select_anchors_uniform(channel, 12, np.random.default_rng(0), exclude)
        assert anchors.shape == (12,) and len(labels) == 12

    def test_n_anchors_is_clamped_to_the_grid_size(self, channel) -> None:
        anchors, _ = U.select_anchors_uniform(channel, channel.n_sites + 50, np.random.default_rng(0))
        assert anchors.shape == (channel.n_sites,)
