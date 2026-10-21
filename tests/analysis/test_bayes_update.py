"""OMEGA ``bayes_update``: quadrature, exact-Bayes floors, power of the coherence tests, distances, anchors.

Fast tests use GP engines, whose Part A answers are known exactly (a coherent Bayesian model has zero drift, a
total-variance ratio of 1 and an implied covariance equal to its posterior covariance). The TabPFN fingerprint
test loads the real checkpoint on CPU and is marked ``slow``.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis.context import draw_context
from pfns4neurostim.data.synthetic_neurostim import gp_channel
from pfns4neurostim.models.gp.surrogates import NaiveGPSurrogate

TRUE_ELL = 0.2
NOISE_SD = 0.3
FIXED = {"lengthscale": TRUE_ELL, "outputscale": 1.0, "noise": NOISE_SD ** 2}
#: A coarse grid that keeps the fast tests fast while still marginalising every hyperparameter.
SMALL_GRID = {"lengthscale": (0.08, 0.8, 4), "outputscale": (0.3, 3.0, 3), "noise": (0.02, 0.3, 3)}


@pytest.fixture(scope="module")
def channel():
    return gp_channel(10, TRUE_ELL, NOISE_SD, 10, np.random.default_rng(1))


@pytest.fixture(scope="module")
def ctx(channel):
    return draw_context(channel, 25, np.random.default_rng(0))


def _fixed(ctx) -> B.GPFixedEngine:
    eng = B.GPFixedEngine(**FIXED)
    eng.fit(ctx.X, ctx.y)
    return eng


def _mixture(weights, means, sds) -> B.MixturePredictive:
    w = np.asarray(weights, dtype=float)[None]                                    # [1, C]
    m = np.asarray(means, dtype=float)[None, None]                                # [1, 1, C]
    v = (np.asarray(sds, dtype=float) ** 2)[None, None]
    return B.MixturePredictive(w, m, v)


# ---------------------------------------------------------------------------
# 1. Gauss-Hermite quadrature in normal-score space
# ---------------------------------------------------------------------------
class TestQuadrature:
    def test_gaussian_first_two_moments_are_exact(self) -> None:
        u, w = B.quadrature_levels("gauss_hermite", 20, u_min=1e-15)
        assert w.sum() == pytest.approx(1.0, abs=1e-14)
        pred = _mixture([1.0], [0.7], [1.3])
        y = B.quadrature_values(pred, 0, u[0])
        assert abs(w @ y - 0.7) < 1e-12
        assert abs(w @ (y - 0.7) ** 2 - 1.3 ** 2) < 1e-12

    def test_truncated_levels_stay_in_the_central_mass(self) -> None:
        u, w = B.quadrature_levels("truncated", 20, u_min=1e-15, central_mass=0.9999)
        assert u.shape == (1, 20) and u.min() >= 0.5e-4 - 1e-15 and u.max() <= 1 - 0.5e-4 + 1e-15

    def test_rqmc_levels_are_scrambles_of_equal_weight_and_reproducible(self) -> None:
        u1, w = B.quadrature_levels("rqmc", 16, u_min=1e-12, n_scrambles=4, rng=np.random.default_rng(3))
        u2, _ = B.quadrature_levels("rqmc", 16, u_min=1e-12, n_scrambles=4, rng=np.random.default_rng(3))
        assert u1.shape == (4, 16) and np.allclose(w, 1 / 16) and np.array_equal(u1, u2)
        assert not np.array_equal(u1[0], u1[1])
        with pytest.raises(ValueError, match="generator"):
            B.quadrature_levels("rqmc", 16, u_min=1e-12)

    def test_gaussian_mixture_moments_within_1e6(self) -> None:
        """A GP-Bayes-like mixture (similar components) is integrated to 1e-6 at the default K = 20."""
        u, w = B.quadrature_levels("gauss_hermite", 20, u_min=1e-15)
        weights, means, sds = np.array([0.5, 0.3, 0.2]), np.array([0.0, 0.1, -0.05]), np.array([0.5, 0.55, 0.6])
        y = B.quadrature_values(_mixture(weights, means, sds), 0, u[0])
        assert abs(w @ y - weights @ means) < 1e-6
        assert abs(w @ y ** 2 - weights @ (sds ** 2 + means ** 2)) < 1e-6

    def test_strongly_non_gaussian_mixture_converges_with_k(self) -> None:
        """A bimodal, unequal-SD mixture is the hard case: 2e-4 at K = 20, 2e-6 at K = 80 (measured 2026-10-15).
        Gate G3 (K = 20 vs 40 on TabPFN) is what decides K on the real predictives."""
        weights, means, sds = np.array([0.6, 0.4]), np.array([-0.3, 0.9]), np.array([0.5, 0.8])
        pred = _mixture(weights, means, sds)
        errs = []
        for K in (20, 40, 80):
            u, w = B.quadrature_levels("gauss_hermite", K, u_min=1e-15)
            errs.append(abs(w @ B.quadrature_values(pred, 0, u[0]) - weights @ means))
        assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-5


# ---------------------------------------------------------------------------
# 2. GP-fixed: the exact floor of Part A, and identity with NaiveGPSurrogate
# ---------------------------------------------------------------------------
class TestGPFixed:
    def test_matches_naive_gp_surrogate(self, channel, ctx) -> None:
        """Agreement to 1e-6: gpytorch stores the fixed hyperparameters in float32 (noise 0.09 reads back as
        0.0900000110), the engine uses them exactly; the difference is that rounding, nothing else."""
        eng = _fixed(ctx)
        ref = NaiveGPSurrogate(**FIXED)
        ref.fit(ctx.X, ctx.y)
        m, s = ref.predict_frozen(channel.X_pool)
        base = eng.base(channel.X_pool)
        np.testing.assert_allclose(base.mean[0], m, atol=1e-6)
        np.testing.assert_allclose(base.sd[0], s, atol=1e-6)
        xs, ys = channel.X_pool[[3, 40]], np.array([0.4, -1.1])
        dm, ds = ref.posterior_update(xs, ys, channel.X_pool)
        upd = eng.probe(xs, ys, channel.X_pool)
        np.testing.assert_allclose(upd.mean - base.mean, dm, atol=1e-6)
        np.testing.assert_allclose(upd.sd - base.sd, ds, atol=1e-6)
        np.testing.assert_allclose(eng.posterior_covariance(channel.X_pool), ref.posterior_covariance(channel.X_pool),
                                   atol=1e-6)

    def test_part_a_is_exactly_bayes_including_the_replicate(self, channel, ctx) -> None:
        eng = _fixed(ctx)
        Q = channel.X_pool
        unobserved = np.setdiff1d(np.arange(channel.n_sites), ctx.sites)[:3]
        anchors = np.concatenate([unobserved, ctx.sites[:2]])                     # the last two are revisits
        rows, C, _ = B.coherence(eng, Q, anchors, K=20, eps=0.01)
        kC = eng.posterior_covariance(Q)
        np.testing.assert_allclose(C, kC[anchors], atol=1e-8)
        for r, a in zip(rows, anchors):
            assert r["max_abs_drift"] < 1e-8
            assert r["max_abs_total_var_dev"] < 1e-8
            assert r["shrink_move_ratio"] == pytest.approx(1.0, abs=1e-8)
            assert r["latent_frac"] == pytest.approx(kC[a, a] / (kC[a, a] + FIXED["noise"]), abs=1e-8)
            assert r["drift_w1"] < 1e-3                                           # grid tolerance
        cells = B.coherence_cell(rows, C, anchors, ["unobserved"] * 3 + ["observed"] * 2, np.ones(5))
        for cell in cells:
            assert cell["cov_asymmetry"] < 1e-8 and cell["psd_neg_mass"] < 1e-8

    @pytest.mark.parametrize("method,kw", [("truncated", {"central_mass": 0.9999}),
                                           ("rqmc", {"n_scrambles": 3, "rng": np.random.default_rng(5)})])
    def test_moment_matched_readouts_are_exact_under_any_measure(self, channel, ctx, method, kw) -> None:
        """G1 re-stated: the raw readouts are biased by a truncated or sampled measure, the ``*_c`` ones are not."""
        eng = _fixed(ctx)
        Q = channel.X_pool
        anchors = np.concatenate([np.setdiff1d(np.arange(channel.n_sites), ctx.sites)[:3], ctx.sites[:2]])
        rows, C, Cc = B.coherence(eng, Q, anchors, K=16, eps=0.01, method=method, **kw)
        np.testing.assert_allclose(Cc, eng.posterior_covariance(Q)[anchors], atol=1e-8)
        for r in rows:
            assert r["max_abs_drift_c"] < 1e-8 and r["max_abs_total_var_dev_c"] < 1e-8
            assert r["shrink_move_ratio_c"] == pytest.approx(1.0, abs=1e-8)
            assert abs(r["shrink_move_ratio"] - 1.0) > 1e-6                     # the raw readout is biased
        if method == "rqmc":
            assert rows[0]["n_scrambles"] == 3 and "shrink_move_ratio_se" in rows[0]
        cells = B.coherence_cell(rows, C, anchors, ["unobserved"] * 3 + ["observed"] * 2, np.ones(5), C_c=Cc)
        assert all(c["cov_asymmetry_c"] < 1e-8 for c in cells)


# ---------------------------------------------------------------------------
# 3. GP-Bayes: mixture over a hyperparameter grid
# ---------------------------------------------------------------------------
class TestGPBayes:
    def test_weights_sum_to_one_and_diagnostics_are_recorded(self, ctx) -> None:
        eng = B.GPBayesEngine(**SMALL_GRID, prune_mass=0.0)
        eng.fit(ctx.X, ctx.y)
        assert np.exp(eng.log_w).sum() == pytest.approx(1.0, abs=1e-12)
        d = eng.diagnostics()
        assert 1.0 <= d["ess"] <= d["n_grid"] == 4 * 4 * 3 * 3
        assert {"post_lengthscale_0", "post_lengthscale_1", "post_outputscale", "post_noise"} <= set(d)

    def test_single_point_grid_reproduces_gp_fixed(self, channel, ctx) -> None:
        one = B.GPBayesEngine((TRUE_ELL, TRUE_ELL, 1), (1.0, 1.0, 1), (FIXED["noise"],) * 2 + (1,), b2=0.0)
        one.fit(ctx.X, ctx.y)
        fixed = _fixed(ctx)
        Q = channel.X_pool
        np.testing.assert_array_equal(one.base(Q).mean, fixed.base(Q).mean)
        np.testing.assert_array_equal(one.probe(Q[[5]], [0.3], Q).var, fixed.probe(Q[[5]], [0.3], Q).var)

    def test_sequential_reweighting_equals_refitting_the_mixture(self, channel, ctx) -> None:
        Q = channel.X_pool
        seq = B.GPBayesEngine(**SMALL_GRID, prune_mass=0.0)
        seq.fit(ctx.X, ctx.y)
        a, y_new = 17, 1.4
        upd = seq.probe(Q[[a]], [y_new], Q)
        full = B.GPBayesEngine(**SMALL_GRID, prune_mass=0.0)
        full.fit(np.vstack([ctx.X, Q[a]]), np.append(ctx.y, y_new))
        again = full.base(Q)
        np.testing.assert_allclose(upd.weights, again.weights, atol=1e-10)
        np.testing.assert_allclose(upd.mean, again.mean, atol=1e-10)
        np.testing.assert_allclose(upd.var, again.var, atol=1e-10)

    def test_recomputed_cross_factor_equals_the_cached_one(self, channel, ctx) -> None:
        """Above ``cache_bytes`` (G4's dense grid) the factor is rebuilt per probe site: same numbers."""
        Q = channel.X_pool
        cached = B.GPBayesEngine(**SMALL_GRID, chunk=7)
        lean = B.GPBayesEngine(**SMALL_GRID, chunk=7, cache_bytes=0)
        for eng in (cached, lean):
            eng.fit(ctx.X, ctx.y)
        a, b = cached.probe(Q[[4, 60]], [0.2, -0.7], Q), lean.probe(Q[[4, 60]], [0.2, -0.7], Q)
        np.testing.assert_allclose(a.mean, b.mean, atol=1e-12)
        np.testing.assert_allclose(a.var, b.var, atol=1e-12)
        np.testing.assert_allclose(cached.latent_covariance(Q), lean.latent_covariance(Q), atol=1e-12)

    def test_rank_one_factors_are_bit_identical_to_the_materialised_mixture(self, channel, ctx) -> None:
        """Task #28 (2026-10-21): large probes keep factors, not [P, M, C] arrays; every readout is unchanged."""
        Q = channel.X_pool
        lean = B.GPBayesEngine(**SMALL_GRID, prune_mass=0.0)                       # 144 components > 64: factors
        dense = B.GPBayesEngine(**SMALL_GRID, prune_mass=0.0, exact_max_components=10_000)   # materialised path
        for eng in (lean, dense):
            eng.fit(ctx.X, ctx.y)
        xs, ys = Q[[4, 4, 4, 60, 60]], np.array([0.2, -0.7, 1.9, 0.1, -2.4])
        r1, mat = lean.probe(xs, ys, Q), dense.probe(xs, ys, Q)
        assert isinstance(r1, B.RankOneMixturePredictive) and type(mat) is B.MixturePredictive
        for got, want in ((r1.weights, mat.weights), (r1.means, mat.means), (r1.variances, mat.variances),
                          (r1.mean, mat.mean), (r1.var, mat.var)):
            np.testing.assert_array_equal(got, want)
        binned = B.MixturePredictive(mat.weights, mat.means, mat.variances)        # same CDF algorithm as r1
        grid = B.site_grid([lean.base(Q), r1], 1e-4, 64)
        np.testing.assert_array_equal(r1.cdf(grid), binned.cdf(grid))
        np.testing.assert_array_equal(r1.support(1e-4), binned.support(1e-4))
        np.testing.assert_array_equal(r1.ppf(np.array([0.1, 0.5, 0.9]), np.array([4, 33])),
                                      binned.ppf(np.array([0.1, 0.5, 0.9]), np.array([4, 33])))
        sub, sub_mat = r1.take([3, 0]), binned.take([3, 0])
        np.testing.assert_array_equal(sub.mean, sub_mat.mean)
        np.testing.assert_array_equal(sub.cdf(grid), sub_mat.cdf(grid))

    def test_part_a_stays_within_gate_g1(self, channel, ctx) -> None:
        eng = B.GPBayesEngine(**SMALL_GRID)
        eng.fit(ctx.X, ctx.y)
        anchors = np.concatenate([np.setdiff1d(np.arange(channel.n_sites), ctx.sites)[:4], ctx.sites[:3]])
        rows, C, _ = B.coherence(eng, channel.X_pool, anchors, K=20, eps=0.01)
        for r in rows:
            assert r["drift_abs"] <= 1e-3
            assert abs(r["total_var_ratio"] - 1.0) <= 1e-2
        for cell in B.coherence_cell(rows, C, anchors, ["unobserved"] * 4 + ["observed"] * 3, np.ones(7)):
            assert cell["cov_asymmetry"] <= 1e-2

    def test_latent_covariance_is_the_implied_covariance(self, channel, ctx) -> None:
        """A3's tower property for a mixture: C(x', .) is the mixture's latent posterior covariance row."""
        eng = B.GPBayesEngine(**SMALL_GRID)
        eng.fit(ctx.X, ctx.y)
        anchors = np.array([11, 52])
        _, C, _ = B.coherence(eng, channel.X_pool, anchors, K=40, eps=0.01)
        cov = eng.latent_covariance(channel.X_pool)
        np.testing.assert_allclose(C, cov[anchors], atol=2e-3)


# ---------------------------------------------------------------------------
# 4. Power: the toy updaters fail what gate G2 says they must
# ---------------------------------------------------------------------------
class TestToyPower:
    def test_inflated_gain_breaks_the_law_of_total_variance(self, channel, ctx) -> None:
        toy = B.ToyUpdater(B.GPFixedEngine(**FIXED), gain_mult=1.5, name="toy_gain")
        toy.fit(ctx.X, ctx.y)
        rows, _, _ = B.coherence(toy, channel.X_pool, np.array([3, 30, 77]), K=20, eps=0.01)
        assert np.median([r["shrink_move_ratio"] for r in rows]) <= 0.5
        assert np.median([r["shrink_move_ratio"] for r in rows]) == pytest.approx(1 / 1.5 ** 2, abs=1e-8)

    def test_mean_bias_breaks_the_martingale(self, channel, ctx) -> None:
        toy = B.ToyUpdater(B.GPFixedEngine(**FIXED), bias_sd=0.25, name="toy_bias")
        toy.fit(ctx.X, ctx.y)
        rows, _, _ = B.coherence(toy, channel.X_pool, np.array([3, 30, 77]), K=20, eps=0.01)
        assert np.median([r["drift_abs"] for r in rows]) >= 0.05


class TestRQMCFloor:
    """G1 under RQMC (user, 2026-10-15): an exact Bayesian engine's residual is consistent with zero in SE units,
    while a biased updater is not."""

    ANCHORS = np.array([3, 30, 52, 77])
    POOLS = ["unobserved"] * 4

    def test_gp_bayes_residual_is_within_three_standard_errors(self, channel, ctx) -> None:
        eng = B.GPBayesEngine(**SMALL_GRID)
        eng.fit(ctx.X, ctx.y)
        rows, cells = B.rqmc_floor(eng, channel.X_pool, self.ANCHORS, self.POOLS, 16, 0.01, n_scrambles=8,
                                   rng=np.random.default_rng(0))
        crit = B.rqmc_critical_z(8, 2 * len(rows) + 1, 0.05)
        assert max(max(r["drift_z"], r["total_var_z"]) for r in rows) <= crit
        assert cells[0]["cov_asymmetry_z"] <= crit

    def test_mean_bias_is_many_standard_errors_from_zero(self, channel, ctx) -> None:
        toy = B.ToyUpdater(B.GPFixedEngine(**FIXED), bias_sd=0.25, name="toy_bias")
        toy.fit(ctx.X, ctx.y)
        rows, _ = B.rqmc_floor(toy, channel.X_pool, self.ANCHORS, self.POOLS, 16, 0.01, n_scrambles=8,
                               rng=np.random.default_rng(0))
        assert min(r["drift_z"] for r in rows) >= 10.0

    def test_needs_two_scrambles(self, channel, ctx) -> None:
        with pytest.raises(ValueError, match="2 scrambles"):
            B.rqmc_floor(_fixed(ctx), channel.X_pool, self.ANCHORS, self.POOLS, 16, 0.01, n_scrambles=1,
                         rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# 5. Distances between distributions on the grid
# ---------------------------------------------------------------------------
class TestGridDistances:
    def test_w1_of_identical_distributions_is_zero(self) -> None:
        a = _mixture([1.0], [0.2], [0.7])
        grid = B.site_grid([a], 1e-9, 2001)
        assert B.w1(a.cdf(grid), a.cdf(grid), grid)[0, 0] == 0.0

    def test_w1_of_equal_sd_gaussians_is_the_mean_shift(self) -> None:
        a, b = _mixture([1.0], [0.0], [0.5]), _mixture([1.0], [0.8], [0.5])
        grid = B.site_grid([a, b], 1e-9, 4001)
        assert B.w1(a.cdf(grid), b.cdf(grid), grid)[0, 0] == pytest.approx(0.8, abs=1e-4)

    def test_grid_crps_of_a_gaussian_matches_the_closed_form(self) -> None:
        mu, sd = 0.3, 0.6
        a = _mixture([1.0], [mu], [sd])
        y = np.array([[-1.5, 0.3, 0.9, 2.4]])                                      # [M=1, H]
        grid = B.site_grid([a], 1e-9, 4001, extra=y)
        crps = B.crps_grid(a.cdf(grid), y, grid)[0, 0]
        z = (y[0] - mu) / sd
        closed = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        np.testing.assert_allclose(crps, closed, atol=1e-5)

    def test_binned_mixture_cdf_matches_the_exact_one(self) -> None:
        """The deposit-and-convolve CDF used above ``exact_max_components`` (GP-Bayes) agrees with the exact sum."""
        rng = np.random.default_rng(4)
        P, M, C = 3, 5, 400
        w = rng.dirichlet(np.ones(C), size=P)                                      # [P, C]
        mu = rng.normal(0.0, 0.4, size=(P, M, C))
        var = rng.uniform(0.04, 0.6, size=(P, M, C))
        binned = B.MixturePredictive(w, mu, var, exact_max_components=10, sd_bins=48)
        exact = B.MixturePredictive(w, mu, var, exact_max_components=10 ** 9)
        grid = B.site_grid([exact], 1e-6, 512)
        F_b, F_e = binned.cdf(grid), exact.cdf(grid)
        assert np.abs(F_b - F_e).max() < 2e-3
        assert (B.w1(F_b, F_e, grid) / exact.sd).max() < 2e-3
        np.testing.assert_array_equal(binned.mean, exact.mean)                     # moments are never binned

    def test_crps_outside_the_grid_adds_the_linear_tail(self) -> None:
        a = _mixture([1.0], [0.0], [0.2])
        grid = B.site_grid([a], 1e-9, 4001)
        far = np.array([[grid[0, -1] + 3.0]])
        wide = B.site_grid([a], 1e-9, 4001, extra=far)
        np.testing.assert_allclose(B.crps_grid(a.cdf(grid), far, grid), B.crps_grid(a.cdf(wide), far, wide),
                                   atol=1e-4)


# ---------------------------------------------------------------------------
# 6. Linear transfer and far field
# ---------------------------------------------------------------------------
class TestTransfer:
    def test_recovers_a_known_linear_transfer(self) -> None:
        T = np.linspace(-1, 1, 7)
        e = np.array([-2.0, -0.5, 0.3, 1.7])
        np.testing.assert_allclose(B.linear_transfer(e[:, None] * T[None], e), T, atol=1e-14)

    def test_robust_to_near_zero_innovations(self) -> None:
        T = np.linspace(0.0, 2.0, 5)
        e = np.array([1e-12, -1e-12, 1.0])
        rng = np.random.default_rng(0)
        dmu = e[:, None] * T[None] + 1e-9 * rng.normal(size=(3, 5))
        np.testing.assert_allclose(B.linear_transfer(dmu, e), T, atol=1e-8)

    def test_all_zero_innovations_raise(self) -> None:
        with pytest.raises(RuntimeError, match="zero"):
            B.linear_transfer(np.ones((2, 3)), np.zeros(2))

    def test_far_field_share(self) -> None:
        T = np.array([1.0, 1.0, 1.0, 1.0])
        dist = np.array([0.0, 1.0, 5.0, 6.0])
        assert B.far_field_share(T, dist, np.ones(4, bool), 4.0) == pytest.approx(0.5)
        assert B.far_field_share(np.zeros(4), dist, np.ones(4, bool), 4.0) is None


class TestHeadToHead:
    def test_an_engine_against_itself_is_a_perfect_match(self, channel, ctx) -> None:
        Q = channel.X_pool
        a = int(np.setdiff1d(np.arange(channel.n_sites), ctx.sites)[0])
        held = B.heldout_trials(channel, ctx.sites, ctx.trial_idx)
        preds = {}
        for name in ("gp_fixed", "twin"):
            eng = _fixed(ctx)
            base = eng.base(Q)
            preds[name] = {"base": base, "trial": eng.probe(Q[[a] * 3], held[a, :3], Q),
                           "ladder": eng.probe(Q[[a] * 4], base.mean[0, a] + np.array([-3, -1, 1, 3.0]), Q)}
        grid = B.site_grid([p for v in preds.values() for p in v.values()], 1e-6, 512, extra=held)
        rows, surprise, profile = B.headtohead_anchor(
            preds, ["gp_fixed"], anchor=a, trial_y=held[a, :3], trial_idx=np.arange(3),
            ladder_c=np.array([-3, -1, 1, 3.0]), s_bayes_anchor=1.0, heldout=held,
            observed=np.isin(np.arange(channel.n_sites), ctx.sites), dist=B.site_distances(channel.ch2xy, [a])[0],
            y_grid=grid, far_pitch=4.0, saturation_c=(1.0, 3.0), distance_bins=[0.0, 2.0, 4.0, 20.0])
        contrast = [r for r in rows if r["reference"] == "gp_fixed"]
        assert len(contrast) == 1 and contrast[0]["engine"] == "twin"
        c = contrast[0]
        assert c["footprint_r"] == pytest.approx(1.0) and c["transfer_slope"] == pytest.approx(1.0)
        assert c["gap_change"] == 0.0 and c["delta_crps"] == 0.0
        sat = [r["saturation"] for r in rows if r["source"] == "ladder"]
        assert sat == pytest.approx([1.0, 1.0])                                   # a GP's update is linear in y'
        assert len(surprise) == 8
        twin = [r for r in profile if r["engine"] == "twin"]
        ref = [r for r in profile if r["engine"] == "gp_fixed"]
        assert [r["abs_transfer"] for r in twin] == pytest.approx([r["abs_transfer"] for r in ref])
        assert {r["site_pool"] for r in profile} <= {"unobserved", "observed"}
        assert sum(r["n_sites"] for r in ref) == channel.n_sites - 1                # every site but the anchor


class TestC1FrozenReference:
    """Hypothesis C1 (user, 2026-10-19): GP-MLL frozen is the GP-like reference; a perfect GP scores 1 against it."""

    def test_frozen_update_is_the_closed_form_and_linear_in_the_surprise(self, channel, ctx) -> None:
        params = {"n_opt_steps": 20, "lr": 1.0, "optimizer": "lbfgs", "dtype": "float64", "noise_floor": 1e-3,
                  "n_restarts": 1, "lengthscale_floor": 0.05}
        eng = B.GPFrozenEngine(params)
        eng.fit(ctx.X, ctx.y)
        Q = channel.X_pool
        a = int(np.setdiff1d(np.arange(channel.n_sites), ctx.sites)[0])
        base = eng.base(Q)
        c = np.array([-3.0, -1.0, 1.0, 3.0])
        pr = eng.probe(Q[[a] * 4], base.mean[0, a] + c * base.sd[0, a], Q)
        dmu = pr.mean - base.mean                                                   # [4, M]
        np.testing.assert_allclose(dmu[3], 3 * dmu[2], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(dmu[0], -dmu[3], rtol=1e-8, atol=1e-10)
        dm_ref, _ = eng.gp.posterior_update(Q[[a]], base.mean[0, a:a + 1] + base.sd[0, a], Q, refit=False)
        np.testing.assert_allclose(dmu[2], dm_ref[0], rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("kind", ["gp_mll_frozen", "gp_fixed"])
    def test_a_gp_scored_against_itself_reaches_every_perfect_gp_value(self, channel, ctx, kind) -> None:
        """Roadmap M10 self-reference test: the values drawn as reference lines in the C1 figures."""
        if kind == "gp_mll_frozen":
            eng = B.GPFrozenEngine({"n_opt_steps": 20, "lr": 1.0, "optimizer": "lbfgs", "dtype": "float64",
                                    "noise_floor": 1e-3, "n_restarts": 1, "lengthscale_floor": 0.05})
            eng.fit(ctx.X, ctx.y)
        else:
            eng = _fixed(ctx)
        Q = channel.X_pool
        a = int(np.setdiff1d(np.arange(channel.n_sites), ctx.sites)[0])
        held = B.heldout_trials(channel, ctx.sites, ctx.trial_idx)
        c = np.array([-3.0, -1.0, 1.0, 3.0])
        base = eng.base(Q)
        pr = {"base": base, "trial": eng.probe(Q[[a] * 3], held[a, :3], Q),
              "ladder": eng.probe(Q[[a] * 4], base.mean[0, a] + c * base.sd[0, a], Q)}
        preds = {kind: pr, "twin": pr}
        grid = B.site_grid(list(pr.values()), 1e-6, 512, extra=held)
        dist = B.site_distances(channel.ch2xy, [a])[0]
        rows, _, profile = B.headtohead_anchor(
            preds, [kind], anchor=a, trial_y=held[a, :3], trial_idx=np.arange(3), ladder_c=c,
            s_bayes_anchor=float(base.sd[0, a]), heldout=held,
            observed=np.isin(np.arange(channel.n_sites), ctx.sites), dist=dist, y_grid=grid, far_pitch=4.0,
            saturation_c=(1.0, 3.0), distance_bins=[0.0, 2.0, 4.0, 20.0])
        (contrast,) = [r for r in rows if r["reference"] == kind]
        assert contrast["transfer_slope"] == pytest.approx(1.0, abs=1e-12)
        assert contrast["footprint_r"] == pytest.approx(1.0, abs=1e-12)
        own = {r["engine"]: r for r in rows if r["reference"] == "none" and r["source"] == "trial"}
        k_aa = float(eng.posterior_covariance(Q)[a, a])                          # latent k_C(x', x')
        gain = k_aa / float(base.sd[0, a]) ** 2                                   # k_C / (k_C + sigma^2)
        assert own[kind]["gain"] == pytest.approx(gain, rel=1e-6)
        assert own["twin"]["far_field_share"] == pytest.approx(own[kind]["far_field_share"])
        sat = [r["saturation"] for r in rows if r["source"] == "ladder"]
        assert sat == pytest.approx([1.0, 1.0], abs=1e-8)
        twin = [r["abs_transfer"] for r in profile if r["engine"] == "twin"]
        assert twin == pytest.approx([r["abs_transfer"] for r in profile if r["engine"] == kind])

    def test_transfer_profile_bins_and_refuses_bad_edges(self) -> None:
        T = np.array([1.0, 0.5, -0.5, 0.25])
        dist = np.array([0.0, 1.0, 1.5, 3.0])
        rows = B.transfer_profile(T, dist, np.array([False, False, True, False]), 0, [0.0, 2.0, 4.0])
        assert rows == [
            {"site_pool": "unobserved", "bin_lo": 0.0, "bin_hi": 2.0, "abs_transfer": 0.5, "n_sites": 1},
            {"site_pool": "unobserved", "bin_lo": 2.0, "bin_hi": 4.0, "abs_transfer": 0.25, "n_sites": 1},
            {"site_pool": "observed", "bin_lo": 0.0, "bin_hi": 2.0, "abs_transfer": 0.5, "n_sites": 1},
        ]
        with pytest.raises(ValueError, match="increase"):
            B.transfer_profile(T, dist, np.zeros(4, bool), 0, [0.0, 0.0])


# ---------------------------------------------------------------------------
# 7. Two-pool uniform anchors (kept stratum-mask tests first)
# ---------------------------------------------------------------------------
class TestStratumMasksAndUniformAnchors:
    """``stratum_masks`` labels and ``select_anchors_uniform`` samples; both on the 10 x 10 fixture."""

    def test_stratum_masks_names_match_anchor_strata(self, channel) -> None:
        masks = B.stratum_masks(channel)
        assert set(masks) == set(B.ANCHOR_STRATA)

    def test_stratum_masks_are_boolean_per_site(self, channel) -> None:
        for mask in B.stratum_masks(channel).values():
            assert mask.dtype == bool and mask.shape == (channel.n_sites,)

    def test_stratum_masks_sizes(self, channel) -> None:
        masks = B.stratum_masks(channel)
        q = channel.n_sites // 4
        assert masks["high"].sum() == masks["low"].sum() == masks["centre"].sum() == q
        assert masks["edge"].sum() == 4 * 10 - 4                      # perimeter of a 10 x 10 grid
        assert not (masks["high"] & masks["low"]).any()

    def test_strata_are_overlapping_labels_not_a_partition(self, channel) -> None:
        """Make the response peak sit at a grid corner: that site is both high and edge."""
        import dataclasses

        X = channel.X_pool
        tilted = dataclasses.replace(channel, y_gt=X[:, 0] + X[:, 1])
        masks = B.stratum_masks(tilted)
        assert (masks["high"] & masks["edge"]).any()

    def test_pools_are_disjoint_and_sizes_respected(self, channel, ctx) -> None:
        anchors, pools, labels = B.select_anchors_uniform(channel, 8, 4, np.random.default_rng(0), ctx.sites)
        assert anchors.shape == (12,) and len(pools) == len(labels) == 12
        assert pools == ["unobserved"] * 8 + ["observed"] * 4
        obs = set(ctx.sites.tolist())
        assert all((a in obs) == (p == "observed") for a, p in zip(anchors.tolist(), pools))
        assert len(set(anchors.tolist())) == 12

    def test_reproducible_per_seed(self, channel, ctx) -> None:
        a1, _, l1 = B.select_anchors_uniform(channel, 8, 4, np.random.default_rng(7), ctx.sites)
        a2, _, l2 = B.select_anchors_uniform(channel, 8, 4, np.random.default_rng(7), ctx.sites)
        a3, _, _ = B.select_anchors_uniform(channel, 8, 4, np.random.default_rng(8), ctx.sites)
        assert np.array_equal(a1, a2) and l1 == l2
        assert not np.array_equal(a1, a3)

    def test_labels_are_the_plus_joined_membership_of_stratum_masks(self, channel, ctx) -> None:
        masks = B.stratum_masks(channel)
        anchors, _, labels = B.select_anchors_uniform(channel, 30, 10, np.random.default_rng(1), ctx.sites)
        for a, label in zip(anchors, labels):
            assert label == ("+".join(n for n in B.ANCHOR_STRATA if masks[n][a]) or "none")

    def test_a_small_pool_contributes_every_site(self, channel) -> None:
        ctx_sites = np.arange(3)
        anchors, pools, _ = B.select_anchors_uniform(channel, 8, 4, np.random.default_rng(0), ctx_sites)
        assert pools.count("observed") == 3 and pools.count("unobserved") == 8


# ---------------------------------------------------------------------------
# 8. TabPFN: the injected row gets the train-mode fingerprint (slow, real checkpoint)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_tabpfn_injected_row_is_train_hashed_unlike_the_query_row(channel, ctx) -> None:
    eng = B.PFNEngine(device="cpu", as_train=True)
    eng.fit(ctx.X, ctx.y)
    a = int(np.setdiff1d(np.arange(channel.n_sites), ctx.sites)[0])
    x = channel.X_pool[[a]]
    injected = eng.engine.transform_x(x, as_train=True).cpu().numpy()
    query = eng.engine.transform_x(x).cpu().numpy()
    assert not np.array_equal(injected, query)
    # Several probes at one anchor share ONE fingerprint, as one refit on D + {x'} would assign it.
    out = eng.probe(np.repeat(x, 3, axis=0), np.array([-0.5, 0.0, 0.5]), channel.X_pool)
    assert out.mean.shape == (3, channel.n_sites) and np.isfinite(out.var).all()
