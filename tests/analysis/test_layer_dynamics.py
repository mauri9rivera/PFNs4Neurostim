"""Layer-depth mechanism measurement primitives (F6-E Step 3, sprint #3).

Pure-math functions (``alignment_stats``, ``make_repeated_kfold``, ``make_spatial_kfold``,
``scale_free_ridge_r2``, ``template_bank``, ``partial_correlation``, ``prior_kernel_slice``,
``common_mode_share``, ``vector_cka``, ``symmetric_sigma_alignment``) are
tested with synthetic arrays and need no TabPFN checkpoint. The hook-based functions
(``capture_block_states``, ``capture_head_states``, ``patched_forward``, ``decoder_projection``,
``logit_lens``)
load the real ``tabpfn==6.3.2`` checkpoint on CPU and are marked ``slow``, following the
convention of ``tests/models/pfn/test_frozen_tabpfn.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pfns4neurostim.analysis import layer_dynamics as L
from pfns4neurostim.analysis import update_rule as U


# ---------------------------------------------------------------------------
# alignment_stats (A2)
# ---------------------------------------------------------------------------
class TestAlignmentStats:
    def test_recovers_a_known_positive_correlation(self, rng: np.random.Generator) -> None:
        g = rng.normal(size=60)
        z = np.abs(2.0 * g + 0.01 * rng.normal(size=60))
        out = L.alignment_stats(z, np.abs(g))
        assert out["pearson"] is not None and out["pearson"] > 0.8
        assert out["spearman"] is not None and out["spearman"] > 0.8

    def test_off_anchor_variant_drops_only_the_anchor_site(self, rng: np.random.Generator) -> None:
        from pfns4neurostim.analysis.update_rule import _pearson

        g = np.abs(rng.normal(size=40))
        z = np.abs(rng.normal(size=40)) * 0.01  # unrelated to g
        anchor = 3
        z[anchor], g[anchor] = 100.0, 100.0  # a single outlier that inflates the raw correlation
        out = L.alignment_stats(z, g, anchor=anchor)
        assert out["pearson"] > out["pearson_off_anchor"]
        mask = np.ones(40, dtype=bool)
        mask[anchor] = False
        assert out["pearson_off_anchor"] == pytest.approx(_pearson(z[mask], g[mask]))

    def test_unobserved_variant_uses_only_unobserved_sites(self, rng: np.random.Generator) -> None:
        n = 50
        g = rng.normal(size=n)
        z = np.abs(g) + 0.01 * rng.normal(size=n)
        observed = np.zeros(n, dtype=bool)
        observed[:25] = True
        out = L.alignment_stats(np.abs(z), np.abs(g), observed=observed)
        from pfns4neurostim.analysis.update_rule import _pearson

        expected = _pearson(np.abs(z)[~observed], np.abs(g)[~observed])
        assert out["pearson_unobserved"] == pytest.approx(expected)

    def test_constant_input_returns_none_spearman_and_pearson(self) -> None:
        z = np.ones(10)
        g = np.arange(10, dtype=np.float64)
        out = L.alignment_stats(z, g)
        assert out["pearson"] is None
        assert out["spearman"] is None

    def test_all_observed_gives_empty_unobserved_variant(self, rng: np.random.Generator) -> None:
        n = 20
        g = rng.normal(size=n)
        z = np.abs(g)
        observed = np.ones(n, dtype=bool)
        out = L.alignment_stats(z, g, observed=observed)
        assert out["pearson_unobserved"] is None


# ---------------------------------------------------------------------------
# make_repeated_kfold (A3)
# ---------------------------------------------------------------------------
class TestMakeRepeatedKfold:
    def test_shape_is_n_repeats_by_n_splits(self) -> None:
        repeats = L.make_repeated_kfold(23, n_splits=5, n_repeats=4, seed=7)
        assert len(repeats) == 4
        assert all(len(splits) == 5 for splits in repeats)

    def test_each_repeat_partitions_every_index_exactly_once_as_validation(self) -> None:
        n = 23
        repeats = L.make_repeated_kfold(n, n_splits=5, n_repeats=3, seed=1)
        for splits in repeats:
            seen = np.zeros(n, dtype=int)
            for train, val in splits:
                assert set(train.tolist()).isdisjoint(set(val.tolist()))
                seen[val] += 1
            np.testing.assert_array_equal(seen, np.ones(n, dtype=int))

    def test_deterministic_given_the_same_seed(self) -> None:
        a = L.make_repeated_kfold(20, n_splits=4, n_repeats=3, seed=5)
        b = L.make_repeated_kfold(20, n_splits=4, n_repeats=3, seed=5)
        for (ra), (rb) in zip(a, b):
            for (t1, v1), (t2, v2) in zip(ra, rb):
                np.testing.assert_array_equal(t1, t2)
                np.testing.assert_array_equal(v1, v2)

    def test_different_seeds_give_different_partitions(self) -> None:
        a = L.make_repeated_kfold(20, n_splits=4, n_repeats=1, seed=1)
        b = L.make_repeated_kfold(20, n_splits=4, n_repeats=1, seed=2)
        assert not np.array_equal(a[0][0][1], b[0][0][1])


# ---------------------------------------------------------------------------
# scale_free_ridge_r2 (A3): rescaling invariance, dual/primal agreement
# ---------------------------------------------------------------------------
class TestScaleFreeRidgeR2:
    def test_recovers_high_r2_for_a_low_noise_linear_signal(self, rng: np.random.Generator) -> None:
        n, p = 80, 5
        A = rng.normal(size=(n, p))
        w = rng.normal(size=p)
        y = A @ w + 0.001 * rng.normal(size=n)
        repeats = L.make_repeated_kfold(n, n_splits=5, n_repeats=3, seed=0)
        r2 = L.scale_free_ridge_r2(A, y, repeats, lambda_rel=0.01)
        assert r2 > 0.9

    def test_near_zero_for_pure_noise(self, rng: np.random.Generator) -> None:
        n, p = 80, 20
        A = rng.normal(size=(n, p))
        y = rng.normal(size=n)
        repeats = L.make_repeated_kfold(n, n_splits=5, n_repeats=3, seed=0)
        r2 = L.scale_free_ridge_r2(A, y, repeats, lambda_rel=1.0)
        assert r2 < 0.3

    def test_invariant_to_positive_rescaling_of_the_design(self, rng: np.random.Generator) -> None:
        """The defining property of this specific lambda formula (lambda ~ tr(A_c^T A_c)):
        rescaling A by any positive scalar must leave R^2 EXACTLY unchanged, to near machine
        precision -- not merely "close" (the ridge penalty and the fitted coefficients both
        scale with A, so predictions are invariant)."""
        n, p = 60, 10
        A = rng.normal(size=(n, p))
        w = rng.normal(size=p)
        y = A @ w + 0.05 * rng.normal(size=n)
        repeats = L.make_repeated_kfold(n, n_splits=5, n_repeats=4, seed=3)
        r2_1 = L.scale_free_ridge_r2(A, y, repeats, lambda_rel=0.5)
        r2_2 = L.scale_free_ridge_r2(37.0 * A, y, repeats, lambda_rel=0.5)
        r2_3 = L.scale_free_ridge_r2(1e-4 * A, y, repeats, lambda_rel=0.5)
        assert r2_1 == pytest.approx(r2_2, abs=1e-8)
        assert r2_1 == pytest.approx(r2_3, abs=1e-8)

    def test_handles_p_greater_than_n_train_via_the_dual(self, rng: np.random.Generator) -> None:
        n, p = 15, 200
        A = rng.normal(size=(n, p))
        w = rng.normal(size=p) / np.sqrt(p)
        y = A @ w + 0.01 * rng.normal(size=n)
        repeats = L.make_repeated_kfold(n, n_splits=3, n_repeats=2, seed=0)  # n_tr=10 < p=200
        r2 = L.scale_free_ridge_r2(A, y, repeats, lambda_rel=0.1)
        assert np.isfinite(r2)

    def test_zero_variance_target_raises(self) -> None:
        rng = np.random.default_rng(0)
        A = rng.normal(size=(10, 3))
        y = np.ones(10)
        repeats = L.make_repeated_kfold(10, n_splits=2, n_repeats=1, seed=0)
        with pytest.raises(RuntimeError, match="variance"):
            L.scale_free_ridge_r2(A, y, repeats, lambda_rel=1.0)

    def test_dual_and_primal_agree_on_a_p_less_than_n_case(self, rng: np.random.Generator) -> None:
        """Internally p < n_train takes the PRIMAL branch; here the dual formula is computed
        independently by hand on the same fold and must agree to machine precision."""
        n, p = 24, 8
        A = rng.normal(size=(n, p))
        w = rng.normal(size=p)
        y = A @ w + 0.01 * rng.normal(size=n)
        train, val = np.arange(16), np.arange(16, 24)
        lam_rel = 1.0

        pred_internal, covered = L._ridge_oof_predict(A, y, [(train, val)], lam_rel)  # noqa: SLF001

        At, yt = A[train], y[train]
        mu_a, mu_y = At.mean(axis=0), yt.mean()
        Ac, yc = At - mu_a, yt - mu_y
        lam = lam_rel * float(np.sum(Ac * Ac)) / p
        Av = A[val] - mu_a
        K = Ac @ Ac.T
        alpha = np.linalg.solve(K + lam * np.eye(len(train)), yc)
        pred_dual = Av @ (Ac.T @ alpha) + mu_y

        np.testing.assert_array_equal(np.flatnonzero(covered), val)
        np.testing.assert_allclose(pred_internal[val], pred_dual, atol=1e-10)

    def test_dual_and_primal_agree_when_p_exceeds_n_train(self, rng: np.random.Generator) -> None:
        """Internally p > n_train takes the DUAL branch; here the primal formula is computed
        independently by hand on the same fold and must agree to machine precision."""
        n, p = 10, 30
        A = rng.normal(size=(n, p))
        w = rng.normal(size=p)
        y = A @ w + 0.01 * rng.normal(size=n)
        train, val = np.arange(7), np.arange(7, 10)
        lam_rel = 1.0

        pred_internal, covered = L._ridge_oof_predict(A, y, [(train, val)], lam_rel)  # noqa: SLF001

        At, yt = A[train], y[train]
        mu_a, mu_y = At.mean(axis=0), yt.mean()
        Ac, yc = At - mu_a, yt - mu_y
        lam = lam_rel * float(np.sum(Ac * Ac)) / p
        Av = A[val] - mu_a
        w_hat = np.linalg.solve(Ac.T @ Ac + lam * np.eye(p), Ac.T @ yc)
        pred_primal = Av @ w_hat + mu_y

        np.testing.assert_array_equal(np.flatnonzero(covered), val)
        np.testing.assert_allclose(pred_internal[val], pred_primal, atol=1e-8)


# ---------------------------------------------------------------------------
# template_bank (B1)
# ---------------------------------------------------------------------------
class TestTemplateBank:
    def test_recovers_the_true_lengthscale_from_a_profile_generated_by_conditioned_profile(
        self, rng: np.random.Generator
    ) -> None:
        side = 8
        ax = np.linspace(0, 1, side)
        Q = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
        anchor = 10
        context_X = rng.uniform(size=(5, 2))
        noise_ratio = 0.1
        true_ell = 0.3
        ladder = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0]

        profile = U.conditioned_profile(true_ell, Q, anchor, context_X, noise_ratio)
        out = L.template_bank(profile, Q, anchor, context_X, noise_ratio, ladder)

        assert out["ell_eff"] == pytest.approx(true_ell)
        assert out["r2_conditioned"] == pytest.approx(1.0, abs=1e-8)

    def test_reports_the_gp_template_r2_when_supplied(self, rng: np.random.Generator) -> None:
        side = 6
        ax = np.linspace(0, 1, side)
        Q = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
        anchor = 5
        context_X = rng.uniform(size=(4, 2))
        profile = U.conditioned_profile(0.25, Q, anchor, context_X, 0.1)
        out = L.template_bank(profile, Q, anchor, context_X, 0.1, [0.25], delta_gp_abs=profile)
        assert out["r2_gp"] == pytest.approx(1.0, abs=1e-8)

    def test_ch2xy_uses_pitch_distance_instead_of_euclidean_q_distance(self, rng: np.random.Generator) -> None:
        M = 12
        Q = rng.uniform(size=(M, 2))  # arbitrary, unrelated to the integer grid below
        anchor = 0
        ch2xy = np.stack(np.meshgrid(np.arange(4), np.arange(3), indexing="ij"), -1).reshape(-1, 2)[:M]
        z = rng.uniform(low=0.1, high=1.0, size=M)
        out = L.template_bank(z, Q, anchor, rng.uniform(size=(3, 2)), 0.1, [0.3], ch2xy=ch2xy)

        dist = U.site_distances(ch2xy, np.asarray([anchor]))[0]
        g = z / z[anchor]
        t = -dist
        scale = float(np.dot(t, g) / np.dot(t, t))
        ss_tot = float(((g - g.mean()) ** 2).sum())
        expected_r2_distance = 1.0 - float(((g - scale * t) ** 2).sum()) / ss_tot
        assert out["r2_distance"] == pytest.approx(expected_r2_distance)

    def test_zero_anchor_value_raises(self, rng: np.random.Generator) -> None:
        Q = rng.uniform(size=(10, 2))
        z = rng.uniform(low=0.1, high=1.0, size=10)
        z[0] = 0.0
        with pytest.raises(RuntimeError, match="anchor"):
            L.template_bank(z, Q, 0, rng.uniform(size=(3, 2)), 0.1, [0.2])


# ---------------------------------------------------------------------------
# common_mode_share (S1 / C4): exact orthogonal decomposition
# ---------------------------------------------------------------------------
class TestCommonModeShare:
    def test_decomposition_is_exactly_orthogonal(self, rng: np.random.Generator) -> None:
        Z = rng.normal(size=(30, 15))
        mean = Z.mean(axis=0, keepdims=True)
        residual = Z - mean
        common = np.repeat(mean, len(Z), axis=0)

        cross_term = float(np.sum(common * residual))
        assert abs(cross_term) < 1e-8  # the mean and the zero-sum residual are orthogonal

        total = float((Z ** 2).sum())
        common_energy = float((common ** 2).sum())
        residual_energy = float((residual ** 2).sum())
        assert total == pytest.approx(common_energy + residual_energy, rel=1e-10)
        assert L.common_mode_share(Z) == pytest.approx(common_energy / total)

    def test_zero_for_a_mean_centered_input(self, rng: np.random.Generator) -> None:
        noise = rng.normal(size=(20, 5))
        centered = noise - noise.mean(axis=0, keepdims=True)
        assert L.common_mode_share(centered) == pytest.approx(0.0, abs=1e-10)

    def test_one_for_a_pure_constant_shift(self, rng: np.random.Generator) -> None:
        shift = rng.normal(size=5)
        Z = np.tile(shift, (20, 1))
        assert L.common_mode_share(Z) == pytest.approx(1.0)

    def test_zero_energy_input_raises(self) -> None:
        with pytest.raises(RuntimeError, match="zero-energy"):
            L.common_mode_share(np.zeros((10, 4)))


# ---------------------------------------------------------------------------
# vector_cka (B4 / C5)
# ---------------------------------------------------------------------------
class TestVectorCKA:
    def test_recovers_a_planted_rank1_signal(self, rng: np.random.Generator) -> None:
        M, d = 30, 10
        v = rng.normal(size=M)
        direction = rng.normal(size=d)
        dZ = np.outer(v, direction)
        assert L.vector_cka(dZ, v) == pytest.approx(1.0, abs=1e-6)

    def test_near_zero_for_unrelated_random_representations(self, rng: np.random.Generator) -> None:
        M, d = 50, 15
        v = rng.normal(size=M)
        dZ = rng.normal(size=(M, d))
        assert abs(L.vector_cka(dZ, v)) < 0.25

    def test_mixing_in_noise_lowers_the_recovered_alignment(self, rng: np.random.Generator) -> None:
        M, d = 40, 12
        v = rng.normal(size=M)
        direction = rng.normal(size=d)
        clean = np.outer(v, direction)
        noisy = clean + 3.0 * rng.normal(size=(M, d))
        assert L.vector_cka(clean, v) > L.vector_cka(noisy, v)


# ---------------------------------------------------------------------------
# symmetric_sigma_alignment (H5)
# ---------------------------------------------------------------------------
class TestSymmetricSigmaAlignment:
    def test_matches_manual_symmetrization_and_pearson(self, rng: np.random.Generator) -> None:
        red = rng.uniform(low=0.1, high=1.0, size=40)
        sigma_plus = 2.0 * red + 0.01 * rng.normal(size=40)
        sigma_minus = 2.0 * red + 0.01 * rng.normal(size=40)
        out = L.symmetric_sigma_alignment(sigma_plus, sigma_minus, red)

        manual_sym = 0.5 * (sigma_plus + sigma_minus)
        expected_pearson = float(np.corrcoef(manual_sym, red)[0, 1])
        assert out["pearson"] == pytest.approx(expected_pearson)
        assert out["spearman"] is not None and out["spearman"] > 0.9

    def test_constant_symmetric_part_returns_none(self) -> None:
        red = np.linspace(0, 1, 20)
        sigma_plus = np.full(20, 3.0)
        sigma_minus = np.full(20, 3.0)
        out = L.symmetric_sigma_alignment(sigma_plus, sigma_minus, red)
        assert out["pearson"] is None
        assert out["spearman"] is None

    def test_antisymmetric_sigma_pair_cancels_to_a_constant(self) -> None:
        """The construction this closes M5 for: a GP-like variance reduction is sign-symmetric
        in the surprise c, so an antisymmetric sigma_plus/sigma_minus pair (what an
        antisymmetrized readout would otherwise destroy) still leaves a well-defined,
        non-degenerate symmetric part when the pair is not purely antisymmetric noise."""
        rng = np.random.default_rng(0)
        base = rng.uniform(low=0.1, high=1.0, size=30)
        antisym_noise = rng.normal(size=30)
        sigma_plus = base + antisym_noise
        sigma_minus = base - antisym_noise
        red = base
        out = L.symmetric_sigma_alignment(sigma_plus, sigma_minus, red)
        # The antisymmetric noise cancels exactly in the symmetric part, leaving base vs base.
        assert out["pearson"] == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# make_spatial_kfold (A3 secondary check): contiguous held-out bands
# ---------------------------------------------------------------------------
def _grid_coords(rows: int = 8, cols: int = 12) -> np.ndarray:
    """Integer electrode-grid coordinates, shape [rows * cols, 2]."""
    return np.stack(np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij"), -1).reshape(-1, 2).astype(float)


def _mean_pairwise_distance(coords: np.ndarray, idx: np.ndarray) -> float:
    pts = coords[idx]
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    n = len(idx)
    return float(d.sum() / (n * (n - 1)))


class TestMakeSpatialKfold:
    def test_structure_matches_make_repeated_kfold(self) -> None:
        coords = _grid_coords()
        out = L.make_spatial_kfold(coords, n_splits=5, n_repeats=4, seed=0)
        ref = L.make_repeated_kfold(len(coords), n_splits=5, n_repeats=4, seed=0)
        assert isinstance(out, tuple) and len(out) == len(ref) == 4
        for repeat in out:
            assert isinstance(repeat, tuple) and len(repeat) == 5
            for pair in repeat:
                assert isinstance(pair, tuple) and len(pair) == 2
                assert all(isinstance(a, np.ndarray) for a in pair)

    def test_train_and_val_are_disjoint_and_cover_every_index(self) -> None:
        coords = _grid_coords()
        n = len(coords)
        for repeat in L.make_spatial_kfold(coords, n_splits=5, n_repeats=3, seed=1):
            for train, val in repeat:
                assert set(train.tolist()).isdisjoint(set(val.tolist()))
                np.testing.assert_array_equal(np.sort(np.concatenate([train, val])), np.arange(n))

    def test_every_index_is_held_out_exactly_once_per_repeat(self) -> None:
        coords = _grid_coords()
        n = len(coords)
        for repeat in L.make_spatial_kfold(coords, n_splits=5, n_repeats=6, seed=2):
            seen = np.zeros(n, dtype=int)
            for _, val in repeat:
                seen[val] += 1
            np.testing.assert_array_equal(seen, np.ones(n, dtype=int))

    def test_deterministic_given_the_same_seed(self) -> None:
        coords = _grid_coords()
        a = L.make_spatial_kfold(coords, n_splits=4, n_repeats=3, seed=5)
        b = L.make_spatial_kfold(coords, n_splits=4, n_repeats=3, seed=5)
        for ra, rb in zip(a, b):
            for (t1, v1), (t2, v2) in zip(ra, rb):
                np.testing.assert_array_equal(t1, t2)
                np.testing.assert_array_equal(v1, v2)

    def test_different_seeds_give_different_partitions(self) -> None:
        coords = _grid_coords()
        a = L.make_spatial_kfold(coords, n_splits=4, n_repeats=1, seed=1)
        b = L.make_spatial_kfold(coords, n_splits=4, n_repeats=1, seed=2)
        assert not np.array_equal(a[0][0][1], b[0][0][1])

    def test_held_out_bands_are_more_compact_than_a_random_split(self) -> None:
        """The property that motivates the scheme: validation sites are spatial neighbours.
        Compared against a random split of the same sizes on the same 8 x 12 grid."""
        coords = _grid_coords()
        n = len(coords)
        spatial = L.make_spatial_kfold(coords, n_splits=5, n_repeats=10, seed=0)
        random = L.make_repeated_kfold(n, n_splits=5, n_repeats=10, seed=0)
        per_repeat_spatial = [np.mean([_mean_pairwise_distance(coords, v) for _, v in r]) for r in spatial]
        per_repeat_random = [np.mean([_mean_pairwise_distance(coords, v) for _, v in r]) for r in random]
        assert np.mean(per_repeat_spatial) < 0.85 * np.mean(per_repeat_random)
        # No single projection direction is as dispersed as a random split.
        assert max(per_repeat_spatial) < np.mean(per_repeat_random)

    def test_degenerate_partition_raises(self) -> None:
        """Two distinct degenerate cases, each caught with its own message: a single fold would hold
        out everything (leaving no training set), and a single site has no spatial extent to cut."""
        with pytest.raises(RuntimeError, match="no usable spatial partition"):
            L.make_spatial_kfold(_grid_coords(), n_splits=1, n_repeats=2, seed=0)
        with pytest.raises(RuntimeError, match="no spatial extent"):
            L.make_spatial_kfold(np.zeros((1, 2)), n_splits=5, n_repeats=2, seed=0)

    def test_all_identical_coordinates_raise(self) -> None:
        """Fixed 2026-10-12: a constant projection used to fall through to contiguous *index* blocks,
        which would have been scored as a leakage-free spatial split while holding out nothing
        spatial at all."""
        with pytest.raises(RuntimeError, match="no spatial extent"):
            L.make_spatial_kfold(np.zeros((20, 2)), n_splits=5, n_repeats=3, seed=0)


# ---------------------------------------------------------------------------
# scale_free_ridge_r2: partial coverage after the (pred, covered) change
# ---------------------------------------------------------------------------
class TestScaleFreeRidgeR2Coverage:
    @staticmethod
    def _problem(rng: np.random.Generator, n: int = 40, p: int = 4) -> tuple[np.ndarray, np.ndarray]:
        A = rng.normal(size=(n, p))
        y = A @ rng.normal(size=p) + 0.01 * rng.normal(size=n)
        return A, y

    def test_oof_predict_returns_pred_and_covered_mask(self, rng: np.random.Generator) -> None:
        A, y = self._problem(rng)
        repeats = L.make_repeated_kfold(len(y), n_splits=4, n_repeats=1, seed=0)
        out = L._ridge_oof_predict(A, y, repeats[0], 0.1)  # noqa: SLF001
        assert isinstance(out, tuple) and len(out) == 2
        pred, covered = out
        assert pred.shape == (len(y),) and covered.shape == (len(y),) and covered.dtype == bool
        assert covered.all()
        assert np.isfinite(pred[covered]).all()

    def test_partition_that_omits_indices_still_gives_a_finite_r2(self, rng: np.random.Generator) -> None:
        A, y = self._problem(rng)
        n = len(y)
        omitted = np.arange(0, n, 5)                       # every 5th site is never held out
        keep = np.setdiff1d(np.arange(n), omitted)
        folds = np.array_split(keep, 4)
        partial = tuple((np.setdiff1d(np.arange(n), val), val) for val in folds)
        _, covered = L._ridge_oof_predict(A, y, partial, 0.1)  # noqa: SLF001
        np.testing.assert_array_equal(np.flatnonzero(~covered), omitted)
        r2 = L.scale_free_ridge_r2(A, y, [partial], lambda_rel=0.1)
        assert np.isfinite(r2)
        assert r2 > 0.9  # low-noise linear signal, scored on covered sites only

    def test_r2_is_scored_only_on_covered_indices(self, rng: np.random.Generator) -> None:
        """Recomputed by hand on the covered slice only; the uncovered entries of ``pred`` are
        uninitialised and must never enter the score."""
        A, y = self._problem(rng)
        n = len(y)
        folds = np.array_split(np.arange(10, n), 3)         # sites 0..9 are never held out
        partial = tuple((np.setdiff1d(np.arange(n), v), v) for v in folds)
        lam = 0.5
        pred, covered = L._ridge_oof_predict(A, y, partial, lam)  # noqa: SLF001
        yc = y[covered]
        expected = 1.0 - float(((yc - pred[covered]) ** 2).sum()) / float(((yc - yc.mean()) ** 2).sum())
        assert L.scale_free_ridge_r2(A, y, [partial], lambda_rel=lam) == pytest.approx(expected, abs=1e-12)

    def test_rescaling_invariance_still_holds_with_a_partial_partition(self, rng: np.random.Generator) -> None:
        A, y = self._problem(rng, n=50, p=6)
        n = len(y)
        folds = np.array_split(np.arange(7, n), 4)         # sites 0..6 uncovered
        partial = tuple((np.setdiff1d(np.arange(n), v), v) for v in folds)
        r2_1 = L.scale_free_ridge_r2(A, y, [partial], lambda_rel=0.3)
        r2_2 = L.scale_free_ridge_r2(123.0 * A, y, [partial], lambda_rel=0.3)
        r2_3 = L.scale_free_ridge_r2(1e-5 * A, y, [partial], lambda_rel=0.3)
        assert r2_1 == pytest.approx(r2_2, abs=1e-8)
        assert r2_1 == pytest.approx(r2_3, abs=1e-8)

    def test_spatial_partition_is_accepted_and_finite(self, rng: np.random.Generator) -> None:
        coords = _grid_coords(6, 8)
        A, y = self._problem(rng, n=len(coords), p=5)
        repeats = L.make_spatial_kfold(coords, n_splits=4, n_repeats=3, seed=0)
        assert np.isfinite(L.scale_free_ridge_r2(A, y, repeats, lambda_rel=0.1))

    def test_raises_when_no_repeat_holds_out_a_scoreable_set(self, rng: np.random.Generator) -> None:
        A, y = self._problem(rng, n=12, p=2)
        # A repeat that holds out a single site: its covered target has zero variance (ss_tot == 0).
        single = ((np.arange(1, 12), np.array([0])),)
        with pytest.raises(RuntimeError, match="no repeat held out a scoreable set"):
            L.scale_free_ridge_r2(A, y, [single], lambda_rel=0.1)

    def test_raises_for_an_empty_list_of_repeats(self, rng: np.random.Generator) -> None:
        A, y = self._problem(rng, n=12, p=2)
        with pytest.raises(RuntimeError, match="no repeat held out a scoreable set"):
            L.scale_free_ridge_r2(A, y, [], lambda_rel=0.1)


# ---------------------------------------------------------------------------
# partial_correlation (C1 / C1')
# ---------------------------------------------------------------------------
class TestPartialCorrelation:
    def test_collapses_when_both_variables_are_driven_by_the_control(self, rng: np.random.Generator) -> None:
        n = 500
        z = rng.normal(size=n)
        x = z + 0.1 * rng.normal(size=n)
        y = 2.0 * z + 0.1 * rng.normal(size=n)
        assert np.corrcoef(x, y)[0, 1] > 0.99
        out = L.partial_correlation(x, y, z)
        assert out is not None and abs(out) < 0.2

    def test_stays_high_when_x_and_y_share_a_component_independent_of_the_control(
        self, rng: np.random.Generator
    ) -> None:
        n = 500
        z = rng.normal(size=n)
        shared = rng.normal(size=n)
        x = shared + z + 0.1 * rng.normal(size=n)
        y = shared - 0.5 * z + 0.1 * rng.normal(size=n)
        out = L.partial_correlation(x, y, z)
        assert out is not None and out > 0.9

    def test_accepts_1d_and_column_controls_identically(self, rng: np.random.Generator) -> None:
        n = 100
        z = rng.normal(size=n)
        x, y = z + rng.normal(size=n), z + rng.normal(size=n)
        a = L.partial_correlation(x, y, z)
        b = L.partial_correlation(x, y, z[:, None])
        assert a is not None and b is not None
        assert a == pytest.approx(b, abs=1e-12)

    def test_2d_controls_are_all_used(self, rng: np.random.Generator) -> None:
        n = 600
        c1, c2 = rng.normal(size=n), rng.normal(size=n)
        x = c1 + c2 + 0.2 * rng.normal(size=n)
        y = c1 - c2 + 0.2 * rng.normal(size=n)
        with_both = L.partial_correlation(x, y, np.column_stack([c1, c2]))
        with_one = L.partial_correlation(x, y, c1)
        assert with_both is not None and with_one is not None
        assert abs(with_both) < 0.2          # both common causes removed
        assert with_one < -0.5               # c2 still induces an anti-correlation

    def test_matches_the_closed_form_for_three_variables(self, rng: np.random.Generator) -> None:
        """Single-control partial correlation equals (r_xy - r_xz r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
        exactly for sample correlations, so the tolerance is numerical, not statistical."""
        cov = np.array([[1.0, 0.5, 0.6], [0.5, 1.0, 0.4], [0.6, 0.4, 1.0]])
        data = rng.normal(size=(2000, 3)) @ np.linalg.cholesky(cov).T
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        r = np.corrcoef(data.T)
        r_xy, r_xz, r_yz = r[0, 1], r[0, 2], r[1, 2]
        expected = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
        assert L.partial_correlation(x, y, z) == pytest.approx(expected, abs=1e-9)

    def test_too_few_samples_returns_none(self) -> None:
        assert L.partial_correlation(np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([0.3, 0.1])) is None

    def test_returns_none_when_a_residual_is_constant(self, rng: np.random.Generator) -> None:
        """Fixed 2026-10-12 with a relative-residual guard: a least-squares residual that should
        vanish comes back at ~1e-17, so an exact `std == 0` check never fired and two dust vectors
        were correlated instead. The near-degenerate regime is the operating regime for the geometry
        test, which is why this path matters rather than being an edge case."""
        z = rng.normal(size=50)
        assert L.partial_correlation(3.0 * z + 2.0, rng.normal(size=50), z) is None
        assert L.partial_correlation(np.full(50, 2.0), rng.normal(size=50), z) is None
        assert L.partial_correlation(z, rng.normal(size=50), z) is None


# ---------------------------------------------------------------------------
# prior_kernel_slice
# ---------------------------------------------------------------------------
class TestPriorKernelSlice:
    def test_equals_one_at_the_anchor_and_is_the_global_maximum(self, rng: np.random.Generator) -> None:
        Q = rng.uniform(size=(30, 3))
        k = L.prior_kernel_slice(Q, 7, np.array([0.2, 0.3, 0.5]))
        assert k.shape == (30,)
        assert k[7] == 1.0
        assert k.max() == 1.0 and int(np.argmax(k)) == 7
        assert ((k > 0) & (k <= 1.0)).all()

    def test_decays_monotonically_with_distance_on_a_1d_grid(self) -> None:
        Q = np.linspace(0, 1, 21)[:, None]
        k = L.prior_kernel_slice(Q, 0, np.array([0.3]))
        assert (np.diff(k) < 0).all()

    def test_matches_the_closed_form_ard_rbf(self, rng: np.random.Generator) -> None:
        Q = rng.uniform(size=(15, 2))
        ell = np.array([0.4, 0.15])
        k = L.prior_kernel_slice(Q, 3, ell)
        expected = np.exp(-0.5 * (((Q - Q[3]) / ell) ** 2).sum(axis=1))
        np.testing.assert_allclose(k, expected, rtol=1e-12)

    def test_per_dimension_lengthscales_make_the_short_axis_decay_faster(self) -> None:
        Q = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]])
        k = L.prior_kernel_slice(Q, 0, np.array([0.1, 1.0]))   # short lengthscale on axis 0
        assert k[1] < k[2]                                      # same offset, faster decay along axis 0
        assert k[1] == pytest.approx(np.exp(-0.5 * (0.2 / 0.1) ** 2))
        assert k[2] == pytest.approx(np.exp(-0.5 * (0.2 / 1.0) ** 2))
        swapped = L.prior_kernel_slice(Q, 0, np.array([1.0, 0.1]))
        assert swapped[1] > swapped[2]

    def test_scalar_lengthscale_broadcasts_over_dimensions(self, rng: np.random.Generator) -> None:
        Q = rng.uniform(size=(12, 3))
        scalar = L.prior_kernel_slice(Q, 2, 0.25)
        vector = L.prior_kernel_slice(Q, 2, np.full(3, 0.25))
        np.testing.assert_allclose(scalar, vector, rtol=1e-12)


# ---------------------------------------------------------------------------
# logit_lens: argument validation (no checkpoint needed)
# ---------------------------------------------------------------------------
class TestLogitLensValidation:
    def test_mismatched_sign_sequences_raise_value_error(self) -> None:
        plus = [np.zeros((3, 192)) for _ in range(4)]
        minus = [np.zeros((3, 192)) for _ in range(3)]
        with pytest.raises(ValueError, match="4 blocks for \\+c but 3 for -c"):
            L.logit_lens(None, plus, minus)


# ===========================================================================
# Hook-based functions: need the real tabpfn==6.3.2 checkpoint (CPU, slow)
# ===========================================================================
@pytest.fixture(scope="module")
def toy_context() -> tuple[np.ndarray, np.ndarray]:
    """A tiny, non-constant 2D context: ``(X_ctx [6, 2], y_ctx [6])``."""
    rng = np.random.default_rng(0)
    X = rng.random((6, 2))
    y = rng.random(6)
    return X, y


@pytest.fixture(scope="module")
def engine(toy_context: tuple[np.ndarray, np.ndarray]):
    """A ``FrozenTabPFN`` fitted on :func:`toy_context`, CPU only."""
    from pfns4neurostim.models.pfn.tabpfn import FrozenTabPFN

    X, y = toy_context
    eng = FrozenTabPFN(device="cpu", random_state=0)
    eng.fit(X, y)
    return eng


class TestCaptureBlockStates:
    pytestmark = pytest.mark.slow

    def test_shapes_and_requested_layers_are_all_present(self, engine) -> None:
        X_query = np.random.default_rng(1).random((5, 2))
        n_blocks = len(engine._model.transformer_encoder.layers)
        out = L.capture_block_states(engine, X_query, layers=[0, 5, n_blocks - 1])
        assert out.layers == (0, 5, n_blocks - 1)
        assert len(out.feature_tokens) == len(out.label_token) == 3
        for feat, lab in zip(out.feature_tokens, out.label_token):
            assert feat.shape[0] == 5
            assert lab.shape == (5, feat.shape[2])

    def test_none_layers_captures_every_block(self, engine) -> None:
        X_query = np.random.default_rng(2).random((3, 2))
        n_blocks = len(engine._model.transformer_encoder.layers)
        out = L.capture_block_states(engine, X_query)
        assert out.layers == tuple(range(n_blocks))

    def test_deterministic_given_the_same_inputs(self, engine) -> None:
        X_query = np.random.default_rng(3).random((4, 2))
        a = L.capture_block_states(engine, X_query, layers=[0, 3])
        b = L.capture_block_states(engine, X_query, layers=[0, 3])
        for fa, fb in zip(a.feature_tokens, b.feature_tokens):
            np.testing.assert_array_equal(fa, fb)
        for la, lb in zip(a.label_token, b.label_token):
            np.testing.assert_array_equal(la, lb)

    def test_injecting_an_extra_row_changes_the_captured_state(self, engine) -> None:
        X_query = np.random.default_rng(4).random((4, 2))
        base = L.capture_block_states(engine, X_query, layers=[1])
        extra = L.capture_block_states(
            engine, X_query, layers=[1], X_extra=np.array([[0.5, 0.5]]), y_extra=[1.0]
        )
        assert not np.allclose(base.feature_tokens[0], extra.feature_tokens[0])

    def test_invalid_layer_index_raises(self, engine) -> None:
        X_query = np.random.default_rng(5).random((2, 2))
        with pytest.raises(IndexError):
            L.capture_block_states(engine, X_query, layers=[999])


class TestCaptureHeadStates:
    pytestmark = pytest.mark.slow

    def test_shapes(self, engine) -> None:
        X_query = np.random.default_rng(6).random((4, 2))
        heads = L.capture_head_states(engine, X_query)
        assert heads.h1.shape == (4, 192)
        assert heads.h2.shape == (4, 384)
        assert heads.h3.shape[0] == 4

    def test_decoder_projection_of_h1_exactly_reproduces_h3(self, engine) -> None:
        """decoder_projection applies the SAME decoder_dict['standard'] module; feeding it H1
        (the decoder's own captured input) must reproduce H3 (the decoder's own captured
        output) exactly, since both come from one real forward pass through one module."""
        X_query = np.random.default_rng(7).random((4, 2))
        heads = L.capture_head_states(engine, X_query)
        projected = L.decoder_projection(engine, heads.h1)
        np.testing.assert_allclose(projected, heads.h3, atol=1e-4)


class TestPatchedForward:
    pytestmark = pytest.mark.slow

    def test_identity_patch_reproduces_the_unpatched_forward_exactly(self, engine) -> None:
        X_query = np.random.default_rng(8).random((4, 2))
        unpatched = L.patched_forward(engine, X_query)
        patched = L.patched_forward(engine, X_query, patches={0: lambda z: z, 5: lambda z: z})
        assert torch.equal(unpatched, patched)

    def test_block17_feature_token_patch_leaves_logits_bit_identical(self, engine) -> None:
        """C0.e: the block-17 feature tokens never reach the decoder (only the label token
        does), so scrambling them must leave the final logits bit-identical."""
        n_blocks = len(engine._model.transformer_encoder.layers)
        last = n_blocks - 1
        X_query = np.random.default_rng(9).random((4, 2))
        unpatched = L.patched_forward(engine, X_query)

        def scramble_features_only(z: torch.Tensor) -> torch.Tensor:
            out = z.clone()
            out[:, :-1] = out[:, :-1] + 100.0
            return out

        patched = L.patched_forward(engine, X_query, patches={last: scramble_features_only})
        assert torch.equal(unpatched, patched)

    def test_block17_label_token_patch_changes_the_logits(self, engine) -> None:
        n_blocks = len(engine._model.transformer_encoder.layers)
        last = n_blocks - 1
        X_query = np.random.default_rng(9).random((4, 2))
        unpatched = L.patched_forward(engine, X_query)

        def scramble_label_only(z: torch.Tensor) -> torch.Tensor:
            out = z.clone()
            out[:, -1] = out[:, -1] + 100.0
            return out

        patched = L.patched_forward(engine, X_query, patches={last: scramble_label_only})
        assert not torch.equal(unpatched, patched)

    def test_patching_a_middle_block_changes_the_output(self, engine) -> None:
        X_query = np.random.default_rng(10).random((4, 2))
        unpatched = L.patched_forward(engine, X_query)
        patched = L.patched_forward(engine, X_query, patches={2: lambda z: z + 5.0})
        assert not torch.equal(unpatched, patched)


class TestLogitLens:
    pytestmark = pytest.mark.slow

    X_EXTRA = np.array([[0.5, 0.5]])

    @staticmethod
    def _surprises(toy_context: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
        y_mean = float(np.mean(toy_context[1]))
        return y_mean + 0.4, y_mean - 0.4

    def _label_tokens(self, engine, toy_context, X_query):
        y_plus, y_minus = self._surprises(toy_context)
        plus = L.capture_block_states(engine, X_query, X_extra=self.X_EXTRA, y_extra=[y_plus])
        minus = L.capture_block_states(engine, X_query, X_extra=self.X_EXTRA, y_extra=[y_minus])
        return plus.label_token, minus.label_token

    def _real_antisymmetric_update(self, engine, toy_context, X_query) -> np.ndarray:
        """The model's actual ``(mean_+ - mean_-) / 2`` from two real forward passes."""
        y_plus, y_minus = self._surprises(toy_context)
        Qt = engine.transform_x(X_query)
        Xe = engine.transform_x(self.X_EXTRA).unsqueeze(0)              # [1, 1, F]
        means = []
        for y_val in (y_plus, y_minus):
            ye = engine.transform_y(torch.tensor([[y_val]], dtype=torch.float32, device=engine.device))
            mean, _ = engine.moments(engine.forward(Xe, ye, Qt))        # [1, q]
            means.append(mean[0])
        return 0.5 * (means[0] - means[1])                              # [q]

    def test_shape_is_blocks_by_query_sites_and_finite(self, engine, toy_context) -> None:
        X_query = np.random.default_rng(11).random((5, 2))
        plus, minus = self._label_tokens(engine, toy_context, X_query)
        lens = L.logit_lens(engine, plus, minus)
        n_blocks = len(engine._model.transformer_encoder.layers)
        assert lens.shape == (n_blocks, 5)
        assert np.isfinite(lens).all()

    def test_last_block_equals_the_models_real_antisymmetrised_prediction_change(
        self, engine, toy_context
    ) -> None:
        """Decoding block 17's label token is exactly what the real forward pass does, so the
        lens value at the final block must equal the real antisymmetrised update (float32
        round-trip only; measured agreement ~1.5e-5)."""
        X_query = np.random.default_rng(12).random((5, 2))
        plus, minus = self._label_tokens(engine, toy_context, X_query)
        lens = L.logit_lens(engine, plus, minus)
        real = self._real_antisymmetric_update(engine, toy_context, X_query)
        assert np.abs(real).max() > 0  # the injected row actually moves the prediction
        np.testing.assert_allclose(lens[-1], real, atol=1e-4)

    def test_lens_is_antisymmetric_under_swapping_the_two_signs(self, engine, toy_context) -> None:
        X_query = np.random.default_rng(13).random((4, 2))
        plus, minus = self._label_tokens(engine, toy_context, X_query)
        np.testing.assert_allclose(
            L.logit_lens(engine, plus, minus), -L.logit_lens(engine, minus, plus), atol=1e-12
        )

    def test_identical_sign_sequences_give_a_zero_update(self, engine, toy_context) -> None:
        X_query = np.random.default_rng(14).random((3, 2))
        plus, _ = self._label_tokens(engine, toy_context, X_query)
        np.testing.assert_array_equal(L.logit_lens(engine, plus, plus), 0.0)
