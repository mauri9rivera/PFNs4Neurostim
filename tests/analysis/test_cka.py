"""Debiased CKA, target kernels, permutation null and controls (task #5 Steps 1-4)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pfns4neurostim.analysis import cka as C


def _grid(side: int = 8) -> np.ndarray:
    ax = np.linspace(0, 1, side)
    return np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)


class TestDebiasedCKA:
    def test_self_similarity_is_one(self, rng: np.random.Generator) -> None:
        Z = rng.normal(size=(50, 20))
        K = C.linear_gram(Z)
        assert C.cka_debiased(K, K) == pytest.approx(1.0)

    def test_invariant_to_orthogonal_transform_and_isotropic_scaling(self, rng: np.random.Generator) -> None:
        Z, W = rng.normal(size=(60, 10)), rng.normal(size=(60, 7))
        Q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
        a = C.cka_debiased(C.linear_gram(Z), C.linear_gram(W))
        b = C.cka_debiased(C.linear_gram(3.7 * Z @ Q), C.linear_gram(W))
        assert a == pytest.approx(b, abs=1e-10)

    def test_unrelated_high_dim_near_zero_where_biased_is_inflated(self, rng: np.random.Generator) -> None:
        """n << d: the biased estimator is large for independent data, the debiased one ~ 0."""
        n, d = 40, 400
        vals_deb, vals_bias = [], []
        for _ in range(10):
            A, B = rng.normal(size=(n, d)), rng.normal(size=(n, d))
            vals_deb.append(C.cka_debiased(C.linear_gram(A), C.linear_gram(B)))
            vals_bias.append(C.biased_linear_cka(torch.tensor(A), torch.tensor(B)))
        assert abs(np.mean(vals_deb)) < 0.05
        assert np.mean(vals_bias) > 0.5

    def test_accepts_precomputed_target_grams(self, rng: np.random.Generator) -> None:
        X = _grid()
        ks = C.target_kernels(X, rng.normal(size=len(X)), gp_posterior_cov=np.eye(len(X)))
        assert set(ks) == {"K_X", "K_GT", "K_GT_linear", "K_GP"}
        assert all(k.shape == (len(X), len(X)) for k in ks.values())


class TestPermutationNull:
    def test_related_representation_beats_null(self, rng: np.random.Generator) -> None:
        X = _grid()
        y = np.sin(4 * X[:, 0]) + np.cos(3 * X[:, 1])
        Z = np.column_stack([y, y ** 2, rng.normal(size=(len(y), 5)) * 0.1])   # encodes the response
        obs, null_mean, p = C.permutation_null(C.linear_gram(Z), C.target_kernels(X, y)["K_GT"], 200, rng)
        assert obs > null_mean + 0.2 and p < 0.01

    def test_unrelated_representation_at_null(self, rng: np.random.Generator) -> None:
        X = _grid()
        y = rng.normal(size=len(X))
        Z = rng.normal(size=(len(X), 30))
        _, _, p = C.permutation_null(C.linear_gram(Z), C.target_kernels(X, y)["K_GT"], 200, rng)
        assert p > 0.01


class TestPositiveNegativeControlMachinery:
    """Step 4 controls on a representation with known content (the PFN versions are slow tests)."""

    def test_map_from_known_kernel_exceeds_null_noise_map_does_not(self, rng: np.random.Generator) -> None:
        from pfns4neurostim.analysis.update_rule import gp_channel

        ch = gp_channel(8, 0.25, 0.2, 5, rng)
        # A kernel-smoother "representation": site features are RBF similarities to the
        # context, weighted by the observed values (what any spatial regressor computes).
        ctx = rng.choice(ch.n_sites, 20, replace=False)
        feats = C.rbf_gram(ch.X_pool, 0.25)[:, ctx] * ch.Y_trials[ctx, 0]
        _, _, p_pos = C.permutation_null(C.linear_gram(feats), C.target_kernels(ch.X_pool, ch.y_gt)["K_GT"], 300, rng)
        y_noise = rng.normal(size=ch.n_sites)
        feats_n = C.rbf_gram(ch.X_pool, 0.25)[:, ctx] * (y_noise[ctx] + 0.2 * rng.normal(size=20))
        _, _, p_neg = C.permutation_null(C.linear_gram(feats_n), C.target_kernels(ch.X_pool, y_noise)["K_GT"], 300, rng)
        assert p_pos < 0.05
        assert p_neg > 0.05


def test_gp_posterior_covariance_is_psd_and_matches_predictive_sd() -> None:
    from pfns4neurostim.models.gp.surrogates import GPSurrogate

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(12, 2))
    gp = GPSurrogate(n_opt_steps=30)
    gp.fit(X, np.sin(4 * X[:, 0]))
    Q = _grid(5)
    S = gp.posterior_covariance(Q)
    assert np.linalg.eigvalsh(0.5 * (S + S.T)).min() > -1e-8
    _, sd = gp.predict_frozen(Q)
    np.testing.assert_allclose(np.sqrt(np.diag(S) + gp.hyperparameters()["noise"]), sd, atol=1e-8)


@pytest.mark.slow
@pytest.mark.gpu
def test_layer_embeddings_shape_and_determinism() -> None:
    from pfns4neurostim.analysis.embeddings import site_embeddings
    from pfns4neurostim.analysis.update_rule import draw_context, gp_channel

    ch = gp_channel(8, 0.25, 0.2, 5, np.random.default_rng(0))
    ctx = draw_context(ch, 20, np.random.default_rng(0))
    E1 = site_embeddings(ctx.X, ctx.y, ch.X_pool, device="cuda", inference_seed=0)
    E2 = site_embeddings(ctx.X, ctx.y, ch.X_pool, device="cuda", inference_seed=0)
    assert E1.shape == (18, ch.n_sites, 192)
    np.testing.assert_allclose(E1, E2, atol=1e-5)
    # The last layer encodes the response: CKA to K_GT beats the permutation null.
    K = C.linear_gram(site_embeddings(ctx.X, ctx.y, ch.X_pool, device="cuda", readout="label_token")[-1])
    _, _, p = C.permutation_null(K, C.target_kernels(ch.X_pool, ch.y_gt)["K_GT"], 200, np.random.default_rng(0))
    assert p < 0.05
