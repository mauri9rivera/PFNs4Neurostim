"""Debiased CKA, target kernels, deconfounding, the smooth-field surrogate and the permutation null."""
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
        from pfns4neurostim.data.synthetic_neurostim import gp_channel

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
def test_feature_mean_embeddings_shape_and_determinism() -> None:
    from pfns4neurostim.analysis.context import draw_context
    from pfns4neurostim.analysis.embeddings import readout_embeddings
    from pfns4neurostim.data.synthetic_neurostim import gp_channel
    from pfns4neurostim.models.pfn.tabpfn import FrozenTabPFN

    ch = gp_channel(8, 0.25, 0.2, 5, np.random.default_rng(0))
    ctx = draw_context(ch, 20, np.random.default_rng(0))

    def embed() -> np.ndarray:
        eng = FrozenTabPFN(device="cuda", random_state=0)
        eng.fit(ctx.X, ctx.y)
        return np.stack(readout_embeddings(eng, ch.X_pool)["feature_mean"].arrays)

    E1, E2 = embed(), embed()
    assert E1.shape == (18, ch.n_sites, 192)
    np.testing.assert_allclose(E1, E2, atol=1e-5)
    # The last layer of the feature tokens carries the response: CKA to K_GT beats the permutation null.
    _, _, p = C.permutation_null(C.linear_gram(E1[-1]), C.target_kernels(ch.X_pool, ch.y_gt)["K_GT"], 200,
                                 np.random.default_rng(0))
    assert p < 0.05


# ---------------------------------------------------------------------------
# B1 (2026-10-06): the batched null (descriptive since BETA D4)
# ---------------------------------------------------------------------------
def _loop_null(K: np.ndarray, L: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """The pre-2026-10-06 null: one gather and one HSIC per permutation."""
    kk, ll = C.hsic_unbiased(K, K), C.hsic_unbiased(L, L)
    out = []
    for _ in range(n_perm):
        p = rng.permutation(len(L))
        out.append(C.hsic_unbiased(K, L[np.ix_(p, p)]) / np.sqrt(kk * ll))
    return np.asarray(out)


def test_batched_null_equals_the_permutation_loop() -> None:
    """Sharing one permutation set across stages changes nothing per stage (same draws, same numbers)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(30, 2))
    L = C.target_kernels(X, np.sin(4 * X[:, 0]))["K_GT"]
    Ks = [C.linear_gram(rng.normal(size=(30, 8)) + i * X @ rng.normal(size=(2, 8))) for i in range(3)]
    obs, null = C.cka_null_draws(Ks, L, 300, np.random.default_rng(7))
    for i, K in enumerate(Ks):
        np.testing.assert_allclose(null[i], _loop_null(K, L, 300, np.random.default_rng(7)), atol=1e-12)
        assert obs[i] == pytest.approx(C.cka_debiased(K, L))
    # Asymmetric target (a GP posterior covariance can be numerically asymmetric) uses the transposed gather.
    A = L + 1e-3 * rng.normal(size=L.shape)
    np.testing.assert_allclose(C.cka_null_draws(Ks[:1], A, 50, np.random.default_rng(1))[1][0],
                               _loop_null(Ks[0], A, 50, np.random.default_rng(1)), atol=1e-12)
    o, m, p = C.permutation_null(Ks[0], L, 300, np.random.default_rng(7))
    assert (o, m) == pytest.approx((obs[0], null[0].mean())) and 0 < p <= 1


@pytest.mark.slow
@pytest.mark.gpu
def test_readout_embeddings_are_feature_tokens_only() -> None:
    """Both readouts come from one pass over the FEATURE tokens; the label token is never read (2026-10-07)."""
    from pfns4neurostim.analysis.embeddings import READOUTS, readout_embeddings
    from pfns4neurostim.analysis.context import draw_context
    from pfns4neurostim.data.synthetic_neurostim import gp_channel
    from pfns4neurostim.models.pfn.tabpfn import FrozenTabPFN

    assert READOUTS == ("feature_mean", "feature_tokens")
    ch = gp_channel(8, 0.25, 0.2, 5, np.random.default_rng(0))
    ctx = draw_context(ch, 20, np.random.default_rng(0))
    eng = FrozenTabPFN(device="cuda", random_state=0)
    eng.fit(ctx.X, ctx.y)
    out = readout_embeddings(eng, ch.X_pool, readouts=READOUTS, layers=[0, 5, 17])
    fm, tok = out["feature_mean"], out["feature_tokens"]
    assert fm.layers == tok.layers == (0, 5, 17)
    assert fm.arrays[0].shape == (ch.n_sites, 192) and tok.arrays[0].shape == (2, ch.n_sites, 192)
    for a, b in zip(fm.arrays, tok.arrays):
        np.testing.assert_allclose(b.mean(axis=0), a, atol=1e-6)
    with pytest.raises(ValueError, match="label_token"):
        readout_embeddings(eng, ch.X_pool, readouts=("label_token",))


# ---------------------------------------------------------------------------
# BETA Phase 1 (2026-10-21): double centring, 4-bandwidth deconfounding, the smooth-field surrogate
# ---------------------------------------------------------------------------
def _nhp_grid() -> np.ndarray:
    """The 8 x 12 NHP electrode grid in [0, 1]^2 (96 sites)."""
    ax, ay = np.linspace(0, 1, 12), np.linspace(0, 1, 8)
    return np.stack(np.meshgrid(ax, ay, indexing="ij"), -1).reshape(-1, 2)


_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)


def test_double_center_is_the_gram_of_the_centred_cloud(rng: np.random.Generator) -> None:
    Z = rng.normal(size=(20, 4))
    np.testing.assert_allclose(C.double_center(C.linear_gram(Z)), C.centered_linear_gram(Z), atol=1e-12)
    H = np.eye(20) - np.ones((20, 20)) / 20
    K = rng.normal(size=(20, 20))
    np.testing.assert_allclose(C.double_center(K), H @ K @ H, atol=1e-12)


def test_unbiased_hsic_ignores_double_centring(rng: np.random.Generator) -> None:
    """Why deconfounding may work on double-centred Grams: HSIC_1 does not see the centring itself."""
    K, L = C.linear_gram(rng.normal(size=(30, 5))), C.linear_gram(rng.normal(size=(30, 3)))
    assert C.hsic_unbiased(C.double_center(K), L) == pytest.approx(C.hsic_unbiased(K, L), rel=1e-10)


def test_pure_geometry_representation_deconfounds_to_zero(rng: np.random.Generator) -> None:
    """A representation in the span of the geometry confounds has no residual: deconfounded CKA 0, flagged."""
    X = _nhp_grid()
    dec = C.Deconfounder(C.geometry_confounds(X, _MULTIPLIERS))
    K_X = C.rbf_gram(X, C.median_distance(X))
    for K in (K_X, 3.0 * K_X + 2.0, C.rbf_gram(X, 0.5 * C.median_distance(X)) + K_X):
        assert dec.residual_fraction(K) < 1e-12
        out = C.cka_deconfounded(K, K_X, dec)
        assert out["value"] == 0.0 and out["degenerate"]
    # ... and against a target with real signal it is still 0 (nothing beyond the confounds to align)
    y = C.smooth_field_surrogate(X, 0.2, 1, rng, jitter=1e-8)[:, 0]
    assert C.cka_deconfounded(K_X, C.centered_linear_gram(y), dec)["value"] == 0.0


def test_smooth_field_surrogate_deconfounds_to_near_zero(rng: np.random.Generator) -> None:
    """Unrelated smooth fields share a smoothness floor in raw CKA; after 4-bandwidth deconfounding it is ~0."""
    X = _nhp_grid()
    dec = C.Deconfounder(C.geometry_confounds(X, _MULTIPLIERS))
    raw, dc = [], []
    for _ in range(20):
        K = C.linear_gram(C.smooth_field_surrogate(X, 0.15, 192, rng, jitter=1e-8))
        L = C.centered_linear_gram(C.smooth_field_surrogate(X, 0.15, 1, rng, jitter=1e-8)[:, 0])
        raw.append(C.cka_debiased(K, L))
        dc.append(C.cka_deconfounded(K, L, dec)["value"])
    assert np.median(raw) > 0.1                                      # the floor a site permutation would hide
    assert abs(np.median(dc)) < 0.05 and np.median(dc) < 0.5 * np.median(raw)


def test_planted_mean_map_survives_deconfounding(rng: np.random.Generator) -> None:
    X = _nhp_grid()
    dec = C.Deconfounder(C.geometry_confounds(X, _MULTIPLIERS))
    y = C.smooth_field_surrogate(X, 0.2, 1, rng, jitter=1e-8)[:, 0]
    geometry = C.rbf_gram(X, C.median_distance(X))[:, :40]            # pure-geometry columns
    Z = np.c_[3.0 * y, geometry, 0.1 * rng.normal(size=(len(X), 10))]
    out = C.cka_deconfounded(C.linear_gram(Z), C.centered_linear_gram(y), dec)
    assert not out["degenerate"] and out["value"] > 0.5


def test_layout_confound_removes_a_covariance_that_is_geometry_plus_layout() -> None:
    """A target equal to GP-fixed's covariance plus geometry is removed by the D3d confound set (layout included)."""
    from pfns4neurostim.analysis.bayes_update import GPFixedEngine

    X = _nhp_grid()
    rng = np.random.default_rng(3)
    ctx = rng.choice(len(X), 25, replace=False)
    fixed = GPFixedEngine(0.6931471805599453, 1.0, 0.05)
    fixed.fit(X[ctx], rng.normal(size=25))
    S = fixed.posterior_covariance(X)
    geom = C.geometry_confounds(X, _MULTIPLIERS)
    with_layout = C.Deconfounder([*geom, S])
    assert with_layout.residual_fraction(S + 0.5 * geom[1]) < 1e-12
    assert C.Deconfounder(geom).residual_fraction(S) > 1e-6          # geometry alone does not explain the layout


def test_deconfounder_validates_and_is_fail_fast() -> None:
    X = _nhp_grid()
    with pytest.raises(ValueError, match="at least one"):
        C.Deconfounder([])
    with pytest.raises(ValueError, match="shape"):
        C.Deconfounder([np.eye(len(X)), np.eye(5)])
    bad = np.eye(len(X))
    bad[0, 1] = np.nan
    with pytest.raises(RuntimeError, match="NaN"):
        C.Deconfounder([bad])
    with pytest.raises(RuntimeError, match="NaN"):
        C.Deconfounder([np.eye(len(X))]).residual(bad)
    with pytest.raises(ValueError, match="positive"):
        C.geometry_confounds(X, [0.5, 0.0])


def test_deconfound_function_matches_the_class_and_zeroes_the_diagonal(rng: np.random.Generator) -> None:
    X = _nhp_grid()
    confs = C.geometry_confounds(X, _MULTIPLIERS)
    K = C.linear_gram(rng.normal(size=(len(X), 7)))
    R = C.deconfound(K, confs)
    np.testing.assert_allclose(R, C.Deconfounder(confs).residual(K), atol=1e-12)
    assert np.all(np.diag(R) == 0.0)


def test_cka_or_none_flags_a_constant_representation(rng: np.random.Generator) -> None:
    """The block-0 label token is constant across sites: its CKA is None (reported), not an exception."""
    const = C.linear_gram(np.ones((30, 4)))
    L = C.linear_gram(rng.normal(size=(30, 3)))
    assert C.cka_or_none(const, L) is None
    assert C.cka_or_none(L, L) == pytest.approx(1.0)


def test_effective_rank_of_a_rank_one_target_is_one(rng: np.random.Generator) -> None:
    y = rng.normal(size=40)
    assert C.effective_rank(C.centered_linear_gram(y)) == pytest.approx(1.0)
    assert C.effective_rank(C.linear_gram(rng.normal(size=(40, 30)))) > 5.0


def test_smooth_field_surrogate_shape_smoothness_and_validation(rng: np.random.Generator) -> None:
    X = _nhp_grid()
    Z = C.smooth_field_surrogate(X, 0.3, 50, rng, jitter=1e-8)
    assert Z.shape == (96, 50)
    # neighbouring sites are more alike than distant ones (a smooth field, unlike a permutation)
    near = np.corrcoef(Z[0], Z[1])[0, 1]
    far = np.corrcoef(Z[0], Z[-1])[0, 1]
    assert near > far
    with pytest.raises(ValueError, match="lengthscale"):
        C.smooth_field_surrogate(X, 0.0, 5, rng, jitter=1e-8)
