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
    # The last layer of the feature tokens carries the response: CKA to K_GT beats the permutation null.
    K = C.linear_gram(site_embeddings(ctx.X, ctx.y, ch.X_pool, device="cuda", readout="feature_mean")[-1])
    _, _, p = C.permutation_null(K, C.target_kernels(ch.X_pool, ch.y_gt)["K_GT"], 200, np.random.default_rng(0))
    assert p < 0.05


# ---------------------------------------------------------------------------
# B1 (2026-10-06): batched null, stage tables, shadow criterion, decoder stages
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


def test_stage_table_averages_feature_tokens_draw_by_draw() -> None:
    """feature_tokens: CKA per token, averaged; the null is averaged per permutation, so p tests the mean."""
    from pfns4neurostim.analysis.embeddings import LayerEmbeddings
    from pfns4neurostim.experiments.mechanism import _cka_layer_table

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(25, 2))
    y = np.sin(5 * X[:, 0])
    tokens = np.stack([np.c_[y, rng.normal(size=(25, 3))], rng.normal(size=(25, 4))])     # [2, 25, 4]
    emb = LayerEmbeddings("feature_tokens", (3,), (tokens,))
    kernels = {"K_GT": C.target_kernels(X, y)["K_GT"]}
    (row,) = _cka_layer_table(emb, kernels, 200, np.random.default_rng(1))
    per_token = [C.cka_debiased(C.linear_gram(t), kernels["K_GT"]) for t in tokens]
    assert row["value"] == pytest.approx(np.mean(per_token)) and row["n_tokens"] == 2
    assert row["layer"] == 3 and "stage" not in row
    single = LayerEmbeddings("feature_mean", (3,), (tokens[0],))
    (r0,) = _cka_layer_table(single, kernels, 200, np.random.default_rng(1))
    assert r0["value"] == pytest.approx(per_token[0]) and r0["n_tokens"] == 1


def test_shadow_criterion_passes_only_when_the_band_narrows() -> None:
    import pandas as pd

    from pfns4neurostim.experiments.mechanism import CKA_SHADOW_KEYS, shadow_criterion

    rng = np.random.default_rng(0)
    rows = []
    for readout, spread in (("feature_mean", 1.0), ("feature_tokens", 0.5)):
        for ch in range(12):
            offset = spread * rng.normal()
            for layer in range(18):
                for target in ("K_GT", "K_GP"):
                    for rep in range(3):
                        rows.append({"readout": readout, "subject": 0, "emg": ch, "layer": layer,
                                     "target": target, "context_t": 25, "level": 1.0, "rep": rep,
                                     "cka_minus_null": offset + 0.01 * rng.normal()})
    df = pd.DataFrame(rows)
    verdict = shadow_criterion(df, dict(CKA_SHADOW_KEYS))
    assert verdict["passed"] and all(0.3 < v < 0.75 for v in verdict["ratios"].values())
    assert not shadow_criterion(df, {**CKA_SHADOW_KEYS, "max_iqr_ratio": 0.3})["passed"]
    with pytest.raises(ValueError, match="bogus"):
        shadow_criterion(df, {**CKA_SHADOW_KEYS, "readout": "bogus"})


@pytest.mark.slow
@pytest.mark.gpu
def test_readout_embeddings_are_feature_tokens_only() -> None:
    """Both readouts come from one pass over the FEATURE tokens; the label token is never read (2026-10-07)."""
    from pfns4neurostim.analysis.embeddings import READOUTS, layer_embeddings, readout_embeddings
    from pfns4neurostim.analysis.update_rule import draw_context, gp_channel
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
    np.testing.assert_allclose(layer_embeddings(eng, ch.X_pool, layers=[0, 5, 17]), np.stack(fm.arrays), atol=1e-6)
    with pytest.raises(ValueError, match="label_token"):
        readout_embeddings(eng, ch.X_pool, readouts=("label_token",))
