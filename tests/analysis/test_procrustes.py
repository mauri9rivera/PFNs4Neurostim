"""Gram-based Procrustes shape metrics (:mod:`pfns4neurostim.analysis.procrustes`, BETA Phase 1).

The load-bearing tests: (1) the Gram form IS ``scipy.spatial.procrustes`` on clouds, including clouds of different
widths; (2) the Bures / Procrustes duality (Harvey, Larsen & Williams 2024) -- the Bures distance between two linear
kernels equals the unnormalised orthogonal Procrustes distance, and between two covariances it equals the closed-form
2-Wasserstein distance between zero-mean Gaussians; (3) the invariances NBS quotients out and the spectrum
reweighting that separates it from CKA.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import orthogonal_procrustes, sqrtm
from scipy.spatial import procrustes as scipy_procrustes

from pfns4neurostim.analysis.cka import cka_debiased, linear_gram
from pfns4neurostim.analysis.procrustes import (
    MIN_ROWS,
    STATISTICS,
    GramNull,
    bures,
    disparity,
    gram_factor,
    gram_null,
    nbs,
    nbs_factors,
    psd_project,
    shape_distance,
)
from pfns4neurostim.seeding import rng_for

#: The NHP operating regime: 96 sites, 192-dimensional tokens (n < d).
N_SITES, N_DIM = 96, 192

#: Replicates used in the null tests. Small on purpose -- these assert behaviour, not a measurement.
N_PERM = 64


def _cloud(seed: int, n: int = N_SITES, d: int = N_DIM) -> np.ndarray:
    """A random point cloud, shape [n, d]."""
    return rng_for("test-procrustes", seed).normal(size=(n, d))


def _rotation(d: int, seed: int) -> np.ndarray:
    """A random orthogonal matrix, shape [d, d]."""
    return np.linalg.qr(rng_for("test-procrustes-rot", seed).normal(size=(d, d)))[0]


def _spd(n: int, seed: int) -> np.ndarray:
    """A random symmetric positive-definite matrix, shape [n, n]."""
    A = rng_for("test-procrustes-spd", seed).normal(size=(n, n))
    return A @ A.T / n + 0.1 * np.eye(n)


# --- scipy equivalence -------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,d", [(40, 8), (20, 20), (N_SITES, N_DIM)])
def test_disparity_from_grams_matches_scipy_on_clouds(n: int, d: int) -> None:
    """n > d, n == d and the n < d regime of the NHP grid all agree with ``scipy.spatial.procrustes``."""
    X = _cloud(10, n, d)
    Y = X @ _rotation(d, 2) * 2.3 + _cloud(11, n, d) * 0.8
    assert disparity(linear_gram(X), linear_gram(Y)) == pytest.approx(scipy_procrustes(X, Y)[2], abs=1e-10)


def test_unequal_widths_need_only_equal_rows() -> None:
    """A 192-d layer against a 32-d embedding: scipy needs zero-padding, the Gram form needs nothing (§2.6 defect)."""
    X, Y = _cloud(12, N_SITES, N_DIM), _cloud(13, N_SITES, 32)
    padded = np.c_[Y, np.zeros((N_SITES, N_DIM - 32))]
    assert disparity(linear_gram(X), linear_gram(Y)) == pytest.approx(scipy_procrustes(X, padded)[2], abs=1e-10)
    # a covariance needs no feature map at all
    assert 0.0 <= nbs(linear_gram(X), _spd(N_SITES, 1)) <= 1.0


def test_shape_distance_and_disparity_are_functions_of_nbs() -> None:
    K, L = linear_gram(_cloud(14, 30, 6)), linear_gram(_cloud(15, 30, 4))
    v = nbs(K, L)
    assert shape_distance(K, L) == pytest.approx(np.arccos(v))
    assert disparity(K, L) == pytest.approx(1.0 - v ** 2)
    Fk, Fl = gram_factor(K, center=True), gram_factor(L, center=True)
    assert nbs_factors(Fk, Fl) == pytest.approx(v, abs=1e-12)


# --- Bures duality ---------------------------------------------------------------------------------------------

def test_bures_equals_the_unnormalised_procrustes_distance() -> None:
    """``bures(XX', YY')`` = ``min_R ||X - Y R||_F`` over orthogonal R (Harvey, Larsen & Williams 2024)."""
    X, Y = _cloud(20, 25, 6), _cloud(21, 25, 6)
    R, _ = orthogonal_procrustes(Y, X)                                     # argmin ||Y R - X||_F
    direct = float(np.linalg.norm(X - Y @ R))
    assert bures(linear_gram(X), linear_gram(Y)) == pytest.approx(direct, rel=1e-10)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Rc, _ = orthogonal_procrustes(Yc, Xc)
    assert bures(linear_gram(X), linear_gram(Y), center=True) == pytest.approx(
        float(np.linalg.norm(Xc - Yc @ Rc)), rel=1e-10)


def test_bures_is_the_gaussian_w2_closed_form() -> None:
    """Commuting covariances: W2(N(0, K), N(0, L))^2 = sum (sqrt k_i - sqrt l_i)^2; general SPD: the trace formula."""
    rng = rng_for("bures-diag")
    k, l = rng.uniform(0.1, 3.0, 12), rng.uniform(0.1, 3.0, 12)
    Q = _rotation(12, 3)
    K, L = Q @ np.diag(k) @ Q.T, Q @ np.diag(l) @ Q.T                      # commuting, not diagonal
    assert bures(K, L) == pytest.approx(np.sqrt(((np.sqrt(k) - np.sqrt(l)) ** 2).sum()), rel=1e-10)
    A, B = _spd(10, 4), _spd(10, 5)
    rA = np.real(sqrtm(A))
    w2sq = np.trace(A) + np.trace(B) - 2.0 * np.trace(np.real(sqrtm(rA @ B @ rA)))
    assert bures(A, B) ** 2 == pytest.approx(w2sq, rel=1e-8)


def test_bures_is_scale_aware_where_nbs_is_not() -> None:
    """The right shape at the wrong size: NBS = 1, Bures = (sqrt(c) - 1) sqrt(tr K) (BETA β1b)."""
    K = _spd(15, 6)
    for c in (0.25, 4.0):
        assert nbs(K, c * K, center=False) == pytest.approx(1.0, abs=1e-10)
        assert bures(K, c * K) == pytest.approx(abs(np.sqrt(c) - 1.0) * np.sqrt(np.trace(K)), rel=1e-10)
    assert bures(K, K) == pytest.approx(0.0, abs=1e-6)


# --- invariances and spectrum weighting --------------------------------------------------------------------------

def test_invariant_to_rotation_isotropic_scale_and_translation() -> None:
    X = _cloud(30)
    shift = rng_for("shift").normal(size=(1, N_DIM)) * 5.0
    for Y in (X @ _rotation(N_DIM, 6) * 7.5, X @ _rotation(N_DIM, 7) * 0.02, X + shift):
        assert nbs(linear_gram(X), linear_gram(Y)) == pytest.approx(1.0, abs=1e-10)
        assert shape_distance(linear_gram(X), linear_gram(Y)) == pytest.approx(0.0, abs=1e-4)


def test_sensitive_to_anisotropic_stretch() -> None:
    """An anisotropic stretch moves the shape metric; the same map with the mean gain does not."""
    X = _cloud(31)
    gains = np.linspace(1.0, 6.0, N_DIM)
    Rm = _rotation(N_DIM, 9)
    assert disparity(linear_gram(X), linear_gram(X @ (Rm * gains[None, :]))) > 0.05
    assert disparity(linear_gram(X), linear_gram(X @ Rm * float(gains.mean()))) == pytest.approx(0.0, abs=1e-10)


def test_spectrum_reweighting_that_cka_is_nearly_blind_to() -> None:
    """NBS is a nuclear-norm quantity, CKA a Frobenius one: inflating the 95 weak directions moves only the former."""
    rng = rng_for("spectrum")
    n, d = N_SITES, N_DIM
    U = np.linalg.qr(rng.normal(size=(n, n)))[0]
    V = np.linalg.qr(rng.normal(size=(d, d)))[0][:, :n]
    s = np.concatenate([[8.0], np.ones(n - 1)])
    s_tail = s.copy()
    s_tail[1:] *= 3.0
    X, Y = (U * s[None, :]) @ V.T, (U * s_tail[None, :]) @ V.T
    d_val = disparity(linear_gram(X), linear_gram(Y))
    cka_drop = 1.0 - cka_debiased(linear_gram(X), linear_gram(Y))
    assert d_val > 0.1 and cka_drop < 0.01 and d_val > 10.0 * max(cka_drop, 1e-3)


# --- PSD projection ---------------------------------------------------------------------------------------------

def test_psd_project_clips_and_reports_the_negative_mass() -> None:
    lam = np.array([3.0, 1.0, -0.5, 0.2])
    Q = _rotation(4, 8)
    K, clipped = psd_project(Q @ np.diag(lam) @ Q.T)
    assert clipped == pytest.approx(0.5 / 4.7)
    np.testing.assert_allclose(np.linalg.eigvalsh(K), np.sort(np.clip(lam, 0, None)), atol=1e-12)
    assert psd_project(_spd(6, 9))[1] == 0.0


# --- shuffled-site null (descriptive) ----------------------------------------------------------------------------

def test_null_of_unrelated_clouds_sits_around_the_observation() -> None:
    res = gram_null(linear_gram(_cloud(40)), linear_gram(_cloud(41)), statistic="disparity", n_perm=256,
                    rng=rng_for("null-indep"))
    assert abs(res.observed - res.null_mean) < 4.0 * res.null_sd
    assert 0.05 < res.null_mean < 0.6                                       # the n < d bias of 96 x 192 clouds


def test_matched_signal_is_separated_from_the_null() -> None:
    X = _cloud(42)
    Y = X @ _rotation(N_DIM, 27) + _cloud(43) * 0.3
    res = gram_null(linear_gram(X), linear_gram(Y), statistic="nbs", n_perm=N_PERM, rng=rng_for("null-signal"))
    assert res.observed > res.null_q95


def test_null_is_reproducible_and_jsonable() -> None:
    K, L = linear_gram(_cloud(44, 30, 5)), linear_gram(_cloud(45, 30, 5))
    a = gram_null(K, L, statistic="shape_distance", n_perm=8, rng=rng_for("repro", 1))
    b = gram_null(K, L, statistic="shape_distance", n_perm=8, rng=rng_for("repro", 1))
    assert a == b and isinstance(a, GramNull)
    row = a.as_row()
    assert row["statistic"] == "shape_distance" and row["n_perm"] == 8 and row["n_rows"] == 30
    assert set(STATISTICS) == {"nbs", "shape_distance", "disparity"}


# --- fail-fast ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_raises(bad: float) -> None:
    K = linear_gram(_cloud(50, 12, 6))
    L = K.copy()
    L[3, 2] = bad
    with pytest.raises(RuntimeError, match="non-finite"):
        nbs(K, L)
    with pytest.raises(RuntimeError, match="non-finite"):
        bures(K, L)


def test_shape_mismatch_too_few_rows_and_degenerate_raise() -> None:
    with pytest.raises(ValueError, match="same sites"):
        nbs(linear_gram(_cloud(51, 10, 5)), linear_gram(_cloud(52, 11, 5)))
    with pytest.raises(ValueError, match="rows"):
        nbs(np.eye(MIN_ROWS - 1), np.eye(MIN_ROWS - 1))
    with pytest.raises(ValueError, match="square"):
        nbs(np.zeros((4, 3)), np.zeros((4, 3)))
    with pytest.raises(RuntimeError, match="zero-trace"):
        nbs(np.ones((8, 8)), linear_gram(_cloud(53, 8, 4)))                 # a constant cloud has no shape
    with pytest.raises(ValueError, match="statistic"):
        gram_null(np.eye(5), np.eye(5), statistic="bogus", n_perm=2, rng=rng_for("x"))
    with pytest.raises(ValueError, match="n_perm"):
        gram_null(np.eye(5), np.eye(5), statistic="nbs", n_perm=0, rng=rng_for("x"))
