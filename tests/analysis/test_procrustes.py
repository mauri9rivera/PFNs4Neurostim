"""Tests of the Procrustes complement to the CKA analysis (:mod:`pfns4neurostim.analysis.procrustes`).

The load-bearing ones are the invariance tests: the statistic must be blind to a global rotation and to isotropic
rescaling (that is what it quotients out) and must *not* be blind to an anisotropic stretch (that is the whole
reason it sits next to CKA, which is invariant to both). The last of those is asserted against debiased CKA on
the same pair, so the test fails if the two statistics stop being complementary.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import procrustes as scipy_procrustes

from pfns4neurostim.analysis.cka import cka_debiased, linear_gram
from pfns4neurostim.analysis.procrustes import (
    HEADLINE_STATISTIC,
    MIN_ROWS,
    ProcrustesNull,
    procrustes_disparity,
    procrustes_with_null,
    standardize_cloud,
)
from pfns4neurostim.seeding import rng_for

#: The NHP operating regime the analysis runs in: 96 sites, 192-dimensional tokens (n < d).
N_SITES, N_DIM = 96, 192

#: Replicates used in the tests. Small on purpose -- these assert behaviour, not a measurement.
N_PERM = 64


def _cloud(seed: int, n: int = N_SITES, d: int = N_DIM) -> np.ndarray:
    """A random point cloud, shape [n, d]."""
    return rng_for("test-procrustes", seed).normal(size=(n, d))


def _random_rotation(d: int, seed: int) -> np.ndarray:
    """A random orthogonal matrix, shape [d, d]."""
    return np.linalg.qr(rng_for("test-procrustes-rot", seed).normal(size=(d, d)))[0]


# --- the statistic itself -----------------------------------------------------------------------------------

def test_matches_scipy_procrustes() -> None:
    """The reduced form is the same quantity as ``scipy.spatial.procrustes``'s disparity."""
    X = _cloud(1)
    Y = X @ _random_rotation(N_DIM, 2) * 2.3 + _cloud(3) * 0.4
    assert procrustes_disparity(X, Y) == pytest.approx(scipy_procrustes(X, Y)[2], abs=1e-10)


@pytest.mark.parametrize("n,d", [(40, 8), (20, 20), (N_SITES, N_DIM)])
def test_matches_scipy_in_every_regime(n: int, d: int) -> None:
    """n > d, n == d and the n < d regime of the NHP grid all agree with scipy."""
    X, Y = _cloud(10, n, d), _cloud(11, n, d)
    assert procrustes_disparity(X, Y) == pytest.approx(scipy_procrustes(X, Y)[2], abs=1e-10)


def test_self_disparity_is_zero() -> None:
    """A cloud against itself has zero disparity, to machine precision."""
    X = _cloud(4)
    assert procrustes_disparity(X, X) == pytest.approx(0.0, abs=1e-12)


def test_invariant_to_global_rotation_and_isotropic_scale() -> None:
    """A global orthogonal map and an isotropic rescaling of one cloud are quotiented out."""
    X = _cloud(5)
    for scale in (1.0, 7.5, 0.02):
        assert procrustes_disparity(X, X @ _random_rotation(N_DIM, 6) * scale) == pytest.approx(0.0, abs=1e-12)


def test_invariant_to_translation() -> None:
    """A translation of one cloud is quotiented out (both clouds are centred)."""
    X = _cloud(7)
    shift = rng_for("test-procrustes-shift").normal(size=(1, N_DIM)) * 5.0
    assert procrustes_disparity(X, X + shift) == pytest.approx(0.0, abs=1e-12)


def test_sensitive_to_anisotropic_stretch() -> None:
    """Required property: an anisotropic stretch moves the disparity, an isotropic one does not.

    This is what separates Procrustes from a pure rotation-magnitude readout. The isotropic arm is the control:
    the *same* orthogonal map with the mean gain leaves the disparity at 0.
    """
    X = _cloud(8)
    gains = np.linspace(1.0, 6.0, N_DIM)                                   # [d] per-direction gain
    stretched = X @ (_random_rotation(N_DIM, 9) * gains[None, :])          # [n, d]
    anisotropic = procrustes_disparity(X, stretched)
    isotropic = procrustes_disparity(X, X @ _random_rotation(N_DIM, 9) * float(gains.mean()))
    assert isotropic == pytest.approx(0.0, abs=1e-12)
    assert anisotropic > 0.05
    assert anisotropic - isotropic > 0.05


def test_spectrum_reweighting_that_cka_is_nearly_blind_to() -> None:
    """The complementarity, stated correctly: Procrustes and CKA weight the spectrum differently.

    Linear CKA and Procrustes have the *same* invariance group (translation, isotropic scale, orthogonal maps),
    so "CKA cannot see anisotropy" is false -- a general anisotropic map changes the Gram matrix and CKA does
    register it. The real difference is the norm each puts on the cross-covariance spectrum: CKA is a Frobenius
    (sum of squares) quantity, so it is dominated by the leading directions, while the Procrustes residual is
    ``1 - (sum of singular values) ** 2``, a nuclear-norm (sum) quantity that weights every direction's alignment
    equally.

    The construction below makes that measurable: a cloud with one dominant direction, whose 95 weak directions
    are then all inflated. The dominant direction is untouched, so CKA's Frobenius-dominant term barely moves;
    the disparity moves a lot. Changes the other way round exist too -- the point is that the two readouts are
    not orderings of one another, not that one dominates.
    """
    rng = rng_for("spectrum")
    n, d = N_SITES, N_DIM
    U = np.linalg.qr(rng.normal(size=(n, n)))[0]                           # [n, n]
    V = np.linalg.qr(rng.normal(size=(d, d)))[0][:, :n]                    # [d, n]
    s = np.concatenate([[8.0], np.ones(n - 1)])                            # [n] one dominant direction
    X = (U * s[None, :]) @ V.T                                             # [n, d]
    s_tail = s.copy()
    s_tail[1:] *= 3.0                                                      # inflate the 95 weak directions only
    Y = (U * s_tail[None, :]) @ V.T                                        # [n, d]
    disparity = procrustes_disparity(X, Y)
    cka_drop = 1.0 - cka_debiased(linear_gram(X), linear_gram(Y))
    assert disparity > 0.1
    assert cka_drop < 0.01
    assert disparity > 10.0 * max(cka_drop, 1e-3)


# --- the shuffled-site null (Caveat 2) ----------------------------------------------------------------------

def test_null_is_near_its_expected_level_on_independent_clouds() -> None:
    """On two independent clouds the measured disparity sits inside the null, as it must.

    The null is what unrelated 96-row clouds in 192 dimensions reach; for a pair that *is* unrelated the
    observation should be indistinguishable from it, so the headline difference is ~0 and the percentile is not
    in either extreme tail.
    """
    res = procrustes_with_null(_cloud(20), _cloud(21), n_perm=256, rng=rng_for("null-indep"))
    assert res.disparity == pytest.approx(res.null_mean, abs=4.0 * res.null_sd)
    assert abs(getattr(res, HEADLINE_STATISTIC)) < 4.0 * res.null_sd
    assert 0.01 < res.null_percentile < 0.99


def test_null_shows_the_low_bias_of_the_regime() -> None:
    """Caveat 2 made measurable: the null of two unrelated 96x192 clouds is far below 1."""
    res = procrustes_with_null(_cloud(22), _cloud(23), n_perm=N_PERM, rng=rng_for("null-bias"))
    assert 0.05 < res.null_mean < 0.6
    assert res.n_rank == N_SITES - 1 < res.n_dim


def test_fewer_rows_raise_the_null() -> None:
    """Shrinking n in the same dimension aligns unrelated clouds better, so the null rises."""
    wide = procrustes_with_null(_cloud(24), _cloud(25), n_perm=N_PERM, rng=rng_for("null-96")).null_mean
    narrow = procrustes_with_null(_cloud(24)[:48], _cloud(25)[:48], n_perm=N_PERM,
                                  rng=rng_for("null-48")).null_mean
    assert narrow < wide


def test_matched_signal_is_separated_from_the_null() -> None:
    """A genuinely row-matched pair scores far below its shuffled-site null (the headline is negative)."""
    X = _cloud(26)
    Y = X @ _random_rotation(N_DIM, 27) + _cloud(28) * 0.3
    res = procrustes_with_null(X, Y, n_perm=N_PERM, rng=rng_for("null-signal"))
    assert res.disparity < res.null_q05
    assert getattr(res, HEADLINE_STATISTIC) < -0.05
    assert res.null_percentile == pytest.approx(0.0)


def test_null_is_reproducible_from_the_seeding_helpers() -> None:
    """Two runs with the same keyed generator give identical null draws (no hidden global state)."""
    X, Y = _cloud(29), _cloud(30)
    a = procrustes_with_null(X, Y, n_perm=N_PERM, rng=rng_for("repro", 1))
    b = procrustes_with_null(X, Y, n_perm=N_PERM, rng=rng_for("repro", 1))
    assert a == b
    assert isinstance(a, ProcrustesNull)


def test_result_row_is_flat_and_jsonable() -> None:
    """``as_row`` gives the plain floats/ints a tidy CSV row needs, including the headline field."""
    row = procrustes_with_null(_cloud(31), _cloud(32), n_perm=4, rng=rng_for("row")).as_row()
    assert HEADLINE_STATISTIC in row
    assert all(isinstance(v, (int, float)) for v in row.values())
    assert row["n_perm"] == 4 and row["n_rows"] == N_SITES and row["n_dim"] == N_DIM


# --- fail-fast ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_raises(bad: float) -> None:
    """NaN/Inf raise instead of propagating as a nan disparity (CLAUDE.md fail-fast rule)."""
    X, Y = _cloud(40, 12, 6), _cloud(41, 12, 6)
    Y = Y.copy()
    Y[3, 2] = bad
    with pytest.raises(RuntimeError, match="non-finite"):
        procrustes_disparity(X, Y)
    with pytest.raises(RuntimeError, match="non-finite"):
        procrustes_with_null(X, Y, n_perm=2, rng=rng_for("nan"))


def test_shape_mismatch_raises() -> None:
    """Procrustes needs a row correspondence, so mismatched shapes are an error."""
    with pytest.raises(ValueError, match="matching shapes"):
        procrustes_disparity(_cloud(42, 10, 5), _cloud(43, 11, 5))
    with pytest.raises(ValueError, match="matching shapes"):
        procrustes_with_null(_cloud(42, 10, 5), _cloud(43, 11, 5), n_perm=2, rng=rng_for("shape"))


def test_degenerate_cloud_raises() -> None:
    """A cloud with no spread has no shape to compare; it raises rather than dividing by zero."""
    flat = np.ones((8, 4))
    with pytest.raises(RuntimeError, match="zero Frobenius norm"):
        procrustes_disparity(flat, _cloud(44, 8, 4))


def test_too_few_rows_raises() -> None:
    """Below ``MIN_ROWS`` the best rigid alignment is exact for any pair, so the statistic is refused."""
    with pytest.raises(ValueError, match="rows"):
        procrustes_disparity(_cloud(45, MIN_ROWS - 1, 4), _cloud(46, MIN_ROWS - 1, 4))


def test_non_2d_and_bad_n_perm_raise() -> None:
    """A 1-D or 3-D cloud and a non-positive replicate count are configuration errors."""
    with pytest.raises(ValueError, match="2-D"):
        standardize_cloud(np.zeros((4, 3, 2)))
    with pytest.raises(ValueError, match="n_perm"):
        procrustes_with_null(_cloud(47, 10, 4), _cloud(48, 10, 4), n_perm=0, rng=rng_for("perm"))


def test_standardize_cloud_is_unit_norm_and_centred() -> None:
    """The standardization matches what ``scipy.spatial.procrustes`` applies."""
    Z = standardize_cloud(_cloud(49, 30, 7))
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    assert float(np.linalg.norm(Z)) == pytest.approx(1.0, abs=1e-12)
