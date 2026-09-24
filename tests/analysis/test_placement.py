"""M0 validation of the placement estimators (task #6 Step 5)."""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.analysis import placement as P
from pfns4neurostim.data.references.grid import noise_grid_bank, standardize_map


def _grid(side: int = 8) -> np.ndarray:
    ax = np.linspace(0, 1, side)
    return np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)


def _smooth_map(X: np.ndarray, rng: np.random.Generator, ell: float = 0.3) -> np.ndarray:
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    K = np.exp(-0.5 * d2 / ell ** 2) + 1e-8 * np.eye(len(X))
    return standardize_map(np.linalg.cholesky(K) @ rng.normal(size=len(X)))


class TestMMD:
    def test_same_distribution_is_near_zero(self, rng: np.random.Generator) -> None:
        a, b = rng.normal(size=(300, 2)), rng.normal(size=(300, 2))
        assert abs(P.mmd2_unbiased(a, b, 1.0)) < 0.01

    @pytest.mark.parametrize("kind", ["mean", "scale"])
    def test_monotone_in_known_shift(self, rng: np.random.Generator, kind: str) -> None:
        base = rng.normal(size=(300, 2))
        other = rng.normal(size=(300, 2))
        shifts = [0.0, 0.5, 1.0, 2.0]
        vals = [
            P.mmd2_unbiased(base, other + s if kind == "mean" else other * (1 + s), 1.0) for s in shifts
        ]
        assert all(np.diff(vals) > 0)

    def test_median_bandwidth_positive(self, rng: np.random.Generator) -> None:
        assert P.median_bandwidth([rng.normal(size=(50, 3)) for _ in range(4)]) > 0


class TestW2:
    def test_self_distance_shrinks_with_n(self, rng: np.random.Generator) -> None:
        proj = P.projection_set(3, 200)
        vals = [np.mean([P.sliced_w2(rng.normal(size=(n, 3)), rng.normal(size=(n, 3)), proj)
                         for _ in range(5)]) for n in (20, 200, 2000)]
        assert vals[0] > vals[1] > vals[2]
        assert vals[2] < 0.1

    def test_pure_shift_analytic(self, rng: np.random.Generator) -> None:
        proj = P.projection_set(3, 500, seed=1)
        Z = rng.normal(size=(100, 3))
        delta = 0.7
        shifted = Z + np.array([0.0, 0.0, delta])
        expected = delta * np.sqrt(np.mean(proj[:, -1] ** 2))
        assert P.sliced_w2(Z, shifted, proj) == pytest.approx(expected, rel=1e-10)

    def test_unequal_n_uses_equal_subsamples(self, rng: np.random.Generator) -> None:
        proj = P.projection_set(2, 100)
        v = P.sliced_w2(rng.normal(size=(40, 2)), rng.normal(size=(400, 2)), proj,
                        rng=np.random.default_rng(0))
        assert np.isfinite(v) and v > 0

    @pytest.mark.parametrize("n_proj", [50, 200, 1000])
    def test_projection_count_sensitivity_is_small(self, rng: np.random.Generator, n_proj: int) -> None:
        Z1, Z2 = rng.normal(size=(200, 3)), rng.normal(size=(200, 3)) + 0.5
        ref = P.sliced_w2(Z1, Z2, P.projection_set(3, 5000))
        assert P.sliced_w2(Z1, Z2, P.projection_set(3, n_proj)) == pytest.approx(ref, rel=0.15)


class TestPlacement:
    def test_placement_scale(self) -> None:
        assert P.placement(0.5, 0.0, 1.0) == 0.5
        assert P.placement(1.0, 1.0, 1.0) is None

    def test_bootstrap_ci_brackets_point(self) -> None:
        p, lo, hi = P.bootstrap_placement(0.5, np.array([0.0, 0.1, 0.05]), np.array([1.0, 0.9, 1.1]), n_boot=200)
        assert lo <= p <= hi

    @staticmethod
    def _ladder(metric: str, mode: str) -> float:
        rng = np.random.default_rng(0)
        X = _grid()
        feat = lambda X_, y_: P.map_features(X_, y_, mode)  # noqa: E731
        bank = [feat(X, _smooth_map(X, rng)) for _ in range(20)]
        y_map = _smooth_map(X, rng)
        y_noise = noise_grid_bank(X, 1, seed=5).maps[0]
        if metric == "mmd":
            m = P.make_metric("mmd", bandwidth=P.median_bandwidth(bank))
        else:
            m = P.make_metric("w2", projections=P.projection_set(bank[0].shape[1], 200))
        return P.known_shift_ladder(X, y_map, y_noise, bank, m, k=5, featurize=feat)["spearman"]

    @pytest.mark.parametrize("mode", ["pairs", "rank_pairs"])
    @pytest.mark.parametrize("metric", ["mmd", "w2"])
    def test_known_shift_ladder_is_monotone(self, metric: str, mode: str) -> None:
        """Pair features: blending a smooth map towards noise moves it away from a smooth bank."""
        assert self._ladder(metric, mode) > 0.9

    def test_normal_scores_equalize_marginals(self) -> None:
        a = P.normal_scores(np.exp(np.linspace(0, 5, 50)))     # heavily skewed
        b = P.normal_scores(np.linspace(-1, 1, 50))
        np.testing.assert_allclose(np.sort(a), np.sort(b))

    def test_joint_points_fail_the_ladder(self) -> None:
        """Why 'pairs' is the default: joint (x, y) points move *towards* the bank."""
        assert self._ladder("mmd", "joint") < 0.0

    def test_pair_features_shape(self) -> None:
        X = _grid(4)
        Z = P.map_features(X, np.arange(16.0), n_neighbors=4)
        assert Z.shape == (64, 3)


@pytest.mark.slow
@pytest.mark.skipif(not __import__("os").path.exists("data/monkeys/Cebus2_M1_200123.mat"), reason="raw NHP data absent")
def test_real_channel_ladder_rank_pairs_passes_raw_pairs_fails() -> None:
    """M0 on a real NHP channel with the real prior bag: the reason rank_pairs is the default."""
    from pfns4neurostim.data.channels import load_channel
    from pfns4neurostim.data.references.grid import prior_grid_bank

    ch = load_channel("nhp", 1, 0)
    bank = prior_grid_bank(ch.X_pool, 20, seed=0)
    y_noise = noise_grid_bank(ch.X_pool, 1, seed=1).maps[0]
    out = {}
    for mode in ("rank_pairs", "pairs"):
        feat = lambda X_, y_: P.map_features(X_, y_, mode)  # noqa: E731
        Zb = [feat(ch.X_pool, m) for m in bank.maps]
        m = P.make_metric("mmd", bandwidth=P.median_bandwidth(Zb))
        out[mode] = P.known_shift_ladder(ch.X_pool, standardize_map(ch.y_gt), y_noise, Zb, m, 10, featurize=feat)["spearman"]
    assert out["rank_pairs"] > 0.9 and out["pairs"] < 0.9, out
