"""Placement of neurostim maps between the prior and noise (task #6, P0.6; Hyp C arm ii).

A *map* (x MinMax-scaled, y standardized per map) is represented as a point set and two
maps are compared as distributions of points. Two representations (:func:`map_features`):

* ``'rank_pairs'`` (**default**, see below) and ``'pairs'``: one point ``(y_i, y_j, d_ij)`` per ordered neighbour pair (each
  site with its ``n_neighbors`` nearest sites) — the distribution of local response pairs,
  i.e. the map's spatial dependence, invariant to *where* on the grid its hotspot sits;
* ``'joint'``: one point ``(x_i, y_i)`` per site — the literal reading of "the map as a
  sample". It **fails the M0 known-shift ladder** (blending a smooth map with noise brings
  it *closer* to a bank of smooth prior maps, Spearman rho = -0.9 on an 8x8 GP test grid):
  a noise map's joint distribution is the product distribution, which is what a bank of
  random prior maps averages to, whereas a specific smooth map is close only to priors with
  the same layout. It is kept for the appendix and for the record, not for placement.
* ``'rank_pairs'``: as ``'pairs'`` on normal scores of y (:func:`normal_scores`), so every
  map has the same marginal and only spatial dependence is compared. Motivated by the NHP
  smoke run (2026-09-23): with raw pairs, real channels sat *beyond* the noise ceiling
  (p ~ 2-3), their coordinate-shuffled self was as far from the prior as they were, and the
  known-shift ladder failed (rho = -0.9) — the distance was the peaky response *marginal*,
  not the spatial structure. Both views are reported; the M0 ladder decides validity.

Two maps are compared with

* **MMD²** — unbiased U-statistic with an RBF kernel and **one fixed bandwidth per
  analysis** (the median of prior self-distances, computed once and written to the
  resolved config; audit F10);
* **sliced W₂** — a fixed set of random unit projections, **equal-n repeated
  subsampling** (no quantile interpolation between unequal samples; audit F9/F10).

Placement ``p = (d - floor) / (ceiling - floor)`` with two floors and two ceilings reported
side by side (2026-09-17): Floor 1 = held-out prior vs the prior bank; Floor 2 = the two
split-half maps of the same channel; Ceiling 1 = noise vs the prior bank; Ceiling 2 = the
same channel with its site-response binding shuffled, vs the prior bank.

Formulations: **B** (per channel: distance to each prior map, median over the *k* nearest),
**C** (random size-*t* noisy contexts along a stress knob, at matched sites), **A**
(pooled marginal, appendix).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import norm, rankdata, spearmanr

__all__ = [
    "map_points",
    "map_features",
    "neighbor_pairs",
    "normal_scores",
    "median_bandwidth",
    "mmd2_unbiased",
    "sliced_w2",
    "projection_set",
    "Metric",
    "PreparedBank",
    "make_metric",
    "prepare_bank",
    "distance_to_bank",
    "placement",
    "bootstrap_placement",
    "known_shift_ladder",
]


def map_points(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stack coordinates and responses into joint points ``z = (x, y)``.

    Args:
        X: Coordinates in [0, 1]^D, shape [N, D].
        y: Standardized responses, shape [N].

    Returns:
        Joint points, shape [N, D + 1].
    """
    return np.column_stack([np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)])


def neighbor_pairs(X: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordered (site, neighbour) index pairs and their distances.

    Args:
        X: Coordinates, shape [N, D].
        n_neighbors: Nearest neighbours per site (ties broken by index).

    Returns:
        ``(i, j, d)`` each of shape [N * n_neighbors].
    """
    d = np.sqrt(_sqdist(X, X))                                   # [N, N]
    np.fill_diagonal(d, np.inf)
    j = np.argsort(d, axis=1, kind="stable")[:, :n_neighbors]    # [N, k]
    i = np.repeat(np.arange(len(X)), n_neighbors)
    j = j.reshape(-1)
    return i, j, d[i, j]


def map_features(
    X: np.ndarray,
    y: np.ndarray,
    mode: str = "rank_pairs",
    n_neighbors: int = 4,
) -> np.ndarray:
    """Point-set representation of a map (see the module docstring).

    Args:
        X: Coordinates in [0, 1]^D, shape [N, D].
        y: Standardized responses, shape [N].
        mode: ``'rank_pairs'``, ``'pairs'`` or ``'joint'``.
        n_neighbors: Neighbours per site for ``'pairs'``.

    Returns:
        ``'pairs'``: shape [N * n_neighbors, 3]; ``'joint'``: shape [N, D + 1].
    """
    if mode == "joint":
        return map_points(X, y)
    if mode not in ("pairs", "rank_pairs"):
        raise ValueError(f"map_features: unknown mode {mode!r}.")
    i, j, d = neighbor_pairs(np.asarray(X, dtype=np.float64), n_neighbors)
    y = np.asarray(y, dtype=np.float64)
    if mode == "rank_pairs":
        y = normal_scores(y)
    return np.column_stack([y[i], y[j], d])


def normal_scores(y: np.ndarray) -> np.ndarray:
    """Rank-based normal scores Phi^-1((rank - 0.5) / N) (average ranks for ties).

    Every map then has the same (Gaussian) marginal, so a distance between two maps'
    pair distributions measures spatial dependence only (a Gaussian copula view).

    Args:
        y: Responses, shape [N].

    Returns:
        Normal scores, shape [N].
    """
    r = rankdata(y)
    return norm.ppf((r - 0.5) / len(y))


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, shape [len(A), len(B)]."""
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(sq, 0.0)


def median_bandwidth(Z_list: Sequence[np.ndarray], max_points: int = 2000, seed: int = 0) -> float:
    """Median pairwise distance of points pooled from reference maps (prior self-distances).

    Args:
        Z_list: Joint point sets of the prior maps.
        max_points: Subsample cap for the pooled set.
        seed: Subsampling seed.

    Returns:
        The bandwidth (> 0).

    Raises:
        RuntimeError: If the median distance is 0.
    """
    pool = np.vstack(Z_list)
    rng = np.random.default_rng(seed)
    if len(pool) > max_points:
        pool = pool[rng.choice(len(pool), max_points, replace=False)]
    d = np.sqrt(_sqdist(pool, pool)[np.triu_indices(len(pool), 1)])
    bw = float(np.median(d))
    if bw <= 0:
        raise RuntimeError("median_bandwidth: zero median distance.")
    return bw


def _mmd_self_term(Z: np.ndarray, g: float) -> float:
    """Within-sample term of the unbiased MMD²: mean off-diagonal RBF kernel value."""
    K = np.exp(-g * _sqdist(Z, Z))
    n = len(Z)
    return (K.sum() - np.trace(K)) / (n * (n - 1))


def mmd2_unbiased(Z1: np.ndarray, Z2: np.ndarray, bandwidth: float) -> float:
    """Unbiased MMD² U-statistic with an RBF kernel (Gretton et al. 2012, Eq. 3).

    May be slightly negative when the two distributions are equal.

    Args:
        Z1: Sample 1, shape [n, F].
        Z2: Sample 2, shape [m, F].
        bandwidth: RBF bandwidth ``sigma`` in ``exp(-||a - b||^2 / (2 sigma^2))``.

    Returns:
        The estimate.
    """
    g = 1.0 / (2.0 * bandwidth ** 2)
    return _mmd_from_terms(_mmd_self_term(Z1, g), _mmd_self_term(Z2, g), Z1, Z2, g)


def _mmd_from_terms(self1: float, self2: float, Z1: np.ndarray, Z2: np.ndarray, g: float) -> float:
    """Unbiased MMD² from precomputed within-sample terms (only the cross term is evaluated)."""
    Kxy = np.exp(-g * _sqdist(Z1, Z2))                              # [n, m]
    return float(self1 + self2 - 2.0 * Kxy.mean())


def projection_set(n_features: int, n_projections: int, seed: int = 0) -> np.ndarray:
    """A fixed set of random unit directions, shared by every comparison of an analysis.

    Args:
        n_features: Point dimensionality.
        n_projections: Number of directions.
        seed: Seed.

    Returns:
        Unit vectors, shape [n_projections, n_features].
    """
    P = np.random.default_rng(seed).normal(size=(n_projections, n_features))
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def sliced_w2(
    Z1: np.ndarray,
    Z2: np.ndarray,
    projections: np.ndarray,
    *,
    n_repeats: int = 20,
    rng: np.random.Generator | None = None,
) -> float:
    """Sliced Wasserstein-2 with equal-n repeated subsampling.

    Each repeat subsamples ``m = min(n1, n2)`` points from both samples without
    replacement, so the 1D optimal couplings are exact sorted matchings; the squared
    distance is averaged over projections, square-rooted, and averaged over repeats.

    Args:
        Z1: Sample 1, shape [n1, F].
        Z2: Sample 2, shape [n2, F].
        projections: Unit directions, shape [P, F].
        n_repeats: Subsampling repeats (1 suffices when ``n1 == n2``).
        rng: Generator for the subsamples.

    Returns:
        The sliced W₂ distance.
    """
    rng = rng or np.random.default_rng(0)
    m = min(len(Z1), len(Z2))
    reps = 1 if len(Z1) == len(Z2) else n_repeats
    vals = []
    for _ in range(reps):
        a = Z1 if len(Z1) == m else Z1[rng.choice(len(Z1), m, replace=False)]
        b = Z2 if len(Z2) == m else Z2[rng.choice(len(Z2), m, replace=False)]
        vals.append(_w2_sorted(_sorted_projections(a, projections), _sorted_projections(b, projections)))
    return float(np.mean(vals))


def _sorted_projections(Z: np.ndarray, projections: np.ndarray) -> np.ndarray:
    """Per-direction sorted projections of a point set, shape [n, P]."""
    return np.sort(Z @ projections.T, axis=0)


def _w2_sorted(pa: np.ndarray, pb: np.ndarray) -> float:
    """Sliced W₂ between two equal-size samples from their sorted projections, each [n, P]."""
    return float(np.sqrt(np.mean((pa - pb) ** 2)))


@dataclass(frozen=True)
class Metric:
    """A named map-distance ``(Z1, Z2) -> float`` with its fixed parameters.

    ``prepare`` / ``between`` are an optional exact fast path for many comparisons against
    the same maps: ``prepare`` computes a map's metric-specific summary once (sorted
    projections for W₂, the within-sample kernel term for MMD²) and ``between`` evaluates
    the distance from two summaries with the same arithmetic as ``fn``.
    """

    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    params: dict[str, Any]
    prepare: Callable[[np.ndarray], Any] | None = None
    between: Callable[[Any, Any], float] | None = None

    def __call__(self, Z1: np.ndarray, Z2: np.ndarray) -> float:
        return self.fn(Z1, Z2)


@dataclass(frozen=True)
class PreparedBank:
    """Reference maps with their ``metric.prepare`` summaries, computed once.

    Attributes:
        metric: Name of the metric the summaries belong to.
        summaries: One summary per reference map.
    """

    metric: str
    summaries: list[Any]


def prepare_bank(bank_Z: Sequence[np.ndarray], metric: Metric) -> PreparedBank:
    """Summarize reference maps once so repeated :func:`distance_to_bank` calls reuse them.

    Args:
        bank_Z: Reference maps' point sets.
        metric: Distance; must define ``prepare``/``between``.

    Returns:
        The prepared bank.

    Raises:
        ValueError: If the metric has no fast path.
    """
    if metric.prepare is None or metric.between is None:
        raise ValueError(f"prepare_bank: metric {metric.name!r} has no prepare/between fast path.")
    return PreparedBank(metric.name, [metric.prepare(R) for R in bank_Z])


def make_metric(name: str, **params: Any) -> Metric:
    """Build ``'mmd'`` (needs ``bandwidth``) or ``'w2'`` (needs ``projections``).

    Args:
        name: Metric name.
        **params: Fixed parameters.

    Returns:
        The metric.
    """
    if name == "mmd":
        bw = float(params["bandwidth"])
        g = 1.0 / (2.0 * bw ** 2)
        return Metric(
            "mmd",
            lambda a, b: mmd2_unbiased(a, b, bw),
            {"bandwidth": bw},
            prepare=lambda Z: (Z, _mmd_self_term(Z, g)),
            between=lambda a, b: _mmd_from_terms(a[1], b[1], a[0], b[0], g),
        )
    if name == "w2":
        proj = np.asarray(params["projections"])
        reps = int(params.get("n_repeats", 20))
        seed = int(params.get("seed", 0))

        def w2(a: np.ndarray, b: np.ndarray) -> float:
            return sliced_w2(a, b, proj, n_repeats=reps, rng=np.random.default_rng(seed))

        def w2_between(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
            # Unequal sizes need the random equal-n subsampling: fall back to the full computation.
            return _w2_sorted(a[1], b[1]) if len(a[0]) == len(b[0]) else w2(a[0], b[0])

        return Metric(
            "w2", w2, {"n_projections": int(len(proj)), "n_repeats": reps},
            prepare=lambda Z: (Z, _sorted_projections(Z, proj)),
            between=w2_between,
        )
    raise ValueError(f"Unknown placement metric {name!r}; expected 'mmd' or 'w2'.")


def distance_to_bank(
    Z: np.ndarray,
    bank_Z: Sequence[np.ndarray] | PreparedBank,
    metric: Metric,
    k: int,
) -> float:
    """Formulation B: median distance to the ``k`` nearest reference maps.

    Args:
        Z: The map's joint points, shape [N, F].
        bank_Z: Reference maps' joint points, or a :func:`prepare_bank` of them (same result,
            reference summaries reused across calls).
        metric: Distance.
        k: Pre-declared number of nearest references.

    Returns:
        Median over the ``k`` smallest distances.

    Raises:
        ValueError: If a prepared bank belongs to another metric.
    """
    if isinstance(bank_Z, PreparedBank):
        if bank_Z.metric != metric.name:
            raise ValueError(f"distance_to_bank: bank prepared for {bank_Z.metric!r}, metric is {metric.name!r}.")
        s = metric.prepare(Z)
        d = np.sort([metric.between(s, r) for r in bank_Z.summaries])
    else:
        d = np.sort([metric(Z, R) for R in bank_Z])
    return float(np.median(d[: max(1, min(k, len(d)))]))


def placement(d: float, floor: float, ceiling: float) -> float | None:
    """``p = (d - floor) / (ceiling - floor)``; ``None`` when the pair has no dynamic range.

    Args:
        d: Channel distance.
        floor: Floor distance.
        ceiling: Ceiling distance.

    Returns:
        The placement (0 = floor, 1 = ceiling; may leave [0, 1]).
    """
    gap = ceiling - floor
    return None if gap <= 0 else float((d - floor) / gap)


def bootstrap_placement(
    d: float,
    floors: np.ndarray,
    ceilings: np.ndarray,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float | None, float | None, float | None]:
    """Placement with a bootstrap CI over the floor and ceiling samples.

    Args:
        d: Channel distance.
        floors: Floor distances (e.g. one per held-out prior map), shape [nf].
        ceilings: Ceiling distances, shape [nc].
        n_boot: Replicates.
        ci: Interval mass.
        seed: Seed.

    Returns:
        ``(p, ci_low, ci_high)`` using the medians of the resampled floors/ceilings.
    """
    p = placement(d, float(np.median(floors)), float(np.median(ceilings)))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        f = float(np.median(rng.choice(floors, len(floors))))
        c = float(np.median(rng.choice(ceilings, len(ceilings))))
        v = placement(d, f, c)
        if v is not None:
            boots.append(v)
    if not boots:
        return p, None, None
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return p, float(lo), float(hi)


def known_shift_ladder(
    X: np.ndarray,
    y_map: np.ndarray,
    y_noise: np.ndarray,
    bank_Z: Sequence[np.ndarray],
    metric: Metric,
    k: int,
    lambdas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    featurize: Callable[[np.ndarray, np.ndarray], np.ndarray] = map_features,
) -> dict[str, Any]:
    """M0 known-shift ladder: blend a map with noise; the distance must rise with lambda.

    ``y_lambda = standardize((1 - lambda) y_map + lambda y_noise)``.

    Args:
        X: Grid coordinates, shape [N, D].
        y_map: Standardized map, shape [N].
        y_noise: Standardized noise map, shape [N].
        bank_Z: Prior bank points (same featurization).
        metric: Distance.
        k: Nearest references.
        lambdas: Blend ladder.
        featurize: ``(X, y) -> points``; must match how ``bank_Z`` was built.

    Returns:
        ``{'lambdas', 'distances', 'spearman'}``.
    """
    from ..data.references.grid import standardize_map  # noqa: PLC0415

    dists = []
    for lam in lambdas:
        y = standardize_map((1.0 - lam) * y_map + lam * y_noise)
        dists.append(distance_to_bank(featurize(X, y), bank_Z, metric, k))
    rho = float(spearmanr(lambdas, dists).correlation)
    return {"lambdas": list(lambdas), "distances": dists, "spearman": rho}
