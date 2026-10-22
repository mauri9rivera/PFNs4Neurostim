"""How Bayesian is TabPFN's in-context update? (Hyp C arm (i), OBJECTIVE OMEGA, 2026-10-15).

Replaces the M10 update-rule alignment. After one observation ``(x', y')`` is added to a fixed context D, the
*full* posterior predictive of every engine is read at all sites, and two families of readouts are computed
(``.research_assistant/omega_sprint4_bayesian_update.md`` §3):

**Part A -- coherence**, valid for *any* Bayesian model whatever its prior. Averaged over the model's own
predictive for ``y'`` (Gauss-Hermite nodes mapped through the model's quantile function at ``x'``):

* A1 no expected drift (martingale): ``delta(x) = [sum_k w_k mu^{y'_k}(x) - mu(x)] / s(x)`` and its distributional
  form ``W1(sum_k w_k F^{y'_k}, F) / s`` (Fong, Holmes & Walker 2023; Falck, Wang & Holmes 2024);
* A2 law of total variance: expected shrink ``U = s^2 - sum_k w_k s_k^2`` against mean movement
  ``M = sum_k w_k (mu_k - mu_bar)^2``; ``U / M = 1`` for Bayes;
* A3 implied posterior covariance (tower property): ``C(x', x) = sum_k w_k (y'_k - y_bar)(mu_k(x) - mu_bar(x))``
  equals ``Cov(f(x'), f(x) | D)``, and at ``x = x'`` the *latent* variance, so ``s^2(x') - C(x', x')`` is the noise
  the model implies.

**Part B -- head-to-head**: TabPFN's update against the GP references (GP-MLL refit and GP-fixed; GP-Bayes, with
hyperparameters marginalised on a grid, is available but optional since 2026-10-21), on shared ``y'`` values
(held-out real trials, and a ladder in the predictive SD of one non-TabPFN *ladder reference* engine).
Every contrast is matched per site (and per held-out trial for CRPS) before it is aggregated (user, 2026-10-15).

Every engine predicts a new **noisy** observation; units are the dataset's offline z-scored y. Distributions are
handled through :class:`Predictive` (Gaussian mixtures for the GPs, the bar distribution for TabPFN) and evaluated
on per-site y grids, so W1 and CRPS are computed the same way for every engine.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from scipy.fft import next_fast_len
from scipy.special import logsumexp, ndtr, ndtri
from scipy.stats import spearmanr

from ..data.channels import ChannelData

__all__ = [
    "ANCHOR_STRATA",
    "ANCHOR_POOLS",
    "stratum_masks",
    "select_anchors_uniform",
    "site_distances",
    "Predictive",
    "MixturePredictive",
    "RankOneMixturePredictive",
    "BarPredictive",
    "UpdateEngine",
    "PFNEngine",
    "GPFixedEngine",
    "GPBayesEngine",
    "GPRefitEngine",
    "GPFrozenEngine",
    "ToyUpdater",
    "PFNRefitEngine",
    "CriterionPredictive",
    "QUADRATURE_METHODS",
    "gh_nodes",
    "quadrature_levels",
    "quadrature_values",
    "site_grid",
    "w1",
    "crps_grid",
    "linear_transfer",
    "far_field_share",
    "coherence_anchor",
    "coherence_scheme",
    "coherence",
    "coherence_cell",
    "shared_values",
    "heldout_trials",
    "headtohead_anchor",
    "transfer_profile",
    # BETA (2026-10-21), purely additive: belief embeddings, the site-matched W2 split, posterior covariances
    "belief_levels",
    "belief_embedding",
    "site_w2_split",
    "latent_covariance_of",
    "implied_posterior_covariance",
]

#: Anchor strata (overlapping labels recorded per anchor; anchors are drawn uniformly within a pool).
ANCHOR_STRATA: tuple[str, ...] = ("high", "low", "edge", "centre")
#: Anchor pools, never pooled in a summary: a new site, or a second trial at an observed site (the revisit).
ANCHOR_POOLS: tuple[str, ...] = ("unobserved", "observed")
#: Pool label of the Part A summary over every anchor of a cell (written at assembly, user 2026-10-22).
POOLED_ANCHORS: str = "all"


# ---------------------------------------------------------------------------
# Anchors and geometry
# ---------------------------------------------------------------------------
def stratum_masks(channel: ChannelData) -> dict[str, np.ndarray]:
    """Boolean membership of every site in each anchor stratum, as *overlapping* labels.

    ``high`` = top quartile of ``y_gt``; ``low`` = bottom quartile; ``edge`` = a coordinate at its min or max;
    ``centre`` = the quartile of sites closest to the grid centre.

    Args:
        channel: Source channel.

    Returns:
        ``{'high', 'low', 'edge', 'centre'} -> bool mask [n_sites]``.
    """
    n = channel.n_sites
    q = max(1, n // 4)
    order = np.argsort(channel.y_gt)
    X = channel.X_pool
    lo, hi = X.min(axis=0), X.max(axis=0)
    centre_dist = np.linalg.norm(X - (lo + hi) / 2.0, axis=1)
    masks = {name: np.zeros(n, dtype=bool) for name in ANCHOR_STRATA}
    masks["high"][order[-q:]] = True
    masks["low"][order[:q]] = True
    masks["edge"] = ((X <= lo) | (X >= hi)).any(axis=1)
    masks["centre"][np.argsort(centre_dist)[:q]] = True
    return masks


def select_anchors_uniform(
    channel: ChannelData,
    n_unobserved: int,
    n_observed: int,
    rng: np.random.Generator,
    context_sites: Sequence[int],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Anchors drawn uniformly without replacement within two pools, with stratum labels attached.

    Uniform sampling makes an unweighted pool unbiased for the grid (user decision 2026-10-12: equal allocation
    over strata is not proportional to their population shares); :func:`stratum_masks` still supports every
    conditional contrast because membership is recorded per anchor. The ``observed`` pool is the revisit case: a
    second trial at a context site. A pool smaller than requested contributes all its sites.

    Args:
        channel: Source channel.
        n_unobserved: Anchors from the sites outside the context.
        n_observed: Anchors from the context sites.
        rng: Generator (the unobserved pool is drawn first).
        context_sites: Site indices of the context, shape [t].

    Returns:
        ``(anchors [A], pool per anchor, labels)`` where a label is the ``'+'``-joined strata of the anchor or
        ``'none'``. Unobserved anchors come first, each pool sorted.
    """
    n = channel.n_sites
    observed = np.zeros(n, dtype=bool)
    observed[np.asarray(context_sites, dtype=int)] = True
    anchors: list[np.ndarray] = []
    pools: list[str] = []
    for pool_name, mask, k in (("unobserved", ~observed, n_unobserved), ("observed", observed, n_observed)):
        pool = np.flatnonzero(mask)
        take = min(int(k), int(pool.size))
        chosen = np.sort(rng.choice(pool, size=take, replace=False)) if take else np.empty(0, dtype=int)
        anchors.append(chosen.astype(int))
        pools += [pool_name] * take
    a = np.concatenate(anchors).astype(int)
    masks = stratum_masks(channel)
    labels = ["+".join(s for s in ANCHOR_STRATA if masks[s][i]) or "none" for i in a]
    return a, pools, labels


def site_distances(ch2xy: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Distance of every grid site from every anchor, in electrode pitches.

    Args:
        ch2xy: Integer grid coordinates, shape [M, 2].
        anchors: Anchor site indices, shape [A].

    Returns:
        Distances, shape [A, M].
    """
    xy = np.asarray(ch2xy, dtype=np.float64)                                       # [M, 2]
    return np.sqrt(((xy[None, :, :] - xy[np.asarray(anchors)][:, None, :]) ** 2).sum(-1))   # [A, M]


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson r, or ``None`` when either vector is constant (undefined, not NaN)."""
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _median(values: np.ndarray) -> float | None:
    """Median of a non-empty array, ``None`` for an empty one."""
    return float(np.median(values)) if np.size(values) else None


# ---------------------------------------------------------------------------
# Predictive distributions
# ---------------------------------------------------------------------------
class Predictive:
    """Posterior predictive of a new noisy observation at ``M`` sites, for ``P`` conditioning states.

    Subclasses hold the distribution; the base class fixes the interface every readout uses.

    Attributes:
        mean: Predictive mean, shape [P, M].
        var: Predictive variance, shape [P, M].
    """

    mean: np.ndarray
    var: np.ndarray

    @property
    def n_probes(self) -> int:
        """Number of conditioning states ``P``."""
        return int(self.mean.shape[0])

    @property
    def sd(self) -> np.ndarray:
        """Predictive SD, shape [P, M]."""
        return np.sqrt(self.var)

    def cdf(self, y_grid: np.ndarray) -> np.ndarray:
        """CDF on per-site grids.

        Args:
            y_grid: Evaluation points per site, shape [M, G].

        Returns:
            ``P(Y <= y)``, shape [P, M, G].
        """
        raise NotImplementedError

    def ppf(self, u: np.ndarray, sites: np.ndarray) -> np.ndarray:
        """Quantiles at a subset of sites.

        Args:
            u: Levels in (0, 1), shape [U].
            sites: Site indices, shape [S].

        Returns:
            Quantiles, shape [P, S, U].
        """
        raise NotImplementedError

    def support(self, u_tail: float) -> np.ndarray:
        """Range holding all but ``2 u_tail`` of the mass at every site.

        Args:
            u_tail: Tail probability on each side.

        Returns:
            ``[..., 0]`` lower and ``[..., 1]`` upper bound, shape [P, M, 2].
        """
        raise NotImplementedError

    def take(self, idx: Sequence[int] | np.ndarray) -> "Predictive":
        """The predictive restricted to some conditioning states.

        Args:
            idx: Probe indices.

        Returns:
            A predictive with ``P = len(idx)``.
        """
        raise NotImplementedError


class MixturePredictive(Predictive):
    """Gaussian-mixture predictive (``C = 1`` for a GP with point hyperparameters).

    Args:
        weights: Component weights, shape [P, C] (rows sum to 1).
        means: Component means, shape [P, M, C].
        variances: Component predictive variances (noise included), shape [P, M, C].
        max_elements: Largest ``P * M * G * chunk`` intermediate of the exact CDF.
        bisection_steps: Iterations of the quantile bisection.
        exact_max_components: Mixtures with more components evaluate :meth:`cdf` by binned convolution
            (:meth:`_binned_cdf`); smaller ones exactly.
        sd_bins: Log-SD bins of the binned CDF.

    Raises:
        RuntimeError: On non-finite parameters or non-positive variances.
    """

    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
        *,
        max_elements: int = 20_000_000,
        bisection_steps: int = 100,
        exact_max_components: int = 64,
        sd_bins: int = 32,
    ) -> None:
        self.exact_max_components = int(exact_max_components)
        self.sd_bins = int(sd_bins)
        self.weights = np.asarray(weights, dtype=np.float64)                       # [P, C]
        self.means = np.asarray(means, dtype=np.float64)                           # [P, M, C]
        self.variances = np.asarray(variances, dtype=np.float64)                   # [P, M, C]
        if not (np.isfinite(self.weights).all() and np.isfinite(self.means).all()
                and np.isfinite(self.variances).all()):
            raise RuntimeError("MixturePredictive: non-finite weights, means or variances.")
        if (self.variances <= 0).any():
            raise RuntimeError("MixturePredictive: non-positive component variance.")
        self.max_elements = int(max_elements)
        self.bisection_steps = int(bisection_steps)
        w = self.weights[:, None, :]                                               # [P, 1, C]
        self.mean = (w * self.means).sum(-1)                                       # [P, M]
        second = (w * (self.variances + self.means ** 2)).sum(-1)                  # [P, M]
        self.var = np.maximum(second - self.mean ** 2, np.finfo(float).tiny)       # [P, M]

    def _site_components(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        """Component means and variances at one site, shape [P, C] each."""
        return self.means[:, m], self.variances[:, m]

    def _mixture_cdf(self, y: np.ndarray, means: np.ndarray, sds: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """``sum_c w_c Phi((y - mu_c) / s_c)`` with ``y`` [..., G] against components [..., C], chunked over C."""
        out = np.zeros(y.shape, dtype=np.float64)
        per_c = max(1, int(np.prod(y.shape)))
        chunk = max(1, self.max_elements // per_c)
        for start in range(0, means.shape[-1], chunk):
            sl = slice(start, start + chunk)
            z = (y[..., None] - means[..., None, sl]) / sds[..., None, sl]       # [..., G, c]
            out += (ndtr(z) * weights[..., None, sl]).sum(-1)
        return out

    def cdf(self, y_grid: np.ndarray) -> np.ndarray:
        """See :meth:`Predictive.cdf`; exact up to :attr:`exact_max_components` components, binned above."""
        if self.weights.shape[1] > self.exact_max_components:
            return self._binned_cdf(np.asarray(y_grid, dtype=np.float64))
        y = np.broadcast_to(np.asarray(y_grid, dtype=np.float64)[None], (self.n_probes, *np.shape(y_grid)))
        w = np.broadcast_to(self.weights[:, None, :], self.means.shape)            # [P, M, C]
        return self._mixture_cdf(np.ascontiguousarray(y), self.means, np.sqrt(self.variances), w)

    def _binned_cdf(self, y_grid: np.ndarray) -> np.ndarray:
        """Mixture CDF by deposit-and-convolve, for mixtures of thousands of components (GP-Bayes).

        Every component's weight is split linearly between the two grid nodes around its mean and the two
        log-SD bins around its SD (a 2-D cloud-in-cell deposit); each SD bin is then convolved with the exact
        Gaussian CDF kernel of its SD. The error is second order in the node spacing over the SD and in the
        log-SD bin width -- the moments (``mean``, ``var``) are never approximated, only this CDF.

        Args:
            y_grid: Uniform grid per site, shape [M, G].

        Returns:
            Shape [P, M, G].
        """
        P, C = self.weights.shape
        M = self.mean.shape[1]
        G = y_grid.shape[1]
        out = np.empty((P, M, G))
        lag = np.arange(-(G - 1), G)                                               # [2G-1]
        n_fft = next_fast_len(3 * G - 2, real=True)
        p_idx = np.repeat(np.arange(P), C)                                         # [P*C]
        w = self.weights.reshape(-1)                                               # [P*C]
        for m in range(M):
            means_m, variances_m = self._site_components(m)                         # [P, C] x2
            lo, dy = y_grid[m, 0], y_grid[m, 1] - y_grid[m, 0]
            pos = np.clip((means_m.reshape(-1) - lo) / dy, 0.0, G - 1.0)            # [P*C]
            j0 = np.minimum(pos.astype(int), G - 2)
            fj = pos - j0
            logs = 0.5 * np.log(variances_m.reshape(-1))                            # [P*C]
            smin, smax = logs.min(), logs.max()
            S = 1 if smax - smin < 1e-12 else self.sd_bins
            ks = np.zeros_like(logs) if S == 1 else (logs - smin) / (smax - smin) * (S - 1)
            k0 = np.minimum(ks.astype(int), max(S - 2, 0))
            fk = ks - k0 if S > 1 else np.zeros_like(ks)
            D = np.zeros(P * S * G)
            for dk, wk in ((0, 1.0 - fk), (1, fk)):
                if S == 1 and dk == 1:
                    continue
                for dj, wj in ((0, 1.0 - fj), (1, fj)):
                    idx = (p_idx * S + k0 + dk) * G + j0 + dj
                    D += np.bincount(idx, weights=w * wk * wj, minlength=P * S * G)
            sig = np.exp(smin + (np.arange(S) * (smax - smin) / max(S - 1, 1)))    # [S]
            kern = ndtr(lag[None, :] * dy / sig[:, None])                          # [S, 2G-1], kernel at lag g - j
            # sum of the S convolutions, formed in the frequency domain (one inverse transform per probe)
            spec = np.einsum("psf,sf->pf", np.fft.rfft(D.reshape(P, S, G), n_fft), np.fft.rfft(kern, n_fft))
            full = np.fft.irfft(spec, n_fft)                                       # [P, n_fft]
            out[:, m] = np.clip(full[:, G - 1:2 * G - 1], 0.0, 1.0)               # [P, G]
        return out

    def ppf(self, u: np.ndarray, sites: np.ndarray) -> np.ndarray:
        """See :meth:`Predictive.ppf` (vectorised bisection on the exact mixture CDF)."""
        u = np.asarray(u, dtype=np.float64).reshape(-1)                            # [U]
        if not ((u > 0) & (u < 1)).all():
            raise ValueError(f"MixturePredictive.ppf: levels must lie in (0, 1), got [{u.min()}, {u.max()}].")
        sites = np.asarray(sites, dtype=int).reshape(-1)
        return self._ppf_components(u, self.means[:, sites], np.sqrt(self.variances[:, sites]))

    def _ppf_components(self, u: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        """Bisection quantiles from component means and SDs at the requested sites.

        Args:
            u: Levels in (0, 1), shape [U].
            mu: Component means, shape [P, S, C].
            sd: Component SDs, shape [P, S, C].

        Returns:
            Quantiles, shape [P, S, U].
        """
        w = np.broadcast_to(self.weights[:, None, :], mu.shape)                    # [P, S, C]
        z = float(-ndtri(min(u.min(), 1.0 - u.max()) / 2.0)) + 1.0               # bracket: every component
        lo = np.broadcast_to((mu - z * sd).min(-1)[..., None], (*mu.shape[:2], u.size)).copy()   # [P, S, U]
        hi = np.broadcast_to((mu + z * sd).max(-1)[..., None], lo.shape).copy()
        target = np.broadcast_to(u, lo.shape)
        for _ in range(self.bisection_steps):
            mid = 0.5 * (lo + hi)
            below = self._mixture_cdf(mid, mu, sd, w) < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def support(self, u_tail: float) -> np.ndarray:
        """See :meth:`Predictive.support`; bounds from the components holding at least ``u_tail`` of the mass."""
        z = float(ndtri(1.0 - u_tail))
        sd = np.sqrt(self.variances)
        heavy = (self.weights >= min(u_tail, self.weights.max(axis=1, keepdims=True).min()))[:, None, :]
        lo = np.where(heavy, self.means - z * sd, np.inf).min(-1)                  # [P, M]
        hi = np.where(heavy, self.means + z * sd, -np.inf).max(-1)
        return np.stack([lo, hi], axis=-1)

    def take(self, idx: Sequence[int] | np.ndarray) -> "MixturePredictive":
        """See :meth:`Predictive.take`."""
        idx = np.asarray(idx, dtype=int)
        return MixturePredictive(self.weights[idx], self.means[idx], self.variances[idx],
                                 max_elements=self.max_elements, bisection_steps=self.bisection_steps,
                                 exact_max_components=self.exact_max_components, sd_bins=self.sd_bins)


class RankOneMixturePredictive(MixturePredictive):
    """A GP-Bayes probe predictive kept as rank-one factors instead of ``[P, M, C]`` component arrays.

    After probe ``p`` at site ``j`` every component is ``mu_c(x) + k_c(x, x_j) r_pc``, with ``r_pc`` the surprise
    over the predictive variance ``d_c`` at ``x_j``, and variance ``max(v_c(x) - k_c(x, x_j)^2 / d_c, 0) +
    sigma_c^2`` (exact sequential Bayes, :meth:`GPBayesEngine.probe`). Components are rebuilt per site or per probe
    with exactly the arithmetic the materialised mixture used, so every value is bit-identical; memory is
    ``O((U + 2) M C + P C)`` for ``U`` probed sites instead of ``O(P M C)`` (~21 GiB for one 80-value probe at
    t = 10 on NHP, task #28, 2026-10-21).

    Args:
        weights: Posterior component weights after each probe, shape [P, C].
        mu: Context latent means per component, shape [C, M].
        kcs: Latent cross-covariance with each probed site, shape [U, C, M].
        site_variances: Component predictive variances after a probe at each probed site, shape [U, M, C].
        site_of_probe: Index into the ``U`` probed sites for every probe, shape [P].
        ratio: ``r_pc / d_c`` for every probe, shape [P, C].
        max_elements: See :class:`MixturePredictive`.
        bisection_steps: See :class:`MixturePredictive`.
        exact_max_components: See :class:`MixturePredictive`; must be below ``C`` (the binned CDF is used).
        sd_bins: See :class:`MixturePredictive`.

    Raises:
        RuntimeError: On non-finite factors or non-positive component variances.
        ValueError: If the mixture is small enough for the exact CDF (materialise it instead).
    """

    def __init__(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        kcs: np.ndarray,
        site_variances: np.ndarray,
        site_of_probe: np.ndarray,
        ratio: np.ndarray,
        *,
        max_elements: int = 20_000_000,
        bisection_steps: int = 100,
        exact_max_components: int = 64,
        sd_bins: int = 32,
    ) -> None:
        self.exact_max_components = int(exact_max_components)
        self.sd_bins = int(sd_bins)
        self.max_elements = int(max_elements)
        self.bisection_steps = int(bisection_steps)
        self.weights = np.asarray(weights, dtype=np.float64)                       # [P, C]
        self._mu = mu                                                              # [C, M]
        self._kcs = kcs                                                            # [U, C, M]
        self._site_var = site_variances                                            # [U, M, C]
        self._site = np.asarray(site_of_probe, dtype=int)                          # [P]
        self._ratio = ratio                                                        # [P, C]
        if self.weights.shape[1] <= self.exact_max_components:
            raise ValueError("RankOneMixturePredictive: a mixture this small uses the exact CDF; materialise it.")
        if not all(np.isfinite(a).all() for a in (self.weights, mu, kcs, site_variances, ratio)):
            raise RuntimeError("RankOneMixturePredictive: non-finite weights, means or variances.")
        if (site_variances <= 0).any():
            raise RuntimeError("RankOneMixturePredictive: non-positive component variance.")
        P, M = len(self._site), mu.shape[1]
        self.mean = np.empty((P, M))
        second = np.empty((P, M))
        for p in range(P):                                                         # the moments, one probe at a time
            means_p = self._probe_means(p)                                         # [M, C]
            w = self.weights[p][None, :]                                           # [1, C]
            self.mean[p] = (w * means_p).sum(-1)
            second[p] = (w * (self._site_var[self._site[p]] + means_p ** 2)).sum(-1)
        self.var = np.maximum(second - self.mean ** 2, np.finfo(float).tiny)       # [P, M]

    def _probe_means(self, p: int) -> np.ndarray:
        """Component means after probe ``p``, shape [M, C] (C-contiguous, as the materialised array was)."""
        return np.ascontiguousarray((self._mu + self._kcs[self._site[p]] * self._ratio[p][:, None]).T)

    def _site_components(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        """See :meth:`MixturePredictive._site_components`, rebuilt from the factors."""
        means_m = self._mu[:, m][None, :] + self._kcs[:, :, m][self._site] * self._ratio   # [P, C]
        return means_m, self._site_var[:, m, :][self._site]                        # [P, C] x2

    @property
    def means(self) -> np.ndarray:
        """Materialised component means, shape [P, M, C] (memory-heavy; the cell path never calls it)."""
        return np.stack([self._probe_means(p) for p in range(len(self._site))])

    @property
    def variances(self) -> np.ndarray:
        """Materialised component variances, shape [P, M, C] (memory-heavy; the cell path never calls it)."""
        return np.ascontiguousarray(self._site_var[self._site])

    def ppf(self, u: np.ndarray, sites: np.ndarray) -> np.ndarray:
        """See :meth:`MixturePredictive.ppf`, from components rebuilt at the requested sites only."""
        u = np.asarray(u, dtype=np.float64).reshape(-1)                            # [U]
        if not ((u > 0) & (u < 1)).all():
            raise ValueError(f"MixturePredictive.ppf: levels must lie in (0, 1), got [{u.min()}, {u.max()}].")
        comps = [self._site_components(int(s)) for s in np.asarray(sites, dtype=int).reshape(-1)]
        mu = np.stack([c[0] for c in comps], axis=1)                               # [P, S, C]
        var = np.stack([c[1] for c in comps], axis=1)                              # [P, S, C]
        return self._ppf_components(u, mu, np.sqrt(var))

    def support(self, u_tail: float) -> np.ndarray:
        """See :meth:`MixturePredictive.support`, one site at a time."""
        z = float(ndtri(1.0 - u_tail))
        heavy = self.weights >= min(u_tail, self.weights.max(axis=1, keepdims=True).min())   # [P, C]
        P, M = self.mean.shape
        out = np.empty((P, M, 2))
        for m in range(M):
            means_m, variances_m = self._site_components(m)                         # [P, C] x2
            sd = np.sqrt(variances_m)
            out[:, m, 0] = np.where(heavy, means_m - z * sd, np.inf).min(-1)
            out[:, m, 1] = np.where(heavy, means_m + z * sd, -np.inf).max(-1)
        return out

    def take(self, idx: Sequence[int] | np.ndarray) -> "RankOneMixturePredictive":
        """See :meth:`Predictive.take`; the factors are shared, not copied."""
        idx = np.asarray(idx, dtype=int)
        return RankOneMixturePredictive(self.weights[idx], self._mu, self._kcs, self._site_var, self._site[idx],
                                        self._ratio[idx], max_elements=self.max_elements,
                                        bisection_steps=self.bisection_steps,
                                        exact_max_components=self.exact_max_components, sd_bins=self.sd_bins)


class BarPredictive(Predictive):
    """TabPFN's bar-distribution predictive, read through a fitted ``FrozenTabPFN``.

    Args:
        engine: The fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN` that produced the logits.
        log_probs: Output of its ``forward``, a torch tensor of shape [P, M, bars].
    """

    def __init__(self, engine: Any, log_probs: Any) -> None:
        self.engine = engine
        self.log_probs = log_probs                                                 # [P, M, bars]
        mean, sd = engine.moments(log_probs)                                       # [P, M] x2
        self.mean = mean
        self.var = sd ** 2

    def cdf(self, y_grid: np.ndarray) -> np.ndarray:
        """See :meth:`Predictive.cdf`."""
        import torch  # noqa: PLC0415

        P, M = self.mean.shape
        ys = torch.as_tensor(np.broadcast_to(np.asarray(y_grid)[None], (P, M, np.shape(y_grid)[-1])).copy())
        return self.engine.cdf(self.log_probs, ys)                                 # [P, M, G]

    def ppf(self, u: np.ndarray, sites: np.ndarray) -> np.ndarray:
        """See :meth:`Predictive.ppf`."""
        return self.engine.ppf(self.log_probs[:, np.asarray(sites, dtype=int)], np.asarray(u))

    def support(self, u_tail: float) -> np.ndarray:
        """See :meth:`Predictive.support`."""
        return self.engine.ppf(self.log_probs, np.array([u_tail, 1.0 - u_tail]))

    def take(self, idx: Sequence[int] | np.ndarray) -> "BarPredictive":
        """See :meth:`Predictive.take`."""
        return BarPredictive(self.engine, self.log_probs[np.asarray(idx, dtype=int)])


class CriterionPredictive(Predictive):
    """Bar-distribution predictive where every conditioning state has its own criterion (one refit each).

    A public-API refit re-standardises y, so the raw-space bucket borders differ from probe to probe.

    Args:
        parts: Per conditioning state, ``(criterion, logits [M, bars])`` from ``predict(output_type='full')``.
    """

    def __init__(self, parts: Sequence[tuple[Any, Any]]) -> None:
        from ..models.pfn.tabpfn import _bar_distribution_moments  # noqa: PLC0415

        self.parts = list(parts)
        moments = [_bar_distribution_moments(c, lg) for c, lg in self.parts]
        self.mean = np.stack([m for m, _ in moments]).astype(np.float64)          # [P, M]
        self.var = np.stack([sd for _, sd in moments]).astype(np.float64) ** 2     # [P, M]

    def cdf(self, y_grid: np.ndarray) -> np.ndarray:
        """See :meth:`Predictive.cdf`."""
        from ..models.pfn.tabpfn import bar_cdf  # noqa: PLC0415

        return np.stack([bar_cdf(c, lg, np.asarray(y_grid)) for c, lg in self.parts])

    def ppf(self, u: np.ndarray, sites: np.ndarray) -> np.ndarray:
        """See :meth:`Predictive.ppf`."""
        from ..models.pfn.tabpfn import bar_quantiles  # noqa: PLC0415

        idx = np.asarray(sites, dtype=int)
        return np.stack([bar_quantiles(c, lg[idx], np.asarray(u)) for c, lg in self.parts])

    def support(self, u_tail: float) -> np.ndarray:
        """See :meth:`Predictive.support`."""
        from ..models.pfn.tabpfn import bar_quantiles  # noqa: PLC0415

        return np.stack([bar_quantiles(c, lg, np.array([u_tail, 1.0 - u_tail])) for c, lg in self.parts])

    def take(self, idx: Sequence[int] | np.ndarray) -> "CriterionPredictive":
        """See :meth:`Predictive.take`."""
        return CriterionPredictive([self.parts[int(i)] for i in np.asarray(idx, dtype=int)])


# ---------------------------------------------------------------------------
# Engines: one fitted context, many one-observation probes
# ---------------------------------------------------------------------------
class UpdateEngine(Protocol):
    """A model conditioned on a context and probed with one extra observation at a time."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Condition on the context, shapes [t, D] and [t]."""

    def base(self, Q: np.ndarray) -> Predictive:
        """Predictive at ``Q`` [M, D] given the context (``P = 1``)."""

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> Predictive:
        """Predictive at ``Q`` after adding ``(xs[p], ys[p])``, one probe each (``P = len(ys)``)."""


class PFNEngine:
    """TabPFN probed through the batched frozen-preprocessing forward.

    Args:
        device: Torch device.
        random_state: Preprocessing seed (the inference seed).
        softmax_temperature: 0.9 is the deployed setting, 1.0 the network's raw PPD.
        as_train: Hash the injected row's fingerprint in train mode, as a refit on ``D + {(x', y')}`` would.
            ``False`` reproduces the test-mode duplicate of the anchor's own query row (A4 experiment).
        precision: ``'fp32'`` or ``'auto'`` (see ``FrozenTabPFN``).
        max_batch_tokens: Forwarded to ``FrozenTabPFN``.
        name: Engine name.
    """

    def __init__(
        self,
        device: str = "cpu",
        random_state: int = 0,
        softmax_temperature: float = 0.9,
        as_train: bool = True,
        precision: str = "fp32",
        max_batch_tokens: int = 400_000,
        name: str = "tabpfn_v2_5",
    ) -> None:
        from ..models.pfn.tabpfn import FrozenTabPFN  # noqa: PLC0415 - heavy import

        self.engine = FrozenTabPFN(device=device, random_state=random_state, max_batch_tokens=max_batch_tokens,
                                   precision=precision, softmax_temperature=softmax_temperature)
        self.as_train = bool(as_train)
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit (and freeze) preprocessing on the context."""
        self.engine.fit(X, y)

    def base(self, Q: np.ndarray) -> BarPredictive:
        """Predictive given the context alone."""
        Qt = self.engine.transform_x(Q)                                            # [M, F]
        return BarPredictive(self.engine, self.engine.forward(None, None, Qt))    # [1, M, bars]

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> BarPredictive:
        """One batched pass over every probe.

        Each distinct injected coordinate is preprocessed *on its own*: a train-mode fingerprint is made unique
        against the context only, as one refit on ``D + {(x', y')}`` would; transforming several probes together
        would make them unique against each other as well, which no single refit does.
        """
        import torch  # noqa: PLC0415

        xs = np.atleast_2d(np.asarray(xs, dtype=np.float64))                     # [P, D]
        uniq, inverse = np.unique(xs, axis=0, return_inverse=True)
        rows = torch.cat([self.engine.transform_x(u[None], as_train=self.as_train) for u in uniq])   # [U, F]
        Xs = rows[torch.as_tensor(np.asarray(inverse).reshape(-1))].unsqueeze(1)                   # [P, 1, F]
        yt = self.engine.transform_y(
            torch.as_tensor(np.asarray(ys, dtype=np.float32), device=self.engine.device)
        ).reshape(-1, 1)                                                           # [P, 1]
        return BarPredictive(self.engine, self.engine.forward(Xs, yt, self.engine.transform_x(Q)))


class PFNRefitEngine:
    """TabPFN updated by a real public-API refit on ``D + {(x', y')}`` for every probe: the deployed update.

    Preprocessing (feature transforms, fingerprint, y standardisation) is refitted with the new point, exactly as
    the BO loop does. This is TabPFN's engine in OMEGA (user, 2026-10-15); how far the frozen-preprocessing
    :class:`PFNEngine` is from it is a reported discrepancy, not a gate.

    Args:
        device: Torch device.
        random_state: Preprocessing seed (the inference seed).
        softmax_temperature: Deployed 0.9.
        name: Engine name.
    """

    def __init__(self, device: str = "cpu", random_state: int = 0, softmax_temperature: float = 0.9,
                 name: str = "tabpfn_v2_5") -> None:
        from tabpfn import TabPFNRegressor  # noqa: PLC0415 - heavy import

        self.reg = TabPFNRegressor(n_estimators=1, softmax_temperature=float(softmax_temperature),
                                   random_state=int(random_state), ignore_pretraining_limits=True, device=device)
        self.name = name

    def _fit_predict(self, X: np.ndarray, y: np.ndarray, Q: np.ndarray) -> tuple[Any, Any]:
        """``(criterion, logits [M, bars])`` of one public-API fit."""
        self.reg.fit(X, y)
        full = self.reg.predict(np.asarray(Q, dtype=np.float64), output_type="full")
        return full["criterion"], full["logits"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Keep the context; every prediction is a fresh fit."""
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)

    def base(self, Q: np.ndarray) -> CriterionPredictive:
        """Public-API predictive given D."""
        return CriterionPredictive([self._fit_predict(self._X, self._y, Q)])

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> CriterionPredictive:
        """One public-API refit per probe."""
        xs = np.atleast_2d(np.asarray(xs, dtype=np.float64))
        return CriterionPredictive([self._fit_predict(np.vstack([self._X, x]), np.append(self._y, yv), Q)
                                    for x, yv in zip(xs, np.atleast_1d(np.asarray(ys, dtype=np.float64)))])


def _log_grid(spec: Sequence[float]) -> np.ndarray:
    """``(lo, hi, n)`` -> ``n`` log-spaced values (one value when ``n == 1`` and ``lo == hi``)."""
    lo, hi, n = float(spec[0]), float(spec[1]), int(spec[2])
    if lo <= 0 or hi < lo or n < 1 or (n == 1 and hi != lo):
        raise ValueError(f"hyperparameter grid spec must be (lo > 0, hi >= lo, n >= 1), got {list(spec)}.")
    return np.geomspace(lo, hi, n)


class GPBayesEngine:
    """GP with hyperparameters marginalised on a fixed grid (OMEGA §3.2.1; exact sequential Bayes).

    Kernel: zero-mean ARD-RBF ``s^2 exp(-0.5 sum_d (x_d - x'_d)^2 / l_d^2)`` plus a constant ``b^2`` (a N(0, b^2)
    prior on the constant mean, integrated analytically), plus noise ``sigma^2``. The grid over
    ``theta = (l_1..l_D, s^2, sigma^2)`` is log-spaced with a log-uniform prior on the box, so the posterior weights
    are proportional to the exact marginal likelihood on D. The predictive is the mixture
    ``sum_theta w_theta N(mu_theta(x), v_theta(x) + sigma^2_theta)``; a probe reweights every component by its
    predictive density of ``y'`` and applies the rank-one update -- exact sequential Bayes, identical to
    recomputing the mixture on ``D + {(x', y')}``.

    Components whose posterior mass is jointly below ``prune_mass`` are dropped after the fit (and the rest
    renormalised); 0 keeps every component.

    Args:
        lengthscale: ``(lo, hi, n)`` per input dimension (normalised coordinates).
        outputscale: ``(lo, hi, n)`` of ``s^2``.
        noise: ``(lo, hi, n)`` of ``sigma^2``.
        b2: Variance of the constant-mean prior (z-units); 0 for a zero-mean GP.
        prune_mass: Posterior mass that may be discarded.
        chunk: Hyperparameter points per batched Cholesky.
        cache_bytes: Largest cross-factor ``L^-1 k(X, Q)`` kept in memory; above it (a dense grid, gate G4) the
            factor is recomputed in chunks per probe site instead.
        max_elements: Forwarded to :class:`MixturePredictive`.
        exact_max_components: Forwarded to :class:`MixturePredictive` (exact CDF up to this many components).
        sd_bins: Forwarded to :class:`MixturePredictive` (log-SD bins of the binned CDF).
        name: Engine name.
    """

    def __init__(
        self,
        lengthscale: Sequence[float] = (0.05, 2.0, 12),
        outputscale: Sequence[float] = (0.05, 5.0, 8),
        noise: Sequence[float] = (1e-3, 1.0, 10),
        *,
        b2: float = 1.0,
        prune_mass: float = 1e-10,
        chunk: int = 2048,
        cache_bytes: int = 2_000_000_000,
        max_elements: int = 20_000_000,
        exact_max_components: int = 64,
        sd_bins: int = 32,
        name: str = "gp_bayes",
    ) -> None:
        self.grid_spec = {"lengthscale": list(lengthscale), "outputscale": list(outputscale), "noise": list(noise)}
        self.b2 = float(b2)
        self.prune_mass = float(prune_mass)
        self.chunk = int(chunk)
        self.cache_bytes = int(cache_bytes)
        self.mixture_kw = {"max_elements": int(max_elements), "exact_max_components": int(exact_max_components),
                           "sd_bins": int(sd_bins)}
        self.name = name
        self._fitted = False

    # ---------------------------------------------------------------- fitting
    def _theta(self, D: int) -> np.ndarray:
        """Every grid point, shape [Theta, D + 2] (lengthscales, outputscale, noise)."""
        axes = [_log_grid(self.grid_spec["lengthscale"])] * D + [
            _log_grid(self.grid_spec["outputscale"]), _log_grid(self.grid_spec["noise"])]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.reshape(-1) for m in mesh], axis=-1)                    # [Theta, D+2]

    def _kernel(self, A: np.ndarray, B: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Prior covariance (ARD-RBF + constant) for every theta, shape [c, |A|, |B|]."""
        D = A.shape[1]
        diff2 = (A[:, None, :] - B[None, :, :]) ** 2                               # [a, b, D]
        scaled = np.einsum("abd,cd->cab", diff2, 1.0 / theta[:, :D] ** 2)          # [c, a, b]
        return theta[:, D, None, None] * np.exp(-0.5 * scaled) + self.b2

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Posterior weights on the grid, and the factors every later prediction needs.

        Args:
            X: Context inputs, shape [t, D].
            y: Context targets, shape [t].

        Raises:
            RuntimeError: On non-finite inputs or log marginal likelihoods.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise RuntimeError(f"{self.name}.fit received NaN/Inf inputs.")
        theta = self._theta(X.shape[1])                                            # [Theta, D+2]
        t = len(y)
        logml = np.empty(len(theta))
        for start in range(0, len(theta), self.chunk):
            th = theta[start:start + self.chunk]
            K = self._kernel(X, X, th) + th[:, -1, None, None] * np.eye(t)        # [c, t, t]
            L = np.linalg.cholesky(K)                                              # [c, t, t]
            a = np.linalg.solve(L, np.broadcast_to(y, (len(th), t))[..., None])[..., 0]   # [c, t]
            logml[start:start + len(th)] = (-0.5 * (a ** 2).sum(-1) - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(-1)
                                            - 0.5 * t * math.log(2.0 * math.pi))
        if not np.isfinite(logml).all():
            raise RuntimeError(f"{self.name}.fit: non-finite log marginal likelihood on {(~np.isfinite(logml)).sum()} "
                               "grid points.")
        logw_full = logml - logsumexp(logml)                                       # log-uniform prior on the box
        w_full = np.exp(logw_full)
        self.ess = float(1.0 / (w_full ** 2).sum())
        self.posterior_mean = {
            **{f"lengthscale_{d}": float(w_full @ theta[:, d]) for d in range(X.shape[1])},
            "outputscale": float(w_full @ theta[:, -2]), "noise": float(w_full @ theta[:, -1]),
        }
        order = np.argsort(w_full)                                                 # ascending
        dropped = np.cumsum(w_full[order]) <= self.prune_mass
        keep = np.sort(order[~dropped])
        self.n_grid, self.n_kept = len(theta), len(keep)
        self.theta = theta[keep]                                                   # [C, D+2]
        self.log_w = logw_full[keep] - logsumexp(logw_full[keep])                  # [C]
        K = self._kernel(X, X, self.theta) + self.theta[:, -1, None, None] * np.eye(t)
        self._L = np.linalg.cholesky(K)                                            # [C, t, t]
        self._a = np.linalg.solve(self._L, np.broadcast_to(y, (len(keep), t))[..., None])[..., 0]   # [C, t]
        self._X, self._y = X, y
        self._fitted = True
        self._state_key: tuple[int, bytes] | None = None

    def diagnostics(self) -> dict[str, float]:
        """ESS of the posterior weights, the hyperparameter posterior means and the grid sizes."""
        self._check()
        return {"ess": self.ess, "n_grid": float(self.n_grid), "n_kept": float(self.n_kept),
                **{f"post_{k}": v for k, v in self.posterior_mean.items()}}

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} used before fit().")

    def _factor(self, Q: np.ndarray, sl: slice) -> np.ndarray:
        """Cross-factor ``V = L^-1 k(X, Q)`` of one slice of components, shape [c, t, M]."""
        return np.linalg.solve(self._L[sl], self._kernel(self._X, Q, self.theta[sl]))

    def _slices(self) -> list[slice]:
        """Component chunks of :attr:`chunk`."""
        return [slice(i, i + self.chunk) for i in range(0, len(self.theta), self.chunk)]

    def _state(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Per-component latent mean and variance at ``Q``, and the cross-factor when it fits in memory.

        Cached per ``Q``.

        Returns:
            ``(mu [C, M], v [C, M], V [C, t, M] or None)`` with ``V = L^-1 k(X, Q)``.
        """
        Q = np.asarray(Q, dtype=np.float64)
        key = (Q.shape[0], Q.tobytes())
        if self._state_key != key:
            C, t = len(self.theta), len(self._y)
            keep = C * t * len(Q) * 8 <= self.cache_bytes
            mu, v, Vs = np.empty((C, len(Q))), np.empty((C, len(Q))), []
            for sl in self._slices():
                V = self._factor(Q, sl)                                            # [c, t, M]
                mu[sl] = np.einsum("ctm,ct->cm", V, self._a[sl])
                v[sl] = np.maximum(self.theta[sl, -2, None] + self.b2 - (V ** 2).sum(1), 0.0)
                if keep:
                    Vs.append(V)
            self._state_cache = (mu, v, np.concatenate(Vs) if keep else None)
            self._state_key = key
        return self._state_cache

    def _cross(self, Q: np.ndarray, j: int) -> np.ndarray:
        """Latent posterior covariance of every site with site ``j``, per component, shape [C, M]."""
        Q = np.asarray(Q, dtype=np.float64)
        _, _, V = self._state(Q)
        out = np.empty((len(self.theta), len(Q)))
        for sl in self._slices():
            Vc = V[sl] if V is not None else self._factor(Q, sl)                  # [c, t, M]
            prior = self._kernel(Q, Q[j:j + 1], self.theta[sl])[:, :, 0]           # [c, M]
            out[sl] = prior - np.einsum("ctm,ct->cm", Vc, Vc[:, :, j])
        return out

    def base(self, Q: np.ndarray) -> MixturePredictive:
        """Mixture predictive given the context."""
        self._check()
        mu, v, _ = self._state(Q)
        noise = self.theta[:, -1]                                                  # [C]
        return MixturePredictive(np.exp(self.log_w)[None], mu.T[None], (v + noise[:, None]).T[None],
                                 **self.mixture_kw)

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> MixturePredictive:
        """Exact sequential update per probe: reweight, then the rank-one mean / variance update.

        Raises:
            ValueError: If a probe coordinate is not one of the readout sites ``Q``.
        """
        self._check()
        Q = np.asarray(Q, dtype=np.float64)
        xs = np.atleast_2d(np.asarray(xs, dtype=np.float64))                     # [P, D]
        ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))                     # [P]
        match = (np.abs(xs[:, None, :] - Q[None, :, :]).max(-1) == 0)             # [P, M]
        if not match.any(axis=1).all():
            raise ValueError(f"{self.name}.probe: every probe site must be a readout site of Q.")
        site = match.argmax(axis=1)                                                # [P]
        mu, v, _ = self._state(Q)
        noise = self.theta[:, -1]                                                  # [C]
        P, (C, M) = len(ys), mu.shape
        if C > self.mixture_kw["exact_max_components"]:
            return self._probe_factors(ys, site, Q, mu, v, noise)
        w_out = np.empty((P, C))
        m_out = np.empty((P, M, C))
        v_out = np.empty((P, M, C))
        for j in np.unique(site):
            kc = self._cross(Q, int(j))                                            # [C, M]
            d = v[:, j] + noise                                                    # [C]
            for p in np.flatnonzero(site == j):
                r = ys[p] - mu[:, j]                                               # [C]
                logw = self.log_w - 0.5 * (r ** 2 / d + np.log(2.0 * math.pi * d))
                w_out[p] = np.exp(logw - logsumexp(logw))
                m_out[p] = (mu + kc * (r / d)[:, None]).T                          # [M, C]
                v_out[p] = (np.maximum(v - kc ** 2 / d[:, None], 0.0) + noise[:, None]).T
        return MixturePredictive(w_out, m_out, v_out, **self.mixture_kw)

    def _probe_factors(self, ys: np.ndarray, site: np.ndarray, Q: np.ndarray, mu: np.ndarray, v: np.ndarray,
                       noise: np.ndarray) -> RankOneMixturePredictive:
        """:meth:`probe` for large mixtures: the same update, kept as rank-one factors (task #28).

        Args:
            ys: Probe values, shape [P].
            site: Readout-site index of every probe, shape [P].
            Q: Readout sites, shape [M, D].
            mu: Context latent means per component, shape [C, M].
            v: Context latent variances per component, shape [C, M].
            noise: Component noise variances, shape [C].

        Returns:
            The probed predictive, bit-identical to the materialised one.
        """
        uniq = np.unique(site)                                                     # [U]
        kcs = np.stack([self._cross(Q, int(j)) for j in uniq])                     # [U, C, M]
        ds = np.stack([v[:, j] + noise for j in uniq])                             # [U, C]
        site_var = np.stack([np.ascontiguousarray((np.maximum(v - kc ** 2 / d[:, None], 0.0) + noise[:, None]).T)
                             for kc, d in zip(kcs, ds)])                           # [U, M, C]
        pos = np.searchsorted(uniq, site)                                          # [P]
        w_out = np.empty((len(ys), len(noise)))
        ratio = np.empty_like(w_out)
        for p, (u, j) in enumerate(zip(pos, site)):
            d = ds[u]
            r = ys[p] - mu[:, j]                                                   # [C]
            logw = self.log_w - 0.5 * (r ** 2 / d + np.log(2.0 * math.pi * d))
            w_out[p] = np.exp(logw - logsumexp(logw))
            ratio[p] = r / d
        return RankOneMixturePredictive(w_out, mu, kcs, site_var, pos, ratio, **self.mixture_kw)

    def latent_covariance(self, Q: np.ndarray) -> np.ndarray:
        """Mixture latent posterior covariance ``sum w Sigma_theta + Cov_theta(mu_theta)`` over ``Q``.

        Returns:
            Shape [M, M].
        """
        self._check()
        Q = np.asarray(Q, dtype=np.float64)
        mu, _, V = self._state(Q)
        w = np.exp(self.log_w)                                                     # [C]
        cov = np.zeros((len(Q), len(Q)))
        step = max(1, self.chunk // 8)
        for start in range(0, len(w), step):
            sl = slice(start, start + step)
            Vc = V[sl] if V is not None else self._factor(Q, sl)                  # [c, t, M]
            prior = self._kernel(Q, Q, self.theta[sl])                             # [c, M, M]
            post = prior - np.einsum("ctm,ctn->cmn", Vc, Vc)
            cov += np.einsum("c,cmn->mn", w[sl], post)
        mbar = w @ mu                                                              # [M]
        cov += np.einsum("c,cm,cn->mn", w, mu, mu) - np.outer(mbar, mbar)
        return 0.5 * (cov + cov.T)


class GPFixedEngine(GPBayesEngine):
    """GP-fixed (ARD-free RBF, fixed hyperparameters, zero mean): GP-Bayes on a one-point grid with ``b^2 = 0``.

    Identical to ``NaiveGPSurrogate`` with the same hyperparameters (tested), and closed form throughout.

    Args:
        lengthscale: RBF lengthscale (every dimension).
        outputscale: Signal variance.
        noise: Noise variance.
        name: Engine name.
    """

    def __init__(self, lengthscale: float, outputscale: float, noise: float, name: str = "gp_fixed") -> None:
        super().__init__((lengthscale, lengthscale, 1), (outputscale, outputscale, 1), (noise, noise, 1),
                         b2=0.0, prune_mass=0.0, name=name)

    def posterior_covariance(self, Q: np.ndarray) -> np.ndarray:
        """Exact latent posterior covariance ``k_C`` over ``Q``, shape [M, M]."""
        return self.latent_covariance(Q)


class GPRefitEngine:
    """Converged type-II ML GP refitted on ``D + {(x', y')}`` for every probe (the empirical-Bayes yardstick).

    Args:
        gp_params: Keyword arguments of :class:`~pfns4neurostim.models.gp.surrogates.GPSurrogate`.
        name: Engine name.
    """

    def __init__(self, gp_params: Mapping[str, Any], name: str = "gp_mll_refit") -> None:
        from ..models.gp.surrogates import GPSurrogate  # noqa: PLC0415

        self.gp = GPSurrogate(**dict(gp_params))
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit hyperparameters on the context."""
        self.gp.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))

    @staticmethod
    def _gaussian(mean: np.ndarray, sd: np.ndarray) -> MixturePredictive:
        """One-component predictive from ``[P, M]`` mean and SD."""
        return MixturePredictive(np.ones((mean.shape[0], 1)), mean[..., None], (sd ** 2)[..., None])

    def base(self, Q: np.ndarray) -> MixturePredictive:
        """Closed-form noisy predictive with the context's fitted hyperparameters."""
        m, s = self.gp.predict_frozen(np.asarray(Q, dtype=np.float64))
        return self._gaussian(m[None], s[None])

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> MixturePredictive:
        """Refit per probe, then the closed-form noisy predictive."""
        Q = np.asarray(Q, dtype=np.float64)
        m, s = self.gp.predict_frozen(Q)
        dm, ds = self.gp.posterior_update(xs, ys, Q, refit=True)                   # [P, M] x2
        return self._gaussian(m[None] + dm, s[None] + ds)

    def posterior_covariance(self, Q: np.ndarray) -> np.ndarray:
        """Latent posterior covariance with the context's fitted hyperparameters, shape [M, M]."""
        return self.gp.posterior_covariance(np.asarray(Q, dtype=np.float64))


class GPFrozenEngine(GPRefitEngine):
    """Converged type-II ML GP fitted on D, updated in closed form with its hyperparameters frozen.

    The reference of Hypothesis C1 ("is the update GP-like?") only: a scoped exception to "GP-MLL frozen is gone
    everywhere" (user, 2026-10-19). Its mean update is ``k_C(x, x') (y' - mu(x')) / (k_C(x', x') + s^2)``, linear in
    the surprise, so every C1 readout has a known perfect-GP value. It is never a coherence (Part A) engine.

    Args:
        gp_params: Keyword arguments of :class:`~pfns4neurostim.models.gp.surrogates.GPSurrogate`.
        name: Engine name.
    """

    def __init__(self, gp_params: Mapping[str, Any], name: str = "gp_mll_frozen") -> None:
        super().__init__(gp_params, name=name)

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> MixturePredictive:
        """Closed-form update with the context's hyperparameters."""
        Q = np.asarray(Q, dtype=np.float64)
        m, s = self.gp.predict_frozen(Q)
        dm, ds = self.gp.posterior_update(xs, ys, Q, refit=False)                  # [P, M] x2
        return self._gaussian(m[None] + dm, s[None] + ds)


class ToyUpdater:
    """Gate G2's deliberately non-Bayesian updater: GP-fixed with a distorted mean update.

    ``mu'(x) = mu(x) + gain_mult * dmu_fixed(x) + bias_sd * s(x')``; the updated variance stays GP-fixed's. A
    gain multiplier breaks the law of total variance (``U / M = 1 / gain_mult^2``), a bias breaks the martingale.

    Args:
        fixed: The GP-fixed engine it distorts.
        gain_mult: Multiplier of the mean update.
        bias_sd: Constant mean offset after the update, in units of the predictive SD at the probe site.
        name: Engine name.
    """

    def __init__(self, fixed: GPFixedEngine, gain_mult: float = 1.0, bias_sd: float = 0.0, name: str = "toy") -> None:
        self.fixed = fixed
        self.gain_mult = float(gain_mult)
        self.bias_sd = float(bias_sd)
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the underlying GP-fixed."""
        self.fixed.fit(X, y)

    def base(self, Q: np.ndarray) -> MixturePredictive:
        """GP-fixed's predictive."""
        return self.fixed.base(Q)

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> MixturePredictive:
        """GP-fixed's probe with the mean update scaled and offset."""
        base = self.fixed.base(Q)
        upd = self.fixed.probe(xs, ys, Q)
        Q = np.asarray(Q, dtype=np.float64)
        xs = np.atleast_2d(np.asarray(xs, dtype=np.float64))
        site = (np.abs(xs[:, None, :] - Q[None, :, :]).max(-1) == 0).argmax(axis=1)  # [P]
        shift = self.bias_sd * base.sd[0, site]                                   # [P]
        means = base.means + self.gain_mult * (upd.means - base.means) + shift[:, None, None]
        return MixturePredictive(upd.weights, means, upd.variances, max_elements=upd.max_elements,
                                 exact_max_components=upd.exact_max_components, sd_bins=upd.sd_bins)


# ---------------------------------------------------------------------------
# Quadrature, grids, distances between distributions
# ---------------------------------------------------------------------------
def gh_nodes(K: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilists' Gauss-Hermite nodes and weights normalised to sum to 1.

    Args:
        K: Number of nodes.

    Returns:
        ``(z [K], w [K])``: ``sum_k w_k g(z_k) ~ E[g(Z)]``, ``Z ~ N(0, 1)``, exact for polynomials of degree < 2K.
    """
    z, w = np.polynomial.hermite_e.hermegauss(int(K))
    return z.astype(np.float64), (w / w.sum()).astype(np.float64)


#: Quadrature schemes over the model's own predictive of ``y'`` at the anchor (all in u = CDF space):
#:   gauss_hermite  u = clip(Phi(z_k)), Gauss-Hermite nodes and weights (the plan's scheme).
#:   truncated      u = alpha/2 + (1 - alpha) Phi(z_k): Gauss-Hermite over the central mass 1 - alpha only.
#:   rqmc           scrambled Sobol points in (0, 1), equal weights, several independent scrambles (randomised QMC;
#:                  the literature's predictive resampling, with QMC variance reduction and an error bar).
QUADRATURE_METHODS: tuple[str, ...] = ("gauss_hermite", "truncated", "rqmc")


def quadrature_levels(
    method: str,
    K: int,
    *,
    u_min: float,
    central_mass: float = 1.0,
    n_scrambles: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Levels ``u`` and weights of one quadrature scheme.

    Args:
        method: One of :data:`QUADRATURE_METHODS`.
        K: Nodes (Gauss-Hermite) or points per scramble (RQMC; a power of 2 keeps Sobol balanced).
        u_min: Levels are clipped to ``[u_min, 1 - u_min]``.
        central_mass: ``1 - alpha`` of the truncated scheme.
        n_scrambles: Independent scrambles (RQMC).
        rng: Generator of the scrambles (RQMC).

    Returns:
        ``(u [S, K], w [K])``; ``S = n_scrambles`` for RQMC, 1 otherwise.

    Raises:
        ValueError: On an unknown method or a missing generator.
    """
    if method in ("gauss_hermite", "truncated"):
        z, w = gh_nodes(K)
        u = ndtr(z)
        if method == "truncated":
            alpha = 1.0 - float(central_mass)
            u = 0.5 * alpha + (1.0 - alpha) * u
        return np.clip(u, u_min, 1.0 - u_min)[None], w
    if method == "rqmc":
        from scipy.stats import qmc  # noqa: PLC0415

        if rng is None:
            raise ValueError("quadrature_levels: rqmc needs a generator.")
        u = np.stack([qmc.Sobol(d=1, scramble=True, seed=rng).random(int(K))[:, 0] for _ in range(int(n_scrambles))])
        return np.clip(u, u_min, 1.0 - u_min), np.full(int(K), 1.0 / int(K))
    raise ValueError(f"quadrature_levels: unknown method {method!r}; known {QUADRATURE_METHODS}.")


def quadrature_values(base: Predictive, anchor: int, u: np.ndarray) -> np.ndarray:
    """The model's own ``y'`` at the anchor: its quantile function at the levels ``u``.

    Args:
        base: Predictive given D.
        anchor: Anchor site.
        u: Levels, shape [K].

    Returns:
        ``y'_k``, shape [K].
    """
    return base.ppf(np.asarray(u), np.array([anchor]))[0, 0]


def site_grid(
    predictives: Sequence[Predictive],
    u_tail: float,
    n_points: int,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    """A uniform y grid per site covering every predictive's central mass (and any extra values).

    Args:
        predictives: Predictives whose ``1 - 2 u_tail`` ranges the grid must cover.
        u_tail: Tail probability left out on each side.
        n_points: Grid points per site.
        extra: Values that must lie on the grid at each site, shape [M, H] (NaN ignored), or None.

    Returns:
        Grid, shape [M, G].
    """
    sup = np.concatenate([p.support(u_tail).reshape(-1, p.mean.shape[1], 2) for p in predictives])   # [*, M, 2]
    lo, hi = sup[..., 0].min(0), sup[..., 1].max(0)                               # [M]
    if extra is not None:
        lo = np.fmin(lo, np.nanmin(extra, axis=1))
        hi = np.fmax(hi, np.nanmax(extra, axis=1))
    return np.linspace(lo, hi, int(n_points), axis=-1)                            # [M, G]


def w1(cdf_a: np.ndarray, cdf_b: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """1-Wasserstein distance ``int |F_a - F_b| dy`` per site (trapezoid on the grid).

    Args:
        cdf_a: CDFs, shape [..., M, G].
        cdf_b: CDFs, broadcastable to ``cdf_a``.
        y_grid: Grid, shape [M, G].

    Returns:
        Shape [..., M].
    """
    return np.trapz(np.abs(cdf_a - cdf_b), y_grid, axis=-1)


def crps_grid(cdf: np.ndarray, y_obs: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """CRPS ``int (F(y) - 1{y >= y_obs})^2 dy`` on a uniform per-site grid (Gneiting & Raftery 2007).

    Cumulative trapezoid integrals of ``F^2`` and ``(1 - F)^2`` are interpolated at each observation; outside the
    grid the integrand is taken as 1 (``F = 0`` below, ``F = 1`` above).

    Args:
        cdf: CDFs, shape [P, M, G].
        y_obs: Observations per site, shape [M, H] (NaN entries give NaN).
        y_grid: Uniform grid per site, shape [M, G].

    Returns:
        Shape [P, M, H].
    """
    dy = (y_grid[:, 1] - y_grid[:, 0])[None, :, None]                             # [1, M, 1]
    f2, g2 = cdf ** 2, (1.0 - cdf) ** 2
    A = np.concatenate([np.zeros(cdf.shape[:-1] + (1,)), np.cumsum(0.5 * (f2[..., 1:] + f2[..., :-1]), -1)], -1)
    B = np.concatenate([np.zeros(cdf.shape[:-1] + (1,)), np.cumsum(0.5 * (g2[..., 1:] + g2[..., :-1]), -1)], -1)
    A, B = A * dy, B * dy                                                          # [P, M, G]
    lo, hi, G = y_grid[:, 0], y_grid[:, -1], y_grid.shape[1]
    pos = (np.asarray(y_obs, dtype=np.float64) - lo[:, None]) / (hi - lo)[:, None] * (G - 1)   # [M, H]
    clipped = np.clip(np.nan_to_num(pos), 0.0, G - 1.0)
    i0 = np.minimum(np.floor(clipped).astype(int), G - 2)                          # [M, H]
    frac = clipped - i0
    rows = np.arange(y_grid.shape[0])[:, None]
    A_at = A[:, rows, i0] * (1.0 - frac) + A[:, rows, i0 + 1] * frac               # [P, M, H]
    B_at = B[:, rows, i0] * (1.0 - frac) + B[:, rows, i0 + 1] * frac
    out = A_at + (B[..., -1][..., None] - B_at)
    below = np.maximum(lo[:, None] - y_obs, 0.0)                                   # mass outside the grid
    above = np.maximum(y_obs - hi[:, None], 0.0)
    out = out + np.nan_to_num(below)[None] + np.nan_to_num(above)[None]
    return np.where(np.isnan(np.asarray(y_obs, dtype=np.float64))[None], np.nan, out)


def linear_transfer(dmu: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Least-squares transfer through the origin: ``T(x) = sum_v e_v dmu_v(x) / sum_v e_v^2``.

    Args:
        dmu: Mean changes, shape [V, M].
        e: Innovations ``y'_v - mu(x')``, shape [V].

    Returns:
        ``T``, shape [M].

    Raises:
        RuntimeError: If every innovation is zero.
    """
    e = np.asarray(e, dtype=np.float64)
    denom = float(e @ e)
    if denom <= 0:
        raise RuntimeError("linear_transfer: every innovation is zero.")
    return (e @ np.asarray(dmu, dtype=np.float64)) / denom


def far_field_share(T: np.ndarray, dist: np.ndarray, mask: np.ndarray, far_pitch: float) -> float | None:
    """Share of the transfer energy ``T^2`` at distance >= ``far_pitch``, over the masked sites.

    Args:
        T: Transfer, shape [M].
        dist: Distance from the anchor in pitches, shape [M].
        mask: Sites included (unobserved), shape [M].
        far_pitch: Far-field threshold.

    Returns:
        The share, ``None`` without energy.
    """
    e = np.asarray(T)[mask] ** 2
    total = float(e.sum())
    return None if total <= 0 else float(e[np.asarray(dist)[mask] >= far_pitch].sum() / total)


def transfer_profile(
    T: np.ndarray,
    dist: np.ndarray,
    observed: np.ndarray,
    anchor: int,
    bins: Sequence[float],
) -> list[dict[str, Any]]:
    """Mean |transfer| per distance bin, observed and unobserved sites apart (C1's update-vs-distance figure).

    Args:
        T: Transfer of one engine, shape [M].
        dist: Distance of every site from the anchor, in pitches, shape [M].
        observed: Context-site mask, shape [M].
        anchor: Anchor site (excluded: its transfer is the gain).
        bins: Bin edges in pitches, increasing; a site falls in ``[lo, hi)``.

    Returns:
        One row per (site pool, bin) with at least one site: ``site_pool``, ``bin_lo``, ``bin_hi``, ``abs_transfer``,
        ``n_sites``.

    Raises:
        ValueError: If fewer than two edges are given or they do not increase.
    """
    edges = np.asarray(bins, dtype=np.float64)
    if edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"transfer_profile: bin edges must increase, got {list(bins)}.")
    T, dist = np.asarray(T, dtype=np.float64), np.asarray(dist, dtype=np.float64)
    keep = np.arange(len(T)) != anchor
    rows = []
    for pool, mask in (("unobserved", ~np.asarray(observed, dtype=bool)), ("observed", np.asarray(observed, dtype=bool))):
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = keep & mask & (dist >= lo) & (dist < hi)
            if sel.any():
                rows.append({"site_pool": pool, "bin_lo": float(lo), "bin_hi": float(hi),
                             "abs_transfer": float(np.abs(T[sel]).mean()), "n_sites": int(sel.sum())})
    return rows


# ---------------------------------------------------------------------------
# Part A: coherence
# ---------------------------------------------------------------------------
def coherence_anchor(
    base: Predictive,
    probes: Predictive,
    y_quad: np.ndarray,
    w: np.ndarray,
    anchor: int,
    y_grid: np.ndarray,
    eps: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Part A at one anchor, from the predictives at the quadrature values of ONE scheme / scramble.

    Raw readouts (OMEGA §3.4) and their **moment-matched** versions (``*_c``, added 2026-10-15). The quadrature
    measure's own first two moments, ``y_bar`` and ``v = sum_k w_k (y'_k - y_bar)^2``, should equal the model's
    stated predictive mean and variance at ``x'``; truncation, clipping and sampling make them differ. The
    corrected readouts remove that error as a control variate through the measure's linear part -- the drift
    subtracts ``T_A(x) (y_bar - mu(x'))`` with ``T_A = C / v``, and ``M`` and ``C`` are rescaled by ``s^2(x') / v``
    -- so for a linear-Gaussian Bayesian model (GP-fixed) they are exact under ANY scheme, while the raw ones are
    exact only for the full Gauss-Hermite measure.

    Args:
        base: Predictive given D (``P = 1``).
        probes: Predictive after each ``(x', y'_k)``, ``P = K``.
        y_quad: ``y'_k``, shape [K].
        w: Quadrature weights, shape [K].
        anchor: Anchor site ``x'``.
        y_grid: Grid per site, shape [M, G] (for the distributional drift).
        eps: Influenced sites: ``M(x) >= eps s^2(x)``.

    Returns:
        ``(row, C, C_c)``: per-anchor metrics and the raw / moment-matched implied covariance ``C(x', .)``, [M].

    Raises:
        RuntimeError: If a quantity is non-finite.
    """
    mu, s2 = base.mean[0], base.var[0]                                             # [M]
    mus, s2s = probes.mean, probes.var                                             # [K, M]
    mubar = w @ mus                                                                # [M]
    move = w @ (mus - mubar) ** 2                                                  # M(x)  [M]
    shrink = s2 - w @ s2s                                                          # U(x)  [M]
    ybar = float(w @ y_quad)
    v = float(w @ (y_quad - ybar) ** 2)                                            # measure's variance of y'
    C = w @ ((y_quad - ybar)[:, None] * (mus - mubar))                             # [M]
    delta = (mubar - mu) / np.sqrt(s2)                                             # [M]
    F_mix = np.einsum("k,kmg->mg", w, probes.cdf(y_grid))                          # [M, G]
    drift_w1 = w1(F_mix, base.cdf(y_grid)[0], y_grid) / np.sqrt(s2)                # [M]
    R = (w @ s2s + move) / s2                                                      # [M]
    rho = float(s2[anchor]) / v                                                    # stated / measured variance
    move_c, C_c = move * rho, C * rho                                              # [M]
    delta_c = (mubar - mu - (C / v) * (ybar - mu[anchor])) / np.sqrt(s2)           # [M]
    R_c = (w @ s2s + move_c) / s2                                                  # [M]
    for name, arr in (("delta", delta), ("C", C), ("R", R), ("drift_w1", drift_w1), ("delta_c", delta_c)):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"coherence_anchor: {name} is non-finite.")
    infl = move >= eps * s2                                                        # [M]
    msum, msum_c = float(move[infl].sum()), float(move_c[infl].sum())
    row = {
        "anchor": int(anchor),
        "drift_abs": _median(np.abs(delta[infl])),
        "drift_w1": _median(drift_w1[infl]),
        # ratio of sums over I (user, 2026-10-15): a median of per-site ratios exploded at small-M sites
        "shrink_move_ratio": None if msum == 0 else float(shrink[infl].sum() / msum),
        "total_var_ratio": _median(R[infl]),
        "drift_abs_anchor": float(abs(delta[anchor])),
        "latent_frac": float(C[anchor] / s2[anchor]),
        "implied_noise": float(s2[anchor] - C[anchor]),
        "n_influenced": int(infl.sum()),
        "drift_abs_c": _median(np.abs(delta_c[infl])),
        "shrink_move_ratio_c": None if msum_c == 0 else float(shrink[infl].sum() / msum_c),
        "total_var_ratio_c": _median(R_c[infl]),
        "latent_frac_c": float(C_c[anchor] / s2[anchor]),
        "implied_noise_c": float(s2[anchor] - C_c[anchor]),
        "measure_mean_err": float((ybar - mu[anchor]) / np.sqrt(s2[anchor])),
        "measure_var_ratio": 1.0 / rho,
        # tail flag (replaces gate G3, 2026-10-16): the most extreme quadrature y', in the model's SDs at x'. TabPFN's
        # outer bar buckets put rare levels at |z| ~ 10^2, where a scramble SE can understate the integration error.
        "y_quad_max_z": float(np.abs(y_quad - mu[anchor]).max() / np.sqrt(s2[anchor])),
        # exact-floor checks (gate G1): the worst site, not the median
        "max_abs_drift": float(np.abs(delta).max()),
        "max_abs_total_var_dev": float(np.abs(R - 1.0).max()),
        "max_abs_drift_c": float(np.abs(delta_c).max()),
        "max_abs_total_var_dev_c": float(np.abs(R_c - 1.0).max()),
    }
    return row, C, C_c


def _average_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean of per-scramble rows (RQMC), with a standard error ``<key>_se`` for every numeric metric."""
    if len(rows) == 1:
        return dict(rows[0])
    out: dict[str, Any] = {"anchor": rows[0]["anchor"], "n_scrambles": len(rows)}
    for key in rows[0]:
        if key == "anchor":
            continue
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            out[key] = None
            continue
        arr = np.asarray(vals, dtype=np.float64)
        if key == "y_quad_max_z":                                                   # a flag: worst scramble
            out[key] = float(arr.max())
            continue
        out[key] = float(arr.mean()) if key != "n_influenced" else int(round(arr.mean()))
        if key != "n_influenced":
            out[f"{key}_se"] = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else None
    return out


def coherence_scheme(
    engine: UpdateEngine,
    base: Predictive,
    Q: np.ndarray,
    anchor: int,
    u: np.ndarray,
    w: np.ndarray,
    eps: float,
    *,
    u_tail: float,
    grid_points: int,
    extra: Sequence[tuple[np.ndarray, Predictive]] = (),
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Part A at one anchor for one scheme, averaged over its scrambles.

    Args:
        engine: Fitted engine.
        base: Its predictive given D.
        Q: Readout sites, shape [M, D].
        anchor: Anchor site.
        u: Levels per scramble, shape [S, K].
        w: Weights, shape [K].
        eps: Influenced-site threshold.
        u_tail: Tail probability left off the W1 grid.
        grid_points: Grid points per site.
        extra: Already-probed ``(y'_k, predictive)`` per scramble, used instead of probing again.

    Returns:
        ``(row, C, C_c)`` averaged over the scrambles.
    """
    rows, Cs, Ccs = [], [], []
    for sidx in range(len(u)):
        if extra:
            yq, probes = extra[sidx]
        else:
            yq = quadrature_values(base, anchor, u[sidx])
            probes = engine.probe(np.repeat(Q[anchor][None], len(yq), axis=0), yq, Q)
        grid = site_grid([base, probes], u_tail, grid_points)
        row, C, Cc = coherence_anchor(base, probes, yq, w, anchor, grid, eps)
        rows.append(row)
        Cs.append(C)
        Ccs.append(Cc)
    return _average_rows(rows), np.mean(Cs, axis=0), np.mean(Ccs, axis=0)


def coherence(
    engine: UpdateEngine,
    Q: np.ndarray,
    anchors: np.ndarray,
    K: int,
    eps: float,
    *,
    method: str = "gauss_hermite",
    u_min: float = 1e-12,
    central_mass: float = 1.0,
    n_scrambles: int = 1,
    rng: np.random.Generator | None = None,
    u_tail: float = 1e-6,
    grid_points: int = 512,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Part A for a fitted engine at every anchor (the self-contained form used by gates and tests).

    Args:
        engine: Engine already fitted on D.
        Q: Readout sites, shape [M, D].
        anchors: Anchor sites, shape [A].
        K: Nodes / points per scramble.
        eps: Influenced-site threshold.
        method: One of :data:`QUADRATURE_METHODS`.
        u_min: Level clip. 1e-12 clips only the outermost Gauss-Hermite pair at K = 20 (second-moment error
            2e-12); 1e-6 also clips the next two pairs (error 1.05e-6), which alone would fail gate G1's 1e-8
            floor for GP-fixed (measured 2026-10-15).
        central_mass: Kept mass of the truncated scheme.
        n_scrambles: RQMC scrambles.
        rng: RQMC generator.
        u_tail: Tail probability left off the W1 grid.
        grid_points: Grid points per site.

    Returns:
        ``(rows, C [A, M], C_c [A, M])``.
    """
    base = engine.base(Q)
    rows, Cs, Ccs = [], [], []
    for a in np.asarray(anchors, dtype=int):
        u, w = quadrature_levels(method, K, u_min=u_min, central_mass=central_mass, n_scrambles=n_scrambles,
                                 rng=rng)
        row, C, Cc = coherence_scheme(engine, base, Q, int(a), u, w, eps, u_tail=u_tail, grid_points=grid_points)
        rows.append(row)
        Cs.append(C)
        Ccs.append(Cc)
    return rows, np.asarray(Cs), np.asarray(Ccs)


def rqmc_floor(
    engine: UpdateEngine,
    Q: np.ndarray,
    anchors: np.ndarray,
    pools: Sequence[str],
    K: int,
    eps: float,
    *,
    n_scrambles: int,
    rng: np.random.Generator,
    u_min: float = 1e-12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Gate G1 for an exactly Bayesian engine under RQMC: is its residual consistent with zero? (user, 2026-10-15)

    A deterministic rule cannot certify GP-Bayes: its quadrature residual (~1e-2) is a bias indistinguishable from a
    violation. Under RQMC the scrambles are independent unbiased estimates, so each moment-matched signed readout
    -- the drift ``delta_c(x)``, the law-of-total-variance deviation ``R_c(x) - 1`` and the implied covariance
    ``C_c(x', x)`` -- has a scramble mean and a standard error per site. For an exact model the mean is zero up to
    its SE. Per anchor, ``z = median_I |mean| / median_I SE`` over the influenced sites I (the same median-on-I as
    ``drift_abs``); per pool, the asymmetry ``z = ||C_AA - C_AA^T||_F / sqrt(sum SE^2)`` over the off-diagonal
    anchor pairs (the two entries of a pair come from different anchors' independent scrambles, so their SEs add in
    quadrature). The influenced sites share the same y' draws and move almost together, so under the null a z is
    close to ONE ``|t|`` variate with ``S - 1`` degrees of freedom (the median of many, about 0.7) -- an upper
    bound, since partially correlated sites only shrink it; see :func:`rqmc_critical_z` for the gate's threshold.

    Args:
        engine: Engine already fitted on D.
        Q: Readout sites, shape [M, D].
        anchors: Anchor sites, shape [A].
        pools: Pool of each anchor.
        K: Sobol points per scramble.
        eps: Influenced-site threshold.
        n_scrambles: Independent scrambles (>= 2).
        rng: Scrambling generator.
        u_min: Level clip.

    Returns:
        ``(rows, cells)``: per anchor ``drift_z``, ``total_var_z``, ``drift_abs_c`` and ``drift_se``; per pool
        ``cov_asymmetry_z``.

    Raises:
        ValueError: If fewer than 2 scrambles are requested.
    """
    if int(n_scrambles) < 2:
        raise ValueError(f"rqmc_floor needs >= 2 scrambles for a standard error, got {n_scrambles}.")
    base = engine.base(Q)
    mu, s2 = base.mean[0], base.var[0]                                              # [M]
    anchors = np.asarray(anchors, dtype=int)
    rows, C_mean, C_se = [], [], []
    for a in anchors:
        u, w = quadrature_levels("rqmc", int(K), u_min=u_min, n_scrambles=int(n_scrambles), rng=rng)  # [S, K], [K]
        d, r, c, mv = [], [], [], []
        for us in u:
            yq = quadrature_values(base, int(a), us)                               # [K]
            pr = engine.probe(np.repeat(Q[[a]], len(yq), axis=0), yq, Q)
            mus, s2s = pr.mean, pr.var                                             # [K, M]
            mubar = w @ mus
            ybar = float(w @ yq)
            v = float(w @ (yq - ybar) ** 2)
            C = w @ ((yq - ybar)[:, None] * (mus - mubar))                          # [M]
            rho = float(s2[a]) / v
            move_c = (w @ (mus - mubar) ** 2) * rho                                 # [M]
            d.append((mubar - mu - (C / v) * (ybar - mu[a])) / np.sqrt(s2))
            r.append((w @ s2s + move_c) / s2 - 1.0)
            c.append(C * rho)
            mv.append(move_c)
        d, r, c = np.asarray(d), np.asarray(r), np.asarray(c)                       # [S, M]
        S = d.shape[0]
        infl = np.mean(mv, axis=0) >= eps * s2                                      # [M]

        def z(x: np.ndarray) -> float | None:
            m, se = np.abs(x.mean(0))[infl], (x.std(0, ddof=1) / np.sqrt(S))[infl]
            return None if not infl.any() or np.median(se) == 0 else float(np.median(m) / np.median(se))

        rows.append({"anchor": int(a), "drift_z": z(d), "total_var_z": z(r),
                     "drift_abs_c": _median(np.abs(d.mean(0))[infl]),
                     "drift_se": _median((d.std(0, ddof=1) / np.sqrt(S))[infl])})
        C_mean.append(c.mean(0))
        C_se.append(c.std(0, ddof=1) / np.sqrt(S))
    C_mean, C_se = np.asarray(C_mean), np.asarray(C_se)                             # [A, M]
    cells = []
    for pool in ANCHOR_POOLS:
        idx = np.flatnonzero(np.asarray(pools) == pool)
        if idx.size < 2:
            continue
        A = C_mean[np.ix_(idx, anchors[idx])]                                       # [a, a]
        E = C_se[np.ix_(idx, anchors[idx])]
        off = ~np.eye(idx.size, dtype=bool)
        se2 = float((E ** 2 + E.T ** 2)[off].sum())
        cells.append({"pool": pool, "cov_asymmetry_z": None if se2 == 0
                      else float(np.linalg.norm((A - A.T)[off]) / np.sqrt(se2))})
    return rows, cells


def rqmc_critical_z(n_scrambles: int, n_tests: int, alpha: float) -> float:
    """Family-wise critical value of the :func:`rqmc_floor` z statistics (Bonferroni over ``n_tests``).

    Each z is at most a ``|t|`` with ``n_scrambles - 1`` degrees of freedom under the null, so a fixed "3 SE" rule
    applied to the worst of ~10^2 anchors fails an exact model by chance (measured 2026-10-15: z = 3.3 on GP-Bayes,
    8 scrambles, 4 anchors, no trend in K). Returns ``t^{-1}_{S-1}(1 - alpha / (2 n_tests))``.
    """
    from scipy.stats import t as student_t  # noqa: PLC0415

    return float(student_t.ppf(1.0 - float(alpha) / (2.0 * int(n_tests)), int(n_scrambles) - 1))


def coherence_cell(
    rows: Sequence[Mapping[str, Any]],
    C: np.ndarray,
    anchors: np.ndarray,
    pools: Sequence[str],
    trial_var: np.ndarray,
    C_c: np.ndarray | None = None,
    pool_names: Sequence[str] = ANCHOR_POOLS,
) -> list[dict[str, Any]]:
    """Per-pool cell summaries of Part A.

    Args:
        rows: Per-anchor rows of :func:`coherence_anchor`, in anchor order.
        C: Implied covariance rows ``C(x'_i, .)``, shape [A, M].
        anchors: Anchor sites, shape [A].
        pools: Pool of each anchor.
        trial_var: Empirical variance of the trials at each anchor, shape [A].
        C_c: Moment-matched implied covariance rows, shape [A, M]; adds the ``*_c`` companions.
        pool_names: Pools summarised, in output order (pass every anchor as one pool for the pooled summary).

    Returns:
        One row per pool present: medians of the per-anchor metrics, ``cov_asymmetry``, ``psd_neg_mass`` and
        ``sigma_recovery_rho`` (and their ``*_c`` versions when ``C_c`` is given).
    """
    out = []
    for pool in pool_names:
        idx = np.flatnonzero(np.asarray(pools) == pool)
        if idx.size == 0:
            continue
        cell: dict[str, Any] = {"pool": pool, "n_anchors": int(idx.size)}
        for key in ("drift_abs", "drift_w1", "shrink_move_ratio", "total_var_ratio", "latent_frac", "implied_noise",
                    "drift_abs_c", "shrink_move_ratio_c", "total_var_ratio_c", "latent_frac_c", "implied_noise_c"):
            vals = np.array([rows[i][key] for i in idx if rows[i].get(key) is not None], dtype=np.float64)
            cell[key] = _median(vals)
        for key in ("max_abs_drift", "max_abs_total_var_dev", "max_abs_drift_c", "max_abs_total_var_dev_c"):
            cell[key] = float(max(rows[i][key] for i in idx))
        tv = np.asarray(trial_var, dtype=np.float64)[idx]
        for suffix, mat in (("", C), ("_c", C_c)):
            if mat is None:
                continue
            CAA = mat[np.ix_(idx, np.asarray(anchors)[idx])]                       # [a, a]
            norm = float(np.linalg.norm(CAA))
            cell[f"cov_asymmetry{suffix}"] = None if norm == 0 else float(np.linalg.norm(CAA - CAA.T) / norm)
            eig = np.linalg.eigvalsh(0.5 * (CAA + CAA.T))
            total = float(np.abs(eig).sum())
            cell[f"psd_neg_mass{suffix}"] = None if total == 0 else float(np.abs(eig[eig < 0]).sum() / total)
            noise = np.array([rows[i][f"implied_noise{suffix}"] for i in idx])
            ok = idx.size >= 3 and np.std(noise) > 0 and np.std(tv) > 0
            cell[f"sigma_recovery_rho{suffix}"] = float(spearmanr(noise, tv).correlation) if ok else None
        out.append(cell)
    return out


# ---------------------------------------------------------------------------
# Part B: head-to-head on shared y'
# ---------------------------------------------------------------------------
def heldout_trials(channel: ChannelData, context_sites: np.ndarray, context_trials: np.ndarray) -> np.ndarray:
    """Every valid trial at every site except the one the context used there.

    Args:
        channel: Channel (stressed when a knob applies).
        context_sites: Context site indices, shape [t].
        context_trials: Trial index used at each, shape [t].

    Returns:
        Trial values, shape [M, R] with NaN where invalid or used by the context.
    """
    Y = np.array(channel.Y_trials, dtype=np.float64, copy=True)                    # [M, R]
    Y[np.asarray(context_sites, dtype=int), np.asarray(context_trials, dtype=int)] = np.nan
    return Y


def shared_values(
    heldout: np.ndarray,
    anchors: np.ndarray,
    ref_base: Predictive,
    n_trials: int,
    ladder: Sequence[float],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """The ``y'`` values every engine is probed with, per anchor (identical for every engine).

    Args:
        heldout: Output of :func:`heldout_trials`, shape [M, R].
        anchors: Anchor sites, shape [A].
        ref_base: The ladder reference's predictive given D (``bayes_update.shared.reference``, a non-TabPFN
            engine): the model-neutral ladder anchor.
        n_trials: Held-out trials per anchor, drawn without replacement; an anchor with fewer valid held-out
            trials contributes all of them (NHP has 28 trials per site on subject 0, fewer at some sites of other
            subjects -- found 2026-10-15).
        ladder: Multipliers ``c`` of ``y' = mu_ref(x') + c s_ref(x')``.
        rng: Generator.

    Returns:
        ``{'trial': list of [R_a], 'trial_idx': list of [R_a], 'ladder': [A, L], 'c': [L]}``; ``R_a <= n_trials``.

    Raises:
        ValueError: If an anchor has fewer than 2 held-out trials (no transfer can be fitted).
    """
    trial, idx = [], []
    for a in np.asarray(anchors, dtype=int):
        valid = np.flatnonzero(np.isfinite(heldout[a]))
        if valid.size < 2:
            raise ValueError(f"shared_values: site {a} has {valid.size} held-out trials; at least 2 are needed.")
        pick = rng.choice(valid, size=min(int(n_trials), int(valid.size)), replace=False)
        trial.append(heldout[a, pick])
        idx.append(pick)
    c = np.asarray(ladder, dtype=np.float64)
    mu, sd = ref_base.mean[0, anchors], ref_base.sd[0, anchors]                    # [A]
    return {"trial": trial, "trial_idx": [np.asarray(i, dtype=int) for i in idx],
            "ladder": mu[:, None] + c[None, :] * sd[:, None], "c": c}


def headtohead_anchor(
    preds: Mapping[str, Mapping[str, Predictive]],
    references: Sequence[str],
    *,
    anchor: int,
    trial_y: np.ndarray,
    trial_idx: np.ndarray,
    ladder_c: np.ndarray,
    s_ref_anchor: float,
    heldout: np.ndarray,
    observed: np.ndarray,
    dist: np.ndarray,
    y_grid: np.ndarray,
    far_pitch: float,
    saturation_c: Sequence[float],
    distance_bins: Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Part B at one anchor: per-engine descriptors and site-matched contrasts against each reference.

    Args:
        preds: ``engine -> {'base': P=1, 'trial': P=R, 'ladder': P=L}``.
        references: Reference engines (each present in ``preds`` is contrasted with every other engine).
        anchor: Anchor site.
        trial_y: Injected held-out trials, shape [R].
        trial_idx: Their trial indices at the anchor, shape [R].
        ladder_c: Ladder multipliers, shape [L].
        s_ref_anchor: The ladder reference's predictive SD at the anchor (the ladder and surprise unit).
        heldout: Held-out trials per site, shape [M, H] (NaN = excluded).
        observed: Context-site mask, shape [M].
        dist: Distance of every site from the anchor, in pitches, shape [M].
        y_grid: Grid per site, shape [M, G], covering every predictive and every held-out trial.
        far_pitch: Far-field threshold.
        saturation_c: ``(c_lin, c_sat)``: the saturation ratio compares the ``+-c_sat`` response with the
            ``+-c_lin`` one, scaled by ``c_sat / c_lin``.
        distance_bins: Bin edges (pitches) of the per-engine transfer-vs-distance profile; ``None`` = no profile.

    Returns:
        ``(rows, surprise_rows, profile_rows)``. Rows carry ``source``, ``engine``, ``reference`` (``'none'`` for a
        per-engine descriptor) and the metrics; contrasts are matched per site (and per held-out trial) before any
        average. Profile rows are :func:`transfer_profile`'s, tagged with ``engine`` and ``anchor``.
    """
    M = len(observed)
    unobs = ~np.asarray(observed, dtype=bool)
    off = unobs & (np.arange(M) != anchor)
    # held-out set per injected trial: the injected one is excluded at the anchor
    masks = np.repeat(np.isfinite(heldout)[None], len(trial_y), axis=0)            # [R, M, H]
    for v, j in enumerate(trial_idx):
        masks[v, anchor, j] = False
    stats: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    surprise: list[dict[str, Any]] = []
    profile: list[dict[str, Any]] = []
    for name, pr in preds.items():
        b, tr, ld = pr["base"], pr["trial"], pr["ladder"]
        e = trial_y - b.mean[0, anchor]                                            # [R]
        T = linear_transfer(tr.mean - b.mean, e)                                   # [M]
        cdf_b, cdf_t = b.cdf(y_grid), tr.cdf(y_grid)                               # [1, M, G], [R, M, G]
        crps_b = crps_grid(cdf_b, heldout, y_grid)[0]                              # [M, H]
        crps_t = crps_grid(cdf_t, heldout, y_grid)                                 # [R, M, H]
        dcrps = np.where(masks, crps_t - crps_b[None], np.nan)                     # [R, M, H]
        d_anchor = ld.mean[:, anchor] - b.mean[0, anchor]                          # [L]
        resp = dict(zip(np.round(ladder_c, 12).tolist(), d_anchor.tolist()))
        c_lin, c_sat = (float(c) for c in saturation_c)
        denom = (c_sat / c_lin) * (resp[c_lin] - resp[-c_lin])
        saturation = None if denom == 0 else float((resp[c_sat] - resp[-c_sat]) / denom)
        stats[name] = {"T": T, "cdf_b": cdf_b[0], "cdf_t": cdf_t, "dcrps": dcrps}
        rows.append({
            "source": "trial", "engine": name, "reference": "none", "anchor": int(anchor),
            "gain": float(T[anchor]),
            "far_field_share": far_field_share(T, dist, unobs, far_pitch),
            "delta_crps": float(np.nanmean(np.nanmean(dcrps.reshape(len(trial_y), -1), axis=1))),
        })
        rows.append({"source": "ladder", "engine": name, "reference": "none", "anchor": int(anchor),
                     "saturation": saturation})
        surprise += [{"engine": name, "anchor": int(anchor), "c": float(c),
                      "update_anchor_sd": float(d / s_ref_anchor)} for c, d in zip(ladder_c, d_anchor)]
        if distance_bins is not None:
            profile += [{"engine": name, "anchor": int(anchor), **p}
                        for p in transfer_profile(T, dist, observed, anchor, distance_bins)]
    for ref in references:
        if ref not in stats:
            continue
        r = stats[ref]
        s_ref = preds[ref]["base"].sd[0]                                           # [M]
        for name, m in stats.items():
            if name == ref:
                continue
            Tm, Tr = m["T"][off], r["T"][off]
            g_before = w1(m["cdf_b"], r["cdf_b"], y_grid) / s_ref                  # [M]
            g_after = w1(m["cdf_t"], r["cdf_t"], y_grid).mean(0) / s_ref           # [M] (mean over injected trials)
            change = g_after - g_before                                            # matched per site
            diff = m["dcrps"] - r["dcrps"]                                         # matched per (site, trial)
            denom = float(Tr @ Tr)
            rows.append({
                "source": "trial", "engine": name, "reference": ref, "anchor": int(anchor),
                "footprint_r": _pearson(Tm, Tr),
                "transfer_slope": None if denom == 0 else float(Tm @ Tr / denom),
                "gap_before": _median(g_before[off]),
                "gap_after": _median(g_after[off]),
                "gap_change": _median(change[off]),
                "gap_change_anchor": float(change[anchor]),
                "delta_crps": float(np.nanmean(np.nanmean(diff.reshape(len(trial_y), -1), axis=1))),
            })
    return rows, surprise, profile


# ---------------------------------------------------------------------------
# BETA (2026-10-21): belief embeddings, site-matched W2 split, posterior covariances. Purely additive -- nothing above
# calls these, so no ``bayes_update`` cell identity or payload depends on them (OMEGA's grid runs at ``bb0bba5``).
# ---------------------------------------------------------------------------
def belief_levels(n_quantiles: int) -> np.ndarray:
    """Midpoint quantile levels ``u_j = (j - 1/2) / J`` of the belief embedding (BETA §4).

    Args:
        n_quantiles: ``J`` (``representation.n_quantiles``; 32 in the plan).

    Returns:
        Levels, shape [J].

    Raises:
        ValueError: For ``J < 2`` (no spread can be read from one quantile).
    """
    if int(n_quantiles) < 2:
        raise ValueError(f"belief_levels: need >= 2 quantiles, got {n_quantiles}.")
    return (np.arange(int(n_quantiles)) + 0.5) / int(n_quantiles)


def belief_embedding(pred: Predictive, levels: np.ndarray, probe: int = 0) -> np.ndarray:
    """Belief embedding ``E(x) = [Q(u_1 | x, D), ..., Q(u_J | x, D)]`` at every site (BETA §4).

    ``||E(x) - E(x')||^2 / J`` approximates ``W2^2`` between the two sites' predictives, so the linear Gram of the
    centred embedding is the 2-Wasserstein geometry of the model's beliefs over the sites.

    Args:
        pred: A predictive over ``M`` sites (TabPFN's bar distribution, a GP's Gaussian mixture).
        levels: Quantile levels, shape [J] (:func:`belief_levels`).
        probe: Conditioning state to read (0 = the base predictive given D).

    Returns:
        Quantiles, shape [M, J].

    Raises:
        RuntimeError: If a quantile is non-finite.
    """
    E = np.asarray(pred.ppf(np.asarray(levels, dtype=np.float64), np.arange(pred.mean.shape[1]))[int(probe)],
                   dtype=np.float64)                                               # [M, J]
    if not np.isfinite(E).all():
        raise RuntimeError(f"belief_embedding: {int((~np.isfinite(E)).sum())} non-finite quantile(s).")
    return E


def site_w2_split(E_a: np.ndarray, E_b: np.ndarray) -> dict[str, np.ndarray]:
    """Per-site 2-Wasserstein distance between two models' predictives, split into location / spread / shape (β1a).

    Computed from the quantile embeddings, ``W2^2(x) = mean_j (Q_a(u_j | x) - Q_b(u_j | x))^2``, and split with the
    embedding's own first two moments ``m = mean_j Q``, ``s = sd_j Q`` (BETA §4)::

        W2^2 = (m_a - m_b)^2 + (s_a - s_b)^2 + 2 s_a s_b (1 - rho_ab)
                location         spread          shape >= 0

    where ``rho_ab`` is the correlation of the two centred quantile vectors (Cauchy-Schwarz makes the shape term
    non-negative exactly, so the split is an identity, not an approximation). Matched per site before any aggregate
    (BETA D3).

    Args:
        E_a: Belief embedding of model a, shape [M, J].
        E_b: Belief embedding of model b, shape [M, J].

    Returns:
        ``{'w2', 'location', 'spread', 'shape', 'sd_a', 'sd_b'}``, each [M]; ``location`` / ``spread`` / ``shape`` are
        squared terms summing to ``w2 ** 2``.

    Raises:
        ValueError: On mismatched shapes.
        RuntimeError: On non-finite input.
    """
    A, Bm = np.asarray(E_a, dtype=np.float64), np.asarray(E_b, dtype=np.float64)
    if A.shape != Bm.shape:
        raise ValueError(f"site_w2_split: embeddings differ in shape, {A.shape} vs {Bm.shape}.")
    if not (np.isfinite(A).all() and np.isfinite(Bm).all()):
        raise RuntimeError("site_w2_split: non-finite quantiles.")
    ma, mb = A.mean(axis=1), Bm.mean(axis=1)                                       # [M]
    Ac, Bc = A - ma[:, None], Bm - mb[:, None]                                     # [M, J]
    sa, sb = np.sqrt((Ac ** 2).mean(axis=1)), np.sqrt((Bc ** 2).mean(axis=1))      # [M]
    w2sq = ((A - Bm) ** 2).mean(axis=1)                                            # [M]
    location, spread = (ma - mb) ** 2, (sa - sb) ** 2
    shape = np.maximum(w2sq - location - spread, 0.0)                              # [M]; >= 0 up to rounding
    return {"w2": np.sqrt(w2sq), "location": location, "spread": spread, "shape": shape, "sd_a": sa, "sd_b": sb}


def latent_covariance_of(engine: Any, Q: np.ndarray) -> np.ndarray:
    """Latent posterior covariance over ``Q`` of a fitted GP engine (Sigma_MLL, Sigma_fixed or GP-Bayes's Sigma_B).

    Args:
        engine: A fitted :class:`GPRefitEngine` / :class:`GPFrozenEngine` (context-fitted hyperparameters),
            :class:`GPFixedEngine` or :class:`GPBayesEngine` (the mixture's total covariance).
        Q: Sites, shape [M, D].

    Returns:
        Symmetric covariance, shape [M, M].

    Raises:
        TypeError: For an engine with no closed-form posterior covariance (TabPFN: use
            :func:`implied_posterior_covariance` on Part A's implied rows).
        RuntimeError: If the covariance is non-finite.
    """
    if isinstance(engine, (GPRefitEngine, GPFixedEngine)):
        S = engine.posterior_covariance(Q)
    elif isinstance(engine, GPBayesEngine):
        S = engine.latent_covariance(Q)
    else:
        raise TypeError(f"latent_covariance_of: {type(engine).__name__} has no closed-form posterior covariance.")
    S = np.asarray(S, dtype=np.float64)
    if not np.isfinite(S).all():
        raise RuntimeError(f"latent_covariance_of: non-finite covariance from {getattr(engine, 'name', engine)}.")
    return 0.5 * (S + S.T)


def implied_posterior_covariance(C_rows: np.ndarray) -> dict[str, Any]:
    """C_PFN from Part A's implied-covariance rows with every site as an anchor (BETA §4, β1b).

    ``C(x'_i, x) = Cov(y', mu_new(x))`` estimates the latent ``Cov(f(x'_i), f(x) | D)`` row by row (tower property,
    OMEGA A3), so with all sites as anchors the rows form a full matrix. It is symmetrised and projected to the PSD
    cone; the asymmetry ``||C - C'||_F / ||C||_F`` and the clipped negative-eigenvalue mass are reported, never hidden.

    Args:
        C_rows: Implied rows, shape [M, M] (row ``i`` anchored at site ``i``).

    Returns:
        ``{'cov': [M, M] symmetric PSD, 'asymmetry': float, 'clipped_mass': float}``.

    Raises:
        ValueError: If the rows do not form a square matrix.
        RuntimeError: On non-finite rows or an all-zero matrix.
    """
    C = np.asarray(C_rows, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"implied_posterior_covariance: need one row per site ([M, M]), got {C.shape}.")
    if not np.isfinite(C).all():
        raise RuntimeError("implied_posterior_covariance: non-finite implied covariance rows.")
    norm = float(np.linalg.norm(C))
    if norm == 0:
        raise RuntimeError("implied_posterior_covariance: all-zero implied covariance.")
    S = 0.5 * (C + C.T)
    lam, V = np.linalg.eigh(S)                                                     # [M], [M, M]
    total = float(np.abs(lam).sum())
    return {"cov": (V * np.clip(lam, 0.0, None)[None, :]) @ V.T,
            "asymmetry": float(np.linalg.norm(C - C.T) / norm),
            "clipped_mass": float(np.abs(lam[lam < 0]).sum() / total) if total > 0 else 0.0}
