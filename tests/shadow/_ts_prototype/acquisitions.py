"""Marginal-based acquisitions (UCB / EI / PI / greedy / random) on the shared protocol.

Phase-3 target: ``src/pfns4neurostim/acquisition/{ucb,ei,pi,greedy,random}.py``.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.stats import norm

from .protocol import AcqResult, BOState, Space, Surrogate
from .thompson import TSEnsemble, TSJoint, TSMarginal, argmax_random_tiebreak


def _gaussian_or_quantile_moments(surrogate: Surrogate, X: np.ndarray, space: Space) -> tuple[np.ndarray, np.ndarray]:
    marg = surrogate.predict_marginals(X, space=space)
    return np.asarray(marg.mean), np.asarray(marg.std)


class UCB:
    """mean + kappa * std on the chosen space."""

    name = "ucb"

    def __init__(self, kappa: float = 2.0, space: Space = "predictive") -> None:
        self.kappa, self.space = kappa, space

    def params(self) -> dict[str, Any]:
        return {"kappa": self.kappa, "space": self.space}

    def __call__(self, surrogate: Surrogate, X_pool: np.ndarray, state: BOState,
                 rng: np.random.Generator) -> AcqResult:
        t0 = time.perf_counter()
        m, s = _gaussian_or_quantile_moments(surrogate, X_pool, self.space)
        sc = m + self.kappa * s
        return AcqResult(argmax_random_tiebreak(sc, rng), sc, {"time_s": time.perf_counter() - t0})


class EI:
    """Expected improvement (Gaussian moments) over best_f = max posterior mean at observed x."""

    name = "ei"

    def __init__(self, space: Space = "predictive") -> None:
        self.space = space

    def params(self) -> dict[str, Any]:
        return {"space": self.space, "best_f": "max_mean_at_observed"}

    def __call__(self, surrogate: Surrogate, X_pool: np.ndarray, state: BOState,
                 rng: np.random.Generator) -> AcqResult:
        t0 = time.perf_counter()
        m, s = _gaussian_or_quantile_moments(surrogate, X_pool, self.space)
        best = float(m[np.unique(state.obs_idx)].max())
        z = (m - best) / np.maximum(s, 1e-12)
        sc = (m - best) * norm.cdf(z) + s * norm.pdf(z)
        return AcqResult(argmax_random_tiebreak(sc, rng), sc, {"time_s": time.perf_counter() - t0})


class PI:
    """Probability of improvement over best_f = max posterior mean at observed x."""

    name = "pi"

    def __init__(self, space: Space = "predictive") -> None:
        self.space = space

    def params(self) -> dict[str, Any]:
        return {"space": self.space, "best_f": "max_mean_at_observed"}

    def __call__(self, surrogate: Surrogate, X_pool: np.ndarray, state: BOState,
                 rng: np.random.Generator) -> AcqResult:
        t0 = time.perf_counter()
        m, s = _gaussian_or_quantile_moments(surrogate, X_pool, self.space)
        best = float(m[np.unique(state.obs_idx)].max())
        sc = norm.cdf((m - best) / np.maximum(s, 1e-12))
        return AcqResult(argmax_random_tiebreak(sc, rng), sc, {"time_s": time.perf_counter() - t0})


class Greedy:
    """argmax posterior mean."""

    name = "greedy"

    def params(self) -> dict[str, Any]:
        return {}

    def __call__(self, surrogate: Surrogate, X_pool: np.ndarray, state: BOState,
                 rng: np.random.Generator) -> AcqResult:
        m = np.asarray(surrogate.predict_marginals(X_pool, "predictive").mean)
        return AcqResult(argmax_random_tiebreak(m, rng), m, {})


class RandomAcq:
    """Uniform random pool index (with replacement across steps)."""

    name = "random"

    def params(self) -> dict[str, Any]:
        return {}

    def __call__(self, surrogate: Surrogate, X_pool: np.ndarray, state: BOState,
                 rng: np.random.Generator) -> AcqResult:
        sc = rng.random(X_pool.shape[0])
        return AcqResult(int(np.argmax(sc)), sc, {})


REGISTRY = {c.name: c for c in (UCB, EI, PI, Greedy, RandomAcq, TSJoint, TSMarginal, TSEnsemble)}
