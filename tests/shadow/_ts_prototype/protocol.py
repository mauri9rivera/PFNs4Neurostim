"""Model-agnostic surrogate / acquisition protocol for the joint-TS prototype.

Phase-3 target: ``src/pfns4neurostim/models/protocol.py``.

Every surrogate exposes *marginals* (per-point predictive distributions) through a
common object with ``mean``, ``std`` and a vectorised inverse CDF, plus a batched
``conditional_marginals`` primitive (condition on per-batch extra rows with FROZEN
hyperparameters / preprocessing). The generic sequential joint sampler in
``thompson.py`` is built only on that primitive, so it runs unchanged on GP and PFN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, runtime_checkable

import numpy as np
import torch

Space = Literal["latent", "predictive"]


class Marginals(Protocol):
    """Batched per-point distributions, arbitrary leading shape ``[...]``."""

    space: Space

    @property
    def mean(self) -> np.ndarray:
        """Distribution means, shape [...]."""
        ...

    @property
    def std(self) -> np.ndarray:
        """Distribution standard deviations, shape [...]."""
        ...

    def icdf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF evaluated elementwise at probabilities ``u`` of shape [...]."""
        ...


@dataclass
class GaussianMarginals:
    """Independent Gaussian marginals.

    Args:
        mean_: Means, shape [...].
        std_: Standard deviations, shape [...].
        space: ``'latent'`` (f) or ``'predictive'`` (y = f + noise).
    """

    mean_: np.ndarray
    std_: np.ndarray
    space: Space

    def __post_init__(self) -> None:
        if not (np.isfinite(self.mean_).all() and np.isfinite(self.std_).all()):
            raise RuntimeError("GaussianMarginals received NaN/Inf mean or std.")

    @property
    def mean(self) -> np.ndarray:
        return self.mean_

    @property
    def std(self) -> np.ndarray:
        return self.std_

    def icdf(self, u: np.ndarray) -> np.ndarray:
        from scipy.special import ndtri

        m, sd = self.mean_, self.std_
        if np.ndim(u) == m.ndim + 1:
            m, sd = m[..., None], sd[..., None]
        return m + sd * ndtri(np.clip(u, 1e-12, 1 - 1e-12))


@dataclass
class BarMarginals:
    """Bar-distribution marginals (TabPFN), optionally affinely shrunk to latent space.

    The latent version is ``f = c + a * (y - c)`` with centre ``c`` = predictive mean and
    factor ``a = sqrt(s)`` from the replicate-sensitivity probe; ``a = 1`` is predictive.

    Args:
        logprobs: Log-probabilities over bars, shape [..., bars].
        borders: Raw-space bar borders, shape [bars + 1].
        space: ``'latent'`` or ``'predictive'``.
        shrink: Optional latent shrink factor ``a``, shape [...] (None -> 1).
    """

    logprobs: torch.Tensor
    borders: torch.Tensor
    space: Space
    shrink: Optional[np.ndarray] = None
    _cache: dict = field(default_factory=dict, repr=False)

    def _moments(self) -> tuple[np.ndarray, np.ndarray]:
        if "m" not in self._cache:
            p = self.logprobs.double().exp()                           # [..., bars]
            b = self.borders.double().to(p.device)                     # [bars+1]
            left, right = b[:-1], b[1:]                                # [bars]
            centre = (left + right) / 2                                # [bars]
            sq = (left.square() + right.square() + left * right) / 3   # [bars] E[y^2 | bin], uniform
            m = (p * centre).sum(-1)                                   # [...]
            v = ((p * sq).sum(-1) - m.square()).clamp_min(1e-12)       # [...]
            m_np, s_np = m.cpu().numpy(), v.sqrt().cpu().numpy()
            if not (np.isfinite(m_np).all() and np.isfinite(s_np).all()):
                raise RuntimeError("BarMarginals: NaN/Inf moments.")
            self._cache["m"], self._cache["s"] = m_np, s_np
        return self._cache["m"], self._cache["s"]

    @property
    def mean(self) -> np.ndarray:
        return self._moments()[0]

    @property
    def std(self) -> np.ndarray:
        s = self._moments()[1]
        return s if self.shrink is None else s * self.shrink

    def predictive_icdf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF of the *unshrunk* bar distribution (uniform within bins).

        Args:
            u: Probabilities of shape [...] (one per distribution) or [..., n]
                (n draws per distribution).

        Returns:
            Quantile values with the shape of ``u``.
        """
        lead = tuple(self.logprobs.shape[:-1])
        multi = np.ndim(u) == len(lead) + 1
        p = self.logprobs.double().exp()                               # [..., bars]
        cdf = p.cumsum(-1)                                             # [..., bars]
        cdf = cdf / cdf[..., -1:].clamp_min(1e-300)
        ut = torch.as_tensor(np.asarray(u), dtype=torch.float64, device=p.device)
        if not multi:
            ut = ut.unsqueeze(-1)                                      # [..., 1]
        idx = torch.searchsorted(cdf.contiguous(), ut.contiguous()).clamp(max=p.shape[-1] - 1)  # [..., n]
        cdf_left = torch.where(idx > 0, cdf.gather(-1, (idx - 1).clamp_min(0)), torch.zeros_like(ut))
        p_bin = p.gather(-1, idx).clamp_min(1e-300)
        frac = ((ut - cdf_left) / p_bin).clamp(0.0, 1.0)
        b = self.borders.double().to(p.device)
        val = b[idx] + frac * (b[idx + 1] - b[idx])                    # [..., n]
        out = (val if multi else val.squeeze(-1)).cpu().numpy()
        if not np.isfinite(out).all():
            raise RuntimeError("BarMarginals.icdf produced NaN/Inf.")
        return out

    def icdf(self, u: np.ndarray) -> np.ndarray:
        y = self.predictive_icdf(u)
        if self.shrink is None:
            return y
        c, a = self.mean, self.shrink
        if np.ndim(u) == c.ndim + 1:
            c, a = c[..., None], a[..., None]
        return c + a * (y - c)


@dataclass
class BOState:
    """Observed data and step counters passed to acquisitions.

    Args:
        X_obs: Observed inputs, shape [n, D].
        y_obs: Observed noisy targets, shape [n].
        obs_idx: Pool indices of observations, shape [n].
        t: Current BO step (0-indexed).
        n_steps: Total BO steps.
    """

    X_obs: np.ndarray
    y_obs: np.ndarray
    obs_idx: np.ndarray
    t: int
    n_steps: int


@dataclass
class AcqResult:
    """Output of an acquisition.

    Args:
        index: Selected pool index.
        scores: Acquisition scores over the pool, shape [M] (-inf = not considered).
        diagnostics: Free-form diagnostics (timings, candidate set size, ...).
    """

    index: int
    scores: np.ndarray
    diagnostics: dict[str, Any]


@runtime_checkable
class Surrogate(Protocol):
    """Surrogate contract required by all acquisitions in this prototype."""

    name: str
    has_exact_joint: bool

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit / set context on observed data, shapes [n, D] and [n]."""
        ...

    def predict_marginals(self, X: np.ndarray, space: Space = "predictive") -> Marginals:
        """Marginals at X, shape [M, D] -> leading shape [M]."""
        ...

    def conditional_marginals(
        self,
        X_query: np.ndarray,
        X_extra: np.ndarray,
        y_extra: np.ndarray,
        include_context: bool = True,
    ) -> Marginals:
        """Predictive marginals given per-batch extra rows (frozen model).

        Args:
            X_query: [B, q, D].
            X_extra: [B, r, D] (r may be 0).
            y_extra: [B, r].
            include_context: If False, the real context is replaced by the extra rows.

        Returns:
            Predictive marginals with leading shape [B, q].
        """
        ...

    def sample_joint(
        self, X: np.ndarray, n: int, rng: np.random.Generator, space: Space = "latent"
    ) -> np.ndarray:
        """Exact joint samples [n, M] (only if ``has_exact_joint``)."""
        ...


class Acquisition(Protocol):
    """Acquisition callable; ``params`` must fully describe it for result metadata."""

    name: str

    def params(self) -> dict[str, Any]:
        ...

    def __call__(
        self,
        surrogate: Surrogate,
        X_pool: np.ndarray,
        state: BOState,
        rng: np.random.Generator,
    ) -> AcqResult:
        ...
