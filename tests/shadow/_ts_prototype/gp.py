"""Exact GP surrogate with a converged marginal-likelihood fit (P0.10).

Phase-3 target: ``src/pfns4neurostim/models/gp/surrogates.py``.

Differences from ``src/models/regressors.GPSurrogate`` (see Phase-1 G1/G2):
  * inputs are the shared preprocessed units (X in [0, 1]^D, y z-scored) — P0.9;
  * float64, ARD-RBF + constant mean + homoscedastic Gaussian noise;
  * L-BFGS (strong-Wolfe) to convergence, multi-start on the first fit, warm start after;
  * exact batched conditioning with frozen hyperparameters (``conditional_marginals``);
  * exact joint latent / predictive sampling driven by an explicit ``np.random.Generator``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .protocol import GaussianMarginals, Space


@dataclass
class GPHyper:
    """Unconstrained GP hyperparameters (softplus-transformed where positive)."""

    raw_lengthscale: torch.Tensor  # [D]
    raw_outputscale: torch.Tensor  # []
    raw_noise: torch.Tensor        # []
    mean: torch.Tensor             # []


def _gamma_logpdf(x: torch.Tensor, concentration: float, rate: float) -> torch.Tensor:
    """Unnormalised Gamma log-density (BoTorch-style weak hyperpriors)."""
    return (concentration - 1.0) * torch.log(x) - rate * x


def _softplus_inv(x: float) -> float:
    return x + math.log(-math.expm1(-x))


class ExactGPSurrogate:
    """Exact GP with converged hyperparameters.

    Args:
        device: Torch device string.
        max_iter: L-BFGS iterations per fit.
        n_restarts: Extra random initialisations on the first fit (warm start afterwards).
        noise_floor: Minimum noise variance (in y units^2).
        jitter: Diagonal jitter added before Cholesky of posterior covariances.
        seed: Seed for the restart initialisations.
        warm_start: Start L-BFGS from the previous fit's hyperparameters.
        use_priors: MAP with weak Gamma hyperpriors (lengthscale Gamma(3, 6), outputscale
            Gamma(2, 0.15), noise Gamma(1.1, 0.05); BoTorch SingleTaskGP legacy defaults).
    """

    name = "GP"
    has_exact_joint = True

    def __init__(
        self,
        device: str = "cpu",
        max_iter: int = 100,
        n_restarts: int = 3,
        noise_floor: float = 1e-4,
        jitter: float = 1e-8,
        seed: int = 0,
        warm_start: bool = True,
        use_priors: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.noise_floor = noise_floor
        self.jitter = jitter
        self.warm_start = warm_start
        self.use_priors = use_priors
        self._rng = np.random.default_rng(seed)
        self._hyper: Optional[GPHyper] = None
        self._X: Optional[torch.Tensor] = None
        self._y: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------- kernels
    def _t(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.array(a, dtype=np.float64), dtype=self.dtype, device=self.device)

    def _params(self, h: GPHyper) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ls = torch.nn.functional.softplus(h.raw_lengthscale) + 1e-4       # [D]
        os_ = torch.nn.functional.softplus(h.raw_outputscale) + 1e-8      # []
        noise = torch.nn.functional.softplus(h.raw_noise) + self.noise_floor  # []
        return ls, os_, noise

    def _k(self, A: torch.Tensor, B: torch.Tensor, h: GPHyper) -> torch.Tensor:
        ls, os_, _ = self._params(h)
        a = A / ls                                   # [..., n, D]
        b = B / ls                                   # [..., m, D]
        d2 = (a.unsqueeze(-2) - b.unsqueeze(-3)).square().sum(-1)  # [..., n, m]
        return os_ * torch.exp(-0.5 * d2)

    def _nll(self, X: torch.Tensor, y: torch.Tensor, h: GPHyper) -> torch.Tensor:
        n = X.shape[0]
        _, _, noise = self._params(h)
        K = self._k(X, X, h) + noise * torch.eye(n, dtype=self.dtype, device=self.device)  # [n, n]
        L = torch.linalg.cholesky(K)
        r = (y - h.mean).unsqueeze(-1)               # [n, 1]
        alpha = torch.cholesky_solve(r, L)           # [n, 1]
        nll = 0.5 * (r * alpha).sum() + torch.log(torch.diagonal(L)).sum() + 0.5 * n * math.log(2 * math.pi)
        if self.use_priors:
            ls, os_, noise = self._params(h)
            nll = (nll - _gamma_logpdf(ls, 3.0, 6.0).sum() - _gamma_logpdf(os_, 2.0, 0.15)
                   - _gamma_logpdf(noise, 1.1, 0.05))
        return nll

    # -------------------------------------------------------------------- fit
    def _init_hyper(self, D: int, y: np.ndarray, random: bool) -> GPHyper:
        var = float(max(np.var(y), 1e-6))
        if random:
            ls0 = np.exp(self._rng.uniform(np.log(0.05), np.log(1.0), size=D))
            sn0 = var * float(np.exp(self._rng.uniform(np.log(0.01), np.log(0.5))))
        else:
            ls0, sn0 = np.full(D, 0.3), 0.1 * var
        mk = lambda v: torch.tensor(v, dtype=self.dtype, device=self.device, requires_grad=True)
        return GPHyper(
            raw_lengthscale=mk(np.array([_softplus_inv(float(v)) for v in ls0])),
            raw_outputscale=mk(_softplus_inv(var)),
            raw_noise=mk(_softplus_inv(max(sn0 - self.noise_floor, 1e-6))),
            mean=mk(float(np.mean(y))),
        )

    def _optimise(self, X: torch.Tensor, y: torch.Tensor, h: GPHyper) -> float:
        params = [h.raw_lengthscale, h.raw_outputscale, h.raw_noise, h.mean]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=self.max_iter, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-6, tolerance_change=1e-9)

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = self._nll(X, y, h)
            loss.backward()
            return loss

        try:
            opt.step(closure)
            with torch.no_grad():
                val = float(self._nll(X, y, h))
        except RuntimeError:  # Cholesky failure for a bad restart: discard this restart only
            return float("inf")
        return val

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit hyperparameters by exact MLL with L-BFGS (multi-start first, warm start after).

        Args:
            X: Observed inputs (shared units), shape [n, D].
            y: Observed targets (shared units), shape [n].

        Raises:
            RuntimeError: On NaN/Inf inputs or if every restart fails.
        """
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise RuntimeError("ExactGPSurrogate.fit received NaN/Inf inputs.")
        Xt, yt = self._t(X), self._t(y)                # [n, D], [n]
        D = X.shape[1]
        cands: list[GPHyper] = []
        if self._hyper is not None and self.warm_start:
            cands.append(GPHyper(*[p.detach().clone().requires_grad_(True) for p in (
                self._hyper.raw_lengthscale, self._hyper.raw_outputscale,
                self._hyper.raw_noise, self._hyper.mean)]))
        else:
            cands.append(self._init_hyper(D, y, random=False))
            cands += [self._init_hyper(D, y, random=True) for _ in range(self.n_restarts)]
        best, best_val = None, float("inf")
        for h in cands:
            val = self._optimise(Xt, yt, h)
            if val < best_val:
                best, best_val = h, val
        if best is None:
            raise RuntimeError("ExactGPSurrogate.fit: all L-BFGS restarts failed.")
        self._hyper = GPHyper(*[p.detach() for p in (best.raw_lengthscale, best.raw_outputscale,
                                                     best.raw_noise, best.mean)])
        self._X, self._y = Xt, yt
        self.nll_ = best_val

    def set_fixed(
        self, X: np.ndarray, y: np.ndarray, lengthscale: float, outputscale: float,
        noise_var: float, mean: float = 0.0,
    ) -> None:
        """Condition on data with *known* hyperparameters (no fitting; ground-truth GP).

        Args:
            X: Inputs [n, D].
            y: Targets [n].
            lengthscale: Isotropic lengthscale.
            outputscale: Signal variance.
            noise_var: Noise variance (must exceed ``noise_floor``).
            mean: Constant mean.
        """
        D = X.shape[1]
        t = lambda v: torch.tensor(v, dtype=self.dtype, device=self.device)
        self._hyper = GPHyper(
            raw_lengthscale=t(np.full(D, _softplus_inv(lengthscale - 1e-4))),
            raw_outputscale=t(_softplus_inv(outputscale - 1e-8)),
            raw_noise=t(_softplus_inv(noise_var - self.noise_floor)),
            mean=t(float(mean)),
        )
        self._X, self._y = self._t(X), self._t(y)

    @property
    def hyperparameters(self) -> dict[str, np.ndarray]:
        """Constrained hyperparameters (lengthscale [D], outputscale, noise variance, mean)."""
        self._check()
        ls, os_, noise = self._params(self._hyper)
        return {"lengthscale": ls.cpu().numpy(), "outputscale": float(os_),
                "noise_var": float(noise), "mean": float(self._hyper.mean)}

    # -------------------------------------------------------------- posterior
    def _posterior(
        self, Xc: torch.Tensor, yc: torch.Tensor, Xq: torch.Tensor, full_cov: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent posterior given (batched) context. Xc [B,n,D], yc [B,n], Xq [B,q,D]."""
        h = self._hyper
        _, os_, noise = self._params(h)
        n = Xc.shape[-2]
        Kcc = self._k(Xc, Xc, h) + noise * torch.eye(n, dtype=self.dtype, device=self.device)  # [B,n,n]
        L = torch.linalg.cholesky(Kcc)                                   # [B,n,n]
        Kqc = self._k(Xq, Xc, h)                                         # [B,q,n]
        alpha = torch.cholesky_solve((yc - h.mean).unsqueeze(-1), L)     # [B,n,1]
        mean = h.mean + (Kqc @ alpha).squeeze(-1)                        # [B,q]
        V = torch.linalg.solve_triangular(L, Kqc.transpose(-1, -2), upper=False)  # [B,n,q]
        if full_cov:
            cov = self._k(Xq, Xq, h) - V.transpose(-1, -2) @ V           # [B,q,q]
            return mean, cov
        var = (os_ - V.square().sum(-2)).clamp_min(1e-12)                # [B,q]
        return mean, var

    def predict_marginals(self, X: np.ndarray, space: Space = "predictive") -> GaussianMarginals:
        """Posterior marginals at X.

        Args:
            X: Query inputs, shape [M, D].
            space: ``'latent'`` (f) or ``'predictive'`` (f + noise).

        Returns:
            GaussianMarginals with leading shape [M].
        """
        self._check()
        with torch.no_grad():
            mean, var = self._posterior(self._X[None], self._y[None], self._t(X)[None], full_cov=False)
            if space == "predictive":
                var = var + self._params(self._hyper)[2]
        return GaussianMarginals(mean[0].cpu().numpy(), var[0].sqrt().cpu().numpy(), space)

    def conditional_marginals(
        self,
        X_query: np.ndarray,
        X_extra: np.ndarray,
        y_extra: np.ndarray,
        include_context: bool = True,
    ) -> GaussianMarginals:
        """Exact predictive marginals given per-batch extra observations (frozen hyperparameters).

        Args:
            X_query: [B, q, D].
            X_extra: [B, r, D].
            y_extra: [B, r].
            include_context: Keep the real context (True) or replace it (False).

        Returns:
            Predictive GaussianMarginals with leading shape [B, q].
        """
        self._check()
        with torch.no_grad():
            Xq, Xe, ye = self._t(X_query), self._t(X_extra), self._t(y_extra)
            B = Xq.shape[0]
            if include_context:
                Xc = torch.cat([self._X.unsqueeze(0).expand(B, -1, -1), Xe], dim=1)  # [B,n+r,D]
                yc = torch.cat([self._y.unsqueeze(0).expand(B, -1), ye], dim=1)      # [B,n+r]
            else:
                Xc, yc = Xe, ye
            mean, var = self._posterior(Xc, yc, Xq, full_cov=False)
            var = var + self._params(self._hyper)[2]
        return GaussianMarginals(mean.cpu().numpy(), var.sqrt().cpu().numpy(), "predictive")

    def sample_joint(
        self, X: np.ndarray, n: int, rng: np.random.Generator, space: Space = "latent"
    ) -> np.ndarray:
        """Exact joint posterior samples over X.

        Args:
            X: Query inputs, shape [M, D].
            n: Number of samples.
            rng: Explicit numpy Generator for the standard-normal draws.
            space: ``'latent'`` (f) or ``'predictive'`` (f + iid noise).

        Returns:
            Samples, shape [n, M].
        """
        self._check()
        with torch.no_grad():
            mean, cov = self._posterior(self._X[None], self._y[None], self._t(X)[None], full_cov=True)
            M = X.shape[0]
            cov = cov[0]
            if space == "predictive":
                cov = cov + self._params(self._hyper)[2] * torch.eye(M, dtype=self.dtype, device=self.device)
            jit = self.jitter * float(cov.diagonal().mean())
            for _ in range(6):
                L, info = torch.linalg.cholesky_ex(cov + jit * torch.eye(M, dtype=self.dtype, device=self.device))
                if int(info) == 0:
                    break
                jit *= 10
            else:
                raise RuntimeError("ExactGPSurrogate.sample_joint: posterior covariance not PSD.")
            z = self._t(rng.standard_normal((n, M)))                      # [n, M]
            s = mean[0] + z @ L.T                                          # [n, M]
        out = s.cpu().numpy()
        if not np.isfinite(out).all():
            raise RuntimeError("ExactGPSurrogate.sample_joint produced NaN/Inf.")
        return out

    def _check(self) -> None:
        if self._hyper is None:
            raise RuntimeError("ExactGPSurrogate used before fit().")
