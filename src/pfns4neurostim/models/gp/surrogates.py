"""Gaussian-process surrogates (task #1 Step 3).

Moved from the flat ``src/models/regressors.py``. ``GPSurrogate`` refits its
kernel hyperparameters by marginal likelihood at every BO step; ``NaiveGPSurrogate``
never tunes them (the "no tuning at all" lower bound); ``DeepKernelGPSurrogate``
puts a small MLP in front of the kernel.

``predict_ts`` is the **joint latent** draw (GP-only reference row) and
``predict_ts_marginal`` the per-site predictive draw that both model families
share as the headline TS (task #4, 2026-09-17).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import gpytorch
import numpy as np
import torch

from .exact_gp import DeepKernelGP, ExactGP

__all__ = ["GPSurrogate", "DeepKernelGPSurrogate", "NaiveGPSurrogate"]


# ---------------------------------------------------------------------------
# GPSurrogate — ExactGP wrapper conforming to SurrogateModel
# ---------------------------------------------------------------------------

class GPSurrogate:
    """ExactGP (RBF kernel) wrapper conforming to the ``SurrogateModel`` protocol.

    Trains kernel hyperparameters via marginal likelihood optimisation at each
    ``fit`` call.  ``predict`` returns the posterior mean and standard deviation
    from the trained likelihood.

    Args:
        device: PyTorch device string ('cpu' or 'cuda').
        n_opt_steps: Number of Adam optimiser steps for hyperparameter training.
        lr: Learning rate for the Adam optimiser.
    """

    def __init__(
        self,
        device: str = 'cpu',
        n_opt_steps: int = 100,
        lr: float = 0.1,
    ) -> None:
        self._device = device
        self._n_opt_steps = n_opt_steps
        self._lr = lr
        self._model: ExactGP | None = None
        self._likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None

    def _build_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: "gpytorch.likelihoods.GaussianLikelihood",
    ) -> ExactGP:
        """Construct the GP model for this surrogate (override point for subclasses).

        Args:
            train_x: Training inputs, shape [N, D].
            train_y: Training targets, shape [N].
            likelihood: The Gaussian likelihood instance.

        Returns:
            An initialised gpytorch ExactGP-derived model on the surrogate device.
        """
        return ExactGP(train_x, train_y, likelihood).to(self._device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train GP hyperparameters via marginal likelihood on observed data.

        Args:
            X: Feature matrix of observed points, shape [N, D].  # [N, D]
            y: Response vector of observed targets, shape [N].   # [N]
        """
        if self._model is not None:
            del self._model, self._likelihood

        train_x = torch.tensor(X, dtype=torch.float32, device=self._device)  # [N, D]
        train_y = torch.tensor(y, dtype=torch.float32, device=self._device)  # [N]

        if torch.isnan(train_x).any() or torch.isnan(train_y).any():
            raise RuntimeError(
                "GPSurrogate.fit received NaN inputs. "
                f"X has {torch.isnan(train_x).sum()} NaNs, "
                f"y has {torch.isnan(train_y).sum()} NaNs."
            )

        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self._device)
        self._model = self._build_model(train_x, train_y, self._likelihood)

        self._model.train()
        self._likelihood.train()

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        for _ in range(self._n_opt_steps):
            optimizer.zero_grad()
            output = self._model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        self._model.eval()
        self._likelihood.eval()

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return GP posterior mean and standard deviation.

        Args:
            X: Query feature matrix, shape [M, D].  # [M, D]

        Returns:
            Tuple of (mean, std), each shape [M].   # [M], [M]

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if self._model is None or self._likelihood is None:
            raise RuntimeError(
                "GPSurrogate.predict called before fit. Call fit() first."
            )
        query_x = torch.tensor(X, dtype=torch.float32, device=self._device)  # [M, D]
        with torch.no_grad():
            posterior = self._likelihood(self._model(query_x))
            mean = posterior.mean.cpu().numpy()   # [M]
            std = posterior.stddev.cpu().numpy()  # [M]

        if np.isnan(mean).any() or np.isnan(std).any():
            raise RuntimeError(
                f"GPSurrogate.predict returned NaN values. "
                f"mean NaNs: {np.isnan(mean).sum()}, std NaNs: {np.isnan(std).sum()}."
            )
        return mean, std

    def predict_ucb(
        self,
        X: np.ndarray,
        kappa: float,
        t: int,
        n_steps: int,
    ) -> np.ndarray:
        """Return UCB values using GP posterior mean and standard deviation.

        Uses cosine-annealed kappa internally (the ``kappa`` argument is the
        pre-annealed value computed by ``run_bo_loop``).

        Args:
            X: Candidate feature matrix, shape [M, D].  # [M, D]
            kappa: Current (already-annealed) UCB exploration coefficient.
            t: Current BO step index (unused; annealing done by caller).
            n_steps: Total BO steps (unused; annealing done by caller).

        Returns:
            UCB values, shape [M].  # [M]
        """
        mean, std = self.predict(X)  # [M], [M]
        return mean + kappa * std    # [M]

    def predict_ts(
        self,
        X: np.ndarray,
        temperature: float = 1.0,
        generator: "torch.Generator | None" = None,
        include_noise: bool = False,
    ) -> np.ndarray:
        """Draw one joint Thompson sample from the GP posterior.

        Default (``include_noise=False``) is textbook GP Thompson sampling: one
        draw from the joint latent posterior N(μ, Σ) over all candidates, so the
        sampled values keep the inter-point correlations dictated by the kernel.
        With ``include_noise=True`` the observation noise is added
        (N(μ, Σ + σ²_n I)); this is diagnostic only, because homoscedastic noise
        σ²_n swamps the posterior covariance once the surface is well observed
        and the "joint" draw then behaves like independent per-site draws.

        This is the GP-only reference row of the 2026-09-17 TS design decision
        (see ``.claude/research_design_log.md``): PFN surrogates expose per-site
        marginals only, so the symmetric GP-vs-PFN headline acquisition is
        ``predict_ts_marginal``.

        Only the GP supports this: PFN surrogates expose per-site marginals
        only, so the symmetric GP-vs-PFN headline acquisition is
        ``predict_ts_marginal``.

        Args:
            X: Candidate feature matrix, shape [M, D].  # [M, D]
            temperature: Variance scaling of the sampled distribution.  ``1.0``
                (default) samples the exact predictive posterior; ``>1`` inflates
                and ``<1`` shrinks its spread, matching the effect of the
                bar-distribution temperature in ``TabPFNSurrogate``.
            generator: Optional torch RNG for reproducible draws.  ``None`` uses
                the global torch RNG.
            include_noise: Sample the noisy predictive posterior instead of the
                latent one (diagnostic; see above).

        Returns:
            Thompson sample, shape [M].  # [M]

        Raises:
            RuntimeError: If ``fit`` has not been called yet, or the sample is
                not finite.
            ValueError: If ``temperature`` is not strictly positive.
        """
        if self._model is None or self._likelihood is None:
            raise RuntimeError(
                "GPSurrogate.predict_ts called before fit. Call fit() first."
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        query_x = torch.tensor(X, dtype=torch.float32, device=self._device)  # [M, D]
        with torch.no_grad():
            latent = self._model(query_x)                        # N(μ, Σ) — [M]
            posterior = self._likelihood(latent) if include_noise else latent
            mean = posterior.mean                                # [M]
            # rsample() has no generator argument: draw standard normals explicitly
            # and push them through the Cholesky factor of the covariance.
            # float64 + escalating jitter: the latent covariance of a well-fitted GP
            # over a dense grid is near-singular in float32, where Cholesky fails.
            cov = posterior.covariance_matrix.double()           # [M, M]
            eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
            scale = float(torch.diagonal(cov).mean())
            chol = None
            for exponent in range(-8, -2):                       # 1e-8 … 1e-3 × mean variance
                try:
                    chol = torch.linalg.cholesky(cov + scale * (10.0 ** exponent) * eye)
                    break
                except RuntimeError:
                    continue
            if chol is None:
                raise RuntimeError(
                    "GPSurrogate.predict_ts: latent covariance not positive definite "
                    f"even with 1e-3 relative jitter (M={cov.shape[0]})."
                )
            eps = torch.randn(
                cov.shape[0], dtype=cov.dtype, device=cov.device, generator=generator
            )                                                    # [M]
            sample = mean.double() + np.sqrt(temperature) * (chol @ eps)  # [M]

        out = sample.cpu().numpy()                               # [M]
        if not np.isfinite(out).all():
            raise RuntimeError(
                f"GPSurrogate.predict_ts produced {(~np.isfinite(out)).sum()} "
                "non-finite sample values."
            )
        return out                                               # [M]

    def predict_ts_marginal(
        self,
        X: np.ndarray,
        temperature: float = 1.0,
        generator: "torch.Generator | None" = None,
    ) -> np.ndarray:
        """Draw independent per-site Thompson samples from the GP predictive marginals.

        Samples each candidate independently from its own predictive marginal
        N(μ_i, temperature · (σ²_i + σ²_n)), discarding cross-site correlations.
        This is the symmetric counterpart of ``TabPFNSurrogate.predict_ts``,
        which can only sample per-site because PFN query rows do not attend to
        each other; it is the headline TS for both model families as of the
        2026-09-17 TS design decision.

        Args:
            X: Candidate feature matrix, shape [M, D].  # [M, D]
            temperature: Variance scaling of the per-site draws (``1.0`` = exact
                predictive marginals).
            generator: Optional torch RNG for reproducible draws.  ``None`` uses
                the global torch RNG.

        Returns:
            Thompson sample values, shape [M].  # [M]

        Raises:
            RuntimeError: If ``fit`` has not been called yet, or the sample is
                not finite.
            ValueError: If ``temperature`` is not strictly positive.
        """
        if self._model is None or self._likelihood is None:
            raise RuntimeError(
                "GPSurrogate.predict_ts_marginal called before fit. Call fit() first."
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        query_x = torch.tensor(X, dtype=torch.float32, device=self._device)  # [M, D]
        with torch.no_grad():
            posterior = self._likelihood(self._model(query_x))   # [M]
            mean = posterior.mean                                # [M]
            std = posterior.stddev                               # [M]
            eps = torch.randn(
                mean.shape[0], dtype=mean.dtype, device=mean.device, generator=generator
            )                                                    # [M]
            sample = mean + np.sqrt(temperature) * std * eps     # [M]

        out = sample.cpu().numpy()                               # [M]
        if not np.isfinite(out).all():
            raise RuntimeError(
                f"GPSurrogate.predict_ts_marginal produced "
                f"{(~np.isfinite(out)).sum()} non-finite sample values."
            )
        return out                                               # [M]


# ---------------------------------------------------------------------------
# DeepKernelGPSurrogate — non-stationary GP baseline (§16, limitation L2/L9)
# ---------------------------------------------------------------------------

class DeepKernelGPSurrogate(GPSurrogate):
    """Deep-kernel (non-stationary) GP surrogate conforming to ``SurrogateModel``.

    Reuses the full :class:`GPSurrogate` fit/predict/acquisition machinery but
    swaps the stationary :class:`ExactGP` for a :class:`DeepKernelGP` whose MLP
    feature extractor is trained jointly with the GP hyperparameters. Serves as
    the "best-tuned, expressive GP" baseline: if TabPFN still matches or beats
    it, the on-par-but-cheaper claim is not an artefact of a weak (stationary)
    GP, and the GFS over-smoothing story (A5) is causal rather than a
    weak-baseline artefact.

    A higher default ``n_opt_steps`` and ``lr`` are used than the stationary GP
    because the extra MLP parameters need more optimisation to converge.

    Args:
        device: PyTorch device string ('cpu' or 'cuda').
        n_opt_steps: Adam steps for joint MLP + GP hyperparameter training.
        lr: Adam learning rate.
        feature_dim: Output dimensionality of the learned feature space.
        hidden: Hidden width of the feature-extractor MLP.
    """

    def __init__(
        self,
        device: str = 'cpu',
        n_opt_steps: int = 120,
        lr: float = 0.02,
        feature_dim: int = 2,
        hidden: int = 32,
    ) -> None:
        super().__init__(device=device, n_opt_steps=n_opt_steps, lr=lr)
        self._feature_dim = feature_dim
        self._hidden = hidden

    def _build_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: "gpytorch.likelihoods.GaussianLikelihood",
    ) -> DeepKernelGP:
        """Construct a :class:`DeepKernelGP` on the surrogate device."""
        return DeepKernelGP(
            train_x, train_y, likelihood,
            feature_dim=self._feature_dim, hidden=self._hidden,
        ).to(self._device)


# ---------------------------------------------------------------------------
# NaiveGPSurrogate — fixed-hyperparameter RBF GP lower-bound baseline (§16, L2)
# ---------------------------------------------------------------------------

class NaiveGPSurrogate(GPSurrogate):
    """Naive-default GP: fixed RBF kernel, *no* marginal-likelihood tuning.

    Sets ``n_opt_steps=0`` so kernel hyperparameters keep their default
    initialisation (a fixed lengthscale/outputscale/noise) and are never fit to
    the data. This is the "no tuning at all" lower bound: it isolates how much
    of a tuned GP's performance comes from hyperparameter optimisation, and
    quantifies the tuning that TabPFN avoids entirely.

    Args:
        device: PyTorch device string ('cpu' or 'cuda').
        lengthscale: Fixed RBF lengthscale applied to every input dimension.
        outputscale: Fixed signal variance (scale-kernel output scale).
        noise: Fixed Gaussian observation noise variance.
    """

    def __init__(
        self,
        device: str = 'cpu',
        lengthscale: float = 0.2,
        outputscale: float = 1.0,
        noise: float = 1e-2,
    ) -> None:
        # n_opt_steps=0 → fit() skips the optimisation loop entirely.
        super().__init__(device=device, n_opt_steps=0)
        self._lengthscale = lengthscale
        self._outputscale = outputscale
        self._noise = noise

    def _build_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: "gpytorch.likelihoods.GaussianLikelihood",
    ) -> ExactGP:
        """Construct an :class:`ExactGP` with fixed (unoptimised) hyperparameters."""
        model = ExactGP(train_x, train_y, likelihood).to(self._device)
        # Pin hyperparameters to the naive defaults; fit() runs 0 opt steps so
        # these values persist through prediction.
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = self._lengthscale
            model.covar_module.outputscale = self._outputscale
            likelihood.noise = self._noise
        return model


