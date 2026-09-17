"""TabPFN surrogate adapter exposing marginals / batched conditioning / latent probe.

Phase-3 target: ``src/pfns4neurostim/models/pfn/tabpfn.py``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .protocol import BarMarginals, Space
from .tabpfn_engine import FrozenTabPFN


class TabPFNJointSurrogate:
    """TabPFN (v2.5, single member, T=1) with frozen preprocessing.

    ``predict_marginals(space='latent')`` shrinks the predictive bar distribution by
    ``sqrt(s)``, where ``s`` = d E[y_rep | x, y_fant] / d y_fant is the PFN's own
    replicate sensitivity (for a Gaussian model s = v_f / (v_f + sigma_eps^2)).

    Args:
        device: Torch device string.
        random_state: TabPFN preprocessing seed (fixed across fits).
        probe_delta: Fantasy offset (in predictive SDs) for the sensitivity probe.
    """

    name = "TabPFN"
    has_exact_joint = False

    def __init__(self, device: str = "cuda", random_state: int = 0, probe_delta: float = 1.0) -> None:
        self.engine = FrozenTabPFN(device=device, random_state=random_state)
        self.probe_delta = probe_delta
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Set the in-context examples (shapes [n, D], [n])."""
        self.engine.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
        self._X_obs = np.asarray(X, dtype=np.float64)
        self._fitted = True

    def _bars(self, logprobs: torch.Tensor, shrink: Optional[np.ndarray] = None,
              space: Space = "predictive") -> BarMarginals:
        return BarMarginals(logprobs=logprobs, borders=self.engine.raw_borders, space=space, shrink=shrink)

    def predict_marginals(self, X: np.ndarray, space: Space = "predictive") -> BarMarginals:
        """Marginals at X, shape [M, D] -> leading [M].

        Args:
            X: Query inputs.
            space: ``'predictive'`` (1 pass) or ``'latent'`` (+ batched probe pass(es)).
        """
        Xt = self.engine.transform_x(X)                        # [M, F]
        lp = self.engine.forward(None, None, Xt)[0]            # [M, bars]
        pred = self._bars(lp)
        if space == "predictive":
            return pred
        s = self.sensitivity(X, pred)                          # [M]
        return self._bars(lp, shrink=np.sqrt(s), space="latent")

    def sensitivity(self, X: np.ndarray, pred: BarMarginals) -> np.ndarray:
        """Replicate-sensitivity probe s_j in [0, 1] via symmetric fantasies (one batched pass).

        Args:
            X: Probe inputs, shape [k, D].
            pred: Predictive marginals at X (leading [k]).

        Returns:
            s, shape [k]; aleatoric variance estimate is (1 - s) * pred.std**2.
        """
        from .thompson import replicate_sensitivity

        return replicate_sensitivity(self, X, pred, self.probe_delta)

    def conditional_marginals(
        self,
        X_query: np.ndarray,
        X_extra: np.ndarray,
        y_extra: np.ndarray,
        include_context: bool = True,
    ) -> BarMarginals:
        """Predictive marginals given per-batch extra rows, one batched pass.

        Args:
            X_query: [B, q, D].
            X_extra: [B, r, D].
            y_extra: [B, r].
            include_context: Must be True (context replacement uses ``with_context``).

        Returns:
            BarMarginals with leading shape [B, q].
        """
        if not include_context:
            raise ValueError("TabPFNJointSurrogate: use predict_with_context for context replacement.")
        B, q, D = X_query.shape
        r = X_extra.shape[1]
        Xq = self.engine.transform_x(X_query.reshape(-1, D)).reshape(B, q, -1)        # [B, q, F]
        if r == 0:
            lp = self.engine.forward(None, None, Xq)                                  # [B, q, bars]
        else:
            Xe = self.engine.transform_x(X_extra.reshape(-1, D)).reshape(B, r, -1)    # [B, r, F]
            ye = self.engine.transform_y(torch.as_tensor(y_extra, dtype=torch.float32,
                                                         device=self.engine.device))  # [B, r]
            lp = self.engine.forward(Xe, ye, Xq)
        return self._bars(lp)

    def predict_with_context(
        self, X_query: np.ndarray, X_ctx: np.ndarray, y_ctx: np.ndarray
    ) -> BarMarginals:
        """Predictive marginals under a *replacement* context per batch (frozen preprocessing).

        Reuses the preprocessing fitted on the real context and runs the transformer on the
        replacement rows only (used by the perturbed-context ensemble, design B).

        Args:
            X_query: [B, q, D].
            X_ctx: [B, r, D].
            y_ctx: [B, r].

        Returns:
            BarMarginals with leading shape [B, q].
        """
        e = self.engine
        B, q, D = X_query.shape
        r = X_ctx.shape[1]
        saved_X, saved_y = e.X_ctx, e.y_ctx
        try:
            e.X_ctx = saved_X[:0]
            e.y_ctx = saved_y[:0]
            Xq = e.transform_x(X_query.reshape(-1, D)).reshape(B, q, -1)
            Xe = e.transform_x(X_ctx.reshape(-1, D)).reshape(B, r, -1)
            ye = e.transform_y(torch.as_tensor(y_ctx, dtype=torch.float32, device=e.device))
            lp = e.forward(Xe, ye, Xq)
        finally:
            e.X_ctx, e.y_ctx = saved_X, saved_y
        return self._bars(lp)

    def sample_joint(self, X: np.ndarray, n: int, rng: np.random.Generator,
                     space: Space = "latent") -> np.ndarray:
        """Not available exactly for a PFN; use ``thompson.sequential_joint_sample``."""
        raise NotImplementedError("TabPFN has no exact joint; use the sequential sampler.")
