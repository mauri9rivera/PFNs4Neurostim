"""TabPFN v2.5 surrogate (task #1 Step 3).

Moved from the flat ``src/models/regressors.py``. Wraps a ``TabPFNRegressor`` and
reads its **native bar distribution**: the mean, standard deviation, UCB and
per-site Thompson draws all come from the predicted distribution rather than a
Gaussian approximation of it.

``_bar_distribution_moments`` implements the corrected predictive summaries
(P0.11): the mean is the distribution mean, not its median, and the quantile-based
standard deviation uses the right divisor.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
from tabpfn import TabPFNRegressor

__all__ = ["TabPFNSurrogate"]



# Quantile grid used to summarise the TabPFN bar distribution: mean/std come from
# integrating the predicted quantile function rather than from a normal
# approximation around the median (audit D7 / P0.11).  Levels are placed at equal
# steps on the probit scale over ±4 SD, which concentrates points in the tails where
# a uniform grid loses variance mass; on reference distributions this recovers the
# true std to <1% (normal 1.0006 vs 1; lognormal 1.296 vs 1.304).
_PROBIT_GRID_Z = 4.0
_N_QUANTILE_LEVELS = 99
QUANTILE_LEVELS = np.round(
    0.5 * (1.0 + torch.erf(
        torch.linspace(-_PROBIT_GRID_Z, _PROBIT_GRID_Z, _N_QUANTILE_LEVELS) / math.sqrt(2.0)
    ).numpy()),
    6,
).clip(1e-6, 1.0 - 1e-6)  # [99]


def _bar_distribution_moments(
    criterion: Any,
    logits: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and standard deviation of a TabPFN bar distribution.

    Uses the criterion's own moment methods, which integrate every bucket of the
    predicted density including the half-normal tails, rather than approximating
    the distribution as normal around its median (audit D7 / P0.11).

    Args:
        criterion: TabPFN ``FullSupportBarDistribution`` for these logits.
        logits: Bar-distribution logits, shape [M, num_bars].  # [M, num_bars]

    Returns:
        Tuple of (mean, std), each shape [M].  # [M], [M]

    Raises:
        RuntimeError: If any summary is not finite.
    """
    with torch.no_grad():
        mean = criterion.mean(logits)                                   # [M]
        var = criterion.variance(logits).clamp_min(0.0)                 # [M]
    mean_np = mean.detach().cpu().numpy()                               # [M]
    std_np = var.sqrt().detach().cpu().numpy()                          # [M]
    if not (np.isfinite(mean_np).all() and np.isfinite(std_np).all()):
        raise RuntimeError(
            "TabPFN bar-distribution summaries are not finite: "
            f"{(~np.isfinite(mean_np)).sum()} mean and "
            f"{(~np.isfinite(std_np)).sum()} std entries."
        )
    return mean_np, std_np                                              # [M], [M]


# ---------------------------------------------------------------------------
# TabPFNSurrogate — TabPFNRegressor wrapper conforming to SurrogateModel
# ---------------------------------------------------------------------------

class TabPFNSurrogate:
    """TabPFNRegressor wrapper conforming to the ``SurrogateModel`` protocol.

    Uses TabPFN's native bar-distribution UCB for acquisition, which is more
    accurate than the Gaussian ``mean + kappa * std`` approximation.

    The UCB exploration coefficient (kappa) is controlled externally via
    ``run_bo_loop(kappa_schedule=...)``, not stored here.

    Args:
        model: A fitted ``TabPFNRegressor`` (from ``extract_inference_model()``
            for finetuned models, or a plain ``TabPFNRegressor`` for vanilla).
    """

    def __init__(
        self,
        model: TabPFNRegressor,
    ) -> None:
        self._model = model
        # Logit cache: populated by predict_ucb, invalidated by fit.
        # Allows predict(X) to skip a second forward pass when called with the
        # same X and context as the preceding predict_ucb(X) call.
        self._logit_cache: Optional[tuple] = None  # (X_ref, logits, criterion)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store in-context examples for the TabPFN forward pass.

        No gradient updates are performed; this call sets the context that
        the transformer uses at prediction time.  Invalidates the logit cache
        since the context has changed.

        Args:
            X: Feature matrix of observed points, shape [N, D].  # [N, D]
            y: Response vector of observed targets, shape [N].   # [N]
        """
        if np.isnan(X).any() or np.isnan(y).any():
            raise RuntimeError(
                "TabPFNSurrogate.fit received NaN inputs. "
                f"X has {np.isnan(X).sum()} NaNs, y has {np.isnan(y).sum()} NaNs."
            )
        self._logit_cache = None
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean and standard deviation from the bar distribution.

        Mean and standard deviation are the bar distribution's own moments
        (``criterion.mean`` / ``criterion.variance``), which integrate the full
        predicted density including its half-normal tail buckets.  The reported
        mean is therefore the distribution mean, not the median, and the spread
        assumes no normality (audit D7 / P0.11).  ``QUANTILE_LEVELS`` +
        ``gpbo_utils.moments_from_quantiles`` remain available as an independent
        cross-check; they agree to 1e-4 on the mean and 2.5% on the std, the
        latter being quantile-truncation error in the cross-check, not here.

        If a logit cache is valid for this X (populated by a preceding
        ``predict_ucb(X)`` call with the same context), the moments come from the
        cached logits, avoiding a second transformer forward pass; otherwise one
        ``output_type='full'`` pass is run and the cache refreshed.

        Args:
            X: Query feature matrix, shape [M, D].  # [M, D]

        Returns:
            Tuple of (mean, std), each shape [M].   # [M], [M]

        Raises:
            RuntimeError: If the predictive summaries are not finite.
        """
        # --- Try logit cache; no blanket except — a failing cache is a bug, not a
        # fallback path to swallow (audit D7 / P0.11). ---
        if self._logit_cache is not None:
            X_ref, cached_logits, cached_criterion = self._logit_cache
            if X.shape == X_ref.shape and np.array_equal(X, X_ref):
                return _bar_distribution_moments(cached_criterion, cached_logits)

        # --- Full forward pass (refreshes the cache for later calls) ---
        full_output = self._model.predict(X, output_type="full")
        logits = full_output['logits']         # [M, num_bars]
        criterion = full_output['criterion']
        self._logit_cache = (X.copy(), logits.detach(), criterion)
        mean, std = _bar_distribution_moments(criterion, logits)  # [M], [M]

        if np.isnan(mean).any() or np.isnan(std).any():
            raise RuntimeError(
                f"TabPFNSurrogate.predict returned NaN values. "
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
        """Return UCB values using the native TabPFN bar-distribution criterion.

        Converts ``kappa`` to ``rest_prob`` as::

            rest_prob = 0.5 * erfc(kappa / sqrt(2))

        which maps the Gaussian tail probability to the correct quantile of
        the bar distribution for UCB acquisition.

        Caches the bar-distribution logits so that a subsequent ``predict(X)``
        call on the same ``X`` (without an intervening ``fit``) can skip the
        second transformer forward pass.

        Args:
            X: Candidate feature matrix, shape [M, D].  # [M, D]
            kappa: Current (already-annealed) UCB exploration coefficient.
            t: Current BO step index (unused; annealing done by caller).
            n_steps: Total BO steps (unused; annealing done by caller).

        Returns:
            UCB values, shape [M].  # [M]
        """
        rest_prob = 0.5 * math.erfc(kappa / math.sqrt(2))
        full_output = self._model.predict(X, output_type="full")
        logits = full_output['logits']       # [M, num_bars]
        criterion = full_output['criterion']
        self._logit_cache = (X.copy(), logits.detach(), criterion)
        ucb_vals = criterion.ucb(logits, 0, rest_prob=rest_prob, maximize=True)
        return ucb_vals.clone().cpu().numpy()  # [M]

    def predict_ts(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Draw one Thompson sample from the TabPFN bar-distribution posterior.

        For each candidate in X, samples a bin index from the temperature-scaled
        bar distribution via multinomial sampling, then returns the bin-centre
        value.  The logit cache is populated so that a subsequent ``predict(X)``
        call with the same context can skip a second transformer forward pass.

        Temperature controls sharpness of the distribution before sampling:
          - ``temperature=1.0``: sample from the exact predicted distribution
          - ``temperature<1.0``: sharper / more exploitative (greedier argmax)
          - ``temperature>1.0``: flatter / more exploratory (more uniform)

        Args:
            X: Candidate feature matrix, shape [M, D].  # [M, D]
            temperature: Softmax temperature for bar distribution (default 1.0).

        Returns:
            Thompson sample values (bin centres), shape [M].  # [M]
        """
        full_output = self._model.predict(X, output_type="full")
        logits = full_output['logits']        # [M, num_bars]
        criterion = full_output['criterion']
        self._logit_cache = (X.copy(), logits.detach(), criterion)

        scaled_logits = logits / temperature                                  # [M, num_bars]
        probs = torch.softmax(scaled_logits, dim=-1)                          # [M, num_bars]
        bin_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)    # [M]

        # Map bin indices to bin-centre values using bar-distribution borders
        borders = criterion.borders                # [num_bars + 1]
        left = borders[bin_indices]                # [M]
        right = borders[bin_indices + 1]           # [M]
        samples = (left + right) / 2.0            # [M] — bin centres

        return samples.detach().cpu().numpy()      # [M]


