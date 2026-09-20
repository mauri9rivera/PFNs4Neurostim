import math
import torch
from torch.distributions import Normal
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm as sp_norm
import numpy as np


def compute_ucb_kappa(t, n_steps, kappa_max, kappa_min, alpha: float = 0.5):
    """
    Cosine annealing of UCB exploration parameter kappa.

    At t=0: returns kappa_min + 2*alpha*(kappa_max - kappa_min).
    At t=n_steps: returns kappa_min (maximum exploitation).
    At t=n_steps/2: returns kappa_min + alpha*(kappa_max - kappa_min).

    With alpha=0.5 (default): range is exactly [kappa_min, kappa_max].
    With alpha<0.5: starts below kappa_max (more exploitative from start).
    With alpha>0.5: starts above kappa_max (more exploratory from start).

    Args:
        t: current step (0-indexed)
        n_steps: total number of BO steps (budget - n_init)
        kappa_max: reference kappa (equals starting value when alpha=0.5)
        kappa_min: final (minimum) kappa
        alpha: amplitude scale for cosine annealing (default 0.5)
    """
    if n_steps <= 0:
        return kappa_min
    return kappa_min + alpha * (kappa_max - kappa_min) * (1 + math.cos(math.pi * t / n_steps))


def _auto_kappa_max(
    d: int, n_iter: int, alpha: float = 2.5, kappa_floor: float = 3.0
) -> float:
    """Compute upper UCB kappa bound from GP-UCB theory scaling.

    k_max = max(alpha * sqrt(d * log(n_iter)), kappa_floor)

    Derived from the Srinivas et al. (2010) GP-UCB beta_t formula, rescaled to
    be dimension- and budget-aware. The kappa_floor prevents near-zero exploration
    when n_iter is very small.

    Example values (alpha=2.5, kappa_floor=3.0):
        NHP  (d=2, n_iter=95):  k_max ≈ 7.54
        Rat  (d=2, n_iter=27):  k_max ≈ 6.01

    Args:
        d: Input dimensionality.
        n_iter: Number of active BO steps (budget - n_init).
        alpha: Scale coefficient (default 2.5).
        kappa_floor: Minimum allowed value (default 3.0).

    Returns:
        Scalar kappa upper bound.
    """
    return max(alpha * np.sqrt(d * np.log(max(n_iter, 2))), kappa_floor)


def _auto_kappa_min(d: int, n_iter: int, beta: float = 0.2) -> float:
    """Compute lower UCB kappa bound from GP-UCB theory scaling.

    k_min = beta * sqrt(d * log(n_iter))

    Maintains a constant ratio k_max / k_min = alpha / beta = 12.5x (default)
    regardless of d and n_iter.

    Example values (beta=0.2):
        NHP  (d=2, n_iter=95):  k_min ≈ 0.60
        Rat  (d=2, n_iter=27):  k_min ≈ 0.48

    Args:
        d: Input dimensionality.
        n_iter: Number of active BO steps (budget - n_init).
        beta: Scale coefficient (default 0.2).

    Returns:
        Scalar kappa lower bound.
    """
    return beta * np.sqrt(d * np.log(max(n_iter, 2)))


def expected_improvement(model, likelihood, X_candidates, y_best, device):
    """
    Computes EI for the GP model on discrete candidates.
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        # Predictive posterior
        posterior = likelihood(model(X_candidates))
        mean = posterior.mean
        sigma = posterior.stddev
        
        # Avoid div by zero
        sigma = torch.clamp(sigma, min=1e-9)
        
        # EI Formula
        z = (mean - y_best) / sigma
        # Using PyTorch Normal distribution for cdf/pdf
        dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        ei = (mean - y_best) * dist.cdf(z) + sigma * dist.log_prob(z).exp()
        
    return ei

def moments_from_quantiles(
    quantiles: "np.ndarray",
    levels: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Estimate predictive mean and std by integrating a quantile function.

    Makes no normality assumption: for a random variable Y with quantile
    function q(p), E[Y] = ∫₀¹ q(p) dp and E[Y²] = ∫₀¹ q(p)² dp, both evaluated
    here by the trapezoid rule over the supplied levels.  This is the
    distribution-faithful replacement for reading the median as the mean
    (audit D7 / P0.11): for a skewed bar distribution the median and mean differ.

    The integral is truncated at the outermost supplied levels, so pass a dense,
    wide grid (e.g. ``np.linspace(0.005, 0.995, 199)``); the residual tail mass
    biases the variance slightly downward.

    Args:
        quantiles: Predicted quantile values, shape [L, M] — one row per level.
        levels: Quantile levels in (0, 1), shape [L], strictly increasing.

    Returns:
        Tuple of (mean, std), each shape [M], with std floored at 1e-9.

    Raises:
        ValueError: If shapes disagree, fewer than three levels are given, or
            levels are not strictly increasing within (0, 1).
        RuntimeError: If any predicted quantile is not finite.
    """
    q = np.asarray(quantiles, dtype=np.float64)      # [L, M]
    p = np.asarray(levels, dtype=np.float64)         # [L]
    if q.ndim != 2 or p.ndim != 1 or q.shape[0] != p.shape[0]:
        raise ValueError(
            f"quantiles must be [L, M] and levels [L]; got {q.shape} and {p.shape}."
        )
    if p.shape[0] < 3:
        raise ValueError(f"Need at least 3 quantile levels, got {p.shape[0]}.")
    if not (np.all(np.diff(p) > 0) and p[0] > 0.0 and p[-1] < 1.0):
        raise ValueError("levels must be strictly increasing and lie inside (0, 1).")
    if not np.isfinite(q).all():
        raise RuntimeError(
            f"moments_from_quantiles received {(~np.isfinite(q)).sum()} non-finite "
            "quantile values."
        )

    width = p[-1] - p[0]                              # mass covered by the grid
    mean = np.trapz(q, p, axis=0) / width             # [M]
    mean_sq = np.trapz(q ** 2, p, axis=0) / width     # [M]
    var = np.maximum(mean_sq - mean ** 2, 0.0)        # [M] — clip integration noise
    return mean, np.maximum(np.sqrt(var), 1e-9)       # [M], [M]


def std_from_quantiles(quantiles):
    """
    Estimate mean and std from quantile predictions.

    Averages three symmetric-pair sigma estimates (5/95, 10/90, 25/75)
    under a normal approximation for a more robust uncertainty estimate
    than using a single pair.

    Expects quantiles requested at levels [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95].

    Args:
        quantiles: np.ndarray of shape (7, n_samples) — one row per
                   level in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95].

    Returns:
        mean: np.ndarray of shape (n_samples,)
        std:  np.ndarray of shape (n_samples,), floored at 1e-9
    """
    # Symmetric pairs: (low_index, high_index, divisor)
    # Divisor = 2 * Φ⁻¹(upper_quantile) for the standard normal distribution.
    std_pairs = [
        (0, 6, 3.290),  # q05 / q95  — 2 * Φ⁻¹(0.95) = 3.290 (was 4.390, the q025/q975
                        # constant, which underestimated this pair's sigma by 25%)
        (1, 5, 2.564),  # q10 / q90
        (2, 4, 1.349),  # q25 / q75
    ]

    std_estimates = []
    for i_low, i_high, divisor in std_pairs:
        sigma = (quantiles[i_high] - quantiles[i_low]) / divisor
        std_estimates.append(sigma)

    mean = quantiles[3]  # median (index of 0.5)
    std = np.mean(std_estimates, axis=0)
    std = np.maximum(std, 1e-9)
    return mean, std


def expected_improvement_numpy(mean, std, y_best):
    """
    Compute Expected Improvement using numpy arrays (for TabPFN surrogate).

    Args:
        mean: np.ndarray of predicted means
        std: np.ndarray of predicted stds
        y_best: float, best observed value so far

    Returns:
        ei: np.ndarray of EI values
    """
    std = np.maximum(std, 1e-9)
    z = (mean - y_best) / std
    ei = (mean - y_best) * sp_norm.cdf(z) + std * sp_norm.pdf(z)
    return ei



