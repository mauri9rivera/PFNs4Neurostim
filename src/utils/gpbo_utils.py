import math
import torch
from torch.distributions import Normal
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm as sp_norm
import numpy as np


def compute_ucb_kappa(t, n_steps, kappa_0, kappa_min):
    """
    Cosine annealing of UCB exploration parameter kappa.

    At t=0: returns kappa_0 (maximum exploration).
    At t=n_steps: returns kappa_min (maximum exploitation).
    At t=n_steps/2: returns (kappa_0 + kappa_min) / 2.

    Args:
        t: current step (0-indexed)
        n_steps: total number of BO steps (budget - n_init)
        kappa_0: initial (maximum) kappa
        kappa_min: final (minimum) kappa
    """
    if n_steps <= 0:
        return kappa_min
    return kappa_min + 0.5 * (kappa_0 - kappa_min) * (1 + math.cos(math.pi * t / n_steps))


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
        (0, 6, 4.390),  # q025 / q975
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



