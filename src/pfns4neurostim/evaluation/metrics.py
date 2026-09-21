"""Metrics for BO runs, in shared range-normalized units.

**P0.9.** Every regret is divided by the channel's ground-truth range, so it is
invariant to any affine rescaling of the responses and comparable across models,
datasets and stress levels. The three co-primary regrets (user decision
2026-09-16) are:

* ``recommended_regret`` — the site the surrogate *recommends* at the end,
  ``argmax`` of its posterior mean over the whole pool. This is the quantity a
  clinician would act on.
* ``best_queried_regret`` — the best ground-truth value actually visited.
* ``cumulative_regret`` — summed instantaneous regret over the queried sites.

Calibration (roadmap S9) adds interval coverage, a regression ECE, Gaussian NLL
and CRPS, none of which the legacy ``evaluate_optimization`` ever recorded.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "regret_metrics",
    "r2_score",
    "exploration_score",
    "surrogate_accuracy",
    "identification_metrics",
    "calibration_metrics",
]

#: Nominal levels at which interval coverage is evaluated for the regression ECE.
_ECE_LEVELS: tuple[float, ...] = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2))


def regret_metrics(
    y_gt: np.ndarray,
    observed_indices: list[int] | np.ndarray,
    recommended_index: int,
) -> dict[str, float]:
    """Compute the three co-primary regrets in units of the GT range.

    Args:
        y_gt: Ground-truth response per site, shape [N].
        observed_indices: Site indices queried during the run, in order.
        recommended_index: Site the surrogate recommends after the last step.

    Returns:
        ``recommended_regret``, ``best_queried_regret``, ``cumulative_regret``.

    Raises:
        RuntimeError: If the GT range is degenerate or a regret is non-finite.
    """
    y_gt = np.asarray(y_gt, dtype=np.float64)                 # [N]
    idx = np.asarray(observed_indices, dtype=int)             # [T]
    gt_range = float(np.max(y_gt) - np.min(y_gt))
    if not np.isfinite(gt_range) or gt_range <= 0.0:
        raise RuntimeError(f"regret_metrics: degenerate ground-truth range {gt_range}.")
    y_star = float(np.max(y_gt))

    out = {
        "recommended_regret": (y_star - float(y_gt[int(recommended_index)])) / gt_range,
        "best_queried_regret": (y_star - float(np.max(y_gt[idx]))) / gt_range,
        "cumulative_regret": float(np.sum(y_star - y_gt[idx])) / gt_range,
    }
    for key, value in out.items():
        if not np.isfinite(value):
            raise RuntimeError(f"regret_metrics: {key} is non-finite ({value}).")
    return out


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination of a prediction over the whole pool.

    Args:
        y_true: Ground-truth responses, shape [N].
        y_pred: Predicted responses on the same sites, shape [N].

    Returns:
        R-squared; NaN only when the ground truth is constant.

    Raises:
        RuntimeError: If ``y_pred`` contains non-finite values.
    """
    y_true = np.asarray(y_true, dtype=np.float64)   # [N]
    y_pred = np.asarray(y_pred, dtype=np.float64)   # [N]
    if not np.isfinite(y_pred).all():
        raise RuntimeError(f"r2_score: y_pred has {int((~np.isfinite(y_pred)).sum())} non-finite values.")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def exploration_score(y_raw: np.ndarray, recommended_index: int) -> float:
    """Exploration score: true response at the recommended site over the true maximum.

    Follows the exploration performance of the autonomous-optimization work of
    Bonizzato et al. as specified by the user (2026-09-21): the response at the
    site the surrogate recommends (argmax of its posterior mean) divided by the
    best achievable response. The ratio is only meaningful in **raw** response
    units (non-negative), not in standardized ones, where the maximum can sit
    near or below zero. 1.0 means the optimum was recommended.

    Args:
        y_raw: Ground-truth response per site in raw (unstandardized) units, shape [N].
        recommended_index: Recommended site.

    Returns:
        The score, in [0, 1] for a non-negative response.

    Raises:
        RuntimeError: If the raw maximum is not positive.
    """
    y_raw = np.asarray(y_raw, dtype=np.float64)   # [N]
    y_max = float(np.max(y_raw))
    if not np.isfinite(y_max) or y_max <= 0.0:
        raise RuntimeError(
            f"exploration_score: the raw response maximum is {y_max}; the score needs raw, "
            "non-negative units (was the standardized response passed by mistake?)."
        )
    return float(y_raw[int(recommended_index)]) / y_max


def surrogate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R-squared and Spearman correlation of the surrogate's final prediction.

    Args:
        y_true: Ground-truth responses, shape [N].
        y_pred: Predicted responses on the same sites, shape [N].

    Returns:
        ``r2`` and ``spearman``. Spearman is NaN only when a vector is constant,
        in which case the coefficient is genuinely undefined.

    Raises:
        RuntimeError: If ``y_pred`` contains non-finite values.
    """
    y_true = np.asarray(y_true, dtype=np.float64)   # [N]
    y_pred = np.asarray(y_pred, dtype=np.float64)   # [N]
    if not np.isfinite(y_pred).all():
        raise RuntimeError(
            f"surrogate_accuracy: y_pred has {int((~np.isfinite(y_pred)).sum())} non-finite values."
        )
    r2 = r2_score(y_true, y_pred)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        rho = float("nan")
    else:
        rho = float(stats.spearmanr(y_true, y_pred).statistic)
    return {"r2": float(r2), "spearman": rho}


def identification_metrics(
    y_gt: np.ndarray,
    recommended_index: int,
    ch2xy: np.ndarray,
    *,
    top_k: int = 3,
) -> dict[str, float]:
    """Did the run identify the optimum, and how far off was it?

    Args:
        y_gt: Ground-truth responses, shape [N].
        recommended_index: Recommended site index.
        ch2xy: Integer grid coordinates per site, shape [N, D].
        top_k: k for the top-k hit rate.

    Returns:
        ``top1_hit``, ``top3_hit`` (0/1 for a single run; averaged over reps
        later) and ``opt_distance`` in electrode pitch units.
    """
    y_gt = np.asarray(y_gt, dtype=np.float64)          # [N]
    coords = np.asarray(ch2xy, dtype=np.float64)       # [N, D]
    order = np.argsort(y_gt)[::-1]                     # [N], best first
    rec = int(recommended_index)
    best = int(order[0])
    distance = float(np.linalg.norm(coords[rec] - coords[best]))
    return {
        "top1_hit": float(rec == best),
        "top3_hit": float(rec in set(order[:top_k].tolist())),
        "opt_distance": distance,
    }


def calibration_metrics(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    gt_range: float,
) -> dict[str, float]:
    """Interval coverage, regression ECE, Gaussian NLL and CRPS.

    The predictive distribution is summarized as ``N(mu, sigma^2)``, which is
    exact for the GP and a Gaussian summary of the PFN's bar distribution. NLL is
    in nats; CRPS is divided by ``gt_range`` so it shares the regret units.

    Args:
        y_true: Ground-truth responses, shape [N].
        mu: Predictive means, shape [N].
        sigma: Predictive standard deviations, shape [N].
        gt_range: Ground-truth range of the channel, for CRPS normalization.

    Returns:
        ``coverage_50``, ``coverage_90``, ``ece``, ``nll``, ``crps``.

    Raises:
        RuntimeError: On non-finite inputs or non-positive sigma.
    """
    y_true = np.asarray(y_true, dtype=np.float64)   # [N]
    mu = np.asarray(mu, dtype=np.float64)           # [N]
    sigma = np.asarray(sigma, dtype=np.float64)     # [N]
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all()):
        raise RuntimeError("calibration_metrics: non-finite predictive summaries.")
    if np.any(sigma <= 0.0):
        raise RuntimeError(
            f"calibration_metrics: {int((sigma <= 0).sum())} non-positive predictive "
            "standard deviations; NLL and coverage would be undefined."
        )
    if not np.isfinite(gt_range) or gt_range <= 0.0:
        raise RuntimeError(f"calibration_metrics: degenerate gt_range {gt_range}.")

    z = (y_true - mu) / sigma                                        # [N]

    def _coverage(level: float) -> float:
        """Empirical coverage of the central interval at ``level``."""
        half = float(stats.norm.ppf(0.5 + level / 2.0))
        return float(np.mean(np.abs(z) <= half))

    coverages = np.array([_coverage(p) for p in _ECE_LEVELS])        # [L]
    ece = float(np.mean(np.abs(coverages - np.asarray(_ECE_LEVELS))))

    nll = float(np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * z**2))

    # Gaussian CRPS closed form (Gneiting & Raftery 2007, Eq. 21).
    crps = float(
        np.mean(
            sigma
            * (
                z * (2.0 * stats.norm.cdf(z) - 1.0)
                + 2.0 * stats.norm.pdf(z)
                - 1.0 / np.sqrt(np.pi)
            )
        )
    ) / gt_range

    out = {
        "coverage_50": _coverage(0.5),
        "coverage_90": _coverage(0.9),
        "ece": ece,
        "nll": nll,
        "crps": crps,
    }
    for key, value in out.items():
        if not np.isfinite(value):
            raise RuntimeError(f"calibration_metrics: {key} is non-finite ({value}).")
    return out
