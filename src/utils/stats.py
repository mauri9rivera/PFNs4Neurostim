"""Statistical utilities: TOST equivalence testing and effect size computation.

Implements the A6 statistical backbone for Hypothesis A: paired Two One-Sided
Tests (TOST) that declare TabPFN+TS *equivalent* to GP within a pre-registered
margin, rather than merely "n.s. Wilcoxon." Also provides the calibration
helper that derives the margin from GP-vs-GP rep-to-rep variance before the
test is run.

Usage (from aggregate.py or standalone)::

    from utils.stats import tost_equivalence, compute_equivalence_margin, run_tost_on_results

    margin = compute_equivalence_margin(gp_results, metric='final_regret')
    tost_df = run_tost_on_results(results_dict, margin=margin)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Core TOST primitive
# ---------------------------------------------------------------------------

def tost_equivalence(
    a: np.ndarray,
    b: np.ndarray,
    margin: float,
    alpha: float = 0.05,
) -> dict[str, float | bool | int]:
    """Paired Two One-Sided Tests (TOST) for statistical equivalence.

    Declares a and b equivalent when the (1−2α)×100% CI of mean(a−b) lies
    entirely within [−margin, +margin].  The TOST p-value is max(p₁, p₂)
    where p₁ and p₂ are the one-sided paired t-test p-values.  Equivalence
    holds when tost_p < alpha.

    Pre-registration note: margin must be fixed *before* inspecting results.
    Use :func:`compute_equivalence_margin` on GP-only results to calibrate.

    Args:
        a: Values for condition A (e.g. TabPFN per-rep final regret), shape [N].
        b: Values for condition B (e.g. GP per-rep final regret), shape [N].
        margin: Pre-registered equivalence margin Δ (same units as a and b).
        alpha: One-sided significance level. Default 0.05 → 90% CI.

    Returns:
        Dict with keys:
          mean_diff (float): mean(a) − mean(b).
          ci_lo, ci_hi (float): (1−2α)×100% CI of the mean paired difference.
          tost_p (float): max(p_lower, p_upper); reject H0 of non-equivalence
                          when tost_p < alpha.
          wilcoxon_p (float): two-sided Wilcoxon signed-rank p (companion).
          effect_size (float): rank-biserial r = z/√N (Rosenthal, 1991).
          equivalent (bool): True when tost_p < alpha.
          n (int): number of paired observations.
          margin (float): the margin used.
          alpha (float): the alpha used.

    Raises:
        ValueError: If a and b have different lengths, margin ≤ 0, or n < 2.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    if len(a) != len(b):
        raise ValueError(
            f"a and b must have the same length, got {len(a)} vs {len(b)}."
        )
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}.")

    d = a - b  # [N] paired differences
    n = len(d)

    if n < 2:
        raise ValueError(f"Need at least 2 paired observations, got {n}.")

    mean_d = float(np.mean(d))
    se_d = float(stats.sem(d))  # std / sqrt(n)

    # (1 − 2·alpha)×100 % CI of mean paired difference
    ci_lo, ci_hi = stats.t.interval(
        1.0 - 2.0 * alpha, df=n - 1, loc=mean_d, scale=max(se_d, 1e-12)
    )

    # Two one-sided paired t-tests
    # H01: μ_d ≤ −Δ  →  reject in favour of mean > −Δ
    # H02: μ_d ≥ +Δ  →  reject in favour of mean < +Δ
    _, p_lower = stats.ttest_1samp(d, -margin, alternative='greater')
    _, p_upper = stats.ttest_1samp(d, margin, alternative='less')
    tost_p = float(max(p_lower, p_upper))

    # Wilcoxon signed-rank (non-parametric companion)
    try:
        _w_stat, wilcoxon_p_raw = stats.wilcoxon(d, alternative='two-sided')
        wilcoxon_p = float(wilcoxon_p_raw)
        n_nz = int(np.sum(d != 0))
        # Rank-biserial r = z / sqrt(N) (Rosenthal, 1991)
        if n_nz > 0 and wilcoxon_p < 1.0:
            z_approx = float(abs(stats.norm.ppf(wilcoxon_p / 2.0)))
            effect_size = z_approx / np.sqrt(n_nz)
        else:
            effect_size = 0.0
    except ValueError:
        wilcoxon_p = 1.0
        effect_size = 0.0

    return {
        'mean_diff': mean_d,
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
        'tost_p': tost_p,
        'wilcoxon_p': wilcoxon_p,
        'effect_size': float(effect_size),
        'equivalent': bool(tost_p < alpha),
        'n': n,
        'margin': float(margin),
        'alpha': alpha,
    }


# ---------------------------------------------------------------------------
# Margin calibration
# ---------------------------------------------------------------------------

def compute_equivalence_margin(
    gp_results: list[dict],
    metric: str = 'final_regret',
    scale_factor: float = 1.0,
) -> float:
    """Estimate the equivalence margin Δ from GP rep-to-rep variability.

    Computes the mean within-experiment standard deviation of the chosen
    metric across repetitions using GP results only.  The intuition: if GP
    itself shows ±σ variation across reps, declaring TabPFN equivalent within
    ±σ is a natural and honest pre-registered threshold.

    This must be called *before* comparing GP to TabPFN to preserve
    pre-registration integrity.

    Args:
        gp_results: List of GP result dicts from ``gp_baseline()``.  Each
            must contain ``'values'`` (list[list[float]], shape [n_reps,
            budget]) and ``'y_test'`` (np.ndarray) for ``'final_regret'``,
            or ``'r2'`` (list[float]) for ``'r2'``.
        metric: ``'final_regret'`` or ``'r2'``.
        scale_factor: Multiply the computed std by this factor to widen or
            narrow the margin (default 1.0 = one within-GP std).

    Returns:
        Estimated equivalence margin Δ (positive float).

    Raises:
        ValueError: If no usable GP results are found for the given metric.
    """
    per_exp_stds: list[float] = []
    for res in gp_results:
        if metric == 'final_regret':
            vals = _final_regret_array(res)
        elif metric == 'r2':
            vals = np.asarray(res.get('r2', []), dtype=float)
        else:
            raise ValueError(f"Unknown metric {metric!r}. Use 'final_regret' or 'r2'.")

        if len(vals) < 2:
            continue
        per_exp_stds.append(float(np.std(vals, ddof=1)))

    if not per_exp_stds:
        raise ValueError(f"No usable GP results found for metric '{metric}'.")

    return float(np.mean(per_exp_stds)) * scale_factor


# ---------------------------------------------------------------------------
# §17 — Predictive-interval calibration
# ---------------------------------------------------------------------------

def compute_calibration(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    quantiles: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray | float]:
    """Reliability of a surrogate's predictive intervals under a Gaussian model.

    For each nominal central-interval level ``q`` the empirical coverage is the
    fraction of test points whose true value falls inside the model's ``q``
    predictive interval ``mean ± z * std`` (``z = Φ⁻¹((1+q)/2)``). A perfectly
    calibrated model traces the diagonal (empirical = nominal). The expected
    calibration error (ECE) summarises the mean absolute deviation from it.

    Acquisition validity (UCB/TS) hinges on this: an over-confident surrogate
    under-explores and an under-confident one wastes budget, so §17 tests it
    directly on neurostim rather than assuming it. Applies equally to a GP
    posterior and to TabPFN's ``(mean, std)`` derived from bar-distribution
    quantiles (see :func:`utils.gpbo_utils.std_from_quantiles`).

    Args:
        y_true: Ground-truth responses, shape [M].
        mean: Predictive means, shape [M].
        std: Predictive standard deviations, shape [M]; floored internally.
        quantiles: Nominal central-interval levels in (0, 1). Defaults to
            ``[0.1, 0.2, …, 0.9]``.

    Returns:
        Dict with keys:
          nominal (np.ndarray): the requested levels.
          empirical (np.ndarray): observed coverage at each level.
          ece (float): mean absolute (empirical − nominal).
          overconfidence (float): mean signed (nominal − empirical); >0 means
                                  intervals are too narrow (over-confident).
          n (int): number of test points.

    Raises:
        ValueError: If array lengths differ or fewer than 2 points are given.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    mean = np.asarray(mean, dtype=float).ravel()
    std = np.maximum(np.asarray(std, dtype=float).ravel(), 1e-12)

    if not (len(y_true) == len(mean) == len(std)):
        raise ValueError(
            f"y_true, mean, std must share length; got "
            f"{len(y_true)}, {len(mean)}, {len(std)}."
        )
    if len(y_true) < 2:
        raise ValueError(f"Need at least 2 points for calibration, got {len(y_true)}.")

    if quantiles is None:
        quantiles = np.arange(0.1, 0.91, 0.1)
    quantiles = np.asarray(quantiles, dtype=float).ravel()

    z_scores = np.abs(y_true - mean) / std  # [M] — standardized abs residuals
    empirical = np.array([
        float(np.mean(z_scores <= stats.norm.ppf((1.0 + q) / 2.0)))
        for q in quantiles
    ])

    ece = float(np.mean(np.abs(empirical - quantiles)))
    overconfidence = float(np.mean(quantiles - empirical))

    return {
        'nominal': quantiles,
        'empirical': empirical,
        'ece': ece,
        'overconfidence': overconfidence,
        'n': int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# §17 — OOD placement → BO regret predictive link
# ---------------------------------------------------------------------------

def ood_regret_correlation(
    ood_scores: np.ndarray,
    regrets: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Spearman correlation between per-experiment OOD score and final regret.

    Tests whether a Hypothesis-B distributional OOD score (MMD / Wasserstein /
    Mahalanobis) for a (subject, EMG) pair predicts that pair's downstream BO
    regret — converting Hyp B from descriptive to *predictive*: a usable
    "when will TabPFN struggle?" diagnostic. A positive ρ means more-OOD
    channels incur higher regret.

    A percentile bootstrap over paired observations gives a 95% CI on ρ that
    is honest about the small number of channels.

    Args:
        ood_scores: Per-experiment OOD scores, shape [K].
        regrets: Per-experiment final BO regret (same ordering), shape [K].
        n_boot: Bootstrap resamples for the CI.
        seed: RNG seed for the bootstrap.

    Returns:
        Dict with keys ``spearman_rho``, ``p_value``, ``ci_lo``, ``ci_hi``,
        ``n`` (number of paired experiments).

    Raises:
        ValueError: If the arrays differ in length or fewer than 3 pairs exist.
    """
    a = np.asarray(ood_scores, dtype=float).ravel()
    b = np.asarray(regrets, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError(f"ood_scores and regrets must match; got {len(a)} vs {len(b)}.")
    if len(a) < 3:
        raise ValueError(f"Need at least 3 paired experiments, got {len(a)}.")

    res = stats.spearmanr(a, b)
    rho = float(res.statistic)
    p_value = float(res.pvalue)

    rng = np.random.default_rng(seed)
    n = len(a)
    boot_rhos: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.ptp(a[idx]) < 1e-12 or np.ptp(b[idx]) < 1e-12:
            continue  # degenerate resample — Spearman undefined
        r = stats.spearmanr(a[idx], b[idx]).statistic
        if np.isfinite(r):
            boot_rhos.append(float(r))

    if boot_rhos:
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    else:
        ci_lo, ci_hi = float('nan'), float('nan')

    return {
        'spearman_rho': rho,
        'p_value': p_value,
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
        'n': int(n),
    }


# ---------------------------------------------------------------------------
# Batch TOST over a combined results_dict
# ---------------------------------------------------------------------------

def run_tost_on_results(
    results_dict: dict,
    margin: float,
    metrics: Optional[list[str]] = None,
    gp_key: str = 'GP',
    tabpfn_key: Optional[str] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run paired TOST on all matching GP vs TabPFN (subject, emg) pairs.

    Matches experiments by ``(subject, emg)`` key, extracts per-rep metric
    values from each matched pair, and runs :func:`tost_equivalence` both
    per-experiment and pooled across all experiments (``label='All'``).

    Args:
        results_dict: ``{'GP': [result_dicts], 'TabPFN': [result_dicts], ...}``
            as returned by ``_load_combined_results_dict()`` in aggregate.py.
        margin: Pre-registered equivalence margin Δ.  See
            :func:`compute_equivalence_margin` to calibrate from GP results.
        metrics: Metrics to test.  Defaults to ``['final_regret', 'r2']``.
        gp_key: Key for GP results in results_dict (default ``'GP'``).
        tabpfn_key: Key for TabPFN results.  If ``None``, the first non-GP
            key is used.
        alpha: One-sided significance level (default 0.05 → 90% CI).

    Returns:
        DataFrame with columns: ``metric``, ``label``, ``mean_diff``,
        ``ci_lo``, ``ci_hi``, ``tost_p``, ``wilcoxon_p``, ``effect_size``,
        ``equivalent``, ``n``, ``margin``.

    Raises:
        ValueError: If gp_key is missing from results_dict, or no TabPFN key
            is resolvable.
    """
    if metrics is None:
        metrics = ['final_regret', 'r2']

    if gp_key not in results_dict:
        raise ValueError(
            f"gp_key {gp_key!r} not found in results_dict. "
            f"Keys: {list(results_dict.keys())}"
        )

    if tabpfn_key is None:
        other_keys = [k for k in results_dict if k != gp_key]
        if not other_keys:
            raise ValueError("No non-GP model found in results_dict.")
        tabpfn_key = other_keys[0]

    gp_list = results_dict[gp_key]
    pfn_list = results_dict[tabpfn_key]

    gp_index = {(r['subject'], r['emg']): r for r in gp_list}
    pfn_index = {(r['subject'], r['emg']): r for r in pfn_list}
    common_keys = sorted(set(gp_index) & set(pfn_index))

    rows: list[dict] = []
    for metric in metrics:
        all_a: list[float] = []
        all_b: list[float] = []

        for key in common_keys:
            a_vals, b_vals = _extract_metric_values(
                pfn_index[key], gp_index[key], metric
            )
            if len(a_vals) < 2:
                continue

            try:
                res = tost_equivalence(a_vals, b_vals, margin, alpha)
                rows.append({'metric': metric, 'label': f"S{key[0]}_EMG{key[1]}", **res})
            except ValueError:
                pass

            all_a.extend(a_vals.tolist())
            all_b.extend(b_vals.tolist())

        if len(all_a) >= 2:
            try:
                res = tost_equivalence(np.array(all_a), np.array(all_b), margin, alpha)
                rows.append({'metric': metric, 'label': 'All', **res})
            except ValueError:
                pass

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_metric_values(
    pfn_res: dict,
    gp_res: dict,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract matching per-rep metric arrays from a (TabPFN, GP) result pair.

    Args:
        pfn_res: Result dict for the TabPFN model.
        gp_res: Result dict for the GP model.
        metric: ``'final_regret'`` or ``'r2'``.

    Returns:
        Tuple ``(a, b)`` of 1-D arrays truncated to the minimum shared length.
        Returns ``(array([]), array([]))`` when data is missing.
    """
    if metric == 'final_regret':
        a = _final_regret_array(pfn_res)
        b = _final_regret_array(gp_res)
    elif metric == 'r2':
        a = np.asarray(pfn_res.get('r2', []), dtype=float)
        b = np.asarray(gp_res.get('r2', []), dtype=float)
    else:
        return np.array([]), np.array([])

    if len(a) == 0 or len(b) == 0:
        return np.array([]), np.array([])

    n = min(len(a), len(b))
    return a[:n], b[:n]


def _final_regret_array(res: dict) -> np.ndarray:
    """Compute per-rep normalized final simple regret from a result dict.

    Args:
        res: Result dict with ``'values'`` ([n_reps, budget]) and
            ``'y_test'`` (np.ndarray).

    Returns:
        1-D array of shape [n_reps].  Empty array if data is missing or
        response range is near zero.
    """
    if 'values' not in res or 'y_test' not in res:
        return np.array([])
    y_range = float(res['y_test'].max() - res['y_test'].min())
    if y_range < 1e-8:
        return np.array([])
    optimal = float(res['y_test'].max())
    running_best = np.maximum.accumulate(
        np.array(res['values']), axis=1
    )  # [n_reps, budget]
    return (optimal - running_best[:, -1]) / y_range  # [n_reps]
