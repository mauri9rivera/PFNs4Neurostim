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
