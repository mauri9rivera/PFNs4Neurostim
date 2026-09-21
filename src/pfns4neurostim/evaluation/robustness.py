"""Robustness metrics for the stress sweeps (roadmap S9).

All four quantities are computed from the tidy sweep CSV, so they can be
recomputed without re-running any experiment:

* **breakdown point** — the first knob level (ordered by increasing severity) at
  which the model stops being statistically equivalent to the reference
  (TOST fails) *or* its mean regret exceeds the random-search baseline. Reported
  with a bootstrap CI over channels.
* **degradation AUC** — normalized area under the degradation curve; higher means
  worse degradation across the whole sweep, and it summarizes the curve without
  presupposing a breakdown.
* **CVaR at 10%** — mean of the worst decile of final regret across repetitions:
  tail risk, which a mean hides.
* **relative robustness** — signed gap to a reference model as a function of
  level, plus the level at which the sign flips.

TOST comes from the legacy ``utils.stats.tost_equivalence`` (paired, pre-declared
margin), reached through the same migration seam as the rest of the package.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "cvar",
    "degradation_auc",
    "breakdown_point",
    "relative_robustness",
    "bootstrap_ci",
]


def _noninferiority(a: np.ndarray, b: np.ndarray, margin: float, alpha: float = 0.05) -> dict[str, Any]:
    """One-sided paired non-inferiority test for a lower-is-better metric.

    H0: mean(a - b) >= margin (the model is worse than the reference by at least ``margin``);
    H1: mean(a - b) < margin. Unlike a two-sided equivalence test, a model that is clearly *better*
    than the reference passes.

    Args:
        a: Metric values of the model, shape [n].
        b: Paired values of the reference, shape [n].
        margin: Non-inferiority margin, in the metric's units.
        alpha: Significance level.

    Returns:
        ``p`` (one-sided p-value) and ``ok`` (True when non-inferiority is shown at ``alpha``).
    """
    from scipy import stats

    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)   # [n]
    if diff.size < 2 or float(np.std(diff, ddof=1)) == 0.0:
        return {"p": float("nan"), "ok": bool(diff.size > 0 and float(np.mean(diff)) < margin)}
    res = stats.ttest_1samp(diff, margin, alternative="less")
    return {"p": float(res.pvalue), "ok": bool(res.pvalue < alpha)}


def _tost(a: np.ndarray, b: np.ndarray, margin: float, alpha: float = 0.05) -> dict[str, Any]:
    """Run the paired TOST equivalence test through the legacy implementation.

    Args:
        a: Model values, shape [R].
        b: Reference values, shape [R].
        margin: Pre-registered equivalence margin, same units as the values.
        alpha: One-sided significance level.

    Returns:
        The legacy result dict (keys ``equivalent``, ``tost_p``, ``mean_diff``, ...).
    """
    from .stats import tost_equivalence  # noqa: PLC0415 - heavy scipy import, load on demand

    return tost_equivalence(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), margin, alpha)


def cvar(values: Sequence[float] | np.ndarray, *, alpha: float = 0.1) -> float:
    """Conditional value at risk: mean of the worst ``alpha`` fraction.

    For regret, "worst" is the *upper* tail, so this is the mean of the largest
    values. With few repetitions at least one value is always included.

    Args:
        values: Per-repetition metric values.
        alpha: Tail fraction, e.g. 0.1 for CVaR at 10%.

    Returns:
        Mean of the upper tail.

    Raises:
        ValueError: If ``values`` is empty or ``alpha`` is outside (0, 1].
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("cvar: no finite values.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"cvar: alpha must be in (0, 1], got {alpha}.")
    k = max(1, int(np.ceil(alpha * arr.size)))
    return float(np.mean(np.sort(arr)[-k:]))


def degradation_auc(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    """Normalized area under a degradation curve.

    The x-axis is normalized to [0, 1] over its own span before integrating, so
    the value is comparable across knobs whose levels live on different scales
    (dB for K2, a fraction for K5). Points are sorted by increasing x first.

    Args:
        x: Knob axis values (e.g. achieved SNR in dB), shape [L].
        y: Metric values (e.g. final regret), shape [L].

    Returns:
        Trapezoidal integral of ``y`` against the normalized x-axis.

    Raises:
        ValueError: If fewer than two finite points remain or the x-span is zero.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 2:
        raise ValueError("degradation_auc: need at least two finite points.")
    order = np.argsort(xa)
    xa, ya = xa[order], ya[order]
    span = float(xa[-1] - xa[0])
    if span <= 0.0:
        raise ValueError("degradation_auc: zero-width x-axis.")
    return float(np.trapz(ya, (xa - xa[0]) / span))


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Percentile bootstrap CI of a statistic over the given values.

    Args:
        values: Sample values (e.g. one breakdown level per channel).
        n_boot: Bootstrap resamples.
        level: Confidence level.
        rng: Seeded generator; a fixed default keeps figures reproducible.
        statistic: ``'mean'`` or ``'median'``.

    Returns:
        ``(lo, hi)``; ``(nan, nan)`` when fewer than two finite values exist.

    Raises:
        ValueError: On an unknown statistic.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return (float("nan"), float("nan"))
    if statistic not in ("mean", "median"):
        raise ValueError(f"bootstrap_ci: unknown statistic {statistic!r}.")
    gen = rng if rng is not None else np.random.default_rng(0)
    fn = np.mean if statistic == "mean" else np.median
    draws = gen.integers(0, arr.size, size=(n_boot, arr.size))
    stats_boot = fn(arr[draws], axis=1)                    # [n_boot]
    half = (1.0 - level) / 2.0
    return (
        float(np.quantile(stats_boot, half)),
        float(np.quantile(stats_boot, 1.0 - half)),
    )


def breakdown_point(
    per_level: dict[float, np.ndarray],
    reference_per_level: dict[float, np.ndarray],
    *,
    margin: float,
    severity_ascending: bool = True,
    random_per_level: dict[float, np.ndarray] | None = None,
    alpha: float = 0.05,
    test: str = "tost",
) -> dict[str, Any]:
    """Find the first knob level at which a model breaks down.

    Breakdown is declared at the mildest level (in severity order) where either

    1. the paired TOST against the reference model fails at ``margin``, or
    2. the model's mean metric is worse than the random-search baseline
       (when ``random_per_level`` is provided).

    Args:
        per_level: Level -> per-repetition metric values for the model.
        reference_per_level: Level -> per-repetition values for the reference
            (normally ``gp_mll``), paired with ``per_level`` by position.
        margin: Pre-registered TOST equivalence margin (range-normalized units).
        severity_ascending: True when a larger level means more stress. The K2
            sweep is evaluated against achieved SNR, which *decreases* with
            severity, so the caller passes False in that case.
        random_per_level: Level -> per-repetition values for random search.
        alpha: Test significance level.
        test: ``'tost'`` (two-sided equivalence, the original pre-registered rule) or ``'noninferiority'``
            (one-sided: the model is not worse than the reference by more than ``margin``). Two-sided
            equivalence also fails for a model that is clearly *better*, which makes it degenerate as a
            breakdown rule when a model beats the reference; non-inferiority does not (2026-09-21).

    Returns:
        ``breakdown_level`` (NaN when the model never breaks down),
        ``breakdown_reason`` (``'tost'``, ``'noninferiority'``, ``'random'`` or ``'none'``), and
        ``levels_tested`` / ``tost_p`` / ``equivalent`` per level for the audit
        trail behind the figure.

    Raises:
        ValueError: If the two level sets disagree or ``test`` is unknown.
    """
    if test not in ("tost", "noninferiority"):
        raise ValueError(f"breakdown_point: unknown test {test!r}.")
    levels = sorted(per_level, reverse=not severity_ascending)
    if set(levels) != set(reference_per_level):
        raise ValueError(
            "breakdown_point: model and reference level sets differ "
            f"({sorted(per_level)} vs {sorted(reference_per_level)})."
        )

    tested: list[float] = []
    tost_p: list[float] = []
    equivalent: list[bool] = []
    breakdown: float = float("nan")
    reason = "none"

    for level in levels:
        a = np.asarray(per_level[level], dtype=np.float64)
        b = np.asarray(reference_per_level[level], dtype=np.float64)
        n = min(a.size, b.size)
        tested.append(float(level))
        if n < 2:
            tost_p.append(float("nan"))
            equivalent.append(False)
            continue
        if test == "tost":
            res = _tost(a[:n], b[:n], margin=margin, alpha=alpha)
            tost_p.append(float(res["tost_p"]))
            is_equiv = bool(res["equivalent"])
        else:
            res = _noninferiority(a[:n], b[:n], margin=margin, alpha=alpha)
            tost_p.append(float(res["p"]))
            is_equiv = bool(res["ok"])
        equivalent.append(is_equiv)

        worse_than_random = False
        if random_per_level is not None and level in random_per_level:
            rnd = np.asarray(random_per_level[level], dtype=np.float64)
            if rnd.size:
                worse_than_random = float(np.mean(a)) > float(np.mean(rnd))

        if np.isnan(breakdown) and (not is_equiv or worse_than_random):
            breakdown = float(level)
            reason = "random" if worse_than_random and is_equiv else test

    return {
        "breakdown_level": breakdown,
        "breakdown_reason": reason,
        "levels_tested": tested,
        "tost_p": tost_p,
        "equivalent": equivalent,
        "margin": float(margin),
    }


def relative_robustness(
    x: Sequence[float] | np.ndarray,
    model_curve: Sequence[float] | np.ndarray,
    reference_curve: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Signed gap to a reference model across the sweep, and its crossing point.

    Args:
        x: Knob axis values, shape [L].
        model_curve: Mean metric per level for the model, shape [L].
        reference_curve: Mean metric per level for the reference, shape [L].

    Returns:
        ``gap_mean`` (mean of model minus reference; negative is better for
        regret), ``gap_max``, and ``crossing_x`` — the linearly interpolated
        x at which the gap first changes sign, NaN when it never does.
    """
    xa = np.asarray(x, dtype=np.float64)
    ma = np.asarray(model_curve, dtype=np.float64)
    ra = np.asarray(reference_curve, dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ma) & np.isfinite(ra)
    xa, gap = xa[ok], (ma - ra)[ok]
    if xa.size == 0:
        return {"gap_mean": float("nan"), "gap_max": float("nan"), "crossing_x": float("nan")}
    order = np.argsort(xa)
    xa, gap = xa[order], gap[order]

    crossing = float("nan")
    signs = np.sign(gap)
    zeros = np.flatnonzero(signs == 0)
    if zeros.size:
        # A gap that touches zero exactly *is* the crossing; no interpolation needed.
        crossing = float(xa[zeros[0]])
    else:
        for i in range(1, xa.size):
            if signs[i - 1] != signs[i]:
                # Linear interpolation of the zero crossing between the two levels.
                x0, x1, g0, g1 = xa[i - 1], xa[i], gap[i - 1], gap[i]
                crossing = float(x0 - g0 * (x1 - x0) / (g1 - g0))
                break

    return {
        "gap_mean": float(np.mean(gap)),
        "gap_max": float(np.max(np.abs(gap))),
        "crossing_x": crossing,
    }
