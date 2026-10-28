"""Closed-form EI / PI / UCB over TabPFN's bar distribution (the ``FullSupportBarDistribution`` forms).

TabPFN's regression head is a distribution over 5000 buckets (``FullSupportBarDistribution``): uniform inside every interior bucket,
half-normal tails in the two outer ones. The criterion ships closed-form acquisition integrals over that distribution
(``ei``, ``pi``) and a quantile (``ucb``). This module is the one place those are called, with two repairs:

* **Zero-width buckets.** The TabPFN-2.5 checkpoint's 5001 borders contain a duplicate, so one bucket has width 0 and
  ``criterion.ei`` divides its improvement integral by that width: 0 / 0, and the sum over bars is NaN at *every* candidate
  (measured with tabpfn 6.3.2 and 9.0.0). A zero-width bucket is a point mass at its border, so :func:`merge_zero_width_buckets`
  adds its probability to the next positive-width bucket and drops the duplicate border; no other bucket changes.
* **UCB quantile in float64.** ``criterion.ucb`` calls ``icdf``, which builds its level tensor in float32; a level such as
  ``1 - 3e-7`` (kappa = 5) is then rounded and can select a zero-mass bucket. The repo's float64 :func:`bar_quantiles` is used.

UCB here is the ``Phi(kappa)`` quantile of the predictive (``rest_prob = Phi(-kappa)``, the mapping ``TabPFNSurrogate.predict_ucb`` has
always used), **not** ``mean + kappa * sd``. For large kappa that level can fall inside the outer bucket, whose extent is a convention
of the criterion rather than a belief of the network: :func:`outer_bucket_share` measures how often.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.special import ndtr

__all__ = [
    "ACQUISITION_FORMS",
    "DEFAULT_ACQUISITION_FORM",
    "BAR_ACQUISITIONS",
    "merge_zero_width_buckets",
    "bar_ei",
    "bar_pi",
    "bar_ucb",
    "outer_bucket_share",
]

#: ``bars``: the closed forms of this module; ``moments``: Gaussian EI / PI and ``mean + kappa sd`` from the two moments.
ACQUISITION_FORMS: tuple[str, ...] = ("bars", "moments")
#: Form used by a bar-capable surrogate unless its model params say otherwise (switched from ``moments`` on 2026-10-27, user).
DEFAULT_ACQUISITION_FORM: str = "bars"
#: Acquisition types that have a bar form.
BAR_ACQUISITIONS: tuple[str, ...] = ("ei", "pi", "ucb")
#: Largest quantile level handed to :func:`bar_quantiles` (float64 resolves ``1 - 1e-12``).
MAX_LEVEL: float = 1.0 - 1e-12


def merge_zero_width_buckets(criterion: Any, logits: torch.Tensor) -> tuple[Any, torch.Tensor]:
    """Merge every zero-width bucket into the next positive-width one.

    Args:
        criterion: ``FullSupportBarDistribution`` the logits belong to.
        logits: Bar logits (or log-probabilities), shape [N, bars].  # [N, bars]

    Returns:
        ``(repaired criterion, log-probabilities over its merged bars [N, bars'])``; the input is returned unchanged
        (as log-probabilities) when no bucket has zero width.

    Raises:
        ValueError: If a bucket has negative width, an *edge* bucket has zero width (its half-normal tail would be
            undefined), or the logits do not match the criterion's bars.
    """
    borders = criterion.borders.detach()                                           # [bars + 1]
    widths = borders[1:] - borders[:-1]                                            # [bars]
    if logits.shape[-1] != widths.numel():
        raise ValueError(f"{logits.shape[-1]} logits for a criterion with {widths.numel()} bars.")
    if bool((widths < 0).any()):
        raise ValueError("bar borders must be non-decreasing.")
    positive = widths > 0                                                          # [bars]
    if not (bool(positive[0]) and bool(positive[-1])):
        raise ValueError("an edge bucket of the bar distribution has zero width; it cannot be merged.")
    probs = torch.softmax(logits.to(borders.device, torch.float64), -1)            # [N, bars]
    if bool(positive.all()):
        return criterion, probs.log().to(logits.dtype)
    from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution  # noqa: PLC0415

    keep = torch.nonzero(positive).flatten()                                       # [bars']
    target = torch.searchsorted(keep, torch.arange(widths.numel(), device=keep.device))   # [bars]
    merged = torch.zeros(probs.shape[0], keep.numel(), dtype=probs.dtype, device=probs.device)   # [N, bars']
    merged.index_add_(1, target, probs)
    repaired = FullSupportBarDistribution(torch.cat([borders[keep], borders[-1:]]))
    return repaired, merged.log().to(logits.dtype)


def bar_ei(criterion: Any, logits: torch.Tensor, incumbent: float) -> np.ndarray:
    """Expected improvement ``E[(y - incumbent)+]`` of the bar distribution at every candidate.

    Args:
        criterion: ``FullSupportBarDistribution`` of the fit.
        logits: Bar logits at the candidates, shape [N, bars].  # [N, bars]
        incumbent: Value to improve on, in the model's y space.

    Returns:
        EI per candidate, shape [N], float64.

    Raises:
        RuntimeError: If any value is not finite.
    """
    crit, lp = merge_zero_width_buckets(criterion, logits)
    with torch.no_grad():
        out = crit.ei(lp, float(incumbent)).double().cpu().numpy()                  # [N]
    if not np.isfinite(out).all():
        raise RuntimeError(f"bar_ei: {int((~np.isfinite(out)).sum())} of {out.size} candidates are not finite.")
    return out


def bar_pi(criterion: Any, logits: torch.Tensor, incumbent: float) -> np.ndarray:
    """Probability of improvement ``P(y > incumbent)`` of the bar distribution at every candidate.

    Args:
        criterion: ``FullSupportBarDistribution`` of the fit.
        logits: Bar logits at the candidates, shape [N, bars].  # [N, bars]
        incumbent: Value to improve on, in the model's y space.

    Returns:
        PI per candidate, shape [N], float64.

    Raises:
        RuntimeError: If any value is not finite.
    """
    crit, lp = merge_zero_width_buckets(criterion, logits)
    with torch.no_grad():
        out = crit.pi(lp, float(incumbent)).double().cpu().numpy()                  # [N]
    if not np.isfinite(out).all():
        raise RuntimeError(f"bar_pi: {int((~np.isfinite(out)).sum())} of {out.size} candidates are not finite.")
    return out


def ucb_level(kappa: float) -> float:
    """Quantile level of the bar UCB: ``Phi(kappa)``, clipped to ``(0, 1 - 1e-12)``.

    Args:
        kappa: Exploration coefficient.

    Returns:
        The level.
    """
    return float(min(max(ndtr(kappa), 1e-12), MAX_LEVEL))


def bar_ucb(criterion: Any, logits: torch.Tensor, kappa: float) -> np.ndarray:
    """The ``Phi(kappa)`` quantile of the bar distribution at every candidate (float64 quantile function).

    Args:
        criterion: ``FullSupportBarDistribution`` of the fit.
        logits: Bar logits at the candidates, shape [N, bars].  # [N, bars]
        kappa: Exploration coefficient (``rest_prob = Phi(-kappa)``).

    Returns:
        UCB per candidate, shape [N], float64.
    """
    from .tabpfn import bar_quantiles  # noqa: PLC0415 - imports tabpfn; the registry must load in envs without it

    return bar_quantiles(criterion, logits, np.array([ucb_level(kappa)]))[:, 0]    # [N]


def outer_bucket_share(criterion: Any, logits: torch.Tensor, kappa: float) -> float:
    """Share of candidates whose UCB quantile lies inside an outer bucket.

    There the value is set by the criterion's finite bucket width (60-260 predictive SDs), not by the network's belief.

    Args:
        criterion: ``FullSupportBarDistribution`` of the fit.
        logits: Bar logits at the candidates, shape [N, bars].  # [N, bars]
        kappa: Exploration coefficient.

    Returns:
        A share in [0, 1].
    """
    probs = torch.softmax(logits.to(criterion.borders.device, torch.float64), -1)   # [N, bars]
    level = ucb_level(kappa)
    upper = level > 1.0 - probs[:, -1]                                              # level above the cumulative mass below the last bucket
    lower = level < probs[:, 0]
    return float((upper | lower).double().mean())
