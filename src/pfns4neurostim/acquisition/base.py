"""Acquisition interface: state, scoring, and candidate selection.

Every acquisition type is a function

    score(surrogate, X_pool, state, rng, params) -> np.ndarray  # [N] higher is better

and :func:`acquire` turns scores into a query: already-observed sites are masked
out, the argmax is taken with a **random tie-break** (defect D6 of the 2026-09-16
TS review — the old code always took the lowest index, which biases every model
toward low-numbered electrodes on flat acquisition surfaces), and non-finite
surfaces raise instead of silently selecting index 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = ["BOState", "AcqResult", "acquire", "masked_argmax"]


@dataclass(frozen=True)
class BOState:
    """State of the BO loop at one acquisition step.

    Attributes:
        observed_indices: Sites queried so far, in order.
        observed_values: Their observed (noisy) responses.
        step: Zero-based acquisition step index.
        n_steps: Total acquisition steps in this run.
        n_dims: Search-space dimensionality (used by ``auto_dim`` schedules).
    """

    observed_indices: tuple[int, ...]
    observed_values: tuple[float, ...]
    step: int
    n_steps: int
    n_dims: int = 1

    @property
    def incumbent_value(self) -> float:
        """Best observed value so far; ``-inf`` before the first observation."""
        return float(max(self.observed_values)) if self.observed_values else float("-inf")


@dataclass(frozen=True)
class AcqResult:
    """Outcome of one acquisition step.

    Attributes:
        index: Selected site index.
        values: The acquisition surface, shape [N].
        params: Parameter values actually used (after any schedule), for logging.
    """

    index: int
    values: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)


def masked_argmax(
    values: np.ndarray,
    observed: tuple[int, ...] | list[int],
    rng: np.random.Generator,
    allowed: np.ndarray | None = None,
) -> int:
    """Pick the best unobserved candidate, breaking ties at random.

    Args:
        values: Acquisition surface, shape [N].
        observed: Indices already queried, which are excluded.
        rng: Seeded generator used for the tie-break.
        allowed: Optional boolean mask of selectable sites, shape [N]. Used by the
            K6 dropout knob, where the ground-truth map still spans every site but
            the optimizer may only query the surviving electrodes.

    Returns:
        The selected index.

    Raises:
        RuntimeError: If every candidate is observed, or the surface is entirely
            non-finite over the unobserved candidates.
    """
    values = np.asarray(values, dtype=np.float64)          # [N]
    mask = np.ones(values.shape[0], dtype=bool)            # [N]
    if allowed is not None:
        mask &= np.asarray(allowed, dtype=bool)
    if len(observed):
        mask[np.asarray(observed, dtype=int)] = False
    if not mask.any():
        raise RuntimeError("masked_argmax: every candidate has already been queried.")

    candidates = np.flatnonzero(mask & np.isfinite(values))
    if candidates.size == 0:
        raise RuntimeError(
            "masked_argmax: the acquisition surface is non-finite at every "
            "unobserved candidate; refusing to select a site by accident."
        )
    best = values[candidates].max()
    # Random tie-break (D6): the legacy argmax always took the lowest index,
    # which systematically favours low-numbered electrodes on flat surfaces.
    tied = candidates[values[candidates] == best]
    return int(tied[0] if tied.size == 1 else rng.choice(tied))


def acquire(
    score_fn: Callable[..., np.ndarray],
    surrogate: Any,
    X_pool: np.ndarray,
    state: BOState,
    rng: np.random.Generator,
    params: Any,
    allowed: np.ndarray | None = None,
) -> AcqResult:
    """Score every candidate and select the next query.

    Args:
        score_fn: The acquisition type's scoring function.
        surrogate: Fitted surrogate (adapter).
        X_pool: Candidate coordinates, shape [N, D].
        state: Loop state at this step.
        rng: Seeded generator.
        params: The acquisition type's params dataclass instance.
        allowed: Optional boolean mask of selectable sites, shape [N].

    Returns:
        The :class:`AcqResult` for this step.
    """
    values = np.asarray(score_fn(surrogate, X_pool, state, rng, params), dtype=np.float64)  # [N]
    if values.shape[0] != X_pool.shape[0]:
        raise RuntimeError(
            f"Acquisition returned {values.shape[0]} values for {X_pool.shape[0]} candidates."
        )
    index = masked_argmax(values, state.observed_indices, rng, allowed=allowed)
    return AcqResult(index=index, values=values, params=params.resolved(state))
