"""Acquisition interface: state, scoring, and candidate selection.

Every acquisition type is a function

    score(surrogate, X_pool, state, rng, params) -> np.ndarray  # [N] higher is better

and :func:`acquire` turns scores into a query. **Every site stays selectable at
every step, including sites already observed** (decision 2026-09-21): neural
responses are noisy, so re-querying a promising configuration is how the
optimizer narrows its posterior there and exploits (a failed electrode under
the K6 knob is no exception: the optimizer must notice it from the data). The argmax is taken with a **random tie-break** (defect D6 of the
2026-09-16 TS review — the old code always took the lowest index, which biases
every model toward low-numbered electrodes on flat acquisition surfaces), and
non-finite surfaces raise instead of silently selecting index 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = ["BOState", "AcqResult", "acquire", "select_argmax"]


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


def select_argmax(
    values: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Pick the best selectable candidate, breaking ties at random.

    Already-observed sites are **not** excluded: re-querying a site draws a fresh
    noisy trial, which is how the optimizer averages out response noise.

    Args:
        values: Acquisition surface, shape [N].
        rng: Seeded generator used for the tie-break.

    Returns:
        The selected index.

    Raises:
        RuntimeError: If the surface is non-finite everywhere.
    """
    values = np.asarray(values, dtype=np.float64)          # [N]
    candidates = np.flatnonzero(np.isfinite(values))
    if candidates.size == 0:
        raise RuntimeError(
            "select_argmax: the acquisition surface is non-finite at every "
            "candidate; refusing to select a site by accident."
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
) -> AcqResult:
    """Score every candidate and select the next query.

    Args:
        score_fn: The acquisition type's scoring function.
        surrogate: Fitted surrogate (adapter).
        X_pool: Candidate coordinates, shape [N, D].
        state: Loop state at this step.
        rng: Seeded generator.
        params: The acquisition type's params dataclass instance.

    Returns:
        The :class:`AcqResult` for this step.
    """
    values = np.asarray(score_fn(surrogate, X_pool, state, rng, params), dtype=np.float64)  # [N]
    if values.shape[0] != X_pool.shape[0]:
        raise RuntimeError(
            f"Acquisition returned {values.shape[0]} values for {X_pool.shape[0]} candidates."
        )
    index = select_argmax(values, rng)
    return AcqResult(index=index, values=values, params=params.resolved(state))
