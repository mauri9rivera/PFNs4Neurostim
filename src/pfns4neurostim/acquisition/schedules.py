"""Parameter schedules for acquisition functions (P0.2 ``schedules:`` block).

A schedule anneals one named parameter of an acquisition type over the BO steps::

    acquisition:
      type: ucb
      params: {kappa: 2.0}
      schedules:
        kappa: {kind: cosine, start: 7.5, end: 0.6}

Four kinds are available: ``constant``, ``linear``, ``cosine``, and ``auto_dim``,
the last reproducing the dimensionality-aware kappa bounds of the legacy
``utils.gpbo_utils`` (alpha = 2.5, beta = 0.2, floor 3.0) as named arguments
rather than hidden constants.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = ["Schedule", "build_schedule", "SCHEDULE_KINDS"]

#: Constants of the legacy auto-kappa rule, promoted to named schedule arguments.
_AUTO_ALPHA: float = 2.5
_AUTO_BETA: float = 0.2
_AUTO_FLOOR: float = 3.0

SCHEDULE_KINDS: tuple[str, ...] = ("constant", "linear", "cosine", "auto_dim")


@dataclass(frozen=True)
class Schedule:
    """One parameter schedule.

    Attributes:
        kind: One of :data:`SCHEDULE_KINDS`.
        start: Value at step 0 (ignored by ``auto_dim``, which derives it).
        end: Value at the last step (ignored by ``auto_dim``).
        alpha: ``auto_dim`` upper-bound coefficient.
        beta: ``auto_dim`` lower-bound coefficient.
        floor: ``auto_dim`` minimum for the upper bound.
    """

    kind: str
    start: float = 0.0
    end: float = 0.0
    alpha: float = _AUTO_ALPHA
    beta: float = _AUTO_BETA
    floor: float = _AUTO_FLOOR

    def __post_init__(self) -> None:
        """Reject unknown schedule kinds rather than silently holding constant."""
        if self.kind not in SCHEDULE_KINDS:
            raise ValueError(
                f"Unknown schedule kind {self.kind!r}; available: {list(SCHEDULE_KINDS)}."
            )

    def value(self, step: int, n_steps: int, *, n_dims: int = 1) -> float:
        """Evaluate the schedule at one BO step.

        Args:
            step: Zero-based acquisition step index.
            n_steps: Total number of acquisition steps.
            n_dims: Search-space dimensionality, used by ``auto_dim``.

        Returns:
            The parameter value at this step.
        """
        if n_steps <= 1:
            return float(self.start if self.kind != "auto_dim" else self._auto_max(n_dims, n_steps))
        frac = min(max(step / (n_steps - 1), 0.0), 1.0)

        if self.kind == "constant":
            return float(self.start)
        if self.kind == "linear":
            return float(self.start + frac * (self.end - self.start))
        if self.kind == "cosine":
            # Cosine anneal from start down to end, flat at both extremes.
            return float(self.end + 0.5 * (self.start - self.end) * (1.0 + math.cos(math.pi * frac)))

        # auto_dim: cosine anneal between dimensionality-derived bounds.
        hi = self._auto_max(n_dims, n_steps)
        lo = self._auto_min(n_dims, n_steps)
        return float(lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * frac)))

    def _auto_max(self, n_dims: int, n_steps: int) -> float:
        """Upper kappa bound: grows with dimensionality, floored.

        Args:
            n_dims: Search-space dimensionality.
            n_steps: Total acquisition steps.

        Returns:
            The upper bound.
        """
        return float(max(self.floor, self.alpha * math.sqrt(max(n_dims, 1))))

    def _auto_min(self, n_dims: int, n_steps: int) -> float:
        """Lower kappa bound: shrinks as the budget grows.

        Args:
            n_dims: Search-space dimensionality.
            n_steps: Total acquisition steps.

        Returns:
            The lower bound.
        """
        return float(self.beta * math.sqrt(max(n_dims, 1) / max(n_steps, 1)))


def build_schedule(spec: dict[str, Any] | None) -> Schedule | None:
    """Build a :class:`Schedule` from its config mapping.

    Args:
        spec: Mapping with a ``kind`` key plus that kind's arguments, or None.

    Returns:
        The schedule, or None when ``spec`` is None.

    Raises:
        ValueError: If ``kind`` is missing or unknown arguments are present.
    """
    if spec is None:
        return None
    data = dict(spec)
    kind = data.pop("kind", None)
    if kind is None:
        raise ValueError(f"Schedule spec {spec!r} has no 'kind'.")
    allowed = {"start", "end", "alpha", "beta", "floor"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"Schedule spec has unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}."
        )
    return Schedule(kind=str(kind), **{k: float(v) for k, v in data.items()})
