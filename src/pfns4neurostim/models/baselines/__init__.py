"""Non-learning baselines: the lower bounds every surrogate must beat."""
from __future__ import annotations

from .random_search import RandomSearchSurrogate

__all__ = ["RandomSearchSurrogate"]
