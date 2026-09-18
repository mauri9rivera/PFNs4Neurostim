"""Acquisition functions, their parameters and schedules (P0.2 schema).

``registry`` holds the type registry and the per-type parameter dataclasses;
``base`` defines the loop state and the masked, randomly tie-broken selection;
``schedules`` anneals named parameters over the BO steps. ``thompson`` re-exports
the two Thompson-sampling types under their own module for discoverability.
"""
from __future__ import annotations

from . import base, registry, schedules, thompson
from .base import AcqResult, BOState, acquire, masked_argmax
from .registry import ACQUISITION_REGISTRY, available_acquisitions, build_acquisition

__all__ = [
    "base",
    "registry",
    "schedules",
    "thompson",
    "BOState",
    "AcqResult",
    "acquire",
    "masked_argmax",
    "ACQUISITION_REGISTRY",
    "build_acquisition",
    "available_acquisitions",
]
