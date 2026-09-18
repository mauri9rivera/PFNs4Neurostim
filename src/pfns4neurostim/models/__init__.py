"""Surrogate models and their registry.

``registry`` maps canonical model keys (matching
``visualization.style.MODEL_STYLES``) to constructors and the version strings
logged in every result row (P0.1).
"""
from __future__ import annotations

from . import protocol, registry
from .protocol import SurrogateAdapter, SurrogateModel
from .registry import MODEL_REGISTRY, build_surrogate, model_version

__all__ = [
    "protocol",
    "registry",
    "MODEL_REGISTRY",
    "SurrogateAdapter",
    "SurrogateModel",
    "build_surrogate",
    "model_version",
]
