"""Surrogate models and their registry.

``registry`` maps canonical model keys (matching
``visualization.style.MODEL_STYLES``) to constructors and the version strings
logged in every result row (P0.1).
"""
from __future__ import annotations

from . import registry
from .registry import MODEL_REGISTRY, build_surrogate, model_version

__all__ = ["registry", "MODEL_REGISTRY", "build_surrogate", "model_version"]
