"""PFN surrogates: TabPFN v2.5 (primary) and the Hyp 0 benchmark models.

``bar_distribution`` implements the bucketized-regression adapter used by the
classification-only models; ``external`` holds the shared machinery and the
integration table for the five external amortized surrogates (task #8).
"""
from __future__ import annotations

from . import bar_distribution, external
from .bar_distribution import BarDistribution
from .external import EXTERNAL_SPECS, availability

__all__ = [
    "bar_distribution",
    "external",
    "BarDistribution",
    "EXTERNAL_SPECS",
    "availability",
]
