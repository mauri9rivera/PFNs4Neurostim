"""Reference banks for the Hyp C placement analysis.

``prior`` samples the TabPFN v1 prior bag (the in-distribution reference) and
``noise`` draws the out-of-distribution anchor. Task #6 adds the on-grid
interpolation and the shared standardization both banks must pass through.
"""
from __future__ import annotations

from . import noise, prior

__all__ = ["noise", "prior"]
