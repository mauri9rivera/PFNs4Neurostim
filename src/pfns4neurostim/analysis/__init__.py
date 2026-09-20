"""Hypothesis C analyses: placement, representation and update-rule comparisons.

``id_ood`` is the pre-restructure metric module, moved here whole. It is
deliberately **not** split by metric yet: tasks #5 (CKA redesign) and #6 (MMD /
sliced-W2 re-formulation) replace those metrics one at a time, and each is carved
into its own module as it is rewritten. Until then every result it produces is
provisional, per the audit.
"""
from __future__ import annotations

__all__ = ["cka", "id_ood", "surface_geometry"]
