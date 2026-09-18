"""Experiment runners, one module per experiment type.

``bo_benchmark`` runs the models x acquisitions grid (Hyp 0 / Hyp A);
``stress_sweep`` runs the Hyp B regime sweeps. Both emit the same tidy schema,
assembled by the shared ``_rows.build_row``.
"""
from __future__ import annotations

__all__ = ["bo_benchmark", "stress_sweep"]
