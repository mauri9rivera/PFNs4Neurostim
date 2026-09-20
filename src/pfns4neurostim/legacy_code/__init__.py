"""Superseded code, kept importable for reproducing earlier results.

Nothing here is part of the supported API. It holds the pre-restructure CLIs
(``evaluation``, ``vanilla_benchmark``, ``aggregate``, ``id_ood_analysis``,
``mechanistic_ablation``, ``latency_benchmark``, ``scaling_timing``), the
finetuning/LoRA stack cut from the research design on 2026-09-16, the legacy BO
loop and plotting modules, and the one-off analysis scripts.

New work uses ``pfns4neurostim.experiments`` and ``pfns4neurostim.visualization``.
Tests covering this subpackage are marked ``legacy``.
"""
from __future__ import annotations

__all__ = ["bo_loops", "gpbo_utils", "visualization", "query_transforms"]
