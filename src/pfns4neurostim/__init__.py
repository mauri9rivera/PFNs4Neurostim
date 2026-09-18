"""PFNs4Neurostim: evaluating TabPFNs as replacements for GP-based Bayesian
optimization in neurostimulation.

Subpackages, in dependency order:

* ``data`` - ``ChannelData`` access, achieved SNR, Hyp B stress knobs.
* ``models`` - surrogate registry (TabPFN v2.5, GP ladder, random baseline).
* ``evaluation`` - BO runner, metrics, robustness statistics, tidy result I/O.
* ``experiments`` - one runner per experiment type, dispatched by ``__main__``.
* ``visualization`` - ``style`` is the single source of figure-style truth.

Built up incrementally per the 2026-09-16 restructure (``.claude/task_plan.md``
tasks #1 and #10); the remaining target modules still live in the legacy flat
``src/`` tree and are reached through narrow, documented seams.
"""
from __future__ import annotations

__all__: list[str] = [
    "config",
    "data",
    "evaluation",
    "experiments",
    "models",
    "seeding",
    "visualization",
]

__version__ = "0.1.0"
