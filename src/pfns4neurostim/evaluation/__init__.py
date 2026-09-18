"""Evaluation: tidy result schema, run-directory I/O, metrics, robustness, BO runner.

``results`` holds the tidy schema and run-dir I/O; ``metrics`` the
range-normalized regrets, accuracy, identification and calibration measures;
``robustness`` the Hyp B S9 statistics; ``bo_runner`` one fully instrumented BO
repetition on one channel.
"""
from __future__ import annotations

from . import metrics, results, robustness

__all__ = ["metrics", "results", "robustness", "bo_runner"]
