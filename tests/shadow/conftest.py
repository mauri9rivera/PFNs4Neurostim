"""Shared fixtures and markers for the joint-TS shadow suite.

Markers:
    shadow: prototype tests that shadow future src/pfns4neurostim code (all tests here).
    slow:   > ~20 s; deselect with ``-m "not slow"``.
    gpu:    need CUDA for acceptable runtime (skipped without CUDA).
"""
from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _ts_prototype.preprocessing import PROJECT_ROOT, SharedData, load_neurostim  # noqa: E402

HAS_CUDA = torch.cuda.is_available()
TABPFN_DEVICE = "cuda" if HAS_CUDA else "cpu"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "shadow: joint-TS shadow prototype tests")
    config.addinivalue_line("markers", "slow: long-running (> ~20 s)")
    config.addinivalue_line("markers", "gpu: requires CUDA")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if str(item.fspath).startswith(_HERE):
            item.add_marker(pytest.mark.shadow)
        if "gpu" in item.keywords and not HAS_CUDA:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def tabpfn_device() -> str:
    """Device for TabPFN (CUDA if available)."""
    return TABPFN_DEVICE


@pytest.fixture(scope="session")
def neurostim_loader() -> Callable[[str, int, int], SharedData]:
    """Loader for real data; skips if the local data directory is absent."""
    def _load(dataset: str, subject: int, emg: int) -> SharedData:
        folder = {"nhp": "monkeys", "5d_rat": "5d_rat", "rat": "rat", "spinal": "spinal"}[dataset]
        if not os.path.isdir(os.path.join(PROJECT_ROOT, "data", folder)):
            pytest.skip(f"data/{folder} not available")
        return load_neurostim(dataset, subject, emg)
    return _load


def pytest_terminal_summary(terminalreporter: "pytest.TerminalReporter") -> None:
    """Always show measured values (report-only tests must print their numbers)."""
    from _shadow_utils import MEASUREMENTS

    if MEASUREMENTS:
        terminalreporter.section("shadow measurements")
        for line in MEASUREMENTS:
            terminalreporter.write_line(line)
