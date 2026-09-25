"""Every experiment YAML in configs/experiment/ must load and validate (no data, no GPU).

Guards against a config naming a group file that does not exist (2026-09-25: `tabpfn_v1` was added to the
PFN-benchmark configs without `configs/model/tabpfn_v1.yaml`, so the configs failed only when a job started).
"""
from __future__ import annotations

import glob
import os

import pytest

from pfns4neurostim.config import load_experiment_config
from pfns4neurostim.experiments.mechanism import load_mechanism_config

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CONFIGS = sorted(glob.glob(os.path.join(_ROOT, "configs", "experiment", "*.yaml")))


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: os.path.basename(p))
def test_experiment_config_loads(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_ROOT)   # group files resolve relative to the repository root
    if os.path.basename(path).startswith("mechanism_"):
        load_mechanism_config(path)
    else:
        cfg = load_experiment_config(path)
        assert cfg.models, f"{path}: no models"


def test_spinal_budgets_fit_the_grid() -> None:
    """Spinal runs on an 8x8 grid: a budget above 64 sites would be undefined."""
    for path in _CONFIGS:
        if path.endswith("_spinal.yaml") and not os.path.basename(path).startswith("mechanism_"):
            assert load_experiment_config(path).budget <= 64, path
