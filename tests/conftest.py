"""Shared pytest fixtures and marker handling (task #1 Step 1).

Markers (declared in ``pyproject.toml``):

* ``slow``   — longer than roughly 20 s; deselect with ``-m 'not slow'``
* ``gpu``    — needs CUDA; **skipped automatically** when it is unavailable
* ``shadow`` — prototypes that shadow future package code
* ``legacy`` — covers ``legacy_code`` (finetuning, LoRA, ...)

The fast suite is::

    pytest tests -m "not slow and not gpu and not legacy" -q
"""
from __future__ import annotations

import numpy as np
import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip GPU-marked tests when no CUDA device is present.

    Args:
        config: Pytest config.
        items: Collected test items.
    """
    try:
        import torch  # noqa: PLC0415 - optional at collection time

        has_cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken torch import means no GPU either
        has_cuda = False
    if has_cuda:
        return
    skip = pytest.mark.skip(reason="CUDA is not available on this machine")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so every test that uses randomness is reproducible."""
    return np.random.default_rng(0)


@pytest.fixture
def tiny_channel():
    """A 20-site, 6-trial synthetic channel with one smooth hotspot.

    Small enough for a full BO run in a fraction of a second, and structured
    enough that a working surrogate beats random search on it.

    Returns:
        A :class:`~pfns4neurostim.data.channels.ChannelData`.
    """
    from pfns4neurostim.data.channels import ChannelData

    generator = np.random.default_rng(0)
    coords = np.stack(np.meshgrid(np.arange(5), np.arange(4)), axis=-1).reshape(-1, 2)  # [20, 2]
    x = coords / np.array([4.0, 3.0])                                                    # [20, 2]
    y_gt = np.exp(-((x[:, 0] - 0.7) ** 2 + (x[:, 1] - 0.3) ** 2) / 0.1)                  # [20]
    y_gt = (y_gt - y_gt.mean()) / y_gt.std()
    Y = y_gt[:, None] + 0.25 * generator.normal(size=(20, 6))                            # [20, 6]
    return ChannelData("nhp", 1, 0, x, Y, y_gt, coords, (5, 4))
