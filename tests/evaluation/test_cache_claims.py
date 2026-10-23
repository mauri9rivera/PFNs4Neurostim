"""Work-pool claims of the cell cache (2026-10-23): one lane per cell, stale claims taken over, claims released."""
from __future__ import annotations

import json
import os
import time
from typing import Any

import pytest

from pfns4neurostim.evaluation.cache import CLAIM_SUFFIX, CellStore, cell_key, cell_path, claim_cell, release_claim

IDENT: dict[str, Any] = {"kind": "cell", "t": 50, "draw": 3, "cache_version": 1}
TTL_S: float = 3600.0


def _claim_path(root: str) -> str:
    return cell_path(root, "nhp", "x", cell_key(IDENT)) + CLAIM_SUFFIX


def test_claim_is_exclusive_and_released(tmp_path: Any) -> None:
    path = cell_path(str(tmp_path), "nhp", "x", cell_key(IDENT))
    assert claim_cell(path, TTL_S)
    assert not claim_cell(path, TTL_S)          # a second lane sees a live claim
    release_claim(path)
    assert not os.path.exists(path + CLAIM_SUFFIX)
    assert claim_cell(path, TTL_S)
    release_claim(path)
    release_claim(path)                         # releasing twice is not an error


def test_stale_claim_is_taken_over(tmp_path: Any) -> None:
    path = cell_path(str(tmp_path), "nhp", "x", cell_key(IDENT))
    assert claim_cell(path, TTL_S)
    old = time.time() - 2 * TTL_S
    os.utime(path + CLAIM_SUFFIX, (old, old))
    assert claim_cell(path, TTL_S)


def test_requeued_job_reclaims_its_own_cells(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    path = cell_path(str(tmp_path), "nhp", "x", cell_key(IDENT))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    assert claim_cell(path, TTL_S)
    monkeypatch.setenv("SLURM_JOB_ID", "456")                       # another job: the claim holds
    assert not claim_cell(path, TTL_S)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")                  # the same job after a requeue: stale
    assert claim_cell(path, TTL_S)
    with open(path + CLAIM_SUFFIX, encoding="utf-8") as fh:
        assert json.load(fh)["restart"] == "1"


def test_store_skips_claimed_cell_and_counts_it(tmp_path: Any) -> None:
    root = str(tmp_path)
    assert claim_cell(cell_path(root, "nhp", "x", cell_key(IDENT)), TTL_S)
    store = CellStore(root, claim_ttl_s=TTL_S)
    assert store.run("nhp", "x", IDENT, "c", lambda: pytest.fail("claimed elsewhere")) is None
    assert (store.computed, store.hits, store.claimed_elsewhere) == (0, 0, 1)


def test_store_releases_claim_after_compute_and_failure(tmp_path: Any) -> None:
    root = str(tmp_path)
    store = CellStore(root, claim_ttl_s=TTL_S)
    assert store.run("nhp", "x", IDENT, "c", lambda: ({"row": {}}, {})) is not None
    assert not os.path.exists(_claim_path(root))
    assert store.run("nhp", "x", IDENT, "c", lambda: pytest.fail("cached")) is not None   # a hit, no claim
    other = {**IDENT, "draw": 4}
    assert store.run("nhp", "x", other, "f", lambda: 1 / 0) is None
    assert not os.path.exists(cell_path(root, "nhp", "x", cell_key(other)) + CLAIM_SUFFIX)
    assert (store.computed, store.hits, len(store.failures)) == (1, 1, 1)


def test_without_ttl_no_claim_is_written(tmp_path: Any) -> None:
    root = str(tmp_path)
    CellStore(root).run("nhp", "x", IDENT, "c", lambda: ({"row": {}}, {}))
    assert not os.path.exists(_claim_path(root))
