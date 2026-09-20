"""Cell cache: key stability, exact-identity hits, atomic writes, failure tolerance."""
from __future__ import annotations

import os
from typing import Any

import pytest

from pfns4neurostim.evaluation.cache import (
    CellStore,
    cell_key,
    cell_path,
    load_cell,
    save_cell,
)

IDENT: dict[str, Any] = {"model": "gp_mll", "rep": 0, "seed": 1, "cache_version": 1}


def test_key_is_stable_and_order_independent() -> None:
    assert cell_key(IDENT) == cell_key(dict(reversed(list(IDENT.items()))))
    assert len(cell_key(IDENT)) == 12


def test_key_changes_with_any_field() -> None:
    assert cell_key(IDENT) != cell_key({**IDENT, "seed": 2})
    assert cell_key(IDENT) != cell_key({**IDENT, "cache_version": 2})


def test_round_trip(tmp_path: Any) -> None:
    root = str(tmp_path)
    key = cell_key(IDENT)
    save_cell(root, "nhp", "bo_benchmark", key, IDENT, {"row": {"a": 1}}, {"y": [1, 2]})
    payload, traj = load_cell(root, "nhp", "bo_benchmark", key, IDENT)
    assert payload == {"row": {"a": 1}} and traj == {"y": [1, 2]}


def test_identity_mismatch_is_a_miss(tmp_path: Any) -> None:
    """A colliding key with a different stored identity must never be served."""
    root = str(tmp_path)
    key = cell_key(IDENT)
    save_cell(root, "nhp", "x", key, {**IDENT, "seed": 99}, {"row": {}}, {})
    assert load_cell(root, "nhp", "x", key, IDENT) is None


def test_missing_and_torn_entries_are_misses(tmp_path: Any) -> None:
    root = str(tmp_path)
    key = cell_key(IDENT)
    assert load_cell(root, "nhp", "x", key, IDENT) is None
    save_cell(root, "nhp", "x", key, IDENT, {"row": {}}, {})
    with open(cell_path(root, "nhp", "x", key), "w") as fh:
        fh.write("{ torn")
    assert load_cell(root, "nhp", "x", key, IDENT) is None


def test_atomic_write_leaves_no_temp_files(tmp_path: Any) -> None:
    root = str(tmp_path)
    save_cell(root, "nhp", "x", cell_key(IDENT), IDENT, {"row": {}}, {})
    leftovers = [f for f in os.listdir(tmp_path / "nhp" / "x") if ".tmp." in f]
    assert leftovers == []


class TestCellStore:
    def test_hit_skips_compute(self, tmp_path: Any) -> None:
        store = CellStore(str(tmp_path))
        calls: list[int] = []

        def compute() -> tuple[dict[str, Any], dict[str, Any]]:
            calls.append(1)
            return {"row": {"v": 1}}, {"t": 1}

        assert store.run("nhp", "x", IDENT, "cell", compute) is not None
        assert store.run("nhp", "x", IDENT, "cell", compute) is not None
        assert len(calls) == 1 and (store.computed, store.hits) == (1, 1)

    def test_no_cache_neither_reads_nor_writes(self, tmp_path: Any) -> None:
        store = CellStore(str(tmp_path), enabled=False)
        store.run("nhp", "x", IDENT, "c", lambda: ({"row": {}}, {}))
        assert not os.listdir(tmp_path)

    def test_only_cached_never_computes(self, tmp_path: Any) -> None:
        store = CellStore(str(tmp_path), only_cached=True)
        assert store.run("nhp", "x", IDENT, "c", lambda: pytest.fail("computed")) is None

    def test_failure_is_collected_not_raised(self, tmp_path: Any) -> None:
        store = CellStore(str(tmp_path))

        def boom() -> tuple[dict[str, Any], dict[str, Any]]:
            raise ValueError("bad cell")

        assert store.run("nhp", "x", IDENT, "cellA", boom) is None
        with pytest.raises(RuntimeError, match="cellA: ValueError: bad cell"):
            store.raise_if_failed()
