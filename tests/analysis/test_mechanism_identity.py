"""Cell identities that must not move: OMEGA's grid (running on Narval at ``bb0bba5``) and the CKA (b) placement.

The digests below were computed at ``bb0bba5`` (HEAD before BETA's Phase 0-2, 2026-10-21) from the configs on disk,
by requesting every cell of the run with a recording cell store (nothing computed). A change to anything that enters
a ``bayes_update`` cell's identity would make the pulled Narval cells unreadable by ``--only-cached``; a change to a
CKA (b) identity would orphan the cells re-run for the subject-2 backfill. Needs the NHP data on disk (skipped
otherwise): the identities hold the real channel labels and the data cohort stamp.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable

import pytest

from pfns4neurostim.evaluation import cache
from pfns4neurostim.experiments import mechanism

#: ``(config, kind) -> (n cells, md5 of the sorted cell keys)`` at ``bb0bba5``.
EXPECTED: dict[tuple[str, str], tuple[int, str]] = {
    ("mechanism_bayes_update_nhp", "cell"): (528, "d3aa586c13ae267cddb5a81d2dca2d91"),
    ("mechanism_bayes_update_nhp", "exemplar"): (4, "372d58030c2074d08a56334c940c54d9"),
    ("mechanism_bayes_update_nhp", "c1_exemplar"): (4, "f6935322936ea5a676133f0bbc35fb63"),
    ("mechanism_cka_nhp", "b"): (40, "993820e1717e207380d0d6fe61556dd5"),
}

_NHP_DATA = os.path.join("data", "monkeys")


def _requested(config: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Every cell key a config requests, by kind, with a store that records and computes nothing."""
    keys: dict[str, list[str]] = {}

    def record(self: Any, dataset: str, experiment: str, identity: dict[str, Any], label: str,
               compute: Callable[[], Any]) -> None:
        keys.setdefault(str(identity.get("kind")), []).append(cache.cell_key(identity))
        return None

    from pfns4neurostim.visualization import mechanism as figs

    monkeypatch.setattr(cache.CellStore, "run", record)
    monkeypatch.setattr(figs, "render_bayes_update", lambda out: [])
    monkeypatch.setattr(figs, "render_cka", lambda out: [])
    mechanism.run_mechanism(f"configs/experiment/{config}.yaml", [f"output_root={tmp_path.as_posix()}"])
    return keys


@pytest.mark.skipif(not os.path.isdir(_NHP_DATA), reason="NHP data not on disk")
@pytest.mark.parametrize("config", sorted({c for c, _ in EXPECTED}))
def test_cell_identities_are_unchanged_since_bb0bba5(config: str, tmp_path: Any,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    keys = _requested(config, tmp_path, monkeypatch)
    for (cfg, kind), (n, digest) in EXPECTED.items():
        if cfg != config:
            continue
        got = sorted(keys.get(kind, []))
        assert len(got) == n, f"{config}:{kind}: {len(got)} cells requested, expected {n}"
        assert hashlib.md5(json.dumps(got).encode()).hexdigest() == digest, f"{config}:{kind} identities moved"
