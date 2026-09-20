"""Cell-level result cache: one mechanism for caching, resume and partial output.

Results used to be durable only after the last cell of a grid, so a crash, OOM,
preemption or timeout at cell N lost cells 1..N. Persisting each cell the moment it
completes solves that, avoids re-running finished cells, and makes partial output
extractable.

A *cell* is one BO repetition. Its identity is everything that can change the number
(dataset, channel, model, acquisition, knob level, preprocessing, budget, seed, ...),
hashed to a short key. A hit additionally requires the **stored identity to equal the
requested one exactly**, so an md5 collision or a stale entry can never be returned as
a result. ``cache_version`` is part of the identity from day one: bumping it
invalidates every entry (the legacy ``output/subjects`` cache lacked such a stamp, so
its early entries could never hit again).

Layout::

    {store_root}/{dataset}/{experiment}/{key}.json   identity, tidy row, extras
    {store_root}/{dataset}/{experiment}/{key}.pkl    trajectory
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

__all__ = [
    "cell_key",
    "cell_path",
    "load_cell",
    "save_cell",
    "CellStore",
    "CellFailure",
    "row_payload",
]

_KEY_LEN: int = 12


def _canonical(identity: Mapping[str, Any]) -> str:
    """Serialize an identity dict deterministically (sorted keys, plain types)."""
    return json.dumps(identity, sort_keys=True, default=str)


def cell_key(identity: Mapping[str, Any]) -> str:
    """Return the short content hash of a cell identity.

    Args:
        identity: Everything that can change the cell's result.

    Returns:
        The first 12 hex characters of the md5 of the canonical JSON.
    """
    return hashlib.md5(_canonical(identity).encode("utf-8")).hexdigest()[:_KEY_LEN]


def cell_path(store_root: str, dataset: str, experiment: str, key: str) -> str:
    """Return the JSON path of a cell (its trajectory sits next to it as ``.pkl``).

    Args:
        store_root: Cache root, e.g. ``output/cells``.
        dataset: Dataset name.
        experiment: Experiment type (``'bo_benchmark'``, ``'stress_sweep'``).
        key: Value from :func:`cell_key`.

    Returns:
        ``{store_root}/{dataset}/{experiment}/{key}.json``.
    """
    return os.path.join(store_root, dataset, experiment, f"{key}.json")


def _atomic_write(path: str, write: Callable[[str], None]) -> None:
    """Write via a temp file then ``os.replace`` so a killed job leaves no torn file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_cell(
    store_root: str,
    dataset: str,
    experiment: str,
    key: str,
    identity: Mapping[str, Any],
    payload: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> None:
    """Atomically persist one finished cell.

    The trajectory is written first: a JSON file without its pickle would be a hit
    that cannot be loaded, whereas a pickle without JSON is simply invisible.

    Args:
        store_root: Cache root.
        dataset: Dataset name.
        experiment: Experiment type.
        key: Value from :func:`cell_key` for ``identity``.
        identity: The identity the key was computed from (stored for verification).
        payload: JSON-serializable result (tidy row dict and extras).
        trajectory: Per-step data (pickled).
    """
    json_path = cell_path(store_root, dataset, experiment, key)

    def _write_pkl(tmp: str) -> None:
        with open(tmp, "wb") as fh:
            pickle.dump(dict(trajectory), fh, protocol=pickle.HIGHEST_PROTOCOL)

    def _write_json(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"identity": json.loads(_canonical(identity)), "payload": payload}, fh)

    _atomic_write(json_path[:-5] + ".pkl", _write_pkl)
    _atomic_write(json_path, _write_json)


def load_cell(
    store_root: str,
    dataset: str,
    experiment: str,
    key: str,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return ``(payload, trajectory)`` for a cell, or ``None`` on any miss.

    Args:
        store_root: Cache root.
        dataset: Dataset name.
        experiment: Experiment type.
        key: Value from :func:`cell_key`.
        identity: The requested identity; a stored entry counts as a hit only if its
            identity equals this exactly.

    Returns:
        The stored payload and trajectory, or ``None`` if the entry is absent,
        unreadable, or was written for a different identity.
    """
    json_path = cell_path(store_root, dataset, experiment, key)
    pkl_path = json_path[:-5] + ".pkl"
    if not (os.path.exists(json_path) and os.path.exists(pkl_path)):
        return None
    try:
        with open(json_path, encoding="utf-8") as fh:
            stored = json.load(fh)
        with open(pkl_path, "rb") as fh:
            trajectory = pickle.load(fh)
    except (OSError, ValueError, pickle.UnpicklingError, EOFError):
        return None
    if stored.get("identity") != json.loads(_canonical(identity)):
        return None
    return stored["payload"], trajectory


@dataclass(frozen=True)
class CellFailure:
    """One cell that raised, kept so the run can finish and report everything at once."""

    label: str
    error: str


@dataclass
class CellStore:
    """Cache policy shared by the runners.

    Attributes:
        root: Cache root directory.
        enabled: ``False`` (``--no-cache``) neither reads nor writes the store.
        only_cached: ``True`` (``--only-cached``) never computes: cached cells are
            harvested, everything else is skipped. Used to assemble partial output.
        hits: Cells served from the cache this run.
        computed: Cells computed this run.
        failures: Cells that raised.
    """

    root: str
    enabled: bool = True
    only_cached: bool = False
    hits: int = 0
    computed: int = 0
    failures: list[CellFailure] = field(default_factory=list)

    def run(
        self,
        dataset: str,
        experiment: str,
        identity: Mapping[str, Any],
        label: str,
        compute: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Serve a cell from the cache, or compute and persist it immediately.

        Args:
            dataset: Dataset name.
            experiment: Experiment type.
            identity: Full cell identity.
            label: Human-readable cell label for logs and the failure report.
            compute: Zero-argument callable returning ``(payload, trajectory)``.

        Returns:
            ``(payload, trajectory)``; ``None`` if the cell failed or was skipped by
            ``only_cached``.
        """
        key = cell_key(identity)
        if self.enabled:
            hit = load_cell(self.root, dataset, experiment, key, identity)
            if hit is not None:
                self.hits += 1
                print(f"[cache] hit {label}", flush=True)
                return hit
        if self.only_cached:
            return None
        try:
            payload, trajectory = compute()
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the grid
            self.failures.append(CellFailure(label, f"{type(exc).__name__}: {exc}"))
            print(f"[cache] FAILED {label}: {type(exc).__name__}: {exc}", flush=True)
            return None
        self.computed += 1
        if self.enabled:
            save_cell(self.root, dataset, experiment, key, identity, payload, trajectory)
        return payload, trajectory

    def raise_if_failed(self) -> None:
        """Raise one error listing every failed cell (call after outputs are written).

        Raises:
            RuntimeError: If any cell failed.
        """
        if self.failures:
            listing = "\n".join(f"  - {f.label}: {f.error}" for f in self.failures)
            raise RuntimeError(
                f"{len(self.failures)} cell(s) failed (completed cells are cached; "
                f"re-run to retry only these):\n{listing}"
            )


def row_payload(row: Any, extras: Mapping[str, Any]) -> dict[str, Any]:
    """Build the JSON payload for a tidy row plus its non-schema extras.

    Args:
        row: A :class:`~pfns4neurostim.evaluation.results.TidyRow`.
        extras: Achieved-metric columns outside the fixed schema.

    Returns:
        ``{'row': <dataclass dict, None kept as null>, 'extras': {...}}``.
    """
    return {"row": asdict(row), "extras": dict(extras)}
