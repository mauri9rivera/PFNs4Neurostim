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
    {store_root}/{dataset}/{experiment}/{key}.claim  work-pool claim, only while a pool lane computes the cell

**Work pool (added 2026-10-23).** With ``claim_ttl_s`` set, a cache miss is first *claimed*: the lane writes the
``.claim`` file to a private temp file and hard-links it into place, which fails if the claim exists, so it is created
with its content in one atomic step and any number of lanes, in any number of jobs, can walk the same cell list and
never compute a cell twice (an ``O_CREAT | O_EXCL`` create-then-write, the first version, was visible empty on Lustre
long enough for other lanes to take it over: 192 lanes computed 43 cells, 2026-10-23). A claim is honoured while it is
younger than ``claim_ttl_s`` and was not written by an earlier run of *this* Slurm job (a requeued job re-claims its
own unfinished cells at once; another job's orphaned claim is taken over after the TTL). The claim is removed when the
cell is saved, fails, or its lane is interrupted (SIGTERM from ``scancel``, see ``experiments/mechanism.py``).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from . import pickle_compat as _pickle_compat

__all__ = [
    "cell_key",
    "cell_path",
    "claim_cell",
    "load_cell",
    "release_claim",
    "save_cell",
    "CellStore",
    "CellFailure",
    "row_payload",
]

_KEY_LEN: int = 12
#: Suffix of a work-pool claim next to the cell's JSON.
CLAIM_SUFFIX: str = ".claim"


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


def _claim_owner() -> dict[str, Any]:
    """Who is claiming: the Slurm job and its restart count (``None`` outside Slurm), host and process."""
    return {"job": os.environ.get("SLURM_JOB_ID"), "restart": os.environ.get("SLURM_RESTART_COUNT", "0"),
            "host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}


def _claim_is_live(path: str, ttl_s: float, me: Mapping[str, Any]) -> bool:
    """Whether an existing claim still holds.

    Args:
        path: Claim file.
        ttl_s: Claim lifetime in seconds.
        me: :func:`_claim_owner` of the caller.

    Returns:
        ``False`` if the claim is gone, older than ``ttl_s``, or left by an earlier run of the caller's own Slurm job
        (a requeue bumps ``SLURM_RESTART_COUNT``). A young claim that cannot be read is **live**: reading it as stale
        let several lanes take over one fresh claim on Narval (2026-10-23, 192 lanes computing 43 cells).
    """
    try:
        age = time.time() - os.path.getmtime(path)
    except OSError:
        return False                        # vanished: nobody holds it
    if age >= ttl_s:
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            owner = json.load(fh)
    except (OSError, ValueError):
        return True                         # young but unreadable: assume its owner is still writing it
    if me["job"] is not None and owner.get("job") == me["job"] and owner.get("restart") != me["restart"]:
        return False
    return True


def _link_claim(path: str, me: Mapping[str, Any]) -> bool:
    """Create the claim with its content in one step: write a private temp file, then hard-link it into place.

    ``os.link`` fails if ``path`` exists, so exactly one process wins, and the claim is never visible empty (an
    ``O_EXCL`` create followed by a write was, on Lustre, visible empty long enough for other lanes to read it as torn).

    Args:
        path: Claim file.
        me: :func:`_claim_owner` of the caller.

    Returns:
        ``True`` if this process created the claim.
    """
    tmp = f"{path}.tmp.{me['host']}.{me['pid']}"
    _write_json_file(tmp, me)
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        os.remove(tmp)


def claim_cell(cell_json: str, ttl_s: float) -> bool:
    """Try to claim a cell for this process (work-pool mode).

    Args:
        cell_json: The cell's JSON path (:func:`cell_path`); the claim sits next to it.
        ttl_s: Claim lifetime in seconds: longer than the slowest cell, so a live claim is never taken over.

    Returns:
        ``True`` if this process now owns the cell and should compute it.
    """
    path = cell_json + CLAIM_SUFFIX
    os.makedirs(os.path.dirname(path), exist_ok=True)
    me = _claim_owner()
    if _link_claim(path, me):
        return True
    if _claim_is_live(path, ttl_s, me):
        return False
    # stale: remove it and race for a fresh link, so at most one lane wins the takeover too
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    return _link_claim(path, me)


def release_claim(cell_json: str) -> None:
    """Remove this cell's claim (after the cell was saved or failed); a missing claim is not an error."""
    try:
        os.remove(cell_json + CLAIM_SUFFIX)
    except FileNotFoundError:
        pass


def _write_json_file(path: str, obj: Mapping[str, Any]) -> None:
    """Write ``obj`` as JSON to ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(obj), fh)


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
            trajectory = _pickle_compat.load(fh)
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
        on_cell: Called once per cell that was served (computed or cached), e.g. to
            feed a progress/throughput counter. Not called for failed cells.
        claim_ttl_s: Work-pool mode (module docstring): claim a missed cell before computing it and skip it when
            another lane holds a live claim. ``None`` (default) computes every miss.
        hits: Cells served from the cache this run.
        computed: Cells computed this run.
        claimed_elsewhere: Misses skipped because another lane held the claim (work-pool mode).
        failures: Cells that raised.
    """

    root: str
    enabled: bool = True
    only_cached: bool = False
    on_cell: Callable[[], None] | None = None
    claim_ttl_s: float | None = None
    hits: int = 0
    computed: int = 0
    claimed_elsewhere: int = 0
    failures: list[CellFailure] = field(default_factory=list)
    host: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Record the machine once; cells computed here carry it in their payload."""
        if not self.host and not self.only_cached:
            from ..config import _host_info  # noqa: PLC0415 - config imports evaluation-free modules only

            self.host = _host_info()

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
                self._served()
                return hit
        if self.only_cached:
            return None
        pooled = self.enabled and self.claim_ttl_s is not None
        path = cell_path(self.root, dataset, experiment, key)
        if pooled:
            if not claim_cell(path, float(self.claim_ttl_s)):
                self.claimed_elsewhere += 1
                print(f"[cache] claimed elsewhere {label}", flush=True)
                return None
            # another lane may have saved it between our miss and our claim
            hit = load_cell(self.root, dataset, experiment, key, identity)
            if hit is not None:
                release_claim(path)
                self.hits += 1
                self._served()
                return hit
        try:
            payload, trajectory = compute()
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the grid
            self.failures.append(CellFailure(label, f"{type(exc).__name__}: {exc}"))
            print(f"[cache] FAILED {label}: {type(exc).__name__}: {exc}", flush=True)
            if pooled:
                release_claim(path)
            return None
        except BaseException:               # SIGTERM (scancel, requeue) or Ctrl-C: give the cell back to the pool
            if pooled:
                release_claim(path)
            raise
        self.computed += 1
        payload = {**payload, "host": self.host}  # provenance only; never part of the cell identity
        if self.enabled:
            save_cell(self.root, dataset, experiment, key, identity, payload, trajectory)
        if pooled:
            release_claim(path)
        self._served()
        return payload, trajectory

    def _served(self) -> None:
        """Notify the progress callback that one cell was served."""
        if self.on_cell is not None:
            self.on_cell()

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
