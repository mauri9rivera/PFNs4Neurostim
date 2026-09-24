"""Shard registry: one run directory per experiment, a provenance record per shard.

Sharded jobs (``--shard i/N`` lanes, or any ``--compute-only`` job) never write a run directory of their
own. They compute cells into the shared cell cache and drop one small JSON record into
``<run_dir>/shards/``, naming the machine (node, GPU model, CPU model, SLURM job), the shard and the
models it ran. The assembly step (no shard, ``--only-cached``) then builds ``tidy.csv``, tables and
figures for the whole experiment from the cache and embeds the compact registry in ``config.yaml`` under
``shards:``. Every tidy row also carries the machine that computed it (``host_node``, ``host_gpu``,
``host_cpu``), so latency can always be broken down by hardware.

Concurrent shards write distinct files, so no locking is needed.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Mapping, Sequence

import pandas as pd

__all__ = [
    "SHARD_DIR",
    "HOST_COLUMNS",
    "UNKNOWN_HOST",
    "write_shard_record",
    "read_shard_records",
    "shard_registry",
    "host_columns",
]

#: Subdirectory of a run directory holding one JSON record per shard job.
SHARD_DIR: str = "shards"

#: Tidy columns describing the machine that computed each row, mapped from the host-info keys.
HOST_COLUMNS: dict[str, str] = {"host_node": "node", "host_gpu": "cuda_device", "host_cpu": "cpu"}

#: Value used for cells cached before host provenance was recorded.
UNKNOWN_HOST: str = "unknown"

_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(text: str) -> str:
    """Filesystem-safe form of ``text``."""
    return _SLUG.sub("-", text).strip("-") or "x"


def write_shard_record(
    run_dir: str,
    *,
    shard: tuple[int, int] | None,
    host: Mapping[str, Any],
    models: Sequence[str],
    device: str,
    model_devices: Mapping[str, str],
    cells_computed: int,
    cells_cached: int,
    cells_failed: int,
    started: float,
    status: str,
) -> str:
    """Write the provenance record of one shard job.

    Args:
        run_dir: The experiment's (merged) run directory.
        shard: ``(i, n)`` of this lane, or ``None`` for an unsharded compute-only job.
        host: Machine description from ``config._host_info`` (node, GPU model, CPU model, job id).
        models: Models this job was asked to run.
        device: Default device of the config.
        model_devices: Per-model device overrides (e.g. GP pinned to CPU).
        cells_computed: Cells computed by this job.
        cells_cached: Cells served from the cache by this job.
        cells_failed: Cells that raised.
        started: ``time.time()`` at job start.
        status: ``'ok'`` or ``'failed'``.

    Returns:
        Path of the record written.
    """
    directory = os.path.join(run_dir, SHARD_DIR)
    os.makedirs(directory, exist_ok=True)
    part = "all" if shard is None else f"{shard[0]}of{shard[1]}"
    job = host.get("slurm_job_id") or "local"
    name = f"{_slug(str(host.get('node', 'node')))}-job{_slug(str(job))}-shard{part}-{'-'.join(_slug(m) for m in models)}.json"
    record = {
        "shard": part,
        "models": list(models),
        "device": device,
        "model_devices": dict(model_devices),
        "host": dict(host),
        "cells_computed": int(cells_computed),
        "cells_cached": int(cells_cached),
        "cells_failed": int(cells_failed),
        "wall_time_s": round(time.time() - started, 1),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
    }
    path = os.path.join(directory, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def read_shard_records(run_dir: str) -> list[dict[str, Any]]:
    """Read every shard record of a run directory, oldest first.

    Args:
        run_dir: The experiment's run directory.

    Returns:
        The parsed records (empty when the run was never sharded).
    """
    directory = os.path.join(run_dir, SHARD_DIR)
    if not os.path.isdir(directory):
        return []
    records: list[dict[str, Any]] = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            with open(os.path.join(directory, name), encoding="utf-8") as fh:
                records.append(json.load(fh))
    return sorted(records, key=lambda r: r.get("finished_utc", ""))


def shard_registry(run_dir: str) -> list[dict[str, Any]]:
    """Compact registry for ``config.yaml``: which machine ran which shard and models.

    Args:
        run_dir: The experiment's run directory.

    Returns:
        One entry per shard record with shard, models, devices, node, GPU, CPU, job and cell counts.
    """
    return [
        {
            "shard": r["shard"],
            "models": r["models"],
            "device": r["device"],
            "model_devices": r["model_devices"],
            "node": r["host"].get("node"),
            "gpu": r["host"].get("cuda_device"),
            "cpu": r["host"].get("cpu"),
            "slurm_job_id": r["host"].get("slurm_job_id"),
            "cells_computed": r["cells_computed"],
            "cells_cached": r["cells_cached"],
            "status": r["status"],
        }
        for r in read_shard_records(run_dir)
    ]


def host_columns(payloads: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Per-row host columns from cached cell payloads.

    Args:
        payloads: The cell payloads, aligned with the tidy rows. Older cells have no ``host`` entry.

    Returns:
        A frame with ``host_node``, ``host_gpu`` and ``host_cpu`` (``'unknown'`` when not recorded,
        ``'none'`` for a machine without a GPU).
    """
    records = []
    for payload in payloads:
        host = payload.get("host") or {}
        records.append(
            {
                col: (UNKNOWN_HOST if not host else (host.get(key) or "none"))
                for col, key in HOST_COLUMNS.items()
            }
        )
    return pd.DataFrame.from_records(records, columns=list(HOST_COLUMNS))
