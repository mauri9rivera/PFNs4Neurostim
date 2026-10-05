"""Fingerprint every synthetic twin a stress config would build, without running any BO (read-only).

The cell key does not hash the twin (``experiments/_cells.py::cell_identity``), so two processes that fit different
twins for the same channel would silently share cache keys. This script records, per channel, the sha256 of the twin's
ground truth, trials and generator parameters, plus the twin's variance ratio ``var(mu) / var(gt)``. Diffing two outputs
shows whether a code change, or another machine, moves any twin (task plan A3 Step 15).

    python scripts/twin_hashes.py --config configs/experiment/stress_k2_channel_synthetic_nhp.yaml --out twins_nhp.json
    python scripts/twin_hashes.py --config A.yaml --config B.yaml --out twins.json --set dataset.data_root=/path/to/data
    python scripts/twin_hashes.py --compare before.json after.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import replace
from typing import Any

import numpy as np

from pfns4neurostim.config import load_experiment_config
from pfns4neurostim.data.synthetic_neurostim import mean_map, synthetic_channels
from pfns4neurostim.experiments.stress_sweep import _channels


def _sha(array: np.ndarray) -> str:
    """Return the sha256 of an array's dtype, shape and bytes."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{a.dtype}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _params_sha(params: Any) -> str:
    """Return the sha256 of a twin's generator parameters, exact to the last bit (``float.hex``)."""
    parts = [float(params.baseline).hex(), float(params.saturation).hex(), float(params.noise_cv).hex()]
    for spot in params.hotspots:
        parts += [float(v).hex() for v in np.ravel(spot.center)]
        parts += [float(v).hex() for v in np.ravel(spot.lengthscale)]
        parts += [float(spot.amplitude).hex(), float(spot.rotation).hex()]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def fingerprint(config_path: str, overrides: list[str]) -> dict[str, dict[str, Any]]:
    """Fingerprint the twins one synthetic stress config builds.

    Args:
        config_path: Experiment YAML with ``dataset.demo: synthetic``.
        overrides: ``--set`` tokens (e.g. ``dataset.data_root=...``).

    Returns:
        ``{channel label: {y_gt, Y_trials, params, var_ratio}}``.

    Raises:
        ValueError: If the config does not build synthetic twins.
    """
    cfg = load_experiment_config(config_path, overrides)
    if cfg.dataset.demo != "synthetic":
        raise ValueError(f"{config_path}: dataset.demo is {cfg.dataset.demo!r}, not 'synthetic'.")
    # The real channels exactly as the sweep loads them (same call with demo=draws), then the twins exactly as the
    # sweep fits them (same arguments as stress_sweep._channels), so the ratio is against the REAL ground truth.
    real = list(_channels(replace(cfg, dataset=replace(cfg.dataset, demo="draws"))))
    twins = synthetic_channels(real, n_hotspots=cfg.dataset.generator_hotspots, seed=cfg.seed,
                               normalization=cfg.dataset.normalization)
    out: dict[str, dict[str, Any]] = {}
    for src, twin in zip(real, twins):
        params = twin.meta["generator"]
        gt = np.asarray(src.to_raw(src.y_gt), dtype=np.float64)            # [N] real ground truth, raw units
        mu = mean_map(params)                                              # [N] twin's noise-free map
        out[twin.label] = {
            "source": twin.meta.get("source_label"),
            "y_gt": _sha(twin.y_gt),
            "Y_trials": _sha(twin.Y_trials),
            "params": _params_sha(params),
            "var_ratio": float(np.var(mu) / np.var(gt)) if np.var(gt) > 0 else float("nan"),
        }
    return out


def compare(path_a: str, path_b: str) -> int:
    """Print every twin that differs between two fingerprint files.

    Args:
        path_a: First JSON written by this script.
        path_b: Second JSON written by this script.

    Returns:
        Number of differing (or missing) twins; 0 means bit-identical.
    """
    with open(path_a, encoding="utf-8") as fh:
        a = json.load(fh)["twins"]
    with open(path_b, encoding="utf-8") as fh:
        b = json.load(fh)["twins"]
    n_diff = 0
    for label in sorted(set(a) | set(b)):
        ta, tb = a.get(label), b.get(label)
        if ta is None or tb is None:
            print(f"  MISSING {label}: only in {'B' if ta is None else 'A'}")
            n_diff += 1
            continue
        fields = [k for k in ("y_gt", "Y_trials", "params") if ta[k] != tb[k]]
        if fields:
            print(f"  DIFF {label}: {', '.join(fields)} (var_ratio A {ta['var_ratio']:.3g}, B {tb['var_ratio']:.3g})")
            n_diff += 1
    print(f"{n_diff} of {len(set(a) | set(b))} twins differ")
    return n_diff


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (for ``--compare``: 1 if any twin differs).
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", action="append", default=[], help="synthetic stress config (repeatable)")
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides applied to every config")
    p.add_argument("--out", help="JSON file to write")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="diff two fingerprint files and exit")
    args = p.parse_args(argv)
    if args.compare:
        return 1 if compare(*args.compare) else 0
    if not args.config or not args.out:
        p.error("--config and --out are required unless --compare is given")
    twins: dict[str, dict[str, Any]] = {}
    for path in args.config:
        twins.update(fingerprint(path, args.set))
    record = {
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "configs": args.config,
        "twins": twins,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    low = sorted(twins.items(), key=lambda kv: kv[1]["var_ratio"])[:3]
    print(f"wrote {len(twins)} twins -> {args.out}; lowest var_ratio: "
          + ", ".join(f"{k} {v['var_ratio']:.3g}" for k, v in low))
    return 0


if __name__ == "__main__":
    sys.exit(main())
