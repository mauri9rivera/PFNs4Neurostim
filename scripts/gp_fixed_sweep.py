"""GP-fixed hyperparameter sweep: lengthscale x outputscale (x noise) on the channels of a config.

Diagnostic for the question "why does GP-fixed beat GP-MLL?": if the advantage is a lucky prior it should
vanish away from the default (lengthscale 0.2, outputscale 1.0, noise 0.01) and the regret surface should
show where it lives.

NOT cached, on purpose. Every run goes straight through ``run_channel_bo``, which never touches
``CellStore``, so nothing is written to ``output/cells/`` and no existing GP-fixed cell can be shadowed or
polluted. The only artefacts are the result table and its summary, under their own directory
(``output/gp_fixed_sweep/<tag>/``).

Seeds are ``seed_for(channel, acquisition, 'gp_naive', rep)``, the same as ``bo_benchmark``, so the grid
point equal to the model defaults reproduces the cells of the Hyp A run (a built-in sanity check; numerically
identical only on the same hardware).

Usage::

    python scripts/gp_fixed_sweep.py --config configs/experiment/hyp_a_nhp.yaml --workers 8
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from pfns4neurostim.config import load_experiment_config
from pfns4neurostim.data.channels import ChannelData, iter_channels
from pfns4neurostim.evaluation.bo_runner import run_channel_bo
from pfns4neurostim.seeding import seed_for

#: Metrics copied from ``BOResult.row`` into the table; ``REQUIRED_METRICS`` must be present and finite.
METRICS: tuple[str, ...] = (
    "final_regret", "cumulative_regret", "recommended_regret", "queries_to_target_90",
    "r2", "coverage_90", "nll", "exploration_score",
)
REQUIRED_METRICS: tuple[str, ...] = ("cumulative_regret", "recommended_regret")
MODEL: str = "gp_naive"

# Shared with the pool workers through fork; set once in ``main`` before the pool exists.
_CHANNELS: list[ChannelData] = []
_CFG: Any = None
_ACQ: Any = None


def _floats(text: str) -> list[float]:
    """Parse a comma-separated list of positive floats.

    Args:
        text: E.g. ``"0.05,0.1,0.2"``.

    Returns:
        The values in order.

    Raises:
        argparse.ArgumentTypeError: On an empty list or a non-positive entry.
    """
    values = [float(t) for t in text.split(",") if t.strip()]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"need a comma-separated list of positive floats, got {text!r}")
    return values


def _run_task(task: tuple[int, float, float, float, int]) -> dict[str, Any]:
    """Run one BO repetition of GP-fixed at one grid point on one channel.

    Args:
        task: ``(channel_index, lengthscale, outputscale, noise, rep)``.

    Returns:
        One result row, or a row with an ``error`` field if the run raised.
    """
    ci, ls, os_, noise, rep = task
    channel = _CHANNELS[ci]
    base = {"channel": channel.label, "lengthscale": ls, "outputscale": os_, "noise": noise, "rep": rep}
    seed = seed_for(channel.label, _ACQ.type, MODEL, rep, base_seed=_CFG.seed)
    t0 = time.time()
    try:
        result = run_channel_bo(
            MODEL, channel,
            acq_fn=_ACQ.type, acq_params=_ACQ.params, acq_schedules=_ACQ.schedules,
            budget=_CFG.budget, n_init=_CFG.n_init, seed=seed, device="cpu",
            model_params={"lengthscale": ls, "outputscale": os_, "noise": noise},
            online_y_scaler=_CFG.online_y_scaler,
        )
    except Exception as exc:  # reported in the table and re-raised at the end; the sweep keeps going
        return {**base, "error": f"{type(exc).__name__}: {exc}"}
    row = {**base, "seed": seed, "wall_s": time.time() - t0, "error": ""}
    for key in METRICS:
        row[key] = result.row.get(key, float("nan"))
    for key in REQUIRED_METRICS:
        if not np.isfinite(row[key]):
            row["error"] = f"non-finite {key}={row[key]}"
    return row


def summarise(df: pd.DataFrame, reference: tuple[float, float, float]) -> pd.DataFrame:
    """Aggregate the per-run table over channels and repetitions.

    Args:
        df: Per-run rows without errors.
        reference: ``(lengthscale, outputscale, noise)`` of the default GP-fixed, for the paired delta.

    Returns:
        One row per grid point: medians and means of the metrics and the paired cumulative-regret
        difference against ``reference`` (positive = worse than the default).
    """
    keys = ["lengthscale", "outputscale", "noise"]
    metrics = [m for m in METRICS if m in df]
    out = df.groupby(keys)[metrics].median().add_suffix("_median")
    out["cumulative_regret_mean"] = df.groupby(keys)["cumulative_regret"].mean()
    out["n_runs"] = df.groupby(keys).size()
    ref = df[(df.lengthscale == reference[0]) & (df.outputscale == reference[1]) & (df.noise == reference[2])]
    if len(ref):
        base = ref.set_index(["channel", "rep"])["cumulative_regret"]
        paired = df.join(base.rename("ref_cum"), on=["channel", "rep"])
        paired["delta"] = paired["cumulative_regret"] - paired["ref_cum"]
        out["delta_vs_default_mean"] = paired.groupby(keys)["delta"].mean()
        out["frac_better_than_default"] = paired.groupby(keys)["delta"].apply(lambda s: float((s < 0).mean()))
    return out.reset_index().sort_values("cumulative_regret_median").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Run the sweep and write ``results.csv``, ``summary.csv`` and ``summary.txt``.

    Args:
        argv: Command-line arguments (default ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 if any run failed (the tables are still written).
    """
    global _CHANNELS, _CFG, _ACQ
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiment/hyp_a_nhp.yaml",
                   help="experiment YAML supplying dataset, budget, n_init, seed and online_y_scaler")
    p.add_argument("--set", nargs="*", default=[], dest="overrides", help="key=value config overrides")
    p.add_argument("--lengthscales", type=_floats, default=_floats("0.05,0.1,0.2,0.4,0.8"))
    p.add_argument("--outputscales", type=_floats, default=_floats("0.01,0.03,0.1,0.3,1,3"))
    p.add_argument("--noises", type=_floats, default=_floats("0.01"))
    p.add_argument("--reference", type=_floats, default=_floats("0.2,1.0,0.01"),
                   help="lengthscale,outputscale,noise of the default GP-fixed (paired baseline)")
    p.add_argument("--acq", default="ts_marginal", help="acquisition type from the config's list")
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--data-root", default=None, help="override dataset.data_root (cluster jobs pass the staged copy)")
    p.add_argument("--tag", default="nhp")
    p.add_argument("--out-root", default="output/gp_fixed_sweep")
    args = p.parse_args(argv)
    if len(args.reference) != 3:
        p.error("--reference needs three values: lengthscale,outputscale,noise")

    overrides = list(args.overrides)
    if args.data_root:
        overrides.append(f"dataset.data_root={args.data_root}")
    _CFG = load_experiment_config(args.config, overrides)
    matches = [a for a in _CFG.acquisitions if a.type == args.acq]
    if not matches:
        p.error(f"acquisition {args.acq!r} not in the config's {[a.type for a in _CFG.acquisitions]}")
    _ACQ = matches[0]

    _CHANNELS = list(iter_channels(
        _CFG.dataset.name, _CFG.dataset.subjects, _CFG.dataset.emgs,
        data_root=_CFG.dataset.data_root, gt_mode="full_mean", normalization=_CFG.dataset.normalization,
    ))
    grid = list(itertools.product(args.lengthscales, args.outputscales, args.noises))
    tasks = [
        (ci, ls, os_, noise, rep)
        for ci in range(len(_CHANNELS)) for (ls, os_, noise) in grid for rep in range(args.n_reps)
    ]
    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"[gp_fixed_sweep] {len(_CHANNELS)} channels x {len(grid)} grid points x {args.n_reps} reps = "
        f"{len(tasks)} runs, workers={args.workers}, online_y_scaler={_CFG.online_y_scaler}, "
        f"budget={_CFG.budget}, acq={_ACQ.type}, out={out_dir}", flush=True,
    )

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_run_task, tasks, chunksize=4), 1):
                rows.append(row)
                if i % 100 == 0:
                    print(f"[gp_fixed_sweep] {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            rows.append(_run_task(task))
            if i % 100 == 0:
                print(f"[gp_fixed_sweep] {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    failed = df[df["error"].astype(bool)]
    ok = df[~df["error"].astype(bool)]
    if len(ok):
        summary = summarise(ok, (args.reference[0], args.reference[1], args.reference[2]))
        summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
            fh.write(summary.round(4).to_string(index=False))
        print(summary.round(3).to_string(index=False), flush=True)
    print(f"[gp_fixed_sweep] done in {time.time() - t0:.0f}s: {len(ok)} ok, {len(failed)} failed -> {out_dir}", flush=True)
    if len(failed):
        print(failed[["channel", "lengthscale", "outputscale", "noise", "rep", "error"]].head(20).to_string(index=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
