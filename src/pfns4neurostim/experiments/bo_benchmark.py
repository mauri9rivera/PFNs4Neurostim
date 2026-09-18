"""Bayesian-optimization benchmark: models x acquisitions on unstressed channels.

Serves **Hyp 0** (does acquisition choice matter less for PFNs than for GPs?) and
**Hyp A** (is the PFN equivalent to the best-tuned GP, better than the naive GP and
random search, and cheaper per query?).

The grid is ``model x acquisition x channel x repetition`` at the nominal anchor —
no stress knob. Combinations a model cannot serve (``ts_joint`` on a PFN) are
skipped with a logged note rather than crashing the sweep, since the Hyp 0 table
is deliberately ragged: joint TS is a GP-only reference row.

Deliverables:
    tidy.csv            one row per BO run (same schema as the stress sweep)
    trajectories.pkl    per-step data
    config.yaml         resolved configuration (P0.1/P0.2 logging)
    acquisition_table.csv   model x acquisition summary of the co-primary regrets

CLI::

    python -m pfns4neurostim bo_benchmark --config configs/experiment/hyp_a_nhp.yaml
    python -m pfns4neurostim bo_benchmark --config configs/experiment/hyp0_acq_table_nhp.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import ExperimentConfig, load_experiment_config, resolved_dict
from ..data.channels import iter_channels
from ..evaluation import results as _results
from ..evaluation.bo_runner import run_channel_bo
from ..evaluation.results import TidyRow
from ..models.registry import MODEL_REGISTRY
from ..seeding import seed_for
from ._rows import build_row

__all__ = ["run_bo_benchmark", "build_acquisition_table", "main"]

#: Metrics summarized in the Hyp 0 acquisition table.
_TABLE_METRICS: tuple[str, ...] = (
    "recommended_regret",
    "best_queried_regret",
    "cumulative_regret",
    "r2",
    "mean_query_latency_s",
)


def _supported(model: str, acq_type: str) -> bool:
    """Whether a model can serve an acquisition type.

    Args:
        model: Canonical model key.
        acq_type: Acquisition type name.

    Returns:
        True when the combination is runnable.
    """
    spec = MODEL_REGISTRY.get(model)
    if spec is None:
        return False
    if acq_type == "ts_joint":
        return spec.family == "gp"
    if spec.family == "baseline":
        return acq_type == "random"
    return True


def build_acquisition_table(df: pd.DataFrame, out_dir: str) -> tuple[pd.DataFrame, str]:
    """Summarize the benchmark as a model x acquisition table.

    Channels are averaged before models are compared, so a subject with many EMGs
    does not dominate the cell means.

    Args:
        df: Tidy benchmark frame.
        out_dir: Destination directory.

    Returns:
        ``(table, csv_path)``.
    """
    metrics = [m for m in _TABLE_METRICS if m in df.columns]
    per_channel = (
        df.groupby(["model", "acq_type", "dataset", "subject", "emg"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    table = (
        per_channel.groupby(["model", "acq_type"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    table.columns = [
        col[0] if not col[1] else f"{col[0]}_{col[1]}" for col in table.columns.to_flat_index()
    ]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "acquisition_table.csv")
    table.to_csv(path, index=False)
    return table, path


def run_bo_benchmark(
    config_path: str,
    overrides: list[str] | None = None,
    *,
    replot: bool = False,
    run_dir: str | None = None,
) -> str:
    """Run (or re-summarize) a models x acquisitions benchmark.

    Args:
        config_path: Path to an experiment YAML. ``acquisitions`` (a list) may be
            given instead of a single ``acquisition`` block to sweep types.
        overrides: ``key=value`` overrides.
        replot: Rebuild tables from the existing ``tidy.csv`` without re-running.
        run_dir: Explicit run directory; defaults to
            ``{output_root}/benchmark/{dataset}``.

    Returns:
        Path to the run directory.

    Raises:
        FileNotFoundError: With ``replot=True`` and no ``tidy.csv``.
        RuntimeError: If every (model, acquisition) combination was skipped.
    """
    cfg = load_experiment_config(config_path, overrides)
    target = run_dir or os.path.join(cfg.output_root, "benchmark", cfg.dataset.name)
    os.makedirs(target, exist_ok=True)
    tidy_path = os.path.join(target, "tidy.csv")

    if replot:
        if not os.path.exists(tidy_path):
            raise FileNotFoundError(f"--replot needs an existing {tidy_path}; run the benchmark first.")
        df = pd.read_csv(tidy_path)
    else:
        run_tag = f"{cfg.dataset.name}-benchmark-{cfg.tag}"
        acquisitions = cfg.acquisitions
        rows: list[TidyRow] = []
        trajectories: dict[tuple[Any, ...], dict[str, Any]] = {}
        skipped: list[str] = []

        for channel in iter_channels(
            cfg.dataset.name,
            cfg.dataset.subjects,
            cfg.dataset.emgs,
            data_root=cfg.dataset.data_root,
            gt_mode=cfg.gt_mode,
        ):
            for model in cfg.models:
                for acq in acquisitions:
                    if not _supported(model, acq.type):
                        note = f"{model} x {acq.type}"
                        if note not in skipped:
                            skipped.append(note)
                        continue
                    for rep in range(cfg.n_reps):
                        seed = seed_for(channel.label, acq.type, model, rep, base_seed=cfg.seed)
                        t0 = time.time()
                        result = run_channel_bo(
                            model,
                            channel,
                            acq_fn=acq.type,
                            acq_params=acq.params,
                            acq_schedules=acq.schedules,
                            budget=cfg.budget,
                            n_init=cfg.n_init,
                            seed=seed,
                            device=cfg.device,
                            model_params=cfg.model_params.get(model, {}),
                        )
                        row = build_row(
                            result,
                            channel,
                            run_tag=run_tag,
                            experiment="bo_benchmark",
                            model=model,
                            acq_type=acq.type,
                            rep=rep,
                        )
                        rows.append(row)
                        trajectories[_results.make_trajectory_key(row)] = result.trajectory
                        print(
                            f"[bo_benchmark] {channel.label} {model:<12} {acq.type:<12} rep{rep} "
                            f"rec_regret={row.recommended_regret:.4f} ({time.time() - t0:.1f}s)",
                            flush=True,
                        )
        if not rows:
            raise RuntimeError(
                "bo_benchmark produced no rows. Skipped combinations: "
                f"{skipped or 'none'}; check dataset.subjects / models / acquisitions."
            )
        if skipped:
            print(f"[bo_benchmark] skipped unsupported combinations: {', '.join(skipped)}")

        # One acquisition block per row: flatten per-acquisition rather than globally.
        frames = [
            _results.rows_to_dataframe(
                [r for r in rows if r.acq_type == acq.type], acquisition=acq.as_block()
            )
            for acq in acquisitions
            if any(r.acq_type == acq.type for r in rows)
        ]
        df = pd.concat(frames, ignore_index=True, sort=False)
        df.to_csv(tidy_path, index=False)
        _results.write_trajectories(target, trajectories)
        _results.write_config(target, resolved_dict(cfg))
        print(f"[bo_benchmark] wrote {len(df)} rows -> {tidy_path}")

    _, table_path = build_acquisition_table(df, target)
    print(f"[bo_benchmark] wrote {table_path}")

    from ..visualization import bo as bo_figs  # deferred: matplotlib import

    for path in bo_figs.render_all(df, target, dataset=cfg.dataset.name):
        print(f"[bo_benchmark] wrote {path}")
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m pfns4neurostim bo_benchmark``.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="pfns4neurostim bo_benchmark",
        description="Models x acquisitions BO benchmark (Hyp 0 / Hyp A).",
    )
    parser.add_argument("--config", required=True, help="Path to an experiment YAML.")
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=None, metavar="KEY=VALUE",
        help="Dotted-key overrides, e.g. --set n_reps=2 budget=30.",
    )
    parser.add_argument("--replot", action="store_true", help="Rebuild tables from tidy.csv.")
    parser.add_argument("--run-dir", default=None, help="Override the output run directory.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    run_bo_benchmark(args.config, args.overrides, replot=args.replot, run_dir=args.run_dir)
    return 0
