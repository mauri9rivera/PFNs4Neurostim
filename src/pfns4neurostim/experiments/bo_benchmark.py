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
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import ExperimentConfig, load_experiment_config, resolved_dict
from ..data.channels import parse_shard, iter_channels
from ..diagnostics import ClusterDiagnostics, diagnostics_enabled
from ..evaluation import results as _results
from ..evaluation.cache import CellStore, row_payload
from ..evaluation.bo_runner import run_channel_bo
from ..evaluation.results import TidyRow
from ..models.registry import MODEL_REGISTRY
from ..seeding import seed_for
from ._cells import cell_identity, count_channels, row_from_payload
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
    # The random acquisition is served only by the random-search baseline: any other model
    # would still fit a surrogate every step and add a full run's cost for a random policy.
    if acq_type == "random" or spec.family == "baseline":
        return acq_type == "random" and spec.family == "baseline"
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
        df.groupby(["model", "acq_label", "dataset", "subject", "emg"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    table = (
        per_channel.groupby(["model", "acq_label"], dropna=False)[metrics]
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


def _shard_suffix(shard: tuple[int, int] | None) -> str:
    """Run-directory suffix that keeps shards from overwriting each other's outputs."""
    return "" if shard is None else f"-shard{shard[0]}of{shard[1]}"


def _compute_cell(
    cfg: ExperimentConfig,
    run_tag: str,
    model: str,
    channel: Any,
    acq: Any,
    rep: int,
    seed: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one BO repetition and return its cacheable ``(payload, trajectory)``.

    Args:
        cfg: Resolved experiment configuration.
        run_tag: Run tag written into the row.
        model: Registered model key.
        channel: Channel to optimize over.
        acq: Acquisition block of this cell.
        rep: Repetition index.
        seed: Seed of this repetition.
        label: Cell label for the progress line.

    Returns:
        The tidy-row payload and the per-step trajectory.
    """
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
        result, channel, run_tag=run_tag, experiment="bo_benchmark",
        model=model, acq_type=acq.type, rep=rep, acq_label=acq.name,
    )
    print(
        f"[bo_benchmark] {label} rec_regret={row.recommended_regret:.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    return row_payload(row, {}), result.trajectory


def run_bo_benchmark(
    config_path: str,
    overrides: list[str] | None = None,
    *,
    replot: bool = False,
    run_dir: str | None = None,
    use_cache: bool = True,
    only_cached: bool = False,
    on_cell: Callable[[], None] | None = None,
    shard: tuple[int, int] | None = None,
) -> str:
    """Run (or re-summarize) a models x acquisitions benchmark.

    Args:
        config_path: Path to an experiment YAML. ``acquisitions`` (a list) may be
            given instead of a single ``acquisition`` block to sweep types.
        overrides: ``key=value`` overrides.
        replot: Rebuild tables from the existing ``tidy.csv`` without re-running.
        run_dir: Explicit run directory; defaults to
            ``{output_root}/benchmark/{dataset}/{family}-{tag}``.
        use_cache: ``False`` (``--no-cache``) neither reads nor writes the cell cache.
        only_cached: ``True`` (``--only-cached``) assembles outputs from cached cells
            without computing any.
        on_cell: Called once per cell served (computed or cached); drives the
            cluster-diagnostics throughput counter.
        shard: ``(i, n)`` runs only every n-th channel starting at ``i``. The default
            run directory gets a ``-shard{i}of{n}`` suffix; assemble the union with a
            final ``--only-cached`` run without ``--shard``.

    Returns:
        Path to the run directory.

    Raises:
        FileNotFoundError: With ``replot=True`` and no ``tidy.csv``.
        RuntimeError: If every (model, acquisition) combination was skipped, or any
            cell failed (raised after every completed cell is written and cached).
    """
    cfg = load_experiment_config(config_path, overrides)
    target = run_dir or os.path.join(
        cfg.output_root, "benchmark", cfg.dataset.name, f"{cfg.family}-{cfg.tag}{_shard_suffix(shard)}"
    )
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
        store = CellStore(
            cfg.cell_cache_root, enabled=use_cache, only_cached=only_cached, on_cell=on_cell
        )

        for channel in iter_channels(
            cfg.dataset.name,
            cfg.dataset.subjects,
            cfg.dataset.emgs,
            data_root=cfg.dataset.data_root,
            gt_mode=cfg.gt_mode,
            normalization=cfg.dataset.normalization,
            shard=shard,
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
                        label = f"{channel.label} {model} {acq.type} rep{rep}"
                        identity = cell_identity(
                            cfg, channel, model, acq, experiment="bo_benchmark",
                            rep=rep, seed=seed, budget=cfg.budget,
                        )
                        cell = store.run(
                            channel.dataset, "bo_benchmark", identity, label,
                            lambda m=model, ch=channel, a=acq, r=rep, s=seed, lb=label: _compute_cell(
                                cfg, run_tag, m, ch, a, r, s, lb
                            ),
                        )
                        if cell is None:
                            continue
                        row = row_from_payload(cell[0], run_tag)
                        rows.append(row)
                        trajectories[_results.make_trajectory_key(row)] = cell[1]
        print(
            f"[bo_benchmark] cells: {store.hits} cached, {store.computed} computed, "
            f"{len(store.failures)} failed",
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
                [r for r in rows if r.acq_label == acq.name], acquisition=acq.as_block()
            )
            for acq in acquisitions
            if any(r.acq_label == acq.name for r in rows)
        ]
        df = pd.concat(frames, ignore_index=True, sort=False)
        df.to_csv(tidy_path, index=False)
        _results.write_trajectories(target, trajectories)
        _results.write_config(target, resolved_dict(cfg))
        print(f"[bo_benchmark] wrote {len(df)} rows -> {tidy_path}")
        store.raise_if_failed()

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
    parser.add_argument("--no-cache", action="store_true", help="Neither read nor write the cell cache.")
    parser.add_argument(
        "--shard", type=parse_shard, default=None, metavar="I/N",
        help="Run only every N-th channel starting at I (multi-process lanes).",
    )
    parser.add_argument(
        "--cluster-diag", action="store_true",
        help="Print the SLURM job-efficiency report at the end (or set CLUSTER_DIAG=1).",
    )
    parser.add_argument(
        "--only-cached", action="store_true",
        help="Assemble outputs from cached cells only; compute nothing.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    cfg = load_experiment_config(args.config, args.overrides)
    diag_on = diagnostics_enabled(args.cluster_diag)
    planned = (
        count_channels(cfg, args.shard)
        * cfg.n_reps
        * sum(_supported(m, a.type) for m in cfg.models for a in cfg.acquisitions)
        if diag_on and not args.replot
        else 0
    )
    with ClusterDiagnostics(
        tag=f"{cfg.family}-{cfg.tag}", device=cfg.device, n_planned=planned,
        enabled=diag_on,
    ) as diag:
        run_bo_benchmark(
            args.config, args.overrides, replot=args.replot, run_dir=args.run_dir,
            use_cache=not args.no_cache, only_cached=args.only_cached,
            on_cell=diag.record_experiment, shard=args.shard,
        )
    return 0
