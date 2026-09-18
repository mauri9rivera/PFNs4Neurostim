"""Hypothesis B stress-regime sweep (roadmap Phase 2).

Runs the grid **knob x level x model x channel x repetition** and writes one tidy
CSV plus a trajectory pickle and the resolved config. The runner is knob-agnostic:
it asks the knob for its levels, applies it, records whatever the knob reports as
achieved, and lets the figure layer pick the x-axis. Adding K5 or the Demo 1
generator therefore changes nothing here.

Deliverables (roadmap S2, S9):
    tidy.csv                     one row per BO run
    trajectories.pkl             per-step data keyed by the tidy key columns
    config.yaml                  resolved configuration (P0.1/P0.2 logging)
    degradation_invivo.svg       regret and R-squared vs achieved SNR
    calibration_invivo.svg       coverage-90 and ECE vs achieved SNR
    robustness.csv               breakdown point, degradation AUC, CVaR-10%

CLI::

    python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_nhp.yaml
    python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_nhp.yaml --replot
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import ExperimentConfig, load_experiment_config, resolved_dict
from ..data.channels import ChannelData, iter_channels
from ..data.stress import build_knob
from ..evaluation import results as _results
from ..evaluation.bo_runner import run_channel_bo
from ..evaluation.results import TidyRow
from ..seeding import seed_for
from ._rows import build_row

__all__ = ["run_stress_sweep", "build_tidy_rows", "main"]


def _channels(cfg: ExperimentConfig) -> Iterable[ChannelData]:
    """Yield the channels selected by the config.

    Args:
        cfg: Resolved experiment configuration.

    Yields:
        One :class:`~pfns4neurostim.data.channels.ChannelData` per (subject, EMG).
    """
    return iter_channels(
        cfg.dataset.name,
        cfg.dataset.subjects,
        cfg.dataset.emgs,
        data_root=cfg.dataset.data_root,
        gt_mode=cfg.gt_mode,
    )


def build_tidy_rows(
    cfg: ExperimentConfig,
    run_tag: str,
    *,
    progress: bool = True,
) -> tuple[list[TidyRow], dict[tuple[Any, ...], dict[str, Any]], list[dict[str, Any]]]:
    """Execute the whole sweep grid.

    For each channel, each knob level is applied once and reused by every model
    and repetition, so all models see *identical* stressed data — the comparison
    is paired, which is what the TOST analysis assumes.

    Args:
        cfg: Resolved experiment configuration.
        run_tag: Run tag written into every row.
        progress: Print a per-cell progress line to stdout.

    Returns:
        ``(rows, trajectories, extras)`` where ``extras`` holds the non-schema
        columns (achieved metrics) aligned with ``rows`` by position.
    """
    knob = build_knob(cfg.knob.type, cfg.knob.levels)
    rows: list[TidyRow] = []
    trajectories: dict[tuple[Any, ...], dict[str, Any]] = {}
    extras: list[dict[str, Any]] = []

    for channel in _channels(cfg):
        for level in knob.levels:
            rng = np.random.default_rng(
                seed_for(channel.label, knob.name, level, base_seed=cfg.seed)
            )
            stressed = knob.apply(channel, level, rng)
            achieved = knob.achieved(stressed)
            budget = knob.budget_for(level, cfg.budget)

            for model in cfg.models:
                for rep in range(cfg.n_reps):
                    seed = seed_for(channel.label, knob.name, level, model, rep, base_seed=cfg.seed)
                    t0 = time.time()
                    result = run_channel_bo(
                        model,
                        stressed,
                        acq_fn=cfg.acquisition.type,
                        acq_params=cfg.acquisition.params,
                        acq_schedules=cfg.acquisition.schedules,
                        budget=budget,
                        n_init=cfg.n_init,
                        seed=seed,
                        device=cfg.device,
                        model_params=cfg.model_params.get(model, {}),
                    )
                    row = build_row(
                        result,
                        stressed,
                        run_tag=run_tag,
                        experiment="stress_sweep",
                        model=model,
                        acq_type=cfg.acquisition.type,
                        rep=rep,
                        knob=knob.name,
                        level=float(level),
                        achieved=achieved,
                    )
                    rows.append(row)
                    extras.append(dict(achieved))
                    trajectories[_results.make_trajectory_key(row)] = result.trajectory

                    if progress:
                        print(
                            f"[stress_sweep] {channel.label} {knob.name}={level:g} "
                            f"snr={achieved.get('achieved_snr_db', float('nan')):6.2f}dB "
                            f"{model:<12} rep{rep} "
                            f"rec_regret={row.recommended_regret:.4f} "
                            f"({time.time() - t0:.1f}s)",
                            flush=True,
                        )
    if not rows:
        raise RuntimeError(
            "stress_sweep produced no rows: check dataset.subjects / dataset.emgs "
            "and that the raw data is present under dataset.data_root."
        )
    return rows, trajectories, extras


def run_stress_sweep(
    config_path: str,
    overrides: list[str] | None = None,
    *,
    replot: bool = False,
    run_dir: str | None = None,
) -> str:
    """Run (or re-plot) one stress sweep.

    Args:
        config_path: Path to an experiment YAML.
        overrides: ``key=value`` overrides applied to the resolved config.
        replot: Skip execution and rebuild every figure and table from the
            existing ``tidy.csv`` in the run directory.
        run_dir: Explicit run directory; defaults to
            ``{output_root}/stress/{knob}/{dataset}``.

    Returns:
        Path to the run directory holding the deliverables.

    Raises:
        FileNotFoundError: With ``replot=True`` and no ``tidy.csv`` present.
    """
    cfg = load_experiment_config(config_path, overrides)
    target = run_dir or os.path.join(cfg.output_root, "stress", cfg.knob.type, cfg.dataset.name)
    os.makedirs(target, exist_ok=True)
    tidy_path = os.path.join(target, "tidy.csv")

    if replot:
        if not os.path.exists(tidy_path):
            raise FileNotFoundError(
                f"--replot needs an existing {tidy_path}; run the sweep first."
            )
        df = pd.read_csv(tidy_path)
    else:
        run_tag = f"{cfg.dataset.name}-{cfg.knob.type}-{cfg.tag}"
        rows, trajectories, extras = build_tidy_rows(cfg, run_tag)
        df = _results.rows_to_dataframe(rows, acquisition=cfg.acquisition.as_block())
        # Achieved metrics that are not part of the fixed schema (future knobs
        # may report e.g. amplitude ratio) ride along as extra columns.
        for key in sorted({k for e in extras for k in e} - set(df.columns)):
            df[key] = [e.get(key, np.nan) for e in extras]
        df.to_csv(tidy_path, index=False)
        _results.write_trajectories(target, trajectories)
        _results.write_config(target, resolved_dict(cfg))
        print(f"[stress_sweep] wrote {len(df)} rows -> {tidy_path}")

    from ..visualization import stress as stress_figs  # deferred: matplotlib import

    written = stress_figs.render_all(df, target, knob=cfg.knob.type, dataset=cfg.dataset.name)
    for path in written:
        print(f"[stress_sweep] wrote {path}")
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m pfns4neurostim stress_sweep``.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="pfns4neurostim stress_sweep",
        description="Hypothesis B stress-regime sweep (roadmap Phase 2).",
    )
    parser.add_argument("--config", required=True, help="Path to an experiment YAML.")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Dotted-key overrides, e.g. --set n_reps=2 dataset.subjects=[1].",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Rebuild figures and tables from the existing tidy.csv without re-running.",
    )
    parser.add_argument("--run-dir", default=None, help="Override the output run directory.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    run_stress_sweep(args.config, args.overrides, replot=args.replot, run_dir=args.run_dir)
    return 0
