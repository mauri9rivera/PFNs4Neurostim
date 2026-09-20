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
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..config import ExperimentConfig, load_experiment_config, resolved_dict
from ..data.channels import ChannelData, iter_channels
from ..data.stress import KnobNotApplicable, build_knob, calibrate_levels, floor_snr_db
from ..diagnostics import ClusterDiagnostics, diagnostics_enabled
from ..evaluation import results as _results
from ..evaluation.cache import CellStore, row_payload
from ..evaluation.bo_runner import run_channel_bo
from ..evaluation.results import TidyRow
from ..seeding import seed_for
from ._cells import cell_identity, count_channels, row_from_payload
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
        normalization=cfg.dataset.normalization,
    )


def _compute_cell(
    cfg: ExperimentConfig,
    run_tag: str,
    knob_name: str,
    level: float,
    stressed: ChannelData,
    achieved: dict[str, Any],
    budget: int,
    model: str,
    rep: int,
    seed: int,
    label: str,
    progress: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one BO repetition on a stressed channel; return its cacheable cell.

    Args:
        cfg: Resolved experiment configuration.
        run_tag: Run tag written into the row.
        knob_name: Registered knob name.
        level: Knob level.
        stressed: The already-stressed channel.
        achieved: Knob-reported achieved metrics for this level.
        budget: Effective budget (K6-budget overrides the config's).
        model: Registered model key.
        rep: Repetition index.
        seed: Seed of this repetition.
        label: Cell label for the progress line.
        progress: Print a progress line.

    Returns:
        The tidy-row payload (with the achieved extras) and the trajectory.
    """
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
        acq_label=cfg.acquisition.name,
        rep=rep,
        knob=knob_name,
        level=float(level),
        achieved=achieved,
    )
    if progress:
        print(
            f"[stress_sweep] {label} snr={achieved.get('achieved_snr_db', float('nan')):6.2f}dB "
            f"rec_regret={row.recommended_regret:.4f} ({time.time() - t0:.1f}s)",
            flush=True,
        )
    return row_payload(row, achieved), result.trajectory


def build_tidy_rows(
    cfg: ExperimentConfig,
    run_tag: str,
    *,
    progress: bool = True,
    store: CellStore | None = None,
) -> tuple[list[TidyRow], dict[tuple[Any, ...], dict[str, Any]], list[dict[str, Any]]]:
    """Execute the whole sweep grid.

    For each channel, each knob level is applied once and reused by every model
    and repetition, so all models see *identical* stressed data — the comparison
    is paired, which is what the TOST analysis assumes.

    Args:
        cfg: Resolved experiment configuration.
        run_tag: Run tag written into every row.
        progress: Print a per-cell progress line to stdout.
        store: Cell cache; ``None`` means a fresh store under the config's cache
            root. Cells are persisted the moment they finish, so a killed run resumes.

    Returns:
        ``(rows, trajectories, extras)`` where ``extras`` holds the non-schema
        columns (achieved metrics) aligned with ``rows`` by position.
    """
    store = store or CellStore(cfg.cell_cache_root)
    knob = build_knob(cfg.knob.type, cfg.knob.levels, **cfg.knob.params)
    rows: list[TidyRow] = []
    trajectories: dict[tuple[Any, ...], dict[str, Any]] = {}
    extras: list[dict[str, Any]] = []
    skipped: list[str] = []

    for channel in _channels(cfg):
        floor_db = floor_snr_db(channel)
        if cfg.knob.targets_db is not None:
            # Solve the ladder against *this channel's* floor SNR, so the same
            # config means the same severity on a clean and a noisy dataset.
            try:
                calibrated = calibrate_levels(
                    knob,
                    channel,
                    cfg.knob.targets_db,
                    np.random.default_rng(seed_for(channel.label, knob.name, base_seed=cfg.seed)),
                )
            except KnobNotApplicable as exc:
                skipped.append(f"{channel.label} @ calibration: {exc}")
                print(f"[stress_sweep] skipping {skipped[-1]}", flush=True)
                continue
            levels = [c.level for c in calibrated]
            targets = [c.target_db for c in calibrated]
            # Apply each level with the seed its calibration used, so the realized
            # degradation equals the ladder the run reports rather than a re-roll.
            level_seeds = {c.level: c.seed for c in calibrated}
            print(
                f"[stress_sweep] {channel.label} floor={floor_db:.2f}dB -> "
                + ", ".join(
                    f"{c.target_db:+.0f}dB@{c.level:.4g}({c.achieved_db:+.1f}dB"
                    + ("" if c.resolved else " UNRESOLVED")
                    + ")"
                    for c in calibrated
                ),
                flush=True,
            )
            unreachable = [c for c in calibrated if not c.resolved]
            if unreachable:
                # A step-function response (K5 with few donor artefacts) cannot hit
                # intermediate targets; say so rather than plotting the miss silently.
                print(
                    f"[stress_sweep] WARNING {channel.label}: "
                    f"{len(unreachable)} target(s) unreachable "
                    f"({', '.join(f'{c.target_db:+.0f}dB' for c in unreachable)}); "
                    "each row still records its achieved SNR.",
                    flush=True,
                )
        else:
            levels = list(knob.levels)
            targets = [float("nan")] * len(levels)
            level_seeds = {}

        for level, target_db in zip(levels, targets):
            rng = np.random.default_rng(
                level_seeds.get(level)
                if level_seeds
                else seed_for(channel.label, knob.name, level, base_seed=cfg.seed)
            )
            try:
                stressed = knob.apply(channel, level, rng)
            except KnobNotApplicable as exc:
                # Some channels simply lack what a knob needs (no lab-flagged
                # artefacts for K5, too few sites for K6 dropout). Skip the cell,
                # keep the rest of the grid, and make the gap visible in the log.
                skipped.append(f"{channel.label} @ {knob.name}={level:g}: {exc}")
                print(f"[stress_sweep] skipping {skipped[-1]}", flush=True)
                continue
            achieved = knob.achieved(stressed)
            achieved["floor_snr_db"] = floor_db
            if np.isfinite(target_db):
                achieved["target_delta_snr_db"] = float(target_db)
            budget = knob.budget_for(level, cfg.budget)

            for model in cfg.models:
                for rep in range(cfg.n_reps):
                    seed = seed_for(channel.label, knob.name, level, model, rep, base_seed=cfg.seed)
                    label = f"{channel.label} {knob.name}={level:g} {model} rep{rep}"
                    identity = cell_identity(
                        cfg, stressed, model, cfg.acquisition, experiment="stress_sweep",
                        rep=rep, seed=seed, budget=budget, knob=knob.name,
                        level=float(level),
                        target_db=float(target_db) if np.isfinite(target_db) else None,
                    )
                    cell = store.run(
                        channel.dataset, "stress_sweep", identity, label,
                        lambda m=model, r=rep, s=seed, lb=label: _compute_cell(
                            cfg, run_tag, knob.name, level, stressed, achieved, budget, m, r, s, lb,
                            progress,
                        ),
                    )
                    if cell is None:
                        continue
                    row = row_from_payload(cell[0], run_tag)
                    rows.append(row)
                    extras.append(dict(cell[0]["extras"]))
                    trajectories[_results.make_trajectory_key(row)] = cell[1]
    print(
        f"[stress_sweep] cells: {store.hits} cached, {store.computed} computed, "
        f"{len(store.failures)} failed",
        flush=True,
    )
    if not rows:
        raise RuntimeError(
            "stress_sweep produced no rows: check dataset.subjects / dataset.emgs, "
            "that the raw data is present under dataset.data_root, and the skip list "
            f"({len(skipped)} cell(s) skipped): {skipped[:3]}"
        )
    if skipped:
        print(f"[stress_sweep] {len(skipped)} cell(s) skipped as not applicable", flush=True)
    return rows, trajectories, extras


def run_stress_sweep(
    config_path: str,
    overrides: list[str] | None = None,
    *,
    replot: bool = False,
    run_dir: str | None = None,
    use_cache: bool = True,
    only_cached: bool = False,
    on_cell: Callable[[], None] | None = None,
) -> str:
    """Run (or re-plot) one stress sweep.

    Args:
        config_path: Path to an experiment YAML.
        overrides: ``key=value`` overrides applied to the resolved config.
        replot: Skip execution and rebuild every figure and table from the
            existing ``tidy.csv`` in the run directory.
        run_dir: Explicit run directory; defaults to
            ``{output_root}/stress/{knob}/{dataset}/{family}-{tag}``.
        use_cache: ``False`` (``--no-cache``) neither reads nor writes the cell cache.
        only_cached: ``True`` (``--only-cached``) assembles outputs from cached cells
            without computing any.
        on_cell: Called once per cell served (computed or cached); drives the
            cluster-diagnostics throughput counter.

    Returns:
        Path to the run directory holding the deliverables.

    Raises:
        FileNotFoundError: With ``replot=True`` and no ``tidy.csv`` present.
        RuntimeError: If any cell failed (raised after every completed cell is
            written and cached).
    """
    cfg = load_experiment_config(config_path, overrides)
    target = run_dir or os.path.join(
        cfg.output_root, "stress", cfg.knob.type, cfg.dataset.name, f"{cfg.family}-{cfg.tag}"
    )
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
        store = CellStore(
            cfg.cell_cache_root, enabled=use_cache, only_cached=only_cached, on_cell=on_cell
        )
        rows, trajectories, extras = build_tidy_rows(cfg, run_tag, store=store)
        df = _results.rows_to_dataframe(rows, acquisition=cfg.acquisition.as_block())
        # Achieved metrics that are not part of the fixed schema (future knobs
        # may report e.g. amplitude ratio) ride along as extra columns.
        for key in sorted({k for e in extras for k in e} - set(df.columns)):
            df[key] = [e.get(key, np.nan) for e in extras]
        df.to_csv(tidy_path, index=False)
        _results.write_trajectories(target, trajectories)
        _results.write_config(target, resolved_dict(cfg))
        print(f"[stress_sweep] wrote {len(df)} rows -> {tidy_path}")
        store.raise_if_failed()

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
    parser.add_argument("--no-cache", action="store_true", help="Neither read nor write the cell cache.")
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
    # Upper bound: a knob that does not apply to a channel skips its cells.
    n_levels = len(cfg.knob.targets_db) if cfg.knob.targets_db is not None else len(
        build_knob(cfg.knob.type, cfg.knob.levels, **cfg.knob.params).levels
    )
    planned = (
        count_channels(cfg) * n_levels * len(cfg.models) * cfg.n_reps
        if diag_on and not args.replot
        else 0
    )
    with ClusterDiagnostics(
        tag=f"{cfg.family}-{cfg.tag}", device=cfg.device, n_planned=planned,
        enabled=diag_on,
    ) as diag:
        run_stress_sweep(
            args.config, args.overrides, replot=args.replot, run_dir=args.run_dir,
            use_cache=not args.no_cache, only_cached=args.only_cached,
            on_cell=diag.record_experiment,
        )
    return 0
