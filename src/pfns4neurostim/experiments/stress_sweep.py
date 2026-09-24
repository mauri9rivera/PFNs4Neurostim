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

    python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_channel_nhp.yaml
    python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_channel_nhp.yaml --replot
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
from ..data.channels import parse_shard, ChannelData, iter_channels
from ..data.ground_truth import instance_for_rep
from ..data.stress import KnobNotApplicable, build_knob, calibrate_levels, floor_snr_db
from ..data.synthetic_neurostim import meta_features, synthetic_channels
from ..diagnostics import ClusterDiagnostics, diagnostics_enabled
from ..evaluation import results as _results
from ..evaluation import shards as _shards
from ..evaluation.cache import CellStore, row_payload
from ..evaluation.bo_runner import run_channel_bo
from ..evaluation.results import TidyRow
from ..seeding import seed_for
from ._cells import cell_identity, count_channels, gt_instances, row_from_payload
from ._rows import build_row

__all__ = ["run_stress_sweep", "build_tidy_rows", "main"]


def _channels(cfg: ExperimentConfig, shard: tuple[int, int] | None = None) -> Iterable[ChannelData]:
    """Yield the nominal channels selected by the config (always full-mean ground truth).

    ``dataset.demo: demo1`` swaps each recorded channel for a synthetic map fitted to it
    (roadmap S0), so a Demo 1 sweep runs on the same channel list as its Demo 2 twin.

    Args:
        cfg: Resolved experiment configuration.
        shard: ``(i, n)`` to yield only that shard's channels.

    Returns:
        One :class:`~pfns4neurostim.data.channels.ChannelData` per (subject, EMG).

    Raises:
        ValueError: For split-half ground truth on Demo 1, where the ground truth is exact.
    """
    real = iter_channels(
        cfg.dataset.name,
        cfg.dataset.subjects,
        cfg.dataset.emgs,
        data_root=cfg.dataset.data_root,
        gt_mode="full_mean",  # split-half instances are expanded per channel (task #7)
        normalization=cfg.dataset.normalization,
        shard=shard,
    )
    if cfg.dataset.demo == "demo2":
        return real
    if cfg.gt_mode != "full_mean":
        raise ValueError("gt_mode='split_half' is meaningless on Demo 1: synthetic ground truth is exact.")
    return synthetic_channels(
        real, n_hotspots=cfg.dataset.generator_hotspots, seed=cfg.seed,
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
    shard: tuple[int, int] | None = None,
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

    for channel in _channels(cfg, shard):
        floor_db = floor_snr_db(channel)
        if cfg.knob.targets_db is not None:
            # Solve the ladder against *this channel's* floor SNR, so the same
            # config means the same severity on a clean and a noisy dataset.
            try:
                calibrated = calibrate_levels(
                    knob,
                    channel,
                    cfg.knob.targets_db,
                    np.random.default_rng(seed_for(channel.label, knob.seed_key, base_seed=cfg.seed)),
                )
            except KnobNotApplicable as exc:
                skipped.append(f"{channel.label} @ calibration: {exc}")
                print(f"[stress_sweep] skipping {skipped[-1]}", flush=True)
                continue
            levels = [c.level for c in calibrated]
            targets = [c.target_db for c in calibrated]
            level_seeds = {c.level: c.seed for c in calibrated}
            print(
                f"[stress_sweep] {channel.label} floor={floor_db:.2f}dB -> "
                + ", ".join(f"{c.target_db:+.0f}dB@{c.level:.4g}" for c in calibrated),
                flush=True,
            )
        else:
            levels = list(knob.levels)
            targets = [float("nan")] * len(levels)
            level_seeds = {}

        # Ground-truth instances: [channel] for full_mean, 2R cross-fitted halves for
        # split_half; repetition i runs on instance i mod len (task #7, roadmap S6).
        instances = gt_instances(cfg, channel)
        for level, target_db in zip(levels, targets):
            level_seed = (
                level_seeds[level] if level_seeds
                else seed_for(channel.label, knob.seed_key, level, base_seed=cfg.seed)
            )
            try:
                # Every instance is stressed with the same stream, and each stressed
                # instance is shared by all models and reps: the comparison stays paired.
                stressed_all = [knob.apply(inst, level, np.random.default_rng(level_seed)) for inst in instances]
            except KnobNotApplicable as exc:
                # Some channels simply lack what a knob needs (too few electrodes to
                # survive K6 failure, a recorded map for K1). Skip the cell, keep the
                # rest of the grid, and make the gap visible in the log.
                skipped.append(f"{channel.label} @ {knob.name}={level:g}: {exc}")
                print(f"[stress_sweep] skipping {skipped[-1]}", flush=True)
                continue
            achieved_all = []
            for inst, stressed in zip(instances, stressed_all):
                achieved = knob.achieved(stressed)
                achieved["floor_snr_db"] = floor_db if inst is channel else floor_snr_db(inst)
                if np.isfinite(target_db):
                    achieved["target_delta_snr_db"] = float(target_db)
                achieved_all.append(achieved)
            budget = knob.budget_for(level, cfg.budget)

            for model in cfg.models:
                for rep in range(cfg.n_reps):
                    i = instance_for_rep(rep, len(instances))
                    stressed, achieved = stressed_all[i], achieved_all[i]
                    seed = seed_for(channel.label, knob.seed_key, level, model, rep, base_seed=cfg.seed)
                    label = f"{channel.label} {knob.name}={level:g} {model} rep{rep}"
                    identity = cell_identity(
                        cfg, stressed, model, cfg.acquisition, experiment="stress_sweep",
                        rep=rep, seed=seed, budget=budget, knob=knob.name,
                        level=float(level),
                        target_db=float(target_db) if np.isfinite(target_db) else None,
                    )
                    cell = store.run(
                        channel.dataset, "stress_sweep", identity, label,
                        lambda m=model, r=rep, s=seed, lb=label, st=stressed, ac=achieved: _compute_cell(
                            cfg, run_tag, knob.name, level, st, ac, budget, m, r, s, lb, progress,
                        ),
                    )
                    if cell is None:
                        continue
                    row = row_from_payload(cell[0], run_tag)
                    rows.append(row)
                    extras.append({**cell[0]["extras"], **_shards.host_columns([cell[0]]).iloc[0].to_dict()})
                    trajectories[_results.make_trajectory_key(row)] = cell[1]
    print(
        f"[stress_sweep] cells: {store.hits} cached, {store.computed} computed, "
        f"{len(store.failures)} failed",
        flush=True,
    )
    if not rows and shard is not None and not store.failures and not skipped:
        # An empty shard (more lanes than channels) is a clean no-op, not an error.
        print(f"[stress_sweep] shard {shard[0]}/{shard[1]} owns no channels; nothing to do.", flush=True)
        return [], {}, []
    if not rows:
        raise RuntimeError(
            "stress_sweep produced no rows: check dataset.subjects / dataset.emgs, "
            "that the raw data is present under dataset.data_root, and the skip list "
            f"({len(skipped)} cell(s) skipped): {skipped[:3]}"
        )
    if skipped:
        print(f"[stress_sweep] {len(skipped)} cell(s) skipped as not applicable", flush=True)
    return rows, trajectories, extras


def _generator_validation(cfg: ExperimentConfig, shard: tuple[int, int] | None) -> pd.DataFrame:
    """Meta-features of every selected real channel and of its fitted synthetic twin (S0).

    Args:
        cfg: Resolved Demo 1 configuration.
        shard: Channel shard, as for the sweep.

    Returns:
        One row per (channel, demo) with the :func:`meta_features` columns.
    """
    records = []
    real = list(iter_channels(
        cfg.dataset.name, cfg.dataset.subjects, cfg.dataset.emgs, data_root=cfg.dataset.data_root,
        normalization=cfg.dataset.normalization, shard=shard,
    ))
    synth = synthetic_channels(real, n_hotspots=cfg.dataset.generator_hotspots, seed=cfg.seed,
                               normalization=cfg.dataset.normalization)
    for r, s in zip(real, synth):
        for ch in (r, s):
            records.append({"demo": ch.demo, "subject": ch.subject, "emg": ch.emg, **meta_features(ch)})
    return pd.DataFrame.from_records(records)


def _render_bridge(target: str, demo2_dir: str, cfg: ExperimentConfig, df: pd.DataFrame) -> list[str]:
    """Place the real channels of a Demo 2 run on this Demo 1 run's regime surface (S10).

    Args:
        target: This (Demo 1) run directory.
        demo2_dir: Run directory of the matching Demo 2 sweep.
        cfg: Resolved Demo 1 configuration.
        df: This run's tidy frame.

    Returns:
        Paths written.

    Raises:
        FileNotFoundError: If either run lacks its tidy table or trajectories.
    """
    from ..visualization import stress as stress_figs  # noqa: PLC0415 - matplotlib import
    from ..visualization import traces as T  # noqa: PLC0415

    real_path = os.path.join(demo2_dir, "tidy.csv")
    frame = T.load_trace_frame(target)
    if frame is None or not os.path.exists(real_path):
        raise FileNotFoundError(f"--bridge needs {real_path} and this run's trajectories.pkl.")
    real_df = pd.read_csv(real_path)
    written = []
    for model in [m for m in cfg.models if m != stress_figs.REFERENCE_MODEL]:
        written += stress_figs.plot_bridge(
            frame, df, real_df, os.path.join(target, "bridge", model),
            knob=cfg.knob.type, margin=cfg.equivalence_margin, model=model,
        )
    return written


def run_stress_sweep(
    config_path: str,
    overrides: list[str] | None = None,
    *,
    replot: bool = False,
    run_dir: str | None = None,
    use_cache: bool = True,
    only_cached: bool = False,
    on_cell: Callable[[], None] | None = None,
    shard: tuple[int, int] | None = None,
    compute_only: bool = False,
    bridge: str | None = None,
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
        shard: ``(i, n)`` runs only every n-th channel starting at ``i``. A shard is compute-only.
        compute_only: Compute (and cache) cells but write no ``tidy.csv`` or figures: only a
            provenance record in ``{run_dir}/shards/``. Every job of an experiment shares one run
            directory; assemble the union with a final ``--only-cached`` run without ``--shard``.
        bridge: Run directory of the Demo 2 sweep matching this Demo 1 sweep; draws the real
            channels on this run's regime surface (roadmap S10).

    Returns:
        Path to the (single, merged) run directory holding the deliverables.

    Raises:
        FileNotFoundError: With ``replot=True`` and no ``tidy.csv`` present.
        RuntimeError: If any cell failed (raised after every completed cell is
            written and cached).
    """
    cfg = load_experiment_config(config_path, overrides)
    compute_only = compute_only or shard is not None
    started = time.time()
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
        rows, trajectories, extras = build_tidy_rows(cfg, run_tag, store=store, shard=shard)
        if compute_only:
            record = _shards.write_shard_record(
                target, shard=shard, host=store.host, models=cfg.models, device=cfg.device,
                model_devices={m: p["device"] for m, p in cfg.model_params.items() if p.get("device")},
                cells_computed=store.computed, cells_cached=store.hits, cells_failed=len(store.failures),
                started=started, status="failed" if store.failures else "ok",
            )
            print(f"[stress_sweep] compute-only: shard record -> {record}", flush=True)
            store.raise_if_failed()
            return target
        if not rows:
            return target
        df = _results.rows_to_dataframe(rows, acquisition=cfg.acquisition.as_block())
        # Achieved metrics that are not part of the fixed schema (future knobs
        # may report e.g. amplitude ratio) ride along as extra columns.
        for key in sorted({k for e in extras for k in e} - set(df.columns)):
            df[key] = [e.get(key, np.nan) for e in extras]
        df.to_csv(tidy_path, index=False)
        _results.write_trajectories(target, trajectories)
        _results.write_config(target, {**resolved_dict(cfg), "shards": _shards.shard_registry(target)})
        print(f"[stress_sweep] wrote {len(df)} rows -> {tidy_path}")
        store.raise_if_failed()

    from ..visualization import stress as stress_figs  # deferred: matplotlib import

    written = stress_figs.render_all(
        df, target, knob=cfg.knob.type, dataset=cfg.dataset.name, margin=cfg.equivalence_margin
    )
    if cfg.dataset.demo == "demo1":
        features_path = os.path.join(target, "generator_validation.csv")
        features = (
            pd.read_csv(features_path) if replot and os.path.exists(features_path)
            else _generator_validation(cfg, shard)
        )
        written += stress_figs.plot_generator_validation(features, target)
    if bridge:
        written += _render_bridge(target, bridge, cfg, df)
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
    parser.add_argument(
        "--compute-only", action="store_true",
        help="Compute and cache cells; write only a shard provenance record (implied by --shard).",
    )
    parser.add_argument(
        "--bridge", default=None, metavar="DEMO2_RUN_DIR",
        help="Demo 1 runs only: overlay that Demo 2 run's real channels on the regime surface (S10).",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    cfg = load_experiment_config(args.config, args.overrides)
    diag_on = diagnostics_enabled(args.cluster_diag)
    # Upper bound: a knob that does not apply to a channel skips its cells.
    n_levels = len(cfg.knob.targets_db) if cfg.knob.targets_db is not None else len(
        build_knob(cfg.knob.type, cfg.knob.levels, **cfg.knob.params).levels
    )
    planned = (
        count_channels(cfg, args.shard) * n_levels * len(cfg.models) * cfg.n_reps
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
            on_cell=diag.record_experiment, shard=args.shard, compute_only=args.compute_only, bridge=args.bridge,
        )
    return 0
