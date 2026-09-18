"""Tidy BO-result schema, acquisition-block logging, and run-directory I/O.

Implements the P0.2 fix (roadmap "F3: ``acq_fn`` not logged") and the tidy
result schema from ``.claude/task_plan.md`` #1 Step 5 ("Tidy result schema"
block). Every BO-type experiment result is one :class:`TidyRow`: a
(run_tag, experiment, dataset, subject, emg, model, model_version, acq_type,
knob, level, gt_mode, rep) key tuple plus a fixed set of metric columns.

Regret units: ``final_regret``, ``cumulative_regret``, ``recommended_regret``,
and ``best_queried_regret`` are the three co-primary regret outcomes decided
2026-09-16 (recommended-site regret, best-queried simple regret, cumulative
regret) plus the pre-existing ``final_regret``. All four are reported in
shared, range-normalized units per P0.9 — this module does not perform that
normalization; callers must normalize before constructing a row.
"""
from __future__ import annotations

import dataclasses
import math
import os
import pickle
from typing import Any, Optional, Sequence, Union

import pandas as pd
import yaml

__all__ = [
    "KEY_COLUMNS",
    "METRIC_COLUMNS",
    "ALL_COLUMNS",
    "TidyRow",
    "flatten_acquisition_block",
    "rows_to_dataframe",
    "create_run_dir",
    "write_config",
    "read_config",
    "write_results_csv",
    "read_results_csv",
    "make_trajectory_key",
    "write_trajectories",
    "read_trajectories",
    "discover_run_dirs",
]

# ---------------------------------------------------------------------------
# Tidy schema definition
# ---------------------------------------------------------------------------

#: Columns that together identify one tidy row. All are required — a
#: :class:`TidyRow` cannot be constructed without them (dataclass fields with
#: no default), which is how "missing key column raises" is enforced.
KEY_COLUMNS: tuple[str, ...] = (
    "run_tag",
    "experiment",
    "dataset",
    "demo",
    "subject",
    "emg",
    "model",
    "model_version",
    "acq_type",
    "knob",
    "level",
    "gt_mode",
    "rep",
)

#: Metric/value columns. Each defaults to ``None`` in :class:`TidyRow`,
#: meaning "not computed by this experiment type"; ``None`` is written out
#: as ``float('nan')`` at serialization time (see :meth:`TidyRow.to_dict`).
#: A metric explicitly passed as NaN/Inf instead of ``None`` raises in
#: ``__post_init__`` — CLAUDE.md's fail-fast rule treats a silently produced
#: non-finite metric as a computation bug, distinct from "not computed".
METRIC_COLUMNS: tuple[str, ...] = (
    "r2",
    "spearman",
    "final_regret",
    "cumulative_regret",
    "recommended_regret",
    "best_queried_regret",
    "queries_to_target_90",
    "queries_to_target_95",
    "top1_hit",
    "top3_hit",
    "opt_distance",
    "coverage_50",
    "coverage_90",
    "ece",
    "nll",
    "crps",
    "mean_query_latency_s",
    "median_query_latency_s",
    "total_time_s",
    "achieved_snr_db",
    "n_sites",
    "budget",
    "n_init",
    "seed",
)

#: Full tidy-table column order: key columns followed by metric columns.
ALL_COLUMNS: tuple[str, ...] = KEY_COLUMNS + METRIC_COLUMNS


@dataclasses.dataclass(frozen=True)
class TidyRow:
    """One row of the tidy BO-result schema.

    Args:
        run_tag: Run directory tag, e.g. ``'nhp-vanilla-benchmark-11cc6'``.
        experiment: Experiment type, e.g. ``'bo_benchmark'``, ``'stress_sweep'``.
        dataset: Dataset identifier, e.g. ``'nhp'``, ``'rat'``, ``'5d_rat'``.
        subject: Integer subject index.
        emg: Integer EMG/channel index.
        model: Model identifier, e.g. ``'tabpfn_v2_5'``, ``'gp_mll'``.
        model_version: Explicit model version string (P0.1), e.g.
            ``'TabPFN v2.5'``.
        acq_type: Acquisition function type, matching
            ``acquisition['type']`` (P0.2) for this run.
        knob: Name of the stress-sweep knob varied in this row, or ``None``
            when the experiment is not a stress sweep.
        level: Value of ``knob`` for this row, or ``None`` when not
            applicable. Heterogeneous stress knobs may be numeric or
            categorical, hence ``str | float | int | None``.
        gt_mode: Ground-truth mode, ``'full_mean'`` or ``'split_half'`` (P0.7).
        demo: ``'demo2'`` for in-vivo channels, ``'demo1'`` for synthetic ones
            (roadmap Hyp B Demo 1 / Demo 2).
        rep: Integer BO-repetition index.
        r2: Final-prediction R² of the surrogate (secondary metric).
        spearman: Spearman correlation of the surrogate's final prediction.
        final_regret: Regret at the last BO step, range-normalized (P0.9).
        cumulative_regret: Cumulative regret over the BO trajectory,
            range-normalized (P0.9). Co-primary regret outcome.
        recommended_regret: Regret of the model's final recommended site,
            range-normalized (P0.9). Co-primary regret outcome.
        best_queried_regret: Best-queried simple regret over the trajectory,
            range-normalized (P0.9). Co-primary regret outcome.
        queries_to_target_90: Number of queries to reach 90% of the optimum.
        queries_to_target_95: Number of queries to reach 95% of the optimum.
        top1_hit: Whether the top-1 site was queried (1.0/0.0).
        top3_hit: Whether a top-3 site was queried (1.0/0.0).
        opt_distance: Distance from the recommended site to the true optimum.
        coverage_50: Empirical coverage of the 50% credible interval.
        coverage_90: Empirical coverage of the 90% credible interval.
        ece: Expected calibration error.
        nll: Negative log-likelihood of held-out predictions.
        crps: Continuous ranked probability score.
        mean_query_latency_s: Mean per-query wall-clock latency, in seconds.
        median_query_latency_s: Median per-query wall-clock latency, in seconds.
        achieved_snr_db: Achieved SNR of the (possibly stressed) channel, in dB.
            The canonical x-axis of every K2 figure (roadmap S2).
        n_sites: Number of candidate electrode sites in the pool.
        total_time_s: Total wall-clock time for the run, in seconds.
        budget: Total BO iteration count, including ``n_init`` (P0.3).
        n_init: Number of initial (non-acquisition-driven) queries.
        seed: Random seed used for this rep.
    """

    # --- key columns (required; no defaults) ---
    run_tag: str
    experiment: str
    dataset: str
    subject: int
    emg: int
    model: str
    model_version: str
    acq_type: str
    knob: Optional[str]
    level: Optional[Union[str, float, int]]
    gt_mode: str
    rep: int

    # --- key column with a default (Demo 2 = in vivo is the common case) ---
    demo: str = "demo2"

    # --- metric / value columns (optional; None => NaN at write time) ---
    r2: Optional[float] = None
    spearman: Optional[float] = None
    final_regret: Optional[float] = None
    cumulative_regret: Optional[float] = None
    recommended_regret: Optional[float] = None
    best_queried_regret: Optional[float] = None
    queries_to_target_90: Optional[float] = None
    queries_to_target_95: Optional[float] = None
    top1_hit: Optional[float] = None
    top3_hit: Optional[float] = None
    opt_distance: Optional[float] = None
    coverage_50: Optional[float] = None
    coverage_90: Optional[float] = None
    ece: Optional[float] = None
    nll: Optional[float] = None
    crps: Optional[float] = None
    mean_query_latency_s: Optional[float] = None
    median_query_latency_s: Optional[float] = None
    total_time_s: Optional[float] = None
    achieved_snr_db: Optional[float] = None
    n_sites: Optional[int] = None
    budget: Optional[int] = None
    n_init: Optional[int] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        for col in METRIC_COLUMNS:
            value = getattr(self, col)
            if value is None:
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise ValueError(
                    f"TidyRow.{col} is non-finite ({value!r}) for "
                    f"run_tag={self.run_tag!r}, rep={self.rep}. Metrics must be "
                    "finite; pass None if this experiment does not compute "
                    "this metric, per CLAUDE.md's fail-fast-on-NaN/Inf rule."
                )

    def to_dict(self) -> dict[str, Any]:
        """Flatten this row to a dict in :data:`ALL_COLUMNS` order.

        Returns:
            Dict with every key column verbatim and every metric column
            either its finite value or ``float('nan')`` when unset.
        """
        out: dict[str, Any] = {col: getattr(self, col) for col in KEY_COLUMNS}
        for col in METRIC_COLUMNS:
            value = getattr(self, col)
            out[col] = float("nan") if value is None else value
        return out


# ---------------------------------------------------------------------------
# Acquisition block flattening (P0.2)
# ---------------------------------------------------------------------------

#: Allowed top-level keys of the P0.2 acquisition config block.
_ACQUISITION_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"type", "params", "schedules"})


def flatten_acquisition_block(acquisition: dict[str, Any]) -> dict[str, Any]:
    """Flatten a resolved P0.2 acquisition block into tidy CSV columns.

    The P0.2 schema (``.claude/task_plan.md`` #1, "Acquisition config schema")
    is::

        acquisition:
          type: ucb
          params: {kappa: 2.0}
          schedules: {kappa: {kind: cosine, start: 7.5, end: 0.6}}

    Args:
        acquisition: Resolved acquisition block with keys drawn from
            ``{'type', 'params', 'schedules'}``. ``schedules`` is optional
            and defaults to ``{}``.

    Returns:
        Flat dict with ``acq_type``, one ``acq_param_<name>`` per key of
        ``params`` and one ``acq_schedule_<name>`` per key of ``schedules``,
        in deterministic (sorted-by-name) column order. Each schedule value
        is serialized as ``"<kind>(k1=v1,k2=v2,...)"`` with its non-``kind``
        keys sorted, e.g. ``"cosine(end=0.6,start=7.5)"``.

    Raises:
        ValueError: If ``acquisition`` has a key outside
            ``{'type', 'params', 'schedules'}``, is missing ``'type'`` or
            ``'params'``, or if any ``schedules`` key is not also a key of
            ``params``.
    """
    unknown = set(acquisition) - _ACQUISITION_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown acquisition config key(s) {sorted(unknown)}; allowed "
            f"top-level keys are {sorted(_ACQUISITION_TOP_LEVEL_KEYS)}."
        )
    if "type" not in acquisition:
        raise ValueError("acquisition block is missing required key 'type'.")
    if "params" not in acquisition:
        raise ValueError("acquisition block is missing required key 'params'.")

    acq_type = acquisition["type"]
    params: dict[str, Any] = acquisition["params"]
    schedules: dict[str, Any] = acquisition.get("schedules", {})

    unknown_schedules = set(schedules) - set(params)
    if unknown_schedules:
        raise ValueError(
            f"acquisition.schedules key(s) {sorted(unknown_schedules)} are not "
            f"declared parameters of acquisition.params ({sorted(params)})."
        )

    out: dict[str, Any] = {"acq_type": acq_type}
    for name in sorted(params):
        out[f"acq_param_{name}"] = params[name]
    for name in sorted(schedules):
        out[f"acq_schedule_{name}"] = _serialize_schedule(schedules[name])
    return out


def _serialize_schedule(schedule: dict[str, Any]) -> str:
    """Deterministically serialize one schedule dict to a compact string.

    Args:
        schedule: A schedule dict with a required ``'kind'`` key (one of
            ``constant``, ``linear``, ``cosine``, ``auto_dim``) plus its
            kind-specific parameters.

    Returns:
        ``"<kind>(k1=v1,k2=v2,...)"`` with parameter keys sorted, so the
        same schedule always serializes to the same string regardless of
        the input dict's key order.

    Raises:
        ValueError: If ``schedule`` is missing the required ``'kind'`` key.
    """
    if "kind" not in schedule:
        raise ValueError(f"Schedule block {schedule!r} is missing required key 'kind'.")
    kind = schedule["kind"]
    rest = {k: v for k, v in schedule.items() if k != "kind"}
    inner = ",".join(f"{k}={rest[k]}" for k in sorted(rest))
    return f"{kind}({inner})"


def rows_to_dataframe(
    rows: Sequence[TidyRow],
    acquisition: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Assemble tidy rows (and an optional shared acquisition block) into a DataFrame.

    Args:
        rows: Tidy rows belonging to one run.
        acquisition: Resolved acquisition config block (P0.2 schema) shared
            by every row in ``rows``. When given, the flattened
            ``acq_param_*`` / ``acq_schedule_*`` columns (see
            :func:`flatten_acquisition_block`) are broadcast onto every row.

    Returns:
        DataFrame with columns in :data:`ALL_COLUMNS` order, followed by any
        flattened acquisition columns in sorted order.

    Raises:
        ValueError: If ``acquisition`` is given and any row's ``acq_type``
            does not match ``acquisition['type']``.
    """
    flat_acq_cols: dict[str, Any] = {}
    if acquisition is not None:
        flat_acq = flatten_acquisition_block(acquisition)
        for row in rows:
            if row.acq_type != flat_acq["acq_type"]:
                raise ValueError(
                    f"Row acq_type={row.acq_type!r} (run_tag={row.run_tag!r}) "
                    f"does not match acquisition['type']={flat_acq['acq_type']!r}."
                )
        flat_acq_cols = {k: v for k, v in flat_acq.items() if k != "acq_type"}

    columns = list(ALL_COLUMNS) + sorted(flat_acq_cols)
    records = [dict(row.to_dict(), **flat_acq_cols) for row in rows]
    return pd.DataFrame.from_records(records, columns=columns)


# ---------------------------------------------------------------------------
# Run-directory I/O
# ---------------------------------------------------------------------------


def create_run_dir(runs_root: str, run_tag: str) -> str:
    """Create (if needed) and return ``{runs_root}/{run_tag}/``.

    Args:
        runs_root: Root directory for all run outputs, e.g. ``output/runs``.
            Never hardcoded — always passed in by the caller.
        run_tag: This run's tag/directory name.

    Returns:
        Path to the (now-existing) run directory.
    """
    run_dir = os.path.join(runs_root, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_config(run_dir: str, resolved_config: dict[str, Any]) -> str:
    """Write the fully resolved experiment config to ``{run_dir}/config.yaml``.

    This is the P0.1/P0.2 logging fix: every run's ``config.yaml`` must
    record the exact model version and acquisition settings used, so a run
    is never ambiguous about which model or acquisition function it used.

    Args:
        run_dir: Run directory, from :func:`create_run_dir`.
        resolved_config: Fully resolved config dict. Must contain
            ``'model_version'`` (P0.1) and ``'acquisition'`` (the P0.2 block,
            verbatim — not pre-flattened) so both are always recoverable
            from disk.

    Returns:
        Path to the written ``config.yaml``.

    Raises:
        ValueError: If ``'model_version'`` or ``'acquisition'`` is missing
            from ``resolved_config``.
    """
    missing = [k for k in ("model_version", "acquisition") if k not in resolved_config]
    if missing:
        raise ValueError(
            f"resolved_config is missing required key(s) {missing}; P0.1/P0.2 "
            "require 'model_version' and the 'acquisition' block to be "
            "logged in every run's config.yaml."
        )
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=True)
    return path


def read_config(run_dir: str) -> dict[str, Any]:
    """Read back a ``config.yaml`` written by :func:`write_config`.

    Args:
        run_dir: Run directory containing ``config.yaml``.

    Returns:
        The resolved config dict, exactly as written.
    """
    path = os.path.join(run_dir, "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_results_csv(run_dir: str, df: pd.DataFrame, filename: str = "results.csv") -> str:
    """Write a tidy results DataFrame to ``{run_dir}/results/{filename}``.

    Args:
        run_dir: Run directory, from :func:`create_run_dir`.
        df: DataFrame produced by :func:`rows_to_dataframe`.
        filename: CSV filename under ``{run_dir}/results/``.

    Returns:
        Path to the written CSV.
    """
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    df.to_csv(path, index=False)
    return path


def read_results_csv(path: str) -> pd.DataFrame:
    """Read back a tidy results CSV written by :func:`write_results_csv`.

    Note:
        Integer-valued columns (``subject``, ``emg``, ``rep``, ``budget``,
        ``n_init``, ``seed``) that contain any missing (NaN) entries in the
        written table are read back as ``float64`` rather than an integer
        dtype, because CSV/pandas cannot represent NaN in an integer column.
        This is a documented, value-exact round trip (100 round-trips to
        100.0, not a different number) but not a dtype-exact one for columns
        with missing entries.

    Args:
        path: Path to a CSV written by :func:`write_results_csv`.

    Returns:
        The DataFrame, as read by :func:`pandas.read_csv`.
    """
    return pd.read_csv(path)


def make_trajectory_key(row: TidyRow) -> tuple[Any, ...]:
    """Return ``row``'s key-column tuple, in :data:`KEY_COLUMNS` order.

    Args:
        row: A tidy row.

    Returns:
        Tuple usable as a dict key into a trajectories mapping (see
        :func:`write_trajectories`).
    """
    return tuple(getattr(row, col) for col in KEY_COLUMNS)


def write_trajectories(
    run_dir: str,
    trajectories: dict[tuple[Any, ...], dict[str, Any]],
    filename: str = "trajectories.pkl",
) -> str:
    """Write per-step trajectories, keyed by the tidy-row key tuple, to a pickle.

    Args:
        run_dir: Run directory, from :func:`create_run_dir`.
        trajectories: Mapping from a :data:`KEY_COLUMNS`-ordered key tuple
            (see :func:`make_trajectory_key`) to a dict of per-step data,
            e.g. ``{'regret': [...], 'times': [...], 'recommendations': [...]}``.
        filename: Pickle filename, written directly under ``run_dir``.

    Returns:
        Path to the written pickle file.
    """
    path = os.path.join(run_dir, filename)
    with open(path, "wb") as f:
        pickle.dump({"key_columns": KEY_COLUMNS, "trajectories": trajectories}, f)
    return path


def read_trajectories(path: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Read back a trajectories pickle written by :func:`write_trajectories`.

    Args:
        path: Path to a pickle written by :func:`write_trajectories`.

    Returns:
        The ``trajectories`` mapping, exactly as written.

    Raises:
        ValueError: If the pickle's recorded key-column order does not match
            the current :data:`KEY_COLUMNS` (schema-drift guard).
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if tuple(payload["key_columns"]) != KEY_COLUMNS:
        raise ValueError(
            f"Trajectory pickle at {path!r} was written with key columns "
            f"{payload['key_columns']}, which no longer matches the current "
            f"KEY_COLUMNS {KEY_COLUMNS}. Re-generate the run or migrate the "
            "pickle before reading it."
        )
    return payload["trajectories"]


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def _path_components(path: str) -> tuple[str, ...]:
    """Split a path into its normalized components, dropping the drive/root."""
    norm = os.path.normpath(path)
    _drive, rest = os.path.splitdrive(norm)
    return tuple(part for part in rest.split(os.sep) if part not in ("", os.sep))


def _is_under_archive(path: str) -> bool:
    """Return whether ``'archive'`` appears as a path component of ``path``."""
    return "archive" in _path_components(path)


def discover_run_dirs(
    runs_root: str,
    family: Optional[str] = None,
    include_archive: bool = False,
) -> list[str]:
    """List run directories under ``runs_root``.

    Mirrors the ``{dataset}-{family}-{hash}`` naming convention used
    elsewhere (e.g. ``src/aggregate.py``). By default, any directory whose
    path contains an ``archive`` path component is skipped — this closes
    task #2 Step 2 for the new-code path: ``output/archive/`` is never
    picked up unless a caller explicitly opts in.

    Args:
        runs_root: Root directory containing one subdirectory per run, e.g.
            ``output/runs``. Never hardcoded — always passed in by the
            caller.
        family: If given, only directories whose name contains
            ``f'-{family}-'`` (matching the ``{dataset}-{family}-{hash}``
            convention) are returned. ``None`` returns all run directories.
        include_archive: If ``False`` (default), directories under an
            ``archive`` path component are skipped. Pass ``True`` to
            include them deliberately.

    Returns:
        Sorted list of run directory paths (each ``os.path.join(runs_root,
        name)``), one per discovered run.
    """
    if not os.path.isdir(runs_root):
        return []

    result: list[str] = []
    for name in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, name)
        if not os.path.isdir(run_dir):
            continue
        if not include_archive and _is_under_archive(run_dir):
            continue
        if family is not None and f"-{family}-" not in name:
            continue
        result.append(run_dir)
    return result
