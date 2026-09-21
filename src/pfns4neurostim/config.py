"""YAML config composition and validation.

An experiment YAML composes config *groups* through a ``defaults:`` block and may
override any resolved key inline::

    defaults:
      dataset: nhp
      model: [tabpfn_v2_5, gp_mll, gp_naive]
      acquisition: ei
    knob:
      type: k2_snr
      levels: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    budget: 50

Groups live in ``configs/{dataset,model,acquisition}/<name>.yaml``. Everything is
validated into frozen dataclasses, and the **resolved** mapping is written to each
run's ``config.yaml`` so a result is always traceable to the exact settings that
produced it (P0.1 model version, P0.2 acquisition block, P0.3 budget semantics).

Unknown keys raise rather than being ignored — a typo in a config must never
silently run the wrong experiment.
"""
from __future__ import annotations

import copy
import json
import os
import platform
from dataclasses import dataclass, field
from typing import Any, Sequence

import yaml

from .data.preprocessing import DEFAULT_NORMALIZATION, get_normalization

__all__ = [
    "DatasetConfig",
    "AcquisitionConfig",
    "KnobConfig",
    "ExperimentConfig",
    "load_experiment_config",
    "resolved_dict",
    "CONFIG_ROOT",
]

#: Root of the config-group tree, overridable for tests.
CONFIG_ROOT: str = os.environ.get("PFNS4NEUROSTIM_CONFIG_ROOT", "configs")




@dataclass(frozen=True)
class DatasetConfig:
    """Dataset selection.

    Attributes:
        name: Dataset name (``'nhp'``, ``'rat'``, ``'spinal'``, ``'5d_rat'``).
        subjects: Subject indices to evaluate.
        emgs: EMG indices, or ``None`` for every EMG of each subject.
        data_root: Directory holding the raw ``.mat`` trees. Cluster jobs set
            this to ``$SLURM_TMPDIR/data``.
        normalization: Preprocessing mode, a key of
            :data:`pfns4neurostim.data.preprocessing.NORMALIZATIONS`. Logged in
            the resolved config and in every tidy row.
    """

    name: str
    subjects: tuple[int, ...]
    emgs: tuple[int, ...] | None = None
    data_root: str = "./data"
    normalization: str = DEFAULT_NORMALIZATION

    def __post_init__(self) -> None:
        """Reject an unregistered normalization at load time.

        Raises:
            ValueError: If ``normalization`` is not registered.
        """
        get_normalization(self.normalization)


@dataclass(frozen=True)
class AcquisitionConfig:
    """Acquisition function and its parameters (P0.2 schema).

    Attributes:
        type: Acquisition name; must be a key of :data:`_ACQ_PARAMS`.
        params: Parameters for that type only; unknown keys raise.
        schedules: Optional per-parameter annealing schedules.
        label: Config-group name (e.g. ``'ucb_k2'``), so several configurations of
            one type stay distinguishable in tables and plots.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    schedules: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    @property
    def name(self) -> str:
        """Display/grouping name: the config-group name, else the type."""
        return self.label or self.type

    def __post_init__(self) -> None:
        """Validate type, params and schedules against the acquisition registry.

        Delegating here means the registry is the single definition of what an
        acquisition type accepts: adding a type or a parameter never requires a
        parallel edit in the config layer.

        Raises:
            KeyError: If the type is not registered.
            ValueError: If a parameter or schedule key is not declared by the type.
        """
        from .acquisition.registry import build_acquisition  # noqa: PLC0415 - cycle

        build_acquisition(self.type, self.params, self.schedules)

    def as_block(self) -> dict[str, Any]:
        """Return the P0.2 acquisition block, as logged into the tidy CSV."""
        return {"type": self.type, "params": dict(self.params), "schedules": dict(self.schedules)}


@dataclass(frozen=True)
class KnobConfig:
    """Stress-knob selection.

    Attributes:
        type: Registered knob name (``'k2_snr'``, ...).
        levels: Explicit level ladder, or ``None`` for the knob's default.
        targets_db: SNR-degradation targets in dB (e.g. ``[0, -1, -2, -4]``). When
            given, the level achieving each target is solved **per channel**, so a
            sweep is comparable across datasets whose floor SNRs differ. Mutually
            exclusive with an explicit ``levels`` ladder.
        params: Knob-specific parameters, e.g. ``{source: heavy_tail}`` for K5.
    """

    type: str
    levels: tuple[float, ...] | None = None
    targets_db: tuple[float, ...] | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a ladder specified two ways at once.

        Raises:
            ValueError: If both ``levels`` and ``targets_db`` are given, or a
                target is positive.
        """
        if self.levels is not None and self.targets_db is not None:
            raise ValueError(
                "knob.levels and knob.targets_db are mutually exclusive: a ladder is "
                "either explicit levels or SNR-degradation targets solved per channel."
            )
        if self.targets_db is not None and any(t > 0 for t in self.targets_db):
            raise ValueError(
                f"knob.targets_db entries are SNR degradations and must be <= 0 dB, "
                f"got {list(self.targets_db)}."
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully resolved experiment configuration.

    Attributes:
        experiment: Experiment type, e.g. ``'stress_sweep'``.
        family: Experiment family, used by the aggregator to group run dirs.
        tag: Short run tag appended to the run directory name.
        dataset: Dataset selection.
        models: Model keys to compare, in legend order.
        model_params: Per-model constructor parameters.
        acquisition: Acquisition block.
        knob: Stress-knob block.
        budget: Total queries including ``n_init`` (P0.3).
        n_init: Random initial queries.
        n_reps: BO repetitions per cell.
        gt_mode: ``'full_mean'`` or ``'split_half'`` (P0.7).
        device: Torch device string.
        seed: Base seed; every cell derives its own seed from it.
        output_root: Root for outputs, normally ``output``.
        equivalence_margin: Pre-registered TOST margin in range-normalized
            regret units, used by the robustness table.
        cache_version: Invalidation stamp of the cell cache; bump it to discard
            every cached cell. Part of every cell identity. Version 2
            (2026-09-21) discards the cells computed while the loop was
            forbidden to re-query observed sites.
        cache_root: Cell-cache directory; empty means ``{output_root}/cells``.
        source_path: Path of the YAML this was loaded from.
    """

    experiment: str
    family: str
    tag: str
    dataset: DatasetConfig
    models: tuple[str, ...]
    acquisition: AcquisitionConfig
    knob: KnobConfig
    budget: int = 50
    n_init: int = 5
    n_reps: int = 5
    gt_mode: str = "full_mean"
    device: str = "cpu"
    seed: int = 42
    output_root: str = "output"
    equivalence_margin: float = 0.05
    cache_version: int = 2
    cache_root: str = ""
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    extra_acquisitions: tuple[AcquisitionConfig, ...] = ()
    source_path: str = ""

    @property
    def cell_cache_root(self) -> str:
        """Resolved cell-cache directory."""
        return self.cache_root or os.path.join(self.output_root, "cells")

    @property
    def acquisitions(self) -> tuple[AcquisitionConfig, ...]:
        """Every acquisition to run: the sweep list, or just the single block.

        ``bo_benchmark`` sweeps acquisition types (Hyp 0's ragged table); every
        other experiment uses exactly one, so this collapses to a 1-tuple.
        """
        return self.extra_acquisitions or (self.acquisition,)

    def __post_init__(self) -> None:
        """Validate cross-field constraints (budget semantics, non-empty grid)."""
        if self.budget <= self.n_init:
            raise ValueError(
                f"budget ({self.budget}) must exceed n_init ({self.n_init}); budget is "
                "the total number of queries including the initial design (P0.3)."
            )
        if self.n_reps < 1:
            raise ValueError(f"n_reps must be >= 1, got {self.n_reps}.")
        if not self.models:
            raise ValueError("models is empty; nothing to compare.")
        if self.equivalence_margin <= 0:
            raise ValueError(
                f"equivalence_margin must be > 0, got {self.equivalence_margin}. "
                "It is pre-registered in range-normalized regret units."
            )


def _read_yaml(path: str) -> dict[str, Any]:
    """Read a YAML file into a dict.

    Args:
        path: File path.

    Returns:
        Parsed mapping (empty for an empty file).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document is not a mapping.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping, got {type(data).__name__}.")
    return data


def _load_group(group: str, name: str) -> dict[str, Any]:
    """Load one config group file.

    Args:
        group: Group directory name (``'dataset'``, ``'model'``, ``'acquisition'``).
        name: File stem within the group.

    Returns:
        The group's mapping.
    """
    return _read_yaml(os.path.join(CONFIG_ROOT, group, f"{name}.yaml"))


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into a copy of ``base``.

    Args:
        base: Base mapping.
        update: Overriding mapping.

    Returns:
        The merged mapping; nested dicts merge, scalars and lists replace.
    """
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_override(token: str) -> tuple[list[str], Any]:
    """Parse one ``dotted.key=value`` CLI override.

    Values are parsed as YAML, so ``2``, ``2.5``, ``true``, ``[1, 2]`` and
    ``cuda`` all do the obvious thing.

    Args:
        token: The override token.

    Returns:
        ``(key_path, value)``.

    Raises:
        ValueError: If the token has no ``=``.
    """
    if "=" not in token:
        raise ValueError(f"Override {token!r} must be of the form key=value.")
    key, raw = token.split("=", 1)
    return key.strip().split("."), yaml.safe_load(raw)


def _apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply dotted-key overrides to a resolved mapping.

    Args:
        cfg: Resolved config mapping.
        overrides: Tokens of the form ``dotted.key=value``.

    Returns:
        A new mapping with the overrides applied.

    Raises:
        KeyError: If a key path does not exist — overrides must not invent keys.
    """
    out = copy.deepcopy(cfg)
    for token in overrides:
        path, value = _parse_override(token)
        node: Any = out
        for part in path[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"Override path {'.'.join(path)!r} does not exist in the config.")
            node = node[part]
        if not isinstance(node, dict) or path[-1] not in node:
            raise KeyError(f"Override path {'.'.join(path)!r} does not exist in the config.")
        node[path[-1]] = value
    return out


def _compose(path: str) -> dict[str, Any]:
    """Compose an experiment YAML with its ``defaults:`` groups.

    Args:
        path: Experiment YAML path.

    Returns:
        The resolved mapping, before dataclass validation.

    Raises:
        ValueError: On an unknown ``defaults`` group.
    """
    raw = _read_yaml(path)
    defaults = raw.pop("defaults", {}) or {}
    resolved: dict[str, Any] = {}

    for group, value in defaults.items():
        if group == "dataset":
            resolved["dataset"] = _load_group("dataset", value)
        elif group == "acquisition":
            if isinstance(value, str):
                resolved["acquisition"] = {**_load_group("acquisition", value), "label": value}
            else:
                blocks = [{**_load_group("acquisition", name), "label": name} for name in value]
                if not blocks:
                    raise ValueError(f"{path}: defaults.acquisition is an empty list.")
                resolved["acquisition"] = blocks[0]
                resolved["acquisitions"] = blocks
        elif group == "model":
            names = [value] if isinstance(value, str) else list(value)
            resolved["models"] = names
            params: dict[str, Any] = {}
            for name in names:
                group_cfg = _load_group("model", name)
                params[name] = group_cfg.get("params", {})
            resolved["model_params"] = params
        else:
            raise ValueError(
                f"Unknown defaults group {group!r} in {path}; "
                "expected one of dataset, model, acquisition."
            )
    return _deep_merge(resolved, raw)


def load_experiment_config(
    path: str,
    overrides: Sequence[str] | None = None,
) -> ExperimentConfig:
    """Load, compose, override and validate an experiment config.

    Args:
        path: Experiment YAML path.
        overrides: ``dotted.key=value`` tokens from ``--set``.

    Returns:
        The validated :class:`ExperimentConfig`.

    Raises:
        ValueError: On unknown keys or invalid values.
    """
    merged = _compose(path)
    if overrides:
        merged = _apply_overrides(merged, list(overrides))

    ds_raw = dict(merged.pop("dataset", {}))
    acq_raw = dict(merged.pop("acquisition", {}))
    knob_raw = dict(merged.pop("knob", {}))

    dataset = DatasetConfig(
        name=ds_raw.pop("name"),
        subjects=tuple(int(s) for s in ds_raw.pop("subjects")),
        emgs=(
            None
            if ds_raw.get("emgs") in (None, "all")
            else tuple(int(e) for e in ds_raw.pop("emgs"))
        ),
        data_root=os.path.expandvars(str(ds_raw.pop("data_root", "./data"))),
        normalization=str(ds_raw.pop("normalization", DEFAULT_NORMALIZATION)),
    )
    ds_raw.pop("emgs", None)
    if ds_raw:
        raise ValueError(f"Unknown dataset config key(s): {sorted(ds_raw)}.")

    def _acq(block: dict[str, Any]) -> AcquisitionConfig:
        """Build and validate one acquisition block."""
        data = dict(block)
        built = AcquisitionConfig(
            type=data.pop("type"),
            params=dict(data.pop("params", {}) or {}),
            schedules=dict(data.pop("schedules", {}) or {}),
            label=str(data.pop("label", "") or ""),
        )
        if data:
            raise ValueError(f"Unknown acquisition config key(s): {sorted(data)}.")
        return built

    acquisition = _acq(acq_raw)
    extra_acquisitions = tuple(_acq(b) for b in (merged.pop("acquisitions", []) or []))

    knob = KnobConfig(
        type=knob_raw.pop("type", "nominal"),
        levels=(
            None if knob_raw.get("levels") is None else tuple(float(v) for v in knob_raw.pop("levels"))
        ),
        targets_db=(
            None
            if knob_raw.get("targets_db") is None
            else tuple(float(v) for v in knob_raw.pop("targets_db"))
        ),
        params=dict(knob_raw.pop("params", {}) or {}),
    )
    knob_raw.pop("levels", None)
    knob_raw.pop("targets_db", None)
    if knob_raw:
        raise ValueError(f"Unknown knob config key(s): {sorted(knob_raw)}.")

    models = tuple(merged.pop("models"))
    model_params = {k: dict(v or {}) for k, v in (merged.pop("model_params", {}) or {}).items()}

    known = {
        "experiment",
        "family",
        "tag",
        "budget",
        "n_init",
        "n_reps",
        "gt_mode",
        "device",
        "seed",
        "output_root",
        "equivalence_margin",
        "cache_version",
        "cache_root",
    }
    unknown = set(merged) - known
    if unknown:
        raise ValueError(
            f"Unknown config key(s) {sorted(unknown)} in {path}; known keys: {sorted(known)}."
        )

    return ExperimentConfig(
        experiment=merged.pop("experiment", "stress_sweep"),
        family=merged.pop("family", "stress-sweep"),
        tag=merged.pop("tag", "run"),
        dataset=dataset,
        models=models,
        acquisition=acquisition,
        extra_acquisitions=extra_acquisitions,
        knob=knob,
        model_params=model_params,
        device=os.path.expandvars(str(merged.pop("device", "cpu"))),
        **{k: v for k, v in merged.items()},
        source_path=os.path.abspath(path),
    )


def _host_info() -> dict[str, Any]:
    """Describe the machine a run executes on, for result provenance.

    Local and cluster cells with the same seed are statistically equivalent but not bitwise
    identical (different GPU floating point), so every run records where it ran.

    Returns:
        ``{'node': hostname, 'cuda_device': GPU name or None, 'slurm_job_id': id or None}``.
    """
    device: str | None = None
    try:
        import torch  # noqa: PLC0415 - optional, keeps config import light

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - provenance must never fail a finished run (e.g. CUDA half-visible)
        device = None
    return {
        "node": platform.node(),
        "cuda_device": device,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def resolved_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Serialize a config for ``config.yaml``, JSON-round-tripped for plain types.

    Args:
        cfg: The resolved configuration.

    The mapping always carries ``model_version`` (P0.1) and the verbatim
    ``acquisition`` block (P0.2), which ``evaluation.results.write_config``
    requires, so every run directory records exactly which model version and
    acquisition settings produced it.

    Returns:
        A plain nested mapping safe for ``yaml.safe_dump``.
    """
    from .models.registry import model_version  # noqa: PLC0415 - avoid import cycle

    out = {
        "experiment": cfg.experiment,
        "family": cfg.family,
        "tag": cfg.tag,
        "dataset": {
            "name": cfg.dataset.name,
            "subjects": list(cfg.dataset.subjects),
            "emgs": None if cfg.dataset.emgs is None else list(cfg.dataset.emgs),
            "data_root": cfg.dataset.data_root,
            "normalization": cfg.dataset.normalization,
        },
        "models": list(cfg.models),
        "model_version": {name: model_version(name) for name in cfg.models},
        "model_params": cfg.model_params,
        "acquisition": cfg.acquisition.as_block(),
        "acquisitions": [a.as_block() for a in cfg.acquisitions],
        "knob": {
            "type": cfg.knob.type,
            "levels": None if cfg.knob.levels is None else list(cfg.knob.levels),
            "targets_db": None if cfg.knob.targets_db is None else list(cfg.knob.targets_db),
            "params": cfg.knob.params,
        },
        "budget": cfg.budget,
        "n_init": cfg.n_init,
        "n_reps": cfg.n_reps,
        "gt_mode": cfg.gt_mode,
        "device": cfg.device,
        "seed": cfg.seed,
        "output_root": cfg.output_root,
        "equivalence_margin": cfg.equivalence_margin,
        "cache_version": cfg.cache_version,
        "cache_root": cfg.cell_cache_root,
        "source_path": cfg.source_path,
        "host": _host_info(),
    }
    # Guarantee plain Python types (no numpy scalars) reach the YAML dump.
    return json.loads(json.dumps(out, default=str))
