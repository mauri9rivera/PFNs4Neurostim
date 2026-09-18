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
from dataclasses import dataclass, field
from typing import Any, Sequence

import yaml

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
    """

    name: str
    subjects: tuple[int, ...]
    emgs: tuple[int, ...] | None = None
    data_root: str = "./data"


@dataclass(frozen=True)
class AcquisitionConfig:
    """Acquisition function and its parameters (P0.2 schema).

    Attributes:
        type: Acquisition name; must be a key of :data:`_ACQ_PARAMS`.
        params: Parameters for that type only; unknown keys raise.
        schedules: Optional per-parameter annealing schedules.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    schedules: dict[str, Any] = field(default_factory=dict)

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
        levels: Level ladder, or ``None`` for the knob's pre-registered default.
    """

    type: str
    levels: tuple[float, ...] | None = None


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
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    extra_acquisitions: tuple[AcquisitionConfig, ...] = ()
    source_path: str = ""

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
                resolved["acquisition"] = _load_group("acquisition", value)
            else:
                blocks = [_load_group("acquisition", name) for name in value]
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
    )
    knob_raw.pop("levels", None)
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
        },
        "models": list(cfg.models),
        "model_version": {name: model_version(name) for name in cfg.models},
        "model_params": cfg.model_params,
        "acquisition": cfg.acquisition.as_block(),
        "acquisitions": [a.as_block() for a in cfg.acquisitions],
        "knob": {
            "type": cfg.knob.type,
            "levels": None if cfg.knob.levels is None else list(cfg.knob.levels),
        },
        "budget": cfg.budget,
        "n_init": cfg.n_init,
        "n_reps": cfg.n_reps,
        "gt_mode": cfg.gt_mode,
        "device": cfg.device,
        "seed": cfg.seed,
        "output_root": cfg.output_root,
        "equivalence_margin": cfg.equivalence_margin,
        "source_path": cfg.source_path,
    }
    # Guarantee plain Python types (no numpy scalars) reach the YAML dump.
    return json.loads(json.dumps(out, default=str))
