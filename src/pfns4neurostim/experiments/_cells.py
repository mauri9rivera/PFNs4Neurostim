"""Cell identity and (de)serialization shared by the cache-backed runners."""
from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from ..acquisition.registry import build_acquisition
from ..config import AcquisitionConfig, ExperimentConfig
from ..data.channels import ChannelData
from ..evaluation.results import TidyRow
from ..models.registry import model_version

__all__ = ["cell_identity", "row_from_payload", "count_channels"]


def count_channels(cfg: ExperimentConfig) -> int:
    """Count the (subject, EMG) channels a config selects, for progress reporting.

    Args:
        cfg: Resolved experiment configuration.

    Returns:
        Number of channels (an upper bound: preprocessing may skip a channel).
    """
    from ..data.channels import count_emgs  # noqa: PLC0415 - loads raw data

    if cfg.dataset.emgs is not None:
        return len(cfg.dataset.subjects) * len(cfg.dataset.emgs)
    return sum(
        count_emgs(cfg.dataset.name, s, cfg.dataset.data_root) for s in cfg.dataset.subjects
    )


def cell_identity(
    cfg: ExperimentConfig,
    channel: ChannelData,
    model: str,
    acq: AcquisitionConfig,
    *,
    experiment: str,
    rep: int,
    seed: int,
    budget: int,
    knob: str | None = None,
    level: float | None = None,
    target_db: float | None = None,
) -> dict[str, Any]:
    """Build the identity of one BO repetition: everything that can change its result.

    Args:
        cfg: Resolved experiment configuration.
        channel: The (possibly stressed) channel the cell runs on.
        model: Registered model key.
        acq: Acquisition block of this cell.
        experiment: Experiment type.
        rep: Repetition index.
        seed: Seed of this repetition.
        budget: Effective budget (K6-budget overrides the config's).
        knob: Stress knob name, or ``None``.
        level: Knob level, or ``None``.
        target_db: Calibrated SNR-degradation target, or ``None``.

    Returns:
        A JSON-serializable identity dict.
    """
    _, resolved = build_acquisition(acq.type, acq.params, acq.schedules)
    return {
        "experiment": experiment,
        "dataset": channel.dataset,
        "subject": channel.subject,
        "emg": channel.emg,
        "model": model,
        "model_version": model_version(model),
        "model_params": cfg.model_params.get(model, {}),
        "device": cfg.device,
        "acq_type": acq.type,
        "acq_params_resolved": {
            f.name: getattr(resolved, f.name) for f in fields(resolved) if f.name != "schedules"
        },
        "acq_schedules": acq.schedules,
        "knob": knob,
        "knob_params": cfg.knob.params if knob else None,
        "level": level,
        "target_db": target_db,
        "gt_mode": channel.gt_mode,
        "normalization": channel.normalization,
        "budget": budget,
        "n_init": cfg.n_init,
        "rep": rep,
        "seed": seed,
        "cache_version": cfg.cache_version,
    }


def row_from_payload(payload: dict[str, Any], run_tag: str) -> TidyRow:
    """Rebuild a :class:`TidyRow` from a cached payload under the current run tag.

    The tag is deliberately not part of a cell identity (renaming a run must not
    invalidate its results), so a cached row is re-stamped with the requesting run's.

    Args:
        payload: Value produced by :func:`~pfns4neurostim.evaluation.cache.row_payload`.
        run_tag: Tag of the run consuming the cell.

    Returns:
        The tidy row.
    """
    return replace(TidyRow(**payload["row"]), run_tag=run_tag)
