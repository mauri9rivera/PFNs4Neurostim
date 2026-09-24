"""Cell identity, (de)serialization and ground-truth expansion shared by the cache-backed runners."""
from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from ..acquisition.registry import build_acquisition
from ..config import AcquisitionConfig, ExperimentConfig
from ..data.channels import ChannelData
from ..data.ground_truth import ground_truth_instances, split_half_reliability
from ..evaluation.results import TidyRow
from ..models.registry import model_version
from ..seeding import rng_for

__all__ = ["cell_identity", "row_from_payload", "count_channels", "gt_instances"]


def count_channels(cfg: ExperimentConfig, shard: tuple[int, int] | None = None) -> int:
    """Count the (subject, EMG) channels a config selects, for progress reporting.

    Args:
        cfg: Resolved experiment configuration.
        shard: ``(i, n)`` to count only that shard's channels.

    Returns:
        Number of channels (an upper bound: preprocessing may skip a channel).
    """
    from ..data.channels import count_emgs, shard_size  # noqa: PLC0415 - loads raw data

    if cfg.dataset.emgs is not None:
        total = len(cfg.dataset.subjects) * len(cfg.dataset.emgs)
    else:
        total = sum(
            count_emgs(cfg.dataset.name, s, cfg.dataset.data_root) for s in cfg.dataset.subjects
        )
    return shard_size(total, shard)


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
    identity = {
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
    # Split-half cells only: added conditionally so every full-mean identity (and hence
    # every existing cached cell) is unchanged.
    if channel.split_id is not None:
        identity["split_id"] = channel.split_id
        identity["gt_n_splits"] = cfg.gt_n_splits
    return identity


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


def gt_instances(cfg: ExperimentConfig, channel: ChannelData) -> list[ChannelData]:
    """Expand a full-mean channel into the ground-truth instances of ``cfg.gt_mode``.

    ``full_mean`` returns ``[channel]`` unchanged. ``split_half`` (task #7, P0.7) returns
    ``2 * cfg.gt_n_splits`` cross-fitted instances drawn from a stream keyed by the
    channel label, each carrying the channel's split-half reliability in ``meta`` so
    the tidy row can report the R² noise ceiling.

    Args:
        cfg: Resolved experiment configuration.
        channel: A full-mean channel.

    Returns:
        The instances; repetition ``i`` uses ``instances[i mod len]``.
    """
    if cfg.gt_mode == "full_mean":
        return [channel]
    instances = ground_truth_instances(
        channel, cfg.gt_mode, rng=rng_for(channel.label, "gt_split", base_seed=cfg.seed),
        n_splits=cfg.gt_n_splits,
    )
    rel = split_half_reliability(
        channel.Y_trials, rng_for(channel.label, "gt_reliability", base_seed=cfg.seed),
        n_splits=cfg.gt_n_splits,
    )
    for inst in instances:
        inst.meta["gt_r_half"] = rel["r_half"]
        inst.meta["gt_reliability"] = rel["spearman_brown"]
    return instances
