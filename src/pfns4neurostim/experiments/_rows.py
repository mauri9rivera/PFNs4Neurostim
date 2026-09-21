"""Shared construction of tidy rows from a :class:`~..evaluation.bo_runner.BOResult`.

Both experiment runners (``bo_benchmark`` and ``stress_sweep``) produce rows of
the same schema; the only difference is whether a stress knob was applied. Keeping
the mapping in one place means a schema change is a one-line edit rather than a
hunt through every runner.
"""
from __future__ import annotations

import math
from typing import Any

from ..data.channels import ChannelData
from ..evaluation.bo_runner import BOResult
from ..evaluation.results import TidyRow

__all__ = ["build_row"]


def _clean(value: Any) -> Any:
    """Map a non-finite metric to None ('not computed') rather than a NaN row value.

    ``TidyRow`` refuses non-finite metrics, which is the fail-fast rule: a NaN that
    *arrived* as a number is a computation bug. A genuinely undefined statistic
    (Spearman of a constant vector, latency with no steps) is reported as None.

    Args:
        value: Candidate metric value.

    Returns:
        The value, or None when it is not a finite number.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        return None
    return value


def build_row(
    result: BOResult,
    channel: ChannelData,
    *,
    run_tag: str,
    experiment: str,
    model: str,
    acq_type: str,
    rep: int,
    acq_label: str | None = None,
    knob: str | None = None,
    level: float | None = None,
    achieved: dict[str, float] | None = None,
) -> TidyRow:
    """Assemble one tidy row from a finished BO run.

    Args:
        result: The BO result.
        channel: The channel the run used (stressed, if a knob was applied).
        run_tag: Run directory tag.
        experiment: Experiment type (``'bo_benchmark'`` or ``'stress_sweep'``).
        model: Canonical model key.
        acq_type: Acquisition type name.
        acq_label: Acquisition config name; defaults to ``acq_type``.
        rep: Repetition index.
        knob: Stress knob name, or None outside a stress sweep.
        level: Knob level, or None.
        achieved: Knob-reported achieved metrics (e.g. ``achieved_snr_db``).

    Returns:
        The populated :class:`TidyRow`.
    """
    row = result.row
    achieved = achieved or {}
    return TidyRow(
        run_tag=run_tag,
        experiment=experiment,
        dataset=channel.dataset,
        demo=channel.demo,
        normalization=channel.normalization,
        subject=channel.subject,
        emg=channel.emg,
        model=model,
        model_version=row["model_version"],
        acq_type=acq_type,
        acq_label=acq_label or acq_type,
        device=str(row.get("device", "")),
        knob=knob,
        level=None if level is None else float(level),
        gt_mode=channel.gt_mode,
        rep=int(rep),
        r2=_clean(row.get("r2")),
        spearman=_clean(row.get("spearman")),
        exploration_score=_clean(row.get("exploration_score")),
        # final_regret is kept as an alias of the recommended-site regret so older
        # aggregation code keeps working; the three co-primary regrets are explicit.
        final_regret=_clean(row.get("recommended_regret")),
        recommended_regret=_clean(row.get("recommended_regret")),
        best_queried_regret=_clean(row.get("best_queried_regret")),
        cumulative_regret=_clean(row.get("cumulative_regret")),
        top1_hit=_clean(row.get("top1_hit")),
        top3_hit=_clean(row.get("top3_hit")),
        opt_distance=_clean(row.get("opt_distance")),
        coverage_50=_clean(row.get("coverage_50")),
        coverage_90=_clean(row.get("coverage_90")),
        ece=_clean(row.get("ece")),
        nll=_clean(row.get("nll")),
        crps=_clean(row.get("crps")),
        mean_query_latency_s=_clean(row.get("mean_query_latency_s")),
        median_query_latency_s=_clean(row.get("median_query_latency_s")),
        total_time_s=_clean(row.get("total_time_s")),
        achieved_snr_db=_clean(achieved.get("achieved_snr_db")),
        n_sites=channel.n_sites,
        budget=row["budget"],
        n_init=row["n_init"],
        seed=row["seed"],
    )
