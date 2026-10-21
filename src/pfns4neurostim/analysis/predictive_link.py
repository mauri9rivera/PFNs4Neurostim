"""Predictive link (roadmap M8): per-channel mechanism metrics -> regret.

Model: ``outcome ~ predictor + (1 | dataset/subject)``. The slope is estimated by the
*within* estimator (predictor and outcome centred inside each dataset x subject group), which
is the fixed-intercept limit of the random-intercept model and needs no extra dependency;
its uncertainty is a cluster bootstrap that resamples subjects (the independent unit), so
channels of one animal never count as independent evidence. When ``statsmodels`` is
installed the random-intercept MixedLM slope is reported alongside for comparison.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

__all__ = ["channel_mechanism_table", "predictive_link"]

_CHANNEL: list[str] = ["dataset", "subject", "emg"]


def channel_mechanism_table(
    cells: pd.DataFrame,
    metrics: Sequence[str],
    *,
    level: float | None = None,
) -> pd.DataFrame:
    """Per-channel medians of per-cell mechanism metrics (the metrics come from the config).

    Args:
        cells: One row per analysis cell with the channel keys, ``level`` and the metric columns.
        metrics: Columns to summarise; absent ones are skipped.
        level: Keep one stress level (``None`` pools every level).

    Returns:
        One row per channel.
    """
    sub = cells if level is None else cells[np.isclose(cells["level"], level)]
    cols = [m for m in metrics if m in sub]
    return sub.groupby(_CHANNEL)[cols].median().reset_index()


def predictive_link(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    *,
    groups: Sequence[str] = ("dataset", "subject"),
    cluster: Sequence[str] = ("dataset", "subject"),
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Within-group slope of ``outcome`` on ``predictor`` with a cluster-bootstrap CI.

    Args:
        df: One row per channel with ``predictor``, ``outcome`` and the group columns.
        predictor: Mechanism metric (e.g. ``'saturation_index_median'``).
        outcome: Regret or breakdown column.
        groups: Random-intercept grouping (centred out).
        cluster: Resampling unit of the bootstrap.
        n_boot: Bootstrap replicates.
        ci: Interval mass.
        seed: Bootstrap seed.

    Returns:
        ``{'slope', 'ci_low', 'ci_high', 'spearman', 'n_channels', 'n_clusters', 'mixedlm_slope'}``.

    Raises:
        ValueError: With fewer than 3 usable channels.
    """
    data = df.dropna(subset=[predictor, outcome]).copy()
    if len(data) < 3:
        raise ValueError(f"predictive_link: need >= 3 channels with {predictor} and {outcome}, got {len(data)}.")

    def _slope(frame: pd.DataFrame) -> float:
        x = frame[predictor] - frame.groupby(list(groups))[predictor].transform("mean")
        y = frame[outcome] - frame.groupby(list(groups))[outcome].transform("mean")
        denom = float((x ** 2).sum())
        return float("nan") if denom == 0 else float((x * y).sum() / denom)

    slope = _slope(data)
    rng = np.random.default_rng(seed)
    keys = data[list(cluster)].drop_duplicates().to_records(index=False).tolist()
    by_key = {k: g for k, g in data.groupby(list(cluster))}
    boots = []
    for _ in range(n_boot):
        pick = [keys[i] for i in rng.integers(0, len(keys), size=len(keys))]
        frame = pd.concat(
            [by_key[k].assign(_rep=j) for j, k in enumerate(pick)], ignore_index=True
        )
        # Each resampled cluster is its own group, so duplicates do not merge.
        frame["_grp"] = frame["_rep"]
        x = frame[predictor] - frame.groupby("_grp")[predictor].transform("mean")
        y = frame[outcome] - frame.groupby("_grp")[outcome].transform("mean")
        denom = float((x ** 2).sum())
        if denom > 0:
            boots.append(float((x * y).sum() / denom))
    lo, hi = (np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2]) if boots else (float("nan"),) * 2)
    mixed = None
    try:  # optional: the random-intercept fit itself
        import statsmodels.formula.api as smf  # noqa: PLC0415

        grp = data[list(groups)].astype(str).agg("/".join, axis=1)
        fit = smf.mixedlm(f"{outcome} ~ {predictor}", data.assign(_g=grp), groups="_g").fit()
        mixed = float(fit.params[predictor])
    except Exception:  # noqa: BLE001 - statsmodels absent or singular fit: report the within slope only
        mixed = None
    return {
        "predictor": predictor,
        "outcome": outcome,
        "slope": slope,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "spearman": float(spearmanr(data[predictor], data[outcome]).correlation),
        "n_channels": int(len(data)),
        "n_clusters": int(len(keys)),
        "mixedlm_slope": mixed,
    }
