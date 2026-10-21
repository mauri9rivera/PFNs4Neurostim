"""Predictive link (roadmap M8, **exploratory**): per-channel mechanism readouts -> BO outcomes.

Model: ``outcome ~ predictor + (1 | subject)`` over the NHP channels (user, 2026-10-19 / 2026-10-21). Slopes with
intervals and no confirmatory claim.

* **Slope.** GLS under the random intercept: the variance ratio ``rho = tau^2 / sigma^2`` is profiled by REML on a
  fixed grid (:data:`RE_RATIO_GRID`) and, at each grid point, the GLS fit is closed form after the Fuller-Battese
  quasi-demeaning ``z - theta_g zbar_g``, ``theta_g = 1 - 1 / sqrt(1 + n_g rho)``. No extra dependency
  (``statsmodels`` is not in the environment). With a single subject the random intercept is not identified and the
  fit is plain OLS. The *within* slope (predictor and outcome centred per subject, the fixed-intercept limit
  ``rho -> inf``) is reported alongside: it is immune to subject-level confounding.
* **Interval.** Channel bootstrap, stratified by subject: each subject's channels are resampled with replacement and
  keep their subject, so the subject count and sizes are those of the data (the decision of 2026-10-08: with three or
  four subjects a subject-level bootstrap is degenerate).
* **Predictors** (:data:`READOUTS`). Every OMEGA ``bayes_update`` readout of the linked engine, aggregated per anchor
  -> per cell (median over the cell's anchors of one pool) -> per channel (median over the cells at one stress level,
  every context size and draw unless ``context_ts`` narrows it). Each predictor is paired with a *reference*
  (:func:`reference_pairs`):

  - a **contrast** readout (Part B: ``transfer_slope``, ``footprint_r``, ``gap_change``, ``delta_crps``) is already a
    site-matched comparison with one reference (``headtohead.csv``'s ``reference`` column);
  - an **engine** readout (Part A coherence, ``saturation``, the C1 ``gain`` / ``far_field_share``) is the engine's own
    value (reference ``'none'``) or, with ``engine_contrasts``, the matched difference engine - reference in the same
    cell (and at the same anchor when the readout is per anchor), aggregated after the difference;
  - the **C1 reference** (GP-MLL frozen, a Part-B-only engine) is paired with the C1 readouts only
    (``transfer_slope``, ``footprint_r``, ``gain``, ``far_field_share``, ``saturation``) and never with a Part A
    coherence readout; the C1-only readouts ``gain`` / ``far_field_share`` (whose perfect-GP value is the
    reference's own) are paired with the C1 reference only.
* **Outcomes.** From a BO ``tidy.csv`` (:func:`channel_outcomes`; mean over repetitions, the median for the
  heavy-tailed ``r2``) and, for ``breakdown``, from a stress ``tidy.csv`` (:func:`channel_breakdown`: the per-channel
  :func:`~pfns4neurostim.evaluation.robustness.breakdown_point`, as the number of ladder levels passed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..evaluation.robustness import breakdown_point

__all__ = [
    "READOUTS",
    "NO_REFERENCE",
    "BREAKDOWN_OUTCOME",
    "Readout",
    "reference_pairs",
    "references_in_run",
    "cell_readouts",
    "channel_readouts",
    "channel_outcomes",
    "channel_breakdown",
    "predictive_link",
]

#: One channel; one subject (the random-intercept group: a dataset never shares an intercept with another).
CHANNEL: list[str] = ["dataset", "subject", "emg"]
SUBJECT: list[str] = ["dataset", "subject"]
#: One OMEGA analysis cell.
CELL: list[str] = ["dataset", "subject", "emg", "level", "context_t", "draw"]

#: Reference label of an engine readout taken on its own (no contrast).
NO_REFERENCE: str = "none"


@dataclass(frozen=True)
class Readout:
    """Where one mechanism readout lives and how it pairs with a reference.

    Attributes:
        part: ``'A'`` (coherence), ``'B'`` (head-to-head) or ``'C1'`` (GP-likeness only).
        table: ``'coherence_cell'`` (one row per cell x engine x pool) or ``'headtohead'`` (one row per anchor).
        kind: ``'engine'`` (the engine's own value) or ``'contrast'`` (a site-matched contrast with a reference).
        source: ``headtohead`` probe source (``'trial'`` or ``'ladder'``); ``None`` for coherence readouts.
        c1: Whether the readout is a C1 readout, i.e. may be paired with the C1 (Part-B-only) reference.
    """

    part: str
    table: str
    kind: str
    source: str | None
    c1: bool


#: Every predictor the link knows. The C1 flags mirror ``visualization.mechanism.C1_READOUTS``.
READOUTS: dict[str, Readout] = {
    "drift_abs": Readout("A", "coherence_cell", "engine", None, False),
    "total_var_ratio": Readout("A", "coherence_cell", "engine", None, False),
    "cov_asymmetry": Readout("A", "coherence_cell", "engine", None, False),
    "latent_frac": Readout("A", "coherence_cell", "engine", None, False),
    "sigma_recovery_rho": Readout("A", "coherence_cell", "engine", None, False),
    "transfer_slope": Readout("B", "headtohead", "contrast", "trial", True),
    "footprint_r": Readout("B", "headtohead", "contrast", "trial", True),
    "gap_change": Readout("B", "headtohead", "contrast", "trial", False),
    "delta_crps": Readout("B", "headtohead", "contrast", "trial", False),
    "saturation": Readout("B", "headtohead", "engine", "ladder", True),
    "gain": Readout("C1", "headtohead", "engine", "trial", True),
    "far_field_share": Readout("C1", "headtohead", "engine", "trial", True),
}

#: Display / iteration order of the references (others follow alphabetically).
REFERENCE_ORDER: tuple[str, ...] = ("gp_bayes", "gp_mll_refit", "gp_fixed", "gp_mll_frozen")

#: Outcome name of the per-channel breakdown point (from a stress run, not the BO tidy).
BREAKDOWN_OUTCOME: str = "breakdown"

#: How repetitions are reduced to one channel value; outcomes not listed use the mean. ``r2`` is heavy-tailed
#: (excess kurtosis up to ~490, ``visualization.stress.HEAVY_TAILED_METRICS``), so it takes the median.
OUTCOME_REDUCERS: dict[str, str] = {"r2": "median"}

#: Breakdown rule (mirrors ``visualization.stress.BREAKDOWN_TEST``, the S9 robustness table's rule).
BREAKDOWN_TEST: str = "noninferiority"

#: A stress tidy carrying this column fully populated has a calibrated ladder (dB of SNR degradation), on which a
#: more negative level is more severe (mirrors ``visualization.stress.CALIBRATED_TARGET_COLUMN``).
CALIBRATED_LADDER_COLUMN: str = "target_delta_snr_db"

#: REML grid of the random-intercept variance ratio tau^2 / sigma^2: 0 (pooled OLS) and a log grid wide enough
#: that its top end is the within estimator to within ~1e-3 for any NHP subject size.
RE_RATIO_GRID: np.ndarray = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 61)])

#: Fixed effects of the link model (intercept, slope): the REML degrees-of-freedom correction.
_N_FIXED: int = 2
#: Fewest channels a slope is fitted on.
MIN_CHANNELS: int = 3
#: Bootstrap replicates fitted per batch (bounds the [chunk, R, N] working arrays to a few MB).
_BOOT_CHUNK: int = 250


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------
def _ordered(refs: Sequence[str]) -> list[str]:
    """References in :data:`REFERENCE_ORDER`, unknown ones after it alphabetically."""
    known = [r for r in REFERENCE_ORDER if r in refs]
    return known + sorted(set(refs) - set(known))


def references_in_run(headtohead: pd.DataFrame, model: str) -> list[str]:
    """Every reference the linked engine was contrasted with in the run (``headtohead.csv``), in display order.

    Args:
        headtohead: ``headtohead.csv`` rows.
        model: Linked engine.

    Returns:
        Reference engines, without ``'none'``.
    """
    refs = headtohead.loc[headtohead["engine"] == model, "reference"].dropna().unique()
    return _ordered([str(r) for r in refs if r != NO_REFERENCE])


def reference_pairs(
    predictors: Sequence[str],
    references: Sequence[str],
    *,
    c1_references: Sequence[str],
    engine_contrasts: bool,
) -> list[tuple[str, str]]:
    """Every admissible (predictor, reference) pair, in predictor then reference order.

    Rules: an engine readout always gets its own value (``'none'``) and, with ``engine_contrasts``, a matched
    difference with each reference; a contrast readout gets one pair per reference; a C1 reference pairs only with
    C1 readouts (never with Part A); a C1-only readout pairs only with a C1 reference.

    Args:
        predictors: Readout names (keys of :data:`READOUTS`).
        references: Reference engines (``'none'`` is implicit and ignored here).
        c1_references: Engines that are C1 references only (Part B references never read in Part A or C2).
        engine_contrasts: Whether engine readouts are also contrasted with each reference.

    Returns:
        ``[(predictor, reference), ...]``.

    Raises:
        ValueError: On an unknown predictor.
    """
    unknown = [p for p in predictors if p not in READOUTS]
    if unknown:
        raise ValueError(f"predictive link: unknown predictor(s) {unknown}; known {sorted(READOUTS)}.")
    pairs: list[tuple[str, str]] = []
    for pred in predictors:
        r = READOUTS[pred]
        if r.kind == "engine":
            pairs.append((pred, NO_REFERENCE))
        for ref in _ordered([x for x in references if x != NO_REFERENCE]):
            is_c1_ref = ref in c1_references
            if is_c1_ref != r.c1 and (is_c1_ref or r.part == "C1"):
                continue                                # C1 reference x non-C1 readout, or C1-only readout x other
            if r.kind == "engine" and not engine_contrasts:
                continue
            pairs.append((pred, ref))
    return pairs


def _finite(values: pd.Series, what: str) -> None:
    """Fail fast on +-inf (NaN means "not computed" in every OMEGA table and is dropped downstream)."""
    arr = values.to_numpy(dtype=np.float64)
    if np.isinf(arr).any():
        raise ValueError(f"predictive link: {int(np.isinf(arr).sum())} infinite value(s) in {what}.")


def _matched(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], col: str, what: str) -> pd.DataFrame:
    """``left[col] - right[col]`` matched one-to-one on ``keys`` (site-/cell-matched before any aggregate)."""
    m = left[keys + [col]].merge(right[keys + [col]], on=keys, how="inner", suffixes=("", "_ref"),
                                 validate="one_to_one")
    if m.empty:
        raise ValueError(f"predictive link: no matched rows for {what}.")
    return m.assign(**{col: m[col] - m[f"{col}_ref"]})[keys + [col]]


def cell_readouts(
    coherence_cell: pd.DataFrame,
    headtohead: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    model: str,
    pool: str,
    c1_references: Sequence[str],
) -> pd.DataFrame:
    """One value per (cell, predictor, reference): per anchor first, median over the cell's anchors of ``pool``.

    Args:
        coherence_cell: ``coherence_cell.csv`` rows (Part A; one row per cell x engine x pool).
        headtohead: ``headtohead.csv`` rows (Part B / C1; one row per cell x engine x reference x anchor x source).
        pairs: From :func:`reference_pairs`.
        model: Linked engine.
        pool: Anchor pool (``'unobserved'`` or ``'observed'``); the two are never pooled.
        c1_references: C1-only references, refused with a Part A readout.

    Returns:
        Long frame: :data:`CELL` + ``predictor``, ``reference``, ``value``.

    Raises:
        ValueError: On a Part A readout paired with a C1 reference, a missing column, an absent reference, a
            non-unique match, or an infinite value.
    """
    coh = coherence_cell[coherence_cell["pool"] == pool]
    h2h = headtohead[headtohead["pool"] == pool]
    out: list[pd.DataFrame] = []
    for pred, ref in pairs:
        r = READOUTS[pred]
        what = f"{pred} vs {ref}"
        if r.part == "A" and ref in c1_references:
            raise ValueError(f"predictive link: {ref} is a Part B (C1) reference only and is never paired with the "
                             f"Part A coherence readout {pred}.")
        if r.table == "coherence_cell":
            if pred not in coh:
                raise ValueError(f"predictive link: coherence_cell.csv has no column {pred!r}.")
            own = coh[coh["engine"] == model]
            cells = own[CELL + [pred]] if ref == NO_REFERENCE else _matched(
                own, coh[coh["engine"] == ref], CELL, pred, what)
        else:
            if pred not in h2h:
                raise ValueError(f"predictive link: headtohead.csv has no column {pred!r}.")
            src = h2h[h2h["source"] == r.source]
            keys = CELL + ["anchor"]
            if r.kind == "contrast":
                per_anchor = src[(src["engine"] == model) & (src["reference"] == ref)][keys + [pred]]
            else:
                own = src[(src["engine"] == model) & (src["reference"] == NO_REFERENCE)]
                per_anchor = own[keys + [pred]] if ref == NO_REFERENCE else _matched(
                    own, src[(src["engine"] == ref) & (src["reference"] == NO_REFERENCE)], keys, pred, what)
            if per_anchor.empty:
                raise ValueError(f"predictive link: headtohead.csv has no {r.source} rows for {what} "
                                 f"(engine {model}, pool {pool}).")
            _finite(per_anchor[pred], what)
            cells = per_anchor.groupby(CELL)[pred].median().reset_index()        # anchors -> cell
        _finite(cells[pred], what)
        out.append(cells.rename(columns={pred: "value"}).assign(predictor=pred, reference=ref))
    if not out:
        return pd.DataFrame(columns=CELL + ["value", "predictor", "reference"])
    return pd.concat(out, ignore_index=True)


def channel_readouts(
    cells: pd.DataFrame,
    *,
    level: float,
    context_ts: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Per-channel medians of the per-cell readouts at one stress level.

    Args:
        cells: From :func:`cell_readouts`.
        level: Stress level kept.
        context_ts: Context sizes kept (``None`` = every size of the ladder).

    Returns:
        :data:`CHANNEL` + ``predictor``, ``reference``, ``value``, ``n_cells`` (non-NaN cells behind the median).
    """
    sub = cells[np.isclose(cells["level"].astype(float), float(level))]
    if context_ts is not None:
        sub = sub[sub["context_t"].astype(int).isin([int(t) for t in context_ts])]
    g = sub.groupby(CHANNEL + ["predictor", "reference"])["value"]
    return g.median().reset_index().merge(g.count().rename("n_cells").reset_index(),
                                          on=CHANNEL + ["predictor", "reference"])


# ---------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------
def _one_model(tidy: pd.DataFrame, model: str, acq_label: str | None, what: str) -> pd.DataFrame:
    """Rows of one model (and one acquisition), refusing an ambiguous tidy."""
    missing = [c for c in CHANNEL + ["model"] if c not in tidy]
    if missing:
        raise ValueError(f"predictive link: {what} has no column(s) {missing}.")
    sub = tidy[tidy["model"] == model]
    if acq_label is not None:
        sub = sub[sub["acq_label"] == acq_label]
    if sub.empty:
        raise ValueError(f"predictive link: {what} has no rows of model {model!r}"
                         + (f" with acq_label {acq_label!r}." if acq_label is not None else "."))
    if acq_label is None and "acq_label" in sub and sub["acq_label"].nunique() > 1:
        raise ValueError(f"predictive link: {what} holds several acquisitions for {model} "
                         f"({sorted(sub['acq_label'].unique())}); set bayes_update.link.acq_label.")
    return sub


def channel_outcomes(
    tidy: pd.DataFrame,
    outcomes: Sequence[str],
    *,
    model: str,
    acq_label: str | None = None,
) -> pd.DataFrame:
    """One value per channel and outcome from a BO ``tidy.csv``: mean over repetitions (median for ``r2``).

    Args:
        tidy: BO tidy frame (one row per channel x repetition).
        outcomes: Metric columns (e.g. ``recommended_regret``, ``cumulative_regret``, ``r2``).
        model: Model whose runs are linked.
        acq_label: Acquisition kept (``None`` = the only one present).

    Returns:
        :data:`CHANNEL` + one column per outcome.

    Raises:
        ValueError: On a missing outcome column, no rows of the model, several acquisitions, or an infinite value.
    """
    missing = [o for o in outcomes if o not in tidy]
    if missing:
        raise ValueError(f"predictive link: the BO tidy.csv has no outcome column(s) {missing}; "
                         f"columns: {sorted(tidy.columns)}.")
    sub = _one_model(tidy, model, acq_label, "the BO tidy.csv")
    for o in outcomes:
        _finite(sub[o], f"BO outcome {o}")
    g = sub.groupby(CHANNEL)
    cols = [g[o].agg(OUTCOME_REDUCERS.get(o, "mean")).rename(o) for o in outcomes]
    return pd.concat(cols, axis=1).reset_index()


def channel_breakdown(
    stress: pd.DataFrame,
    *,
    model: str,
    reference: str,
    metric: str,
    margin: float,
    acq_label: str | None = None,
) -> pd.DataFrame:
    """Per-channel breakdown point from a stress ``tidy.csv``, as the number of ladder levels passed.

    Each channel's model and reference runs are paired by (level, repetition) and handed to
    :func:`~pfns4neurostim.evaluation.robustness.breakdown_point` with the S9 rule (:data:`BREAKDOWN_TEST`). The
    outcome is the position of the breakdown level on the severity-ordered ladder -- the count of levels passed
    before the first failure -- and the ladder length when the channel never breaks down, so a robust channel is
    the top of the scale rather than a missing value.

    Args:
        stress: Stress tidy frame (one row per channel x level x model x repetition).
        model: Model whose breakdown is the outcome.
        reference: Reference model of the rule (S9: ``gp_mll``).
        metric: Lower-is-better metric of the rule.
        margin: Pre-registered non-inferiority margin, in the metric's units.
        acq_label: Acquisition kept (``None`` = the only one present).

    Returns:
        :data:`CHANNEL` + ``breakdown`` (levels passed), ``breakdown_level`` (NaN = none), ``n_levels``.

    Raises:
        ValueError: On missing columns, no paired rows, several knobs or demos, or channels on different ladders.
    """
    need = ["level", "rep", metric]
    missing = [c for c in need if c not in stress]
    if missing:
        raise ValueError(f"predictive link: the stress tidy.csv has no column(s) {missing}.")
    m = _one_model(stress, model, acq_label, "the stress tidy.csv")
    r = _one_model(stress, reference, acq_label, "the stress tidy.csv")
    for col in ("knob", "demo"):
        if col in stress and pd.concat([m[col], r[col]]).nunique() > 1:
            raise ValueError(f"predictive link: the stress tidy.csv mixes several {col}s; one breakdown per ladder.")
    _finite(m[metric], f"stress metric {metric}")
    _finite(r[metric], f"stress metric {metric}")
    keys = CHANNEL + ["level", "rep"]
    paired = m[keys + [metric]].merge(r[keys + [metric]], on=keys, suffixes=("", "_ref"), validate="one_to_one")
    if paired.empty:
        raise ValueError(f"predictive link: no (channel, level, rep) is shared by {model} and {reference}.")
    calibrated = CALIBRATED_LADDER_COLUMN in stress and bool(stress[CALIBRATED_LADDER_COLUMN].notna().all())
    ladders = paired.groupby(CHANNEL)["level"].apply(lambda s: tuple(sorted(s.astype(float).unique())))
    if ladders.nunique() > 1:
        raise ValueError("predictive link: channels of the stress run sit on different level ladders, so their "
                         "'levels passed' are not comparable.")
    rows: list[dict[str, Any]] = []
    for key, grp in paired.sort_values(keys).groupby(CHANNEL):
        per = {float(lv): g[metric].to_numpy(np.float64) for lv, g in grp.groupby("level")}
        ref = {float(lv): g[f"{metric}_ref"].to_numpy(np.float64) for lv, g in grp.groupby("level")}
        bp = breakdown_point(per, ref, margin=float(margin), severity_ascending=not calibrated, test=BREAKDOWN_TEST)
        ladder = bp["levels_tested"]                                               # severity order, [L]
        level = bp["breakdown_level"]
        passed = len(ladder) if np.isnan(level) else ladder.index(level)
        rows.append({**dict(zip(CHANNEL, key)), BREAKDOWN_OUTCOME: float(passed), "breakdown_level": level,
                     "n_levels": len(ladder)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the link model
# ---------------------------------------------------------------------------
def _re_fit(
    x: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    n_groups: int,
    rho_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """REML random-intercept GLS fits of ``y = b0 + b1 x + u_g + e`` on a grid of ``rho = tau^2 / sigma^2``.

    Batched over data sets that share one group layout (the point estimate, or every stratified bootstrap
    replicate at once).

    Args:
        x: Predictor, shape [B, N].
        y: Outcome, shape [B, N].
        g: Group code per channel in ``[0, n_groups)``, shape [N], shared by the B data sets.
        n_groups: Number of groups.
        rho_grid: Variance ratios tried, shape [R] (one group: only 0 is used, the ratio is not identified).

    Returns:
        ``(intercept, slope, rho_hat)``, each shape [B]; NaN for a data set whose predictor is constant.
    """
    rho = np.zeros(1) if n_groups == 1 else np.asarray(rho_grid, dtype=np.float64)    # [R]
    onehot = np.eye(n_groups)[g]                                                        # [N, G]
    counts = onehot.sum(0)                                                              # [G]
    xbar = (x @ onehot / counts)[:, g]                                                  # [B, N]
    ybar = (y @ onehot / counts)[:, g]                                                  # [B, N]
    theta = (1.0 - 1.0 / np.sqrt(1.0 + counts[None, :] * rho[:, None]))[:, g]          # [R, N]
    c = (1.0 - theta)[None]                                                             # [1, R, N] intercept column
    xs = x[:, None, :] - theta[None] * xbar[:, None, :]                                 # [B, R, N]
    ys = y[:, None, :] - theta[None] * ybar[:, None, :]                                 # [B, R, N]
    a, b, d = (c * c).sum(-1), (c * xs).sum(-1), (xs * xs).sum(-1)                      # [1, R], [B, R], [B, R]
    e, f = (c * ys).sum(-1), (xs * ys).sum(-1)                                          # [B, R] each
    det = a * d - b * b                                                                 # [B, R]
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = (d * e - b * f) / det                                                      # [B, R]
        b1 = (a * f - b * e) / det                                                      # [B, R]
        rss = ((ys - b0[..., None] * c - b1[..., None] * xs) ** 2).sum(-1)              # [B, R]
        ll = (-(x.shape[1] - _N_FIXED) / 2.0 * np.log(rss)
              - 0.5 * np.log1p(counts[None, :] * rho[:, None]).sum(-1)[None] - 0.5 * np.log(det))
    ok = np.isfinite(ll) & (det > 0)                                                    # [B, R]
    k = np.argmax(np.where(ok, ll, -np.inf), axis=1)                                    # [B]; all False -> 0
    rows = np.arange(x.shape[0])
    # a perfect fit (rss = 0 at every rho) has no finite likelihood: its rho = 0 fit (pooled OLS) is kept
    usable = (np.ptp(x, axis=1) > 0) & ((det[:, 0] > 0) | ok.any(axis=1))               # [B]
    nan = np.full(x.shape[0], np.nan)
    return (np.where(usable, b0[rows, k], nan), np.where(usable, b1[rows, k], nan), np.where(usable, rho[k], nan))


def _within(x: np.ndarray, y: np.ndarray, g: np.ndarray, n_groups: int) -> float:
    """Within-group slope (predictor and outcome centred per group); NaN when no group varies in ``x``."""
    counts = np.bincount(g, minlength=n_groups).astype(np.float64)                      # [G]
    xc = x - (np.bincount(g, x, n_groups) / counts)[g]                                  # [N]
    yc = y - (np.bincount(g, y, n_groups) / counts)[g]                                  # [N]
    denom = float((xc ** 2).sum())
    return float("nan") if denom == 0 else float((xc * yc).sum() / denom)


def predictive_link(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    *,
    groups: Sequence[str] = ("dataset", "subject"),
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    rho_grid: np.ndarray = RE_RATIO_GRID,
) -> dict[str, Any]:
    """Random-intercept slope of ``outcome`` on ``predictor`` with a subject-stratified channel-bootstrap CI.

    Args:
        df: One row per channel with ``predictor``, ``outcome`` and the group columns.
        predictor: Predictor column.
        outcome: Outcome column.
        groups: Random-intercept grouping (one subject).
        n_boot: Bootstrap replicates.
        ci: Interval mass.
        seed: Bootstrap seed.
        rho_grid: REML grid of tau^2 / sigma^2.

    Returns:
        ``slope``, ``intercept``, ``ci_low``, ``ci_high`` (percentile, of the slope), ``within_slope``,
        ``re_ratio`` (REML tau^2 / sigma^2; 0 = pooled OLS), ``spearman``, ``n_channels``, ``n_subjects``,
        ``n_boot_valid`` (replicates whose predictor still varied).

    Raises:
        ValueError: With fewer than :data:`MIN_CHANNELS` usable channels, a constant predictor or outcome, or an
            infinite value.
    """
    data = df.dropna(subset=[predictor, outcome])
    _finite(data[predictor], predictor)
    _finite(data[outcome], outcome)
    if len(data) < MIN_CHANNELS:
        raise ValueError(f"predictive_link: need >= {MIN_CHANNELS} channels with {predictor} and {outcome}, "
                         f"got {len(data)}.")
    x = data[predictor].to_numpy(np.float64)                                            # [N]
    y = data[outcome].to_numpy(np.float64)                                              # [N]
    for name, v in ((predictor, x), (outcome, y)):
        if np.ptp(v) == 0:
            raise ValueError(f"predictive_link: {name} is constant over the {len(v)} channels; no slope.")
    g = data.groupby(list(groups)).ngroup().to_numpy()                                  # [N]
    n_groups = int(g.max()) + 1
    order = np.argsort(g, kind="stable")                                                # strata contiguous
    x, y, g = x[order], y[order], g[order]
    intercept, slope, ratio = (float(v[0]) for v in _re_fit(x[None], y[None], g, n_groups, rho_grid))
    # stratified channel bootstrap: every replicate keeps each subject's channel count, so all replicates share
    # the group layout ``g`` and are fitted in one batch
    rng = np.random.default_rng(seed)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])                               # [G]
    sizes = np.diff(np.r_[starts, len(g)])                                              # [G]
    idx = np.concatenate([s + rng.integers(0, n, size=(int(n_boot), n)) for s, n in zip(starts, sizes)],
                         axis=1)                                                        # [n_boot, N]
    boots = np.concatenate([_re_fit(x[chunk], y[chunk], g, n_groups, rho_grid)[1]
                            for chunk in np.array_split(idx, -(-len(idx) // _BOOT_CHUNK))])  # [n_boot]
    boots = boots[np.isfinite(boots)]
    tail = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [tail, 1.0 - tail]) if boots.size else (float("nan"), float("nan"))
    return {
        "predictor": predictor,
        "outcome": outcome,
        "slope": slope,
        "intercept": intercept,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "within_slope": _within(x, y, g, n_groups),
        "re_ratio": ratio,
        "spearman": float(spearmanr(x, y).correlation),
        "n_channels": int(len(data)),
        "n_subjects": n_groups,
        "n_boot_valid": int(boots.size),
    }
