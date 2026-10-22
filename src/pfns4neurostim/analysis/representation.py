"""OBJECTIVE BETA statistics: belief targets, deconfounded alignment, smooth-field chance level, site matching (β1, β2).

This module composes the primitives of :mod:`analysis.cka`, :mod:`analysis.procrustes` and
:mod:`analysis.bayes_update` into the quantities the ``representation`` runner writes
(``.research_assistant/beta_cka_procrustes_review.md`` §4-§5, amendments D3c, D3d, D3e):

* **Belief targets** (:func:`target_grams`), one Gram per *question*, never merged (D3c): the **mean map** and the
  **SD map** of a model's predictive given D (centred linear kernels), its 32-quantile **belief embedding** (secondary)
  and, for a GP, its **latent posterior covariance**. Models are named by a short code (:data:`BELIEF_CODES`):
  ``own`` = TabPFN's final predictive, ``mll`` = GP-MLL refit, ``fixed`` = GP-fixed, ``bayes`` = GP-Bayes (optional,
  D3e). ``K_X`` (geometry, the confound itself) and ``K_GT`` (accuracy, a secondary reference) complete the set.
* **Alignment rows** (:func:`alignment_rows`): debiased CKA, 4-bandwidth deconfounded CKA (GP-fixed's covariance is an
  extra confound for covariance targets, D3d), NBS and the shape distance, a descriptive site-permutation null, and
  the effective rank of both sides.
* **Chance level** (:func:`surrogate_rows`): the same statistics for smooth-field surrogate representations.
* **Site matching** (:func:`site_match_rows`, :func:`lens_rows`): per-site W2 with its location / spread / shape split,
  computed site by site and only then aggregated (D3). These are output-space quantities, reportable under G6.
* **Channel inference** (:func:`channel_summary`): draws reduced within a channel, then the median over channels with
  a channel bootstrap that resamples within subject (D4).

Guardrail G6: the claim these numbers support is *decodability* ("the GP-MLL refit mean map is linearly readable from
the label token at block l beyond geometry"), never "GP-like"; no conclusion is drawn from the gap between two CKA
scores. The headline statistic is OPEN (``research_design.md`` §4, 2026-10-19): :data:`HEADLINE_STATISTICS` holds the
provisional one and the alternative, selected by config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import cka as C
from . import procrustes as P
from .bayes_update import site_w2_split

__all__ = [
    "BELIEF_CODES",
    "TARGET_KINDS",
    "FIXED_TARGETS",
    "HEADLINE_STATISTICS",
    "SITE_SETS",
    "Belief",
    "parse_target",
    "target_grams",
    "target_confounds",
    "GramStats",
    "alignment_rows",
    "surrogate_rows",
    "surrogate_floors",
    "pair_rows",
    "site_match_rows",
    "lens_rows",
    "add_headline",
    "channel_summary",
]

#: Belief code -> engine name. ``own`` is TabPFN's final predictive of the same forward pass as the states.
BELIEF_CODES: dict[str, str] = {"own": "tabpfn_v2_5", "mll": "gp_mll_refit", "fixed": "gp_fixed", "bayes": "gp_bayes"}

#: Target kinds of a belief: location, spread, the quantile embedding (secondary), the latent posterior covariance.
TARGET_KINDS: tuple[str, ...] = ("mean", "sd", "quant", "cov")

#: Targets that are not a belief.
FIXED_TARGETS: tuple[str, ...] = ("K_X", "K_GT")

#: Headline statistics (OPEN, 2026-10-19). ``deconfounded_vs_surrogate`` is the provisional headline (deconfounded
#: debiased CKA, read against the surrogate's upper quantile); ``floor_subtracted_raw`` is alternative (b) (raw
#: debiased CKA minus the surrogate median, the deconfounded value secondary).
HEADLINE_STATISTICS: tuple[str, ...] = ("deconfounded_vs_surrogate", "floor_subtracted_raw")

#: Site sets a statistic can be computed on: every site, or the sites outside the context (BETA §2.5, demoted).
SITE_SETS: tuple[str, ...] = ("all", "unobserved")

#: Statistics a surrogate replicate is scored on.
_SURROGATE_STATS: tuple[str, ...] = ("cka", "cka_deconf", "nbs")


@dataclass(frozen=True)
class Belief:
    """A model's predictive given D at every site.

    Attributes:
        mean: Predictive mean map, shape [M].
        sd: Predictive SD map (noise included), shape [M].
        quantiles: Belief embedding, shape [M, J].
        cov: Latent posterior covariance, shape [M, M], or ``None`` (TabPFN: no closed form; see β1b).
    """

    mean: np.ndarray
    sd: np.ndarray
    quantiles: np.ndarray
    cov: np.ndarray | None


def parse_target(name: str) -> tuple[str, str | None]:
    """Split a target name into ``(kind, belief code)``; ``(name, None)`` for :data:`FIXED_TARGETS`.

    Args:
        name: ``<kind>_<code>`` (e.g. ``mean_mll``, ``cov_fixed``) or one of :data:`FIXED_TARGETS`.

    Returns:
        The kind and the code.

    Raises:
        ValueError: On an unknown kind or code, or ``cov_own`` (TabPFN has no closed-form covariance; β1b).
    """
    if name in FIXED_TARGETS:
        return name, None
    kind, _, code = name.partition("_")
    if kind not in TARGET_KINDS or code not in BELIEF_CODES:
        raise ValueError(f"unknown representation target {name!r}: expected <kind>_<code> with kind in "
                         f"{TARGET_KINDS}, code in {sorted(BELIEF_CODES)}, or one of {FIXED_TARGETS}.")
    if kind == "cov" and code == "own":
        raise ValueError("target 'cov_own' is not available: TabPFN's implied covariance costs 96 anchors of refits "
                         "per cell and is the beta1b object (representation.posterior_cov), not a beta2 target.")
    return kind, code


def target_grams(
    targets: Sequence[str],
    beliefs: Mapping[str, Belief],
    X: np.ndarray,
    y_gt: np.ndarray,
) -> dict[str, np.ndarray]:
    """The Gram of every requested target over the sites (BETA §4, D3c / D3e).

    Args:
        targets: Target names (:func:`parse_target`).
        beliefs: Belief per code.
        X: Site coordinates, shape [M, D].
        y_gt: Ground-truth map, shape [M].

    Returns:
        ``{target: Gram [M, M]}``.

    Raises:
        KeyError: If a target's belief is missing.
    """
    fixed = C.target_kernels(X, y_gt)
    out: dict[str, np.ndarray] = {}
    for name in targets:
        kind, code = parse_target(name)
        if code is None:
            out[name] = fixed[name]
            continue
        b = beliefs[code]
        if kind == "mean":
            out[name] = C.centered_linear_gram(b.mean)
        elif kind == "sd":
            out[name] = C.centered_linear_gram(b.sd)
        elif kind == "quant":
            out[name] = C.centered_linear_gram(b.quantiles)
        else:
            if b.cov is None:
                raise KeyError(f"target {name!r}: belief {code!r} has no posterior covariance.")
            out[name] = np.asarray(b.cov, dtype=np.float64)
    return out


def target_confounds(name: str, geometry: Sequence[np.ndarray], layout: np.ndarray | None,
                     layout_code: str) -> list[np.ndarray]:
    """Confound set of one target (D3d): geometry always; the layout covariance for covariance targets.

    A covariance target of the layout engine itself (``cov_fixed`` when GP-fixed is the layout confound) is
    deconfounded on geometry only: removing it from itself would leave nothing by construction.

    Args:
        name: Target name.
        geometry: Geometry confound Grams (:func:`analysis.cka.geometry_confounds`).
        layout: GP-fixed's latent posterior covariance on the same context, or ``None``.
        layout_code: Belief code of the layout engine.

    Returns:
        The confound Grams.
    """
    kind, code = parse_target(name)
    if kind == "cov" and layout is not None and code != layout_code:
        return [*geometry, layout]
    return list(geometry)


@dataclass
class GramStats:
    """A Gram with the quantities every statistic of a cell reuses (computed once).

    Attributes:
        gram: The Gram, [M, M].
        factor: Factor of its double-centred PSD part (NBS), [M, r].
        self_hsic: Unbiased self-HSIC (> 0 for a usable side).
        eff_rank: Participation-ratio effective rank.
    """

    gram: np.ndarray
    factor: np.ndarray
    self_hsic: float
    eff_rank: float

    @classmethod
    def of(cls, K: np.ndarray) -> "GramStats":
        """Precompute the statistics of one Gram."""
        K = np.asarray(K, dtype=np.float64)
        return cls(gram=K, factor=P.gram_factor(K, center=True), self_hsic=C.hsic_unbiased(K, K),
                   eff_rank=C.effective_rank(K))

    @property
    def degenerate(self) -> bool:
        """No usable site structure: a non-positive self-HSIC or an all-zero centred part."""
        return self.self_hsic <= 0 or self.factor.shape[1] == 0


def _nbs(a: GramStats, b: GramStats) -> float | None:
    """NBS of two precomputed sides (``None`` when either is degenerate)."""
    if a.degenerate or b.degenerate:
        return None
    return P.nbs_factors(a.factor, b.factor)


def _pair_stats(a: GramStats, b: GramStats, dec: C.Deconfounder, min_residual_frac: float) -> dict[str, Any]:
    """CKA, deconfounded CKA, NBS and shape distance of one (representation, target) pair."""
    raw = None if (a.degenerate or b.degenerate) else C.cka_or_none(a.gram, b.gram)
    dc = (C.cka_deconfounded(a.gram, b.gram, dec, min_residual_frac=min_residual_frac)
          if not (a.degenerate or b.degenerate) else {"value": None, "degenerate": True,
                                                       "residual_frac_rep": None, "residual_frac_target": None})
    v = _nbs(a, b)
    return {"cka": raw, "cka_deconf": dc["value"], "deconf_degenerate": bool(dc["degenerate"]),
            "residual_frac_rep": dc["residual_frac_rep"], "residual_frac_target": dc["residual_frac_target"],
            "nbs": v, "shape_distance": None if v is None else float(np.arccos(np.clip(v, 0.0, 1.0)))}


def alignment_rows(
    reps: Mapping[tuple[str, int], np.ndarray],
    targets: Mapping[str, np.ndarray],
    deconfounders: Mapping[str, C.Deconfounder],
    *,
    n_perm: int,
    rng: np.random.Generator,
    min_residual_frac: float,
) -> list[dict[str, Any]]:
    """β2: every (representation, target) pair of one cell and site set.

    Args:
        reps: ``{(readout, block): Gram [M, M]}``.
        targets: ``{target: Gram [M, M]}``.
        deconfounders: Confound set per target (:func:`target_confounds`), applied to both sides identically.
        n_perm: Site permutations of the descriptive null of the raw CKA (0 = none).
        rng: Generator of the permutations (one set per target, shared by every representation).
        min_residual_frac: See :func:`analysis.cka.cka_deconfounded`.

    Returns:
        One row per (readout, layer, target): ``cka``, ``cka_deconf``, ``deconf_degenerate``, the residual fractions,
        ``nbs``, ``shape_distance``, ``perm_null_mean``, ``rep_eff_rank``, ``target_eff_rank``, ``rep_degenerate``.
    """
    rep_stats = {k: GramStats.of(G) for k, G in reps.items()}
    usable = [k for k, s in rep_stats.items() if not s.degenerate]
    rows: list[dict[str, Any]] = []
    for name, L in targets.items():
        ts = GramStats.of(L)
        null_mean: dict[tuple[str, int], float] = {}
        if int(n_perm) > 0 and usable and not ts.degenerate:
            _, null = C.cka_null_draws([reps[k] for k in usable], L, int(n_perm), rng)   # [k, n_perm]
            null_mean = {k: float(null[i].mean()) for i, k in enumerate(usable)}
        for (readout, layer), rs in rep_stats.items():
            rows.append({"readout": readout, "layer": int(layer), "target": name,
                         **_pair_stats(rs, ts, deconfounders[name], min_residual_frac),
                         "perm_null_mean": null_mean.get((readout, layer)),
                         "rep_eff_rank": rs.eff_rank, "target_eff_rank": ts.eff_rank,
                         "rep_degenerate": rs.degenerate})
    return rows


def surrogate_rows(
    surrogates: Mapping[float, Sequence[np.ndarray]],
    targets: Mapping[str, np.ndarray],
    deconfounders: Mapping[str, C.Deconfounder],
    *,
    quantile: float,
    min_residual_frac: float,
) -> list[dict[str, Any]]:
    """The smooth-field chance level of every statistic, per target and surrogate lengthscale (D3d).

    Args:
        surrogates: ``{lengthscale: [surrogate Gram [M, M], ...]}`` (:func:`analysis.cka.smooth_field_surrogate`).
        targets: ``{target: Gram}``.
        deconfounders: Confound set per target.
        quantile: Upper quantile a claim must exceed (``representation.surrogate.quantile``; 0.95).
        min_residual_frac: See :func:`analysis.cka.cka_deconfounded`.

    Returns:
        One row per (target, lengthscale, statistic): ``median``, ``q`` (the upper quantile), ``n``.
    """
    sur_stats = {ell: [GramStats.of(G) for G in grams] for ell, grams in surrogates.items()}
    rows: list[dict[str, Any]] = []
    for name, L in targets.items():
        ts = GramStats.of(L)
        for ell, stats in sur_stats.items():
            vals: dict[str, list[float]] = {s: [] for s in _SURROGATE_STATS}
            for ss in stats:
                pair = _pair_stats(ss, ts, deconfounders[name], min_residual_frac)
                for s in _SURROGATE_STATS:
                    if pair[s] is not None:
                        vals[s].append(float(pair[s]))
            for s, v in vals.items():
                rows.append({"target": name, "lengthscale": float(ell), "statistic": s, "n": len(v),
                             "median": float(np.median(v)) if v else None,
                             "q": float(np.quantile(v, float(quantile))) if v else None})
    return rows


def surrogate_floors(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Per target, the most demanding surrogate level over lengthscales (max of the medians and of the quantiles).

    Args:
        rows: Output of :func:`surrogate_rows` for one cell and site set.

    Returns:
        ``{target: {'surrogate_median_<stat>', 'surrogate_q_<stat>'}}``.
    """
    out: dict[str, dict[str, float | None]] = {}
    for r in rows:
        d = out.setdefault(r["target"], {})
        for col, key in ((f"surrogate_median_{r['statistic']}", "median"), (f"surrogate_q_{r['statistic']}", "q")):
            if r[key] is None:
                d.setdefault(col, None)
                continue
            prev = d.get(col)
            d[col] = float(r[key]) if prev is None else max(prev, float(r[key]))
    return out


def pair_rows(
    pairs: Sequence[tuple[str, str]],
    grams_a: Mapping[str, np.ndarray],
    grams_b: Mapping[str, np.ndarray],
    deconfounders: Mapping[str, C.Deconfounder],
    *,
    min_residual_frac: float,
    label: str,
) -> list[dict[str, Any]]:
    """CKA / deconfounded CKA / NBS between pairs of targets (belief ceilings, D2; the ceiling of §5).

    Args:
        pairs: ``(name_a, name_b)``; ``name_a`` is read from ``grams_a``, ``name_b`` from ``grams_b``.
        grams_a: Gram per target name.
        grams_b: Gram per target name.
        deconfounders: Confound set per target; the pair uses ``name_a``'s.
        min_residual_frac: See :func:`analysis.cka.cka_deconfounded`.
        label: Value of the ``relation`` column (``belief_ceiling``, ``trial_ceiling``).

    Returns:
        One row per pair.
    """
    rows = []
    for a, b in pairs:
        stats = _pair_stats(GramStats.of(grams_a[a]), GramStats.of(grams_b[b]), deconfounders[a], min_residual_frac)
        rows.append({"relation": label, "target_a": a, "target_b": b, **stats})
    return rows


def site_match_rows(
    E_own: np.ndarray,
    beliefs: Mapping[str, Belief],
    references: Sequence[str],
    observed: np.ndarray,
) -> list[dict[str, Any]]:
    """β1a: TabPFN's final predictive against each reference, **site by site** (D3).

    Args:
        E_own: TabPFN's final belief embedding, shape [M, J].
        beliefs: Belief per code (the references among them).
        references: Belief codes to compare against.
        observed: Context-site mask, shape [M].

    Returns:
        One row per (reference, site): ``w2``, ``location``, ``spread``, ``shape`` (squared terms summing to
        ``w2 ** 2``), ``w2_over_sd`` (``w2`` over the reference's stated predictive SD at the site), and both models'
        stated SDs.
    """
    rows = []
    for code in references:
        ref = beliefs[code]
        split = site_w2_split(E_own, ref.quantiles)
        for s in range(E_own.shape[0]):
            rows.append({"reference": BELIEF_CODES[code], "site": int(s), "observed": bool(observed[s]),
                         "w2": float(split["w2"][s]), "location": float(split["location"][s]),
                         "spread": float(split["spread"][s]), "shape": float(split["shape"][s]),
                         "w2_over_sd": float(split["w2"][s] / ref.sd[s]), "sd_ref": float(ref.sd[s]),
                         "sd_own": float(beliefs["own"].sd[s])})
    return rows


def lens_rows(
    lens_quantiles: np.ndarray,
    lens_means: np.ndarray,
    layers: Sequence[int],
    beliefs: Mapping[str, Belief],
    references: Sequence[str],
    observed: np.ndarray,
) -> list[dict[str, Any]]:
    """Lens-decoded predictive of every block against TabPFN's final one and each reference, site-matched (β2, G6).

    Per block, the per-site W2 split is computed first and then its median taken over the observed and the
    unobserved sites apart. Two lens diagnostics are written beside it (their flag rule is OPEN, ``research_design.md``
    §4, 2026-10-19; nothing is flagged here): the site SD of the lens mean map relative to the final one, and the
    Pearson r of the two maps.

    Args:
        lens_quantiles: Lens belief embedding per block, shape [L, M, J].
        lens_means: Lens predictive mean per block, shape [L, M].
        layers: Block index of each lens row.
        beliefs: Belief per code (``own`` = the final predictive).
        references: Belief codes to compare against besides ``own``.
        observed: Context-site mask, shape [M].

    Returns:
        One row per (block, comparison, site pool).
    """
    own = beliefs["own"]
    sd_final = float(np.std(own.mean))
    rows = []
    for i, layer in enumerate(layers):
        m = lens_means[i]
        sd_ratio = None if sd_final == 0 else float(np.std(m) / sd_final)
        r_final = None if np.std(m) == 0 or sd_final == 0 else float(np.corrcoef(m, own.mean)[0, 1])
        for code in ["own", *references]:
            ref = beliefs[code]
            split = site_w2_split(lens_quantiles[i], ref.quantiles)
            ratio = split["w2"] / ref.sd                                              # [M]
            for pool, mask in (("unobserved", ~observed), ("observed", observed)):
                if not mask.any():
                    continue
                rows.append({"layer": int(layer), "comparison": BELIEF_CODES[code], "site_pool": pool,
                             "n_sites": int(mask.sum()),
                             **{k: float(np.median(split[k][mask])) for k in ("w2", "location", "spread", "shape")},
                             "w2_over_sd": float(np.median(ratio[mask])),
                             "lens_site_sd_ratio": sd_ratio, "lens_r_final": r_final})
    return rows


def add_headline(df: pd.DataFrame, statistic: str) -> pd.DataFrame:
    """Add the selected headline columns to β2 rows that carry their surrogate floors (OPEN, 2026-10-19).

    Args:
        df: ``representation.csv`` rows with ``cka``, ``cka_deconf`` and the ``surrogate_*`` columns.
        statistic: One of :data:`HEADLINE_STATISTICS`.

    Returns:
        A copy with ``headline_statistic``, ``headline_value``, ``headline_floor`` and ``headline_above_floor``.

    Raises:
        ValueError: On an unknown statistic.
    """
    out = df.copy()
    if statistic == "deconfounded_vs_surrogate":
        value, floor = out["cka_deconf"], out["surrogate_q_cka_deconf"]
    elif statistic == "floor_subtracted_raw":
        value = out["cka"] - out["surrogate_median_cka"]
        floor = out["surrogate_q_cka"] - out["surrogate_median_cka"]
    else:
        raise ValueError(f"unknown headline statistic {statistic!r}; known {HEADLINE_STATISTICS}.")
    out["headline_statistic"] = statistic
    out["headline_value"] = pd.to_numeric(value, errors="coerce")
    out["headline_floor"] = pd.to_numeric(floor, errors="coerce")
    out["headline_above_floor"] = out["headline_value"] > out["headline_floor"]
    return out


#: Site-pool label of the β1a summary over every site, observed and unobserved together (user, 2026-10-22).
SITE_POOL_ALL: str = "all"


def channel_summary(
    df: pd.DataFrame,
    values: Sequence[str],
    by: Sequence[str],
    *,
    n_boot: int,
    ci: float,
    seed: int,
    draw_col: str = "draw",
    reduce: Callable[[pd.Series], float] = lambda s: float(s.median()),
) -> pd.DataFrame:
    """Median over channels with a within-subject channel bootstrap interval (BETA D4).

    Each channel ``(subject, emg)`` is first reduced over its context draws (median), so the interval is
    between-channel heterogeneity, not draw noise. The bootstrap resamples channels with replacement *within each
    subject* (channels are the independent unit; subjects are not mixed).

    Args:
        df: Rows with ``subject``, ``emg``, ``draw_col``, the grouping columns and the value columns.
        values: Value columns to summarise.
        by: Grouping columns (e.g. level, context_t, layer, readout, target).
        n_boot: Bootstrap replicates (``representation.inference.n_boot``).
        ci: Central interval mass (``representation.inference.ci``).
        seed: Bootstrap seed.
        draw_col: Column of the context draw.
        reduce: Within-channel reduction over draws.

    Returns:
        One row per group and value column: ``value``, ``median``, ``ci_low``, ``ci_high``, ``n_channels``,
        ``n_subjects``.
    """
    rng = np.random.default_rng(int(seed))
    lo_q, hi_q = (1.0 - float(ci)) / 2.0, 1.0 - (1.0 - float(ci)) / 2.0
    out: list[dict[str, Any]] = []
    keys = [*by, "subject", "emg"]
    index_cache: dict[tuple, np.ndarray] = {}
    for value in values:
        sub = df.dropna(subset=[value])
        if sub.empty:
            continue
        per_ch = sub.groupby(keys)[value].apply(reduce).reset_index()
        for gkey, part in per_ch.groupby(list(by)):
            gkey = gkey if isinstance(gkey, tuple) else (gkey,)
            part = part.sort_values(["subject", "emg"])
            vals = part[value].to_numpy(dtype=np.float64)                              # [n_ch]
            subjects = part["subject"].to_numpy()
            ck = tuple(subjects.tolist())
            if ck not in index_cache:
                idx = np.empty((int(n_boot), len(vals)), dtype=int)
                for s in np.unique(subjects):
                    pos = np.flatnonzero(subjects == s)
                    idx[:, pos] = rng.choice(pos, size=(int(n_boot), pos.size), replace=True)
                index_cache[ck] = idx
            boots = np.median(vals[index_cache[ck]], axis=1)                          # [n_boot]
            out.append({**dict(zip(by, gkey)), "value": value, "median": float(np.median(vals)),
                        "ci_low": float(np.quantile(boots, lo_q)), "ci_high": float(np.quantile(boots, hi_q)),
                        "n_channels": int(len(vals)), "n_subjects": int(len(np.unique(subjects)))})
    return pd.DataFrame(out)
