"""Hyp C mechanism analyses: Bayesian update (OMEGA, arm i), placement (MMD/W2), CKA, Procrustes.

One CLI, one config family: the ``analysis`` key picks the runner and a same-named block
holds its parameters (unknown keys raise, as everywhere else)::

    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_bayes_update_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_placement_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_cka_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_cka_nhp.yaml --replot
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_cka_nhp.yaml --only-cached

Outputs go to ``{output_root}/mechanism/{analysis}/{dataset}/{tag}/`` with the resolved
``config.yaml``; every figure is rebuilt from the CSVs alone with ``--replot``.

**Shared context ladder (B3 Step 4).** Every analysis conditions on ``context.sizes``, composed from the
``configs/context/<dataset>.yaml`` group, so the analyses of one dataset describe the same contexts
(:mod:`analysis.context`: sites keyed by (grid, t, draw), trials by (channel, level, t, draw)).

**Cells and resume (B3 Steps 2-3, 2026-10-06).** ``mechanism`` used to be the only runner without a cache:
a 5-12 h single process that lost everything when killed. Each analysis is now a grid of *cells* -- the
smallest unit worth resuming -- persisted the moment it finishes through the same
:class:`~pfns4neurostim.evaluation.cache.CellStore` the BO runners use, under
``{output_root}/cells/{dataset}/mechanism_{analysis}/``. The CSVs are assembled from the cells at the end, so
a re-run serves every finished cell from disk, ``--only-cached`` assembles whatever exists, and ``--no-cache``
recomputes everything. A cell's identity holds every setting that can change its numbers and nothing else
(not the ladder or the draw count, so extending either reuses the finished cells).

Cells by analysis:

====================  ===================================================================================
bayes_update          ``cell`` (channel, level, t, draw: every engine of the cell, metric rows only -- the
                      distributions are never cached); ``gate`` (G1, G2, G4); ``discrepancy`` (channel, t: the
                      frozen-preprocessing probe vs the refit, the former G0); ``exemplar`` (Omega-5, per t)
cka                   ``a`` (channel, level, t, draw, readout); ``b`` (grid, t, draw; feature mean);
                      ``gate`` (controls, per readout)
placement             ``B`` (channel); ``C`` (channel: every level and context size); ``gate`` (M0 ladder)
procrustes            ``trajectory`` (channel, level, draw, readout): the whole context ladder of one draw,
                      every layer, disparity + shuffled-site null + the CKA companion on the same pairs
====================  ===================================================================================

One analysis does **not** follow the ``{analysis}`` output path: ``procrustes`` writes under
``{output_root}/mechanism/procrustes/cka/{dataset}/{tag}/`` (:data:`ANALYSIS_SUBDIR`, user requirement
2026-10-14). It is the CKA analysis' complement -- the same clouds, the same pairs, one statistic that keeps the
anisotropy CKA quotients out -- and the nesting says so on disk.

**Artifacts** (same store, ``kind`` in the identity): ``bank`` / ``noise_bank`` per grid and ``gpfit``
(the ``K_GP`` posterior covariance per context). Embeddings are deliberately *not* cached: at 18 blocks x
96 sites x 192 dims a cell's embeddings are ~1.3 MB, ~4 GB+ for NHP, while recomputing one costs ~0.2 s; the
cells above are what makes a re-run free. Caching the banks also pins them across hosts (G5): a
requeued job on another node serves the same maps instead of re-sampling them with that node's numerics.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import yaml

from ..config import DatasetConfig, apply_overrides, compose, dataset_config_from_raw
from ..analysis.context import context_for, context_sites, grid_id
from ..data.channels import ChannelData, iter_channels
from ..evaluation.cache import CellStore
from ..seeding import rng_for, set_seed

__all__ = [
    "MechanismConfig",
    "RunOptions",
    "load_mechanism_config",
    "run_mechanism",
    "main",
    "ANALYSES",
    "ANALYSIS_SUBDIR",
    "MECHANISM_CACHE_VERSION",
]

#: Output sub-path of an analysis under ``{output_root}/mechanism/``, when it is not the analysis name.
#: ``procrustes`` is nested under ``cka`` because it is that analysis' complement on the same clouds and the
#: same pairs (user requirement, 2026-10-14): a reader who opens the CKA tree finds the Procrustes curve in it.
ANALYSIS_SUBDIR: dict[str, str] = {"procrustes": os.path.join("procrustes", "cka")}

#: Invalidation stamp of every mechanism cell and artifact; bump it to discard them all.
MECHANISM_CACHE_VERSION: int = 1

#: The TabPFN checkpoint every mechanism analysis probes (its registry version string enters the identity).
MECHANISM_MODEL: str = "tabpfn_v2_5"


@dataclass(frozen=True)
class MechanismConfig:
    """Resolved mechanism-analysis configuration.

    Attributes:
        analysis: A key of :data:`ANALYSES`.
        tag: Run tag.
        dataset: Dataset selection.
        device: Torch device for TabPFN.
        seed: Base seed.
        output_root: Output root.
        context_sizes: The shared context ladder (``context.sizes``, B3 Step 4).
        params: The analysis block (validated by the analysis runner).
        source_path: YAML path.
    """

    analysis: str
    tag: str
    dataset: DatasetConfig
    device: str
    seed: int
    output_root: str
    context_sizes: tuple[int, ...]
    params: dict[str, Any]
    source_path: str

    @property
    def run_dir(self) -> str:
        """Output directory of this analysis (see :data:`ANALYSIS_SUBDIR` for the one nested exception)."""
        sub = ANALYSIS_SUBDIR.get(self.analysis, self.analysis)
        return os.path.join(self.output_root, "mechanism", sub, self.dataset.name, self.tag)

    @property
    def cell_root(self) -> str:
        """Root of the cell cache (shared with the BO runners)."""
        return os.path.join(self.output_root, "cells")


@dataclass(frozen=True)
class RunOptions:
    """How a runner treats the cell cache.

    Attributes:
        replot: Rebuild figures from the CSVs only; compute and assemble nothing.
        use_cache: ``False`` (``--no-cache``) neither reads nor writes cells.
        only_cached: ``True`` (``--only-cached``) assembles from cached cells and computes nothing.
    """

    replot: bool = False
    use_cache: bool = True
    only_cached: bool = False


def load_mechanism_config(path: str, overrides: list[str] | None = None) -> MechanismConfig:
    """Compose (``defaults: {dataset: ..., context: ...}``), override and validate a mechanism config.

    Args:
        path: YAML path.
        overrides: Dotted ``key=value`` overrides (never invent keys).

    Returns:
        The resolved config.

    Raises:
        ValueError: On unknown keys, an unknown analysis, or a missing / malformed context ladder.
    """
    merged = compose(path)
    if overrides:
        merged = apply_overrides(merged, list(overrides))
    dataset = dataset_config_from_raw(merged.pop("dataset"))
    analysis = str(merged.pop("analysis"))
    if analysis not in ANALYSES:
        raise ValueError(f"Unknown analysis {analysis!r}; expected one of {sorted(ANALYSES)}.")
    context = dict(merged.pop("context", {}) or {})
    unknown_ctx = set(context) - {"sizes"}
    if unknown_ctx:
        raise ValueError(f"Unknown context key(s) {sorted(unknown_ctx)}; known: ['sizes'].")
    if not context.get("sizes"):
        raise ValueError(
            f"{path}: no context ladder. Compose one with `defaults: {{context: <dataset>}}` "
            "(configs/context/) or set `context: {sizes: [...]}`."
        )
    sizes = tuple(int(t) for t in context["sizes"])
    if any(t < 2 for t in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError(f"context.sizes must be distinct integers >= 2, got {list(sizes)}.")
    params = dict(merged.pop(analysis, {}) or {})
    merged.pop("experiment", None)
    cfg = MechanismConfig(
        analysis=analysis,
        tag=str(merged.pop("tag", analysis)),
        dataset=dataset,
        device=os.path.expandvars(str(merged.pop("device", "cpu"))),
        seed=int(merged.pop("seed", 42)),
        output_root=str(merged.pop("output_root", "output")),
        context_sizes=tuple(sorted(sizes)),
        params=params,
        source_path=os.path.abspath(path),
    )
    if merged:
        raise ValueError(f"Unknown mechanism config key(s): {sorted(merged)}.")
    return cfg


def _check_keys(block: dict[str, Any], defaults: dict[str, Any], name: str) -> dict[str, Any]:
    """Merge an analysis block over its defaults, refusing unknown keys.

    Args:
        block: User block.
        defaults: Known keys with default values.
        name: Analysis name, for the error.

    Returns:
        The merged parameters.
    """
    unknown = set(block) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} key(s) {sorted(unknown)}; known: {sorted(defaults)}.")
    return {**defaults, **block}


def _subset_of_ladder(values: Sequence[int], ladder: Sequence[int], name: str) -> None:
    """Refuse an analysis-specific context list that is not drawn from the shared ladder.

    Args:
        values: The analysis' context sizes.
        ladder: ``context.sizes``.
        name: Config key, for the error.

    Raises:
        ValueError: If a value is not in the ladder.
    """
    outside = sorted(set(int(v) for v in values) - set(int(t) for t in ladder))
    if outside:
        raise ValueError(
            f"{name}={list(values)} lists context sizes {outside} that are not in the shared ladder "
            f"context.sizes={list(ladder)}; a cell outside the ladder would not exist to read from."
        )


def _write_config(cfg: MechanismConfig, params: dict[str, Any], out_dir: str) -> None:
    """Write the resolved config (with every default filled) to ``config.yaml``."""
    os.makedirs(out_dir, exist_ok=True)
    doc = {
        "analysis": cfg.analysis,
        "tag": cfg.tag,
        "dataset": {
            "name": cfg.dataset.name, "subjects": list(cfg.dataset.subjects),
            "emgs": None if cfg.dataset.emgs is None else list(cfg.dataset.emgs),
            "data_root": cfg.dataset.data_root, "normalization": cfg.dataset.normalization,
        },
        "context": {"sizes": list(cfg.context_sizes)},
        "device": cfg.device,
        "seed": cfg.seed,
        "output_root": cfg.output_root,
        "model_version": _model_version(),
        "cache_version": MECHANISM_CACHE_VERSION,
        cfg.analysis: json.loads(json.dumps(params, default=_json_default)),
        "source_path": cfg.source_path,
    }
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)


def _json_default(obj: Any) -> Any:
    """JSON fallback for numpy scalars/arrays."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _jsonable(obj: Any) -> Any:
    """Round-trip through JSON so a cell payload holds plain types only (numpy scalars become floats)."""
    return json.loads(json.dumps(obj, default=_json_default))


def _model_version() -> str:
    """Registry version string of the probed TabPFN checkpoint (P0.1)."""
    from ..models.registry import model_version  # noqa: PLC0415 - avoid importing the registry for --help

    return model_version(MECHANISM_MODEL)


def _channels(cfg: MechanismConfig) -> list[ChannelData]:
    """Load every selected channel (full-mean GT)."""
    return list(iter_channels(
        cfg.dataset.name, cfg.dataset.subjects, cfg.dataset.emgs,
        data_root=cfg.dataset.data_root, normalization=cfg.dataset.normalization,
    ))


class _Cells:
    """The mechanism namespace of the cell store: one place that builds identities and serves cells.

    Args:
        cfg: Resolved config.
        opts: Cache policy.
    """

    def __init__(self, cfg: MechanismConfig, opts: RunOptions) -> None:
        self.cfg = cfg
        self.store = CellStore(cfg.cell_root, enabled=opts.use_cache, only_cached=opts.only_cached)
        self.experiment = f"mechanism_{cfg.analysis}"
        self.base = {
            "cache_version": MECHANISM_CACHE_VERSION,
            "analysis": cfg.analysis,
            "dataset": cfg.dataset.name,
            "normalization": cfg.dataset.normalization,
            "seed": cfg.seed,
            "device": cfg.device,
            "model_version": _model_version(),
        }

    @staticmethod
    def channel(ch: ChannelData) -> dict[str, Any]:
        """Identity fields of a channel (the cohort stamp only where the loader versions its data)."""
        ident: dict[str, Any] = {"subject": int(ch.subject), "emg": int(ch.emg), "label": ch.label}
        cohort = ch.meta.get("data_cohort")
        if cohort is not None:
            ident["data_cohort"] = cohort
        return ident

    def run(
        self,
        kind: str,
        identity: dict[str, Any],
        label: str,
        compute: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Serve one cell from the cache or compute and persist it.

        Args:
            kind: Cell kind (see the module docstring).
            identity: Every setting that can change the cell's numbers.
            label: Log label.
            compute: ``() -> (payload, trajectory)``; the payload is made JSON-safe here.

        Returns:
            ``(payload, trajectory)``, or ``None`` on a failure or an ``--only-cached`` miss.
        """
        def safe() -> tuple[dict[str, Any], dict[str, Any]]:
            payload, trajectory = compute()
            return _jsonable(payload), trajectory

        return self.store.run(self.cfg.dataset.name, self.experiment, {**self.base, "kind": kind, **identity},
                              f"{self.cfg.analysis}:{kind} {label}", safe)

    def report(self) -> None:
        """Print the cache tally and raise if any cell failed (after the outputs are written)."""
        s = self.store
        print(f"[mechanism] cells: {s.computed} computed, {s.hits} cached, {len(s.failures)} failed", flush=True)
        s.raise_if_failed()


def _grid_banks(cells: _Cells, ref: ChannelData, n_prior_total: int, n_noise: int, bank: dict[str, Any],
                ) -> tuple[Any, Any] | None:
    """The prior and noise banks of one grid, as cached artifacts.

    Args:
        cells: Cell namespace.
        ref: Any channel on the grid.
        n_prior_total: Prior maps to attempt (bank + hold-out).
        n_noise: Noise maps.
        bank: ``n_dense``, ``holdout_frac``, ``rmse_threshold``, ``prior_type``.

    Returns:
        ``(prior, noise)`` :class:`~pfns4neurostim.data.references.grid.GridBank` objects, or ``None`` when an
        ``--only-cached`` run lacks them.
    """
    from ..data.references import grid as G  # noqa: PLC0415 - module attribute so tests can inject a sampler

    seed = cells.cfg.seed
    gid = grid_id(ref.X_pool)
    p_ident = {"grid_id": gid, "n_total": int(n_prior_total), "n_dense": int(bank["n_dense"]),
               "holdout_frac": float(bank["holdout_frac"]), "rmse_threshold": float(bank["rmse_threshold"]),
               "prior_type": str(bank["prior_type"]), "bank_seed": seed}

    def prior() -> tuple[dict[str, Any], dict[str, Any]]:
        b = G.prior_grid_bank(ref.X_pool, int(n_prior_total), n_dense=int(bank["n_dense"]),
                              holdout_frac=float(bank["holdout_frac"]),
                              rmse_threshold=float(bank["rmse_threshold"]), prior_type=bank["prior_type"],
                              seed=seed)
        return {"n_kept": len(b.maps), "rejection_rate": b.rejection_rate}, {"bank": b}

    def noise() -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {"bank": G.noise_grid_bank(ref.X_pool, int(n_noise), seed=seed + 1)}

    hp = cells.run("bank", p_ident, f"prior bank {gid}", prior)
    hn = cells.run("noise_bank", {"grid_id": gid, "n_noise": int(n_noise), "bank_seed": seed + 1},
                   f"noise bank {gid}", noise)
    if hp is None or hn is None:
        return None
    return hp[1]["bank"], hn[1]["bank"]


# ---------------------------------------------------------------------------
# bayes_update (OBJECTIVE OMEGA, 2026-10-15): how Bayesian is TabPFN's in-context update?
# ---------------------------------------------------------------------------
BAYES_UPDATE_DEFAULTS: dict[str, Any] = {
    # Engines (OMEGA §3.2). TabPFN's update is the deployed one -- a public-API refit on D + {(x', y')} per probe
    # (user, 2026-10-15); the variants differ only in temperature. The frozen-preprocessing probe is not an engine.
    "engines": ["tabpfn_v2_5", "tabpfn_v2_5_t1", "gp_bayes", "gp_mll_refit", "gp_fixed"],
    # Part B references: every other engine of a cell is contrasted with each of them, site-matched.
    "references": ["gp_bayes", "gp_mll_refit", "gp_fixed"],
    # Which cells run which part, per engine; an engine or part not listed runs everywhere. Filter keys:
    # levels, context_ts, draws, channels_per_subject (the first n channels of each subject, in load order).
    "coverage": {},
    "stress": {"knob": "k2_channel", "levels": [1.0, 2.0, 4.0]},
    "n_context_draws": 2,
    "anchors": {"n_unobserved": 8, "n_observed": 4},
    # method: gauss_hermite | truncated | rqmc (analysis/bayes_update.QUADRATURE_METHODS); central_mass for truncated;
    # n_scrambles for rqmc (K is then the points per scramble).
    "quadrature": {"method": "gauss_hermite", "K": 20, "u_min": 1e-12, "central_mass": 1.0, "n_scrambles": 1},
    "eps": 0.01,
    "y_grid": {"points": 512, "u_tail": 1e-4},
    "shared": {"n_trials": 8, "ladder": [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0], "saturation_c": [1.0, 3.0]},
    "far_pitch": 4.0,
    # precision / max_batch_tokens apply to the frozen probe of ``discrepancy`` only (the refit uses the public API).
    # The inference seed of a grid cell is inference_seed + its context draw (user, 2026-10-16, replacing gate G5):
    # each draw is then also an independent TabPFN instance, so every between-draw interval carries seed variance.
    "tabpfn": {"inference_seed": 0, "precision": "fp32", "max_batch_tokens": 400000, "softmax_temperature": 0.9},
    "gp_bayes": {"lengthscale": [0.05, 2.0, 12], "outputscale": [0.05, 5.0, 8], "noise": [1.0e-3, 1.0, 10],
                 "b2": 1.0, "prune_mass": 1.0e-10, "chunk": 2048, "exact_max_components": 64, "sd_bins": 32},
    "gp_params": {"n_opt_steps": 50, "lr": 1.0, "optimizer": "lbfgs", "dtype": "float64", "noise_floor": 1.0e-3,
                  "n_restarts": 5, "lengthscale_floor": 0.05},
    "fixed_gp_params": {"lengthscale": 0.6931471805599453, "outputscale": 1.0, "noise": 0.05},
    # Omega-5: C(x', .) maps at ONE common anchor (the most central site unobserved in every exemplar context).
    "exemplar": {"enabled": True, "subject": None, "emg": None, "level": 1.0, "draw": 0,
                 "engines": ["tabpfn_v2_5", "gp_bayes", "gp_mll_refit"]},
    "panel_ts": None,
    # Hypothesis C1, "is the update GP-like?" (user, 2026-10-19): descriptive readouts against GP-MLL frozen, inside the
    # same cells (a Part B reference only). distance_bins: edges (pitches) of the transfer-vs-distance profile;
    # exemplar_c: the shared ladder multiplier of the change maps at the common anchor (y' = mu_B + c s_B).
    "c1": {"reference": "gp_mll_frozen", "engines": ["tabpfn_v2_5", "gp_mll_frozen", "gp_mll_refit", "gp_fixed"],
           "distance_bins": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 14.0], "exemplar_c": 1.0},
    "run": {"gates": True, "grid": True, "discrepancy": True},
    # The former gate G0, archived as a reported discrepancy (user, 2026-10-15): the frozen-preprocessing probe vs
    # the refit, per site and probe, by t -- the input of the analysis of why the two differ. Never a gate.
    # context_ts null = the whole ladder; channels: the first channels_per_subject of every subject.
    "discrepancy": {"channels_per_subject": 2, "context_ts": None, "n_trials": 3, "ladder": [-3.0, 3.0]},
    "gates": {
        # Run a subset (e.g. [G1, G2] for a quadrature shadow); null = all of G1, G2, G4. G3 (quadrature K) and G5
        # (seed floor) were dropped on 2026-10-16 (user): RQMC gives every anchor its own SE, and the seed is varied
        # with the context draw inside the grid.
        "only": None,
        "n_channels": 2,
        # bayes_alpha: GP-Bayes under rqmc (user, 2026-10-15, option 2 -- the residual must be consistent with zero
        # in SE units): every anchor's drift / total-variance z and every pool's asymmetry z <= the family-wise
        # critical |t| (analysis/bayes_update.rqmc_floor, rqmc_critical_z). The *_tol rules apply to the
        # deterministic schemes, whose residual cannot be told from a violation.
        "g1": {"fixed_tol": 1.0e-8, "bayes_drift_tol": 1.0e-3, "bayes_tol": 1.0e-2, "bayes_alpha": 0.05},
        "g2": {"gain_mult": 1.5, "bias_sd": 0.25, "max_shrink_move": 0.5, "min_drift": 0.05},
        # dense: grid sizes to compare against, one refinement per entry (G4 (b), user 2026-10-15: refine the axis
        # that matters -- the lengthscales -- and check the other axes separately, rather than every axis at once).
        "g4": {"min_ess": 10.0, "context_ts": [10, 50], "rel_tol": 0.10, "abs_tol": 0.01,
               "dense": [{"lengthscale": 45}, {"outputscale": 15, "noise": 19}]},
    },
    # M8 (exploratory): per-channel readouts of ``model`` (anchors of ``pool`` -> cell median -> channel median at
    # ``level``, over ``context_ts``; null = the whole ladder) joined with per-channel outcomes; skipped while both
    # tidy paths are null. predictors / references: a list or "all" (every readout of analysis.predictive_link.READOUTS
    # / every reference the run contrasted ``model`` with). engine_contrasts: engine readouts (Part A, saturation,
    # gain, far_field_share) also as the matched difference engine - reference. outcomes: tidy.csv metric columns
    # (``r2`` included) and ``breakdown`` (per-channel S9 breakdown from ``stress_tidy_csv``; ``breakdown.margin``
    # null = the stress run's own equivalence_margin). acq_label: the acquisition kept in both tidies (null = the
    # only one present).
    "link": {"tidy_csv": None, "model": "tabpfn_v2_5", "level": 1.0,
             "outcomes": ["recommended_regret", "cumulative_regret"],
             "predictors": ["drift_abs", "total_var_ratio", "transfer_slope", "delta_crps"], "n_boot": 2000,
             "references": ["gp_bayes"], "engine_contrasts": False, "pool": "unobserved", "context_ts": None,
             "acq_label": None, "stress_tidy_csv": None,
             "breakdown": {"reference": "gp_mll", "metric": "recommended_regret", "margin": None}},
}

#: TabPFN engines, all the deployed update -- a public-API refit on D + {(x', y')} per probe (user, 2026-10-15):
#: the deployed temperature, and the network's raw PPD (temperature 1 by definition).
TABPFN_ENGINES: dict[str, dict[str, Any]] = {
    "tabpfn_v2_5": {},
    "tabpfn_v2_5_t1": {"softmax_temperature": 1.0},
}

#: Engines that are Part B references only (never a coherence engine): GP-MLL frozen exists for Hypothesis C1 alone.
B_ONLY_ENGINES: tuple[str, ...] = ("gp_mll_frozen",)

#: Precision of TabPFN's quantile function (``FrozenTabPFN.ppf``), recorded in every TabPFN cell identity: float32
#: quantiles broke at the clipped extreme quadrature levels on 2026-10-15, so cells computed before the fix never
#: serve a run after it.
TABPFN_QUANTILE_DTYPE: str = "float64"

#: Quadrature schemes (mirrors ``analysis.bayes_update.QUADRATURE_METHODS``; kept here so validating a config does
#: not import the analysis module).
B_QUADRATURE_METHODS: tuple[str, ...] = ("gauss_hermite", "truncated", "rqmc")

#: Pre-declared headline metrics (OMEGA §3.6): Part A per-anchor, Part A per-cell, Part B (TabPFN vs GP-Bayes).
#: Part A's law-of-total-variance headline is total_var_ratio (user, 2026-10-16): the ratio of sums SumU / SumM is
#: ill-conditioned where M is a few % of s^2 (G3 3/8, G5 ICC <= 0.44 on the refit) and is reported as secondary.
HEADLINE_A: tuple[str, ...] = ("drift_abs", "total_var_ratio", "latent_frac")
HEADLINE_A_CELL: tuple[str, ...] = ("cov_asymmetry", "sigma_recovery_rho")
HEADLINE_B: tuple[str, ...] = ("transfer_slope", "footprint_r", "gap_change", "delta_crps")


def _validate_bayes_update(p: dict[str, Any], ladder: Sequence[int]) -> dict[str, Any]:
    """Resolve the ``bayes_update`` block over its defaults, refusing unknown keys at every level.

    Args:
        p: User block.
        ladder: ``context.sizes``.

    Returns:
        The resolved parameters.

    Raises:
        ValueError: On unknown keys, unknown engines, or settings the analysis cannot honour.
    """
    d = BAYES_UPDATE_DEFAULTS
    p = _check_keys(p, d, "bayes_update")
    for sub in ("stress", "anchors", "quadrature", "y_grid", "shared", "tabpfn", "gp_bayes", "fixed_gp_params",
                "exemplar", "c1", "run", "discrepancy", "gates", "link"):
        p[sub] = _check_keys(p[sub], d[sub], f"bayes_update.{sub}")
    for g in d["gates"]:
        if isinstance(d["gates"][g], dict):
            p["gates"][g] = _check_keys(p["gates"][g], d["gates"][g], f"bayes_update.gates.{g}")
    known = set(TABPFN_ENGINES) | {"gp_bayes", "gp_mll_refit", "gp_fixed", *B_ONLY_ENGINES}
    bad = [e for e in p["engines"] if e not in known]
    if bad:
        raise ValueError(f"bayes_update.engines: unknown {bad}; known {sorted(known)}.")
    if "gp_bayes" not in p["engines"]:
        raise ValueError("bayes_update.engines must include gp_bayes: it anchors the shared y' ladder.")
    bad = [r for r in p["references"] if r not in p["engines"]]
    if bad:
        raise ValueError(f"bayes_update.references {bad} are not in bayes_update.engines.")
    for eng, parts in p["coverage"].items():
        if eng not in p["engines"]:
            raise ValueError(f"bayes_update.coverage names {eng!r}, which is not an engine.")
        for part, spec in (parts or {}).items():
            if part not in ("A", "B"):
                raise ValueError(f"bayes_update.coverage.{eng}: parts are 'A' and 'B', got {part!r}.")
            unknown = set(spec or {}) - {"levels", "context_ts", "draws", "channels_per_subject"}
            if unknown:
                raise ValueError(f"bayes_update.coverage.{eng}.{part}: unknown filter key(s) {sorted(unknown)}.")
    lad = [float(c) for c in p["shared"]["ladder"]]
    for c in p["shared"]["saturation_c"]:
        if float(c) not in lad or -float(c) not in lad:
            raise ValueError(f"bayes_update.shared.saturation_c={p['shared']['saturation_c']} needs +-{c} on the ladder.")
    if p["quadrature"]["method"] not in B_QUADRATURE_METHODS:
        raise ValueError(f"bayes_update.quadrature.method must be one of {B_QUADRATURE_METHODS}, "
                         f"got {p['quadrature']['method']!r}.")
    for eng, parts in p["coverage"].items():
        if eng in B_ONLY_ENGINES and (parts or {}).get("A"):
            raise ValueError(f"bayes_update.coverage.{eng}.A: {eng} is a Part B reference only (Hypothesis C1).")
    bad = [e for e in p["exemplar"]["engines"] if e in B_ONLY_ENGINES]
    if bad:
        raise ValueError(f"bayes_update.exemplar.engines: {bad} have no implied covariance (Part B references only).")
    # C1 is active when its reference is an engine; its figures then show the listed engines that are present.
    c1 = p["c1"]
    if c1["reference"] in p["engines"]:
        if c1["reference"] not in p["references"]:
            raise ValueError(f"bayes_update.c1.reference {c1['reference']!r} must be in bayes_update.references.")
        if float(c1["exemplar_c"]) not in [float(c) for c in p["shared"]["ladder"]]:
            raise ValueError(f"bayes_update.c1.exemplar_c={c1['exemplar_c']} is not on the shared ladder.")
    c1["engines"] = [e for e in c1["engines"] if e in p["engines"]]
    if p["panel_ts"] is None:
        p["panel_ts"] = sorted({int(ladder[0]), int(ladder[len(ladder) // 2])})
    _subset_of_ladder(p["panel_ts"], ladder, "bayes_update.panel_ts")
    if p["discrepancy"]["context_ts"] is None:
        p["discrepancy"]["context_ts"] = [int(t) for t in ladder]
    _subset_of_ladder(p["discrepancy"]["context_ts"], ladder, "bayes_update.discrepancy.context_ts")
    if p["run"]["gates"]:
        _subset_of_ladder(p["gates"]["g4"]["context_ts"], ladder, "bayes_update.gates.g4.context_ts")
    p["link"] = _validate_link(p["link"], p, ladder)
    return p


#: Anchor pools a link predictor may be read from (never pooled).
LINK_POOLS: tuple[str, ...] = ("unobserved", "observed")


def _validate_link(link: dict[str, Any], p: dict[str, Any], ladder: Sequence[int]) -> dict[str, Any]:
    """Resolve and check ``bayes_update.link`` (roadmap M8), refusing what the link could not honour.

    Args:
        link: The link block, already merged over its defaults (one level).
        p: The resolved ``bayes_update`` block (references, stress levels).
        ladder: ``context.sizes``.

    Returns:
        The link block with ``breakdown`` merged over its defaults and ``predictors: all`` expanded.

    Raises:
        ValueError: On an unknown predictor, reference, outcome or pool, a context size off the ladder, or an
            outcome whose tidy path is missing while the link is active.
    """
    from ..analysis.predictive_link import BREAKDOWN_OUTCOME, READOUTS  # noqa: PLC0415
    from ..evaluation.results import METRIC_COLUMNS  # noqa: PLC0415

    d = BAYES_UPDATE_DEFAULTS["link"]
    link["breakdown"] = _check_keys(link["breakdown"] or {}, d["breakdown"], "bayes_update.link.breakdown")
    if link["predictors"] == "all":
        link["predictors"] = list(READOUTS)
    bad = [x for x in link["predictors"] if x not in READOUTS]
    if bad:
        raise ValueError(f"bayes_update.link.predictors: unknown {bad}; known {list(READOUTS)} (or 'all').")
    if link["references"] != "all":
        bad = [r for r in link["references"] if r not in p["references"]]
        if bad:
            raise ValueError(f"bayes_update.link.references {bad} are not Part B references of this run "
                             f"(bayes_update.references = {p['references']}); use a subset or 'all'.")
    bad = [o for o in link["outcomes"] if o not in METRIC_COLUMNS and o != BREAKDOWN_OUTCOME]
    if bad:
        raise ValueError(f"bayes_update.link.outcomes: unknown {bad}; a tidy.csv metric column or "
                         f"{BREAKDOWN_OUTCOME!r}.")
    if link["pool"] not in LINK_POOLS:
        raise ValueError(f"bayes_update.link.pool must be one of {LINK_POOLS}, got {link['pool']!r}.")
    if link["context_ts"] is not None:
        _subset_of_ladder(link["context_ts"], ladder, "bayes_update.link.context_ts")
    if int(link["n_boot"]) < 1:
        raise ValueError(f"bayes_update.link.n_boot must be >= 1, got {link['n_boot']}.")
    if link["tidy_csv"] or link["stress_tidy_csv"]:
        if any(o != BREAKDOWN_OUTCOME for o in link["outcomes"]) and not link["tidy_csv"]:
            raise ValueError("bayes_update.link: BO outcomes need link.tidy_csv.")
        if BREAKDOWN_OUTCOME in link["outcomes"] and not link["stress_tidy_csv"]:
            raise ValueError(f"bayes_update.link: outcome {BREAKDOWN_OUTCOME!r} needs link.stress_tidy_csv.")
        if not any(np.isclose(float(link["level"]), float(lv)) for lv in p["stress"]["levels"]):
            raise ValueError(f"bayes_update.link.level={link['level']} is not a stress level of this run "
                             f"({p['stress']['levels']}).")
    return link


def _make_engine(name: str, p: dict[str, Any], device: str, draw: int = 0) -> Any:
    """Build one update engine by name (OMEGA §3.2).

    Args:
        name: A key of :data:`TABPFN_ENGINES`, ``'gp_bayes'``, ``'gp_mll_refit'`` or ``'gp_fixed'``.
        p: Resolved ``bayes_update`` block.
        device: Torch device for TabPFN.
        draw: Context draw; TabPFN's inference seed is ``tabpfn.inference_seed + draw``.

    Returns:
        The engine.
    """
    from ..analysis import bayes_update as B  # noqa: PLC0415

    if name in TABPFN_ENGINES:
        tp = {**p["tabpfn"], **TABPFN_ENGINES[name]}
        return B.PFNRefitEngine(device=device, random_state=int(tp["inference_seed"]) + int(draw),
                                softmax_temperature=float(tp["softmax_temperature"]), name=name)
    if name == "gp_bayes":
        g = p["gp_bayes"]
        return B.GPBayesEngine(g["lengthscale"], g["outputscale"], g["noise"], b2=float(g["b2"]),
                               prune_mass=float(g["prune_mass"]), chunk=int(g["chunk"]), name=name,
                               exact_max_components=int(g["exact_max_components"]), sd_bins=int(g["sd_bins"]))
    if name == "gp_mll_refit":
        return B.GPRefitEngine(p["gp_params"], name=name)
    if name == "gp_mll_frozen":
        return B.GPFrozenEngine(p["gp_params"], name=name)
    if name == "gp_fixed":
        return B.GPFixedEngine(**p["fixed_gp_params"], name=name)
    raise ValueError(f"Unknown bayes_update engine {name!r}.")


def _engine_params(name: str, p: dict[str, Any], draw: int = 0) -> dict[str, Any]:
    """The settings of one engine that enter a cell identity (seed, temperature and quantile precision included)."""
    if name in TABPFN_ENGINES:
        tp = {**p["tabpfn"], **TABPFN_ENGINES[name]}
        return {"inference_seed": int(tp["inference_seed"]) + int(draw), "softmax_temperature": float(tp["softmax_temperature"]),
                "refit": True, "quantile_dtype": TABPFN_QUANTILE_DTYPE}
    if name == "gp_bayes":
        return {k: v for k, v in p["gp_bayes"].items() if k != "chunk"}
    if name == "gp_mll_refit":
        return {"gp_params": p["gp_params"]}
    if name == "gp_mll_frozen":
        return {"gp_params": p["gp_params"], "refit": False}
    return {"fixed_gp_params": p["fixed_gp_params"]}


def _covered(spec: dict[str, Any] | None, label: str, allowed: dict[int, set[str]], level: float, t: int,
             draw: int) -> bool:
    """Whether a cell passes one coverage filter (``None`` = every cell)."""
    if not spec:
        return True
    if "levels" in spec and not any(np.isclose(level, lv) for lv in spec["levels"]):
        return False
    if "context_ts" in spec and int(t) not in {int(v) for v in spec["context_ts"]}:
        return False
    if "draws" in spec and int(draw) not in {int(v) for v in spec["draws"]}:
        return False
    if "channels_per_subject" in spec and label not in allowed.get(int(spec["channels_per_subject"]), set()):
        return False
    return True


def _first_per_subject(channels: Sequence[ChannelData], n: int) -> set[str]:
    """Labels of the first ``n`` channels of every subject, in load order."""
    seen: dict[int, int] = {}
    keep = set()
    for ch in channels:
        if seen.get(ch.subject, 0) < n:
            keep.add(ch.label)
        seen[ch.subject] = seen.get(ch.subject, 0) + 1
    return keep


def _bayes_cell(
    stressed: ChannelData,
    label: str,
    level: float,
    t: int,
    draw: int,
    p: dict[str, Any],
    cfg: MechanismConfig,
    engines_a: Sequence[str],
    engines_b: Sequence[str],
    make: Callable[[str], Any],
) -> dict[str, Any]:
    """One (channel, level, t, draw) cell: Part A for ``engines_a``, Part B for ``engines_b``, metric rows only.

    Args:
        stressed: The channel at this stress level.
        label: Channel label (seed streams).
        level: Stress level.
        t: Context size.
        draw: Context draw.
        p: Resolved ``bayes_update`` block.
        cfg: Resolved config.
        engines_a: Engines that run Part A here.
        engines_b: Engines that run Part B here (``gp_bayes`` included whenever this is non-empty).
        make: ``name -> unfitted engine``.

    Returns:
        ``coherence`` (per anchor), ``coherence_cell`` (per pool), ``headtohead``, ``surprise`` and ``profile`` rows
        (the transfer-vs-distance profile of Hypothesis C1), the
        ``gp_bayes`` diagnostics, the host and the wall time (guardrail G5).
    """
    import socket  # noqa: PLC0415

    from ..analysis import bayes_update as B  # noqa: PLC0415

    t0 = time.time()
    ctx = context_for(stressed, int(t), int(draw), base_seed=cfg.seed, level=float(level))
    rng = rng_for(label, "bayes_update", level, t, draw, base_seed=cfg.seed)
    anchors, pools, strata = B.select_anchors_uniform(stressed, int(p["anchors"]["n_unobserved"]),
                                                      int(p["anchors"]["n_observed"]), rng, ctx.sites)
    Q = stressed.X_pool
    held = B.heldout_trials(stressed, ctx.sites, ctx.trial_idx)                   # [M, R]
    observed = np.isin(np.arange(stressed.n_sites), ctx.sites)                     # [M]
    dist = B.site_distances(stressed.ch2xy, anchors)                               # [A, M]
    trial_var = np.nanvar(stressed.Y_trials[anchors], axis=1, ddof=1)             # [A]
    names = list(dict.fromkeys([*engines_a, *engines_b]))                          # every engine requested, once
    engines = {n: make(n) for n in names}
    bases = {}
    for n, eng in engines.items():
        eng.fit(ctx.X, ctx.y)
        bases[n] = eng.base(Q)
    sh = p["shared"]
    shared = None
    if engines_b:
        shared = B.shared_values(held, anchors, bases["gp_bayes"], int(sh["n_trials"]), sh["ladder"],
                                 rng_for(label, "bayes_update_shared", level, t, draw, base_seed=cfg.seed))
    qd = p["quadrature"]
    yg = p["y_grid"]
    coh: dict[str, list[dict[str, Any]]] = {n: [] for n in engines_a}
    Cs: dict[str, list[np.ndarray]] = {n: [] for n in engines_a}
    Ccs: dict[str, list[np.ndarray]] = {n: [] for n in engines_a}
    h2h: list[dict[str, Any]] = []
    surprise: list[dict[str, Any]] = []
    profile: list[dict[str, Any]] = []
    refs = [r for r in p["references"] if r in engines_b]
    for i, a in enumerate(anchors):
        a = int(a)
        preds_b: dict[str, dict[str, Any]] = {}
        # One quadrature draw per anchor, shared by every engine (identical levels u; each maps them through its
        # own quantile function).
        u, w = B.quadrature_levels(str(qd["method"]), int(qd["K"]), u_min=float(qd["u_min"]),
                                   central_mass=float(qd["central_mass"]), n_scrambles=int(qd["n_scrambles"]),
                                   rng=rng_for(label, "bayes_update_quadrature", level, t, draw, a,
                                               base_seed=cfg.seed))
        for n in names:
            vals, parts = [], {}
            yqs = []
            if n in engines_a:
                for sidx in range(len(u)):
                    yq = B.quadrature_values(bases[n], a, u[sidx])
                    parts[f"quad{sidx}"] = np.arange(len(vals), len(vals) + len(yq))
                    vals += list(yq)
                    yqs.append(yq)
            if n in engines_b:
                for key in ("trial", "ladder"):
                    parts[key] = np.arange(len(vals), len(vals) + len(shared[key][i]))
                    vals += list(shared[key][i])
            pr = engines[n].probe(np.repeat(Q[a][None], len(vals), axis=0), np.asarray(vals), Q)
            if n in engines_a:
                extra = [(yqs[sidx], pr.take(parts[f"quad{sidx}"])) for sidx in range(len(u))]
                row, C, Cc = B.coherence_scheme(engines[n], bases[n], Q, a, u, w, float(p["eps"]),
                                                u_tail=float(yg["u_tail"]), grid_points=int(yg["points"]), extra=extra)
                coh[n].append({**row, "pool": pools[i], "strata": strata[i], "trial_var": float(trial_var[i])})
                Cs[n].append(C)
                Ccs[n].append(Cc)
            if n in engines_b:
                preds_b[n] = {"base": bases[n], "trial": pr.take(parts["trial"]), "ladder": pr.take(parts["ladder"])}
        if preds_b:
            grid = B.site_grid([q for v in preds_b.values() for q in v.values()], float(yg["u_tail"]),
                               int(yg["points"]), extra=held)
            rows, sur, prof = B.headtohead_anchor(
                preds_b, refs, anchor=a, trial_y=shared["trial"][i], trial_idx=shared["trial_idx"][i],
                ladder_c=shared["c"], s_bayes_anchor=float(bases["gp_bayes"].sd[0, a]), heldout=held,
                observed=observed, dist=dist[i], y_grid=grid, far_pitch=float(p["far_pitch"]),
                saturation_c=sh["saturation_c"], distance_bins=p["c1"]["distance_bins"])
            h2h += [{**r, "pool": pools[i], "strata": strata[i]} for r in rows]
            surprise += [{**r, "pool": pools[i]} for r in sur]
            profile += [{**r, "pool": pools[i]} for r in prof]
    coh_cell = []
    for n in engines_a:
        extra = engines[n].diagnostics() if n == "gp_bayes" else {}
        coh_cell += [{"engine": n, **c, **extra}
                     for c in B.coherence_cell(coh[n], np.asarray(Cs[n]), anchors, pools, trial_var,
                                               C_c=np.asarray(Ccs[n]))]
    return {
        "coherence": [{"engine": n, **r} for n in engines_a for r in coh[n]],
        "coherence_cell": coh_cell,
        "headtohead": h2h,
        "surprise": surprise,
        "profile": profile,
        "anchors": anchors.tolist(),
        "C": {n: np.asarray(Cs[n]).tolist() for n in engines_a},
        "host": socket.gethostname(),
        "device": cfg.device,
        "seconds": time.time() - t0,
    }


def _rel_ok(a: float | None, b: float | None, rel: float, abs_tol: float) -> bool:
    """``|a - b| <= rel |a|`` or ``<= abs_tol`` (gate G4); two undefined values agree."""
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= max(rel * abs(a), abs_tol)


def _bayes_gates(
    cfg: MechanismConfig,
    p: dict[str, Any],
    cells: _Cells,
    channels: Sequence[ChannelData],
    knob: Any,
) -> list[dict[str, Any]]:
    """Gates G1, G2 and G4 (OMEGA §3.6) on the first ``gates.n_channels`` channels at the nominal level, draw 0.

    G3 and G5 were dropped on 2026-10-16 (user): under RQMC every anchor carries its own standard error (with the
    largest quadrature ``|y'|`` as a tail flag), which measures what G3's K-doubling estimated; and the inference seed
    varies with the context draw inside the grid, so seed variance enters every interval instead of a separate ICC.

    G0 (frozen probe vs refit) is no longer a gate: TabPFN's update is the refit itself, and the frozen probe's
    distance from it is reported by :func:`_frozen_refit_discrepancy` (user, 2026-10-15).

    Every gate is a cached cell whose identity holds its thresholds, so re-running reassembles ``gates.json``
    without recomputing.

    Returns:
        One record per gate with ``passed`` and the measured quantities.
    """
    from ..analysis import bayes_update as B  # noqa: PLC0415

    g = p["gates"]
    only = set(g["only"] or ("G1", "G2", "G4"))
    gate_channels = list(channels[: int(g["n_channels"])])
    level, draw = float(knob.levels[0]), 0
    common = {"knob": knob.name, "level": level, "draw": draw, "anchors": p["anchors"], "eps": p["eps"],
              "quadrature": p["quadrature"], "y_grid": p["y_grid"], "shared": p["shared"]}
    out: list[dict[str, Any]] = []

    def stressed_of(ch: ChannelData) -> ChannelData:
        return knob.apply(ch, level, rng_for(ch.label, "bayes_update", level, base_seed=cfg.seed))

    def run(ident: dict[str, Any], label: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
        hit = cells.run("gate", {**common, **ident}, label, lambda: (fn(), {}))
        return None if hit is None else hit[0]

    # ---- G1 (exact floor), G2 (power), G4 ESS: GP engines at every t ----
    g1, g2 = g["g1"], g["g2"]
    fixed_rows, bayes_rows, gain_rows, bias_rows, ess = [], [], [], [], {}
    for ch in gate_channels if only & {"G1", "G2", "G4"} else []:
        for t in cfg.context_sizes:
            def g12_cell(ch: ChannelData = ch, t: int = int(t)) -> dict[str, Any]:
                st = stressed_of(ch)
                ctx = context_for(st, t, draw, base_seed=cfg.seed, level=level)
                rng = rng_for(ch.label, "bayes_update", level, t, draw, base_seed=cfg.seed)
                anchors, pools, _ = B.select_anchors_uniform(st, int(p["anchors"]["n_unobserved"]),
                                                             int(p["anchors"]["n_observed"]), rng, ctx.sites)
                tv = np.nanvar(st.Y_trials[anchors], axis=1, ddof=1)
                qd = p["quadrature"]
                kw = {"method": str(qd["method"]), "u_min": float(qd["u_min"]),
                      "central_mass": float(qd["central_mass"]), "n_scrambles": int(qd["n_scrambles"]),
                      "rng": rng_for(ch.label, "bayes_update_g1_quadrature", level, t, draw, base_seed=cfg.seed),
                      "u_tail": float(p["y_grid"]["u_tail"]), "grid_points": int(p["y_grid"]["points"])}
                res: dict[str, Any] = {}
                for name, eng in (
                        ("gp_fixed", _make_engine("gp_fixed", p, cfg.device)),
                        ("gp_bayes", _make_engine("gp_bayes", p, cfg.device)),
                        ("toy_gain", B.ToyUpdater(_make_engine("gp_fixed", p, cfg.device), gain_mult=g2["gain_mult"])),
                        ("toy_bias", B.ToyUpdater(_make_engine("gp_fixed", p, cfg.device), bias_sd=g2["bias_sd"]))):
                    eng.fit(ctx.X, ctx.y)
                    rows, C, Cc = B.coherence(eng, st.X_pool, anchors, int(qd["K"]), float(p["eps"]), **kw)
                    cell = B.coherence_cell(rows, C, anchors, pools, tv, C_c=Cc)
                    res[name] = {"rows": rows, "cell": cell}
                    if name == "gp_fixed":
                        kC = eng.posterior_covariance(st.X_pool)[anchors]
                        res[name]["max_cov_error"] = float(np.abs(C - kC).max())
                        res[name]["max_cov_error_c"] = float(np.abs(Cc - kC).max())
                    if name == "gp_bayes":
                        res[name]["ess"] = eng.diagnostics()["ess"]
                        if qd["method"] == "rqmc":
                            fr, fc = B.rqmc_floor(
                                eng, st.X_pool, anchors, pools, int(qd["K"]), float(p["eps"]),
                                n_scrambles=int(qd["n_scrambles"]), u_min=float(qd["u_min"]),
                                rng=rng_for(ch.label, "bayes_update_g1_rqmc", level, t, draw, base_seed=cfg.seed))
                            res[name]["floor"] = {"rows": fr, "cell": fc}
                return res

            hit = run({"gate": "G1G2", **cells.channel(ch), "t": int(t), "g1": g1, "g2": g2, "quadrature": p["quadrature"],
                       "fixed": _engine_params("gp_fixed", p), "bayes": _engine_params("gp_bayes", p)},
                      f"G1/G2 {ch.label} t={t}", g12_cell)
            if hit is None:
                continue
            fixed_rows.append(hit["gp_fixed"])
            bayes_rows.append(hit["gp_bayes"])
            gain_rows += hit["toy_gain"]["rows"]
            bias_rows += hit["toy_bias"]["rows"]
            ess.setdefault(int(t), []).append(hit["gp_bayes"]["ess"])

    def worst(blocks: list[dict[str, Any]], key: str, src: str = "rows", fn: Callable[[float], float] = abs) -> float:
        vals = [fn(r[key]) for b in blocks for r in b[src] if r.get(key) is not None]
        return float(max(vals)) if vals else float("nan")

    if fixed_rows:
        # G1 re-stated (user, 2026-10-15): the full Gauss-Hermite measure is checked on the raw readouts; a truncated
        # or randomised measure on the moment-matched ones, which are exact for GP-fixed under any measure.
        sfx = "" if p["quadrature"]["method"] == "gauss_hermite" else "_c"
        f = {"max_abs_drift": worst(fixed_rows, f"max_abs_drift{sfx}"),
             "max_abs_total_var_dev": worst(fixed_rows, f"max_abs_total_var_dev{sfx}"),
             "max_cov_asymmetry": worst(fixed_rows, f"cov_asymmetry{sfx}", "cell"),
             "max_cov_error": float(max(b[f"max_cov_error{sfx}"] for b in fixed_rows))}
        b = {"max_drift_abs": worst(bayes_rows, f"drift_abs{sfx}"),
             "max_total_var_dev": worst(bayes_rows, f"total_var_ratio{sfx}", fn=lambda v: abs(v - 1.0)),
             "max_cov_asymmetry": worst(bayes_rows, f"cov_asymmetry{sfx}", "cell")}
        if p["quadrature"]["method"] == "rqmc":
            # G1 re-stated in SE units (user, 2026-10-15): GP-Bayes's residual must be consistent with zero.
            fl = [blk["floor"] for blk in bayes_rows]
            zs = [r[k] for blk in fl for r in blk["rows"] for k in ("drift_z", "total_var_z") if r[k] is not None]
            za = [c["cov_asymmetry_z"] for blk in fl for c in blk["cell"] if c["cov_asymmetry_z"] is not None]
            crit = B.rqmc_critical_z(int(p["quadrature"]["n_scrambles"]), len(zs) + len(za), float(g1["bayes_alpha"]))
            b.update({"max_drift_z": worst(fl, "drift_z"), "max_total_var_z": worst(fl, "total_var_z"),
                      "max_cov_asymmetry_z": worst(fl, "cov_asymmetry_z", "cell"),
                      "median_drift_z": float(np.median([r["drift_z"] for blk in fl for r in blk["rows"]
                                                         if r["drift_z"] is not None])),
                      "median_total_var_z": float(np.median([r["total_var_z"] for blk in fl for r in blk["rows"]
                                                             if r["total_var_z"] is not None])),
                      "median_drift_se": float(np.median([r["drift_se"] for blk in fl for r in blk["rows"]
                                                          if r["drift_se"] is not None])),
                      "critical_z": crit, "n_tests": len(zs) + len(za)})
            bayes_ok = max(zs + za) <= crit
        else:
            bayes_ok = (b["max_drift_abs"] <= g1["bayes_drift_tol"] and b["max_total_var_dev"] <= g1["bayes_tol"]
                        and b["max_cov_asymmetry"] <= g1["bayes_tol"])
        out.append({"gate": "G1", "quadrature": p["quadrature"], "gp_fixed": f, "gp_bayes": b, "thresholds": g1,
                    "passed": bool(all(v <= g1["fixed_tol"] for v in f.values()) and bayes_ok)})
        sm = float(np.median([r["shrink_move_ratio"] for r in gain_rows if r["shrink_move_ratio"] is not None]))
        dr = float(np.median([r["drift_abs"] for r in bias_rows if r["drift_abs"] is not None]))
        out.append({"gate": "G2", "toy_gain_shrink_move_median": sm, "toy_bias_drift_median": dr, "thresholds": g2,
                    "passed": bool(sm <= g2["max_shrink_move"] and dr >= g2["min_drift"])})

    # ---- G4 (grid density): TabPFN-vs-GP-Bayes cells at two GP-Bayes grids ----
    def paired_cells(gate: str, spec: dict[str, Any], alt: dict[str, Any]) -> list[tuple[dict, dict]]:
        pairs = []
        for ch in gate_channels:
            for t in spec["context_ts"]:
                def compute(ch: ChannelData = ch, t: int = int(t)) -> dict[str, Any]:
                    st = stressed_of(ch)
                    res = {}
                    for key, kw in (("base", {}), ("alt", alt)):
                        pp = json.loads(json.dumps(p))
                        if "gp_bayes" in kw:
                            pp["gp_bayes"].update(kw["gp_bayes"])
                        res[key] = _bayes_cell(st, ch.label, level, t, draw, pp, cfg, [], ["tabpfn_v2_5", "gp_bayes"],
                                               lambda n, pp=pp: _make_engine(n, pp, cfg.device))
                    return _jsonable(res)

                hit = run({"gate": gate, **cells.channel(ch), "t": int(t), "spec": spec, "alt": alt,
                           "engine_params": _engine_params("tabpfn_v2_5", p), "bayes": _engine_params("gp_bayes", p)},
                          f"{gate} {ch.label} t={t}", compute)
                if hit is not None:
                    pairs.append((hit["base"], hit["alt"]))
        return pairs

    g4 = g["g4"]
    gb = p["gp_bayes"]
    moves = []
    for refinement in (g4["dense"] if "G4" in only else []):
        dense = {"gp_bayes": {k: [gb[k][0], gb[k][1], int(refinement.get(k, gb[k][2]))]
                              for k in ("lengthscale", "outputscale", "noise")}}
        for base, alt in paired_cells("G4", g4, dense):
            for metric in HEADLINE_B:
                def med(rows: list[dict[str, Any]], metric: str = metric) -> float | None:
                    vals = [r[metric] for r in rows if r["engine"] == "tabpfn_v2_5" and r["reference"] == "gp_bayes"
                            and r.get(metric) is not None]
                    return float(np.median(vals)) if vals else None

                a, b2 = med(base["headtohead"]), med(alt["headtohead"])
                moves.append({"refinement": refinement, "metric": metric, "grid": a, "grid_dense": b2,
                              "ok": _rel_ok(a, b2, g4["rel_tol"], g4["abs_tol"])})
    ess_med = {t: float(np.median(v)) for t, v in sorted(ess.items())}
    if moves or ess_med:
        out.append({"gate": "G4", "median_ess": ess_med, "density": moves, "thresholds": g4,
                    "passed": bool(all(v >= g4["min_ess"] for v in ess_med.values()) and all(mv["ok"] for mv in moves))})

    return out


def _frozen_refit_discrepancy(
    cfg: MechanismConfig,
    p: dict[str, Any],
    cells: _Cells,
    channels: Sequence[ChannelData],
    knob: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frozen-preprocessing probe vs the deployed refit, per site and probe, by t (the former gate G0).

    Archived as a report on 2026-10-15 (user): TabPFN's update is the refit, so nothing downstream depends on the
    frozen probe; how far it is from the refit -- and why, the follow-up analysis -- is a result in itself. Every
    cell keeps the raw moments so that analysis needs no recompute: per anchor and probe y', both engines' mean and
    SD at every site after the update, and both engines' base (D only) moments, which must agree (same
    preprocessing fitted on D). Cells: the first ``discrepancy.channels_per_subject`` channels of every subject, the
    nominal stress level, draw 0, each t of ``discrepancy.context_ts``; anchors as in the grid cell.

    Args:
        cfg: Resolved config.
        p: Resolved ``bayes_update`` block.
        cells: Cell store.
        channels: Every channel of the run.
        knob: Stress knob (its first level is the nominal one).

    Returns:
        ``(sites, summary)``: one row per (channel, t, anchor, probe, site) with the signed discrepancies
        ``d_mean = (mu_frozen - mu_refit) / s_refit`` and ``d_sd = (s_frozen - s_refit) / s_refit``; and their
        summary by t x probe kind x pool (plus ``all`` rows): n, median and p95 of ``|d_mean|``, median of
        ``|d_sd|``, the signed medians, and the base-moment check.
    """
    from ..analysis import bayes_update as B  # noqa: PLC0415

    dc = p["discrepancy"]
    tp = p["tabpfn"]
    level, draw = float(knob.levels[0]), 0
    keep = _first_per_subject(channels, int(dc["channels_per_subject"]))
    rows: list[dict[str, Any]] = []
    for ch in [c for c in channels if c.label in keep]:
        st = knob.apply(ch, level, rng_for(ch.label, "bayes_update", level, base_seed=cfg.seed))
        for t in dc["context_ts"]:
            if int(t) >= st.n_sites:
                continue

            def cell(st: ChannelData = st, ch: ChannelData = ch, t: int = int(t)) -> dict[str, Any]:
                ctx = context_for(st, t, draw, base_seed=cfg.seed, level=level)
                rng = rng_for(ch.label, "bayes_update", level, t, draw, base_seed=cfg.seed)
                anchors, pools, _ = B.select_anchors_uniform(st, int(p["anchors"]["n_unobserved"]),
                                                             int(p["anchors"]["n_observed"]), rng, ctx.sites)
                Q = st.X_pool
                bayes = _make_engine("gp_bayes", p, cfg.device)
                bayes.fit(ctx.X, ctx.y)
                held = B.heldout_trials(st, ctx.sites, ctx.trial_idx)
                sv = B.shared_values(held, anchors, bayes.base(Q), int(dc["n_trials"]), dc["ladder"],
                                     rng_for(ch.label, "bayes_update_g0", level, t, draw, base_seed=cfg.seed))
                frozen = B.PFNEngine(device=cfg.device, random_state=int(tp["inference_seed"]),
                                     softmax_temperature=float(tp["softmax_temperature"]),
                                     precision=str(tp["precision"]), max_batch_tokens=int(tp["max_batch_tokens"]),
                                     name="tabpfn_v2_5_frozen")
                refit = _make_engine("tabpfn_v2_5", p, cfg.device)
                frozen.fit(ctx.X, ctx.y)
                refit.fit(ctx.X, ctx.y)
                bf, br = frozen.base(Q), refit.base(Q)
                per_anchor = []
                for i, a in enumerate(anchors):
                    ys = np.concatenate([sv["trial"][i], sv["ladder"][i]])                 # [P]
                    xs = np.repeat(Q[[int(a)]], len(ys), axis=0)                         # [P, D]
                    pf, pr = frozen.probe(xs, ys, Q), refit.probe(xs, ys, Q)
                    per_anchor.append({"anchor": int(a), "pool": pools[i], "y": ys.tolist(),
                                       "kind": ["trial"] * len(sv["trial"][i]) + ["ladder"] * len(sv["ladder"][i]),
                                       "c": [None] * len(sv["trial"][i]) + [float(c) for c in sv["c"]],
                                       "mu_frozen": pf.mean.tolist(), "sd_frozen": pf.sd.tolist(),
                                       "mu_refit": pr.mean.tolist(), "sd_refit": pr.sd.tolist()})
                return {"anchors": per_anchor, "observed": np.isin(np.arange(st.n_sites), ctx.sites).tolist(),
                        "dist": B.site_distances(st.ch2xy, anchors).tolist(),
                        "y_mean": float(np.mean(ctx.y)), "y_sd": float(np.std(ctx.y)),
                        "base": {"mu_frozen": bf.mean[0].tolist(), "sd_frozen": bf.sd[0].tolist(),
                                 "mu_refit": br.mean[0].tolist(), "sd_refit": br.sd[0].tolist()}}

            ident = {**cells.channel(ch), "knob": knob.name, "level": level, "draw": draw, "t": int(t),
                     "anchors": p["anchors"], "discrepancy": {k: dc[k] for k in ("n_trials", "ladder")},
                     "frozen": {k: tp[k] for k in ("precision", "max_batch_tokens")},
                     "engine_params": _engine_params("tabpfn_v2_5", p), "bayes": _engine_params("gp_bayes", p)}
            hit = cells.run("discrepancy", ident, f"discrepancy {ch.label} t={t}", lambda cell=cell: (cell(), {}))
            if hit is None:
                continue
            pay = hit[0]
            b = {k: np.asarray(v) for k, v in pay["base"].items()}
            base_mu = np.abs(b["mu_frozen"] - b["mu_refit"]) / b["sd_refit"]                # [M]
            observed = np.asarray(pay["observed"])
            for i, an in enumerate(pay["anchors"]):
                mf, sf = np.asarray(an["mu_frozen"]), np.asarray(an["sd_frozen"])         # [P, M]
                mr, sr = np.asarray(an["mu_refit"]), np.asarray(an["sd_refit"])           # [P, M]
                dist = np.asarray(pay["dist"][i])                                         # [M]
                for k, yv in enumerate(an["y"]):
                    for s in range(mf.shape[1]):
                        rows.append({"label": ch.label, "subject": ch.subject, "emg": ch.emg, "level": level,
                                     "context_t": int(t), "anchor": an["anchor"], "pool": an["pool"],
                                     "kind": an["kind"][k], "c": an["c"][k], "y_prime": float(yv),
                                     "z_prime": (float(yv) - pay["y_mean"]) / pay["y_sd"], "site": s,
                                     "observed": bool(observed[s]), "dist": float(dist[s]),
                                     "mu_frozen": mf[k, s], "sd_frozen": sf[k, s], "mu_refit": mr[k, s],
                                     "sd_refit": sr[k, s], "d_mean": (mf[k, s] - mr[k, s]) / sr[k, s],
                                     "d_sd": (sf[k, s] - sr[k, s]) / sr[k, s], "base_d_mean": base_mu[s]})
    sites = pd.DataFrame(rows)
    if sites.empty:
        return sites, sites
    sites["probe"] = np.where(sites["kind"] == "trial", "trial",
                              np.where(sites["c"].fillna(0) > 0, "ladder+", "ladder-"))

    def summ(g: pd.DataFrame) -> pd.Series:
        a = g["d_mean"].abs()
        return pd.Series({"n": len(g), "median_abs_d_mean": a.median(), "p95_abs_d_mean": a.quantile(0.95),
                          "median_abs_d_sd": g["d_sd"].abs().median(), "median_d_mean": g["d_mean"].median(),
                          "median_d_sd": g["d_sd"].median(), "max_base_d_mean": g["base_d_mean"].max()})

    parts = []
    for by in (["context_t"], ["context_t", "probe"], ["context_t", "pool"], ["context_t", "probe", "pool"]):
        s = sites.groupby(by)[["d_mean", "d_sd", "base_d_mean"]].apply(summ).reset_index()
        for col in ("probe", "pool"):
            if col not in s:
                s[col] = "all"
        parts.append(s)
    summary = pd.concat(parts, ignore_index=True)[
        ["context_t", "probe", "pool", "n", "median_abs_d_mean", "p95_abs_d_mean", "median_abs_d_sd",
         "median_d_mean", "median_d_sd", "max_base_d_mean"]]
    return sites, summary


def _common_anchor(channel: ChannelData, contexts: Sequence[Any]) -> int:
    """The most central site of the grid that is unobserved in every context (Omega-5's common anchor).

    Raises:
        ValueError: If every site is observed in some context.
    """
    free = np.ones(channel.n_sites, dtype=bool)
    for ctx in contexts:
        free[ctx.sites] = False
    if not free.any():
        raise ValueError("no site is unobserved in every exemplar context.")
    X = channel.X_pool
    centre = (X.min(0) + X.max(0)) / 2.0
    cand = np.flatnonzero(free)
    return int(cand[np.argmin(np.linalg.norm(X[cand] - centre, axis=1))])


def run_bayes_update(cfg: MechanismConfig, opts: RunOptions) -> str:
    """OMEGA: Parts A and B on the grid channel x level x t x draw, gates G1, G2, G4, the frozen-vs-refit discrepancy
    report, the exemplar, the figures.

    Args:
        cfg: Resolved config.
        opts: Cache policy.

    Returns:
        The run directory.

    Raises:
        RuntimeError: If gate G1 or G2 fails while the grid is requested (stop and report, OMEGA §3.6).
    """
    from ..analysis import bayes_update as B  # noqa: PLC0415
    from ..data.snr import achieved_snr_db  # noqa: PLC0415
    from ..data.stress import build_knob  # noqa: PLC0415
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    out = cfg.run_dir
    if not opts.replot:
        set_seed(cfg.seed)
        _write_config(cfg, p, out)
        cells = _Cells(cfg, opts)
        knob = build_knob(p["stress"]["knob"], p["stress"]["levels"])
        channels = _channels(cfg)
        if p["run"]["gates"]:
            gates = _bayes_gates(cfg, p, cells, channels, knob)
            with open(os.path.join(out, "gates.json"), "w", encoding="utf-8") as fh:
                json.dump(gates, fh, indent=2, default=_json_default)
            for gt in gates:
                print(f"[mechanism] gate {gt['gate']} {gt.get('channel', '')}: "
                      f"{'PASS' if gt['passed'] else 'FAIL'}", flush=True)
            failed = sorted({gt["gate"] for gt in gates if gt["gate"] in ("G1", "G2") and not gt["passed"]})
            if failed and p["run"]["grid"]:
                cells.report()
                raise RuntimeError(f"bayes_update: gate(s) {failed} failed -- stop and report (OMEGA §3.6); "
                                   f"see {os.path.join(out, 'gates.json')}.")
        if p["run"]["discrepancy"]:
            sites, summary = _frozen_refit_discrepancy(cfg, p, cells, channels, knob)
            if not sites.empty:
                sites.to_csv(os.path.join(out, "frozen_refit_discrepancy_sites.csv"), index=False)
                summary.to_csv(os.path.join(out, "frozen_refit_discrepancy.csv"), index=False)
                print(f"[mechanism] wrote {os.path.join(out, 'frozen_refit_discrepancy.csv')}", flush=True)
        if p["run"]["grid"]:
            allowed = {n: _first_per_subject(channels, n) for n in {
                int(spec["channels_per_subject"]) for parts in p["coverage"].values() for spec in (parts or {}).values()
                if spec and "channels_per_subject" in spec}}
            coh_rows, cell_rows, h2h_rows, sur_rows, prof_rows = [], [], [], [], []
            t0 = time.time()
            for ch in channels:
                for level in knob.levels:
                    st = knob.apply(ch, level, rng_for(ch.label, "bayes_update", level, base_seed=cfg.seed))
                    snr = achieved_snr_db(st)
                    for t in cfg.context_sizes:
                        if t >= st.n_sites:
                            continue
                        for draw in range(int(p["n_context_draws"])):
                            cov = {part: [e for e in p["engines"] if _covered(
                                (p["coverage"].get(e) or {}).get(part), ch.label, allowed, level, t, draw)
                                and not (part == "A" and e in B_ONLY_ENGINES)]
                                for part in ("A", "B")}
                            if cov["B"] and "gp_bayes" not in cov["B"]:
                                cov["B"].append("gp_bayes")
                            keys = {"dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg, "knob": knob.name,
                                    "level": float(level), "achieved_snr_db": snr, "context_t": int(t), "draw": draw}
                            ident = {**cells.channel(ch), "knob": knob.name, "level": float(level), "t": int(t),
                                     "draw": draw, "engines_a": cov["A"], "engines_b": cov["B"],
                                     "engine_params": {e: _engine_params(e, p, draw)
                                                       for e in set(cov["A"]) | set(cov["B"])},
                                     **{k: p[k] for k in ("references", "anchors", "quadrature", "eps", "y_grid",
                                                          "shared", "far_pitch")}}
                            hit = cells.run(
                                "cell", ident, f"{ch.label} a={level:g} t={t} d={draw}",
                                lambda st=st, ch=ch, level=level, t=t, draw=draw, cov=cov: (_bayes_cell(
                                    st, ch.label, float(level), int(t), draw, p, cfg, cov["A"], cov["B"],
                                    lambda n, draw=draw: _make_engine(n, p, cfg.device, draw)), {}))
                            if hit is None:
                                continue
                            pay = hit[0]
                            prov = {"host": pay["host"], "seconds": pay["seconds"]}
                            coh_rows += [{**keys, **r, **prov} for r in pay["coherence"]]
                            cell_rows += [{**keys, **r, **prov} for r in pay["coherence_cell"]]
                            h2h_rows += [{**keys, **r} for r in pay["headtohead"]]
                            sur_rows += [{**keys, **r} for r in pay["surprise"]]
                            prof_rows += [{**keys, **r} for r in pay.get("profile", [])]
                    print(f"[mechanism] {ch.label} level={level} done ({time.time() - t0:.0f}s)", flush=True)
            for rows, name in ((coh_rows, "coherence.csv"), (cell_rows, "coherence_cell.csv"),
                               (h2h_rows, "headtohead.csv"), (sur_rows, "surprise.csv"),
                               (prof_rows, "transfer_profile.csv")):
                _write_csv(rows, out, name)
            ex = p["exemplar"]
            if ex["enabled"] and channels:
                pick = [c for c in channels if (ex["subject"] is None or c.subject == ex["subject"])
                        and (ex["emg"] is None or c.emg == ex["emg"])]
                if not pick:
                    raise ValueError(f"bayes_update.exemplar: no channel matches subject={ex['subject']} emg={ex['emg']}.")
                ch = pick[0]
                st = knob.apply(ch, ex["level"], rng_for(ch.label, "bayes_update", ex["level"], base_seed=cfg.seed))
                ts = [t for t in cfg.context_sizes if t < st.n_sites]
                ctxs = {t: context_for(st, t, int(ex["draw"]), base_seed=cfg.seed, level=float(ex["level"])) for t in ts}
                a = _common_anchor(st, list(ctxs.values()))
                maps = {}
                for t in ts:
                    def ex_cell(t: int = t) -> tuple[dict[str, Any], dict[str, Any]]:
                        res = {}
                        for n in ex["engines"]:
                            eng = _make_engine(n, p, cfg.device, int(ex["draw"]))
                            eng.fit(ctxs[t].X, ctxs[t].y)
                            qd = p["quadrature"]
                            _, C, _ = B.coherence(
                                eng, st.X_pool, np.array([a]), int(qd["K"]), float(p["eps"]), method=str(qd["method"]),
                                u_min=float(qd["u_min"]), central_mass=float(qd["central_mass"]),
                                n_scrambles=int(qd["n_scrambles"]),
                                rng=rng_for(ch.label, "bayes_update_exemplar", t, base_seed=cfg.seed),
                                u_tail=float(p["y_grid"]["u_tail"]), grid_points=int(p["y_grid"]["points"]))
                            res[n] = C[0].tolist()
                        return {"C": res}, {}

                    hit = cells.run("exemplar", {**cells.channel(ch), "level": float(ex["level"]), "draw": int(ex["draw"]),
                                                 "t": int(t), "anchor": a, "quadrature": p["quadrature"], "eps": p["eps"],
                                                 "engine_params": {n: _engine_params(n, p, int(ex["draw"]))
                                                                   for n in ex["engines"]}},
                                    f"exemplar {ch.label} t={t}", ex_cell)
                    if hit is not None:
                        maps[t] = hit[0]["C"]
                if maps:
                    np.savez(os.path.join(out, "exemplar.npz"), context_ts=np.array(sorted(maps)), anchor=a,
                             engines=np.array(ex["engines"]), ch2xy=np.asarray(st.ch2xy), channel=ch.label,
                             observed=np.array([np.isin(np.arange(st.n_sites), ctxs[t].sites) for t in sorted(maps)]),
                             C=np.array([[maps[t][n] for n in ex["engines"]] for t in sorted(maps)]))
                c1 = p["c1"]
                dmaps = {}
                for t in (ts if c1["reference"] in p["engines"] else []):
                    def c1_cell(t: int = t) -> tuple[dict[str, Any], dict[str, Any]]:
                        bayes = _make_engine("gp_bayes", p, cfg.device, int(ex["draw"]))
                        bayes.fit(ctxs[t].X, ctxs[t].y)
                        bb = bayes.base(st.X_pool)
                        y = float(bb.mean[0, a] + float(c1["exemplar_c"]) * bb.sd[0, a])
                        res = {}
                        for n in c1["engines"]:
                            eng = _make_engine(n, p, cfg.device, int(ex["draw"]))
                            eng.fit(ctxs[t].X, ctxs[t].y)
                            base = eng.base(st.X_pool)
                            pr = eng.probe(st.X_pool[[a]], np.array([y]), st.X_pool)
                            res[n] = (pr.mean[0] - base.mean[0]).tolist()                 # [M]
                        return {"dmu": res, "y_prime": y}, {}

                    hit = cells.run("c1_exemplar", {**cells.channel(ch), "level": float(ex["level"]),
                                                    "draw": int(ex["draw"]), "t": int(t), "anchor": a,
                                                    "exemplar_c": float(c1["exemplar_c"]),
                                                    "engine_params": {n: _engine_params(n, p, int(ex["draw"]))
                                                                      for n in [*c1["engines"], "gp_bayes"]}},
                                    f"C1 exemplar {ch.label} t={t}", c1_cell)
                    if hit is not None:
                        dmaps[t] = hit[0]["dmu"]
                if dmaps:
                    np.savez(os.path.join(out, "c1_exemplar.npz"), context_ts=np.array(sorted(dmaps)), anchor=a,
                             engines=np.array(c1["engines"]), ch2xy=np.asarray(st.ch2xy), channel=ch.label,
                             exemplar_c=float(c1["exemplar_c"]),
                             observed=np.array([np.isin(np.arange(st.n_sites), ctxs[t].sites) for t in sorted(dmaps)]),
                             dmu=np.array([[dmaps[t][n] for n in c1["engines"]] for t in sorted(dmaps)]))
        cells.report()
    link = p["link"]
    if link["tidy_csv"] or link["stress_tidy_csv"]:
        for path in _predictive_link(out, link, cfg.seed):
            print(f"[mechanism] wrote {path}")
    for path in figs.render_bayes_update(out):
        print(f"[mechanism] wrote {path}")
    return out


# ---------------------------------------------------------------------------
# shared writers
# ---------------------------------------------------------------------------
def _write_csv(rows: list[dict[str, Any]], out: str, name: str) -> None:
    """Write assembled rows to ``out/name``."""
    pd.DataFrame(rows).to_csv(os.path.join(out, name), index=False)



#: Column order of ``predictive_link.csv``.
LINK_COLUMNS: tuple[str, ...] = (
    "predictor", "part", "kind", "reference", "outcome", "slope", "ci_low", "ci_high", "intercept", "within_slope",
    "re_ratio", "spearman", "n_channels", "n_subjects", "n_boot_valid", "model", "pool", "level", "context_ts",
)


def _breakdown_margin(link: dict[str, Any]) -> float:
    """The breakdown margin: ``link.breakdown.margin``, else the stress run's own ``equivalence_margin``.

    Raises:
        ValueError: If neither is set.
    """
    margin = link["breakdown"]["margin"]
    if margin is not None:
        return float(margin)
    path = os.path.join(os.path.dirname(os.path.abspath(link["stress_tidy_csv"])), "config.yaml")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            margin = (yaml.safe_load(fh) or {}).get("equivalence_margin")
    if margin is None:
        raise ValueError(f"bayes_update.link.breakdown.margin is null and {path} gives no equivalence_margin.")
    return float(margin)


def _predictive_link(out: str, link: dict[str, Any], seed: int) -> list[str]:
    """M8 (exploratory): per-channel slopes of every outcome on every (predictor, reference) pair of the linked engine.

    Predictors come from the run's tables (:func:`analysis.predictive_link.cell_readouts`): per anchor of
    ``link.pool``, median per cell, median per channel at ``link.level`` over ``link.context_ts``. Part A from
    ``coherence_cell.csv``; Part B and C1 from ``headtohead.csv``; GP-MLL frozen (:data:`B_ONLY_ENGINES`) is paired
    with the C1 readouts only. Outcomes: ``link.tidy_csv`` (BO metrics, ``r2`` included) and
    ``link.stress_tidy_csv`` (per-channel breakdown). A pair with too few channels or a constant column is reported
    on stdout and left out of the table.

    Args:
        out: Run directory.
        link: Resolved ``bayes_update.link`` block.
        seed: Bootstrap seed.

    Returns:
        Paths of ``predictive_link.csv`` (one row per predictor x reference x outcome) and
        ``predictive_link_channels.csv`` (the per-channel points behind every slope, which the figures read).

    Raises:
        ValueError: If a listed reference has no contrast with the linked engine in this run.
    """
    from ..analysis import predictive_link as L  # noqa: PLC0415

    coh = pd.read_csv(os.path.join(out, "coherence_cell.csv"))
    h2h = pd.read_csv(os.path.join(out, "headtohead.csv"))
    present = L.references_in_run(h2h, link["model"])
    refs = present if link["references"] == "all" else list(link["references"])
    absent = [r for r in refs if r not in present]
    if absent:
        raise ValueError(f"bayes_update.link.references {absent}: the run has no {link['model']} contrast with "
                         f"them (present: {present}).")
    pairs = L.reference_pairs(link["predictors"], refs, c1_references=B_ONLY_ENGINES,
                              engine_contrasts=bool(link["engine_contrasts"]))
    cells = L.cell_readouts(coh, h2h, pairs, model=link["model"], pool=link["pool"], c1_references=B_ONLY_ENGINES)
    mech = L.channel_readouts(cells, level=float(link["level"]), context_ts=link["context_ts"])
    bo_outcomes = [o for o in link["outcomes"] if o != L.BREAKDOWN_OUTCOME]
    frames: list[pd.DataFrame] = []
    if bo_outcomes:
        frames.append(L.channel_outcomes(pd.read_csv(link["tidy_csv"]), bo_outcomes, model=link["model"],
                                         acq_label=link["acq_label"]))
    if L.BREAKDOWN_OUTCOME in link["outcomes"]:
        bd = link["breakdown"]
        frames.append(L.channel_breakdown(pd.read_csv(link["stress_tidy_csv"]), model=link["model"],
                                          reference=bd["reference"], metric=bd["metric"],
                                          margin=_breakdown_margin(link), acq_label=link["acq_label"]))
    outcomes = frames[0]
    for frame in frames[1:]:
        outcomes = outcomes.merge(frame, on=L.CHANNEL, how="outer")
    joined = mech.merge(outcomes, on=L.CHANNEL, how="inner")
    context = "all" if link["context_ts"] is None else "/".join(str(int(t)) for t in link["context_ts"])
    rows: list[dict[str, Any]] = []
    for outcome in link["outcomes"]:
        for pred, ref in pairs:
            sub = joined[(joined["predictor"] == pred) & (joined["reference"] == ref)]
            try:
                res = L.predictive_link(sub.rename(columns={"value": pred}), pred, outcome,
                                        n_boot=int(link["n_boot"]), seed=seed)
            except ValueError as exc:
                print(f"[mechanism] link {pred} vs {ref} -> {outcome}: skipped ({exc})", flush=True)
                continue
            readout = L.READOUTS[pred]
            rows.append({**res, "part": readout.part, "kind": readout.kind, "reference": ref, "model": link["model"],
                         "pool": link["pool"], "level": float(link["level"]), "context_ts": context})
    path = os.path.join(out, "predictive_link.csv")
    pd.DataFrame(rows, columns=list(LINK_COLUMNS)).to_csv(path, index=False)
    chan_path = os.path.join(out, "predictive_link_channels.csv")
    joined.to_csv(chan_path, index=False)
    return [path, chan_path]


# ---------------------------------------------------------------------------
# placement (task #6)
# ---------------------------------------------------------------------------
PLACEMENT_DEFAULTS: dict[str, Any] = {
    "metrics": ["mmd", "w2"],
    "representation": "rank_pairs",
    "n_neighbors": 4,
    "n_prior": 200,
    "n_prior_holdout": 50,
    "n_noise": 50,
    "n_dense": 1024,
    "holdout_frac": 0.1,
    "rmse_threshold": 0.5,
    "prior_type": "prior_bag",
    "k_nearest": 10,
    "n_projections": 200,
    "w2_repeats": 20,
    "n_splits": 5,
    "n_shuffles": 10,
    "n_boot": 1000,
    "ladder_lambdas": [0.0, 0.25, 0.5, 0.75, 1.0],
    "ladder_min_spearman": 0.9,
    # Formulation C: the context sizes are the shared ladder (``context.sizes``) plus the full map.
    "context": {
        "n_draws": 10,
        "stress": {"knob": "k2_channel", "levels": [1.0, 2.0, 4.0]},
    },
}


def run_placement(cfg: MechanismConfig, opts: RunOptions) -> str:
    """Task #6: placement of every channel between the prior and noise (B, C, A).

    Args:
        cfg: Resolved config.
        opts: Cache policy.

    Returns:
        The run directory.
    """
    from scipy.stats import spearmanr  # noqa: PLC0415

    from ..analysis import placement as P  # noqa: PLC0415
    from ..data.references.grid import standardize_map  # noqa: PLC0415
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _check_keys(cfg.params, PLACEMENT_DEFAULTS, "placement")
    p["context"] = _check_keys(p["context"], PLACEMENT_DEFAULTS["context"], "placement.context")
    out = cfg.run_dir
    if not opts.replot:
        set_seed(cfg.seed)
        channels = _channels(cfg)
        if not channels:
            raise RuntimeError("placement: no channels loaded.")
        cells = _Cells(cfg, opts)

        def feat(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            return P.map_features(X, y, p["representation"], int(p["n_neighbors"]))

        k = int(p["k_nearest"])
        bank_keys = {key: p[key] for key in ("n_dense", "holdout_frac", "rmse_threshold", "prior_type")}
        n_total = int(p["n_prior"]) + int(p["n_prior_holdout"])
        grids: dict[bytes, list[ChannelData]] = {}
        for ch in channels:
            grids.setdefault(ch.X_pool.tobytes(), []).append(ch)
        banks: dict[bytes, dict[str, Any]] = {}
        bank_log: list[dict[str, Any]] = []
        for key, members in grids.items():
            pair = _grid_banks(cells, members[0], n_total, int(p["n_noise"]), bank_keys)
            if pair is None:
                continue
            prior, noise = pair
            n_hold = min(int(p["n_prior_holdout"]), len(prior.maps) // 2)
            ch = members[0]
            banks[key] = {
                "train": [feat(ch.X_pool, m) for m in prior.maps[n_hold:]],
                "holdout": [feat(ch.X_pool, m) for m in prior.maps[:n_hold]],
                "prior_maps": prior.maps, "n_hold": n_hold,
                "noise": [feat(ch.X_pool, m) for m in noise.maps], "noise_maps": noise.maps,
                "grid_id": grid_id(ch.X_pool),
            }
            finite = [r for r in prior.rmse if np.isfinite(r)]
            bank_log.append({
                "grid_of": ch.label, "n_sites": ch.n_sites, "n_attempted": n_total,
                "n_kept": int(len(prior.maps)), "rejection_rate": prior.rejection_rate,
                "rmse_median": float(np.median(finite)) if finite else None,
                "rmse_threshold": float(p["rmse_threshold"]),
            })
        if not banks:
            raise RuntimeError("placement: no prior bank available (an --only-cached run needs the bank artifacts).")

        # One fixed parameter set per metric per analysis (Step 2), from the first grid's bank.
        first = banks[next(iter(banks))]
        metrics: dict[str, Any] = {}
        for name in p["metrics"]:
            if name == "mmd":
                metrics[name] = P.make_metric("mmd", bandwidth=P.median_bandwidth(first["train"], seed=cfg.seed))
            else:
                proj = P.projection_set(first["train"][0].shape[1], int(p["n_projections"]), cfg.seed)
                metrics[name] = P.make_metric("w2", projections=proj, n_repeats=int(p["w2_repeats"]), seed=cfg.seed)
        common = {key: p[key] for key in ("metrics", "representation", "n_neighbors", "k_nearest", "n_projections",
                                          "w2_repeats", "n_prior", "n_prior_holdout", "n_noise")}
        common.update(bank_keys)

        # M0 known-shift ladder on the first channel (Step 5): a failing metric is flagged invalid.
        gates: list[dict[str, Any]] = []
        ch0 = channels[0]
        for name, m in metrics.items():
            def ladder_gate(name: str = name, m: Any = m) -> tuple[dict[str, Any], dict[str, Any]]:
                lad = P.known_shift_ladder(
                    ch0.X_pool, standardize_map(ch0.y_gt), first["noise_maps"][0], first["train"], m, k,
                    p["ladder_lambdas"], featurize=feat,
                )
                return {"gate": {"gate": "known_shift_ladder", "metric": name, "channel": ch0.label, **lad,
                                 "passed": bool(lad["spearman"] > float(p["ladder_min_spearman"]))}}, {}

            hit = cells.run("gate", {**common, **cells.channel(ch0), "gate": "known_shift_ladder", "metric": name,
                                     "ladder_lambdas": p["ladder_lambdas"],
                                     "ladder_min_spearman": p["ladder_min_spearman"]},
                            f"ladder {name}", ladder_gate)
            if hit is not None:
                gates.append(hit[0]["gate"])
        valid = {g["metric"]: g["passed"] for g in gates}

        # Formulation B with both floors and both ceilings (Steps 3-4), one cell per channel.
        train_prep: dict[tuple[bytes, str], Any] = {}

        def prepared(key: bytes, name: str) -> Any:
            if (key, name) not in train_prep:
                train_prep[(key, name)] = P.prepare_bank(banks[key]["train"], metrics[name])
            return train_prep[(key, name)]

        ref_cache: dict[tuple[bytes, str], tuple[np.ndarray, np.ndarray]] = {}
        rows: list[dict[str, Any]] = []
        for ch in channels:
            key = ch.X_pool.tobytes()
            if key not in banks:
                continue
            bank = banks[key]

            def formulation_b(ch: ChannelData = ch, key: bytes = key, bank: dict[str, Any] = bank,
                              ) -> tuple[dict[str, Any], dict[str, Any]]:
                from ..data.ground_truth import build_ground_truth  # noqa: PLC0415

                y = standardize_map(ch.y_gt)
                Z = feat(ch.X_pool, y)
                rng = rng_for(ch.label, "placement", base_seed=cfg.seed)
                halves = build_ground_truth(ch.Y_trials, "split_half", rng, int(p["n_splits"]))
                split_maps = [(halves[r][0], halves[r + 1][0]) for r in range(0, len(halves), 2)]
                X_split = ch.X_pool[np.isfinite(ch.Y_trials).sum(axis=1) >= 2]
                out_rows = []
                for name, m in metrics.items():
                    train = prepared(key, name)
                    if (key, name) not in ref_cache:
                        ref_cache[(key, name)] = (
                            np.array([P.distance_to_bank(h, train, m, k) for h in bank["holdout"]]),
                            np.array([P.distance_to_bank(nz, train, m, k) for nz in bank["noise"]]),
                        )
                    floor1, ceil1 = ref_cache[(key, name)]
                    floor2 = np.array([
                        m(feat(X_split, standardize_map(a)), feat(X_split, standardize_map(b))) for a, b in split_maps
                    ])
                    ceil2 = np.array([
                        P.distance_to_bank(feat(ch.X_pool, rng.permutation(y)), train, m, k)
                        for _ in range(int(p["n_shuffles"]))
                    ])
                    d = P.distance_to_bank(Z, train, m, k)
                    row: dict[str, Any] = {
                        "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg, "metric": name,
                        "representation": p["representation"], "d": d,
                        "floor1": float(np.median(floor1)), "floor2": float(np.median(floor2)),
                        "ceiling1": float(np.median(ceil1)), "ceiling2": float(np.median(ceil2)),
                    }
                    for fi, fl in (("1", floor1), ("2", floor2)):
                        for ci, ce in (("1", ceil1), ("2", ceil2)):
                            pv, lo, hi = P.bootstrap_placement(d, fl, ce, n_boot=int(p["n_boot"]), seed=cfg.seed)
                            tag = f"f{fi}c{ci}"
                            row[f"p_{tag}"], row[f"p_{tag}_lo"], row[f"p_{tag}_hi"] = pv, lo, hi
                            row[f"gap_{tag}"] = float(np.median(ce) - np.median(fl))
                    out_rows.append(row)
                return {"rows": out_rows}, {}

            hit = cells.run("B", {**common, **cells.channel(ch), "n_splits": p["n_splits"],
                                  "n_shuffles": p["n_shuffles"], "n_boot": p["n_boot"]},
                            f"{ch.label} formulation B", formulation_b)
            if hit is not None:
                rows += [{**r, "metric_valid": valid.get(r["metric"])} for r in hit[0]["rows"]]
        df = pd.DataFrame(rows)

        # Formulation C: random size-t noisy contexts along the knob, at matched sites (Step 4); one cell per
        # channel (every level and context size), so the full-map references are computed once per channel.
        from ..data.snr import achieved_snr_db  # noqa: PLC0415
        from ..data.stress import build_knob  # noqa: PLC0415

        cp = p["context"]
        knob = build_knob(cp["stress"]["knob"], cp["stress"]["levels"])
        crow: list[dict[str, Any]] = []
        t0 = time.time()
        for ch in channels:
            key = ch.X_pool.tobytes()
            if key not in banks:
                continue

            def formulation_c(ch: ChannelData = ch, key: bytes = key) -> tuple[dict[str, Any], dict[str, Any]]:
                bank = banks[key]
                train_maps = bank["prior_maps"][bank["n_hold"]:]
                hold_maps = bank["prior_maps"][: bank["n_hold"]]
                full_cache: dict[str, tuple[Any, float, float]] = {}
                out_rows: list[dict[str, Any]] = []
                for level in knob.levels:
                    stressed = knob.apply(ch, level, rng_for(ch.label, "placement_c", level, base_seed=cfg.seed))
                    snr = achieved_snr_db(stressed)
                    for t in list(cfg.context_sizes) + ["full"]:
                        tt = ch.n_sites if t == "full" else int(t)
                        if tt > ch.n_sites or tt <= int(p["n_neighbors"]):
                            continue
                        for draw in range(int(cp["n_draws"])):
                            # Shared sites, this channel's own trials (#19 Step 1).
                            ctx = context_for(stressed, tt, draw, base_seed=cfg.seed, level=float(level))
                            sites, yv = ctx.sites, ctx.y
                            if np.std(yv) == 0:
                                continue
                            Xs = ch.X_pool[sites]
                            Zc = feat(Xs, standardize_map(yv))

                            def at_sites(maps: np.ndarray, sites: np.ndarray = sites, Xs: np.ndarray = Xs,
                                         ) -> list[np.ndarray]:
                                return [feat(Xs, standardize_map(mp[sites])) for mp in maps]

                            is_full = tt == ch.n_sites
                            refs: dict[str, tuple[Any, float, float]] = {}
                            if is_full and full_cache:
                                refs = full_cache
                            else:
                                ref, hold, noise_s = at_sites(train_maps), at_sites(hold_maps), at_sites(bank["noise_maps"])
                                for name, m in metrics.items():
                                    ref_p = P.prepare_bank(ref, m)
                                    refs[name] = (
                                        ref_p,
                                        float(np.median([P.distance_to_bank(h, ref_p, m, k) for h in hold])),
                                        float(np.median([P.distance_to_bank(nz, ref_p, m, k) for nz in noise_s])),
                                    )
                                if is_full:
                                    full_cache.update(refs)
                            for name, m in metrics.items():
                                ref_p, fl, ce = refs[name]
                                d = P.distance_to_bank(Zc, ref_p, m, k)
                                out_rows.append({
                                    "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg, "metric": name,
                                    "knob": knob.name, "level": float(level), "achieved_snr_db": snr,
                                    "context_t": tt, "is_full": t == "full", "draw": draw, "d": d,
                                    "floor1": fl, "ceiling1": ce, "p": P.placement(d, fl, ce),
                                })
                return {"rows": out_rows}, {}

            hit = cells.run("C", {**common, **cells.channel(ch), "sizes": list(cfg.context_sizes),
                                  "n_draws": int(cp["n_draws"]), "stress": cp["stress"]},
                            f"{ch.label} formulation C", formulation_c)
            if hit is not None:
                crow += hit[0]["rows"]
            print(f"[mechanism] placement-C {ch.label} done ({time.time() - t0:.0f}s)", flush=True)

        # Formulation A: pooled marginal of y (appendix, labelled "marginal"). Cheap; computed at assembly.
        arow: list[dict[str, Any]] = []
        pooled_ch = np.concatenate([standardize_map(ch.y_gt) for ch in channels])[:, None]
        pooled_pr = np.concatenate([b["prior_maps"].reshape(-1) for b in banks.values()])[:, None]
        pooled_nz = np.concatenate([b["noise_maps"].reshape(-1) for b in banks.values()])[:, None]
        n_eq = min(len(pooled_ch), len(pooled_pr) // 2, len(pooled_nz))
        rng = np.random.default_rng(cfg.seed)

        def take(a: np.ndarray) -> np.ndarray:
            return a[rng.choice(len(a), n_eq, replace=False)]

        pr_a, pr_b = take(pooled_pr), take(pooled_pr)
        for name in p["metrics"]:
            if name == "mmd":
                bw = P.median_bandwidth([pr_a], seed=cfg.seed)

                def fn(a: np.ndarray, b: np.ndarray) -> float:
                    return P.mmd2_unbiased(a, b, bw)
            else:
                proj1 = P.projection_set(1, 1, cfg.seed)

                def fn(a: np.ndarray, b: np.ndarray) -> float:
                    return P.sliced_w2(a, b, proj1)
            d, fl, ce = fn(take(pooled_ch), pr_a), fn(pr_b, pr_a), fn(take(pooled_nz), pr_a)
            arow.append({"dataset": cfg.dataset.name, "metric": name, "label": "marginal", "n": n_eq,
                         "d": d, "floor1": fl, "ceiling1": ce, "p": P.placement(d, fl, ce)})

        _write_config(cfg, p, out)
        df.to_csv(os.path.join(out, "placement.csv"), index=False)
        pd.DataFrame(crow).to_csv(os.path.join(out, "placement_context.csv"), index=False)
        pd.DataFrame(arow).to_csv(os.path.join(out, "placement_pooled.csv"), index=False)
        summary: dict[str, Any] = {
            "banks": bank_log,
            "metric_params": {n: m.params for n, m in metrics.items()},
            "gates": gates,
        }
        if {"mmd", "w2"} <= set(metrics) and len(channels) >= 3 and not df.empty:
            wide = df.pivot_table(index=["subject", "emg"], columns="metric", values="d")
            summary["rho_mmd_w2_channel"] = float(spearmanr(wide["mmd"], wide["w2"]).correlation)
        with open(os.path.join(out, "placement_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=_json_default)
        for g in gates:
            print(f"[mechanism] gate {g['gate']} {g['metric']}: rho={g['spearman']:.2f} "
                  f"{'PASS' if g['passed'] else 'FAIL (metric flagged invalid)'}", flush=True)
    for path in figs.render_placement(out):
        print(f"[mechanism] wrote {path}")
    if not opts.replot:
        cells.report()
    return out


# ---------------------------------------------------------------------------
# cka (task #5, redesigned in B1 on 2026-10-06; feature tokens only since 2026-10-07)
# ---------------------------------------------------------------------------
#: The readout CKA (b) places. Placement asks where the network's representation of the electrode geometry sits
#: between the prior and noise references; the per-token shadow readout is a CKA (a) question only.
CKA_PLACEMENT_READOUT: str = "feature_mean"

#: Statistic set the placement (``b``) cell payload carries. A ``b`` cell caches its finished rows, so an older
#: cell cannot serve a column it never computed; this stamp is part of the cell identity, so bumping it supersedes
#: the previous generation without discarding it (the expensive ``a`` / ``gpfit`` cells are untouched). Bumped to 2
#: on 2026-10-14, when ``rank_vs_floor`` / ``rank_vs_ceiling`` / ``p_rank`` and the reference counts joined the rows.
CKA_PLACEMENT_STATS_VERSION: int = 2

CKA_DEFAULTS: dict[str, Any] = {
    "layers": None,
    # Feature-token readouts (analysis/embeddings.READOUTS): ``feature_mean``, and ``feature_tokens`` -- the
    # per-token shadow readout of B1 Step 5 (CKA (a) only). The label token and the decoder stages that follow it
    # were dropped on 2026-10-07 (user): they re-measure the prediction, not the representation of the geometry.
    "readouts": ["feature_mean"],
    "targets": ["K_X", "K_GT", "K_GT_linear", "K_GP"],
    "stress": {"knob": "k2_channel", "levels": [1.0, 2.0, 4.0]},
    "n_context_draws": 3,
    "n_perm": 1000,
    "inference_seed": 0,
    "gp_params": {"n_opt_steps": 100, "lr": 0.1},
    "controls": {
        "grid_side": 10, "lengthscale": 0.2, "noise_sd": 0.3, "n_trials": 10, "t": 25,
        "alpha": 0.05, "n_seeds": 10, "icc_min": 0.75, "n_seed_channels": 1,
    },
    # CKA (b): placement of a channel's layer-wise feature-mean embedding between the on-grid prior bank and the noise bank,
    # at every context size of the shared ladder (B3 Step 4 replaced the old ``ts`` key).
    #   n_prior    on-grid prior maps forming the comparison bank.
    #   n_noise    noise maps forming the ceiling.
    #   n_draws    independent context draws; the figure averages within a channel first.
    #   min_gap    rows whose (ceiling - floor) is below this are marked ``gap_ok = False`` (see 2026-09-27).
    #   bank_quantile
    #              which quantile of the channel-to-bank distances defines "distance to the prior bag" (it
    #              converges as n_prior grows, unlike the old absolute k_nearest).
    "placement": {
        "enabled": True, "n_prior": 100, "n_prior_holdout": 30, "n_noise": 30,
        "n_draws": 5, "min_gap": 0.02,
        "bank_quantile": 0.10, "n_dense": 1024, "holdout_frac": 0.1, "rmse_threshold": 0.5,
        "prior_type": "prior_bag",
    },
    # B1 Step 5: the pre-registered success criterion of a shadow readout, evaluated at assembly and written to
    # ``shadow_criterion.json``; ``null`` disables it. See configs/experiment/mechanism_cka_tokens_nhp.yaml.
    "shadow": None,
    # The CKA **ceiling** (user, 2026-10-12). Debiased CKA against a structured target is bounded well below 1 even
    # under a perfect monotone relationship, because the two Gram matrices have different spectra -- so an absolute
    # value like "TabPFN scores 0.23 against K_GP" is uninterpretable on its own, and the permutation null only
    # supplies a FLOOR. This computes the reachable ceiling instead: ``noise_ceiling`` is the CKA between two GP
    # posterior covariances fitted on the SAME sites with INDEPENDENT trial draws (so they differ only by measurement
    # noise, which is the best any method could score against a noisily estimated target), and ``gp_vs_gt`` is the CKA
    # between the GP's posterior covariance and the ground-truth response kernel (how well the comparator models the
    # surface at all). Reported so a measured CKA can be read as a fraction of what is achievable.
    #   trial_draw_offset  re-keys the trial stream only (analysis/context.context_for), holding the sites fixed.
    "ceiling": {"enabled": True, "trial_draw_offset": 100},
}

#: Keys of a ``cka.shadow`` block.
CKA_SHADOW_KEYS: dict[str, Any] = {
    "readout": "feature_tokens", "baseline": "feature_mean", "targets": ["K_GT", "K_GP"],
    "level": 1.0, "max_iqr_ratio": 0.75,
}


def _layer_grams(emb: Any, index: int) -> list[np.ndarray]:
    """Linear Gram matrices of one layer: one per feature token for ``feature_tokens``, else one."""
    from ..analysis.cka import linear_gram  # noqa: PLC0415

    arr = emb.arrays[index]
    if arr.ndim == 3:                                                       # [n_tok, q, d]
        return [linear_gram(a) for a in arr]
    return [linear_gram(arr)]                                               # [q, d]


def _cka_layer_table(
    emb: Any, kernels: dict[str, np.ndarray], n_perm: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Debiased CKA with a site-permutation null for every (layer, target).

    One set of permutations per target is shared by every layer (and every feature token) of the cell, see
    :func:`analysis.cka.cka_null_draws`. For ``feature_tokens`` the statistic is the CKA averaged over tokens and
    the null is averaged over tokens draw by draw, so ``p`` tests the averaged statistic.

    Args:
        emb: :class:`~pfns4neurostim.analysis.embeddings.LayerEmbeddings` of one readout.
        kernels: Target Gram matrices, each [N, N].
        n_perm: Permutations.
        rng: Generator.

    Returns:
        Rows with ``layer, target, value, null_mean, cka_minus_null, p, n_tokens``; ``layer`` is the block index.
    """
    from ..analysis.cka import cka_null_draws, permutation_p  # noqa: PLC0415

    grams = [_layer_grams(emb, i) for i in range(len(emb.arrays))]
    flat = [G for gs in grams for G in gs]
    owner = np.repeat(np.arange(len(grams)), [len(gs) for gs in grams])    # [K] layer of each Gram
    rows = []
    for target, L in kernels.items():
        obs, null = cka_null_draws(flat, L, n_perm, rng)                    # [K], [K, P]
        for i in range(len(grams)):
            sel = owner == i
            o, nl = float(obs[sel].mean()), null[sel].mean(axis=0)          # token-averaged
            rows.append({"layer": int(emb.layers[i]), "target": target, "value": o,
                         "null_mean": float(nl.mean()), "cka_minus_null": o - float(nl.mean()),
                         "p": permutation_p(o, nl), "n_tokens": int(sel.sum())})
    return rows


def _layer_cka(emb: Any, kernels: dict[str, np.ndarray]) -> list[float]:
    """Debiased CKA (token-averaged) at every (layer, target), without a null (seed-stability control)."""
    from ..analysis.cka import cka_debiased  # noqa: PLC0415

    return [float(np.mean([cka_debiased(G, K) for G in _layer_grams(emb, i)]))
            for i in range(len(emb.arrays)) for K in kernels.values()]


def run_cka(cfg: MechanismConfig, opts: RunOptions) -> str:
    """Task #5 / B1: CKA (a) of the feature tokens against target kernels, controls, and CKA (b) placement.

    Args:
        cfg: Resolved config.
        opts: Cache policy.

    Returns:
        The run directory.
    """
    from ..analysis.cka import linear_gram, target_kernels  # noqa: PLC0415
    from ..analysis.embeddings import READOUTS, readout_embeddings  # noqa: PLC0415
    from ..analysis import placement as P  # noqa: PLC0415
    from ..analysis.placement import placement as place  # noqa: PLC0415
    from ..analysis.context import draw_context  # noqa: PLC0415
    from ..data.snr import achieved_snr_db  # noqa: PLC0415
    from ..data.stress import build_knob  # noqa: PLC0415
    from ..data.synthetic_neurostim import gp_channel  # noqa: PLC0415
    from ..evaluation.stats import icc_oneway  # noqa: PLC0415
    from ..models.gp.surrogates import GPSurrogate  # noqa: PLC0415
    from ..models.pfn.tabpfn import FrozenTabPFN  # noqa: PLC0415
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _check_keys(cfg.params, CKA_DEFAULTS, "cka")
    p["controls"] = _check_keys(p["controls"], CKA_DEFAULTS["controls"], "cka.controls")
    p["placement"] = _check_keys(p["placement"], CKA_DEFAULTS["placement"], "cka.placement")
    p["ceiling"] = _check_keys(p["ceiling"], CKA_DEFAULTS["ceiling"], "cka.ceiling")
    if p["shadow"] is not None:
        p["shadow"] = _check_keys(p["shadow"], CKA_SHADOW_KEYS, "cka.shadow")
    bad = [r for r in p["readouts"] if r not in READOUTS]
    if bad:
        raise ValueError(f"cka.readouts: unknown {bad}; expected a subset of {READOUTS}.")
    n_layers = 18 if p["layers"] is None else len(p["layers"])
    if 1.0 / (int(p["n_perm"]) + 1) >= p["controls"]["alpha"] / n_layers:
        raise ValueError(
            f"cka.n_perm={p['n_perm']} cannot resolve the Bonferroni threshold "
            f"{p['controls']['alpha']}/{n_layers}: the smallest attainable p is 1/(n_perm+1)."
        )
    out = cfg.run_dir
    if not opts.replot:
        set_seed(cfg.seed)
        _write_config(cfg, p, out)
        cells = _Cells(cfg, opts)
        engine_box: dict[str, Any] = {}

        def engine() -> Any:
            if "e" not in engine_box:
                engine_box["e"] = FrozenTabPFN(device=cfg.device, random_state=int(p["inference_seed"]))
            return engine_box["e"]

        def embed(X_ctx: np.ndarray, y_ctx: np.ndarray, X_sites: np.ndarray, readout: str,
                  eng: Any | None = None) -> Any:
            eng = engine() if eng is None else eng
            eng.fit(X_ctx, y_ctx)
            return readout_embeddings(eng, X_sites, readouts=(readout,), layers=p["layers"])[readout]

        def embed_ident(readout: str) -> dict[str, Any]:
            return {"readout": readout, "layers": p["layers"], "inference_seed": int(p["inference_seed"])}

        def gp_cov(ch: ChannelData, ctx: Any, ident: dict[str, Any]) -> np.ndarray:
            """The MLL-GP posterior covariance of a context: a cached artifact shared by every readout."""
            def compute() -> tuple[dict[str, Any], dict[str, Any]]:
                gp = GPSurrogate(**p["gp_params"])
                gp.fit(ctx.X, ctx.y)
                return {"diagnostics": gp.fit_diagnostics()}, {"cov": gp.posterior_covariance(ch.X_pool)}

            hit = cells.run("gpfit", {**ident, "gp_params": p["gp_params"]}, f"GP {ident.get('label')}", compute)
            if hit is None:
                raise RuntimeError("the context's GP fit failed (see the gpfit failure above).")
            return hit[1]["cov"]

        def kernels_for(ch: ChannelData, ctx: Any, ident: dict[str, Any]) -> dict[str, np.ndarray]:
            cov = gp_cov(ch, ctx, ident) if "K_GP" in p["targets"] else None
            ks = target_kernels(ch.X_pool, ch.y_gt, cov)
            return {k: v for k, v in ks.items() if k in p["targets"]}

        channels = _channels(cfg)
        knob = build_knob(p["stress"]["knob"], p["stress"]["levels"])
        rows: list[dict[str, Any]] = []
        for ch in channels:
            for level in knob.levels:
                st = knob.apply(ch, level, rng_for(ch.label, "cka", level, base_seed=cfg.seed))
                snr = achieved_snr_db(st)
                for t in cfg.context_sizes:
                    if t >= st.n_sites:
                        continue
                    for rep in range(int(p["n_context_draws"])):
                        ctx_ident = {**cells.channel(ch), "knob": knob.name, "level": float(level), "t": int(t),
                                     "draw": rep}
                        for readout in p["readouts"]:
                            def compute(level: float = level, t: int = t, rep: int = rep, readout: str = readout,
                                        st: ChannelData = st, ctx_ident: dict[str, Any] = ctx_ident,
                                        ) -> tuple[dict[str, Any], dict[str, Any]]:
                                ctx = context_for(st, int(t), rep, base_seed=cfg.seed, level=float(level))
                                kernels = kernels_for(st, ctx, ctx_ident)
                                emb = embed(ctx.X, ctx.y, st.X_pool, readout)
                                # Keyed per cell, so the null is reproducible and independent of run order.
                                perm_rng = rng_for(ch.label, "cka-perm", level, t, rep, readout, base_seed=cfg.seed)
                                return {"rows": _cka_layer_table(emb, kernels, int(p["n_perm"]), perm_rng)}, {}

                            hit = cells.run("a", {**ctx_ident, **embed_ident(readout), "targets": list(p["targets"]),
                                                  "n_perm": int(p["n_perm"]), "gp_params": p["gp_params"]},
                                            f"{ch.label} a={level:g} t={t} d={rep} {readout}", compute)
                            if hit is None:
                                continue
                            for r in hit[0]["rows"]:
                                rows.append({
                                    "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg,
                                    "model": MECHANISM_MODEL, "readout": readout, "context_t": int(t),
                                    "knob": knob.name, "level": float(level), "achieved_snr_db": snr,
                                    "rep": rep, **r,
                                })
            print(f"[mechanism] cka {ch.label} done", flush=True)
        _write_csv(rows, out, "cka.csv")

        # --- the reachable CKA ceiling (user, 2026-10-12) ---
        if p["ceiling"]["enabled"] and "K_GP" in p["targets"]:
            ceil_rows = _cka_ceiling(cfg, p, cells, channels, knob)
            if ceil_rows:
                _write_csv(ceil_rows, out, "cka_ceiling.csv")

        # --- M0 controls (Step 4), per readout ---
        c = p["controls"]
        gates: list[dict[str, Any]] = []
        for readout in p["readouts"]:
            for label, noise_only in (("positive_control", False), ("negative_control", True)):
                def control(readout: str = readout, noise_only: bool = noise_only, label: str = label,
                            ) -> tuple[dict[str, Any], dict[str, Any]]:
                    from dataclasses import replace  # noqa: PLC0415

                    rng = np.random.default_rng(cfg.seed)
                    ctrl = gp_channel(c["grid_side"], c["lengthscale"], c["noise_sd"], c["n_trials"], rng)
                    if noise_only:
                        y = rng.normal(size=ctrl.n_sites)
                        ctrl = replace(ctrl, y_gt=y,
                                       Y_trials=y[:, None] + c["noise_sd"] * rng.normal(size=ctrl.Y_trials.shape))
                    ctx = draw_context(ctrl, int(c["t"]), rng)
                    emb = embed(ctx.X, ctx.y, ctrl.X_pool, readout)
                    tab = _cka_layer_table(emb, {"K_GT": target_kernels(ctrl.X_pool, ctrl.y_gt)["K_GT"]},
                                           int(p["n_perm"]), rng)
                    min_p = min(r["p"] for r in tab)
                    bonf = c["alpha"] / len(tab)
                    return {"gate": {
                        "gate": label, "readout": readout, "min_p": min_p, "alpha_bonferroni": bonf,
                        "max_cka_minus_null": max(r["cka_minus_null"] for r in tab),
                        "passed": bool(min_p < bonf) if not noise_only else bool(min_p >= bonf),
                    }}, {}

                hit = cells.run("gate", {"gate": label, "controls": c, **embed_ident(readout),
                                         "n_perm": int(p["n_perm"])}, f"{label} {readout}", control)
                if hit is not None:
                    gates.append(hit[0]["gate"])
            for ch in channels[: int(c["n_seed_channels"])]:
                def seed_stability(readout: str = readout, ch: ChannelData = ch,
                                   ) -> tuple[dict[str, Any], dict[str, Any]]:
                    ctx = draw_context(ch, min(int(c["t"]), ch.n_sites - 1), np.random.default_rng(cfg.seed))
                    gp = GPSurrogate(**p["gp_params"])
                    gp.fit(ctx.X, ctx.y)
                    ks = {k: v for k, v in target_kernels(ch.X_pool, ch.y_gt, gp.posterior_covariance(ch.X_pool)).items()
                          if k in p["targets"]}
                    per_seed = [_layer_cka(embed(ctx.X, ctx.y, ch.X_pool, readout,
                                                 eng=FrozenTabPFN(device=cfg.device, random_state=s)), ks)
                                for s in range(int(c["n_seeds"]))]
                    icc = icc_oneway(np.asarray(per_seed).T)
                    return {"gate": {"gate": "seed_stability", "readout": readout, "channel": ch.label,
                                     "n_seeds": int(c["n_seeds"]), "icc": icc,
                                     "passed": bool(icc >= c["icc_min"])}}, {}

                hit = cells.run("gate", {"gate": "seed_stability", "controls": c, **cells.channel(ch),
                                         **embed_ident(readout), "targets": list(p["targets"]),
                                         "gp_params": p["gp_params"]},
                                f"seed stability {readout} {ch.label}", seed_stability)
                if hit is not None:
                    gates.append(hit[0]["gate"])

        # --- CKA (b) placement (Step 6) of the feature-token mean, one cell per (grid, t, draw) ---
        # Channel-independent work at a fixed (grid, t, draw) -- the banks, the bank maps' embeddings at the
        # context sites, and hence the floor and ceiling -- is done once per cell for every channel of the grid
        # (2026-09-30). The context sites come from a grid-keyed stream, so the comparison is paired.
        pl = p["placement"]
        prow: list[dict[str, Any]] = []
        if pl["enabled"]:
            q = float(pl["bank_quantile"])
            min_gap = float(pl["min_gap"])
            n_total = int(pl["n_prior"]) + int(pl["n_prior_holdout"])
            bank_keys = {key: pl[key] for key in ("n_dense", "holdout_frac", "rmse_threshold", "prior_type")}
            grids: dict[bytes, list[ChannelData]] = {}
            for ch in channels:
                grids.setdefault(ch.X_pool.tobytes(), []).append(ch)
            for grid_channels in grids.values():
                ref = grid_channels[0]
                grid = grid_id(ref.X_pool)
                bank_box: dict[str, Any] = {}

                def banks() -> tuple[Any, Any]:
                    if "b" not in bank_box:
                        pair = _grid_banks(cells, ref, n_total, int(pl["n_noise"]), bank_keys)
                        if pair is None:
                            raise RuntimeError("prior / noise bank unavailable.")
                        bank_box["b"] = pair
                    return bank_box["b"]

                readout = CKA_PLACEMENT_READOUT
                for t_ctx, draw in itertools.product(
                        [t for t in cfg.context_sizes if t < ref.n_sites], range(int(pl["n_draws"]))):
                    def compute(t_ctx: int = t_ctx, draw: int = draw,
                                ref: ChannelData = ref, grid_channels: list[ChannelData] = grid_channels,
                                grid: str = grid) -> tuple[dict[str, Any], dict[str, Any]]:
                        prior, noise = banks()
                        n_hold = min(int(pl["n_prior_holdout"]), len(prior.maps) // 2)
                        sites = context_sites(ref.X_pool, t_ctx, draw, base_seed=cfg.seed)          # [t]

                        layer_ids: list[tuple[int, ...]] = []

                        def grams(y_map: np.ndarray) -> list[np.ndarray]:
                            emb = embed(ref.X_pool[sites], y_map[sites], ref.X_pool, readout)
                            if not layer_ids:
                                layer_ids.append(emb.layers)
                            return [linear_gram(a) for a in emb.arrays]

                        bank = [grams(m) for m in prior.maps[n_hold:]]
                        hold = [grams(m) for m in prior.maps[:n_hold]]
                        nz = [grams(m) for m in noise.maps]

                        def dist(Gs: list[np.ndarray], layer: int) -> float:
                            """``q``-quantile of the CKA distances from one map's layer embedding to the bank."""
                            from ..analysis.cka import cka_debiased  # noqa: PLC0415

                            d = [1.0 - cka_debiased(Gs[layer], B[layer]) for B in bank]
                            return float(np.quantile(d, q))

                        n_lay = len(bank[0])
                        # The per-map distances are kept, not just their medians: the rank placement
                        # (2026-10-12) needs the reference *distributions*, and the ratio placement's
                        # floor/ceiling are still their medians, so both can be reported side by side.
                        floor_refs = [np.array([dist(h, s) for h in hold]) for s in range(n_lay)]
                        ceil_refs = [np.array([dist(z, s) for z in nz]) for s in range(n_lay)]
                        floors = [float(np.median(f)) for f in floor_refs]
                        ceilings = [float(np.median(c)) for c in ceil_refs]
                        out_rows = []
                        for ch in grid_channels:
                            ctx = context_for(ch, t_ctx, draw, base_seed=cfg.seed)
                            G_ch = [linear_gram(a) for a in embed(ctx.X, ctx.y, ch.X_pool, readout).arrays]
                            for s in range(n_lay):
                                d = dist(G_ch, s)
                                fl, ce = floors[s], ceilings[s]
                                gap = ce - fl
                                out_rows.append({
                                    "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg,
                                    "layer": int(layer_ids[0][s]), "context_t": int(t_ctx), "draw": draw, "d": d,
                                    "floor1": fl, "ceiling1": ce, "gap": gap,
                                    "gap_ok": bool(abs(gap) >= min_gap and gap > 0.0),
                                    "min_gap": min_gap, "p": place(d, fl, ce), "grid_id": grid,
                                    "prior_rejection_rate": prior.rejection_rate,
                                    "n_floor_refs": int(floor_refs[s].size),
                                    "n_ceiling_refs": int(ceil_refs[s].size),
                                    **P.rank_placement(d, floor_refs[s], ceil_refs[s]),
                                })
                        return {"rows": out_rows}, {}

                    ident = {"grid_id": grid, "channels": [cells.channel(ch) for ch in grid_channels],
                             "t": int(t_ctx), "draw": draw,
                             "stats_version": CKA_PLACEMENT_STATS_VERSION, **embed_ident(readout),
                             **{key: pl[key] for key in ("n_prior", "n_prior_holdout", "n_noise", "bank_quantile",
                                                          "min_gap", "n_dense", "holdout_frac", "rmse_threshold",
                                                          "prior_type")}}
                    hit = cells.run("b", ident, f"grid {grid} t={t_ctx} d={draw}", compute)
                    if hit is not None:
                        prow += hit[0]["rows"]
                    print(f"[mechanism] cka placement grid {grid} t={t_ctx} draw {draw} done "
                          f"({len(grid_channels)} channels)", flush=True)
        if prow:
            _write_csv(prow, out, "cka_placement.csv")
        with open(os.path.join(out, "gates.json"), "w", encoding="utf-8") as fh:
            json.dump(gates, fh, indent=2, default=_json_default)
        for g in gates:
            print(f"[mechanism] gate {g['gate']} ({g.get('readout', '')}): "
                  f"{'PASS' if g['passed'] else 'FAIL'}", flush=True)
        if p["shadow"] is not None and rows:
            verdict = shadow_criterion(pd.DataFrame(rows), p["shadow"])
            with open(os.path.join(out, "shadow_criterion.json"), "w", encoding="utf-8") as fh:
                json.dump(verdict, fh, indent=2, default=_json_default)
            print(f"[mechanism] shadow criterion ({verdict['readout']} vs {verdict['baseline']}): "
                  f"{'PASS' if verdict['passed'] else 'FAIL'} {verdict['ratios']}", flush=True)
    for path in figs.render_cka(out):
        print(f"[mechanism] wrote {path}")
    if not opts.replot:
        cells.report()
    return out


def shadow_criterion(df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    """B1 Step 5's pre-registered test: does a shadow readout narrow the between-channel band?

    Per target and context size, at stress level ``spec['level']``: average ``cka_minus_null`` over draws
    *within* a channel, take the interquartile range over channels at every block, divide the shadow
    readout's IQR by the baseline's, and take the median of that ratio over blocks. The criterion passes when
    the median ratio is below ``max_iqr_ratio`` for **every** listed target at **every** context size in the
    run (pre-registered 2026-10-06: ratio < 0.75, targets K_GT and K_GP).

    Args:
        df: Rows of ``cka.csv`` holding both readouts.
        spec: The resolved ``cka.shadow`` block.

    Returns:
        ``{readout, baseline, ratios: {"<target>@t=<t>": ratio}, max_iqr_ratio, passed}``.

    Raises:
        ValueError: If either readout is missing from ``df``.
    """
    for r in (spec["readout"], spec["baseline"]):
        if r not in set(df["readout"]):
            raise ValueError(f"shadow_criterion: readout {r!r} is not in the run (readouts: {sorted(set(df['readout']))}).")
    blocks = df[df["level"] == float(spec["level"])]
    ratios: dict[str, float | None] = {}
    for target in spec["targets"]:
        for t in sorted(blocks["context_t"].unique()):
            sub = blocks[(blocks["target"] == target) & (blocks["context_t"] == t)]
            iqr = {}
            for readout in (spec["readout"], spec["baseline"]):
                per_ch = (sub[sub["readout"] == readout]
                          .groupby(["subject", "emg", "layer"])["cka_minus_null"].mean().reset_index())
                g = per_ch.groupby("layer")["cka_minus_null"]
                iqr[readout] = g.quantile(0.75) - g.quantile(0.25)
            joined = pd.concat(iqr, axis=1).dropna()
            base = joined[spec["baseline"]]
            ratio = (joined[spec["readout"]] / base)[base > 0]
            ratios[f"{target}@t={int(t)}"] = float(ratio.median()) if len(ratio) else None
    passed = bool(ratios) and all(v is not None and v < float(spec["max_iqr_ratio"]) for v in ratios.values())
    return {"readout": spec["readout"], "baseline": spec["baseline"], "level": float(spec["level"]),
            "ratios": ratios, "max_iqr_ratio": float(spec["max_iqr_ratio"]), "passed": passed}


def _cka_ceiling(
    cfg: MechanismConfig,
    p: dict[str, Any],
    cells: _Cells,
    channels: list[ChannelData],
    knob: Any,
) -> list[dict[str, Any]]:
    """The reachable CKA ceiling per (channel, level, context size): what a perfect method could score.

    Debiased CKA between two Gram matrices with different spectra is bounded well below 1 even when one
    is a perfect monotone function of the other, so an absolute CKA against ``K_GP`` cannot be read
    without a ceiling, and the site-permutation null only supplies a floor. Two references are computed,
    both from closed-form GP fits (no transformer forward passes, so this is cheap):

    * ``noise_ceiling`` -- CKA between two GP posterior covariances fitted on the **same sites** with
      **independent trial draws**. They differ only by measurement noise, so this is the most any
      method could score against a target that is itself estimated from noisy data.
    * ``gp_vs_gt`` -- CKA between the GP's posterior covariance and the ground-truth response kernel:
      how well the comparator models the surface at all.

    Args:
        cfg: Resolved config.
        p: Resolved ``cka`` block.
        cells: Cell namespace.
        channels: Loaded channels.
        knob: Built stress knob.

    Returns:
        One row per (channel, level, context size, draw).
    """
    from ..analysis.cka import cka_debiased, target_kernels  # noqa: PLC0415
    from ..data.snr import achieved_snr_db  # noqa: PLC0415
    from ..models.gp.surrogates import GPSurrogate  # noqa: PLC0415

    offset = int(p["ceiling"]["trial_draw_offset"])
    rows: list[dict[str, Any]] = []
    for ch in channels:
        for level in knob.levels:
            st = knob.apply(ch, level, rng_for(ch.label, "cka", level, base_seed=cfg.seed))
            snr = achieved_snr_db(st)
            for t in cfg.context_sizes:
                if t >= st.n_sites:
                    continue
                for rep in range(int(p["n_context_draws"])):
                    ident = {**cells.channel(ch), "knob": knob.name, "level": float(level), "t": int(t),
                             "draw": rep, "gp_params": p["gp_params"], "trial_draw_offset": offset}

                    def compute(ch: ChannelData = ch, st: ChannelData = st, t: int = t, rep: int = rep,
                                level: float = level) -> tuple[dict[str, Any], dict[str, Any]]:
                        covs = []
                        for trial_draw in (rep, rep + offset):
                            ctx = context_for(st, int(t), rep, base_seed=cfg.seed, level=float(level),
                                              trial_draw=trial_draw)
                            gp = GPSurrogate(**p["gp_params"])
                            gp.fit(ctx.X, ctx.y)
                            covs.append(gp.posterior_covariance(st.X_pool))
                        k_gt = target_kernels(st.X_pool, st.y_gt)["K_GT"]
                        return {
                            "noise_ceiling": cka_debiased(covs[0], covs[1]),
                            "gp_vs_gt": cka_debiased(covs[0], k_gt),
                        }, {}

                    hit = cells.run("ceiling", ident, f"ceiling {ch.label} a={level:g} t={t} d={rep}", compute)
                    if hit is not None:
                        rows.append({
                            "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg,
                            "knob": knob.name, "level": float(level), "achieved_snr_db": snr,
                            "context_t": int(t), "rep": rep,
                            **{k: v for k, v in hit[0].items() if k != "host"},
                        })
    return rows


# ---------------------------------------------------------------------------
# procrustes (M14 revived, 2026-10-14): the spectrum-weighting complement of the CKA analysis
# ---------------------------------------------------------------------------
#: Readouts this analysis accepts. ``feature_tokens`` is deliberately excluded: Procrustes is a statistic of one
#: ``[n_sites, d]`` cloud, and a per-token variant would need a per-token null whose averaging rule is a separate
#: design question (CKA (a)'s shadow readout, B1 Step 5). The feature-token *mean* is the readout CKA (b) places,
#: so the two analyses compare the same clouds.
PROCRUSTES_READOUTS: tuple[str, ...] = ("feature_mean",)

PROCRUSTES_DEFAULTS: dict[str, Any] = {
    "layers": None,                      # None = all 18 blocks
    "readouts": list(PROCRUSTES_READOUTS),
    # The disparity baseline: the cloud every other context size is compared against. 2 is the smallest context
    # TabPFN accepts (analysis/context.context_sites refuses t < 2), so it is the least-informed geometry.
    # It must be IN the shared ladder, so the cell it needs exists (the loader's _subset_of_ladder rule).
    "baseline_t": 2,
    "n_context_draws": 3,
    # Shuffled-site null replicates (CAVEAT 2). Each costs one [r, r] SVD with r = min(n_sites - 1, d), so the
    # null runs ~0.8 ms per replicate on the 96-site NHP grid -- unlike CKA's null it cannot be collapsed into a
    # single contraction, which is why this is 200 while cka.n_perm is 1000. The headline statistic is the
    # DIFFERENCE disparity - null_mean (analysis/procrustes.HEADLINE_STATISTIC), which 200 replicates resolve far
    # better than the between-channel spread; only the secondary percentile readout is limited to 1/(n_perm + 1).
    "n_perm": 200,
    "inference_seed": 0,
    # Unstressed by default: this analysis asks how the geometry moves with EVIDENCE, and a stress level would
    # confound that with signal quality. The knob is still a config key, so a stressed sweep needs no code change.
    "stress": {"knob": "k2_channel", "levels": [1.0]},
    # Row subsampling. The legacy M14 code subsampled 300 rows to escape the n < d bias; the NHP grid has 96
    # sites, so there is nothing to subsample and ``n_rows: null`` (every site) is the only honest setting here.
    # Kept configurable for larger grids, and because a smaller n is the direct way to SHOW the bias: the same
    # cell at n_rows = 48 has a higher null. Indices are drawn once per cell and shared by every layer and every
    # context size, so the row correspondence Procrustes depends on survives.
    "subsample": {"n_rows": None},
    # Debiased linear CKA between the SAME two clouds, so a Procrustes panel and a CKA panel are read on
    # identical pairs (that comparison is the point of the analysis). Cheap: one Gram pair per (layer, t).
    "cka_companion": True,
    # Layers drawn in the disparity-vs-context figure (every layer is in the CSV; an 18-trace legend is
    # unreadable). Must be a subset of the computed layers.
    "figures": {"context_layers": [0, 1, 5, 11, 17]},
}


def _procrustes_rows(
    clouds: dict[int, Any],
    baseline_t: int,
    n_perm: int,
    rng: np.random.Generator,
    *,
    cka_companion: bool,
) -> list[dict[str, Any]]:
    """Disparity + shuffled-site null (and the CKA companion) for every (layer, context size) of one cell.

    Args:
        clouds: ``{context_t: LayerEmbeddings}`` of one channel, level, draw and readout; all share the layer set
            and the site rows, so row ``i`` is the same electrode everywhere.
        baseline_t: The context size whose cloud every other one is compared against.
        n_perm: Shuffled-site replicates per pair.
        rng: Generator for the null (keyed per cell by the caller).
        cka_companion: Also report debiased linear CKA between the same two clouds.

    Returns:
        Rows with ``layer``, ``context_t``, ``is_baseline`` and every field of
        :class:`~pfns4neurostim.analysis.procrustes.ProcrustesNull` (plus ``cka`` when requested).

    Raises:
        KeyError: If ``baseline_t`` was not embedded.
    """
    from ..analysis.cka import cka_debiased, linear_gram  # noqa: PLC0415
    from ..analysis.procrustes import procrustes_with_null  # noqa: PLC0415

    base = clouds[int(baseline_t)]
    rows: list[dict[str, Any]] = []
    for t in sorted(clouds):
        emb = clouds[t]
        for i, layer in enumerate(emb.layers):
            X = base.arrays[i]                                             # [n_sites, d]
            Y = emb.arrays[i]                                              # [n_sites, d]
            row: dict[str, Any] = {"layer": int(layer), "context_t": int(t),
                                   "is_baseline": int(t) == int(baseline_t)}
            row.update(procrustes_with_null(X, Y, n_perm=n_perm, rng=rng).as_row())
            if cka_companion:
                row["cka"] = cka_debiased(linear_gram(X), linear_gram(Y))
            rows.append(row)
    return rows


def run_procrustes(cfg: MechanismConfig, opts: RunOptions) -> str:
    """M14 revived: Procrustes disparity of the layer-wise site geometry along the context ladder.

    For one channel and one context draw, TabPFN is conditioned on ``t`` observations and all N grid sites are
    embedded at every block -- exactly the clouds :func:`run_cka` forms. The disparity between the cloud at
    ``baseline_t`` and the cloud at each ``t`` is then reported **against a shuffled-site null**, per layer, with
    debiased linear CKA on the same pair beside it.

    Two things the reporting must keep apart (they answer different questions):

    * the **t ladder** tests whether the geometric drift depends on how much evidence the network has;
    * the **shuffled-site null** tests whether a disparity number means anything at all at ``n < d``
      (96 rows in 192 dimensions). It is not a context-size control, and the ladder is not a null.

    Procrustes quotients out rotation, which the fixed, non-rotation-equivariant prediction head does not: these
    numbers are representation-similarity readouts to be read beside the output-space ones, never a claim about
    the model's prediction (Caveat 1, :mod:`pfns4neurostim.analysis.procrustes`).

    Args:
        cfg: Resolved config.
        opts: Cache policy.

    Returns:
        The run directory.
    """
    from ..analysis.embeddings import READOUTS, readout_embeddings  # noqa: PLC0415
    from ..analysis.procrustes import HEADLINE_STATISTIC  # noqa: PLC0415
    from ..data.snr import achieved_snr_db  # noqa: PLC0415
    from ..data.stress import build_knob  # noqa: PLC0415
    from ..models.pfn.tabpfn import FrozenTabPFN  # noqa: PLC0415
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _check_keys(cfg.params, PROCRUSTES_DEFAULTS, "procrustes")
    p["stress"] = _check_keys(p["stress"], PROCRUSTES_DEFAULTS["stress"], "procrustes.stress")
    p["subsample"] = _check_keys(p["subsample"], PROCRUSTES_DEFAULTS["subsample"], "procrustes.subsample")
    p["figures"] = _check_keys(p["figures"], PROCRUSTES_DEFAULTS["figures"], "procrustes.figures")
    bad = [r for r in p["readouts"] if r not in READOUTS]
    if bad:
        raise ValueError(f"procrustes.readouts: unknown {bad}; expected a subset of {READOUTS}.")
    unsupported = [r for r in p["readouts"] if r not in PROCRUSTES_READOUTS]
    if unsupported:
        raise ValueError(
            f"procrustes.readouts: {unsupported} is not supported here (only {list(PROCRUSTES_READOUTS)}); a "
            "per-token Procrustes would need a per-token null whose averaging rule is undefined."
        )
    _subset_of_ladder([p["baseline_t"]], cfg.context_sizes, "procrustes.baseline_t")
    if int(p["n_perm"]) < 1:
        raise ValueError(f"procrustes.n_perm must be >= 1, got {p['n_perm']}.")
    out = cfg.run_dir
    if not opts.replot:
        set_seed(cfg.seed)
        _write_config(cfg, p, out)
        cells = _Cells(cfg, opts)
        engine_box: dict[str, Any] = {}

        def engine() -> Any:
            if "e" not in engine_box:
                engine_box["e"] = FrozenTabPFN(device=cfg.device, random_state=int(p["inference_seed"]))
            return engine_box["e"]

        def embed(ctx: Any, X_sites: np.ndarray, readout: str) -> Any:
            eng = engine()
            eng.fit(ctx.X, ctx.y)
            return readout_embeddings(eng, X_sites, readouts=(readout,), layers=p["layers"])[readout]

        channels = _channels(cfg)
        knob = build_knob(p["stress"]["knob"], p["stress"]["levels"])
        rows: list[dict[str, Any]] = []
        for ch in channels:
            for level in knob.levels:
                st = knob.apply(ch, level, rng_for(ch.label, "procrustes", level, base_seed=cfg.seed))
                snr = achieved_snr_db(st)
                # t == n_sites is allowed here (CKA skips it): the fully-observed grid is the informed end of the
                # ladder, and context_sites accepts t up to N. It is NOT a fix for n < d -- the cloud's rows are
                # the readout sites, so no context size changes how many rows it has (that is the null's job).
                ladder = tuple(int(t) for t in cfg.context_sizes if t <= st.n_sites)
                if int(p["baseline_t"]) not in ladder:
                    raise ValueError(
                        f"procrustes: baseline_t={p['baseline_t']} exceeds {st.label}'s {st.n_sites} sites."
                    )
                for rep in range(int(p["n_context_draws"])):
                    for readout in p["readouts"]:
                        ident = {
                            **cells.channel(ch), "knob": knob.name, "level": float(level), "draw": rep,
                            "readout": readout, "layers": p["layers"], "ladder": list(ladder),
                            "baseline_t": int(p["baseline_t"]), "n_perm": int(p["n_perm"]),
                            "subsample": p["subsample"], "cka_companion": bool(p["cka_companion"]),
                            "inference_seed": int(p["inference_seed"]),
                        }

                        def compute(level: float = level, rep: int = rep, readout: str = readout,
                                    st: ChannelData = st, ladder: tuple[int, ...] = ladder,
                                    ) -> tuple[dict[str, Any], dict[str, Any]]:
                            # One row subsample per cell, shared by every layer and every context size, so the
                            # site-to-row correspondence holds across the whole cell.
                            n_rows = p["subsample"]["n_rows"]
                            sel: np.ndarray | None = None
                            if n_rows is not None:
                                if not 3 <= int(n_rows) <= st.n_sites:
                                    raise ValueError(
                                        f"procrustes.subsample.n_rows={n_rows} must be in [3, {st.n_sites}]."
                                    )
                                sub_rng = rng_for(st.label, "procrustes-subsample", level, rep, base_seed=cfg.seed)
                                sel = np.sort(sub_rng.choice(st.n_sites, size=int(n_rows), replace=False))
                            X_sites = st.X_pool if sel is None else st.X_pool[sel]    # [n_rows, D]
                            clouds: dict[int, Any] = {}
                            for t in ladder:
                                ctx = context_for(st, int(t), rep, base_seed=cfg.seed, level=float(level))
                                clouds[int(t)] = embed(ctx, X_sites, readout)
                            null_rng = rng_for(st.label, "procrustes-null", level, rep, readout, base_seed=cfg.seed)
                            return {"rows": _procrustes_rows(
                                clouds, int(p["baseline_t"]), int(p["n_perm"]), null_rng,
                                cka_companion=bool(p["cka_companion"]),
                            )}, {}

                        hit = cells.run("trajectory", ident, f"{ch.label} a={level:g} d={rep} {readout}", compute)
                        if hit is None:
                            continue
                        for r in hit[0]["rows"]:
                            rows.append({
                                "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg,
                                "model": MECHANISM_MODEL, "readout": readout, "knob": knob.name,
                                "level": float(level), "achieved_snr_db": snr, "rep": rep,
                                "baseline_t": int(p["baseline_t"]), **r,
                            })
            print(f"[mechanism] procrustes {ch.label} done", flush=True)
        _write_csv(rows, out, "procrustes.csv")
        with open(os.path.join(out, "headline.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "headline_statistic": HEADLINE_STATISTIC,
                "null": "shuffled-site (the row correspondence of one cloud is permuted)",
                "n_perm": int(p["n_perm"]),
                "rotation_quotient_caveat": (
                    "Procrustes quotients out rotation; the prediction head is not rotation-equivariant. "
                    "Representation-similarity readout only -- never a claim about the model's output."
                ),
            }, fh, indent=2)
        cells.report()
    for path in figs.render_procrustes(out):
        print(f"[mechanism] wrote {path}", flush=True)
    return out


#: Analysis runners by name.
ANALYSES: dict[str, Callable[[MechanismConfig, RunOptions], str]] = {
    "bayes_update": run_bayes_update,
    "placement": run_placement,
    "cka": run_cka,
    "procrustes": run_procrustes,
}


def run_mechanism(
    config_path: str,
    overrides: list[str] | None = None,
    *,
    replot: bool = False,
    use_cache: bool = True,
    only_cached: bool = False,
) -> str:
    """Load a mechanism config and run its analysis.

    Args:
        config_path: YAML path.
        overrides: ``key=value`` overrides.
        replot: Rebuild figures from existing CSVs.
        use_cache: Read and write the cell cache (``--no-cache`` turns it off).
        only_cached: Assemble from cached cells only (``--only-cached``).

    Returns:
        The run directory.
    """
    cfg = load_mechanism_config(config_path, overrides)
    return ANALYSES[cfg.analysis](cfg, RunOptions(replot=replot, use_cache=use_cache, only_cached=only_cached))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m pfns4neurostim mechanism``.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="pfns4neurostim mechanism", description="Hyp C mechanism analyses.")
    parser.add_argument("--config", required=True, help="Path to a mechanism YAML.")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None, metavar="KEY=VALUE")
    parser.add_argument("--replot", action="store_true", help="Rebuild figures from the CSVs.")
    parser.add_argument("--no-cache", action="store_true", help="Neither read nor write the mechanism cells.")
    parser.add_argument("--only-cached", action="store_true",
                        help="Assemble the CSVs from cached cells only; compute nothing.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_mechanism(args.config, args.overrides, replot=args.replot, use_cache=not args.no_cache,
                  only_cached=args.only_cached)
    return 0
