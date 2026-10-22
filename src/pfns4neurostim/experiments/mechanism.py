"""Hyp C mechanism analyses: Bayesian update (OMEGA), placement (MMD/W2), CKA (b) placement, representation (BETA).

One CLI, one config family: the ``analysis`` key picks the runner and a same-named block
holds its parameters (unknown keys raise, as everywhere else)::

    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_bayes_update_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_placement_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_cka_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_cka_nhp.yaml --replot
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_cka_nhp.yaml --only-cached
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_representation_nhp.yaml

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
cka                   ``b`` (grid, t, draw; feature mean): the embedding placement. CKA (a) (kind ``a``), its
                      controls and the ``ceiling`` cells were removed on 2026-10-21 (BETA §7); their cached cells
                      stay on disk, unread
placement             ``B`` (channel); ``C`` (channel: every level and context size); ``gate`` (M0 ladder)
representation        ``cell`` (channel, level, t, draw: β2 alignment of the label token / feature mean / lens with
                      belief targets, the smooth-field surrogate, ceilings, β1a site-matched W2 and the lens rows);
                      ``posterior_cov`` (channel, level, t, draw on its coverage: β1b, C_PFN vs the references)
====================  ===================================================================================

The Procrustes trajectory (``procrustes``, nested under ``procrustes/cka``) was folded into ``representation`` on
2026-10-21 (BETA D5): NBS and the shape distance are computed there on Gram matrices.

**Artifacts** (same store, ``kind`` in the identity): ``bank`` / ``noise_bank`` per grid. Embeddings are
deliberately *not* cached: at 18 blocks x
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

#: Output sub-path of an analysis under ``{output_root}/mechanism/``, when it is not the analysis name. Empty since
#: the Procrustes analysis (nested under ``procrustes/cka``) was folded into ``representation`` (BETA D5,
#: 2026-10-21); kept as the one place a nested layout would be declared (``scripts/replot_all.py`` reads it).
ANALYSIS_SUBDIR: dict[str, str] = {}

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
    # reference: the ladder-anchor engine (user, 2026-10-21): y' = mu_ref(x') + c s_ref(x'), and the surprise response
    # (Omega-4, C1) is read in its predictive SD. One of ``engines``, never a TabPFN engine; fitted (fit + base only)
    # on every cell where Part B runs, whether or not coverage gives it Part A or Part B there.
    "shared": {"n_trials": 8, "ladder": [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0], "saturation_c": [1.0, 3.0],
               "reference": "gp_bayes"},
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
    # exemplar_c: the shared ladder multiplier of the change maps at the common anchor (y' = mu_ref + c s_ref, the
    # ladder reference of ``shared.reference``).
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
             "references": "all", "engine_contrasts": False, "pool": "unobserved", "context_ts": None,
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

#: Pre-declared headline metrics (OMEGA §3.6): Part A per-anchor, Part A per-cell, Part B (TabPFN vs each reference).
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
    # The ladder reference (user, 2026-10-21): any non-TabPFN engine of the run -- GP-Bayes is optional.
    ref = p["shared"]["reference"]
    if ref not in p["engines"]:
        raise ValueError(f"bayes_update.shared.reference {ref!r} is not in bayes_update.engines {p['engines']}: "
                         f"the y' ladder is anchored on an engine of the run.")
    if ref in TABPFN_ENGINES:
        raise ValueError(f"bayes_update.shared.reference {ref!r} is a TabPFN engine; the ladder anchor must be "
                         f"model-neutral (a GP engine).")
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
    bad = [e for e in p["exemplar"]["engines"] if e not in p["engines"]]
    if p["exemplar"]["enabled"] and bad:
        raise ValueError(f"bayes_update.exemplar.engines: {bad} are not in bayes_update.engines {p['engines']}.")
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
        name: A key of :data:`TABPFN_ENGINES`, ``'gp_bayes'``, ``'gp_mll_refit'``, ``'gp_mll_frozen'`` or
            ``'gp_fixed'``.
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

    Whenever Part B runs, the ladder reference ``p['shared']['reference']`` is fitted and its base predictive anchors
    the shared ``y'`` ladder and the surprise unit; it is probed (Part A / Part B) only if it is in ``engines_a`` /
    ``engines_b``.

    Args:
        stressed: The channel at this stress level.
        label: Channel label (seed streams).
        level: Stress level.
        t: Context size.
        draw: Context draw.
        p: Resolved ``bayes_update`` block.
        cfg: Resolved config.
        engines_a: Engines that run Part A here.
        engines_b: Engines that run Part B here.
        make: ``name -> unfitted engine``.

    Returns:
        ``coherence`` (per anchor), ``coherence_cell`` (per pool), ``headtohead``, ``surprise`` and ``profile`` rows
        (the transfer-vs-distance profile of Hypothesis C1), the
        ``gp_bayes`` diagnostics (when it runs Part A), the host and the wall time (guardrail G5).
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
    sh = p["shared"]
    ref = str(sh["reference"])                                                     # the ladder anchor
    names = list(dict.fromkeys([*engines_a, *engines_b]))                          # every engine probed, once
    fitted = list(dict.fromkeys([*names, *([ref] if engines_b else [])]))          # + the ladder anchor (base only)
    engines = {n: make(n) for n in fitted}
    bases = {}
    for n, eng in engines.items():
        eng.fit(ctx.X, ctx.y)
        bases[n] = eng.base(Q)
    shared = None
    if engines_b:
        shared = B.shared_values(held, anchors, bases[ref], int(sh["n_trials"]), sh["ladder"],
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
                ladder_c=shared["c"], s_ref_anchor=float(bases[ref].sd[0, a]), heldout=held,
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
    # The gates validate GP-Bayes itself, so G4's ladder is always anchored on it whatever ``shared.reference`` the
    # grid uses; the reference is therefore not part of a gate identity.
    gate_shared = {k: v for k, v in p["shared"].items() if k != "reference"}
    common = {"knob": knob.name, "level": level, "draw": draw, "anchors": p["anchors"], "eps": p["eps"],
              "quadrature": p["quadrature"], "y_grid": p["y_grid"], "shared": gate_shared}
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
                        pp["shared"]["reference"] = "gp_bayes"
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
                            # the ladder reference is fitted wherever Part B runs (shared.reference), never added
                            # to Part A or Part B; its settings enter the identity through engine_params
                            fitted = set(cov["A"]) | set(cov["B"]) | ({p["shared"]["reference"]} if cov["B"] else set())
                            keys = {"dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg, "knob": knob.name,
                                    "level": float(level), "achieved_snr_db": snr, "context_t": int(t), "draw": draw}
                            ident = {**cells.channel(ch), "knob": knob.name, "level": float(level), "t": int(t),
                                     "draw": draw, "engines_a": cov["A"], "engines_b": cov["B"],
                                     "engine_params": {e: _engine_params(e, p, draw) for e in fitted},
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
                ladder_ref = str(p["shared"]["reference"])
                dmaps = {}
                for t in (ts if c1["reference"] in p["engines"] else []):
                    def c1_cell(t: int = t) -> tuple[dict[str, Any], dict[str, Any]]:
                        anchor_eng = _make_engine(ladder_ref, p, cfg.device, int(ex["draw"]))
                        anchor_eng.fit(ctxs[t].X, ctxs[t].y)
                        bb = anchor_eng.base(st.X_pool)
                        y = float(bb.mean[0, a] + float(c1["exemplar_c"]) * bb.sd[0, a])  # y' = mu_ref + c s_ref
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
                                                    "ladder_reference": ladder_ref,
                                                    "engine_params": {n: _engine_params(n, p, int(ex["draw"]))
                                                                      for n in [*c1["engines"], ladder_ref]}},
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
# cka (task #5; B1 2026-10-06): CKA (b) embedding placement only since BETA (2026-10-21)
# ---------------------------------------------------------------------------
#: The readout CKA (b) places. Placement asks where the network's representation of the electrode geometry sits
#: between the prior and noise references.
CKA_PLACEMENT_READOUT: str = "feature_mean"

#: Statistic set the placement (``b``) cell payload carries. A ``b`` cell caches its finished rows, so an older
#: cell cannot serve a column it never computed; this stamp is part of the cell identity, so bumping it supersedes
#: the previous generation without discarding it. Bumped to 2 on 2026-10-14, when ``rank_vs_floor`` /
#: ``rank_vs_ceiling`` / ``p_rank`` and the reference counts joined the rows.
CKA_PLACEMENT_STATS_VERSION: int = 2

#: The ``cka`` block. CKA (a) -- the feature tokens against K_X / K_GT / K_GP with site-permutation nulls, its M0
#: controls, the per-token shadow criterion and the K_GP ceiling -- was removed on 2026-10-21 (BETA §7, D5): the
#: ``representation`` analysis replaces it. ``layers`` and ``inference_seed`` stay because they enter every CKA (b)
#: cell identity (``embed_ident``), so the cells computed for the subject-2 backfill keep serving.
CKA_DEFAULTS: dict[str, Any] = {
    "layers": None,
    "inference_seed": 0,
    # CKA (b): placement of a channel's layer-wise feature-mean embedding between the on-grid prior bank and the noise
    # bank, at every context size of the shared ladder (B3 Step 4 replaced the old ``ts`` key).
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
}


def run_cka(cfg: MechanismConfig, opts: RunOptions) -> str:
    """CKA (b): placement of the feature-token mean between the on-grid prior bank and the noise bank.

    One cell per (grid, t, draw) over every channel of the grid; the channel-independent work (banks, the bank maps'
    embeddings at the context sites, hence the floor and ceiling) is done once per cell. The context sites come from
    a grid-keyed stream, so the comparison is paired.

    Args:
        cfg: Resolved config.
        opts: Cache policy.

    Returns:
        The run directory.
    """
    from ..analysis.cka import linear_gram  # noqa: PLC0415
    from ..analysis.embeddings import readout_embeddings  # noqa: PLC0415
    from ..analysis import placement as P  # noqa: PLC0415
    from ..analysis.placement import placement as place  # noqa: PLC0415
    from ..models.pfn.tabpfn import FrozenTabPFN  # noqa: PLC0415
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _check_keys(cfg.params, CKA_DEFAULTS, "cka")
    p["placement"] = _check_keys(p["placement"], CKA_DEFAULTS["placement"], "cka.placement")
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

        def embed(X_ctx: np.ndarray, y_ctx: np.ndarray, X_sites: np.ndarray, readout: str) -> Any:
            eng = engine()
            eng.fit(X_ctx, y_ctx)
            return readout_embeddings(eng, X_sites, readouts=(readout,), layers=p["layers"])[readout]

        def embed_ident(readout: str) -> dict[str, Any]:
            return {"readout": readout, "layers": p["layers"], "inference_seed": int(p["inference_seed"])}

        channels = _channels(cfg)
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
    for path in figs.render_cka(out):
        print(f"[mechanism] wrote {path}")
    if not opts.replot:
        cells.report()
    return out


# ---------------------------------------------------------------------------
# representation (OBJECTIVE BETA, 2026-10-21): label-token belief alignment (β2), site-matched predictives (β1a),
# posterior-covariance alignment (β1b). Replaces CKA (a) and the Procrustes trajectory (BETA §7, D5).
# ---------------------------------------------------------------------------
#: Readouts of the representation analysis: the label token per block (the only token the decoder reads; reportable
#: against belief targets, G6 / D1), the feature-token mean (a diagnostic line), and the zero-parameter lens's
#: quantile embedding per block (output space).
REPRESENTATION_READOUTS: tuple[str, ...] = ("label", "feature_mean", "lens")

#: Statistic set a ``cell`` payload carries; in the identity, so a change supersedes old cells without deleting them.
REPRESENTATION_STATS_VERSION: int = 1

#: GP engines whose predictive given D can be a belief (BETA D3e: GP-MLL refit and GP-fixed; GP-Bayes optional).
REPRESENTATION_GP_ENGINES: tuple[str, ...] = ("gp_mll_refit", "gp_fixed", "gp_bayes")

REPRESENTATION_DEFAULTS: dict[str, Any] = {
    "stress": {"knob": "k2_channel", "levels": [1.0, 2.0, 4.0]},
    "n_context_draws": 2,
    "layers": None,                                    # null = all 18 blocks
    "readouts": list(REPRESENTATION_READOUTS),
    # Targets <kind>_<code>: kind in mean | sd | quant | cov; code in own (TabPFN's final predictive) | mll (GP-MLL
    # refit) | fixed (GP-fixed) | bayes (GP-Bayes, optional); plus K_X and K_GT (analysis/representation.py).
    "targets": ["mean_own", "sd_own", "mean_mll", "sd_mll", "cov_mll", "quant_own", "quant_mll", "K_X", "K_GT"],
    "belief_engines": ["gp_mll_refit", "gp_fixed"],
    "layout_confound": "gp_fixed",                     # its posterior covariance: the extra confound of cov targets
    "n_quantiles": 32,
    "site_sets": ["all"],                              # all | unobserved (demoted pitfall; never at t = 80)
    # The seed rule of every Hyp C analysis: the inference seed of a cell is inference_seed + its context draw.
    "tabpfn": {"inference_seed": 0, "softmax_temperature": 0.9, "precision": "fp32", "max_batch_tokens": 400000},
    "gp_params": {"n_opt_steps": 50, "lr": 1.0, "optimizer": "lbfgs", "dtype": "float64", "noise_floor": 1.0e-3,
                  "n_restarts": 5, "lengthscale_floor": 0.05},
    "fixed_gp_params": {"lengthscale": 0.6931471805599453, "outputscale": 1.0, "noise": 0.05},
    "gp_bayes": {"lengthscale": [0.05, 2.0, 23], "outputscale": [0.05, 5.0, 11], "noise": [1.0e-3, 1.0, 13],
                 "b2": 1.0, "prune_mass": 1.0e-10, "chunk": 2048, "exact_max_components": 64, "sd_bins": 32},
    "deconfound": {"bandwidth_multipliers": [0.25, 0.5, 1.0, 2.0], "min_residual_frac": 1.0e-10},
    "surrogate": {"lengthscales": [0.10, 0.15, 0.30], "n_fields": 192, "n_replicates": 20, "jitter": 1.0e-8,
                  "quantile": 0.95},
    "permutation": {"n_perm": 200},                    # descriptive only (D4); 0 = off
    # The reachable ceiling (user, 2026-10-12, generalised in BETA §5): the same target built from an independent
    # trial draw at the same sites; trial_draw_offset re-keys the trial stream only (analysis/context.context_for).
    "ceiling": {"enabled": True, "trial_draw_offset": 100},
    # Belief ceilings (D2 with D3e): pairs (target_a, target_b) scored per cell; never a pair of two models' beliefs
    # (BETA §4: models are compared site by site, not through their across-site Grams).
    "target_relations": [["quant_mll", "K_GT"], ["mean_mll", "K_GT"]],
    "site_match": {"references": ["gp_mll_refit", "gp_fixed"]},
    # OPEN (research_design §4, 2026-10-19): the provisional headline and its alternative (b) are selectable here;
    # both are computable from representation.csv whatever is selected.
    "headline": {"statistic": "deconfounded_vs_surrogate", "readout": "label"},
    "inference": {"n_boot": 2000, "ci": 0.95},
    # β1b: C_PFN from Part A's implied covariance with every site as an anchor (OMEGA A3, refit engine), against the
    # references' latent covariances. Coverage filters as in bayes_update.coverage (context_ts null = the ladder).
    "posterior_cov": {
        "enabled": False, "engine": "tabpfn_v2_5", "references": ["gp_mll_refit", "gp_fixed"],
        "coverage": {"levels": [1.0], "context_ts": None, "draws": [0], "channels_per_subject": 2},
        "quadrature": {"method": "rqmc", "K": 8, "u_min": 1.0e-12, "central_mass": 1.0, "n_scrambles": 8},
        "eps": 0.01, "y_grid": {"points": 512, "u_tail": 1.0e-4}, "moment_matched": True,
    },
    "figures": {"level": None},                        # null = the nominal (lowest) stress level
}

#: Engines β1b can build C_PFN for (a GP engine reproduces its own closed form: the exactness check of the tests).
POSTERIOR_COV_ENGINES: tuple[str, ...] = (*TABPFN_ENGINES, *REPRESENTATION_GP_ENGINES)


def _validate_representation(p: dict[str, Any], ladder: Sequence[int]) -> dict[str, Any]:
    """Resolve the ``representation`` block over its defaults, refusing unknown keys and impossible settings.

    Args:
        p: User block.
        ladder: ``context.sizes``.

    Returns:
        The resolved parameters.

    Raises:
        ValueError: On unknown keys, readouts, targets, engines or statistics, or a target whose belief is not built.
    """
    from ..analysis import representation as R  # noqa: PLC0415

    d = REPRESENTATION_DEFAULTS
    p = _check_keys(p, d, "representation")
    for sub in ("stress", "tabpfn", "fixed_gp_params", "gp_bayes", "deconfound", "surrogate", "permutation",
                "ceiling", "site_match", "headline", "inference", "posterior_cov", "figures"):
        p[sub] = _check_keys(p[sub] or {}, d[sub], f"representation.{sub}")
    pc = p["posterior_cov"]
    for sub in ("coverage", "quadrature", "y_grid"):
        pc[sub] = _check_keys(pc[sub] or {}, d["posterior_cov"][sub], f"representation.posterior_cov.{sub}")
    bad = [r for r in p["readouts"] if r not in REPRESENTATION_READOUTS]
    if bad or not p["readouts"]:
        raise ValueError(f"representation.readouts: unknown {bad}; known {list(REPRESENTATION_READOUTS)}.")
    bad = [e for e in p["belief_engines"] if e not in REPRESENTATION_GP_ENGINES]
    if bad:
        raise ValueError(f"representation.belief_engines: unknown {bad}; known {list(REPRESENTATION_GP_ENGINES)}.")
    codes = {"own"} | {c for c, e in R.BELIEF_CODES.items() if e in p["belief_engines"]}
    for name in p["targets"]:
        _, code = R.parse_target(name)
        if code is not None and code not in codes:
            raise ValueError(f"representation.targets: {name!r} needs {R.BELIEF_CODES[code]!r} in "
                             f"representation.belief_engines {p['belief_engines']}.")
    if p["layout_confound"] not in p["belief_engines"]:
        raise ValueError(f"representation.layout_confound {p['layout_confound']!r} must be one of "
                         f"representation.belief_engines {p['belief_engines']} (its covariance is the confound).")
    bad = [r for r in p["site_match"]["references"] if r not in p["belief_engines"]]
    if bad:
        raise ValueError(f"representation.site_match.references {bad} are not in representation.belief_engines.")
    for pair in p["target_relations"]:
        if len(pair) != 2 or any(n not in p["targets"] for n in pair):
            raise ValueError(f"representation.target_relations: {pair} must name two of representation.targets.")
        codes_pair = {R.parse_target(n)[1] for n in pair} - {None}
        if len(codes_pair) > 1:
            raise ValueError(f"representation.target_relations: {pair} compares two models' beliefs through their "
                             f"across-site Grams; models are compared site by site (BETA §4, D3).")
    bad = [s for s in p["site_sets"] if s not in R.SITE_SETS]
    if bad or not p["site_sets"]:
        raise ValueError(f"representation.site_sets: unknown {bad}; known {list(R.SITE_SETS)}.")
    if p["headline"]["statistic"] not in R.HEADLINE_STATISTICS:
        raise ValueError(f"representation.headline.statistic must be one of {R.HEADLINE_STATISTICS}, got "
                         f"{p['headline']['statistic']!r}.")
    if p["headline"]["readout"] not in p["readouts"]:
        raise ValueError(f"representation.headline.readout {p['headline']['readout']!r} is not a computed readout.")
    if int(p["n_quantiles"]) < 2:
        raise ValueError(f"representation.n_quantiles must be >= 2, got {p['n_quantiles']}.")
    s = p["surrogate"]
    if not s["lengthscales"] or int(s["n_replicates"]) < 2 or int(s["n_fields"]) < 1 or not 0 < float(s["quantile"]) < 1:
        raise ValueError(f"representation.surrogate: need lengthscales, n_replicates >= 2, n_fields >= 1 and "
                         f"0 < quantile < 1, got {s}.")
    if int(p["inference"]["n_boot"]) < 1 or not 0 < float(p["inference"]["ci"]) < 1:
        raise ValueError(f"representation.inference: need n_boot >= 1 and 0 < ci < 1, got {p['inference']}.")
    if pc["enabled"]:
        if pc["engine"] not in POSTERIOR_COV_ENGINES:
            raise ValueError(f"representation.posterior_cov.engine must be one of {POSTERIOR_COV_ENGINES}.")
        bad = [r for r in pc["references"] if r not in p["belief_engines"]]
        if bad:
            raise ValueError(f"representation.posterior_cov.references {bad} are not in representation.belief_engines.")
        if pc["quadrature"]["method"] not in B_QUADRATURE_METHODS:
            raise ValueError(f"representation.posterior_cov.quadrature.method must be one of {B_QUADRATURE_METHODS}.")
        if pc["coverage"]["context_ts"] is None:
            pc["coverage"]["context_ts"] = [int(t) for t in ladder]
        _subset_of_ladder(pc["coverage"]["context_ts"], ladder, "representation.posterior_cov.coverage.context_ts")
    return p


def _representation_gp(name: str, p: dict[str, Any]) -> Any:
    """An unfitted GP belief engine of the representation analysis (same classes as ``bayes_update``)."""
    from ..analysis import bayes_update as B  # noqa: PLC0415

    if name == "gp_mll_refit":
        return B.GPRefitEngine(p["gp_params"], name=name)
    if name == "gp_fixed":
        return B.GPFixedEngine(**p["fixed_gp_params"], name=name)
    g = p["gp_bayes"]
    return B.GPBayesEngine(g["lengthscale"], g["outputscale"], g["noise"], b2=float(g["b2"]),
                           prune_mass=float(g["prune_mass"]), chunk=int(g["chunk"]), name=name,
                           exact_max_components=int(g["exact_max_components"]), sd_bins=int(g["sd_bins"]))


def _representation_gp_params(name: str, p: dict[str, Any]) -> dict[str, Any]:
    """The settings of one GP belief engine that enter a cell identity."""
    if name == "gp_mll_refit":
        return {"gp_params": p["gp_params"]}
    if name == "gp_fixed":
        return {"fixed_gp_params": p["fixed_gp_params"]}
    return {k: v for k, v in p["gp_bayes"].items() if k != "chunk"}


def _representation_beliefs(
    ctx: Any,
    Q: np.ndarray,
    p: dict[str, Any],
    tabpfn: Any,
    u: np.ndarray,
) -> tuple[dict[str, Any], Any, Any]:
    """Fit TabPFN (one hooked pass) and every GP belief engine on one context.

    Args:
        ctx: The context.
        Q: Readout sites, shape [M, D].
        p: Resolved ``representation`` block.
        tabpfn: A :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN` (refitted here).
        u: Quantile levels, shape [J].

    Returns:
        ``(beliefs, states, lens)``: belief per code (:class:`analysis.representation.Belief`), the block states of
        the pass, and ``{'quantiles' [L, M, J], 'means' [L, M]}`` of the zero-parameter lens.

    Raises:
        RuntimeError: On non-finite TabPFN quantiles or lens moments.
    """
    from ..analysis import bayes_update as B  # noqa: PLC0415
    from ..analysis import representation as R  # noqa: PLC0415
    from ..analysis.embeddings import capture_states_and_predictive, lens_log_probs  # noqa: PLC0415

    tabpfn.fit(ctx.X, ctx.y)
    states, lp = capture_states_and_predictive(tabpfn, Q, layers=p["layers"])     # lp [1, M, bars]
    mean, sd = tabpfn.moments(lp)                                                   # [1, M] x2
    E = np.asarray(tabpfn.ppf(lp, u), dtype=np.float64)[0]                          # [M, J]
    lens_lp = lens_log_probs(tabpfn, states.label_token)                            # [L, M, bars]
    lens_mean, _ = tabpfn.moments(lens_lp)                                          # [L, M]
    lens_q = np.asarray(tabpfn.ppf(lens_lp, u), dtype=np.float64)                   # [L, M, J]
    for name, arr in (("final quantiles", E), ("lens quantiles", lens_q), ("lens means", lens_mean)):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"representation: non-finite TabPFN {name}.")
    beliefs: dict[str, Any] = {"own": R.Belief(mean[0], sd[0], E, None)}
    codes = {e: c for c, e in R.BELIEF_CODES.items()}
    for name in p["belief_engines"]:
        g = _representation_gp(name, p)
        g.fit(ctx.X, ctx.y)
        base = g.base(Q)
        beliefs[codes[name]] = R.Belief(base.mean[0], base.sd[0], B.belief_embedding(base, u),
                                        B.latent_covariance_of(g, Q))
    return beliefs, states, {"quantiles": lens_q, "means": lens_mean}


def _representation_cell(
    st: ChannelData,
    level: float,
    t: int,
    draw: int,
    p: dict[str, Any],
    cfg: MechanismConfig,
    tabpfn: Any,
    surrogate_grams: Callable[[float], list[np.ndarray]],
) -> dict[str, Any]:
    """One (channel, level, t, draw) cell of BETA: β2 alignment rows, the surrogate floor, ceilings, β1a, the lens.

    Args:
        st: The channel at this stress level.
        level: Stress level.
        t: Context size.
        draw: Context draw.
        p: Resolved ``representation`` block.
        cfg: Resolved config.
        tabpfn: The cell's ``FrozenTabPFN`` (inference seed ``tabpfn.inference_seed + draw``).
        surrogate_grams: ``lengthscale -> [surrogate Gram over every site of the grid]`` (channel-independent).

    Returns:
        ``alignment``, ``surrogate``, ``ceiling``, ``relations``, ``site_match`` and ``lens`` rows, diagnostics, the
        host and the wall time (guardrail G5).
    """
    import socket  # noqa: PLC0415

    from ..analysis import bayes_update as B  # noqa: PLC0415
    from ..analysis import cka as C  # noqa: PLC0415
    from ..analysis import representation as R  # noqa: PLC0415

    t0 = time.time()
    Q = st.X_pool
    M = st.n_sites
    u = B.belief_levels(int(p["n_quantiles"]))
    ctx = context_for(st, int(t), int(draw), base_seed=cfg.seed, level=float(level))
    observed = np.isin(np.arange(M), ctx.sites)                                     # [M]
    beliefs, states, lens = _representation_beliefs(ctx, Q, p, tabpfn, u)
    codes = {e: c for c, e in R.BELIEF_CODES.items()}
    layout_code = codes[p["layout_confound"]]
    layout = beliefs[layout_code].cov
    targets = R.target_grams(p["targets"], beliefs, Q, st.y_gt)
    reps: dict[tuple[str, int], np.ndarray] = {}
    for i, layer in enumerate(states.layers):
        if "label" in p["readouts"]:
            reps[("label", int(layer))] = C.linear_gram(states.label_token[i])                    # [M, M]
        if "feature_mean" in p["readouts"]:
            reps[("feature_mean", int(layer))] = C.linear_gram(states.feature_tokens[i].mean(axis=1))
        if "lens" in p["readouts"]:
            reps[("lens", int(layer))] = C.centered_linear_gram(lens["quantiles"][i])
    geometry = C.geometry_confounds(Q, p["deconfound"]["bandwidth_multipliers"])
    min_res = float(p["deconfound"]["min_residual_frac"])
    alt_targets: dict[str, np.ndarray] = {}
    if p["ceiling"]["enabled"]:
        alt_ctx = context_for(st, int(t), int(draw), base_seed=cfg.seed, level=float(level),
                              trial_draw=int(draw) + int(p["ceiling"]["trial_draw_offset"]))
        alt_beliefs, _, _ = _representation_beliefs(alt_ctx, Q, p, tabpfn, u)
        alt_targets = R.target_grams(p["targets"], alt_beliefs, Q, st.y_gt)
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in ("alignment", "surrogate", "ceiling", "relations")}
    sur_full = {float(ell): surrogate_grams(float(ell)) for ell in p["surrogate"]["lengthscales"]}
    for site_set in p["site_sets"]:
        idx = np.arange(M) if site_set == "all" else np.flatnonzero(~observed)
        if idx.size < 4:
            continue

        def sub(G: np.ndarray, idx: np.ndarray = idx) -> np.ndarray:
            return G[np.ix_(idx, idx)]

        dec_cache: dict[bool, C.Deconfounder] = {}
        decs: dict[str, C.Deconfounder] = {}
        for name in p["targets"]:
            confs = R.target_confounds(name, geometry, layout, layout_code)
            with_layout = len(confs) > len(geometry)
            if with_layout not in dec_cache:
                dec_cache[with_layout] = C.Deconfounder([sub(c) for c in confs])
            decs[name] = dec_cache[with_layout]
        tg = {n: sub(G) for n, G in targets.items()}
        rows = R.alignment_rows({k: sub(G) for k, G in reps.items()}, tg, decs,
                                n_perm=int(p["permutation"]["n_perm"]),
                                rng=rng_for(st.label, "representation-perm", level, t, draw, site_set,
                                            base_seed=cfg.seed),
                                min_residual_frac=min_res)
        sur = R.surrogate_rows({ell: [sub(G) for G in gs] for ell, gs in sur_full.items()}, tg, decs,
                               quantile=float(p["surrogate"]["quantile"]), min_residual_frac=min_res)
        floors = R.surrogate_floors(sur)
        out["alignment"] += [{"site_set": site_set, **r, **floors.get(r["target"], {})} for r in rows]
        out["surrogate"] += [{"site_set": site_set, **r} for r in sur]
        out["relations"] += [{"site_set": site_set, **r} for r in R.pair_rows(
            [tuple(x) for x in p["target_relations"]], tg, tg, decs, min_residual_frac=min_res,
            label="belief_ceiling")]
        if alt_targets:
            beliefs_only = [n for n in p["targets"] if R.parse_target(n)[1] is not None]
            out["ceiling"] += [{"site_set": site_set, **r} for r in R.pair_rows(
                [(n, n) for n in beliefs_only], tg, {n: sub(G) for n, G in alt_targets.items()}, decs,
                min_residual_frac=min_res, label="trial_ceiling")]
    refs = [codes[r] for r in p["site_match"]["references"]]
    last = len(states.layers) - 1
    diagnostics = {"n_observed": int(observed.sum())}
    if int(states.layers[last]) == len(tabpfn._model.transformer_encoder.layers) - 1:  # noqa: SLF001
        # The lens at the last block IS the final predictive (same pass, same head): a built-in check.
        diagnostics["lens_final_max_abs_mean"] = float(np.abs(lens["means"][last] - beliefs["own"].mean).max())
    return {
        **out,
        "site_match": R.site_match_rows(beliefs["own"].quantiles, beliefs, refs, observed),
        "lens": R.lens_rows(lens["quantiles"], lens["means"], states.layers, beliefs, refs, observed),
        "diagnostics": diagnostics,
        "host": socket.gethostname(),
        "device": cfg.device,
        "seconds": time.time() - t0,
    }


def _posterior_cov_cell(
    st: ChannelData,
    level: float,
    t: int,
    draw: int,
    p: dict[str, Any],
    cfg: MechanismConfig,
) -> dict[str, Any]:
    """β1b at one (channel, level, t, draw): C_PFN with every site as an anchor against each reference's covariance.

    Statistics per reference: debiased CKA and NBS / shape distance of the double-centred matrices (shape), the
    deconfounded CKA (geometry, plus GP-fixed's covariance for any other reference, D3d), the Bures distance of the
    uncentred matrices (= W2 between the zero-mean Gaussian posteriors on the grid) and its ratio to
    ``sqrt(tr Sigma_ref)``, and the trace ratio ``tr C_PFN / tr Sigma_ref`` (overall latent uncertainty).

    Args:
        st: The channel at this stress level.
        level: Stress level.
        t: Context size.
        draw: Context draw.
        p: Resolved ``representation`` block.
        cfg: Resolved config.

    Returns:
        ``rows`` (one per reference), C_PFN's ``asymmetry`` and ``clipped_mass``, the host and the wall time.
    """
    import socket  # noqa: PLC0415

    from ..analysis import bayes_update as B  # noqa: PLC0415
    from ..analysis import cka as C  # noqa: PLC0415
    from ..analysis import procrustes as P  # noqa: PLC0415

    t0 = time.time()
    pc = p["posterior_cov"]
    Q = st.X_pool
    M = st.n_sites
    ctx = context_for(st, int(t), int(draw), base_seed=cfg.seed, level=float(level))
    name = pc["engine"]
    if name in TABPFN_ENGINES:
        tp = {**p["tabpfn"], **TABPFN_ENGINES[name]}
        eng: Any = B.PFNRefitEngine(device=cfg.device, random_state=int(tp["inference_seed"]) + int(draw),
                                    softmax_temperature=float(tp["softmax_temperature"]), name=name)
    else:
        eng = _representation_gp(name, p)
    eng.fit(ctx.X, ctx.y)
    qd = pc["quadrature"]
    _, C_raw, C_c = B.coherence(
        eng, Q, np.arange(M), int(qd["K"]), float(pc["eps"]), method=str(qd["method"]), u_min=float(qd["u_min"]),
        central_mass=float(qd["central_mass"]), n_scrambles=int(qd["n_scrambles"]),
        rng=rng_for(st.label, "representation_posterior_cov", level, t, draw, base_seed=cfg.seed),
        u_tail=float(pc["y_grid"]["u_tail"]), grid_points=int(pc["y_grid"]["points"]))
    imp = B.implied_posterior_covariance(C_c if pc["moment_matched"] else C_raw)
    C_pfn = imp["cov"]                                                              # [M, M]
    geometry = C.geometry_confounds(Q, p["deconfound"]["bandwidth_multipliers"])
    layout_eng = _representation_gp(p["layout_confound"], p)
    layout_eng.fit(ctx.X, ctx.y)
    layout = B.latent_covariance_of(layout_eng, Q)
    rows = []
    for ref in pc["references"]:
        g = _representation_gp(ref, p)
        g.fit(ctx.X, ctx.y)
        S = B.latent_covariance_of(g, Q)                                            # [M, M]
        confs = geometry if ref == p["layout_confound"] else [*geometry, layout]
        dc = C.cka_deconfounded(C_pfn, S, C.Deconfounder(confs),
                                min_residual_frac=float(p["deconfound"]["min_residual_frac"]))
        v = P.nbs(C_pfn, S)
        bd = P.bures(C_pfn, S)
        rows.append({"engine": name, "reference": ref, "cka": C.cka_or_none(C_pfn, S), "cka_deconf": dc["value"],
                     "deconf_degenerate": bool(dc["degenerate"]), "nbs": v,
                     "shape_distance": float(np.arccos(np.clip(v, 0.0, 1.0))), "bures": bd,
                     "bures_rel": bd / float(np.sqrt(np.trace(S))),
                     "trace_ratio": float(np.trace(C_pfn) / np.trace(S))})
    return {"rows": rows, "asymmetry": imp["asymmetry"], "clipped_mass": imp["clipped_mass"],
            "host": socket.gethostname(), "device": cfg.device, "seconds": time.time() - t0}


def run_representation(cfg: MechanismConfig, opts: RunOptions) -> str:
    """BETA β2 / β1a / β1b on the grid channel x level x t x draw; summaries and figures.

    Cells (one per (channel, level, t, draw), kind ``cell``; β1b's kind ``posterior_cov`` on its coverage) are cached
    as everywhere else, so ``--only-cached`` assembles what exists and a subject can be added without recomputing
    the others. Per channel rather than per grid (the plan's §7 layout): β3, the only analysis that needs every
    channel of a grid in one cell, moved to an optional appendix (D6a), and per-channel cells shard by subject.

    Args:
        cfg: Resolved config.
        opts: Cache policy.

    Returns:
        The run directory.
    """
    from ..analysis import cka as C  # noqa: PLC0415
    from ..analysis import representation as R  # noqa: PLC0415
    from ..data.snr import achieved_snr_db  # noqa: PLC0415
    from ..data.stress import build_knob  # noqa: PLC0415
    from ..models.pfn.tabpfn import FrozenTabPFN  # noqa: PLC0415
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _validate_representation(cfg.params, cfg.context_sizes)
    out = cfg.run_dir
    if not opts.replot:
        set_seed(cfg.seed)
        _write_config(cfg, p, out)
        cells = _Cells(cfg, opts)
        knob = build_knob(p["stress"]["knob"], p["stress"]["levels"])
        channels = _channels(cfg)
        tp = p["tabpfn"]
        engines: dict[int, Any] = {}

        def tabpfn(draw: int) -> Any:
            seed = int(tp["inference_seed"]) + int(draw)
            if seed not in engines:
                engines[seed] = FrozenTabPFN(device=cfg.device, random_state=seed, precision=str(tp["precision"]),
                                             max_batch_tokens=int(tp["max_batch_tokens"]),
                                             softmax_temperature=float(tp["softmax_temperature"]))
            return engines[seed]

        sur_cache: dict[tuple[str, float], list[np.ndarray]] = {}

        def surrogate_grams_for(X: np.ndarray) -> Callable[[float], list[np.ndarray]]:
            gid = grid_id(X)
            sp = p["surrogate"]

            def grams(ell: float) -> list[np.ndarray]:
                if (gid, ell) not in sur_cache:
                    sur_cache[(gid, ell)] = [C.linear_gram(C.smooth_field_surrogate(
                        X, ell, int(sp["n_fields"]), rng_for(gid, "representation-surrogate", ell, rep,
                                                             base_seed=cfg.seed), jitter=float(sp["jitter"])))
                        for rep in range(int(sp["n_replicates"]))]
                return sur_cache[(gid, ell)]
            return grams

        gp_ident = {e: _representation_gp_params(e, p) for e in p["belief_engines"]}
        common = {k: p[k] for k in ("layers", "readouts", "targets", "belief_engines", "layout_confound", "n_quantiles",
                                    "site_sets", "deconfound", "surrogate", "permutation", "ceiling",
                                    "target_relations", "site_match")}
        tables: dict[str, list[dict[str, Any]]] = {k: [] for k in (
            "alignment", "surrogate", "ceiling", "relations", "site_match", "lens", "posterior_cov")}
        pc = p["posterior_cov"]
        allowed = _first_per_subject(channels, int(pc["coverage"]["channels_per_subject"])) \
            if pc["coverage"]["channels_per_subject"] is not None else {c.label for c in channels}
        t0 = time.time()
        for ch in channels:
            for level in knob.levels:
                st = knob.apply(ch, level, rng_for(ch.label, "representation", level, base_seed=cfg.seed))
                snr = achieved_snr_db(st)
                for t in cfg.context_sizes:
                    if t >= st.n_sites:
                        continue
                    for draw in range(int(p["n_context_draws"])):
                        keys = {"dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg, "knob": knob.name,
                                "level": float(level), "achieved_snr_db": snr, "context_t": int(t), "draw": draw}
                        ident = {**cells.channel(ch), "knob": knob.name, "level": float(level), "t": int(t),
                                 "draw": draw, "stats_version": REPRESENTATION_STATS_VERSION,
                                 "tabpfn": {"inference_seed": int(tp["inference_seed"]) + draw,
                                            "softmax_temperature": float(tp["softmax_temperature"]),
                                            "precision": str(tp["precision"])},
                                 "gp": gp_ident, **common}
                        hit = cells.run("cell", ident, f"{ch.label} a={level:g} t={t} d={draw}",
                                        lambda st=st, level=level, t=t, draw=draw: (_representation_cell(
                                            st, float(level), int(t), draw, p, cfg, tabpfn(draw),
                                            surrogate_grams_for(st.X_pool)), {}))
                        if hit is not None:
                            pay = hit[0]
                            prov = {"host": pay["host"], "seconds": pay["seconds"]}
                            for k in ("alignment", "surrogate", "ceiling", "relations", "site_match", "lens"):
                                tables[k] += [{**keys, **r, **(prov if k == "alignment" else {})} for r in pay[k]]
                        cov = pc["coverage"]
                        if pc["enabled"] and ch.label in allowed and any(np.isclose(level, lv) for lv in cov["levels"]) \
                                and int(t) in {int(v) for v in cov["context_ts"]} and draw in {int(v) for v in cov["draws"]}:
                            pident = {**cells.channel(ch), "knob": knob.name, "level": float(level), "t": int(t),
                                      "draw": draw, "engine": pc["engine"],
                                      "engine_params": _engine_params(pc["engine"], p, draw),
                                      "references": pc["references"], "gp": gp_ident,
                                      "layout_confound": p["layout_confound"], "deconfound": p["deconfound"],
                                      **{k: pc[k] for k in ("quadrature", "eps", "y_grid", "moment_matched")}}
                            hit = cells.run("posterior_cov", pident, f"posterior_cov {ch.label} a={level:g} t={t} d={draw}",
                                            lambda st=st, level=level, t=t, draw=draw: (_posterior_cov_cell(
                                                st, float(level), int(t), draw, p, cfg), {}))
                            if hit is not None:
                                pay = hit[0]
                                tables["posterior_cov"] += [
                                    {**keys, **r, "asymmetry": pay["asymmetry"], "clipped_mass": pay["clipped_mass"],
                                     "host": pay["host"], "seconds": pay["seconds"]} for r in pay["rows"]]
                print(f"[mechanism] representation {ch.label} level={level} done ({time.time() - t0:.0f}s)",
                      flush=True)
        names = {"alignment": "representation.csv", "surrogate": "surrogate.csv", "ceiling": "ceiling.csv",
                 "relations": "target_relations.csv", "site_match": "site_match.csv", "lens": "lens.csv",
                 "posterior_cov": "posterior_cov.csv"}
        for k, rows in tables.items():
            if rows:
                _write_csv(rows, out, names[k])
        if tables["alignment"]:
            _representation_summaries(out, p, cfg.seed)
        cells.report()
    for path in figs.render_representation(out):
        print(f"[mechanism] wrote {path}")
    return out


def _representation_summaries(out: str, p: dict[str, Any], seed: int) -> None:
    """Headline columns, channel-bootstrap summaries and ``headline.json`` from the assembled CSVs (BETA D4).

    Args:
        out: Run directory (holds the assembled CSVs).
        p: Resolved ``representation`` block.
        seed: Bootstrap seed.
    """
    from ..analysis import representation as R  # noqa: PLC0415

    inf = p["inference"]
    kw = {"n_boot": int(inf["n_boot"]), "ci": float(inf["ci"]), "seed": int(seed)}
    stat = p["headline"]["statistic"]
    rep = R.add_headline(pd.read_csv(os.path.join(out, "representation.csv")), stat)
    rep["headline_margin"] = rep["headline_value"] - rep["headline_floor"]
    rep.to_csv(os.path.join(out, "representation.csv"), index=False)
    R.channel_summary(rep, ["headline_value", "headline_floor", "headline_margin", "cka", "cka_deconf", "nbs"],
                      ["level", "context_t", "site_set", "readout", "layer", "target"], **kw).to_csv(
        os.path.join(out, "representation_summary.csv"), index=False)
    path = os.path.join(out, "site_match.csv")
    if os.path.exists(path):
        sm = pd.read_csv(path)
        sm["site_pool"] = np.where(sm["observed"].astype(bool), "observed", "unobserved")
        # matched per site first (site_match.csv), then the median over the sites of a pool, per cell
        cell = (sm.groupby(["subject", "emg", "level", "context_t", "draw", "reference", "site_pool"])
                [["w2", "w2_over_sd", "location", "spread", "shape"]].median().reset_index())
        R.channel_summary(cell, ["w2", "w2_over_sd", "location", "spread", "shape"],
                          ["level", "context_t", "reference", "site_pool"], **kw).to_csv(
            os.path.join(out, "site_match_summary.csv"), index=False)
    path = os.path.join(out, "lens.csv")
    if os.path.exists(path):
        R.channel_summary(pd.read_csv(path), ["w2", "w2_over_sd", "location", "spread", "shape",
                                              "lens_site_sd_ratio", "lens_r_final"],
                          ["level", "context_t", "layer", "comparison", "site_pool"], **kw).to_csv(
            os.path.join(out, "lens_summary.csv"), index=False)
    path = os.path.join(out, "posterior_cov.csv")
    if os.path.exists(path):
        R.channel_summary(pd.read_csv(path), ["nbs", "cka", "cka_deconf", "shape_distance", "bures_rel",
                                              "trace_ratio", "asymmetry", "clipped_mass"],
                          ["level", "context_t", "engine", "reference"], **kw).to_csv(
            os.path.join(out, "posterior_cov_summary.csv"), index=False)
    with open(os.path.join(out, "headline.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "status": "OPEN (research_design.md §4, 2026-10-19): settled after the real-grid re-check",
            "statistic": stat, "readout": p["headline"]["readout"],
            "computable_alternatives": [s for s in R.HEADLINE_STATISTICS if s != stat],
            "floor": "smooth-field surrogate: the max over lengthscales of the per-lengthscale upper quantile",
            "surrogate": p["surrogate"],
            "lens_flag_rule": "OPEN (research_design.md §4, 2026-10-19): lens diagnostics written, nothing flagged",
            "claim": "decodability beyond geometry and layout (G6); never 'GP-like' from a gap between two CKA scores",
        }, fh, indent=2)


#: Analysis runners by name.
ANALYSES: dict[str, Callable[[MechanismConfig, RunOptions], str]] = {
    "bayes_update": run_bayes_update,
    "placement": run_placement,
    "cka": run_cka,
    "representation": run_representation,
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
