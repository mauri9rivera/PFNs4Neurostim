"""Hyp C mechanism analyses: update rule (M10, primary), placement (MMD/W2), CKA.

One CLI, one config family: the ``analysis`` key picks the runner and a same-named block
holds its parameters (unknown keys raise, as everywhere else)::

    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_update_rule_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_placement_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_cka_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_update_rule_nhp.yaml --replot
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
update_rule           ``probe`` (channel, level, t, draw, engine) -- the full probe result is cached, so F7
                      and the exemplar are derived at assembly; ``layers``; ``offmap``;
                      ``gate`` (each validation gate); ``profile_control`` (F7 known-GP / shuffled)
cka                   ``a`` (channel, level, t, draw, readout); ``b`` (grid, t, draw; feature mean);
                      ``gate`` (controls, per readout)
placement             ``B`` (channel); ``C`` (channel: every level and context size); ``gate`` (M0 ladder)
====================  ===================================================================================

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
from dataclasses import asdict, dataclass
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
    "MECHANISM_CACHE_VERSION",
]

#: Invalidation stamp of every mechanism cell and artifact; bump it to discard them all.
MECHANISM_CACHE_VERSION: int = 1

#: The TabPFN checkpoint every mechanism analysis probes (its registry version string enters the identity).
MECHANISM_MODEL: str = "tabpfn_v2_5"


@dataclass(frozen=True)
class MechanismConfig:
    """Resolved mechanism-analysis configuration.

    Attributes:
        analysis: ``'update_rule'``, ``'placement'`` or ``'cka'``.
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
        """Output directory of this analysis."""
        return os.path.join(self.output_root, "mechanism", self.analysis, self.dataset.name, self.tag)

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
# update_rule (task #9)
# ---------------------------------------------------------------------------
UPDATE_RULE_DEFAULTS: dict[str, Any] = {
    "engines": ["tabpfn_v2_5", "gp_mll_frozen", "gp_mll_refit"],
    "stress": {"knob": "k2_channel", "levels": [1.0, 2.0, 4.0]},
    "n_context_draws": 3,
    "n_anchors": 12,
    "surprises": [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
    "refit_surprises": [-1.0, -0.5, 0.5, 1.0],
    "gp_params": {"n_opt_steps": 100, "lr": 0.1},
    # Which GP every metric is measured against (rho_shape, the surprise-response reference, the exemplar's GP row):
    # ``mll`` is the converged type-II ML GP of ``gp_params``; ``fixed`` is the GP-fixed arm of the paper (no fitting,
    # the hyperparameters of ``fixed_gp_params``). Added 2026-10-02 because the converged fit can drive one ARD
    # lengthscale to ~0 and another to ~1e4 on a 10-point context, which makes the GP update a one-column stripe.
    "reference_gp": "mll",
    # Keep in step with configs/model/gp_naive.yaml by hand: the engines build the surrogate directly.
    "fixed_gp_params": {"lengthscale": 0.6931471805599453, "outputscale": 1.0, "noise": 0.05},
    "inference_seed": 0,
    "max_batch_tokens": 400000,
    # Secondary layer-wise arm. ``context_ts`` (a subset of the shared ladder; ``None`` = all of it): the layer
    # profile is a property of the surrogate *given its evidence*. It reads the feature-token mean only: the label
    # token was dropped from every mechanism analysis on 2026-10-07 (see analysis/embeddings.py).
    "layer_arm": {"enabled": True, "context_ts": None, "level": 1.0, "ridge": 1.0},
    # Which cell figure F1 (the exemplar delta-maps) is drawn from (2026-09-27 / 09-30, unchanged):
    #   context_ts      one column per context size (a subset of the shared ladder; ``None`` = all of it) for
    #                   ONE channel.
    #   level           the nominal anchor (alpha = 1.0).
    #   anchor_stratum  'centre' gives the full kernel bump (used by anchor: per_cell).
    #   select          'median_alignment' (representative by construction), 'explicit' or 'first'.
    #   anchor          'common' (default since 2026-10-07): ONE anchor site for every column -- the most central site of
    #                   the chosen channel that is unobserved in all of its exemplar contexts -- probed in a dedicated
    #                   ``exemplar`` cell per context size, so the columns differ only in the evidence. The grid cells
    #                   draw 12 anchors each, independently per context, and almost never share one, which made the
    #                   columns show different sites. 'per_cell' keeps that older behaviour (first anchor of the stratum).
    #   reference_gp    the GP of F1's second row: 'mll' or 'fixed'; ``None`` = ``update_rule.reference_gp``. Independent of
    #                   the reference every metric uses, because the converged ML fit's ARD lengthscales collapse on small
    #                   contexts and turn its update into a one-row / one-column stripe, which is a poor picture of "the GP
    #                   update" (user, 2026-10-07: F1 compares TabPFN with GP-fixed). Needs ``anchor: common``.
    # The plotted profile is the antisymmetrized response at the SMALLEST surprise magnitude, which is the
    # same estimator ``ell_hat`` uses, so the figure and the metric cannot disagree.
    "exemplar": {
        "context_ts": None,
        "level": 1.0,
        "draw": 0,
        "anchor_stratum": "centre",
        "select": "median_alignment",
        "subject": None,
        "emg": None,
        "anchor": "common",
        "reference_gp": None,
    },
    "gates": {
        "positive_r": 0.7, "lengthscale_tol": 0.25, "icc_min": 0.75, "linear_regime_tol": 0.25,
        "null_quantile": 0.95, "context_t": 25,
    },
    "positive_control": {"grid_side": 10, "lengthscale": 0.2, "noise_sd": 0.3, "n_trials": 10},
    "seed_floor": {"n_seeds": 10, "n_channels": 2},
    # F7 update_vs_distance (B2 Step 11, 2026-10-06). Derived at assembly from the cached probe results, so
    # changing a bin or a threshold re-assembles without recomputing; only the off-map and control cells compute.
    #   bin_edges     distance bins in electrode pitches (a final open bin is appended); [0, 0.5) holds the anchor.
    #   far_pitch     far field = d >= 4 pitches (half the NHP grid's short side).
    #   level         stress level of the F7 figures and of the off-map control (the nominal one).
    #   panel_ts      context sizes that get a radial-profile panel (the preview layout, restored 2026-10-07);
    #                 on the ladder, ``None`` = its first and middle sizes (10 and 50 on NHP). Keep them <= half the
    #                 grid: only 16 unobserved sites remain at t = 80 on NHP. The energy panel spans the whole ladder.
    #   headline_gp   the stated comparator whose difference to TabPFN the F7 supplement prints (user, 2026-10-06:
    #                 GP-fixed, the blind default). The main figure draws TabPFN, GP-fixed and GP-MLL (2026-10-07).
    #   offmap        control (1): a virtual anchor ``distance`` grid widths past the edge.
    #   controls      (3) known-GP synthetic channel, (4) shuffled coordinates on the first ``n_channels``, at
    #                 context size ``context_t``.
    "profile": {
        "enabled": True,
        "bin_edges": [0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 11.0],
        "far_pitch": 4.0,
        "level": 1.0,
        "panel_ts": None,
        "headline_gp": "gp_fixed_frozen",
        "offmap": {"enabled": True, "distance": 2.0},
        "controls": {"known_gp": True, "shuffled": True, "context_t": 25, "n_channels": 2},
    },
    # Step 7 (M8): join per-channel metrics with a Hyp A/B tidy.csv; skipped when null.
    "link": {
        "tidy_csv": None,
        "model": "tabpfn_v2_5",
        "outcomes": ["recommended_regret", "cumulative_regret"],
        "predictors": ["saturation_index_median", "ell_adaptivity", "rho_shape_median"],
        "n_boot": 2000,
    },
}

#: Engines whose update F7 profiles (the GP-refit arm re-optimizes per probe and has no single update).
PROFILE_ENGINES: tuple[str, ...] = ("tabpfn_v2_5", "gp_mll_frozen", "gp_fixed_frozen")


def _make_engine(name: str, p: dict[str, Any], device: str, inference_seed: int) -> Any:
    """Build one update engine by name.

    Args:
        name: ``'tabpfn_v2_5'``, ``'gp_mll_frozen'``, ``'gp_mll_refit'`` or ``'gp_fixed_frozen'``.
        p: Resolved ``update_rule`` block.
        device: Torch device for TabPFN.
        inference_seed: TabPFN preprocessing seed.

    Returns:
        The engine.
    """
    from ..analysis import update_rule as U  # noqa: PLC0415
    from ..models.gp.surrogates import GPSurrogate, NaiveGPSurrogate  # noqa: PLC0415

    if name == "tabpfn_v2_5":
        return U.PFNEngine(device=device, random_state=inference_seed,
                           max_batch_tokens=int(p["max_batch_tokens"]), name=name)
    if name == "gp_mll_frozen":
        return U.GPFrozenEngine(GPSurrogate(**p["gp_params"]), name=name)
    if name == "gp_mll_refit":
        return U.GPRefitEngine(GPSurrogate(**p["gp_params"]), name=name)
    if name == "gp_fixed_frozen":
        return U.GPFrozenEngine(NaiveGPSurrogate(**p["fixed_gp_params"]), name=name)
    raise ValueError(f"Unknown update_rule engine {name!r}.")


def _engine_params(name: str, p: dict[str, Any]) -> dict[str, Any]:
    """The settings of one engine that enter a cell identity."""
    if name == "tabpfn_v2_5":
        return {"inference_seed": int(p["inference_seed"])}
    if name in ("gp_mll_frozen", "gp_mll_refit"):
        return {"gp_params": p["gp_params"]}
    return {"fixed_gp_params": p["fixed_gp_params"]}


def _reference_params(p: dict[str, Any]) -> dict[str, Any]:
    """Identity of the reference GP (its kind and hyperparameter settings)."""
    kind = p["reference_gp"]
    return {"reference_gp": kind, "params": p["gp_params"] if kind == "mll" else p["fixed_gp_params"]}


def _make_reference(p: dict[str, Any], name: str = "reference") -> Any:
    """The frozen GP every update is measured against, chosen by ``update_rule.reference_gp``.

    Args:
        p: Resolved ``update_rule`` block.
        name: Engine name (``'reference'`` inside the grid).

    Returns:
        A :class:`GPFrozenEngine`: the converged ML GP for ``'mll'``, the fixed-kernel GP-fixed for ``'fixed'``.

    Raises:
        ValueError: For any other ``reference_gp``.
    """
    from ..analysis import update_rule as U  # noqa: PLC0415
    from ..models.gp.surrogates import GPSurrogate, NaiveGPSurrogate  # noqa: PLC0415

    kind = p["reference_gp"]
    if kind == "mll":
        return U.GPFrozenEngine(GPSurrogate(**p["gp_params"]), name=name)
    if kind == "fixed":
        return U.GPFrozenEngine(NaiveGPSurrogate(**p["fixed_gp_params"]), name=name)
    raise ValueError(f"update_rule.reference_gp must be 'mll' or 'fixed', got {kind!r}.")


def _res_payload(res: Any) -> dict[str, Any]:
    """A probe result as a picklable dict (the trajectory of a ``probe`` cell)."""
    return {"res": asdict(res)}


def _res_from(traj: dict[str, Any]) -> Any:
    """Rebuild a :class:`~pfns4neurostim.analysis.update_rule.ProbeResult` from a cell trajectory."""
    from ..analysis.update_rule import ProbeResult  # noqa: PLC0415

    return ProbeResult(**traj["res"])


def _has_map(ch: ChannelData) -> bool:
    """Whether a channel has a 2-D electrode map (F7 needs pitch distances; 5D condition sets have none)."""
    return ch.ch2xy is not None and np.asarray(ch.ch2xy).ndim == 2 and np.asarray(ch.ch2xy).shape[1] == 2


def run_update_rule(cfg: MechanismConfig, opts: RunOptions) -> str:
    """Task #9 Step 4: the M10 grid, its gates, the F6 layer arm and the F7 distance profile.

    Grid: channel x stress level x context size t x context draw x engine; every engine of a
    cell shares one context, one anchor set and one frozen reference GP.

    Args:
        cfg: Resolved config.
        opts: Cache policy.

    Returns:
        The run directory.
    """
    from ..analysis import update_rule as U  # noqa: PLC0415
    from ..data.snr import achieved_snr_db  # noqa: PLC0415
    from ..data.stress import build_knob  # noqa: PLC0415
    from ..models.gp.surrogates import NaiveGPSurrogate  # noqa: PLC0415
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _check_keys(cfg.params, UPDATE_RULE_DEFAULTS, "update_rule")
    for sub in ("gates", "layer_arm", "exemplar", "profile"):
        p[sub] = _check_keys(p[sub], UPDATE_RULE_DEFAULTS[sub], f"update_rule.{sub}")
    pf = p["profile"]
    pf["offmap"] = _check_keys(pf["offmap"], UPDATE_RULE_DEFAULTS["profile"]["offmap"], "update_rule.profile.offmap")
    pf["controls"] = _check_keys(pf["controls"], UPDATE_RULE_DEFAULTS["profile"]["controls"],
                                 "update_rule.profile.controls")
    ladder = cfg.context_sizes
    for sub, key in (("layer_arm", "context_ts"), ("exemplar", "context_ts")):
        if p[sub][key] is None:
            p[sub][key] = list(ladder)
        _subset_of_ladder(p[sub][key], ladder, f"update_rule.{sub}.{key}")
    ex = p["exemplar"]
    if ex["reference_gp"] is None:
        ex["reference_gp"] = p["reference_gp"]
    if ex["reference_gp"] not in ("mll", "fixed"):
        raise ValueError(f"update_rule.exemplar.reference_gp must be 'mll', 'fixed' or null, got {ex['reference_gp']!r}.")
    if ex["anchor"] == "per_cell" and ex["reference_gp"] != p["reference_gp"]:
        raise ValueError("update_rule.exemplar.reference_gp differs from update_rule.reference_gp, which needs "
                         "exemplar.anchor: common (per-cell columns reuse the grid's reference GP).")
    if pf["panel_ts"] is None:
        pf["panel_ts"] = sorted({int(ladder[0]), int(ladder[len(ladder) // 2])})
    if pf["enabled"]:
        _subset_of_ladder(pf["panel_ts"], ladder, "update_rule.profile.panel_ts")
    out = cfg.run_dir
    if not opts.replot:
        set_seed(cfg.seed)
        _write_config(cfg, p, out)
        cells = _Cells(cfg, opts)
        knob = build_knob(p["stress"]["knob"], p["stress"]["levels"])
        anchor_rows: list[dict[str, Any]] = []
        cell_rows: list[dict[str, Any]] = []
        surprise_rows: list[dict[str, Any]] = []
        layer_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        profile_c_rows: list[dict[str, Any]] = []
        offmap_rows: list[dict[str, Any]] = []
        exemplar_candidates: list[dict[str, Any]] = []
        channels = _channels(cfg)
        t0 = time.time()
        engines: dict[str, Any] = {}

        def engine(name: str) -> Any:
            if name not in engines:
                engines[name] = _make_engine(name, p, cfg.device, int(p["inference_seed"]))
            return engines[name]

        la = p["layer_arm"]
        common = {"n_anchors": int(p["n_anchors"]), "reference": _reference_params(p),
                  "knob": knob.name}
        for ch in channels:
            for level in knob.levels:
                stressed = knob.apply(ch, level, rng_for(ch.label, "update_rule", level, base_seed=cfg.seed))
                snr = achieved_snr_db(stressed)
                for t in ladder:
                    if t >= stressed.n_sites:
                        continue
                    for draw in range(int(p["n_context_draws"])):
                        # One shared context per (grid, t, draw) with this channel's own trials (#19 Step 1).
                        ctx = context_for(stressed, int(t), draw, base_seed=cfg.seed, level=float(level))
                        rng = rng_for(ch.label, "update_rule", level, t, draw, base_seed=cfg.seed)
                        anchors, strata = U.select_anchors(stressed, int(p["n_anchors"]), rng, exclude=ctx.sites)
                        ref_box: dict[str, Any] = {}

                        def reference() -> Any:
                            # Fitted lazily: a cell served from the cache never pays for the reference fit.
                            if "ref" not in ref_box:
                                ref_box["ref"] = _make_reference(p)
                                ref_box["ref"].fit(ctx.X, ctx.y)
                            return ref_box["ref"]

                        keys = {
                            "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg,
                            "knob": knob.name, "level": float(level), "achieved_snr_db": snr,
                            "context_t": int(t), "draw": draw, "reference_gp": p["reference_gp"],
                        }
                        cell_id = {**cells.channel(ch), **common, "level": float(level), "t": int(t), "draw": draw}
                        for name in p["engines"]:
                            surprises = p["refit_surprises"] if name == "gp_mll_refit" else p["surprises"]
                            ident = {**cell_id, "engine": name, "engine_params": _engine_params(name, p),
                                     "surprises": list(surprises)}

                            def compute(name: str = name, surprises: Sequence[float] = surprises,
                                        ) -> tuple[dict[str, Any], dict[str, Any]]:
                                res = U.run_probes(engine(name), reference(), ctx, stressed.X_pool, anchors, strata,
                                                   surprises)
                                rows, cell = U.update_metrics(res, stressed.X_pool)
                                return {"anchor_rows": rows, "cell": cell}, _res_payload(res)

                            hit = cells.run("probe", ident, f"{ch.label} a={level:g} t={t} d={draw} {name}", compute)
                            if hit is None:
                                continue
                            payload, traj = hit
                            res = _res_from(traj)
                            cell = dict(payload["cell"])
                            anchor_rows += [{**keys, **r} for r in payload["anchor_rows"]]
                            for i, a in enumerate(res.anchors):
                                for j, c in enumerate(res.surprises):
                                    surprise_rows.append({
                                        **keys, "engine": name, "anchor": int(a), "c": float(c),
                                        "u": float(c * res.base_sd[a]),
                                        "d_anchor": float(res.d_mean[i, j, a]),
                                        "d_gp_anchor": float(res.d_gp[i, j, a]),
                                        "dsd_anchor": float(res.d_sd[i, j, a]),
                                        "base_sd_anchor": float(res.base_sd[a]),
                                    })
                            if pf["enabled"] and name in PROFILE_ENGINES and _has_map(stressed):
                                prow, pcell = U.distance_profile(res, stressed.ch2xy, bin_edges=pf["bin_edges"],
                                                                 far_pitch=float(pf["far_pitch"]))
                                cell.update(pcell)
                                profile_rows += [{**keys, **r} for r in prow]
                                profile_c_rows += [{**keys, **r} for r in U.surprise_profile(
                                    res, stressed.ch2xy, far_pitch=float(pf["far_pitch"]))]
                            cell_rows.append({**keys, **cell})
                            if _exemplar_cell_matches(p["exemplar"], ch, level, t, draw) and (
                                    name == "tabpfn_v2_5" or "tabpfn_v2_5" not in p["engines"]):
                                exemplar_candidates.append(
                                    _exemplar(stressed, res, res.strata, p["exemplar"], keys, cell)
                                )
                            if (name == "tabpfn_v2_5" and la["enabled"]
                                    and int(t) in {int(v) for v in la["context_ts"]}
                                    and float(level) == float(la["level"])):
                                def compute_layers(res: Any = res) -> tuple[dict[str, Any], dict[str, Any]]:
                                    eng = engine("tabpfn_v2_5")
                                    eng.fit(ctx.X, ctx.y)
                                    return {"rows": U.layer_alignment(eng, res, stressed.X_pool,
                                                                      ridge=float(la["ridge"]))}, {}

                                lh = cells.run("layers", {**ident, "ridge": float(la["ridge"])},
                                               f"{ch.label} t={t} d={draw} layers", compute_layers)
                                if lh is not None:
                                    layer_rows += [{**keys, **r} for r in lh[0]["rows"]]
                            if (pf["enabled"] and pf["offmap"]["enabled"] and name in PROFILE_ENGINES
                                    and float(level) == float(pf["level"]) and _has_map(stressed)):
                                oh = cells.run(
                                    "offmap", {**ident, "offmap_distance": float(pf["offmap"]["distance"])},
                                    f"{ch.label} t={t} d={draw} {name} off-map",
                                    lambda name=name, surprises=surprises: _offmap_cell(
                                        engine(name), reference(), ctx, stressed, surprises, float(pf["offmap"]["distance"])),
                                )
                                if oh is not None:
                                    offmap_rows.append({**keys, "engine": name, **oh[0]})
                print(f"[mechanism] {ch.label} level={level} done ({time.time() - t0:.0f}s)", flush=True)
        _write_csv(anchor_rows, out, "update_rule.csv")
        _write_csv(cell_rows, out, "update_rule_cell.csv")
        _write_csv(surprise_rows, out, "update_rule_surprise.csv")
        for rows, name in ((layer_rows, "update_rule_layers.csv"), (profile_rows, "update_rule_profile.csv"),
                           (profile_c_rows, "update_rule_profile_c.csv"), (offmap_rows, "update_rule_offmap.csv")):
            if rows:
                _write_csv(rows, out, name)
        exemplar = _select_exemplar(exemplar_candidates, p["exemplar"])
        if exemplar is not None and p["exemplar"]["anchor"] == "common":
            exemplar = _common_anchor_exemplar(exemplar, p, cfg, cells, channels, knob, engine)
        elif p["exemplar"]["anchor"] not in ("common", "per_cell"):
            raise ValueError(f"update_rule.exemplar.anchor must be 'common' or 'per_cell', got {p['exemplar']['anchor']!r}.")
        if exemplar is not None:
            np.savez(os.path.join(out, "update_rule_exemplar.npz"), **exemplar)
        else:
            print(f"[mechanism] no exemplar cell matched {p['exemplar']}; F1 will be skipped", flush=True)

        # --- validation gates (Step 2), one cell each ---
        g = p["gates"]
        pc = p["positive_control"]
        truth = lambda: U.GPFrozenEngine(  # noqa: E731
            NaiveGPSurrogate(lengthscale=pc["lengthscale"], outputscale=1.0, noise=pc["noise_sd"] ** 2),
            name="truth",
        )
        gates: list[dict[str, Any]] = []
        gate_common = {"gates": g, "positive_control": pc, "n_anchors": int(p["n_anchors"]),
                       "surprises": list(p["surprises"]), "reference": _reference_params(p)}

        def gate(ident: dict[str, Any], label: str, fn: Callable[[], dict[str, Any]]) -> None:
            hit = cells.run("gate", {**gate_common, **ident}, label, lambda: ({"gate": fn()}, {}))
            if hit is not None:
                gates.append(hit[0]["gate"])

        def _positive(eng: Any) -> dict[str, Any]:
            return U.positive_control(
                eng, truth(), grid_side=pc["grid_side"], lengthscale=pc["lengthscale"],
                noise_sd=pc["noise_sd"], n_trials=pc["n_trials"], t=g["context_t"],
                n_anchors=int(p["n_anchors"]), surprises=p["surprises"], min_r=g["positive_r"],
                lengthscale_tol=g["lengthscale_tol"], seed=cfg.seed,
            )

        # Probe validity first: the true GP through the probe must pass, otherwise "the probe,
        # not the model, is at fault" and no model result below is interpretable.
        gate({"gate": "probe_validity"}, "probe validity", lambda: {**_positive(truth()), "role": "probe_validity"})
        for name in p["engines"]:
            eparams = {"engine": name, "engine_params": _engine_params(name, p)}
            gate({"gate": "positive_control", **eparams}, f"positive {name}",
                 lambda name=name: {**_positive(engine(name)), "role": "model"})
            if channels:
                ch0 = channels[0]
                gate({"gate": "negative_control", **eparams, **cells.channel(ch0)}, f"negative {name}",
                     lambda name=name, ch0=ch0: {**U.negative_control(
                         engine(name), _make_reference(p), ch0,
                         t=min(g["context_t"], ch0.n_sites - 1), n_anchors=int(p["n_anchors"]),
                         surprises=p["surprises"], null_quantile=g["null_quantile"], seed=cfg.seed,
                     ), "channel": ch0.label})
            eng_rows = [r for r in anchor_rows if r["engine"] == name]
            gates.append({**U.linear_regime_check(eng_rows, g["linear_regime_tol"]), "engine": name})
        if "tabpfn_v2_5" in p["engines"]:
            sf = p["seed_floor"]
            for ch in channels[: int(sf["n_channels"])]:
                extra = None
                if pf["enabled"] and _has_map(ch):
                    def extra(res: Any, ch: ChannelData = ch) -> dict[str, float | None]:
                        cell = U.distance_profile(res, ch.ch2xy, bin_edges=pf["bin_edges"],
                                                  far_pitch=float(pf["far_pitch"]))[1]
                        return {"far_field_share": cell["far_field_share"], "locality_index": cell["locality_index"]}
                gate({"gate": "seed_floor", "n_seeds": int(sf["n_seeds"]), **cells.channel(ch),
                      "engine_params": _engine_params("tabpfn_v2_5", p), "profile_far_pitch": float(pf["far_pitch"]),
                      "profile_bins": list(pf["bin_edges"]) if extra else None},
                     f"seed floor {ch.label}",
                     lambda ch=ch, extra=extra: {**U.seed_floor(
                         lambda s: _make_engine("tabpfn_v2_5", p, cfg.device, s),
                         _make_reference(p), ch,
                         n_seeds=int(sf["n_seeds"]), t=min(g["context_t"], ch.n_sites - 1),
                         n_anchors=int(p["n_anchors"]), surprises=p["surprises"], icc_min=g["icc_min"],
                         seed=cfg.seed, extra_metrics=extra,
                     ), "engine": "tabpfn_v2_5", "channel": ch.label})
        with open(os.path.join(out, "gates.json"), "w", encoding="utf-8") as fh:
            json.dump(gates, fh, indent=2, default=_json_default)
        for gt in gates:
            print(f"[mechanism] gate {gt['gate']} {gt.get('engine', '')}: "
                  f"{'PASS' if gt['passed'] else 'FAIL'}", flush=True)

        # --- F7 supplement controls (3) known GP and (4) shuffled coordinates ---
        if pf["enabled"]:
            ctrl_rows = _profile_controls(cfg, p, cells, channels, engine, truth)
            if ctrl_rows:
                _write_csv(ctrl_rows, out, "update_rule_profile_controls.csv")
    link = p["link"]
    if link.get("tidy_csv"):
        path = _predictive_link(out, link, cfg.seed)
        print(f"[mechanism] wrote {path}")
    for path in figs.render_update_rule(out):
        print(f"[mechanism] wrote {path}")
    if not opts.replot:
        cells.report()
    return out


def _write_csv(rows: list[dict[str, Any]], out: str, name: str) -> None:
    """Write assembled rows to ``out/name``."""
    pd.DataFrame(rows).to_csv(os.path.join(out, name), index=False)


def _offmap_cell(
    engine: Any,
    reference: Any,
    ctx: Any,
    channel: ChannelData,
    surprises: Sequence[float],
    distance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """F7 control (1): the update on the grid caused by an observation placed off the map.

    Args:
        engine: Update engine (fitted inside :func:`run_probes`).
        reference: Fitted frozen reference GP of the cell.
        ctx: The cell's context.
        channel: The (stressed) channel.
        surprises: Surprise multipliers.
        distance: Off-map distance in grid widths (:func:`analysis.update_rule.offmap_point`).

    Returns:
        ``(payload, {})`` with the median ``|T|`` over unobserved and observed grid sites, the reference GP's,
        and the off-map point's separation from the grid in the engine's own input space relative to the grid's
        diameter there. TabPFN quantile-transforms its inputs, which can map an off-map point onto the edge of
        the grid -- a separation near 0 says the control is uninformative for that engine.
    """
    from ..analysis import update_rule as U  # noqa: PLC0415

    X = channel.X_pool
    x_off = U.offmap_point(X, distance)                                     # [D]
    Q = np.vstack([X, x_off[None]])                                         # [N+1, D]
    res = U.run_probes(engine, reference, ctx, Q, np.array([len(X)]), ["offmap"], surprises)
    T, c0 = U.transfer(res)                                                 # [1, N+1]
    observed = np.zeros(len(X), dtype=bool)
    observed[ctx.sites] = True
    grid_T = np.abs(T[0, :-1])                                              # [N]
    g_ref = U._antisym(res.d_gp, res.surprises, c0)[0, :-1] / (c0 * res.base_sd[len(X)])
    if hasattr(engine, "engine"):
        space = engine.engine.transform_x(Q).double().cpu().numpy()        # [N+1, F] model space
    else:
        space = Q
    sep = float(np.min(np.linalg.norm(space[:-1] - space[-1], axis=1)))
    diam = float(np.max(np.linalg.norm(space[:-1, None] - space[None, :-1], axis=-1)))
    return {
        "offmap_transfer_median": float(np.median(grid_T[~observed])),
        "offmap_transfer_observed_median": float(np.median(grid_T[observed])) if observed.any() else None,
        "offmap_reference_transfer_median": float(np.median(np.abs(g_ref[~observed]))),
        "offmap_separation": sep / diam if diam > 0 else None,
        "offmap_distance": float(distance),
        "c0": float(c0),
    }, {}


def _profile_controls(
    cfg: MechanismConfig,
    p: dict[str, Any],
    cells: _Cells,
    channels: list[ChannelData],
    engine: Callable[[str], Any],
    truth: Callable[[], Any],
) -> list[dict[str, Any]]:
    """F7 controls (3) the known-GP channel and (4) shuffled coordinates, one cell per (control, engine).

    Args:
        cfg: Resolved config.
        p: Resolved ``update_rule`` block.
        cells: Cell namespace.
        channels: Loaded channels.
        engine: ``name -> engine`` (shared, lazily built).
        truth: Factory of the true-GP engine of the positive control.

    Returns:
        Profile rows with a ``control`` column (``known_gp``, ``original``, ``shuffled``).
    """
    from ..analysis import update_rule as U  # noqa: PLC0415

    pf = p["profile"]
    cc = pf["controls"]
    pc = p["positive_control"]
    rows: list[dict[str, Any]] = []
    profile_kw = {"bin_edges": pf["bin_edges"], "far_pitch": float(pf["far_pitch"])}
    names = [n for n in p["engines"] if n in PROFILE_ENGINES]
    base = {"n_anchors": int(p["n_anchors"]), "surprises": list(p["surprises"]), "t": int(cc["context_t"])}
    if cc["known_gp"]:
        ctrl = U.gp_channel(pc["grid_side"], pc["lengthscale"], pc["noise_sd"], pc["n_trials"],
                            np.random.default_rng(cfg.seed))
        t = min(int(cc["context_t"]), ctrl.n_sites - 1)
        for name in names + ["truth"]:
            def compute(name: str = name) -> tuple[dict[str, Any], dict[str, Any]]:
                eng = truth() if name == "truth" else engine(name)
                _, _, res = U._probe_cell(ctrl, eng, truth(), t, int(p["n_anchors"]), p["surprises"],
                                          np.random.default_rng(cfg.seed))
                prow, pcell = U.distance_profile(res, ctrl.ch2xy, **profile_kw)
                return {"rows": prow, "cell": pcell}, {}

            ident = {**base, "control": "known_gp", "positive_control": pc, "engine": name,
                     "engine_params": None if name == "truth" else _engine_params(name, p), "profile": profile_kw}
            hit = cells.run("profile_control", ident, f"known-GP {name}", compute)
            if hit is not None:
                rows += [{"control": "known_gp", "context_t": t, **r} for r in hit[0]["rows"]]
    if cc["shuffled"]:
        for ch in [c for c in channels if _has_map(c)][: int(cc["n_channels"])]:
            t = min(int(cc["context_t"]), ch.n_sites - 1)
            for control in ("original", "shuffled"):
                for name in names:
                    def compute(name: str = name, ch: ChannelData = ch, control: str = control,
                                ) -> tuple[dict[str, Any], dict[str, Any]]:
                        rng = rng_for(ch.label, "f7_shuffle", base_seed=cfg.seed)
                        shown = U.shuffle_coordinates(ch, rng) if control == "shuffled" else ch
                        _, _, res = U._probe_cell(shown, engine(name), _make_reference(p), t, int(p["n_anchors"]),
                                                  p["surprises"], rng_for(ch.label, "f7_context", base_seed=cfg.seed),
                                                  readout_X=ch.X_pool)
                        prow, pcell = U.distance_profile(res, ch.ch2xy, **profile_kw)
                        return {"rows": prow, "cell": pcell}, {}

                    ident = {**base, "control": control, **cells.channel(ch), "engine": name,
                             "engine_params": _engine_params(name, p), "reference": _reference_params(p),
                             "profile": profile_kw}
                    hit = cells.run("profile_control", ident, f"{control} {ch.label} {name}", compute)
                    if hit is not None:
                        rows += [{"control": control, "subject": ch.subject, "emg": ch.emg, "context_t": t,
                                  **hit[0]["cell"], **r} for r in hit[0]["rows"]]
    return rows


def _predictive_link(out: str, link: dict[str, Any], seed: int) -> str:
    """Task #9 Step 7: regress per-channel regret on per-channel M10 metrics.

    Args:
        out: Update-rule run directory (holds ``update_rule_cell.csv``).
        link: Resolved ``update_rule.link`` block.
        seed: Bootstrap seed.

    Returns:
        Path of ``predictive_link.csv``.
    """
    from ..analysis.predictive_link import channel_mechanism_table, predictive_link  # noqa: PLC0415

    cell = pd.read_csv(os.path.join(out, "update_rule_cell.csv"))
    mech = channel_mechanism_table(cell)
    tidy = pd.read_csv(link["tidy_csv"])
    tidy = tidy[tidy["model"] == link["model"]]
    regret = tidy.groupby(["dataset", "subject", "emg"])[list(link["outcomes"])].mean().reset_index()
    joined = mech.merge(regret, on=["dataset", "subject", "emg"], how="inner")
    rows = []
    for outcome in link["outcomes"]:
        for predictor in link["predictors"]:
            if predictor in joined and joined[predictor].notna().sum() >= 3:
                rows.append(predictive_link(joined, predictor, outcome, n_boot=int(link["n_boot"]), seed=seed))
    path = os.path.join(out, "predictive_link.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _exemplar_cell_matches(
    spec: dict[str, Any],
    channel: ChannelData,
    level: float,
    context_t: int,
    draw: int,
) -> bool:
    """Whether this (channel, level, t, draw) cell is a candidate for figure F1.

    Args:
        spec: The resolved ``update_rule.exemplar`` block.
        channel: Channel being probed.
        level: Stress level of the cell.
        context_t: Context size of the cell.
        draw: Context-draw index of the cell.

    Returns:
        True when the cell matches the requested operating point (and, for ``select: explicit``, the
        requested channel).
    """
    if (int(context_t) not in {int(v) for v in spec["context_ts"]}
            or float(level) != float(spec["level"]) or int(draw) != int(spec["draw"])):
        return False
    if spec["select"] == "explicit":
        return (int(channel.subject) == int(spec["subject"]) and int(channel.emg) == int(spec["emg"]))
    return True


def _exemplar(
    channel: ChannelData,
    res: Any,
    strata: Sequence[str],
    spec: dict[str, Any],
    keys: dict[str, Any],
    cell: dict[str, Any],
) -> dict[str, Any]:
    """Arrays and full provenance of one candidate exemplar for figure F1 (grid heatmaps).

    The plotted profile is the antisymmetrized change at the smallest surprise magnitude -- the same
    estimator :func:`analysis.update_rule.implicit_kernel` uses for ``ell_hat`` -- so the figure and the
    quantitative metric describe the same object. The anchor is the first one of the requested stratum,
    falling back to the first anchor when that stratum was not sampled in this cell.

    Args:
        channel: Channel being probed.
        res: Probe result for one engine.
        strata: Stratum name of each sampled anchor, parallel to ``res.anchors``.
        spec: The resolved ``update_rule.exemplar`` block.
        keys: Cell key columns (dataset, subject, emg, knob, level, achieved SNR, context size, draw).
        cell: Per-cell metrics, used to rank candidates under ``select: median_alignment``.

    Returns:
        A mapping ready for :func:`numpy.savez`, carrying the arrays and every field needed to caption
        and reproduce the figure.
    """
    from ..analysis.update_rule import _antisym, _symmetric_magnitudes  # noqa: PLC0415

    c0 = _symmetric_magnitudes(res.surprises)[0]
    wanted = str(spec["anchor_stratum"])
    idx = next((i for i, s in enumerate(strata) if s == wanted), 0)
    return {
        "engine": np.array(res.engine),
        "label": np.array(channel.label),
        "anchor": np.array(int(res.anchors[idx])),
        "anchor_stratum": np.array(strata[idx] if idx < len(strata) else "unknown"),
        "anchor_stratum_requested": np.array(wanted),
        "surprise_c": np.array(float(c0)),
        "select_rule": np.array(str(spec["select"])),
        "subject": np.array(int(keys["subject"])),
        "emg": np.array(int(keys["emg"])),
        "knob": np.array(str(keys["knob"])),
        "level": np.array(float(keys["level"])),
        "achieved_snr_db": np.array(float(keys["achieved_snr_db"])),
        "context_t": np.array(int(keys["context_t"])),
        "draw": np.array(int(keys["draw"])),
        "reference_gp": np.array(str(keys.get("reference_gp", "mll"))),
        "rho_shape_offanchor": np.array(np.nan if cell.get("rho_shape_offanchor_median") is None
                                        else float(cell["rho_shape_offanchor_median"])),
        "ch2xy": channel.ch2xy,
        "grid_shape": np.array(channel.grid_shape),
        "g_model": _antisym(res.d_mean, res.surprises, c0)[idx],
        "g_gp": _antisym(res.d_gp, res.surprises, c0)[idx],
    }


def _select_exemplar(
    candidates: Sequence[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any] | None:
    """Pick ONE channel and stack its exemplars over the requested context sizes (figure F1).

    The channel is chosen once and then shown at every context size, so the columns of F1 differ only in
    how much evidence the surrogate was given. ``median_alignment`` ranks channels by the median of their
    ``rho_shape_offanchor`` over those context sizes and takes the channel at the median of that ranking: a
    median exemplar is representative by construction, whereas selecting the best-aligned channel would
    turn F1 into an upper bound rather than a typical case, so that option is deliberately absent.

    The anchor is drawn per (context, draw) cell upstream, so it may differ between columns; the per-column
    anchors are stored and each panel marks its own.

    Args:
        candidates: Candidate exemplars from :func:`_exemplar`, over channels and context sizes.
        spec: The resolved ``update_rule.exemplar`` block.

    Returns:
        A mapping ready for :func:`numpy.savez` whose ``g_model`` / ``g_gp`` are ``[T, N]`` and whose
        ``context_ts`` / ``anchors`` are ``[T]``, ordered by context size; or ``None`` when no cell matched.

    Raises:
        ValueError: If ``spec['select']`` is not a known rule.
    """
    if not candidates:
        return None
    rule = str(spec["select"])
    if rule not in ("first", "explicit", "median_alignment"):
        raise ValueError(
            f"update_rule.exemplar.select={rule!r} is not one of 'median_alignment', 'explicit', 'first'."
        )
    by_channel: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cand in candidates:
        by_channel.setdefault((int(cand["subject"]), int(cand["emg"])), []).append(cand)
    if rule in ("first", "explicit"):
        chosen = by_channel[(int(candidates[0]["subject"]), int(candidates[0]["emg"]))]
    else:
        keys = list(by_channel)
        scores = np.array([
            np.nanmedian([float(c["rho_shape_offanchor"]) for c in by_channel[k]]) for k in keys
        ])
        finite = np.flatnonzero(np.isfinite(scores))
        if finite.size == 0:
            chosen = by_channel[keys[0]]
        else:
            # Lower median of the finite scores, so the pick is an actual channel rather than an interpolant.
            ordered = finite[np.argsort(scores[finite], kind="stable")]
            chosen = by_channel[keys[int(ordered[(len(ordered) - 1) // 2])]]
    chosen = sorted(chosen, key=lambda c: int(c["context_t"]))
    ref = chosen[len(chosen) // 2]     # provenance scalars come from the middle context
    stacked = {k: v for k, v in ref.items() if k not in ("g_model", "g_gp", "anchor", "context_t")}
    stacked["g_model"] = np.stack([c["g_model"] for c in chosen])          # [T, N]
    stacked["g_gp"] = np.stack([c["g_gp"] for c in chosen])                # [T, N]
    stacked["anchors"] = np.array([int(c["anchor"]) for c in chosen])      # [T]
    stacked["context_ts"] = np.array([int(c["context_t"]) for c in chosen])  # [T]
    return stacked


def _common_anchor_exemplar(
    stacked: dict[str, Any],
    p: dict[str, Any],
    cfg: MechanismConfig,
    cells: _Cells,
    channels: list[ChannelData],
    knob: Any,
    engine: Callable[[str], Any],
) -> dict[str, Any] | None:
    """Re-probe the selected F1 channel at ONE anchor site shared by every context column.

    The channel picked by :func:`_select_exemplar` is kept. Among its sites that are unobserved in all of the
    exemplar's contexts (same draw, same level), the one closest to the electrode-map centroid is the anchor (the
    'centre' stratum's full kernel bump); for each context size one ``exemplar`` cell probes it with the exemplar's
    engine and the GP of ``exemplar.reference_gp`` fitted on that context, exactly as
    :func:`analysis.update_rule.run_probes` does in the grid: the GP's update is its exact posterior update to the same
    injected observation (x*, y*) the probed model received. Falls back to the per-cell anchors when no site is
    unobserved in every context (only possible when ``exemplar.reference_gp`` equals the grid's reference).

    Args:
        stacked: The per-cell exemplar from :func:`_select_exemplar`.
        p: Resolved ``update_rule`` block.
        cfg: Resolved config.
        cells: Cell namespace.
        channels: Loaded channels.
        knob: The stress knob of the grid.
        engine: ``name -> engine`` (shared, lazily built).

    Returns:
        The exemplar mapping with ``g_model`` / ``g_gp`` re-probed at the common anchor, ``anchors`` repeating it,
        ``reference_gp`` naming the GP row and ``anchor_rule`` recording which rule produced the columns; ``None`` when
        an ``--only-cached`` run lacks a common-anchor cell.

    Raises:
        RuntimeError: If no site is unobserved in every context while F1's GP differs from the grid's reference.
    """
    from ..analysis import update_rule as U  # noqa: PLC0415

    spec = p["exemplar"]
    p_ex = {**p, "reference_gp": spec["reference_gp"]}       # the GP of F1's second row
    ch = next(c for c in channels if (int(c.subject), int(c.emg)) == (int(stacked["subject"]), int(stacked["emg"])))
    level, draw = float(spec["level"]), int(spec["draw"])
    stressed = knob.apply(ch, level, rng_for(ch.label, "update_rule", level, base_seed=cfg.seed))
    ts = [int(t) for t in stacked["context_ts"]]
    ctxs = {t: context_for(stressed, t, draw, base_seed=cfg.seed, level=level) for t in ts}
    observed = set().union(*(set(int(i) for i in c.sites) for c in ctxs.values()))
    free = np.array(sorted(set(range(stressed.n_sites)) - observed), dtype=int)                # [F]
    if free.size == 0:
        if spec["reference_gp"] != p["reference_gp"]:
            raise RuntimeError(f"F1: no site of {ch.label} is unobserved in every exemplar context, and the per-cell "
                               "fallback cannot show exemplar.reference_gp; pick another exemplar draw or channel.")
        print(f"[mechanism] F1: no site of {ch.label} is unobserved in every context; keeping per-cell anchors",
              flush=True)
        return {**stacked, "anchor_rule": np.array("per_cell (no common unobserved site)")}
    xy = np.asarray(stressed.ch2xy, dtype=np.float64)                                          # [N, 2]
    site = int(free[np.argmin(np.linalg.norm(xy[free] - xy.mean(axis=0), axis=1))])
    name = str(stacked["engine"])
    g_model, g_gp = [], []
    for t in ts:
        ctx = ctxs[t]

        def compute(ctx: Any = ctx) -> tuple[dict[str, Any], dict[str, Any]]:
            reference = _make_reference(p_ex)
            reference.fit(ctx.X, ctx.y)
            res = U.run_probes(engine(name), reference, ctx, stressed.X_pool, np.array([site]), ["centre"],
                               p["surprises"])
            c0 = U._symmetric_magnitudes(res.surprises)[0]
            return {}, {"g_model": U._antisym(res.d_mean, res.surprises, c0)[0],
                        "g_gp": U._antisym(res.d_gp, res.surprises, c0)[0]}

        ident = {**cells.channel(ch), "knob": knob.name, "level": level, "t": t, "draw": draw, "anchor": site,
                 "engine": name, "engine_params": _engine_params(name, p), "reference": _reference_params(p_ex),
                 "surprises": list(p["surprises"])}
        hit = cells.run("exemplar", ident, f"F1 {ch.label} t={t} anchor {site}", compute)
        if hit is None:
            print(f"[mechanism] F1: common-anchor cell missing for t={t}; F1 is skipped", flush=True)
            return None
        g_model.append(hit[1]["g_model"])
        g_gp.append(hit[1]["g_gp"])
    return {**stacked, "g_model": np.stack(g_model), "g_gp": np.stack(g_gp),                  # [T, N] each
            "anchors": np.full(len(ts), site), "anchor_stratum": np.array("centre (common)"),
            "reference_gp": np.array(str(spec["reference_gp"])),
            "anchor_rule": np.array("common: most central site unobserved in every column")}


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
    from ..analysis import update_rule as U  # noqa: PLC0415
    from ..analysis.cka import linear_gram, target_kernels  # noqa: PLC0415
    from ..analysis.embeddings import READOUTS, readout_embeddings  # noqa: PLC0415
    from ..analysis.placement import placement as place  # noqa: PLC0415
    from ..data.snr import achieved_snr_db  # noqa: PLC0415
    from ..data.stress import build_knob  # noqa: PLC0415
    from ..models.gp.surrogates import GPSurrogate  # noqa: PLC0415
    from ..models.pfn.tabpfn import FrozenTabPFN  # noqa: PLC0415
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _check_keys(cfg.params, CKA_DEFAULTS, "cka")
    p["controls"] = _check_keys(p["controls"], CKA_DEFAULTS["controls"], "cka.controls")
    p["placement"] = _check_keys(p["placement"], CKA_DEFAULTS["placement"], "cka.placement")
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

        # --- M0 controls (Step 4), per readout ---
        c = p["controls"]
        gates: list[dict[str, Any]] = []
        for readout in p["readouts"]:
            for label, noise_only in (("positive_control", False), ("negative_control", True)):
                def control(readout: str = readout, noise_only: bool = noise_only, label: str = label,
                            ) -> tuple[dict[str, Any], dict[str, Any]]:
                    from dataclasses import replace  # noqa: PLC0415

                    rng = np.random.default_rng(cfg.seed)
                    ctrl = U.gp_channel(c["grid_side"], c["lengthscale"], c["noise_sd"], c["n_trials"], rng)
                    if noise_only:
                        y = rng.normal(size=ctrl.n_sites)
                        ctrl = replace(ctrl, y_gt=y,
                                       Y_trials=y[:, None] + c["noise_sd"] * rng.normal(size=ctrl.Y_trials.shape))
                    ctx = U.draw_context(ctrl, int(c["t"]), rng)
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
                    ctx = U.draw_context(ch, min(int(c["t"]), ch.n_sites - 1), np.random.default_rng(cfg.seed))
                    gp = GPSurrogate(**p["gp_params"])
                    gp.fit(ctx.X, ctx.y)
                    ks = {k: v for k, v in target_kernels(ch.X_pool, ch.y_gt, gp.posterior_covariance(ch.X_pool)).items()
                          if k in p["targets"]}
                    per_seed = [_layer_cka(embed(ctx.X, ctx.y, ch.X_pool, readout,
                                                 eng=FrozenTabPFN(device=cfg.device, random_state=s)), ks)
                                for s in range(int(c["n_seeds"]))]
                    icc = U.icc_oneway(np.asarray(per_seed).T)
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
                        floors = [float(np.median([dist(h, s) for h in hold])) for s in range(n_lay)]
                        ceilings = [float(np.median([dist(z, s) for z in nz])) for s in range(n_lay)]
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
                                })
                        return {"rows": out_rows}, {}

                    ident = {"grid_id": grid, "channels": [cells.channel(ch) for ch in grid_channels],
                             "t": int(t_ctx), "draw": draw, **embed_ident(readout),
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


#: Analysis runners by name.
ANALYSES: dict[str, Callable[[MechanismConfig, RunOptions], str]] = {
    "update_rule": run_update_rule,
    "placement": run_placement,
    "cka": run_cka,
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
