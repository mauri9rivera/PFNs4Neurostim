"""Hyp C mechanism analyses: update rule (M10, primary), placement (MMD/W2), CKA.

One CLI, one config family: the ``analysis`` key picks the runner and a same-named block
holds its parameters (unknown keys raise, as everywhere else)::

    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_update_rule_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_placement_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_cka_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_update_rule_nhp.yaml --replot

Outputs go to ``{output_root}/mechanism/{analysis}/{dataset}/{tag}/`` with the resolved
``config.yaml``; every figure is rebuilt from the CSVs alone with ``--replot``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from ..config import DatasetConfig, apply_overrides, compose, dataset_config_from_raw
from ..data.channels import ChannelData, iter_channels
from ..seeding import rng_for, set_seed

__all__ = ["MechanismConfig", "load_mechanism_config", "run_mechanism", "main", "ANALYSES"]


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
        params: The analysis block (validated by the analysis runner).
        source_path: YAML path.
    """

    analysis: str
    tag: str
    dataset: DatasetConfig
    device: str
    seed: int
    output_root: str
    params: dict[str, Any]
    source_path: str

    @property
    def run_dir(self) -> str:
        """Output directory of this analysis."""
        return os.path.join(self.output_root, "mechanism", self.analysis, self.dataset.name, self.tag)


def load_mechanism_config(path: str, overrides: list[str] | None = None) -> MechanismConfig:
    """Compose (``defaults: {dataset: ...}``), override and validate a mechanism config.

    Args:
        path: YAML path.
        overrides: Dotted ``key=value`` overrides (never invent keys).

    Returns:
        The resolved config.

    Raises:
        ValueError: On unknown keys or an unknown analysis.
    """
    merged = compose(path)
    if overrides:
        merged = apply_overrides(merged, list(overrides))
    dataset = dataset_config_from_raw(merged.pop("dataset"))
    analysis = str(merged.pop("analysis"))
    if analysis not in ANALYSES:
        raise ValueError(f"Unknown analysis {analysis!r}; expected one of {sorted(ANALYSES)}.")
    params = dict(merged.pop(analysis, {}) or {})
    merged.pop("experiment", None)
    cfg = MechanismConfig(
        analysis=analysis,
        tag=str(merged.pop("tag", analysis)),
        dataset=dataset,
        device=os.path.expandvars(str(merged.pop("device", "cpu"))),
        seed=int(merged.pop("seed", 42)),
        output_root=str(merged.pop("output_root", "output")),
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
        "device": cfg.device,
        "seed": cfg.seed,
        "output_root": cfg.output_root,
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


def _channels(cfg: MechanismConfig) -> list[ChannelData]:
    """Load every selected channel (full-mean GT)."""
    return list(iter_channels(
        cfg.dataset.name, cfg.dataset.subjects, cfg.dataset.emgs,
        data_root=cfg.dataset.data_root, normalization=cfg.dataset.normalization,
    ))


# ---------------------------------------------------------------------------
# update_rule (task #9)
# ---------------------------------------------------------------------------
UPDATE_RULE_DEFAULTS: dict[str, Any] = {
    "engines": ["tabpfn_v2_5", "gp_mll_frozen", "gp_mll_refit"],
    "context_sizes": [10, 25, 50],
    "stress": {"knob": "k2_channel", "levels": [1.0, 2.0, 4.0]},
    "n_context_draws": 3,
    "n_anchors": 12,
    "surprises": [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
    "refit_surprises": [-1.0, -0.5, 0.5, 1.0],
    "gp_params": {"n_opt_steps": 100, "lr": 0.1},
    "inference_seed": 0,
    "max_batch_tokens": 400000,
    "layer_arm": {"enabled": True, "context_t": 25, "level": 1.0, "ridge": 1.0},
    "gates": {
        "positive_r": 0.7, "lengthscale_tol": 0.25, "icc_min": 0.75, "linear_regime_tol": 0.25,
        "null_quantile": 0.95, "context_t": 25,
    },
    "positive_control": {"grid_side": 10, "lengthscale": 0.2, "noise_sd": 0.3, "n_trials": 10},
    "seed_floor": {"n_seeds": 10, "n_channels": 2},
    # Step 7 (M8): join per-channel metrics with a Hyp A/B tidy.csv; skipped when null.
    "link": {
        "tidy_csv": None,
        "model": "tabpfn_v2_5",
        "outcomes": ["recommended_regret", "cumulative_regret"],
        "predictors": ["saturation_index_median", "ell_adaptivity", "rho_shape_median"],
        "n_boot": 2000,
    },
}


def _make_engine(name: str, p: dict[str, Any], device: str, inference_seed: int) -> Any:
    """Build one update engine by name.

    Args:
        name: ``'tabpfn_v2_5'``, ``'gp_mll_frozen'`` or ``'gp_mll_refit'``.
        p: Resolved ``update_rule`` block.
        device: Torch device for TabPFN.
        inference_seed: TabPFN preprocessing seed.

    Returns:
        The engine.
    """
    from ..analysis import update_rule as U  # noqa: PLC0415
    from ..models.gp.surrogates import GPSurrogate  # noqa: PLC0415

    if name == "tabpfn_v2_5":
        return U.PFNEngine(device=device, random_state=inference_seed,
                           max_batch_tokens=int(p["max_batch_tokens"]), name=name)
    if name == "gp_mll_frozen":
        return U.GPFrozenEngine(GPSurrogate(**p["gp_params"]), name=name)
    if name == "gp_mll_refit":
        return U.GPRefitEngine(GPSurrogate(**p["gp_params"]), name=name)
    raise ValueError(f"Unknown update_rule engine {name!r}.")


def run_update_rule(cfg: MechanismConfig, replot: bool = False) -> str:
    """Task #9 Step 4: the M10 grid and its gates.

    Grid: channel x stress level x context size t x context draw x engine; every engine of a
    cell shares one context, one anchor set and one frozen MLL-GP reference.

    Args:
        cfg: Resolved config.
        replot: Rebuild figures from the CSVs only.

    Returns:
        The run directory.
    """
    from ..analysis import update_rule as U  # noqa: PLC0415
    from ..data.snr import achieved_snr_db  # noqa: PLC0415
    from ..data.stress import build_knob  # noqa: PLC0415
    from ..models.gp.surrogates import GPSurrogate, NaiveGPSurrogate  # noqa: PLC0415
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _check_keys(cfg.params, UPDATE_RULE_DEFAULTS, "update_rule")
    p["gates"] = _check_keys(p["gates"], UPDATE_RULE_DEFAULTS["gates"], "update_rule.gates")
    out = cfg.run_dir
    if not replot:
        set_seed(cfg.seed)
        _write_config(cfg, p, out)
        knob = build_knob(p["stress"]["knob"], p["stress"]["levels"])
        anchor_rows: list[dict[str, Any]] = []
        cell_rows: list[dict[str, Any]] = []
        surprise_rows: list[dict[str, Any]] = []
        layer_rows: list[dict[str, Any]] = []
        exemplar: dict[str, Any] | None = None
        channels = _channels(cfg)
        t0 = time.time()
        engines = {name: _make_engine(name, p, cfg.device, int(p["inference_seed"])) for name in p["engines"]}
        for ch in channels:
            for level in knob.levels:
                stressed = knob.apply(ch, level, rng_for(ch.label, "update_rule", level, base_seed=cfg.seed))
                snr = achieved_snr_db(stressed)
                for t in p["context_sizes"]:
                    if t >= stressed.n_sites:
                        continue
                    for draw in range(int(p["n_context_draws"])):
                        rng = rng_for(ch.label, "update_rule", level, t, draw, base_seed=cfg.seed)
                        ctx = U.draw_context(stressed, int(t), rng)
                        anchors, strata = U.select_anchors(stressed, int(p["n_anchors"]), rng, exclude=ctx.sites)
                        reference = U.GPFrozenEngine(GPSurrogate(**p["gp_params"]), name="reference")
                        reference.fit(ctx.X, ctx.y)
                        keys = {
                            "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg,
                            "knob": knob.name, "level": float(level), "achieved_snr_db": snr,
                            "context_t": int(t), "draw": draw,
                        }
                        for name, engine in engines.items():
                            surprises = p["refit_surprises"] if name == "gp_mll_refit" else p["surprises"]
                            res = U.run_probes(engine, reference, ctx, stressed.X_pool, anchors, strata, surprises)
                            rows, cell = U.update_metrics(res, stressed.X_pool)
                            la = p["layer_arm"]
                            if (name == "tabpfn_v2_5" and la["enabled"] and int(t) == int(la["context_t"])
                                    and float(level) == float(la["level"])):
                                layer_rows += [{**keys, **r} for r in U.layer_alignment(
                                    engine, res, stressed.X_pool, ridge=float(la["ridge"]))]
                            anchor_rows += [{**keys, **r} for r in rows]
                            cell_rows.append({**keys, **cell})
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
                            if exemplar is None and (name == "tabpfn_v2_5" or "tabpfn_v2_5" not in engines):
                                exemplar = _exemplar(stressed, res)
                print(f"[mechanism] {ch.label} level={level} done ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(anchor_rows).to_csv(os.path.join(out, "update_rule.csv"), index=False)
        pd.DataFrame(cell_rows).to_csv(os.path.join(out, "update_rule_cell.csv"), index=False)
        pd.DataFrame(surprise_rows).to_csv(os.path.join(out, "update_rule_surprise.csv"), index=False)
        if layer_rows:
            pd.DataFrame(layer_rows).to_csv(os.path.join(out, "update_rule_layers.csv"), index=False)
        if exemplar is not None:
            np.savez(os.path.join(out, "update_rule_exemplar.npz"), **exemplar)

        # --- validation gates (Step 2) ---
        g = p["gates"]
        pc = p["positive_control"]
        truth = lambda: U.GPFrozenEngine(  # noqa: E731
            NaiveGPSurrogate(lengthscale=pc["lengthscale"], outputscale=1.0, noise=pc["noise_sd"] ** 2),
            name="truth",
        )
        gates: list[dict[str, Any]] = []

        def _positive(eng: Any) -> dict[str, Any]:
            return U.positive_control(
                eng, truth(), grid_side=pc["grid_side"], lengthscale=pc["lengthscale"],
                noise_sd=pc["noise_sd"], n_trials=pc["n_trials"], t=g["context_t"],
                n_anchors=int(p["n_anchors"]), surprises=p["surprises"], min_r=g["positive_r"],
                lengthscale_tol=g["lengthscale_tol"], seed=cfg.seed,
            )

        # Probe validity first: the true GP through the probe must pass, otherwise "the probe,
        # not the model, is at fault" and no model result below is interpretable.
        gates.append({**_positive(truth()), "role": "probe_validity"})
        for name in p["engines"]:
            eng = _make_engine(name, p, cfg.device, int(p["inference_seed"]))
            gates.append({**_positive(eng), "role": "model"})
            if channels:
                gates.append({**U.negative_control(
                    eng, U.GPFrozenEngine(GPSurrogate(**p["gp_params"])), channels[0],
                    t=min(g["context_t"], channels[0].n_sites - 1), n_anchors=int(p["n_anchors"]),
                    surprises=p["surprises"], null_quantile=g["null_quantile"], seed=cfg.seed,
                ), "channel": channels[0].label})
            eng_rows = [r for r in anchor_rows if r["engine"] == name]
            gates.append({**U.linear_regime_check(eng_rows, g["linear_regime_tol"]), "engine": name})
        if "tabpfn_v2_5" in p["engines"]:
            for ch in channels[: int(p["seed_floor"]["n_channels"])]:
                gates.append({**U.seed_floor(
                    lambda s: _make_engine("tabpfn_v2_5", p, cfg.device, s),
                    U.GPFrozenEngine(GPSurrogate(**p["gp_params"])), ch,
                    n_seeds=int(p["seed_floor"]["n_seeds"]), t=min(g["context_t"], ch.n_sites - 1),
                    n_anchors=int(p["n_anchors"]), surprises=p["surprises"], icc_min=g["icc_min"],
                    seed=cfg.seed,
                ), "engine": "tabpfn_v2_5", "channel": ch.label})
        with open(os.path.join(out, "gates.json"), "w", encoding="utf-8") as fh:
            json.dump(gates, fh, indent=2, default=_json_default)
        for gate in gates:
            print(f"[mechanism] gate {gate['gate']} {gate.get('engine', '')}: "
                  f"{'PASS' if gate['passed'] else 'FAIL'}", flush=True)
    link = p["link"]
    if link.get("tidy_csv"):
        path = _predictive_link(out, link, cfg.seed)
        print(f"[mechanism] wrote {path}")
    for path in figs.render_update_rule(out):
        print(f"[mechanism] wrote {path}")
    return out


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


def _exemplar(channel: ChannelData, res: Any) -> dict[str, Any]:
    """Arrays of the first anchor for figure F1 (grid heatmaps)."""
    from ..analysis.update_rule import _antisym, _symmetric_magnitudes  # noqa: PLC0415

    c0 = _symmetric_magnitudes(res.surprises)[0]
    return {
        "engine": np.array(res.engine),
        "label": np.array(channel.label),
        "anchor": np.array(int(res.anchors[0])),
        "ch2xy": channel.ch2xy,
        "grid_shape": np.array(channel.grid_shape),
        "g_model": _antisym(res.d_mean, res.surprises, c0)[0],
        "g_gp": _antisym(res.d_gp, res.surprises, c0)[0],
    }


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
    "context": {
        "sizes": [5, 10, 20, 40],
        "n_draws": 10,
        "stress": {"knob": "k2_channel", "levels": [1.0, 2.0, 4.0]},
    },
}


def _placement_banks(
    cfg: MechanismConfig, p: dict[str, Any], channels: list[ChannelData], feat: Callable[..., np.ndarray]
) -> tuple[dict[bytes, dict[str, Any]], list[dict[str, Any]]]:
    """Build the on-grid prior and noise banks once per distinct grid (task #6 Step 1).

    Args:
        cfg: Resolved config.
        p: Resolved ``placement`` block.
        channels: Channels to place.
        feat: Map featurizer.

    Returns:
        ``(banks keyed by grid bytes, bank log rows)``.
    """
    from ..data.references.grid import noise_grid_bank, prior_grid_bank  # noqa: PLC0415

    n_total = int(p["n_prior"]) + int(p["n_prior_holdout"])
    banks: dict[bytes, dict[str, Any]] = {}
    log: list[dict[str, Any]] = []
    for ch in channels:
        key = ch.X_pool.tobytes()
        if key in banks:
            continue
        prior = prior_grid_bank(
            ch.X_pool, n_total, n_dense=int(p["n_dense"]), holdout_frac=float(p["holdout_frac"]),
            rmse_threshold=float(p["rmse_threshold"]), prior_type=p["prior_type"], seed=cfg.seed,
        )
        n_hold = min(int(p["n_prior_holdout"]), len(prior.maps) // 2)
        noise = noise_grid_bank(ch.X_pool, int(p["n_noise"]), seed=cfg.seed + 1)
        banks[key] = {
            "train": [feat(ch.X_pool, m) for m in prior.maps[n_hold:]],
            "holdout": [feat(ch.X_pool, m) for m in prior.maps[:n_hold]],
            "prior_maps": prior.maps,
            "n_hold": n_hold,
            "noise": [feat(ch.X_pool, m) for m in noise.maps],
            "noise_maps": noise.maps,
        }
        finite = [r for r in prior.rmse if np.isfinite(r)]
        log.append({
            "grid_of": ch.label, "n_sites": ch.n_sites, "n_attempted": n_total,
            "n_kept": int(len(prior.maps)), "rejection_rate": prior.rejection_rate,
            "rmse_median": float(np.median(finite)) if finite else None,
            "rmse_threshold": float(p["rmse_threshold"]),
        })
        print(f"[mechanism] prior bank for {ch.label}: kept {len(prior.maps)}/{n_total} "
              f"(rejection {prior.rejection_rate:.2f})", flush=True)
    return banks, log


def run_placement(cfg: MechanismConfig, replot: bool = False) -> str:
    """Task #6: placement of every channel between the prior and noise (B, C, A).

    Args:
        cfg: Resolved config.
        replot: Rebuild figures from the CSVs only.

    Returns:
        The run directory.
    """
    from scipy.stats import spearmanr  # noqa: PLC0415

    from ..analysis import placement as P  # noqa: PLC0415
    from ..data.ground_truth import build_ground_truth  # noqa: PLC0415
    from ..data.references.grid import standardize_map  # noqa: PLC0415
    from ..data.snr import achieved_snr_db  # noqa: PLC0415
    from ..data.stress import build_knob  # noqa: PLC0415
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _check_keys(cfg.params, PLACEMENT_DEFAULTS, "placement")
    out = cfg.run_dir
    if not replot:
        set_seed(cfg.seed)
        channels = _channels(cfg)
        if not channels:
            raise RuntimeError("placement: no channels loaded.")

        def feat(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            return P.map_features(X, y, p["representation"], int(p["n_neighbors"]))

        k = int(p["k_nearest"])
        banks, bank_log = _placement_banks(cfg, p, channels, feat)

        # One fixed parameter set per metric per analysis (Step 2).
        first = banks[channels[0].X_pool.tobytes()]
        metrics: dict[str, Any] = {}
        for name in p["metrics"]:
            if name == "mmd":
                metrics[name] = P.make_metric("mmd", bandwidth=P.median_bandwidth(first["train"], seed=cfg.seed))
            else:
                proj = P.projection_set(first["train"][0].shape[1], int(p["n_projections"]), cfg.seed)
                metrics[name] = P.make_metric("w2", projections=proj, n_repeats=int(p["w2_repeats"]), seed=cfg.seed)

        # M0 known-shift ladder on the first channel (Step 5): a failing metric is flagged invalid.
        gates: list[dict[str, Any]] = []
        ch0 = channels[0]
        for name, m in metrics.items():
            lad = P.known_shift_ladder(
                ch0.X_pool, standardize_map(ch0.y_gt), first["noise_maps"][0], first["train"], m, k,
                p["ladder_lambdas"], featurize=feat,
            )
            gates.append({"gate": "known_shift_ladder", "metric": name, "channel": ch0.label, **lad,
                          "passed": bool(lad["spearman"] > float(p["ladder_min_spearman"]))})
        valid = {g["metric"]: g["passed"] for g in gates}

        # Formulation B with both floors and both ceilings (Steps 3-4).
        rows: list[dict[str, Any]] = []
        ref_cache: dict[tuple[bytes, str], tuple[np.ndarray, np.ndarray]] = {}
        for ch in channels:
            bank = banks[ch.X_pool.tobytes()]
            y = standardize_map(ch.y_gt)
            Z = feat(ch.X_pool, y)
            rng = rng_for(ch.label, "placement", base_seed=cfg.seed)
            halves = build_ground_truth(ch.Y_trials, "split_half", rng, int(p["n_splits"]))
            split_maps = [(halves[r][0], halves[r + 1][0]) for r in range(0, len(halves), 2)]
            X_split = ch.X_pool[np.isfinite(ch.Y_trials).sum(axis=1) >= 2]
            for name, m in metrics.items():
                ck = (ch.X_pool.tobytes(), name)
                if ck not in ref_cache:
                    ref_cache[ck] = (
                        np.array([P.distance_to_bank(h, bank["train"], m, k) for h in bank["holdout"]]),
                        np.array([P.distance_to_bank(nz, bank["train"], m, k) for nz in bank["noise"]]),
                    )
                floor1, ceil1 = ref_cache[ck]
                floor2 = np.array([
                    m(feat(X_split, standardize_map(a)), feat(X_split, standardize_map(b))) for a, b in split_maps
                ])
                ceil2 = np.array([
                    P.distance_to_bank(feat(ch.X_pool, rng.permutation(y)), bank["train"], m, k)
                    for _ in range(int(p["n_shuffles"]))
                ])
                d = P.distance_to_bank(Z, bank["train"], m, k)
                row: dict[str, Any] = {
                    "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg, "metric": name,
                    "metric_valid": valid[name], "representation": p["representation"], "d": d,
                    "floor1": float(np.median(floor1)), "floor2": float(np.median(floor2)),
                    "ceiling1": float(np.median(ceil1)), "ceiling2": float(np.median(ceil2)),
                }
                for fi, fl in (("1", floor1), ("2", floor2)):
                    for ci, ce in (("1", ceil1), ("2", ceil2)):
                        pv, lo, hi = P.bootstrap_placement(d, fl, ce, n_boot=int(p["n_boot"]), seed=cfg.seed)
                        tag = f"f{fi}c{ci}"
                        row[f"p_{tag}"], row[f"p_{tag}_lo"], row[f"p_{tag}_hi"] = pv, lo, hi
                        row[f"gap_{tag}"] = float(np.median(ce) - np.median(fl))
                rows.append(row)
            print(f"[mechanism] placement {ch.label} done", flush=True)
        df = pd.DataFrame(rows)

        # Formulation C: random size-t noisy contexts along the knob, at matched sites (Step 4).
        cp = p["context"]
        knob = build_knob(cp["stress"]["knob"], cp["stress"]["levels"])
        crow: list[dict[str, Any]] = []
        for ch in channels:
            bank = banks[ch.X_pool.tobytes()]
            train_maps = bank["prior_maps"][bank["n_hold"]:]
            hold_maps = bank["prior_maps"][: bank["n_hold"]]
            for level in knob.levels:
                stressed = knob.apply(ch, level, rng_for(ch.label, "placement_c", level, base_seed=cfg.seed))
                snr = achieved_snr_db(stressed)
                for t in list(cp["sizes"]) + ["full"]:
                    tt = ch.n_sites if t == "full" else int(t)
                    if tt > ch.n_sites or tt <= int(p["n_neighbors"]):
                        continue
                    for draw in range(int(cp["n_draws"])):
                        rng = rng_for(ch.label, "placement_c", level, t, draw, base_seed=cfg.seed)
                        sites = np.sort(rng.choice(ch.n_sites, tt, replace=False))
                        yv = np.array([
                            stressed.Y_trials[s, rng.choice(np.flatnonzero(np.isfinite(stressed.Y_trials[s])))]
                            for s in sites
                        ])
                        if np.std(yv) == 0:
                            continue
                        Xs = ch.X_pool[sites]
                        Zc = feat(Xs, standardize_map(yv))

                        def at_sites(maps: np.ndarray) -> list[np.ndarray]:
                            return [feat(Xs, standardize_map(mp[sites])) for mp in maps]

                        ref, hold, noise_s = at_sites(train_maps), at_sites(hold_maps), at_sites(bank["noise_maps"])
                        for name, m in metrics.items():
                            d = P.distance_to_bank(Zc, ref, m, k)
                            fl = float(np.median([P.distance_to_bank(h, ref, m, k) for h in hold]))
                            ce = float(np.median([P.distance_to_bank(nz, ref, m, k) for nz in noise_s]))
                            crow.append({
                                "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg, "metric": name,
                                "knob": knob.name, "level": float(level), "achieved_snr_db": snr,
                                "context_t": tt, "is_full": t == "full", "draw": draw, "d": d,
                                "floor1": fl, "ceiling1": ce, "p": P.placement(d, fl, ce),
                            })

        # Formulation A: pooled marginal of y (appendix, labelled "marginal").
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
        if {"mmd", "w2"} <= set(metrics) and len(channels) >= 3:
            wide = df.pivot_table(index=["subject", "emg"], columns="metric", values="d")
            summary["rho_mmd_w2_channel"] = float(spearmanr(wide["mmd"], wide["w2"]).correlation)
        with open(os.path.join(out, "placement_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=_json_default)
        for g in gates:
            print(f"[mechanism] gate {g['gate']} {g['metric']}: rho={g['spearman']:.2f} "
                  f"{'PASS' if g['passed'] else 'FAIL (metric flagged invalid)'}", flush=True)
    for path in figs.render_placement(out):
        print(f"[mechanism] wrote {path}")
    return out


# ---------------------------------------------------------------------------
# cka (task #5)
# ---------------------------------------------------------------------------
CKA_DEFAULTS: dict[str, Any] = {
    "layers": None,
    "readouts": ["feature_mean"],
    "targets": ["K_X", "K_GT", "K_GT_linear", "K_GP"],
    "context_sizes": [5, 10, 20, 40],
    "stress": {"knob": "k2_channel", "levels": [1.0, 2.0, 4.0]},
    "n_context_draws": 3,
    "n_perm": 1000,
    "inference_seed": 0,
    "gp_params": {"n_opt_steps": 100, "lr": 0.1},
    "controls": {
        "grid_side": 10, "lengthscale": 0.2, "noise_sd": 0.3, "n_trials": 10, "t": 25,
        "alpha": 0.05, "n_seeds": 10, "icc_min": 0.75, "n_seed_channels": 1,
    },
    "placement": {
        "enabled": True, "t": 25, "n_prior": 30, "n_prior_holdout": 10, "n_noise": 10,
        "k_nearest": 5, "n_dense": 1024, "holdout_frac": 0.1, "rmse_threshold": 0.5,
        "prior_type": "prior_bag",
    },
}


def _cka_layer_table(
    E: np.ndarray, kernels: dict[str, np.ndarray], n_perm: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Debiased CKA with a site-permutation null for every (layer, target).

    Args:
        E: Embeddings, shape [L, N, d].
        kernels: Target Gram matrices, each [N, N].
        n_perm: Permutations.
        rng: Generator.

    Returns:
        Rows with ``layer, target, value, null_mean, cka_minus_null, p``.
    """
    from ..analysis.cka import linear_gram, permutation_null  # noqa: PLC0415

    rows = []
    for layer in range(E.shape[0]):
        K = linear_gram(E[layer])
        for target, L in kernels.items():
            obs, null_mean, p = permutation_null(K, L, n_perm, rng)
            rows.append({"layer": layer, "target": target, "value": obs, "null_mean": null_mean,
                         "cka_minus_null": obs - null_mean, "p": p})
    return rows


def run_cka(cfg: MechanismConfig, replot: bool = False) -> str:
    """Task #5: CKA (a) against target kernels, controls, and CKA (b) placement.

    Args:
        cfg: Resolved config.
        replot: Rebuild figures from the CSVs only.

    Returns:
        The run directory.
    """
    from ..analysis import update_rule as U  # noqa: PLC0415
    from ..analysis.cka import cka_debiased, linear_gram, target_kernels  # noqa: PLC0415
    from ..analysis.embeddings import layer_embeddings  # noqa: PLC0415
    from ..analysis.placement import placement as place  # noqa: PLC0415
    from ..data.references.grid import noise_grid_bank, prior_grid_bank  # noqa: PLC0415
    from ..data.snr import achieved_snr_db  # noqa: PLC0415
    from ..data.stress import build_knob  # noqa: PLC0415
    from ..models.gp.surrogates import GPSurrogate  # noqa: PLC0415
    from ..models.pfn.tabpfn import FrozenTabPFN  # noqa: PLC0415
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _check_keys(cfg.params, CKA_DEFAULTS, "cka")
    p["controls"] = _check_keys(p["controls"], CKA_DEFAULTS["controls"], "cka.controls")
    p["placement"] = _check_keys(p["placement"], CKA_DEFAULTS["placement"], "cka.placement")
    n_layers_max = 18 if p["layers"] is None else len(p["layers"])
    if 1.0 / (int(p["n_perm"]) + 1) >= p["controls"]["alpha"] / n_layers_max:
        raise ValueError(
            f"cka.n_perm={p['n_perm']} cannot resolve the Bonferroni threshold "
            f"{p['controls']['alpha']}/{n_layers_max}: the smallest attainable p is 1/(n_perm+1)."
        )
    out = cfg.run_dir
    if not replot:
        set_seed(cfg.seed)
        _write_config(cfg, p, out)
        engine = FrozenTabPFN(device=cfg.device, random_state=int(p["inference_seed"]))

        def embed(X_ctx: np.ndarray, y_ctx: np.ndarray, X_sites: np.ndarray, readout: str,
                  eng: Any = engine) -> np.ndarray:
            eng.fit(X_ctx, y_ctx)
            return layer_embeddings(eng, X_sites, layers=p["layers"], readout=readout)

        def kernels_for(ch: ChannelData, ctx: Any) -> dict[str, np.ndarray]:
            gp = GPSurrogate(**p["gp_params"])
            gp.fit(ctx.X, ctx.y)
            ks = target_kernels(ch.X_pool, ch.y_gt, gp.posterior_covariance(ch.X_pool))
            return {k: v for k, v in ks.items() if k in p["targets"]}

        channels = _channels(cfg)
        knob = build_knob(p["stress"]["knob"], p["stress"]["levels"])
        rows: list[dict[str, Any]] = []
        for ch in channels:
            for level in knob.levels:
                st = knob.apply(ch, level, rng_for(ch.label, "cka", level, base_seed=cfg.seed))
                snr = achieved_snr_db(st)
                for t in p["context_sizes"]:
                    if t >= st.n_sites:
                        continue
                    for rep in range(int(p["n_context_draws"])):
                        rng = rng_for(ch.label, "cka", level, t, rep, base_seed=cfg.seed)
                        ctx = U.draw_context(st, int(t), rng)
                        kernels = kernels_for(st, ctx)
                        for readout in p["readouts"]:
                            E = embed(ctx.X, ctx.y, st.X_pool, readout)
                            for r in _cka_layer_table(E, kernels, int(p["n_perm"]), rng):
                                rows.append({
                                    "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg,
                                    "model": "tabpfn_v2_5", "readout": readout, "context_t": int(t),
                                    "knob": knob.name, "level": float(level), "achieved_snr_db": snr,
                                    "rep": rep, **r,
                                })
            print(f"[mechanism] cka {ch.label} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, "cka.csv"), index=False)

        # --- M0 controls (Step 4) ---
        c = p["controls"]
        gates: list[dict[str, Any]] = []
        n_layers = None
        for label, noise_only in (("positive_control", False), ("negative_control", True)):
            rng = np.random.default_rng(cfg.seed)
            ctrl = U.gp_channel(c["grid_side"], c["lengthscale"], c["noise_sd"], c["n_trials"], rng)
            if noise_only:
                from dataclasses import replace  # noqa: PLC0415

                y = rng.normal(size=ctrl.n_sites)
                ctrl = replace(ctrl, y_gt=y, Y_trials=y[:, None] + c["noise_sd"] * rng.normal(size=ctrl.Y_trials.shape))
            ctx = U.draw_context(ctrl, int(c["t"]), rng)
            E = embed(ctx.X, ctx.y, ctrl.X_pool, p["readouts"][0])
            n_layers = E.shape[0]
            tab = _cka_layer_table(E, {"K_GT": target_kernels(ctrl.X_pool, ctrl.y_gt)["K_GT"]}, int(p["n_perm"]), rng)
            min_p = min(r["p"] for r in tab)
            bonf = c["alpha"] / len(tab)
            gates.append({
                "gate": label, "min_p": min_p, "alpha_bonferroni": bonf,
                "max_cka_minus_null": max(r["cka_minus_null"] for r in tab),
                "passed": bool(min_p < bonf) if not noise_only else bool(min_p >= bonf),
            })
        for ch in channels[: int(c["n_seed_channels"])]:
            ctx = U.draw_context(ch, min(int(c["t"]), ch.n_sites - 1), np.random.default_rng(cfg.seed))
            kernels = kernels_for(ch, ctx)
            per_seed = []
            for s in range(int(c["n_seeds"])):
                E = embed(ctx.X, ctx.y, ch.X_pool, p["readouts"][0],
                          eng=FrozenTabPFN(device=cfg.device, random_state=s))
                per_seed.append([cka_debiased(linear_gram(E[l]), K) for l in range(E.shape[0]) for K in kernels.values()])
            icc = U.icc_oneway(np.asarray(per_seed).T)
            gates.append({"gate": "seed_stability", "channel": ch.label, "n_seeds": int(c["n_seeds"]),
                          "icc": icc, "passed": bool(icc >= c["icc_min"])})

        # --- CKA (b) placement (Step 6) ---
        pl = p["placement"]
        prow: list[dict[str, Any]] = []
        if pl["enabled"]:
            for ch in channels:
                n_total = int(pl["n_prior"]) + int(pl["n_prior_holdout"])
                prior = prior_grid_bank(ch.X_pool, n_total, n_dense=int(pl["n_dense"]),
                                        holdout_frac=float(pl["holdout_frac"]),
                                        rmse_threshold=float(pl["rmse_threshold"]),
                                        prior_type=pl["prior_type"], seed=cfg.seed)
                noise = noise_grid_bank(ch.X_pool, int(pl["n_noise"]), seed=cfg.seed + 1)
                n_hold = min(int(pl["n_prior_holdout"]), len(prior.maps) // 2)
                rng = rng_for(ch.label, "cka_placement", base_seed=cfg.seed)
                ctx = U.draw_context(ch, min(int(pl["t"]), ch.n_sites - 1), rng)
                sites = ctx.sites
                readout = p["readouts"][0]

                def grams(y_map: np.ndarray) -> list[np.ndarray]:
                    E = embed(ch.X_pool[sites], y_map[sites], ch.X_pool, readout)
                    return [linear_gram(E[l]) for l in range(E.shape[0])]

                G_ch = [linear_gram(E) for E in embed(ctx.X, ctx.y, ch.X_pool, readout)]
                bank = [grams(m) for m in prior.maps[n_hold:]]
                hold = [grams(m) for m in prior.maps[:n_hold]]
                nz = [grams(m) for m in noise.maps]
                k = int(pl["k_nearest"])

                def dist(Gs: list[np.ndarray], layer: int) -> float:
                    d = np.sort([1.0 - cka_debiased(Gs[layer], B[layer]) for B in bank])
                    return float(np.median(d[:k]))

                for layer in range(len(G_ch)):
                    d = dist(G_ch, layer)
                    fl = float(np.median([dist(h, layer) for h in hold]))
                    ce = float(np.median([dist(z, layer) for z in nz]))
                    prow.append({"dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg,
                                 "layer": layer, "context_t": len(sites), "d": d, "floor1": fl,
                                 "ceiling1": ce, "gap": ce - fl, "p": place(d, fl, ce),
                                 "prior_rejection_rate": prior.rejection_rate})
                print(f"[mechanism] cka placement {ch.label} done", flush=True)
        pd.DataFrame(prow).to_csv(os.path.join(out, "cka_placement.csv"), index=False)
        with open(os.path.join(out, "gates.json"), "w", encoding="utf-8") as fh:
            json.dump(gates, fh, indent=2, default=_json_default)
        for g in gates:
            print(f"[mechanism] gate {g['gate']}: {'PASS' if g['passed'] else 'FAIL'}", flush=True)
    for path in figs.render_cka(out):
        print(f"[mechanism] wrote {path}")
    return out


#: Analysis runners by name (placement and cka register below as they land).
ANALYSES: dict[str, Callable[[MechanismConfig, bool], str]] = {
    "update_rule": run_update_rule,
    "placement": run_placement,
    "cka": run_cka,
}


def run_mechanism(config_path: str, overrides: list[str] | None = None, *, replot: bool = False) -> str:
    """Load a mechanism config and run its analysis.

    Args:
        config_path: YAML path.
        overrides: ``key=value`` overrides.
        replot: Rebuild figures from existing CSVs.

    Returns:
        The run directory.
    """
    cfg = load_mechanism_config(config_path, overrides)
    return ANALYSES[cfg.analysis](cfg, replot)


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
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_mechanism(args.config, args.overrides, replot=args.replot)
    return 0
