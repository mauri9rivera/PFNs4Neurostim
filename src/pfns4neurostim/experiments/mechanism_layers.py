"""Layer-depth mechanism analysis runner (F6, sprint #3 -- Phase 0 measurement gate + Phase 1 first pass).

Dispatches from ``analysis: layer_dynamics`` in the shared ``mechanism`` CLI (F6-E Step 1: the
fourth mechanism analysis gets its own module rather than growing ``update_rule.py`` further, since
the layer-capture machinery -- hooks, patching -- is a different concern from the probe grid)::

    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_layer_dynamics_nhp.yaml
    python -m pfns4neurostim mechanism --config configs/experiment/mechanism_layer_dynamics_nhp.yaml --only-cached

Reuses :mod:`pfns4neurostim.analysis.update_rule`'s context and anchor machinery (``context_for``,
``select_anchors``, ``run_probes``) and the shared :class:`~pfns4neurostim.experiments.mechanism._Cells`
cache wrapper, under its own cache namespace (``{cell_root}/{dataset}/mechanism_layer_dynamics/``).

**First-pass scoping note.** This recomputes the probe grid rather than reading it back from an
existing ``update_rule`` run's cache cells -- true cross-analysis cache sharing (so a layer-dynamics
run pays nothing for probes an ``update_rule`` run already cached) is a follow-up once the Phase-0
gate (A0-A5) decides whether the probe grid itself needs to change. Probes are cheap relative to the
18-block hooked forward this module adds, so the duplication costs little; it is called out here so
a future change does not assume the sharing already exists.

Covers, of this sprint's Phase 0/1 work: A0 (fact sheet), A2 (robust per-block alignment, written as
``layer_dynamics.csv``) and A3 (the rebuilt, shared-partition, scale-free ridge probe, written as
``layer_probe.csv``, alongside the frozen ``update_rule.layer_alignment`` reference for a
like-for-like C0.c comparison). A1 (the 3-way precision comparison) and B1's richer template bank are
follow-ups once the gate's other items are in; this module exposes the primitives they need
(:mod:`pfns4neurostim.analysis.layer_dynamics`) so adding them is a runner-only change.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from ..analysis import layer_dynamics as L
from ..analysis import update_rule as U
from ..analysis.context import context_for
from ..data.snr import achieved_snr_db
from ..data.stress import build_knob
from ..seeding import rng_for, set_seed
from .mechanism import (
    MechanismConfig,
    RunOptions,
    _Cells,
    _channels,
    _check_keys,
    _engine_params,
    _json_default,
    _make_engine,
    _make_reference,
    _reference_params,
    _subset_of_ladder,
)

__all__ = ["LAYER_DYNAMICS_DEFAULTS", "run_layer_dynamics"]


#: Pre-registered thresholds (frozen before the first F6-0 run; copied verbatim into the F6-4
#: verdict table, never chosen after the fact). See task_plan.md "Conjectures and decision rules"
#: for the derivation of each number; the key names mirror the conjecture ids there.
LAYER_DYNAMICS_GATE_DEFAULTS: dict[str, Any] = {
    "c0a_alignment_delta": 0.05,
    "c0a_cosine_min": 0.9,
    "c0b_margin": 0.10,
    "c0c_spearman_max": -0.1,
    "c0d_rho_shape": 0.065,
    "c0d_ell_hat": 0.33,
    "c0d_far_field_share": 0.135,
    "c0d_locality": 0.20,
    "c1_partial_r_max": 0.1,
    "c1_gp_fixed_delta_max": 0.05,
    "c1_gp_specific_min": 0.2,
    "c1p_segment_gap": 0.15,
    "c2_label_removed_min": 0.5,
    "c2_feature_removed_max": 0.2,
    "c3_axis_delta_r2": 0.05,
    "c3_icc_max": 0.5,
    "c4c_block5_lift": 0.1,
    "c4c_far_field_cut_t10": 0.135,
    "c5_relative_dip_max": 0.25,
    "c5_norm_plateau_dip": 0.57,
    "c6a_block0_delta": 0.2,
    "c6a_mid_block_delta_max": 0.065,
}

LAYER_DYNAMICS_DEFAULTS: dict[str, Any] = {
    "engine": "tabpfn_v2_5",
    "reference_gp": "mll",
    "gp_params": {"n_opt_steps": 100, "lr": 0.1},
    "fixed_gp_params": {"lengthscale": 0.6931471805599453, "outputscale": 1.0, "noise": 0.05},
    "stress": {"knob": "k2_channel", "levels": [1.0]},
    "n_context_draws": 1,
    # 45 uniformly sampled anchors (user decision 2026-10-12), replacing 12 stratified ones. Equal
    # allocation over the four strata is not proportional to their population shares (the perimeter
    # is 37.5 % of an 8x12 grid but took 25 % of the sample), so the unweighted pooled statistic did
    # not estimate the grid-level quantity it was read as. Strata are now recorded per anchor
    # instead (update_rule.stratum_masks), which keeps every conditional contrast available.
    "anchor_selection": "uniform",
    "n_anchors": 45,
    # 'test' is the historical probe path; 'train' replays the uniqueness-checked training-mode
    # fingerprint hash for the injected row only, which is what a refit on context + {(x*, y*)}
    # would assign it (F6-A4). It enters the cell identity, so the two arms cache side by side and
    # the comparison is paired per cell.
    "injection_mode": "test",
    "surprises": [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
    "inference_seed": 0,
    "max_batch_tokens": 400_000,
    # Subset of the shared context ladder (configs/context/<dataset>.yaml); None = all of it.
    "context_ts": None,
    # The rebuilt probe. ``spatial_*`` adds a second fold scheme that holds out contiguous patches
    # of the grid: random site folds put near-neighbours on both sides of the split, and the
    # monotone preference for the weakest penalty measured on 2026-10-09 is what that leakage would
    # look like. One penalty and fewer repeats on the spatial arm, because the contrast of interest
    # is scheme-vs-scheme and a k-means partition of a fixed grid barely moves with its seed.
    "probe": {
        "n_folds": 5, "n_repeats": 10, "lambda_rels": [0.1, 1.0, 10.0],
        "spatial": True, "spatial_n_repeats": 5, "spatial_lambda_rel": 1.0,
    },
    # The prior-kernel control of the geometry test, and the fixed-kernel GP second reference.
    "geometry": {"enabled": True, "fixed_reference": True, "stage_cka": True},
    # S4: the output-space curve, read through the checkpoint's own decoder at every block.
    "lens": {"enabled": True},
    # The two causal tests (C6a block influence, C2 where y* enters). Both need extra forward passes
    # per anchor, so they run on their own smaller subset rather than the full grid: `skip_blocks`
    # costs one pass per block, `swap_blocks` four.
    # C0.a, the precision gate. Two of its three arms need CUDA; the reference arm supplies the
    # injected values so the arms differ only in arithmetic.
    "precision_check": {
        "enabled": False, "n_channels": 6, "context_ts": [10, 25, 50, 80],
        "reference_arm": "cpu_fp32",
    },
    # Which precision the main grid runs at: 'auto' lets TabPFN pick (fp16 autocast on CUDA, fp32 on
    # CPU), 'fp32' forces full precision everywhere. Part of the cell identity.
    "precision": "auto",
    "causal": {
        "enabled": False, "n_channels": 6, "context_ts": [10, 80], "n_anchors": 12,
        "skip_blocks": [0, 1, 2, 5, 12, 13, 14, 15, 16, 17],
        "swap_blocks": [0, 1, 2, 3, 5, 8, 12, 17],
    },
    "gates": LAYER_DYNAMICS_GATE_DEFAULTS,
}


def _fact_sheet(cfg: MechanismConfig, p: dict[str, Any], channel: Any) -> dict[str, Any]:
    """A0: per-dataset facts needed before any F6 number is trusted.

    Args:
        cfg: Resolved mechanism config.
        p: Resolved ``layer_dynamics`` block.
        channel: One channel, used to fit a real engine for the fingerprint/architecture checks.

    Returns:
        The fact sheet (written verbatim to ``fact_sheet.json``).
    """
    from ..models.pfn.tabpfn import FrozenTabPFN  # noqa: PLC0415 - heavy import

    t = min(int(p["context_ts"][0]) if p["context_ts"] else 25, channel.n_sites - 1)
    ctx = context_for(channel, t, 0, base_seed=cfg.seed)
    eng = FrozenTabPFN(device=cfg.device, random_state=int(p["inference_seed"]),
                       max_batch_tokens=int(p["max_batch_tokens"]))
    eng.fit(ctx.X, ctx.y)
    model = eng._model  # noqa: SLF001 - pinned internal (tabpfn 6.3.2)
    n_blocks = len(model.transformer_encoder.layers)
    decoder = model.decoder_dict["standard"]
    probe_x = channel.X_pool[:1]
    Xt_test = eng.transform_x(probe_x).numpy()
    Xt_train = eng.transform_x(probe_x, as_train=True).numpy()
    differing_cols = np.flatnonzero(np.any(np.abs(Xt_test - Xt_train) > 1e-9, axis=0))
    return {
        "n_blocks": int(n_blocks),
        "emsize": int(model.ninp),
        "n_heads_feature_attn": int(model.transformer_encoder.layers[0].self_attn_between_features._nhead),  # noqa: SLF001
        "n_heads_item_attn": int(model.transformer_encoder.layers[0].self_attn_between_items._nhead),  # noqa: SLF001
        "n_preprocessed_features": int(eng.transform_x(probe_x).shape[1]),
        "decoder_shapes": [list(decoder[0].weight.shape)[::-1], list(decoder[2].weight.shape)[::-1]],
        "n_bars": int(eng.criterion.borders.numel() - 1),
        "test_mode_vs_train_mode_fingerprint_columns_differ": differing_cols.tolist(),
        "n_test_mode_vs_train_mode_columns_differing": int(len(differing_cols)),
        "context_t_used": int(t),
    }


def _alignment_row(
    norm: np.ndarray,
    g_gp: np.ndarray,
    anchor: int,
    observed: np.ndarray,
    k_prior: np.ndarray | None,
    g_fixed: np.ndarray | None,
    lens: np.ndarray | None,
    dZ: np.ndarray | None = None,
) -> dict[str, Any]:
    """Every alignment readout of one stage, for one anchor.

    Args:
        norm: Per-site size of the representation change at this stage, shape [M].
        g_gp: The reference GP's signed update at the same sites, shape [M].
        anchor: Anchor site index.
        observed: Context-site mask, shape [M].
        k_prior: The prior (context-blind) kernel slice, or ``None`` to skip the geometry test.
        g_fixed: The fixed-kernel GP's signed update, or ``None``.
        lens: This stage's output-space update from the logit lens, or ``None``.
        dZ: The full per-site representation change, shape [M, d], for the direction-aware
            readouts of C5; ``None`` skips them.

    Returns:
        Flat column -> value mapping (``None`` wherever a statistic is undefined).
    """
    unobs_off = ~np.asarray(observed, dtype=bool)
    unobs_off[anchor] = False
    out: dict[str, Any] = {
        **L.alignment_stats(norm, np.abs(g_gp), anchor=anchor, observed=observed),
        # The statistic the pre-registered rule actually names: sites that are BOTH never observed
        # AND away from the injected site. The two single-exclusion variants disagreed (2026-10-09),
        # which is exactly why the combined one is the one that settles it.
        "pearson_unobserved_off_anchor": (
            L.alignment_stats(norm[unobs_off], np.abs(g_gp[unobs_off]))["pearson"]
            if unobs_off.sum() >= 3 else None
        ),
    }
    if dZ is not None:
        # C5: two direction-aware readouts. Parameter-free post-LN puts every token on a sphere of
        # fixed radius, so `norm` is a rotation magnitude and is blind to the SIGN of the GP update.
        # These recover it: the fraction of across-site variation in dZ carried by a single
        # GP-shaped component, and whether one principal direction tracks the signed update.
        out["vector_cka"] = L.vector_cka(dZ, g_gp)
        out["signed_pc_r"] = L.signed_pc_projection(dZ, g_gp)
    if k_prior is not None:
        out["pearson_prior_kernel"] = L.alignment_stats(norm, np.abs(k_prior))["pearson"]
        out["partial_r_gp_given_prior"] = L.partial_correlation(norm, np.abs(g_gp), np.abs(k_prior))
        out["partial_r_prior_given_gp"] = L.partial_correlation(norm, np.abs(k_prior), np.abs(g_gp))
        # How separable the two templates are at all. The prior and conditioned kernel slices are
        # both "similarity to x*", so they are collinear by construction and conditioning only
        # pulls them apart as the context grows. Without this column a partial correlation near
        # zero cannot be told from a partial correlation that is simply not identified, so it is
        # recorded beside them rather than left for the reader to assume.
        out["kernel_collinearity"] = L.alignment_stats(np.abs(k_prior), np.abs(g_gp))["pearson"]
    if g_fixed is not None:
        out["pearson_gp_fixed"] = L.alignment_stats(norm, np.abs(g_fixed))["pearson"]
    if lens is not None:
        out["lens_pearson"] = L.alignment_stats(np.abs(lens), np.abs(g_gp))["pearson"]
        out["lens_signed_pearson"] = L.alignment_stats(lens, g_gp)["pearson"]
        out["lens_abs_mean"] = float(np.abs(lens).mean())
    return out


def _head_stage_deltas(engine: Any, h1_plus: np.ndarray, h1_minus: np.ndarray) -> dict[str, np.ndarray]:
    """The H1/H2/H3 representation changes, derived from the block-17 label token.

    Everything after block 17 is a deterministic function of that one 192-d vector, so the two
    later head states need no further forward pass: ``H2 = GELU(Linear1(H1))`` and
    ``H3 = Linear2(H2)``, using the checkpoint's own decoder modules.

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        h1_plus: Block-17 label token at the positive surprise, shape [M, 192].
        h1_minus: The same at the negative surprise, shape [M, 192].

    Returns:
        ``{'H1': [M, 192], 'H2': [M, 384], 'H3': [M, n_bars]}`` of antisymmetrised changes.
    """
    import torch  # noqa: PLC0415

    decoder = engine._model.decoder_dict["standard"]  # noqa: SLF001 - pinned internal (tabpfn 6.3.2)
    out: dict[str, np.ndarray] = {"H1": 0.5 * (h1_plus - h1_minus)}
    stages: dict[str, list[np.ndarray]] = {"H2": [], "H3": []}
    for h1 in (h1_plus, h1_minus):
        with torch.no_grad():
            z = torch.as_tensor(np.asarray(h1, dtype=np.float32), device=engine.device)
            h2 = decoder[1](decoder[0](z))                                   # [M, 384] post-GELU
            stages["H2"].append(h2.cpu().numpy().astype(np.float64))
            stages["H3"].append(decoder[2](h2).cpu().numpy().astype(np.float64))  # [M, n_bars]
    for stage, (plus, minus) in stages.items():
        out[stage] = 0.5 * (plus - minus)
    return out


def _fixed_gp_update(
    p: dict[str, Any], ctx: Any, res: Any, Q: np.ndarray, c0: float,
) -> np.ndarray:
    """The fixed-kernel GP's antisymmetrised update for the same probes, shape [A, M].

    The second reference of the geometry test: if the representation change tracks a blind,
    fixed-hyperparameter GP as well as it tracks the fitted one, the agreement is not evidence that
    the network inferred anything about *this* response surface.

    Args:
        p: Resolved ``layer_dynamics`` block.
        ctx: The cell's context.
        res: The probe result (supplies anchors, surprises and the injected values).
        Q: Readout coordinates, shape [M, D].
        c0: Smallest symmetric surprise magnitude.

    Returns:
        The antisymmetrised update at every site, per anchor.
    """
    from ..models.gp.surrogates import NaiveGPSurrogate  # noqa: PLC0415

    gp = NaiveGPSurrogate(**p["fixed_gp_params"])
    gp.fit(ctx.X, ctx.y)
    xs = np.repeat(Q[res.anchors], len(res.surprises), axis=0)               # [A*S, D]
    d_gp, _ = gp.posterior_update(xs, res.y_star.reshape(-1), Q)             # [A*S, M]
    shaped = d_gp.reshape(len(res.anchors), len(res.surprises), len(Q))      # [A, S, M]
    return U._antisym(shaped, res.surprises, c0)  # noqa: SLF001 - [A, M]


def _causal_rows(
    cfg: MechanismConfig,
    p: dict[str, Any],
    cells: _Cells,
    channels: list[ChannelData],
    knob: Any,
    level: float,
) -> list[dict[str, Any]]:
    """C6a (block influence) and C2 (where y\\* enters), on the causal subset.

    Two interventions per anchor, both read in *output* space so the effect is on the model's
    answer rather than on a representation:

    * **skip** -- identity-skip one block and ask how much the update changes. A block the answer
      does not depend on can be removed with little effect; the pre-registered reading is that the
      first block (latent mapping) matters a lot and the middle blocks individually little.
    * **swap** -- exchange the injected row's state at one block between the two surprise signs. If
      that row is still the sole carrier of y\\*, the swap reverses the update
      (``cos = -1``); once the information has reached the query rows, the swap does little
      (``cos = +1``). Sweeping the block dates the hand-off.

    Args:
        cfg: Resolved config.
        p: Resolved ``layer_dynamics`` block.
        cells: Cell namespace.
        channels: Loaded channels.
        knob: Built stress knob.
        level: Stress level to run at.

    Returns:
        One row per (channel, context size, anchor, intervention, block).
    """
    cz = p["causal"]
    rows: list[dict[str, Any]] = []
    for ch in channels[: int(cz["n_channels"])]:
        stressed = knob.apply(ch, level, rng_for(ch.label, "layer_dynamics", level, base_seed=cfg.seed))
        for t in cz["context_ts"]:
            t = int(t)
            if t >= stressed.n_sites:
                continue
            ctx = context_for(stressed, t, 0, base_seed=cfg.seed, level=level)
            rng = rng_for(ch.label, "layer_dynamics_causal", level, t, base_seed=cfg.seed)
            anchors, strata = U.select_anchors_uniform(stressed, int(cz["n_anchors"]), rng,
                                                       exclude=ctx.sites)
            engine = _make_engine(p["engine"], p, cfg.device, int(p["inference_seed"]))
            reference = _make_reference(p)
            reference.fit(ctx.X, ctx.y)
            ident = {"subject": int(ch.subject), "emg": int(ch.emg), "level": level, "context_t": t,
                     "n_anchors": int(cz["n_anchors"]), "surprises": list(p["surprises"]),
                     "engine_params": _engine_params(p["engine"], p), "injection_mode": p["injection_mode"],
                     "reference": _reference_params(p), "skip_blocks": list(cz["skip_blocks"]),
                     "swap_blocks": list(cz["swap_blocks"])}
            keys = {"dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg, "context_t": t,
                    "level": level}

            def compute(engine: Any = engine, reference: Any = reference, ctx: Any = ctx,
                        stressed: ChannelData = stressed, anchors: Any = anchors,
                        strata: Any = strata, keys: dict[str, Any] = keys,
                        ) -> tuple[dict[str, Any], dict[str, Any]]:
                Q = stressed.X_pool
                res = U.run_probes(engine, reference, ctx, Q, anchors, strata, p["surprises"])
                c0 = U._symmetric_magnitudes(res.surprises)[0]  # noqa: SLF001
                i_pos = int(np.flatnonzero(np.isclose(res.surprises, c0))[0])
                i_neg = int(np.flatnonzero(np.isclose(res.surprises, -c0))[0])
                g_gp = U._antisym(res.d_gp, res.surprises, c0)   # noqa: SLF001 - [A, M]
                as_train = p["injection_mode"] == "train"
                out: list[dict[str, Any]] = []
                for i, a in enumerate(res.anchors):
                    xs, yp, yn = Q[a][None], [res.y_star[i, i_pos]], [res.y_star[i, i_neg]]
                    base = {**keys, "anchor": int(a), "stratum": strata[i]}

                    def update(**kw: Any) -> np.ndarray:
                        """Antisymmetrised output-space update under one intervention."""
                        m = [L.patched_forward(engine.engine, Q, X_extra=xs, y_extra=y,
                                               as_train=as_train, **kw) for y in (yp, yn)]
                        return 0.5 * (engine.engine.moments(m[0])[0][0]
                                      - engine.engine.moments(m[1])[0][0])          # [M]

                    plain = update()
                    rho_plain = L.alignment_stats(plain, g_gp[i])["pearson"]
                    for k in cz["skip_blocks"]:
                        sk = update(skip=(int(k),))
                        out.append({**base, "intervention": "skip", "block": int(k),
                                    "rho_shape": L.alignment_stats(sk, g_gp[i])["pearson"],
                                    "rho_shape_plain": rho_plain,
                                    "update_norm_ratio": float(np.linalg.norm(sk)
                                                               / (np.linalg.norm(plain) + 1e-30))})
                    for k in cz["swap_blocks"]:
                        sw = L.swap_injected_row(engine.engine, Q, xs, yp, yn, int(k),
                                                 as_train=as_train)
                        u, q_ = sw["unpatched"], sw["patched"]
                        denom = float(np.linalg.norm(u) * np.linalg.norm(q_)) + 1e-30
                        cos = float(np.dot(u, q_) / denom)
                        out.append({**base, "intervention": "swap", "block": int(k),
                                    "cos_patched_unpatched": cos,
                                    # 1 = the injected row still carries the whole update at this
                                    # block (the swap reverses it); 0 = it carries none of it.
                                    "injected_row_share": 0.5 * (1.0 - cos),
                                    "update_norm_ratio": float(np.linalg.norm(q_)
                                                               / (np.linalg.norm(u) + 1e-30))})
                return {"rows": out}, {}

            hit = cells.run("causal", ident, f"causal {ch.label} t={t}", compute)
            if hit is not None:
                rows += hit[0]["rows"]
        print(f"[mechanism] causal {ch.label} done", flush=True)
    return rows


#: The three arms of the precision comparison, as ``(label, device, precision)``. Two contrasts fall
#: out of three arms: fp16 vs fp32 on the *same* device isolates precision, and fp32 on two devices
#: isolates the device. Changing both at once -- which a plain "re-run it on CPU" would do -- cannot
#: attribute a difference to either (G5).
PRECISION_ARMS: tuple[tuple[str, str, str], ...] = (
    ("cuda_fp16", "cuda", "auto"),
    ("cuda_fp32", "cuda", "fp32"),
    ("cpu_fp32", "cpu", "auto"),
)


def _precision_rows(
    cfg: MechanismConfig,
    p: dict[str, Any],
    cells: _Cells,
    channels: list[ChannelData],
    knob: Any,
    level: float,
) -> list[dict[str, Any]]:
    """C0.a: is the depth profile an artifact of reduced-precision arithmetic?

    The three arms of :data:`PRECISION_ARMS` are run on the *same* contexts, anchors and injected
    values, so the only thing that differs between them is the arithmetic. That last point needs
    care: the injected value is normally placed at the model's own predictive mean plus a multiple
    of its own predictive spread, and those differ slightly between arms -- so letting each arm pick
    its own would confound "precision changes the arithmetic" with "precision changes which probe
    was run". The values are therefore computed once, on the arm the rest of the sprint uses, and
    injected verbatim into all three.

    Pre-registered rule: the conjecture is confirmed -- and every F6 number must be recomputed in
    fp32 -- if any block's median alignment moves by more than 0.05, or if the median cosine between
    the fp16 and fp32 representation changes falls below 0.9.

    Args:
        cfg: Resolved config.
        p: Resolved ``layer_dynamics`` block.
        cells: Cell namespace.
        channels: Loaded channels.
        knob: Built stress knob.
        level: Stress level to run at.

    Returns:
        One row per (channel, context size, anchor, block), carrying every arm's readout side by
        side plus the two cosines.
    """
    import torch  # noqa: PLC0415

    from ..models.pfn.tabpfn import FrozenTabPFN  # noqa: PLC0415 - heavy import

    pz = p["precision_check"]
    if not torch.cuda.is_available():
        raise RuntimeError("layer_dynamics.precision_check needs a CUDA device for two of its three arms.")
    # Built once: each construction loads the checkpoint, and `fit` is the cheap part.
    engines = {
        label: FrozenTabPFN(device=dev, random_state=int(p["inference_seed"]),
                            max_batch_tokens=int(p["max_batch_tokens"]), precision=prec)
        for label, dev, prec in PRECISION_ARMS
    }
    ref_arm = str(pz["reference_arm"])
    rows: list[dict[str, Any]] = []
    for ch in channels[: int(pz["n_channels"])]:
        stressed = knob.apply(ch, level, rng_for(ch.label, "layer_dynamics", level, base_seed=cfg.seed))
        for t in pz["context_ts"]:
            t = int(t)
            if t >= stressed.n_sites:
                continue
            Q = stressed.X_pool
            ctx = context_for(stressed, t, 0, base_seed=cfg.seed, level=level)
            rng = rng_for(ch.label, "layer_dynamics", level, t, 0, base_seed=cfg.seed)
            anchors, strata = U.select_anchors_uniform(stressed, int(p["n_anchors"]), rng,
                                                       exclude=ctx.sites)
            ident = {"subject": int(ch.subject), "emg": int(ch.emg), "level": level, "context_t": t,
                     "n_anchors": int(p["n_anchors"]), "arms": [a[0] for a in PRECISION_ARMS],
                     "reference_arm": ref_arm, "surprises": list(p["surprises"]),
                     "inference_seed": int(p["inference_seed"]), "reference": _reference_params(p)}

            def compute(ch: ChannelData = ch, stressed: ChannelData = stressed, t: int = t,
                        Q: np.ndarray = Q, ctx: Any = ctx, anchors: Any = anchors,
                        strata: Any = strata) -> tuple[dict[str, Any], dict[str, Any]]:
                for eng in engines.values():
                    eng.fit(ctx.X, ctx.y)
                reference = _make_reference(p)
                reference.fit(ctx.X, ctx.y)
                # One shared set of injected values, from the reference arm, so the arms differ
                # only in arithmetic (see the docstring).
                ref = engines[ref_arm]
                base_mean, base_sd = ref.moments(ref.forward(None, None, ref.transform_x(Q)))
                base_mean, base_sd = base_mean[0], base_sd[0]                        # [M], [M]
                c0 = U._symmetric_magnitudes(np.asarray(p["surprises"], dtype=float))[0]  # noqa: SLF001
                out: list[dict[str, Any]] = []
                for i, a in enumerate(anchors):
                    y_hi = float(base_mean[a] + c0 * base_sd[a])
                    y_lo = float(base_mean[a] - c0 * base_sd[a])
                    d_gp, _ = reference.gp.posterior_update(
                        np.repeat(Q[a][None], 2, axis=0), np.array([y_hi, y_lo]), Q)   # [2, M]
                    g_gp = 0.5 * (d_gp[0] - d_gp[1])                                   # [M]
                    dZ = {}
                    for label, eng in engines.items():
                        zp = L.capture_block_states(eng, Q, X_extra=Q[a][None], y_extra=[y_hi])
                        zn = L.capture_block_states(eng, Q, X_extra=Q[a][None], y_extra=[y_lo])
                        dZ[label] = [(0.5 * (zp.feature_tokens[k] - zn.feature_tokens[k])).mean(axis=1)
                                     for k in zp.layers]                              # per block [M, d]
                    for k in range(len(dZ[ref_arm])):
                        row: dict[str, Any] = {
                            "dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg,
                            "context_t": t, "level": level, "anchor": int(a), "stratum": strata[i],
                            "layer": k,
                        }
                        for label in dZ:
                            v = dZ[label][k]                                           # [M, d]
                            row[f"norm_{label}"] = float(np.linalg.norm(v, axis=1).mean())
                            row[f"align_{label}"] = L.alignment_stats(
                                np.linalg.norm(v, axis=1), np.abs(g_gp))["pearson"]
                        row["cos_precision"] = _median_cosine(dZ["cuda_fp16"][k], dZ["cuda_fp32"][k])
                        row["cos_device"] = _median_cosine(dZ["cuda_fp32"][k], dZ["cpu_fp32"][k])
                        out.append(row)
                return {"rows": out}, {}

            hit = cells.run("precision", ident, f"precision {ch.label} t={t}", compute)
            if hit is not None:
                rows += hit[0]["rows"]
        print(f"[mechanism] precision {ch.label} done ({cells.store.host.get('hostname', '?')})", flush=True)
    return rows


def _median_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    """Median over sites of the cosine between two per-site representation changes.

    Args:
        a: Representation change, shape [M, d].
        b: The same quantity from another arm, shape [M, d].

    Returns:
        The median cosine, or ``None`` if either side is degenerate at every site.
    """
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)                      # [M], [M]
    ok = (na > 0) & (nb > 0)
    if not ok.any():
        return None
    return float(np.median((a[ok] * b[ok]).sum(axis=1) / (na[ok] * nb[ok])))


def run_layer_dynamics(cfg: MechanismConfig, opts: RunOptions) -> str:
    """F6-0/F6-1 first pass: fact sheet, per-block alignment and the rebuilt ridge probe.

    Grid: channel x context size t (subset of the shared ladder) x draw 0 (``n_context_draws``
    controls how many draws), one engine's probes read out at every block.

    Args:
        cfg: Resolved mechanism config.
        opts: Cache policy.

    Returns:
        The run directory.
    """
    from ..visualization import mechanism as figs  # noqa: PLC0415

    p = _check_keys(cfg.params, LAYER_DYNAMICS_DEFAULTS, "layer_dynamics")
    if p["engine"] != "tabpfn_v2_5":
        raise ValueError(f"layer_dynamics.engine must be 'tabpfn_v2_5' (block capture needs the transformer), "
                         f"got {p['engine']!r}.")
    if p["anchor_selection"] not in ("uniform", "stratified"):
        raise ValueError(f"layer_dynamics.anchor_selection must be 'uniform' or 'stratified', "
                         f"got {p['anchor_selection']!r}.")
    for sub in ("gates", "probe", "geometry", "lens", "causal", "precision_check"):
        p[sub] = _check_keys(p[sub], LAYER_DYNAMICS_DEFAULTS[sub], f"layer_dynamics.{sub}")
    ladder = cfg.context_sizes
    if p["context_ts"] is None:
        p["context_ts"] = list(ladder)
    _subset_of_ladder(p["context_ts"], ladder, "layer_dynamics.context_ts")
    out = cfg.run_dir
    if opts.replot:
        for path in figs.render_layer_dynamics(out):
            print(f"[mechanism] wrote {path}")
        return out

    set_seed(cfg.seed)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as fh:
        import yaml  # noqa: PLC0415

        yaml.safe_dump({
            "analysis": "layer_dynamics", "tag": cfg.tag,
            "dataset": {"name": cfg.dataset.name, "subjects": list(cfg.dataset.subjects),
                       "emgs": None if cfg.dataset.emgs is None else list(cfg.dataset.emgs)},
            "context": {"sizes": list(ladder)}, "device": cfg.device, "seed": cfg.seed,
            "layer_dynamics": json.loads(json.dumps(p, default=_json_default)),
        }, fh, sort_keys=False)

    cells = _Cells(cfg, opts)
    channels = _channels(cfg)
    if not channels:
        raise RuntimeError("run_layer_dynamics: no channels selected.")
    knob = build_knob(p["stress"]["knob"], p["stress"]["levels"])
    level = float(knob.levels[0])

    fact_sheet = _fact_sheet(cfg, p, channels[0])
    with open(os.path.join(out, "fact_sheet.json"), "w", encoding="utf-8") as fh:
        json.dump(fact_sheet, fh, indent=2, default=_json_default)
    print(f"[mechanism] wrote {os.path.join(out, 'fact_sheet.json')}", flush=True)

    align_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    n_folds, n_repeats = int(p["probe"]["n_folds"]), int(p["probe"]["n_repeats"])
    lambda_rels = [float(v) for v in p["probe"]["lambda_rels"]]
    geom = p["geometry"]
    stage_cka = bool(geom["stage_cka"])
    if p["injection_mode"] not in ("test", "train"):
        raise ValueError(f"layer_dynamics.injection_mode must be 'test' or 'train', "
                         f"got {p['injection_mode']!r}.")
    as_train = p["injection_mode"] == "train"
    t0 = time.time()

    for ch in channels:
        stressed = knob.apply(ch, level, rng_for(ch.label, "layer_dynamics", level, base_seed=cfg.seed))
        snr = achieved_snr_db(stressed)
        for t in p["context_ts"]:
            t = int(t)
            if t >= stressed.n_sites:
                continue
            for draw in range(int(p["n_context_draws"])):
                ctx = context_for(stressed, t, draw, base_seed=cfg.seed, level=level)
                rng = rng_for(ch.label, "layer_dynamics", level, t, draw, base_seed=cfg.seed)
                select = (U.select_anchors_uniform if p["anchor_selection"] == "uniform"
                          else U.select_anchors)
                anchors, strata = select(stressed, int(p["n_anchors"]), rng, exclude=ctx.sites)
                engine = _make_engine(p["engine"], p, cfg.device, int(p["inference_seed"]))
                reference = _make_reference(p)
                reference.fit(ctx.X, ctx.y)
                ident = {
                    "subject": int(ch.subject), "emg": int(ch.emg), "knob": knob.name,
                    "level": level, "context_t": t, "draw": draw, "n_anchors": int(p["n_anchors"]),
                    "anchor_selection": p["anchor_selection"],
                    "injection_mode": p["injection_mode"],
                    # Recorded only when it is not the default, so adding this key does not orphan
                    # cells computed before it existed (the same reason `injection_mode` defaults
                    # inside `_make_engine`). 'auto' is what every run so far used.
                    **({} if p["precision"] == "auto" else {"precision": p["precision"]}),
                    "surprises": list(p["surprises"]), "engine": p["engine"],
                    "engine_params": _engine_params(p["engine"], p), "reference": _reference_params(p),
                    "probe": p["probe"], "geometry": geom, "lens": p["lens"],
                    "fixed_gp_params": p["fixed_gp_params"] if geom["fixed_reference"] else None,
                }
                keys = {"dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg, "level": level,
                       "achieved_snr_db": snr, "context_t": t, "draw": draw}

                def compute(engine: Any = engine, reference: Any = reference, ctx: Any = ctx,
                            anchors: Any = anchors, strata: Any = strata, keys: dict[str, Any] = keys,
                            ) -> tuple[dict[str, Any], dict[str, Any]]:
                    Q = stressed.X_pool
                    res = U.run_probes(engine, reference, ctx, Q, anchors, strata, p["surprises"])
                    c0 = U._symmetric_magnitudes(res.surprises)[0]  # noqa: SLF001
                    i_pos = int(np.flatnonzero(np.isclose(res.surprises, c0))[0])
                    i_neg = int(np.flatnonzero(np.isclose(res.surprises, -c0))[0])
                    g_pfn = U._antisym(res.d_mean, res.surprises, c0)   # noqa: SLF001 - [A, M]
                    g_gp = U._antisym(res.d_gp, res.surprises, c0)      # noqa: SLF001 - [A, M]
                    observed = np.zeros(len(Q), dtype=bool)
                    observed[ctx.sites] = True
                    ell = np.asarray(reference.gp.hyperparameters()["lengthscale"], dtype=np.float64).ravel()
                    g_fixed = _fixed_gp_update(p, ctx, res, Q, c0) if geom["fixed_reference"] else None
                    align_out: list[dict[str, Any]] = []
                    probe_out: list[dict[str, Any]] = []
                    for i, a in enumerate(res.anchors):
                        Zp = L.capture_block_states(engine.engine, Q, X_extra=Q[a][None],
                                                    y_extra=[res.y_star[i, i_pos]], as_train=as_train)
                        Zn = L.capture_block_states(engine.engine, Q, X_extra=Q[a][None],
                                                    y_extra=[res.y_star[i, i_neg]], as_train=as_train)
                        lens = (L.logit_lens(engine.engine, Zp.label_token, Zn.label_token)
                                if p["lens"]["enabled"] else None)                                  # [L, M]
                        k_prior = L.prior_kernel_slice(Q, int(a), ell) if geom["enabled"] else None  # [M]
                        fold_seed = int(rng_for(ch.label, "layer_dynamics_probe_folds", level, t, draw,
                                                int(a), base_seed=cfg.seed).integers(2**31))
                        schemes = {"random": L.make_repeated_kfold(
                            len(Q), n_splits=n_folds, n_repeats=n_repeats, seed=fold_seed)}
                        if p["probe"]["spatial"]:
                            schemes["spatial"] = L.make_spatial_kfold(
                                Q, n_splits=n_folds, n_repeats=int(p["probe"]["spatial_n_repeats"]),
                                seed=fold_seed)
                        row = {**keys, "anchor": int(a), "stratum": strata[i]}
                        # C6b: where consecutive blocks are least alike marks a change of character
                        # in the representation. Computed on the representations themselves (the
                        # +c run's), not on the probe-induced change in them.
                        adj = (L.adjacent_block_cka([Zp.feature_tokens[k].mean(axis=1) for k in Zp.layers])
                               if stage_cka else [None] * (len(Zp.layers) - 1))
                        for layer in Zp.layers:
                            dZ_mean = (0.5 * (Zp.feature_tokens[layer]
                                              - Zn.feature_tokens[layer])).mean(axis=1)             # [M, d]
                            norm = np.linalg.norm(dZ_mean, axis=1)                                  # [M]
                            align_out.append({
                                **row, "stage": f"block{layer}", "layer": layer,
                                "common_mode_share": L.common_mode_share(dZ_mean),
                                "cka_next_block": adj[layer] if layer < len(adj) else None,
                                **_alignment_row(norm, g_gp[i], int(a), observed, k_prior,
                                                 None if g_fixed is None else g_fixed[i],
                                                 None if lens is None else lens[layer], dZ=dZ_mean),
                            })
                            for lam in lambda_rels:
                                probe_out.append({
                                    **row, "layer": layer, "fold_scheme": "random", "lambda_rel": lam,
                                    "ridge_r2": L.scale_free_ridge_r2(dZ_mean, g_pfn[i],
                                                                      schemes["random"], lambda_rel=lam)})
                            if "spatial" in schemes:
                                lam_s = float(p["probe"]["spatial_lambda_rel"])
                                probe_out.append({
                                    **row, "layer": layer, "fold_scheme": "spatial", "lambda_rel": lam_s,
                                    "ridge_r2": L.scale_free_ridge_r2(dZ_mean, g_pfn[i],
                                                                      schemes["spatial"], lambda_rel=lam_s)})
                        # H1 / H2 / H3: deterministic functions of the block-17 label token, so they need
                        # no extra forward pass. Same y-axis as the blocks (norm alignment), appended as
                        # three further x positions -- and H2 is the first state not under a parameter-free
                        # LayerNorm, where the length of a change is meaningful again.
                        for stage, dH in _head_stage_deltas(engine.engine, Zp.label_token[-1],
                                                            Zn.label_token[-1]).items():
                            align_out.append({
                                **row, "stage": stage, "layer": None,
                                "common_mode_share": L.common_mode_share(dH),
                                "cka_next_block": None,
                                **_alignment_row(np.linalg.norm(dH, axis=1), g_gp[i], int(a), observed,
                                                 k_prior, None if g_fixed is None else g_fixed[i], None,
                                                 dZ=dH),
                            })
                    return {"align": align_out, "probe": probe_out}, {}

                hit = cells.run("block_probe", ident, f"{ch.label} t={t} d={draw}", compute)
                if hit is not None:
                    align_rows += hit[0]["align"]
                    probe_rows += hit[0]["probe"]
        print(f"[mechanism] {ch.label} done ({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(align_rows).to_csv(os.path.join(out, "layer_dynamics.csv"), index=False)
    pd.DataFrame(probe_rows).to_csv(os.path.join(out, "layer_probe.csv"), index=False)
    if p["causal"]["enabled"]:
        causal = _causal_rows(cfg, p, cells, channels, knob, level)
        if causal:
            pd.DataFrame(causal).to_csv(os.path.join(out, "layer_causal.csv"), index=False)
    if p["precision_check"]["enabled"]:
        prec = _precision_rows(cfg, p, cells, channels, knob, level)
        if prec:
            pd.DataFrame(prec).to_csv(os.path.join(out, "layer_precision.csv"), index=False)
    cells.report()
    for path in figs.render_layer_dynamics(out):
        print(f"[mechanism] wrote {path}")
    return out
