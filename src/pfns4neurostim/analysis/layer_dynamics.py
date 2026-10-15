"""Layer-depth mechanism analysis of the TabPFN update rule (F6, sprint #3).

Explains the block-by-block profile of :func:`pfns4neurostim.analysis.update_rule.layer_alignment`
(F6): representation-change alignment with the GP update jumps at block 1, dips at block 5, and
recovers with depth. This module holds the *measurement primitives* -- hooks into a fitted
:class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN` and the statistics read from them -- used by
:mod:`pfns4neurostim.experiments.mechanism_layers` (the cached-cell runner) across Phases 0-3 of the
sprint (``.research_assistant/task_plan.md``).

**First-pass scope.** This module covers Phase 0's measurement gate (A0-A5) and the
representation-level statistics of Phase 1 (B1, B4) and Phase 3 (S1, S4 preview): capturing block
and head states, a patching primitive for the C0.e bit-identity test and later causal work, the
rebuilt (shared-partition, scale-free, dual-capable) ridge probe of A3, and the alignment / template
/ common-mode / CKA / decoder-projection / sigma statistics of B1, B4 and H5. The attention-map
captures of B2/B3 (``capture_item_attention``, ``capture_feature_attention``) are Phase 1 work that
depends on the F6-0 gate passing first and are deferred to when that phase starts.

**Output-space vs diagnostic (G6).** The label-token and block-ℓ representations this module
captures are read for *diagnostic* statistics only (patching, logit-lens projections); only
output-space quantities derived from them (``rho_shape_ℓ``, ``gain_ℓ``, H5's sigma alignment) are
reportable in F6 and its tables, per the roadmap guardrail split on 2026-10-08/09.

Pinned to ``tabpfn==6.3.2`` internals (the same contract as
:class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN` and
:mod:`pfns4neurostim.analysis.embeddings`): every block is a ``PerFeatureEncoderLayer`` whose
forward-hook output is already its post-LN state (no separate pre-LN state exists -- ``layer.py``
raises if ``pre_norm=True``, which this checkpoint never sets), and the prediction head is
``decoder_dict['standard'] == Sequential(Linear(192, 384), GELU(), Linear(384, n_out))``
(``transformer.py:255-263``) applied to the block-17 test label token (``transformer.py:568``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import spearmanr

from .cka import cka_debiased, linear_gram
from .update_rule import _pearson, conditioned_profile, site_distances

__all__ = [
    "BlockStates",
    "capture_block_states",
    "HeadStates",
    "capture_head_states",
    "patched_forward",
    "alignment_stats",
    "make_repeated_kfold",
    "make_spatial_kfold",
    "scale_free_ridge_r2",
    "template_bank",
    "common_mode_share",
    "vector_cka",
    "decoder_projection",
    "symmetric_sigma_alignment",
    "partial_correlation",
    "prior_kernel_slice",
    "logit_lens",
    "signed_pc_projection",
    "adjacent_block_cka",
    "injected_row_slice",
    "swap_injected_row",
]


# ---------------------------------------------------------------------------
# A0/A1/A2/C0.e/P1 primitives: hook a fitted FrozenTabPFN
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlockStates:
    """Per-block post-LN states of the query (test) rows, both token kinds.

    Attributes:
        layers: Block indices captured, in hook order.
        feature_tokens: Per block, feature-group tokens, float64 ``[q, n_tok, d]``.
        label_token: Per block, the label token (diagnostic only, G6), float64 ``[q, d]``.
    """

    layers: tuple[int, ...]
    feature_tokens: tuple[np.ndarray, ...]
    label_token: tuple[np.ndarray, ...]


def _run_forward(
    engine: Any,
    X_query: np.ndarray,
    X_extra: np.ndarray | None,
    y_extra: np.ndarray | None,
    as_train: bool = False,
) -> Any:
    """Shared single-probe forward: context (already fitted into ``engine``) + one extra row.

    ``as_train`` applies only to the injected row, matching
    :class:`~pfns4neurostim.analysis.update_rule.PFNEngine`'s ``injection_mode``: the readout rows
    are genuine test rows and always take the test preprocessing path. Keeping the two in step
    matters because the probe result and the captured states must describe the *same* forward pass.
    """
    import torch  # noqa: PLC0415

    Qt = engine.transform_x(X_query)
    if X_extra is None:
        return engine.forward(None, None, Qt)
    Xe = engine.transform_x(np.atleast_2d(X_extra), as_train=as_train).unsqueeze(0)  # [1, r, F]
    ye = engine.transform_y(
        torch.as_tensor(np.asarray(y_extra, dtype=np.float32), device=engine.device)
    ).reshape(1, -1)  # [1, r]
    return engine.forward(Xe, ye, Qt)


def capture_block_states(
    engine: Any,
    X_query: np.ndarray,
    *,
    X_extra: np.ndarray | None = None,
    y_extra: np.ndarray | None = None,
    layers: Sequence[int] | None = None,
    as_train: bool = False,
) -> BlockStates:
    """Hook every requested block in one forward pass, keeping both token kinds.

    Generalises :func:`pfns4neurostim.analysis.embeddings.readout_embeddings` (which keeps the
    feature-token mean only) to also keep the label token per block -- F6's logit-lens readout
    (S4) and C0.e's block-17 bit-identity check both need it.

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        X_query: Query coordinates (raw units, as passed to ``fit``), shape [q, D].
        X_extra: Extra context row(s) injected for this probe, shape [r, D], or None.
        y_extra: Their targets (raw units), shape [r], or None.
        layers: Block indices; ``None`` = all.

    Returns:
        Post-LN states of every hooked block, query rows only.

    Raises:
        RuntimeError: If a hook does not fire.
    """
    import torch  # noqa: PLC0415

    blocks = engine._model.transformer_encoder.layers  # noqa: SLF001 - pinned internal (tabpfn 6.3.2)
    idx = list(range(len(blocks))) if layers is None else [int(i) for i in layers]
    q = len(X_query)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(k: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            act = out if isinstance(out, torch.Tensor) else out[0]   # [B, rows, tokens, d]
            captured[k] = act[0, -q:].detach().float().cpu()         # [q, tokens, d]
        return hook

    handles = [blocks[k].register_forward_hook(make_hook(k)) for k in idx]
    try:
        _run_forward(engine, X_query, X_extra, y_extra, as_train)
    finally:
        for h in handles:
            h.remove()
    missing = [k for k in idx if k not in captured]
    if missing:
        raise RuntimeError(f"capture_block_states: hooks on blocks {missing} did not fire.")
    feature_tokens = tuple(captured[k][:, :-1].numpy().astype(np.float64) for k in idx)
    label_token = tuple(captured[k][:, -1].numpy().astype(np.float64) for k in idx)
    return BlockStates(layers=tuple(idx), feature_tokens=feature_tokens, label_token=label_token)


@dataclass(frozen=True)
class HeadStates:
    """Prediction-head readouts of the query rows (H1-H3; output-space downstream only, G6).

    Attributes:
        h1: Block-17 test label token (post-LN), the decoder's input, shape ``[q, 192]``.
        h2: Post-GELU decoder hidden state, shape ``[q, 384]``.
        h3: Logits before the temperature division, shape ``[q, n_out]``.
    """

    h1: np.ndarray
    h2: np.ndarray
    h3: np.ndarray


def capture_head_states(
    engine: Any,
    X_query: np.ndarray,
    *,
    X_extra: np.ndarray | None = None,
    y_extra: np.ndarray | None = None,
    as_train: bool = False,
) -> HeadStates:
    """Hook the prediction head (``decoder_dict['standard']``) in one forward pass.

    Pinned to ``tabpfn==6.3.2``: the decoder is called as ``Sequential(Linear(192, 384), GELU(),
    Linear(384, n_out))`` on a seq-first ``[q, b, *]`` tensor (``transformer.py:567-572``), so every
    capture here indexes batch position 0 along dim 1, not dim 0.

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        X_query: Query coordinates, shape [q, D].
        X_extra: Extra context row(s), shape [r, D], or None.
        y_extra: Their targets, shape [r], or None.

    Returns:
        The three head readouts.

    Raises:
        RuntimeError: If a hook does not fire.
    """
    import torch  # noqa: PLC0415

    seq = engine._model.decoder_dict["standard"]  # noqa: SLF001 - pinned internal (tabpfn 6.3.2)
    captured: dict[str, torch.Tensor] = {}

    def pre_hook_0(_mod: Any, inp: Any) -> None:
        captured["h1"] = inp[0][:, 0].detach().float().cpu()   # [q, 192]

    def pre_hook_2(_mod: Any, inp: Any) -> None:
        captured["h2"] = inp[0][:, 0].detach().float().cpu()   # [q, 384]

    def hook_2(_mod: Any, _inp: Any, out: Any) -> None:
        captured["h3"] = out[:, 0].detach().float().cpu()      # [q, n_out]

    handles = [
        seq[0].register_forward_pre_hook(pre_hook_0),
        seq[2].register_forward_pre_hook(pre_hook_2),
        seq[2].register_forward_hook(hook_2),
    ]
    try:
        _run_forward(engine, X_query, X_extra, y_extra, as_train)
    finally:
        for h in handles:
            h.remove()
    missing = [k for k in ("h1", "h2", "h3") if k not in captured]
    if missing:
        raise RuntimeError(f"capture_head_states: hooks {missing} did not fire.")
    return HeadStates(**{k: v.numpy().astype(np.float64) for k, v in captured.items()})


def patched_forward(
    engine: Any,
    X_query: np.ndarray,
    *,
    X_extra: np.ndarray | None = None,
    y_extra: np.ndarray | None = None,
    patches: dict[int, Callable[[Any], Any]] | None = None,
    as_train: bool = False,
    skip: Sequence[int] = (),
    row_patches: dict[int, tuple[int, int, Any]] | None = None,
) -> Any:
    """Run one forward pass with selected blocks' query-row output replaced.

    A patch ``patches[k]`` receives the hooked block's query-row slice, shape ``[q, tokens, d]``
    (feature tokens then the label token, matching :func:`capture_block_states`), and must return a
    tensor of the same shape; it replaces that slice in the block's output before the next block
    runs, so every later block recomputes from the patched state. An identity patch (``lambda z:
    z``) must reproduce the unpatched forward exactly, and a patch that replaces block 17's
    feature-token columns ``z[:, :-1]`` while leaving its label-token column ``z[:, -1]`` untouched
    must leave the logits bit-identical, since the block-17 feature tokens never reach the decoder
    (C0.e).

    Three interventions, each on its own argument:

    * ``patches`` replaces a block's **query-row** output (the slice above).
    * ``skip`` identity-skips a whole block: its output is replaced by its own input, for every row,
      so the block contributes nothing. This is the block-influence test of C6a -- how much the
      final answer depends on each block being there at all.
    * ``row_patches`` replaces an arbitrary **row range** of a block's output with a supplied
      tensor, which is what the swapped-pair path patching of P1/P2 needs. Rows at a block are laid
      out ``[thinking (64), context (n), injected (r), query (q)]``, so the injected rows occupy
      ``[-(q + r), -q)``; :func:`injected_row_slice` computes that span rather than leaving callers
      to rederive it.

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        X_query: Query coordinates, shape [q, D].
        X_extra: Extra context row(s), shape [r, D], or None.
        y_extra: Their targets, shape [r], or None.
        patches: Block index -> replacement function for the query rows.
        as_train: Preprocess the injected row on the training path (see :func:`_run_forward`).
        skip: Block indices to identity-skip entirely.
        row_patches: Block index -> ``(start, stop, value)``; the block's output rows in
            ``[start, stop)`` are replaced by ``value``, which must match that slice's shape.

    Returns:
        Log-probabilities over the raw-space bar distribution, as
        :meth:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN.forward` (shape ``[1, q, bars]``).
    """
    import torch  # noqa: PLC0415

    blocks = engine._model.transformer_encoder.layers  # noqa: SLF001 - pinned internal (tabpfn 6.3.2)
    q = len(X_query)
    patches = patches or {}
    row_patches = row_patches or {}

    def split(out: Any) -> tuple[bool, Any]:
        is_tensor = isinstance(out, torch.Tensor)
        return is_tensor, (out if is_tensor else out[0])

    def rebuild(out: Any, is_tensor: bool, act: Any) -> Any:
        return act if is_tensor else (act, *tuple(out[1:]))

    def query_hook(fn: Callable[[Any], Any]) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> Any:
            is_tensor, act = split(out)
            patched = act.clone()
            patched[0, -q:] = fn(act[0, -q:])
            return rebuild(out, is_tensor, patched)
        return hook

    def rows_hook(start: int, stop: int, value: Any) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> Any:
            is_tensor, act = split(out)
            patched = act.clone()
            patched[0, start:stop] = value.to(patched.dtype)
            return rebuild(out, is_tensor, patched)
        return hook

    def skip_hook(_mod: Any, inp: Any, out: Any) -> Any:
        """Return the block's own input, so the block is a no-op for every row."""
        is_tensor, _ = split(out)
        return rebuild(out, is_tensor, inp[0])

    handles = [blocks[k].register_forward_hook(query_hook(fn)) for k, fn in patches.items()]
    handles += [blocks[k].register_forward_hook(rows_hook(*spec)) for k, spec in row_patches.items()]
    handles += [blocks[int(k)].register_forward_hook(skip_hook) for k in skip]
    try:
        return _run_forward(engine, X_query, X_extra, y_extra, as_train)
    finally:
        for h in handles:
            h.remove()


def injected_row_slice(n_query: int, n_injected: int = 1) -> tuple[int, int]:
    """Row span of the injected probe row(s) at any block's output, as ``(start, stop)``.

    Rows are laid out ``[thinking, context, injected, query]`` and the thinking rows are prepended
    inside the model, so counting from the *end* is the only stable way to address the injected
    rows: they sit immediately before the query rows.

    Args:
        n_query: Number of query (readout) rows.
        n_injected: Number of injected rows.

    Returns:
        ``(start, stop)`` as negative indices suitable for ``act[0, start:stop]``.
    """
    return -(n_query + n_injected), -n_query


def swap_injected_row(
    engine: Any,
    X_query: np.ndarray,
    X_extra: np.ndarray,
    y_plus: Sequence[float],
    y_minus: Sequence[float],
    block: int,
    *,
    as_train: bool = False,
) -> dict[str, np.ndarray]:
    """Swapped-pair patch of the injected row's state at one block (P1: where does y\\* enter?).

    Runs the two surprise signs, then re-runs each with the *other* sign's injected-row state
    substituted at ``block``'s output. The swap is symmetric on purpose: patching one side only
    would leave a half-difference whose symmetric component contaminates every "share removed"
    readout, whereas swapping both keeps the patched update exactly antisymmetric.

    Reading it: if the injected row's state at ``block`` is still the only carrier of y\\*, swapping
    it there reverses the downstream update, and the patched update is close to the negation of the
    unpatched one. Once the information has already spread to the query rows, swapping the source
    row changes little. Sweeping ``block`` over the first few blocks therefore dates the hand-off.

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        X_query: Query coordinates, shape [q, D].
        X_extra: The injected row(s), shape [r, D].
        y_plus: Injected target(s) at the positive surprise, length r.
        y_minus: The same at the negative surprise, length r.
        block: Block whose injected-row output is swapped.
        as_train: Preprocess the injected row on the training path.

    Returns:
        ``{'unpatched': [q], 'patched': [q]}`` antisymmetrised output-space updates.
    """
    import torch  # noqa: PLC0415

    q, r = len(X_query), len(np.atleast_2d(X_extra))
    start, stop = injected_row_slice(q, r)
    captured: dict[str, Any] = {}

    def grab(tag: str) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            act = out if isinstance(out, torch.Tensor) else out[0]
            captured[tag] = act[0, start:stop].detach().clone()
        return hook

    blocks = engine._model.transformer_encoder.layers  # noqa: SLF001 - pinned internal (tabpfn 6.3.2)
    means: dict[str, np.ndarray] = {}
    for tag, y in (("plus", y_plus), ("minus", y_minus)):
        handle = blocks[int(block)].register_forward_hook(grab(tag))
        try:
            lp = _run_forward(engine, X_query, X_extra, y, as_train)
        finally:
            handle.remove()
        means[tag] = engine.moments(lp)[0][0]                                   # [q]
    for tag, y, other in (("plus", y_plus, "minus"), ("minus", y_minus, "plus")):
        lp = patched_forward(engine, X_query, X_extra=X_extra, y_extra=y, as_train=as_train,
                             row_patches={int(block): (start, stop, captured[other])})
        means[f"{tag}_patched"] = engine.moments(lp)[0][0]                      # [q]
    return {
        "unpatched": 0.5 * (means["plus"] - means["minus"]),
        "patched": 0.5 * (means["plus_patched"] - means["minus_patched"]),
    }


# ---------------------------------------------------------------------------
# A2 / B4: alignment statistics
# ---------------------------------------------------------------------------
def alignment_stats(
    delta_z_norm: np.ndarray,
    delta_gp_abs: np.ndarray,
    *,
    anchor: int | None = None,
    observed: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Pearson/Spearman alignment of one block's representation change with the GP update (A2).

    Args:
        delta_z_norm: Per-site norm of the block's representation change, shape [M].
        delta_gp_abs: Per-site ``|Delta_GP|`` at the same sites, shape [M].
        anchor: Anchor site index; when given, also reports the off-anchor variant (C0.b).
        observed: Context-site mask, shape [M]; when given, also reports the unobserved-only
            variant (C0.b).

    Returns:
        ``{'pearson', 'spearman'[, 'pearson_off_anchor'][, 'pearson_unobserved']}``.
    """
    z = np.asarray(delta_z_norm, dtype=np.float64)
    g = np.asarray(delta_gp_abs, dtype=np.float64)
    out: dict[str, float | None] = {
        "pearson": _pearson(z, g),
        "spearman": float(spearmanr(z, g).statistic) if np.std(z) > 0 and np.std(g) > 0 else None,
    }
    if anchor is not None:
        mask = np.ones(len(z), dtype=bool)
        mask[int(anchor)] = False
        out["pearson_off_anchor"] = _pearson(z[mask], g[mask])
    if observed is not None:
        unobs = ~np.asarray(observed, dtype=bool)
        out["pearson_unobserved"] = _pearson(z[unobs], g[unobs]) if unobs.any() else None
    return out


# ---------------------------------------------------------------------------
# A3: rebuilt scale-free ridge probe (shared partition, training-fold centering, dual form)
# ---------------------------------------------------------------------------
def make_repeated_kfold(
    n: int,
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...]:
    """``n_repeats`` independent ``n_splits``-fold partitions of ``range(n)``.

    Drawn ONCE per anchor and shared by every block and head stage (A3): the block axis is the
    figure's entire contrast, so re-drawing folds per block -- as the frozen
    :func:`pfns4neurostim.analysis.update_rule.layer_alignment` reference does -- adds an
    independent noise term per point, in the worst regime for it (``d=192`` against ``~77``
    training rows).

    Args:
        n: Number of sites.
        n_splits: Folds per repeat.
        n_repeats: Independent partitions.
        seed: RNG seed (draw from ``seeding.rng_for``, not a hardcoded constant).

    Returns:
        ``n_repeats`` tuples of ``n_splits`` ``(train_idx, val_idx)`` pairs.
    """
    rng = np.random.default_rng(seed)
    repeats = []
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        folds = np.array_split(perm, n_splits)
        splits = tuple(
            (np.concatenate([folds[j] for j in range(n_splits) if j != k]), folds[k])
            for k in range(n_splits)
        )
        repeats.append(splits)
    return tuple(repeats)


def make_spatial_kfold(
    coords: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...]:
    """Like :func:`make_repeated_kfold`, but each held-out group is a contiguous patch of the grid.

    The representation change is spatially smooth over the electrode grid, so a *random* split of
    sites puts near-neighbours on both sides of it and a weakly penalised probe can interpolate
    across the split rather than generalise. Holding out contiguous patches (k-means on the site
    coordinates, one patch per fold) removes that route. This changes what the score means -- it is
    now "predict a region the probe has never seen" rather than "predict a site among its own
    neighbours" -- so it is reported as a second fold scheme beside the random one, never as a
    replacement for it.

    Each repeat projects the sites onto a random direction and cuts the ordering into ``n_splits``
    contiguous bands. Bands rather than blobs on purpose: for a given held-out area a band has less
    perimeter than a compact patch, so fewer held-out sites sit adjacent to a training site, which
    is the leakage this scheme exists to remove. Rotating the direction per repeat varies the
    partition without any clustering fit.

    Args:
        coords: Site coordinates, shape [n, D].
        n_splits: Bands per repeat.
        n_repeats: Independent partitions (independent projection directions).
        seed: Base RNG seed.

    Returns:
        Same structure as :func:`make_repeated_kfold`.

    Raises:
        RuntimeError: If no usable partition exists (degenerate coordinates).
    """
    X = np.asarray(coords, dtype=np.float64)
    n, d = X.shape
    idx = np.arange(n)
    # Refuse degenerate coordinates rather than silently returning index blocks: a stable argsort of
    # a constant projection preserves index order, so the "spatial" bands would be contiguous blocks
    # of row indices with no spatial meaning at all -- and the resulting score would be read as a
    # leakage-free one when nothing had been held out spatially.
    if n < 2 or np.allclose(X, X[0]):
        raise RuntimeError(
            f"make_spatial_kfold: coordinates have no spatial extent ({n} site(s), "
            f"{'all identical' if n >= 2 else 'too few'}); a spatial partition is undefined."
        )
    rng = np.random.default_rng(seed)
    repeats = []
    for _ in range(n_repeats):
        u = rng.normal(size=d)
        norm = np.linalg.norm(u)
        if norm == 0:
            continue
        proj = X @ (u / norm)                                              # [n]
        if np.allclose(proj, proj[0]):  # this direction is degenerate; try the next one
            continue
        order = idx[np.argsort(proj, kind="stable")]                       # [n]
        bands = np.array_split(order, n_splits)
        splits = tuple(
            (np.setdiff1d(idx, val, assume_unique=False), val)
            for val in bands if 0 < val.size < n
        )
        if splits:
            repeats.append(splits)
    if not repeats:
        raise RuntimeError("make_spatial_kfold: no usable spatial partition (degenerate coordinates?).")
    return tuple(repeats)


def _ridge_oof_predict(
    A: np.ndarray, y: np.ndarray, splits: Sequence[tuple[np.ndarray, np.ndarray]], lambda_rel: float,
) -> np.ndarray:
    """Out-of-fold ridge predictions for one repeated-CV partition (training-fold centering).

    Returns ``(pred, covered)``: a partition need not cover every index (a spatial partition can
    drop a degenerate patch), and only covered entries are scored, so an uncovered index can never
    be read as an uninitialised value.
    """
    pred = np.empty(len(y), dtype=np.float64)
    covered = np.zeros(len(y), dtype=bool)
    p = A.shape[1]
    for train, val in splits:
        covered[val] = True
        At, yt = A[train], y[train]
        mu_a, mu_y = At.mean(axis=0), yt.mean()
        Ac, yc = At - mu_a, yt - mu_y
        lam = lambda_rel * float(np.sum(Ac * Ac)) / p
        Av = A[val] - mu_a
        n_tr = len(train)
        if p > n_tr:
            K = Ac @ Ac.T                                                  # [n_tr, n_tr]
            alpha = np.linalg.solve(K + lam * np.eye(n_tr), yc)
            pred[val] = Av @ (Ac.T @ alpha) + mu_y
        else:
            w = np.linalg.solve(Ac.T @ Ac + lam * np.eye(p), Ac.T @ yc)
            pred[val] = Av @ w + mu_y
    return pred, covered


def scale_free_ridge_r2(
    A: np.ndarray,
    y: np.ndarray,
    repeats: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    *,
    lambda_rel: float,
) -> float:
    """Repeated-K-fold ridge R^2 of ``y`` from ``A``, scale-free and shared-partition (A3).

    Ridge penalty ``lambda = lambda_rel * tr(A_c^T A_c) / p`` on the training-fold-centered
    design (an implicit intercept), so R^2 is invariant to a uniform positive rescaling of ``A``
    (the fitted coefficients and the penalty scale together, so predictions are unchanged) --
    unlike the frozen ``update_rule.layer_alignment`` reference, which regresses an un-centered
    design at a fixed ``ridge=1.0``. Solved in the dual whenever ``p > n_train`` (required for
    H3, where ``p=5000``); the primal and dual solutions agree to machine precision.

    Args:
        A: Design matrix (e.g. one block's per-site representation change), shape [n, p].
        y: Target (e.g. ``Delta_PFN``), shape [n].
        repeats: Output of :func:`make_repeated_kfold` (or :func:`make_spatial_kfold`) -- the SAME
            partition for every block and head stage of this anchor.
        lambda_rel: Ridge strength relative to the training fold's feature energy.

    Returns:
        Mean R^2 over the repeats' pooled out-of-fold predictions. Each repeat is scored only on
        the indices its partition actually held out, and ``y``'s reference variance is taken on the
        same indices, so a partition that drops a patch stays comparable with one that does not.

    Raises:
        RuntimeError: If the target has zero variance.
    """
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if float(((y - y.mean()) ** 2).sum()) == 0:
        raise RuntimeError("scale_free_ridge_r2: target has zero variance; R^2 undefined.")
    r2s = []
    for splits in repeats:
        pred, covered = _ridge_oof_predict(A, y, splits, lambda_rel)
        yc = y[covered]
        ss_tot = float(((yc - yc.mean()) ** 2).sum())
        if ss_tot == 0:
            continue
        r2s.append(1.0 - float(((yc - pred[covered]) ** 2).sum()) / ss_tot)
    if not r2s:
        raise RuntimeError("scale_free_ridge_r2: no repeat held out a scoreable set of sites.")
    return float(np.mean(r2s))


# ---------------------------------------------------------------------------
# B1: template bank (which kernel/geometry explains the profile)
# ---------------------------------------------------------------------------
def template_bank(
    delta_z_norm: np.ndarray,
    Q: np.ndarray,
    anchor: int,
    context_X: np.ndarray,
    noise_ratio: float,
    ell_ladder: Sequence[float],
    *,
    delta_gp_abs: np.ndarray | None = None,
    ch2xy: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Winning distance/kernel template for one block's representation-change profile (B1).

    Regresses ``delta_z_norm`` (normalized to 1 at the anchor) against a bank of one-parameter
    (scale-only) templates: a ladder of conditioned-RBF profiles
    (:func:`pfns4neurostim.analysis.update_rule.conditioned_profile`, which this sprint factored
    out of ``fit_conditioned_lengthscale`` so it can be evaluated at a ladder rather than only the
    fitted optimum), the raw distance decay, and -- when supplied -- ``|Delta_GP|`` itself. This is
    the first-pass template set (B1's richer axis/stripe bank for C3 is added when Phase 1 starts).

    Args:
        delta_z_norm: Per-site representation-change norm, shape [M].
        Q: Readout coordinates (the same space ``context_X`` and the reference GP use), shape
            [M, D].
        anchor: Anchor site index.
        context_X: Context coordinates, shape [t, D].
        noise_ratio: Reference GP's noise/outputscale.
        ell_ladder: Candidate RBF lengthscales for the conditioned-kernel template.
        delta_gp_abs: Optional ``|Delta_GP|`` at the same sites, shape [M].
        ch2xy: Optional integer electrode-grid coordinates, shape [M, 2]; when given, the
            distance template is pitch distance (:func:`update_rule.site_distances`) rather than
            Euclidean distance in ``Q``.

    Returns:
        ``{'ell_eff', 'r2_conditioned', 'r2_distance'[, 'r2_gp']}``.

    Raises:
        RuntimeError: If the anchor's own value is zero (the profile cannot be normalized).
    """
    z = np.asarray(delta_z_norm, dtype=np.float64)
    denom = z[int(anchor)]
    if denom == 0:
        raise RuntimeError("template_bank: zero anchor value; cannot normalize the profile.")
    g = z / denom

    def r2_of(template: np.ndarray) -> float | None:
        t = np.asarray(template, dtype=np.float64)
        if np.std(t) == 0:
            return None
        scale = float(np.dot(t, g) / np.dot(t, t))
        ss_tot = float(((g - g.mean()) ** 2).sum())
        return None if ss_tot == 0 else 1.0 - float(((g - scale * t) ** 2).sum()) / ss_tot

    best_ell: float | None = None
    best_r2: float | None = None
    for ell in ell_ladder:
        prof = conditioned_profile(float(ell), Q, int(anchor), context_X, noise_ratio)
        r2 = r2_of(prof)
        if r2 is not None and (best_r2 is None or r2 > best_r2):
            best_ell, best_r2 = float(ell), r2
    if ch2xy is not None:
        dist = site_distances(ch2xy, np.asarray([anchor]))[0]
    else:
        dist = np.linalg.norm(Q - Q[int(anchor)], axis=1)
    out: dict[str, float | None] = {"ell_eff": best_ell, "r2_conditioned": best_r2, "r2_distance": r2_of(-dist)}
    if delta_gp_abs is not None:
        out["r2_gp"] = r2_of(np.asarray(delta_gp_abs, dtype=np.float64))
    return out


# ---------------------------------------------------------------------------
# C1 / C1': is it geometry, or is it the posterior? (partial correlations)
# ---------------------------------------------------------------------------
def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
    *,
    min_residual_frac: float = 1e-8,
) -> float | None:
    """Pearson correlation of ``x`` and ``y`` after regressing both on ``controls``.

    The discriminating statistic of the geometry question. The exact GP update at a fixed anchor is
    ``Delta_GP(x) = k_C(x, x*) * (y* - mu_C(x*)) / (k_C(x*, x*) + s2)``, and the second factor does
    not depend on ``x`` -- so ``|Delta_GP|`` is, across sites, *proportional to the conditioned
    kernel slice* and nothing else. A plain correlation of the representation change with
    ``|Delta_GP|`` therefore cannot separate "updates like a GP" from "spreads the new value by
    input similarity": both are kernel-shaped. What distinguishes them is *which* kernel, so the
    test is whether the conditioned (context-aware) slice still explains the representation change
    once a context-blind geometric template has been partialled out.

    Args:
        x: First variable, shape [n].
        y: Second variable, shape [n].
        controls: Variables to partial out, shape [n] or [n, k].
        min_residual_frac: Smallest fraction of a variable's own spread that may remain after the
            control is removed. Below it the residual is numerical dust rather than signal and the
            partial correlation is undefined.

    Returns:
        The partial correlation, or ``None`` when it is undefined: too few points, or either
        variable is (numerically) fully explained by the controls.

    Note:
        The emptiness check is on the *relative* residual norm, not on an exact zero. A least-squares
        residual that should vanish comes back at ~1e-17 rather than 0, so a plain ``std == 0`` guard
        never fires and a meaningless correlation of two dust vectors is returned instead of
        ``None``. This matters here because the two templates being compared -- the prior and the
        conditioned kernel slice -- are collinear by construction (0.80 to 0.96 on the NHP grid,
        rising as the context shrinks), so the near-degenerate regime is the operating regime, not
        an edge case. Statistical identifiability is a separate question and is reported separately
        by the runner as ``kernel_collinearity``; this guard only refuses to return arithmetic noise.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 3:
        return None
    Z = np.asarray(controls, dtype=np.float64).reshape(len(x), -1)
    design = np.column_stack([np.ones(len(x)), Z])                      # [n, 1+k]

    def residual(v: np.ndarray) -> np.ndarray | None:
        r = v - design @ np.linalg.lstsq(design, v, rcond=None)[0]      # [n]
        spread = float(np.linalg.norm(v - v.mean()))
        if spread == 0.0 or float(np.linalg.norm(r)) / spread < min_residual_frac:
            return None
        return r

    rx, ry = residual(x), residual(y)
    return None if rx is None or ry is None else _pearson(rx, ry)


def prior_kernel_slice(Q: np.ndarray, anchor: int, lengthscales: np.ndarray) -> np.ndarray:
    """The *prior* (context-blind) ARD-RBF kernel between every site and the anchor.

    The control template of :func:`partial_correlation`: the similarity a GP would assign before it
    has seen any data, using the same fitted lengthscales as the reference GP, so the comparison
    isolates the effect of conditioning rather than a change of kernel shape.

    Args:
        Q: Site coordinates, shape [M, D].
        anchor: Anchor index into ``Q``.
        lengthscales: Per-dimension lengthscales of the reference GP, shape [D] (a scalar is
            broadcast).

    Returns:
        ``k(x, x*)`` for every site, shape [M].
    """
    X = np.asarray(Q, dtype=np.float64)
    ell = np.broadcast_to(np.asarray(lengthscales, dtype=np.float64).ravel(), (X.shape[1],))
    d2 = (((X - X[int(anchor)]) / ell) ** 2).sum(axis=1)                # [M]
    return np.exp(-0.5 * d2)                                           # [M]


# ---------------------------------------------------------------------------
# S4: the logit lens — what the network would predict if it stopped at block l
# ---------------------------------------------------------------------------
def logit_lens(
    engine: Any,
    label_plus: Sequence[np.ndarray],
    label_minus: Sequence[np.ndarray],
) -> np.ndarray:
    """Per-block output-space update, read through the checkpoint's own decoder.

    Applies ``decoder_dict['standard']`` -- the real, already-trained prediction head, with zero
    fitted parameters added -- to each block's *label* token, then the same post-head chain the
    forward pass uses (:meth:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN.decode_logits` and the
    bar distribution's mean). The result is the prediction the network would make if it stopped at
    block l, so the antisymmetrised difference between the two surprise signs is an *output-space*
    update defined at every depth and directly comparable with the GP's.

    This is the readout that makes the depth profile interpretable without a learned probe: the
    label token is the only block-17 output the decoder actually reads, and at block 17 the lens
    value equals the model's real prediction by construction (a useful internal check).

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        label_plus: Per block, the label token at the positive surprise, each ``[q, 192]``.
        label_minus: Per block, the same at the negative surprise, each ``[q, 192]``.

    Returns:
        Antisymmetrised output-space update ``(mean_+ - mean_-) / 2``, shape [L, q].

    Raises:
        ValueError: If the two sign sequences do not match in length or shape.
    """
    import torch  # noqa: PLC0415

    if len(label_plus) != len(label_minus):
        raise ValueError(f"logit_lens: {len(label_plus)} blocks for +c but {len(label_minus)} for -c.")
    decoder = engine._model.decoder_dict["standard"]  # noqa: SLF001 - pinned internal (tabpfn 6.3.2)
    means = []
    for sign_states in (label_plus, label_minus):
        stacked = np.stack([np.asarray(z, dtype=np.float32) for z in sign_states])   # [L, q, 192]
        L, q, d = stacked.shape
        flat = torch.as_tensor(stacked.reshape(L * q, d), device=engine.device)      # [L*q, 192]
        with torch.no_grad():
            logits = decoder(flat)                                                  # [L*q, bars]
            mean, _ = engine.moments(engine.decode_logits(logits))                   # [L*q]
        means.append(np.asarray(mean, dtype=np.float64).reshape(L, q))               # [L, q]
    return 0.5 * (means[0] - means[1])                                               # [L, q]


# ---------------------------------------------------------------------------
# S1 / C4: common-mode share
# ---------------------------------------------------------------------------
def common_mode_share(dZ: np.ndarray) -> float:
    """Share of one block's representation-change energy carried by a site-independent shift.

    ``dZ`` decomposes orthogonally into a common mode (the across-site mean, repeated at every
    site) and a residual: for any fixed feature dimension, the mean is orthogonal to the
    zero-sum residual, so the two energies add exactly (no cross term). The reported share is
    the common mode's share of the total sum of squares (C4a: over-smoothing as a site-independent
    level shift; C4b: whether it is washed out by evidence).

    Args:
        dZ: Per-site representation change (one token kind, one block), shape [M, d].

    Returns:
        The share, in [0, 1].

    Raises:
        RuntimeError: If ``dZ`` has zero total energy.
    """
    Z = np.asarray(dZ, dtype=np.float64)
    mean = Z.mean(axis=0, keepdims=True)                        # [1, d]
    total = float((Z ** 2).sum())
    if total == 0:
        raise RuntimeError("common_mode_share: zero-energy representation change.")
    common_energy = float(Z.shape[0]) * float((mean ** 2).sum())
    return common_energy / total


# ---------------------------------------------------------------------------
# B4 / C5: direction-aware alignment and the decoder's own readout
# ---------------------------------------------------------------------------
def vector_cka(dZ: np.ndarray, delta_gp: np.ndarray) -> float:
    """Debiased linear CKA of a block's representation change against a rank-1 GP target (C5).

    Parameter-free post-LN puts every token on a fixed-radius sphere, so the norm-only alignment
    of :func:`alignment_stats` is blind to the *sign* of the GP update. Against the rank-1 target
    ``v v'`` with ``v = delta_gp``, debiased linear CKA (:func:`pfns4neurostim.analysis.cka.cka_debiased`)
    reduces, after centering, to the fraction of across-site variation in ``dZ`` explained by a
    single component proportional to the signed GP update -- exactly what the norm cannot see.

    Args:
        dZ: Per-site representation change, shape [M, d].
        delta_gp: Signed GP update at the same sites, shape [M].

    Returns:
        CKA in roughly [-small, 1]; exactly 1 for ``dZ`` rank-1 and proportional to ``delta_gp``.
    """
    v = np.asarray(delta_gp, dtype=np.float64)
    K = linear_gram(np.asarray(dZ, dtype=np.float64))
    L = np.outer(v, v)
    return cka_debiased(K, L)


def signed_pc_projection(dZ: np.ndarray, delta_gp: np.ndarray) -> float | None:
    """Correlation of the leading principal component of ``dZ`` with the **signed** GP update (C5).

    The companion to :func:`vector_cka`, and the second way around the blindness of the norm
    readout. Centering ``dZ`` over sites and projecting onto its leading principal component gives
    each site one signed coordinate; if that coordinate tracks the signed GP update, the
    representation change encodes the *direction* of the update and not merely its size.

    The sign of a principal component is arbitrary (``v`` and ``-v`` are the same component), so the
    magnitude of the correlation is the identified quantity and that is what is returned. A value
    near 1 means a single direction in representation space carries the signed update; the norm
    readout cannot distinguish that from the same-sized change in an unrelated direction.

    Args:
        dZ: Per-site representation change, shape [M, d].
        delta_gp: Signed GP update at the same sites, shape [M].

    Returns:
        ``|r|`` between the leading-component score and the signed update, or ``None`` when either
        is degenerate.
    """
    Z = np.asarray(dZ, dtype=np.float64)
    Z = Z - Z.mean(axis=0, keepdims=True)                              # [M, d]
    if Z.shape[0] < 3 or not np.any(Z):
        return None
    # Leading right singular vector; the score is the matching left vector scaled by its value.
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    if s[0] <= 0:
        return None
    score = Z @ vt[0]                                                  # [M]
    r = _pearson(score, np.asarray(delta_gp, dtype=np.float64))
    return None if r is None else abs(r)


def adjacent_block_cka(states: Sequence[np.ndarray]) -> list[float | None]:
    """Debiased CKA between each consecutive pair of block representations (C6b).

    Where the representation changes *character* with depth, consecutive blocks are least alike. The
    pre-registered reading is that such boundaries mark the transitions between stages of inference
    (latent mapping, then feature engineering, then committing to an answer), so the minima of this
    curve are the quantity of interest rather than its level.

    Note this is computed on the representations themselves, not on the probe-induced *change* in
    them -- it describes the network's own depth structure given a context, independent of any
    injected observation.

    Args:
        states: Per block, the site representations, each ``[M, d]`` (blocks in depth order).

    Returns:
        ``len(states) - 1`` values, entry ``l`` comparing block ``l`` with block ``l + 1``;
        ``None`` where a pair is degenerate.
    """
    grams = [linear_gram(np.asarray(z, dtype=np.float64)) for z in states]
    out: list[float | None] = []
    for lo, hi in zip(grams, grams[1:]):
        try:
            out.append(cka_debiased(lo, hi))
        except (RuntimeError, ValueError):
            out.append(None)
    return out


def decoder_projection(engine: Any, dZ17: np.ndarray) -> np.ndarray:
    """Push a block-17 label-token representation change through the real decoder head (C5).

    Zero fitted parameters: ``decoder_dict['standard']`` applied to ``dZ17`` is the model's own
    statement of which directions in the 192-d label token matter for its prediction -- C5's
    second sufficient criterion, used when the norm-only readout's block-5 dip does not meet the
    relative threshold against its plateau.

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        dZ17: Per-site block-17 label-token change, shape [M, 192].

    Returns:
        Decoded logit-space change, shape [M, n_out].
    """
    import torch  # noqa: PLC0415

    with torch.no_grad():
        z = torch.as_tensor(np.asarray(dZ17, dtype=np.float32), device=engine.device)
        out = engine._model.decoder_dict["standard"](z)  # noqa: SLF001 - pinned internal (tabpfn 6.3.2)
    return out.detach().float().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# H5: symmetric sigma alignment (closes roadmap M5)
# ---------------------------------------------------------------------------
def symmetric_sigma_alignment(
    sigma_plus: np.ndarray,
    sigma_minus: np.ndarray,
    gp_variance_reduction: np.ndarray,
) -> dict[str, float | None]:
    """H5: the symmetric part of the predictive-SD change against the GP's variance reduction.

    The GP's variance reduction ``k_C(x, x*)^2 / (k_C(x*, x*) + sigma^2)`` is sign-symmetric in
    the injected surprise ``c``, so the antisymmetrisation ``(f(+c) - f(-c)) / 2`` used everywhere
    else in M10 would cancel it exactly. This instead takes the SYMMETRIC part
    ``(sigma(+c) + sigma(-c)) / 2`` of the predictive-SD change, closing roadmap M5's missing
    sigma-recovery probe.

    Args:
        sigma_plus: Predictive-SD change at the positive surprise, shape [M].
        sigma_minus: Predictive-SD change at the matching negative surprise, shape [M].
        gp_variance_reduction: The GP's own (sign-symmetric) variance reduction, shape [M].

    Returns:
        ``{'pearson', 'spearman'}`` of the symmetric sigma change against the GP's reduction.
    """
    sym = 0.5 * (np.asarray(sigma_plus, dtype=np.float64) + np.asarray(sigma_minus, dtype=np.float64))
    red = np.asarray(gp_variance_reduction, dtype=np.float64)
    spearman = float(spearmanr(sym, red).statistic) if np.std(sym) > 0 and np.std(red) > 0 else None
    return {"pearson": _pearson(sym, red), "spearman": spearman}
