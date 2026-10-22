"""Layer-wise TabPFN site states from one hooked forward pass (task #5, P0.5; B1; BETA 2026-10-21).

Protocol: the context is ``t`` observed noisy trials; *all* N grid sites are query rows (TabPFN gives query rows
no label); ``n_estimators=1`` with a fixed inference seed (the preprocessing ``random_state``); every transformer
block is hooked in the same forward pass.

Token layout of one row at a block's output, ``[B, rows, tokens, d]`` with ``d = 192``: the leading tokens are
*feature-group* tokens (TabPFN-2.5 groups ``features_per_group = 3`` preprocessed features per token; NHP's 2
inputs become 5 preprocessed features, so **2 feature tokens**, measured 2026-10-06), and the last token is the
label (target) token. The row axis also holds 64 *thinking* rows between the context and the queries, so query
rows are always the last ``q`` rows.

**Feature-token readouts** (:func:`readout_embeddings`, :data:`READOUTS`) serve the CKA (b) embedding placement,
which asks where the network's representation of the electrode geometry sits between prior and noise:

``feature_mean``    mean over the feature tokens.
``feature_tokens``  every feature token kept separately, ``[n_tok, q, d]`` (the retired B1 shadow readout).

**Capture layer** (moved here on 2026-10-15 from the cancelled Sprint #3 module): :func:`capture_block_states`
keeps both token kinds per block, :func:`capture_states_and_predictive` adds the final predictive of the same pass,
:func:`lens_log_probs` decodes every block's label token with the checkpoint's own head (the zero-parameter lens),
:func:`capture_head_states` keeps the prediction head's input, hidden state and logits, and :func:`logit_lens` is
the antisymmetrised lens mean. The label token is reportable against belief targets (guardrail G6 as rewritten per
BETA D1, 2026-10-15); the decoder hidden state stays diagnostic.

Uses :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`, so extra context rows (the update-rule probe of task
#9 Step 6) can be added without refitting preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "READOUTS", "LayerEmbeddings", "readout_embeddings",
    "BlockStates", "capture_block_states", "capture_states_and_predictive", "lens_log_probs",
    "HeadStates", "capture_head_states", "logit_lens",
]

#: Feature-token readouts over the per-row token axis.
READOUTS: tuple[str, ...] = ("feature_mean", "feature_tokens")


@dataclass(frozen=True)
class LayerEmbeddings:
    """One readout's embeddings at every hooked block.

    Attributes:
        readout: The readout (see :data:`READOUTS`).
        layers: Block index of each array.
        arrays: Per block, float64 ``[q, d]`` -- or ``[n_tok, q, d]`` for ``feature_tokens``.
    """

    readout: str
    layers: tuple[int, ...]
    arrays: tuple[np.ndarray, ...]


def readout_embeddings(
    engine: Any,
    X_query: np.ndarray,
    *,
    readouts: Sequence[str] = ("feature_mean",),
    layers: Sequence[int] | None = None,
    X_extra: np.ndarray | None = None,
    y_extra: np.ndarray | None = None,
) -> dict[str, LayerEmbeddings]:
    """Hook the requested blocks of a fitted ``FrozenTabPFN`` in ONE forward pass and read every readout from it.

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        X_query: Query coordinates (raw units, as passed to ``fit``), shape [q, D].
        readouts: Readouts to return, each one of :data:`READOUTS`.
        layers: Block indices; ``None`` = all.
        X_extra: Optional extra context rows, shape [r, D] (one batch element).
        y_extra: Their targets (raw units), shape [r].

    Returns:
        ``{readout: LayerEmbeddings}``.

    Raises:
        ValueError: On an unknown readout.
        RuntimeError: If a hook did not fire or produced non-finite values.
    """
    import torch  # noqa: PLC0415

    unknown = [r for r in readouts if r not in READOUTS]
    if unknown:
        raise ValueError(f"readout_embeddings: unknown readout(s) {unknown}; expected {READOUTS}.")
    blocks = engine._model.transformer_encoder.layers  # noqa: SLF001 - pinned internal (tabpfn 6.3.2)
    idx = list(range(len(blocks))) if layers is None else [int(i) for i in layers]
    q = len(X_query)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(k: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            act = out if isinstance(out, torch.Tensor) else out[0]   # [B, rows, tokens, d]
            captured[k] = act[0, -q:, :-1].detach().float().cpu()   # [q, n_tok, d]  feature tokens only
        return hook

    handles = [blocks[k].register_forward_hook(make_hook(k)) for k in idx]
    try:
        Qt = engine.transform_x(X_query)                              # [q, F]
        if X_extra is None:
            engine.forward(None, None, Qt)
        else:
            Xe = engine.transform_x(np.atleast_2d(X_extra)).unsqueeze(0)   # [1, r, F]
            ye = engine.transform_y(torch.as_tensor(
                np.asarray(y_extra, dtype=np.float32), device=engine.device)).reshape(1, -1)  # [1, r]
            engine.forward(Xe, ye, Qt)
    finally:
        for h in handles:
            h.remove()
    missing = [k for k in idx if k not in captured]
    if missing:
        raise RuntimeError(f"readout_embeddings: hooks on blocks {missing} did not fire.")
    out: dict[str, LayerEmbeddings] = {}
    for readout in readouts:
        arrays = []
        for k in idx:
            tokens = captured[k]                                      # [q, n_tok, d]
            z = tokens.mean(dim=1) if readout == "feature_mean" else tokens.transpose(0, 1)   # [q, d] | [n_tok, q, d]
            a = z.numpy().astype(np.float64)
            if not np.isfinite(a).all():
                raise RuntimeError(f"readout_embeddings: non-finite activations at block {k} ({readout}).")
            arrays.append(a)
        out[readout] = LayerEmbeddings(readout, tuple(idx), tuple(arrays))
    return out


# ---------------------------------------------------------------------------
# Capture layer: both token kinds and the prediction head (pinned to tabpfn 6.3.2)
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
    the update probe's injected row: the readout rows
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


def _capture(
    engine: Any,
    X_query: np.ndarray,
    X_extra: np.ndarray | None,
    y_extra: np.ndarray | None,
    layers: Sequence[int] | None,
    as_train: bool,
) -> tuple[BlockStates, Any]:
    """One hooked forward pass: the block states of the query rows and the pass's own log-probabilities."""
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
        log_probs = _run_forward(engine, X_query, X_extra, y_extra, as_train)   # [1, q, bars]
    finally:
        for h in handles:
            h.remove()
    missing = [k for k in idx if k not in captured]
    if missing:
        raise RuntimeError(f"capture_block_states: hooks on blocks {missing} did not fire.")
    feature_tokens = tuple(captured[k][:, :-1].numpy().astype(np.float64) for k in idx)
    label_token = tuple(captured[k][:, -1].numpy().astype(np.float64) for k in idx)
    for name, arrs in (("feature", feature_tokens), ("label", label_token)):
        if not all(np.isfinite(a).all() for a in arrs):
            raise RuntimeError(f"capture_block_states: non-finite {name}-token activations.")
    return BlockStates(layers=tuple(idx), feature_tokens=feature_tokens, label_token=label_token), log_probs


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
    feature-token mean only) to also keep the label token per block: the only token the decoder
    reads, which the logit lens and BETA's label-token readout need.

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
    return _capture(engine, X_query, X_extra, y_extra, layers, as_train)[0]


def capture_states_and_predictive(
    engine: Any,
    X_query: np.ndarray,
    *,
    layers: Sequence[int] | None = None,
) -> tuple[BlockStates, Any]:
    """Block states AND the final predictive of the same forward pass (BETA β2 / β1a).

    The representation and the belief targets built from TabPFN's own predictive (its mean and SD maps, E_out) then
    describe literally one pass, so the zero-parameter lens at the last block reproduces the final predictive.

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN` (context only, no injected row).
        X_query: Query coordinates, shape [q, D].
        layers: Block indices; ``None`` = all.

    Returns:
        ``(states, log_probs)`` with ``log_probs`` the raw-space bar log-probabilities, shape [1, q, bars].
    """
    return _capture(engine, X_query, None, None, layers, False)


def lens_log_probs(engine: Any, label_states: Sequence[np.ndarray]) -> Any:
    """Zero-parameter logit lens: each block's label token decoded by the checkpoint's own head (BETA §4).

    Applies ``decoder_dict['standard']`` (no fitted parameters added) and then exactly the post-head chain of the
    forward pass (:meth:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN.decode_logits`: softmax temperature, logit
    mask, re-binning, log), so the result is the full predictive the network would emit if it stopped at block l.
    :func:`logit_lens` is its antisymmetrised-mean special case. Whether a block's lens is degenerate is an OPEN rule
    (``research_design.md`` §4, 2026-10-19); this function only decodes.

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        label_states: Per block, the label token of the query rows, each ``[q, 192]``.

    Returns:
        Log-probabilities over the raw-space bar distribution, torch tensor of shape [L, q, bars].

    Raises:
        ValueError: On an empty block list or blocks of different shapes.
    """
    import torch  # noqa: PLC0415

    if not len(label_states):
        raise ValueError("lens_log_probs: no blocks given.")
    stacked = np.stack([np.asarray(z, dtype=np.float32) for z in label_states])   # [L, q, 192]
    L, q, d = stacked.shape
    decoder = engine._model.decoder_dict["standard"]  # noqa: SLF001 - pinned internal (tabpfn 6.3.2)
    flat = torch.as_tensor(stacked.reshape(L * q, d), device=engine.device)        # [L*q, 192]
    with torch.no_grad():
        logits = decoder(flat)                                                     # [L*q, bars]
        out = engine.decode_logits(logits)                                         # [L*q, bars]
    return out.reshape(L, q, -1)                                                   # [L, q, bars]


@dataclass(frozen=True)
class HeadStates:
    """Prediction-head readouts of the query rows (the decoder hidden state stays diagnostic, G6).

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
