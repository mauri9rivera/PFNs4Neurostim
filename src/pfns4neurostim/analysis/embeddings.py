"""Layer-wise TabPFN site embeddings of the FEATURE tokens, from one hooked forward pass (task #5, P0.5; B1).

Protocol: the context is ``t`` observed noisy trials; *all* N grid sites are query rows (TabPFN gives query rows
no label); ``n_estimators=1`` with a fixed inference seed (the preprocessing ``random_state``); every transformer
block is hooked in the same forward pass.

Token layout of one row at a block's output, ``[B, rows, tokens, d]`` with ``d = 192``: the leading tokens are
*feature-group* tokens (TabPFN-2.5 groups ``features_per_group = 3`` preprocessed features per token; NHP's 2
inputs become 5 preprocessed features, so **2 feature tokens**, measured 2026-10-06), and the last token is the
label (target) token. The row axis also holds 64 *thinking* rows between the context and the queries, so query
rows are always the last ``q`` rows.

Only the feature tokens are read out (decision 2026-10-07, user). The label token is the one the decoder turns
into the prediction, so a CKA of it re-measures prediction accuracy (Hyp A's R^2) and its placement re-places the
predicted response map among prior maps (C3's MMD / W2 placement); neither says anything about how the network
represents the electrode geometry, which is the question the mechanism analyses ask. Readouts
(:data:`READOUTS`):

``feature_mean``    mean over the feature tokens.
``feature_tokens``  every feature token kept separately, ``[n_tok, q, d]``; CKA is then computed per token and
                    averaged (B1 Step 5 shadow run), which is not the CKA of the averaged tokens.

Uses :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`, so extra context rows (the update-rule probe of task
#9 Step 6) can be added without refitting preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["READOUTS", "LayerEmbeddings", "readout_embeddings", "layer_embeddings", "site_embeddings"]

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


def layer_embeddings(
    engine: Any,
    X_query: np.ndarray,
    *,
    X_extra: np.ndarray | None = None,
    y_extra: np.ndarray | None = None,
    layers: Sequence[int] | None = None,
    readout: str = "feature_mean",
) -> np.ndarray:
    """Block embeddings of one readout, stacked.

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        X_query: Query coordinates (raw units, as passed to ``fit``), shape [q, D].
        X_extra: Optional extra context rows, shape [r, D] (one batch element).
        y_extra: Their targets (raw units), shape [r].
        layers: Block indices; ``None`` = all.
        readout: One of :data:`READOUTS`.

    Returns:
        Embeddings, shape [L, q, d] (float64); [L, n_tok, q, d] for ``feature_tokens``.
    """
    emb = readout_embeddings(engine, X_query, readouts=(readout,), layers=layers,
                             X_extra=X_extra, y_extra=y_extra)[readout]
    return np.stack(emb.arrays)


def site_embeddings(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_sites: np.ndarray,
    *,
    device: str = "cpu",
    inference_seed: int = 0,
    layers: Sequence[int] | None = None,
    readout: str = "feature_mean",
    engine: Any | None = None,
) -> np.ndarray:
    """Fit TabPFN on a context and embed every site: the P0.5 protocol in one call.

    Args:
        X_context: Context coordinates, shape [t, D].
        y_context: Context values (one noisy trial each), shape [t].
        X_sites: All grid sites (query rows), shape [N, D].
        device: Torch device.
        inference_seed: TabPFN preprocessing seed (fixed).
        layers: Layer indices; ``None`` = all 18.
        readout: See :data:`READOUTS`.
        engine: Reuse an existing (unfitted or fitted) ``FrozenTabPFN``; it is refitted.

    Returns:
        Embeddings, shape [L, N, d].
    """
    if engine is None:
        from ..models.pfn.tabpfn import FrozenTabPFN  # noqa: PLC0415

        engine = FrozenTabPFN(device=device, random_state=inference_seed)
    engine.fit(np.asarray(X_context), np.asarray(y_context))
    return layer_embeddings(engine, np.asarray(X_sites), layers=layers, readout=readout)
