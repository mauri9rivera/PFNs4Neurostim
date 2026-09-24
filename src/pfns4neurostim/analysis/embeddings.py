"""Layer-wise TabPFN site embeddings from one hooked forward pass (task #5 Step 0, P0.5).

Protocol: the context is ``t`` observed noisy trials; *all* N grid sites are query rows
(TabPFN gives query rows no label, i.e. a dummy label token); ``n_estimators=1`` with a fixed
inference seed (the preprocessing ``random_state``); every transformer layer is hooked in the
same forward pass. The readout of site ``i`` at layer ``l`` is the layer-``l`` output of its
query row, **averaged over the feature tokens** (the last token of each row is the target
token that the decoder reads, ``encoder_out[:, single_eval_pos:, -1]`` in
``tabpfn/architectures/base/transformer.py``; it is available as ``readout='label_token'``).

Uses :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`, so extra context rows (the
update-rule probe of task #9 Step 6) can be added without refitting preprocessing.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["READOUTS", "layer_embeddings", "site_embeddings"]

#: Row readouts over the per-row token axis.
READOUTS: tuple[str, ...] = ("feature_mean", "label_token", "all_mean")


def layer_embeddings(
    engine: Any,
    X_query: np.ndarray,
    *,
    X_extra: np.ndarray | None = None,
    y_extra: np.ndarray | None = None,
    layers: Sequence[int] | None = None,
    readout: str = "feature_mean",
) -> np.ndarray:
    """Hook every requested layer in one forward pass of a fitted ``FrozenTabPFN``.

    Args:
        engine: Fitted :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        X_query: Query coordinates (raw units, as passed to ``fit``), shape [q, D].
        X_extra: Optional extra context rows, shape [r, D] (one batch element).
        y_extra: Their targets (raw units), shape [r].
        layers: Layer indices; ``None`` = all.
        readout: One of :data:`READOUTS`.

    Returns:
        Embeddings, shape [L, q, d_model] (float64).

    Raises:
        ValueError: On an unknown readout.
        RuntimeError: If a hook did not fire or produced non-finite values.
    """
    import torch  # noqa: PLC0415

    if readout not in READOUTS:
        raise ValueError(f"layer_embeddings: unknown readout {readout!r}; expected {READOUTS}.")
    blocks = engine._model.transformer_encoder.layers  # noqa: SLF001 - pinned internal (tabpfn 6.3.2)
    idx = list(range(len(blocks))) if layers is None else [int(i) for i in layers]
    q = len(X_query)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(k: int) -> Any:
        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            act = out if isinstance(out, torch.Tensor) else out[0]   # [B, rows, tokens, d]
            rows = act[0, -q:].float()                               # [q, tokens, d]
            if readout == "feature_mean":
                rows = rows[:, :-1].mean(dim=1)                      # [q, d]
            elif readout == "label_token":
                rows = rows[:, -1]                                   # [q, d]
            else:
                rows = rows.mean(dim=1)                              # [q, d]
            captured[k] = rows.detach().cpu()
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
        raise RuntimeError(f"layer_embeddings: hooks on layers {missing} did not fire.")
    out = np.stack([captured[k].numpy() for k in idx]).astype(np.float64)   # [L, q, d]
    if not np.isfinite(out).all():
        raise RuntimeError("layer_embeddings: non-finite activations.")
    return out


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
