"""Batched low-level TabPFN forward with frozen preprocessing (pinned to tabpfn==6.3.2).

The public ``TabPFNRegressor.predict`` re-runs sklearn preprocessing on every call and
accepts exactly one context. Joint Thompson sampling needs many forward passes whose
contexts differ only by appended *fantasy* rows, and calibration tests need hundreds of
such contexts. This engine:

1. Fits a ``TabPFNRegressor`` once on the real context (``fit_mode='fit_preprocessors'``,
   ``n_estimators=1``, ``softmax_temperature=1.0``).
2. Freezes that single ensemble member's CPU preprocessing and target normalisation.
3. Runs the underlying ``PerFeatureTransformer`` directly on a *batch* of B contexts
   ``[N_ctx + N_fant + q, B, F]`` in one pass, and post-processes the logits exactly as
   ``TabPFNRegressor.predict`` does (border transform -> translate -> log).

Fantasy rows are transformed with the frozen (context-fitted) preprocessing rather than
refitting it on context + fantasies, so every conditional in a sampling chain comes from
the *same* model. ``test_ts_tabpfn_engine_pinned`` verifies bit-level agreement (up to
float tolerance) with the public API and fails loudly if TabPFN internals change.
"""
from __future__ import annotations

import importlib.metadata
from typing import Optional

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
from tabpfn.utils import transform_borders_one, translate_probs_across_borders

PINNED_TABPFN_VERSION = "6.3.2"


def check_tabpfn_version() -> None:
    """Raise if the installed tabpfn is not the version this engine was written against.

    Raises:
        RuntimeError: If ``tabpfn`` != ``PINNED_TABPFN_VERSION``.
    """
    installed = importlib.metadata.version("tabpfn")
    if installed != PINNED_TABPFN_VERSION:
        raise RuntimeError(
            f"FrozenTabPFN relies on tabpfn=={PINNED_TABPFN_VERSION} internals "
            f"(executor_.ensemble_members, model_caches, transform_borders_one); "
            f"found tabpfn=={installed}. Re-validate test_ts_tabpfn_engine_pinned."
        )


class FrozenTabPFN:
    """Single-member TabPFN with frozen preprocessing and a batched forward pass.

    Args:
        device: Torch device string (``'cuda'`` or ``'cpu'``).
        random_state: Integer seed for TabPFN's preprocessing-config sampling. Fixed so the
            same preprocessing is used across all fits (no hidden sampling).
        max_batch_tokens: Soft cap on ``B * N_rows`` per internal forward; larger batches
            are split into sub-batches to bound GPU memory.
    """

    def __init__(
        self,
        device: str = "cuda",
        random_state: int = 0,
        max_batch_tokens: int = 400_000,
    ) -> None:
        check_tabpfn_version()
        self.device = torch.device(device)
        self.random_state = random_state
        self.max_batch_tokens = max_batch_tokens
        self._reg = TabPFNRegressor(
            device=device,
            n_estimators=1,
            softmax_temperature=1.0,
            ignore_pretraining_limits=True,
            fit_mode="fit_preprocessors",
            random_state=random_state,
        )
        self._fitted = False

    # ------------------------------------------------------------------ fitting
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit preprocessing on the real context and cache transformed tensors.

        Args:
            X: Context inputs, shape [n, D].
            y: Context targets (raw units), shape [n].

        Raises:
            RuntimeError: On NaN/Inf inputs, constant targets, or unexpected internals.
        """
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise RuntimeError("FrozenTabPFN.fit received NaN/Inf inputs.")
        if np.unique(y).size == 1:
            raise RuntimeError("FrozenTabPFN.fit: constant target is not supported.")
        self._reg.fit(X, y)
        executor = self._reg.executor_
        members = getattr(executor, "ensemble_members", None)
        if members is None or len(members) != 1:
            raise RuntimeError(
                "FrozenTabPFN expected executor_.ensemble_members with exactly one member "
                f"(tabpfn internals changed?): got {type(executor).__name__}."
            )
        member = members[0]
        if member.config.subsample_ix is not None:
            raise RuntimeError("FrozenTabPFN does not support row subsampling configs.")
        if member.gpu_preprocessor is not None:
            raise RuntimeError("FrozenTabPFN does not support GPU preprocessing pipelines.")
        self._member = member
        cache = executor.model_caches[member.config._model_index]
        cached = list(getattr(cache, "_models", {}).values())
        if len(cached) != 1:
            raise RuntimeError(
                "FrozenTabPFN expected exactly one cached model copy "
                f"(tabpfn internals changed?): got {len(cached)}."
            )
        self._model = cached[0]
        self.device = next(self._model.parameters()).device
        self._cat_ix = list(member.cat_ix)
        self._y_mean = float(self._reg.y_train_mean_)
        self._y_std = float(self._reg.y_train_std_)
        self._autocast = bool(self._reg.use_autocast_)

        self.X_ctx = torch.as_tensor(
            np.asarray(member.X_train), dtype=torch.float32, device=self.device
        )  # [n, F]
        self.y_ctx = torch.as_tensor(
            np.asarray(member.y_train), dtype=torch.float32, device=self.device
        )  # [n]

        znorm_borders = self._reg.znorm_space_bardist_.borders.to(self.device)  # [bars+1]
        target_transform = member.config.target_transform
        if target_transform is None:
            borders_t = znorm_borders.clone()
            self._logit_cancel_mask: Optional[torch.Tensor] = None
            self._descending = False
        else:
            mask, descending, borders_np = transform_borders_one(
                znorm_borders.cpu().numpy(),
                target_transform=target_transform,
                repair_nan_borders_after_transform=(
                    self._reg.inference_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM
                ),
            )
            borders_t = torch.as_tensor(borders_np, device=self.device)
            if descending:
                borders_t = borders_t.flip(-1)
            self._logit_cancel_mask = (
                None if mask is None else torch.as_tensor(mask, device=self.device)
            )
            self._descending = bool(descending)
        self._borders_t = borders_t.float()
        self._znorm_borders = znorm_borders.float()
        self.criterion: FullSupportBarDistribution = self._reg.raw_space_bardist_
        self.raw_borders = self.criterion.borders.to(self.device).float()  # [bars+1]
        self._fitted = True

    # --------------------------------------------------------------- transforms
    def transform_x(self, X: np.ndarray) -> torch.Tensor:
        """Apply the frozen feature preprocessing.

        Args:
            X: Raw inputs, shape [q, D].

        Returns:
            Model-space features, shape [q, F].
        """
        self._check_fitted()
        Xt = self._member.transform_X_test(np.asarray(X, dtype=np.float64))
        return torch.as_tensor(np.asarray(Xt), dtype=torch.float32, device=self.device)

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        """Map raw-unit targets into the model's target space (frozen normalisation).

        Args:
            y: Raw-unit targets, any shape [...].

        Returns:
            Model-space targets, same shape.
        """
        self._check_fitted()
        z = (y - self._y_mean) / self._y_std  # [...]
        tt = self._member.config.target_transform
        if tt is None:
            return z
        flat = z.detach().cpu().numpy().reshape(-1, 1)
        out = tt.transform(flat).reshape(z.shape)
        return torch.as_tensor(out, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        X_extra: Optional[torch.Tensor],
        y_extra: Optional[torch.Tensor],
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """Batched forward: shared real context + per-batch fantasies -> test logits.

        Args:
            X_extra: Fantasy inputs in model space, shape [B, r, F] or None (r=0).
            y_extra: Fantasy targets in *model* space, shape [B, r] or None.
            X_test: Test inputs in model space, shape [B, q, F] or [q, F] (broadcast).

        Returns:
            Log-probabilities over the raw-space bar distribution, shape [B, q, bars].

        Raises:
            RuntimeError: If the output contains NaN.
        """
        self._check_fitted()
        if X_test.dim() == 2:
            B = 1 if X_extra is None else X_extra.shape[0]
            X_test = X_test.unsqueeze(0).expand(B, -1, -1)  # [B, q, F]
        B, q, _ = X_test.shape
        n = self.X_ctx.shape[0]
        r = 0 if X_extra is None else X_extra.shape[1]
        rows = n + r + q
        sub = max(1, self.max_batch_tokens // max(rows, 1))
        outs = []
        for start in range(0, B, sub):
            stop = min(B, start + sub)
            b = stop - start
            parts_x = [self.X_ctx.unsqueeze(0).expand(b, -1, -1)]  # [b, n, F]
            parts_y = [self.y_ctx.unsqueeze(0).expand(b, -1)]      # [b, n]
            if r > 0:
                parts_x.append(X_extra[start:stop])                 # [b, r, F]
                parts_y.append(y_extra[start:stop])                 # [b, r]
            parts_x.append(X_test[start:stop])                      # [b, q, F]
            x_full = torch.cat(parts_x, dim=1).transpose(0, 1).contiguous()  # [n+r+q, b, F]
            y_train = torch.cat(parts_y, dim=1).transpose(0, 1).contiguous() # [n+r, b]
            with torch.autocast(
                device_type=self.device.type, enabled=self._autocast and self.device.type == "cuda"
            ), torch.inference_mode():
                raw = self._model(
                    x_full,
                    y_train,
                    only_return_standard_out=True,
                    categorical_inds=[self._cat_ix] * b,
                )  # [q, b, bars]
            outs.append(raw.float().transpose(0, 1))  # [b, q, bars]
        logits = torch.cat(outs, dim=0)  # [B, q, bars]
        if self._logit_cancel_mask is not None:
            logits = logits.clone()
            logits[..., self._logit_cancel_mask] = float("-inf")
        probs = translate_probs_across_borders(
            logits, frm=self._borders_t, to=self._znorm_borders
        )  # [B, q, bars]
        out = probs.log()
        if torch.isnan(out).any():
            raise RuntimeError("FrozenTabPFN.forward produced NaN log-probabilities.")
        return out

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FrozenTabPFN used before fit().")

    @property
    def wrapper(self) -> TabPFNRegressor:
        """The underlying fitted public-API regressor (for pinning tests)."""
        return self._reg
