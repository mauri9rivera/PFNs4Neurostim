"""TabPFN v2.5 surrogate (task #1 Step 3).

Moved from the flat ``src/models/regressors.py``. Wraps a ``TabPFNRegressor`` and
reads its **native bar distribution**: the mean, standard deviation, UCB and
per-site Thompson draws all come from the predicted distribution rather than a
Gaussian approximation of it.

``_bar_distribution_moments`` implements the corrected predictive summaries
(P0.11): the mean is the distribution mean, not its median, and the quantile-based
standard deviation uses the right divisor.
"""

from __future__ import annotations

import importlib.metadata
import math
from typing import Any, Optional

import numpy as np
import torch
from tabpfn import TabPFNRegressor

__all__ = ["TabPFNSurrogate", "FrozenTabPFN", "PINNED_TABPFN_VERSION"]



# Quantile grid used to summarise the TabPFN bar distribution: mean/std come from
# integrating the predicted quantile function rather than from a normal
# approximation around the median (audit D7 / P0.11).  Levels are placed at equal
# steps on the probit scale over ±4 SD, which concentrates points in the tails where
# a uniform grid loses variance mass; on reference distributions this recovers the
# true std to <1% (normal 1.0006 vs 1; lognormal 1.296 vs 1.304).
_PROBIT_GRID_Z = 4.0
_N_QUANTILE_LEVELS = 99
QUANTILE_LEVELS = np.round(
    0.5 * (1.0 + torch.erf(
        torch.linspace(-_PROBIT_GRID_Z, _PROBIT_GRID_Z, _N_QUANTILE_LEVELS) / math.sqrt(2.0)
    ).numpy()),
    6,
).clip(1e-6, 1.0 - 1e-6)  # [99]


def _bar_distribution_moments(
    criterion: Any,
    logits: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and standard deviation of a TabPFN bar distribution.

    Uses the criterion's own moment methods, which integrate every bucket of the
    predicted density including the half-normal tails, rather than approximating
    the distribution as normal around its median (audit D7 / P0.11).

    Args:
        criterion: TabPFN ``FullSupportBarDistribution`` for these logits.
        logits: Bar-distribution logits, shape [M, num_bars].  # [M, num_bars]

    Returns:
        Tuple of (mean, std), each shape [M].  # [M], [M]

    Raises:
        RuntimeError: If any summary is not finite.
    """
    with torch.no_grad():
        mean = criterion.mean(logits)                                   # [M]
        var = criterion.variance(logits).clamp_min(0.0)                 # [M]
    mean_np = mean.detach().cpu().numpy()                               # [M]
    std_np = var.sqrt().detach().cpu().numpy()                          # [M]
    if not (np.isfinite(mean_np).all() and np.isfinite(std_np).all()):
        raise RuntimeError(
            "TabPFN bar-distribution summaries are not finite: "
            f"{(~np.isfinite(mean_np)).sum()} mean and "
            f"{(~np.isfinite(std_np)).sum()} std entries."
        )
    return mean_np, std_np                                              # [M], [M]


# ---------------------------------------------------------------------------
# TabPFNSurrogate — TabPFNRegressor wrapper conforming to SurrogateModel
# ---------------------------------------------------------------------------

class TabPFNSurrogate:
    """TabPFNRegressor wrapper conforming to the ``SurrogateModel`` protocol.

    Uses TabPFN's native bar-distribution UCB for acquisition, which is more
    accurate than the Gaussian ``mean + kappa * std`` approximation.

    The UCB exploration coefficient (kappa) is controlled externally via
    ``run_bo_loop(kappa_schedule=...)``, not stored here.

    Args:
        model: A fitted ``TabPFNRegressor`` (from ``extract_inference_model()``
            for finetuned models, or a plain ``TabPFNRegressor`` for vanilla).
    """

    def __init__(
        self,
        model: TabPFNRegressor,
    ) -> None:
        self._model = model
        # Logit cache: populated by predict_ucb, invalidated by fit.
        # Allows predict(X) to skip a second forward pass when called with the
        # same X and context as the preceding predict_ucb(X) call.
        self._logit_cache: Optional[tuple] = None  # (X_ref, logits, criterion)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store in-context examples for the TabPFN forward pass.

        No gradient updates are performed; this call sets the context that
        the transformer uses at prediction time.  Invalidates the logit cache
        since the context has changed.

        Args:
            X: Feature matrix of observed points, shape [N, D].  # [N, D]
            y: Response vector of observed targets, shape [N].   # [N]
        """
        if np.isnan(X).any() or np.isnan(y).any():
            raise RuntimeError(
                "TabPFNSurrogate.fit received NaN inputs. "
                f"X has {np.isnan(X).sum()} NaNs, y has {np.isnan(y).sum()} NaNs."
            )
        self._logit_cache = None
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean and standard deviation from the bar distribution.

        Mean and standard deviation are the bar distribution's own moments
        (``criterion.mean`` / ``criterion.variance``), which integrate the full
        predicted density including its half-normal tail buckets.  The reported
        mean is therefore the distribution mean, not the median, and the spread
        assumes no normality (audit D7 / P0.11).  ``QUANTILE_LEVELS`` +
        ``gpbo_utils.moments_from_quantiles`` remain available as an independent
        cross-check; they agree to 1e-4 on the mean and 2.5% on the std, the
        latter being quantile-truncation error in the cross-check, not here.

        If a logit cache is valid for this X (populated by a preceding
        ``predict_ucb(X)`` call with the same context), the moments come from the
        cached logits, avoiding a second transformer forward pass; otherwise one
        ``output_type='full'`` pass is run and the cache refreshed.

        Args:
            X: Query feature matrix, shape [M, D].  # [M, D]

        Returns:
            Tuple of (mean, std), each shape [M].   # [M], [M]

        Raises:
            RuntimeError: If the predictive summaries are not finite.
        """
        # --- Try logit cache; no blanket except — a failing cache is a bug, not a
        # fallback path to swallow (audit D7 / P0.11). ---
        if self._logit_cache is not None:
            X_ref, cached_logits, cached_criterion = self._logit_cache
            if X.shape == X_ref.shape and np.array_equal(X, X_ref):
                return _bar_distribution_moments(cached_criterion, cached_logits)

        # --- Full forward pass (refreshes the cache for later calls) ---
        full_output = self._model.predict(X, output_type="full")
        logits = full_output['logits']         # [M, num_bars]
        criterion = full_output['criterion']
        self._logit_cache = (X.copy(), logits.detach(), criterion)
        mean, std = _bar_distribution_moments(criterion, logits)  # [M], [M]

        if np.isnan(mean).any() or np.isnan(std).any():
            raise RuntimeError(
                f"TabPFNSurrogate.predict returned NaN values. "
                f"mean NaNs: {np.isnan(mean).sum()}, std NaNs: {np.isnan(std).sum()}."
            )
        return mean, std

    def predict_ucb(
        self,
        X: np.ndarray,
        kappa: float,
        t: int,
        n_steps: int,
    ) -> np.ndarray:
        """Return UCB values using the native TabPFN bar-distribution criterion.

        Converts ``kappa`` to ``rest_prob`` as::

            rest_prob = 0.5 * erfc(kappa / sqrt(2))

        which maps the Gaussian tail probability to the correct quantile of
        the bar distribution for UCB acquisition.

        Caches the bar-distribution logits so that a subsequent ``predict(X)``
        call on the same ``X`` (without an intervening ``fit``) can skip the
        second transformer forward pass.

        Args:
            X: Candidate feature matrix, shape [M, D].  # [M, D]
            kappa: Current (already-annealed) UCB exploration coefficient.
            t: Current BO step index (unused; annealing done by caller).
            n_steps: Total BO steps (unused; annealing done by caller).

        Returns:
            UCB values, shape [M].  # [M]
        """
        rest_prob = 0.5 * math.erfc(kappa / math.sqrt(2))
        full_output = self._model.predict(X, output_type="full")
        logits = full_output['logits']       # [M, num_bars]
        criterion = full_output['criterion']
        self._logit_cache = (X.copy(), logits.detach(), criterion)
        ucb_vals = criterion.ucb(logits, 0, rest_prob=rest_prob, maximize=True)
        return ucb_vals.clone().cpu().numpy()  # [M]

    def predict_ts(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Draw one Thompson sample from the TabPFN bar-distribution posterior.

        For each candidate in X, samples a bin index from the temperature-scaled
        bar distribution via multinomial sampling, then returns the bin-centre
        value.  The logit cache is populated so that a subsequent ``predict(X)``
        call with the same context can skip a second transformer forward pass.

        Temperature controls sharpness of the distribution before sampling:
          - ``temperature=1.0``: sample from the exact predicted distribution
          - ``temperature<1.0``: sharper / more exploitative (greedier argmax)
          - ``temperature>1.0``: flatter / more exploratory (more uniform)

        Args:
            X: Candidate feature matrix, shape [M, D].  # [M, D]
            temperature: Softmax temperature for bar distribution (default 1.0).

        Returns:
            Thompson sample values (bin centres), shape [M].  # [M]
        """
        full_output = self._model.predict(X, output_type="full")
        logits = full_output['logits']        # [M, num_bars]
        criterion = full_output['criterion']
        self._logit_cache = (X.copy(), logits.detach(), criterion)

        scaled_logits = logits / temperature                                  # [M, num_bars]
        probs = torch.softmax(scaled_logits, dim=-1)                          # [M, num_bars]
        bin_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)    # [M]

        # Map bin indices to bin-centre values using bar-distribution borders
        borders = criterion.borders                # [num_bars + 1]
        left = borders[bin_indices]                # [M]
        right = borders[bin_indices + 1]           # [M]
        samples = (left + right) / 2.0            # [M] — bin centres

        return samples.detach().cpu().numpy()      # [M]


# ---------------------------------------------------------------------------
# FrozenTabPFN — batched multi-context forward (promoted from the TS prototype, task #9 Step 3)
# ---------------------------------------------------------------------------
# Single-member TabPFN whose CPU preprocessing and target normalisation are fitted once on a
# real context and then frozen; the underlying transformer is run directly on a *batch* of
# contexts that share that context plus per-batch extra rows. The update-rule probes of M10
# (one extra observation per probe, ~96 probes per cell) therefore run as a few batched
# passes, and every probe is read out through the *same* preprocessing as the base context,
# so a measured change is the network's, not a refitted preprocessor's. Pinned to
# tabpfn==6.3.2 internals; ``tests/shadow/test_ts_sampler.py`` checks agreement with the
# public API for the prototype this was copied from.

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
        from tabpfn.utils import transform_borders_one  # noqa: PLC0415 - pinned internal

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
        self.criterion: Any = self._reg.raw_space_bardist_  # FullSupportBarDistribution
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
        from tabpfn.utils import translate_probs_across_borders  # noqa: PLC0415 - pinned internal

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

    def moments(self, log_probs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Mean and SD of the raw-space bar distributions returned by :meth:`forward`.

        Args:
            log_probs: Output of :meth:`forward`, shape [..., bars].

        Returns:
            ``(mean, sd)`` as float64 arrays of shape [...] (P0.11 moments).

        Raises:
            RuntimeError: If a summary is not finite.
        """
        self._check_fitted()
        lead = log_probs.shape[:-1]
        flat = log_probs.reshape(-1, log_probs.shape[-1])                   # [B*q, bars]
        flat = flat.to(self.criterion.borders.device)
        mean, sd = _bar_distribution_moments(self.criterion, flat)          # [B*q], [B*q]
        return mean.reshape(lead).astype(np.float64), sd.reshape(lead).astype(np.float64)

    def _check_fitted(self) -> None:
        """Raise if :meth:`fit` has not been called."""
        if not self._fitted:
            raise RuntimeError("FrozenTabPFN used before fit().")

    @property
    def wrapper(self) -> TabPFNRegressor:
        """The underlying fitted public-API regressor (for pinning tests)."""
        return self._reg
