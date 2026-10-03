"""Per-model wrappers for the external amortized surrogates of Hypothesis 0 (task #8 Step 3).

One class per benchmark model, each reaching its **official implementation** —
vendored under ``libs/`` as a git submodule, or a pinned pip dependency — through
:mod:`~pfns4neurostim.models.pfn.external`. Nothing here re-implements a model.

Status as of 2026-09-20, after reading each upstream API:

============  =========  ==============================================================
Model         Runnable   Notes
============  =========  ==============================================================
TabFlex       **yes**    ``libs/ticl``; classification-only, so it goes through the
                         bucketized adapter. Weights download on first use.
TabICL v2     py>=3.10   ``libs/tabicl`` @ v2.2.0. **Native regressor** with a quantile
                         predictive distribution — not a classification-head adaptation.
TabFM         py>=3.11   ``libs/tabfm``. Native regressor, but ``predict`` returns point
                         predictions only; uncertainty comes from ensemble spread.
PFNs4BO       **yes**    Vendored checkpoint. An end-to-end BO model: it owns the query decision
                         (its acquisition criterion, computed inside the network), exposed as
                         a *native policy* that the ``native`` acquisition type delegates to.
Mitra         pending    Needs AutoGluon; predictive-distribution access unconfirmed.
============  =========  ==============================================================

The wrappers for TabICL and TabFM are written and will run as-is under a suitable
interpreter; they are blocked by the environment, not by missing code.
"""
from __future__ import annotations

import gzip
import io
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .external import BucketizedClassifierSurrogate, ExternalSurrogate, require_backend

__all__ = [
    "PFNs4BOSurrogate",
    "TabICLSurrogate",
    "CausiloSurrogate",
    "TabPFN35Surrogate",
    "TabFMSurrogate",
    "MitraSurrogate",
    "TabFlexSurrogate",
]

#: Quantile levels used to summarize a quantile-based predictive distribution.
#: Equally spaced on the probit scale over +-3 SD, so points concentrate in the
#: tails where a uniform grid loses variance mass (same rationale as the TabPFN
#: bar-distribution summary in models/pfn/tabpfn.py).
_QUANTILE_ALPHAS: tuple[float, ...] = (
    0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99,
)


def _midpoint_levels(n: int) -> tuple[float, ...]:
    """Equally spaced probability levels ``(i + 0.5) / n`` -- the grid a Thompson draw inverts.

    Args:
        n: Number of levels.

    Returns:
        ``n`` increasing levels strictly inside (0, 1).

    Raises:
        ValueError: If ``n`` < 2.
    """
    if n < 2:
        raise ValueError(f"need at least 2 sampling levels, got {n}.")
    return tuple((i + 0.5) / n for i in range(n))


def _inverse_cdf_draws(
    quantiles: np.ndarray,
    levels: tuple[float, ...],
    rng: np.random.Generator,
    temperature: float,
) -> np.ndarray:
    """One draw per row by inverting a predicted quantile function at a uniform level.

    A draw ``u ~ U(0, 1)`` is mapped through the model's own quantile function (linear interpolation between
    ``levels``; clamped to the end values outside them, i.e. at most ``0.5 / len(levels)`` of tail mass per side).

    Args:
        quantiles: Predicted quantiles, shape [N, Q], increasing along axis 1.
        levels: The Q probability levels, increasing.
        rng: Seeded generator.
        temperature: Variance scaling around the predictive mean (1.0 = the exact predictive).

    Returns:
        Samples, shape [N].

    Raises:
        ValueError: If ``temperature`` is not positive or the shapes disagree.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    q = np.asarray(quantiles, dtype=np.float64)                  # [N, Q]
    a = np.asarray(levels, dtype=np.float64)                     # [Q]
    if q.ndim != 2 or q.shape[1] != a.size:
        raise ValueError(f"quantiles {q.shape} do not match {a.size} levels.")
    u = rng.random(q.shape[0])                                   # [N]
    draws = np.array([np.interp(u[i], a, q[i]) for i in range(q.shape[0])])   # [N]
    if temperature == 1.0:
        return draws
    mean, _ = _moments_from_quantiles(q, tuple(a))
    return mean + np.sqrt(temperature) * (draws - mean)


def _moments_from_quantiles(
    quantiles: np.ndarray,
    alphas: tuple[float, ...] = _QUANTILE_ALPHAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a quantile function into a mean and standard deviation.

    Treats the predictive distribution as piecewise-uniform between the given
    quantiles, which is exact in the limit and avoids assuming normality — the
    point of using a distributional model in the first place.

    Args:
        quantiles: Predicted quantiles, shape [N, Q], increasing along axis 1.
        alphas: The probability levels of those quantiles, length Q.

    Returns:
        ``(mean, std)``, each shape [N].

    Raises:
        ValueError: If the shapes disagree.
    """
    q = np.asarray(quantiles, dtype=np.float64)          # [N, Q]
    a = np.asarray(alphas, dtype=np.float64)             # [Q]
    if q.shape[1] != a.size:
        raise ValueError(f"_moments_from_quantiles: {q.shape[1]} quantiles for {a.size} levels.")
    # Probability mass attributed to each quantile point (trapezoid in alpha).
    edges = np.concatenate(([0.0], 0.5 * (a[1:] + a[:-1]), [1.0]))    # [Q + 1]
    weights = np.diff(edges)                                          # [Q]
    mean = q @ weights                                                # [N]
    var = ((q - mean[:, None]) ** 2) @ weights                        # [N]
    return mean, np.sqrt(np.maximum(var, 1e-12))


class TabICLSurrogate(ExternalSurrogate):
    """TabICL v2 surrogate — **native regression with a quantile predictive distribution**.

    ``TabICLRegressor.predict(X, output_type='quantiles', alphas=...)`` returns the
    predicted quantile function, which this wrapper integrates into a mean and a
    standard deviation and samples by inverse-CDF for Thompson sampling. No
    bucketization is involved, so TabICL v2 is a first-class row of the Hyp 0
    table rather than a classification-head adaptation.

    Requires Python >= 3.10 (upstream declaration), so it cannot run in the
    project's pinned 3.9 environment.

    Args:
        device: Torch device string.
        n_estimators: Ensemble members; 1 keeps BO steps cheap.
        n_sample_levels: Resolution of the quantile function a Thompson draw inverts (equally spaced
            levels ``(i + 0.5) / n``). The mean/std summary keeps its own coarser grid.
        **backend_kwargs: Forwarded to ``TabICLRegressor``.
    """

    def __init__(
        self, device: str = "cpu", n_estimators: int = 1, n_sample_levels: int = 200, **backend_kwargs: Any
    ) -> None:
        super().__init__("tabicl", device=device, **backend_kwargs)
        self.n_estimators = int(n_estimators)
        self._sample_levels = _midpoint_levels(int(n_sample_levels))
        self._model: Any = None

    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a fresh TabICL regressor on the observed context.

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].
        """
        backend = require_backend("tabicl")
        self._model = backend.TabICLRegressor(
            device=self.device,
            n_estimators=self.n_estimators,
            **self.backend_kwargs,
        )
        self._model.fit(X, y)

    def _predict_backend(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Read the predicted quantile function and integrate it.

        Args:
            X: Candidate coordinates, shape [N, D].

        Returns:
            ``(mean, std)``, each shape [N].
        """
        quantiles = np.asarray(
            self._model.predict(X, output_type="quantiles", alphas=list(_QUANTILE_ALPHAS)),
            dtype=np.float64,
        )                                                  # [N, Q]
        return _moments_from_quantiles(quantiles)

    def sample_marginal(
        self, X: np.ndarray, rng: np.random.Generator, temperature: float = 1.0
    ) -> np.ndarray:
        """Draw per-site Thompson samples by inverting TabICL's own quantile function.

        The quantile function is evaluated on ``n_sample_levels`` equally spaced levels (not the coarse 15-level
        summary grid, which would collapse the outer 2 % of mass onto its end points), so a skewed or
        multi-modal predictive is sampled faithfully rather than flattened into a Gaussian.

        Args:
            X: Candidate coordinates, shape [N, D].
            rng: Seeded generator.
            temperature: Variance scaling applied around the predictive mean.

        Returns:
            One sample per candidate, shape [N].

        Raises:
            ValueError: If ``temperature`` is not positive.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        quantiles = np.asarray(
            self._model.predict(X, output_type="quantiles", alphas=list(self._sample_levels)),
            dtype=np.float64,
        )                                                  # [N, n_sample_levels]
        return _inverse_cdf_draws(np.sort(quantiles, axis=1), self._sample_levels, rng, temperature)


class CausiloSurrogate(ExternalSurrogate):
    """Causilo surrogate -- **native regression with a quantile predictive distribution**.

    ``CausiloRegressor.predict(X, output_type='quantiles', quantiles=[...])`` returns the predicted
    quantile function (its native grid is 999 levels, 0.001-0.999), which this wrapper integrates into a
    mean and a standard deviation and samples by inverse-CDF, exactly as :class:`TabICLSurrogate` does.
    No bucketization is involved, so Causilo is a first-class row of the Hyp 0 table.

    Requires torch >= 2.13 and Python >= 3.10 (upstream declaration), i.e. the ``latest`` environment.
    Not yet executed against the real package: the call signature is taken from upstream's
    ``docs/inference.md``, so the first cell run must confirm it.

    Args:
        device: Torch device string (``'cpu'`` or ``'cuda'``; upstream's own default is ``'auto'``).
        n_estimators: Ensemble members. Upstream defaults to 8; the BO loops use 1, one forward per step,
            like every other model of the benchmark.
        n_sample_levels: Resolution of the quantile function a Thompson draw inverts.
        **backend_kwargs: Forwarded to ``CausiloRegressor``.
    """

    def __init__(
        self, device: str = "cpu", n_estimators: int = 1, n_sample_levels: int = 200, **backend_kwargs: Any
    ) -> None:
        super().__init__("causilo", device=device, **backend_kwargs)
        self.n_estimators = int(n_estimators)
        self._sample_levels = _midpoint_levels(int(n_sample_levels))
        self._model: Any = None

    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a fresh Causilo regressor on the observed context.

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].
        """
        backend = require_backend("causilo")
        self._model = backend.CausiloRegressor(
            device=self.device, n_estimators=self.n_estimators, **self.backend_kwargs
        )
        self._model.fit(X, y)

    def _quantiles(self, X: np.ndarray) -> np.ndarray:
        """Predicted quantile function at the summary levels.

        Args:
            X: Candidate coordinates, shape [N, D].

        Returns:
            Quantiles, shape [N, Q], sorted along axis 1.

        Raises:
            RuntimeError: If upstream returns a shape other than [N, Q].
        """
        q = np.asarray(
            self._model.predict(X, output_type="quantiles", quantiles=list(_QUANTILE_ALPHAS)),
            dtype=np.float64,
        )                                                  # [N, Q]
        if q.shape != (X.shape[0], len(_QUANTILE_ALPHAS)):
            raise RuntimeError(
                f"causilo returned quantiles of shape {q.shape}, expected "
                f"({X.shape[0]}, {len(_QUANTILE_ALPHAS)}) (upstream API changed?)."
            )
        return np.sort(q, axis=1)

    def _predict_backend(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the predicted quantile function into a mean and a standard deviation.

        Args:
            X: Candidate coordinates, shape [N, D].

        Returns:
            ``(mean, std)``, each shape [N].
        """
        return _moments_from_quantiles(self._quantiles(X))

    def sample_marginal(
        self, X: np.ndarray, rng: np.random.Generator, temperature: float = 1.0
    ) -> np.ndarray:
        """Draw per-site Thompson samples by inverting Causilo's own quantile function.

        Args:
            X: Candidate coordinates, shape [N, D].
            rng: Seeded generator.
            temperature: Variance scaling applied around the predictive mean.

        Returns:
            One sample per candidate, shape [N].

        Raises:
            ValueError: If ``temperature`` is not positive.
            RuntimeError: If upstream returns a shape other than [N, n_sample_levels].
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        q = np.asarray(
            self._model.predict(X, output_type="quantiles", quantiles=list(self._sample_levels)),
            dtype=np.float64,
        )                                                  # [N, n_sample_levels]
        if q.shape != (X.shape[0], len(self._sample_levels)):
            raise RuntimeError(
                f"causilo returned quantiles of shape {q.shape}, expected "
                f"({X.shape[0]}, {len(self._sample_levels)}) (upstream API changed?)."
            )
        return _inverse_cdf_draws(np.sort(q, axis=1), self._sample_levels, rng, temperature)


class TabPFN35Surrogate(ExternalSurrogate):
    """TabPFN-3.5 surrogate: the same bar-distribution predictive as TabPFN-2.5, newer checkpoint.

    Builds the regressor with ``TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, ...)`` and
    reads it through the existing :class:`~pfns4neurostim.models.pfn.tabpfn.TabPFNSurrogate` (public API
    only: ``fit`` and ``predict(output_type='full')``), so TabPFN-2.5 and TabPFN-3.5 differ only in the
    checkpoint. Needs ``tabpfn >= 9`` (``latest`` env); the main env pins 6.3.2 for TabPFN-2.5.
    Not yet executed against the real package: ``output_type='full'`` keeping its ``logits`` /
    ``criterion`` keys in 9.x is assumed, and the first cell run must confirm it.

    Args:
        device: Torch device string.
        n_estimators: Ensemble members; 1 for BO loops, as for TabPFN-2.5.
        **backend_kwargs: Forwarded to ``TabPFNRegressor.create_default_for_version``.
    """

    def __init__(self, device: str = "cpu", n_estimators: int = 1, **backend_kwargs: Any) -> None:
        super().__init__("tabpfn_v3_5", device=device, **backend_kwargs)
        self.n_estimators = int(n_estimators)
        self._inner: Any = None

    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a fresh TabPFN-3.5 regressor on the observed context.

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].
        """
        from tabpfn.constants import ModelVersion  # noqa: PLC0415 - V3_5 exists only in tabpfn >= 9

        from .tabpfn import TabPFNSurrogate  # noqa: PLC0415 - imports tabpfn at module level

        regressor = self._backend.TabPFNRegressor.create_default_for_version(
            ModelVersion.V3_5,
            device=self.device,
            n_estimators=self.n_estimators,
            ignore_pretraining_limits=True,
            **self.backend_kwargs,
        )
        self._inner = TabPFNSurrogate(regressor)
        self._inner.fit(X, y)

    def _predict_backend(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Moments of the bar distribution.

        Args:
            X: Candidate coordinates, shape [N, D].

        Returns:
            ``(mean, std)``, each shape [N].
        """
        return self._inner.predict(X)

    def predict_ts(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Per-site Thompson draw from the bar distribution (picked up by the surrogate adapter).

        Args:
            X: Candidate coordinates, shape [N, D].
            temperature: Softmax temperature of the bar distribution.

        Returns:
            One sample per candidate, shape [N].
        """
        return self._inner.predict_ts(np.asarray(X, dtype=np.float64), temperature=temperature)


class TabFlexSurrogate(BucketizedClassifierSurrogate):
    """TabFlex as a bucketized regressor (classification-head adaptation).

    TabFlex is classification-only upstream, so the standardized response is
    binned and the classifier's class probabilities are read as a bar distribution
    (see :mod:`~pfns4neurostim.models.pfn.bar_distribution`). Every result row
    must be labelled "classification-head adaptation": its resolution is bounded
    by the bin width, which is a property of the adaptation, not of the model.

    **Weights unavailable (checked 2026-09-20):** upstream downloads its checkpoints from
    ``amuellermothernet.blob.core.windows.net``, which no longer resolves (microsoft/ticl issue #27,
    NXDOMAIN); no mirror exists, so this wrapper cannot run unless the ``.cpkt`` files are placed in
    ``libs/ticl/ticl/models_diff/`` by hand. It is therefore left out of the D3 benchmark.

    Upstream's ``TabFlex`` convenience class hardcodes ``device='cuda'`` and picks
    one of three checkpoints by data size. This wrapper calls the underlying
    ``TabPFNClassifier`` directly so the device is configurable, and defaults to
    the small-data checkpoint, which is the one upstream's own routing selects for
    our regime (N < 3000, D <= 100).

    Args:
        device: Torch device string.
        n_bins: Number of response bins.
        n_ensemble_configurations: Upstream ensemble size.
        model_string: Checkpoint stem; defaults to the small-data TabFlex model.
        epoch: Checkpoint epoch tag.
        **backend_kwargs: Forwarded to ``TabPFNClassifier``.
    """

    #: Upstream's small-data checkpoint, selected by TabFlex.fit for N<3000, D<=100.
    DEFAULT_MODEL = "ssm_tabpfn_modellinear_attention_08_28_2024_19_00_44"
    DEFAULT_EPOCH = "3110"

    def __init__(
        self,
        device: str = "cpu",
        n_bins: int = 32,
        n_ensemble_configurations: int = 3,
        model_string: str | None = None,
        epoch: str | None = None,
        **backend_kwargs: Any,
    ) -> None:
        super().__init__("tabflex", device=device, n_bins=n_bins, **backend_kwargs)
        self.n_ensemble_configurations = int(n_ensemble_configurations)
        self.model_string = model_string or self.DEFAULT_MODEL
        self.epoch = epoch or self.DEFAULT_EPOCH

    def _make_classifier(self) -> Any:
        """Construct the underlying ticl classifier, fetching weights on first use.

        Returns:
            A ``ticl.prediction.tabpfn.TabPFNClassifier`` on the requested device.
        """
        from ticl.prediction.tabpfn import TabPFNClassifier  # noqa: PLC0415 - submodule
        from ticl.utils import fetch_model  # noqa: PLC0415 - submodule

        # Downloads into libs/ticl/ticl/models_diff/ on first use (excluded locally
        # from the submodule's index; see scripts/mila_setup.sh submodules).
        fetch_model(f"{self.model_string}_epoch_{self.epoch}.cpkt")
        return TabPFNClassifier(
            device=self.device,
            model_string=self.model_string,
            N_ensemble_configurations=self.n_ensemble_configurations,
            epoch=self.epoch,
            **self.backend_kwargs,
        )


#: Predictive-SD constructions of :class:`TabFMSurrogate` (TabFM itself emits a point prediction only).
TABFM_PREDICTIVE_SD: tuple[str, ...] = ("spread_oof", "spread")


class TabFMSurrogate(ExternalSurrogate):
    """Google TabFM surrogate — native point regression with a constructed predictive SD.

    TabFM's regression head emits **one value per row** (``_check_regressor_output_dim``): it is a
    point regressor, trained and ranked (TabArena) on point accuracy, with no predictive distribution.
    BO needs one, so the wrapper builds a Gaussian marginal around upstream's own point prediction:

    * **mean** = ``TabFMRegressor._combine_predictions`` of the members, i.e. exactly what
      ``predict`` returns, in the caller's response scale. Upstream standardizes ``y`` on the
      context internally; the members of ``_predict_internal`` live in that internal scale and must
      be inverse-transformed (before 2026-09-25 they were not, so R^2 / NLL / CRPS / coverage were
      computed on the wrong scale; the argmax-based BO choices were unaffected).
    * **std** (``predictive_sd``):

      - ``'spread'``: disagreement across ensemble members only. The members are deterministic
        views of the same context (feature order x ``none``/``power`` normalization), so with two
        features there are at most four distinct views and the spread is near zero: Thompson
        sampling then collapses to greedy exploitation (90% coverage 0.04 on NHP, 2026-09-24).
      - ``'spread_oof'`` (default): ``sqrt(spread^2 + s_oof^2)``, where ``s_oof`` is the RMSE of
        upstream's out-of-fold predictions of the context (``_compute_oof_preds_scaled``, k-fold,
        k = min(``oof_folds``, n)). A homoscedastic residual scale, the same idea as split-conformal
        intervals; still a constructed SD, not the model's own, and must be labelled so (G3).

    Requires Python >= 3.11 (upstream declaration), so it cannot run in the
    project's pinned 3.9 environment.

    Args:
        device: Torch device string (TabFM also has a JAX backend).
        n_estimators: Ensemble members; must be >= 2 for a usable spread.
        tabfm_backend: ``'torch'`` (default) or ``'jax'`` upstream checkpoint implementation.
        predictive_sd: ``'spread_oof'`` (default) or ``'spread'``.
        oof_folds: Folds of the out-of-fold residual estimate (capped at the context size).
        batch_size: Ensemble members per forward pass, passed to ``TabFMRegressor``; ``0`` = all at
            once. Upstream's default of 1 runs the members one by one: measured 2026-09-25 on NHP,
            0 is 2.7x faster per BO step and makes the 5-fold OOF estimate cost ~30% over the old step.
        **backend_kwargs: Forwarded to ``TabFMRegressor``.

    Raises:
        ValueError: If ``n_estimators`` < 2, ``oof_folds`` < 2 or ``predictive_sd`` is unknown.
    """

    #: Loaded checkpoints, shared across instances (one per backend and device).
    _CHECKPOINTS: dict[tuple[str, str], Any] = {}

    def __init__(
        self,
        device: str = "cpu",
        n_estimators: int = 8,
        tabfm_backend: str = "torch",
        predictive_sd: str = "spread_oof",
        oof_folds: int = 5,
        batch_size: int = 0,
        **backend_kwargs: Any,
    ) -> None:
        if tabfm_backend not in ("torch", "jax"):
            raise ValueError(f"tabfm_backend must be 'torch' or 'jax', got {tabfm_backend!r}.")
        if predictive_sd not in TABFM_PREDICTIVE_SD:
            raise ValueError(f"predictive_sd must be one of {TABFM_PREDICTIVE_SD}, got {predictive_sd!r}.")
        if oof_folds < 2:
            raise ValueError(f"oof_folds must be >= 2, got {oof_folds}.")
        if n_estimators < 2:
            raise ValueError(
                "TabFMSurrogate needs n_estimators >= 2: its only uncertainty signal is "
                "the spread across ensemble members, which is identically zero for one."
            )
        super().__init__("tabfm", device=device, batch_size=int(batch_size), **backend_kwargs)
        self.tabfm_backend = tabfm_backend
        self.n_estimators = int(n_estimators)
        self.predictive_sd = predictive_sd
        self.oof_folds = int(oof_folds)
        self._model: Any = None
        self._oof_sd: float = 0.0

    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a fresh TabFM regressor on the observed context.

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].
        """
        backend = require_backend("tabfm")
        del backend
        self._model = self._regressor(self._load_checkpoint())
        self._model.fit(X, y)
        self._oof_sd = self._oof_residual_sd(y) if self.predictive_sd == "spread_oof" else 0.0

    def _oof_residual_sd(self, y: np.ndarray) -> float:
        """RMSE of upstream's out-of-fold predictions of the context, in the response scale.

        Args:
            y: Observed responses, shape [n].

        Returns:
            The residual scale ``s_oof`` (0.0 for a context too small to split).

        Raises:
            RuntimeError: If the out-of-fold predictions are non-finite.
        """
        n = int(y.shape[0])
        if n < 2:
            return 0.0
        scaled, val_idx = self._model._compute_oof_preds_scaled(cv=min(self.oof_folds, n))   # [E, n]
        pred = np.asarray(self._model._combine_predictions(np.asarray(scaled, dtype=np.float64)))  # [n]
        idx = np.arange(n) if val_idx is None else np.asarray(val_idx)
        resid = np.asarray(y, dtype=np.float64)[idx] - pred[idx]                             # [n_val]
        if not np.isfinite(resid).all():
            raise RuntimeError(f"TabFM out-of-fold predictions are non-finite (n={n}).")
        return float(np.sqrt(np.mean(resid ** 2)))

    def _load_checkpoint(self) -> Any:
        """Load the TabFM regression checkpoint once per process and backend.

        Upstream's ``TabFMRegressor`` takes a *loaded model* (``model=``), not a device or a name;
        reloading it at every BO step would dominate the cost, so it is cached on the class.

        Returns:
            The upstream model object for ``model_type='regression'``.
        """
        key = (self.tabfm_backend, self.device)
        if key not in TabFMSurrogate._CHECKPOINTS:
            if self.tabfm_backend == "torch":
                from tabfm import tabfm_v1_0_0_pytorch as upstream  # noqa: PLC0415 - optional dep

                # Upstream defaults to device='cpu' and bfloat16 compute: on CPU that is ~340 s to
                # predict 96 sites (measured 2026-09-20), so the configured device must be honoured.
                model = upstream.load(model_type="regression", device=self.device)
            else:
                from tabfm import tabfm_v1_0_0_jax as upstream  # noqa: PLC0415 - optional dep

                model = upstream.load(model_type="regression")
            TabFMSurrogate._CHECKPOINTS[key] = model
        return TabFMSurrogate._CHECKPOINTS[key]

    def _regressor(self, model: Any) -> Any:
        """Build upstream's sklearn-style regressor around a loaded checkpoint.

        Args:
            model: Object returned by :meth:`_load_checkpoint`.

        Returns:
            An unfitted ``TabFMRegressor``.
        """
        from tabfm import TabFMRegressor  # noqa: PLC0415 - optional dep

        return TabFMRegressor(model=model, n_estimators=self.n_estimators, **self.backend_kwargs)

    def _predict_backend(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return upstream's point prediction and the constructed predictive SD.

        Args:
            X: Candidate coordinates, shape [N, D].

        Returns:
            ``(mean, std)``, each shape [N], in the response scale passed to ``fit``.
        """
        scaled = np.asarray(self._model._predict_internal(X), dtype=np.float64)    # [E, N], upstream's z-scale
        if scaled.ndim == 1:
            scaled = scaled[None, :]
        mean = np.asarray(self._model._combine_predictions(scaled), dtype=np.float64)  # [N], == predict(X)
        members = np.stack([self._model._inverse_transform_y(m) for m in scaled])      # [E, N], response scale
        spread = members.std(axis=0, ddof=1) if members.shape[0] > 1 else np.zeros_like(mean)  # [N]
        std = np.sqrt(spread ** 2 + self._oof_sd ** 2)                                  # [N]
        # A zero SD would make EI and TS degenerate, so floor it at a small
        # fraction of the response scale rather than returning a fake certainty.
        floor = 1e-6 * max(float(np.ptp(mean)), 1.0)
        return mean, np.maximum(std, floor)


#: Vendored HEBO+ checkpoint (the authors' main model), relative to the repository root. The pip
#: package ships no weights, and unzipping into ``libs/`` would modify a read-only submodule, so the
#: gzip archive is decompressed in memory.
PFNS4BO_DEFAULT_CHECKPOINT: str = "libs/PFNs4BO/pfns4bo/final_models/model_hebo_morebudget_9_unused_features_3.pt.gz"
_REPO_ROOT: Path = Path(__file__).resolve().parents[4]   # src/pfns4neurostim/models/pfn/wrappers.py -> repo


@lru_cache(maxsize=2)
def _load_pfns4bo_model(path: str) -> Any:
    """Load a PFNs4BO transformer once per process (about 100 MB, pickled whole).

    Args:
        path: Absolute path of a ``.pt`` or ``.pt.gz`` checkpoint.

    Returns:
        The transformer in eval mode, on CPU.

    Raises:
        FileNotFoundError: If the checkpoint is missing (submodule not initialised).
    """
    import torch  # noqa: PLC0415 - heavy import, load on demand

    ckpt = Path(path)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"PFNs4BO checkpoint not found: {ckpt}. Initialise the submodule "
            "(`bash scripts/mila_setup.sh submodules`) or set `checkpoint` in the model params."
        )
    if ckpt.suffix == ".gz":
        with gzip.open(ckpt, "rb") as fh:
            source: Any = io.BytesIO(fh.read())
    else:
        source = ckpt
    model = torch.load(source, map_location="cpu", weights_only=False)
    return model.eval()


class PFNs4BOSurrogate(ExternalSurrogate):
    """PFNs4BO (HEBO prior): an end-to-end BO model with a **native policy**.

    PFNs4BO is a PFN *and* an acquisition rule: its transformer scores every candidate
    with its own criterion (EI by default, the authors' HPO-B setting) computed inside
    the pipeline. That surface is exposed as :meth:`policy_scores`, which the ``native``
    acquisition type delegates to, so the shared BO loop selects the query (random
    tie-break, re-querying) exactly as for any other model.

    The predictive summary (:meth:`_predict_backend`) reads the same network's bar
    distribution on the same context. It is taken on the z-scored response *without*
    the authors' internal power transform, so mean and std live in the response scale
    that R-squared and the calibration metrics assume; the policy itself keeps the
    authors' default pipeline.

    Args:
        device: Torch device string.
        checkpoint: Checkpoint path (``.pt`` or ``.pt.gz``); relative paths resolve
            against the repository root.
        acq_function: PFNs4BO criterion (``'ei'``, ``'pi'``, ``'ucb'``, ``'mean'``).
        **acq_kwargs: Further keyword arguments of
            ``pfns4bo.scripts.acquisition_functions.general_acq_function``.
    """

    def __init__(
        self,
        device: str = "cpu",
        checkpoint: str = PFNS4BO_DEFAULT_CHECKPOINT,
        acq_function: str = "ei",
        **acq_kwargs: Any,
    ) -> None:
        super().__init__("pfns4bo", device=device, **acq_kwargs)
        from pfns4bo.scripts.acquisition_functions import TransformerBOMethod  # noqa: PLC0415

        path = Path(checkpoint)
        self.checkpoint = str(path if path.is_absolute() else _REPO_ROOT / path)
        self.acq_function = acq_function
        self._model = _load_pfns4bo_model(self.checkpoint)
        self._method = TransformerBOMethod(
            self._model, device=device, acq_function=acq_function, **acq_kwargs
        )
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store the context; the transformer is conditioned at query time.

        Args:
            X: Observed coordinates in [0, 1]^D, shape [n, D].
            y: Observed responses (z-scored), shape [n].
        """
        self._X, self._y = X, y

    def policy_scores(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Score every candidate with PFNs4BO's own acquisition criterion.

        Uses the official ``observe_and_suggest(..., return_actual_ei=True)`` and keeps
        the *surface*, discarding its internally drawn index so that tie-breaking stays
        with the shared loop. (The official call draws that discarded
        index from the global torch RNG, which the runner seeds per repetition.)

        Args:
            X: Candidate coordinates in [0, 1]^D, shape [N, D].
            rng: Unused; the criterion is deterministic given the context.

        Returns:
            Criterion value per candidate, shape [N]; higher is preferred.

        Raises:
            RuntimeError: If called before :meth:`fit`, or on non-finite scores.
        """
        if not self._fitted:
            raise RuntimeError("pfns4bo.policy_scores called before fit().")
        _, scores = self._method.observe_and_suggest(self._X, self._y, X, return_actual_ei=True)
        scores = np.asarray(scores.numpy(), dtype=np.float64).reshape(-1)   # [N]
        if not np.isfinite(scores).all():
            raise RuntimeError("pfns4bo.policy_scores produced non-finite values.")
        return scores

    def _predict_backend(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and std from the bar distribution at each candidate.

        Args:
            X: Candidate coordinates in [0, 1]^D, shape [N, D].

        Returns:
            ``(mean, std)``, each shape [N].
        """
        import torch  # noqa: PLC0415 - heavy import, load on demand

        device = torch.device(self.device)
        self._model.to(device)
        x_given = torch.as_tensor(self._X, dtype=torch.float32, device=device).unsqueeze(1)   # [n, 1, D]
        y_given = torch.as_tensor(self._y, dtype=torch.float32, device=device).unsqueeze(1)   # [n, 1]
        x_eval = torch.as_tensor(X, dtype=torch.float32, device=device).unsqueeze(1)          # [N, 1, D]
        with torch.no_grad():
            logits = self._model(x_given, y_given, x_eval)                                    # [N, 1, n_bins]
            criterion = self._model.criterion
            mean = criterion.mean(logits).reshape(-1)                                         # [N]
            var = criterion.variance(logits).reshape(-1)                                      # [N]
        return mean.cpu().numpy(), var.clamp_min(1e-12).sqrt().cpu().numpy()


class MitraSurrogate(ExternalSurrogate):
    """Mitra regression surrogate (native regression, via AutoGluon).

    Args:
        device: Torch device string.
        **backend_kwargs: Forwarded to the backend regressor.
    """

    def __init__(self, device: str = "cpu", **backend_kwargs: Any) -> None:
        super().__init__("mitra", device=device, **backend_kwargs)

    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Mitra regressor. **Not implemented.**

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].

        Raises:
            NotImplementedError: Task #8 Step 3; predictive-distribution access
                must be confirmed first.
        """
        raise NotImplementedError(
            "MitraSurrogate is not implemented yet (task #8 Step 3); confirm first that "
            "AutoGluon exposes a predictive distribution and not only a point prediction."
        )
