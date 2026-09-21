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
PFNs4BO       pending    Backend installed; needs our layout mapped onto its input.
TabPFN v1     pending    Needs ``tabpfn<2``, which cannot coexist with the pinned 6.3.2.
Mitra         pending    Needs AutoGluon; predictive-distribution access unconfirmed.
============  =========  ==============================================================

The wrappers for TabICL and TabFM are written and will run as-is under a suitable
interpreter; they are blocked by the environment, not by missing code.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .external import BucketizedClassifierSurrogate, ExternalSurrogate, require_backend

__all__ = [
    "PFNs4BOSurrogate",
    "TabICLSurrogate",
    "TabPFNv1Surrogate",
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
        **backend_kwargs: Forwarded to ``TabICLRegressor``.
    """

    def __init__(self, device: str = "cpu", n_estimators: int = 1, **backend_kwargs: Any) -> None:
        super().__init__("tabicl", device=device, **backend_kwargs)
        self.n_estimators = int(n_estimators)
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
        """Draw per-site Thompson samples by inverting the predicted quantile function.

        Sampling the model's own quantiles keeps the draw faithful to a skewed or
        multi-modal predictive, which a Gaussian summary would flatten.

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
            self._model.predict(X, output_type="quantiles", alphas=list(_QUANTILE_ALPHAS)),
            dtype=np.float64,
        )                                                  # [N, Q]
        alphas = np.asarray(_QUANTILE_ALPHAS)              # [Q]
        u = rng.random(quantiles.shape[0])                 # [N]
        draws = np.array(
            [np.interp(u[i], alphas, quantiles[i]) for i in range(quantiles.shape[0])]
        )                                                  # [N]
        if temperature == 1.0:
            return draws
        mean, _ = _moments_from_quantiles(quantiles)
        return mean + np.sqrt(temperature) * (draws - mean)


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


class TabFMSurrogate(ExternalSurrogate):
    """Google TabFM surrogate — native regression, uncertainty from ensemble spread.

    ``TabFMRegressor.predict`` returns point predictions only. The wrapper therefore
    derives a standard deviation from the spread across ensemble members
    (``_predict_internal`` returns one row per member). **This is a proxy, not a
    calibrated predictive distribution**, and must be labelled as such wherever
    TabFM's calibration metrics appear; with ``n_estimators=1`` there is no spread
    at all, so the wrapper requires at least two members.

    Requires Python >= 3.11 (upstream declaration), so it cannot run in the
    project's pinned 3.9 environment.

    Args:
        device: Torch device string (TabFM also has a JAX backend).
        n_estimators: Ensemble members; must be >= 2 for a usable spread.
        tabfm_backend: ``'torch'`` (default) or ``'jax'`` upstream checkpoint implementation.
        **backend_kwargs: Forwarded to ``TabFMRegressor``.

    Raises:
        ValueError: If ``n_estimators`` < 2.
    """

    #: Loaded checkpoints, shared across instances (one per backend and device).
    _CHECKPOINTS: dict[tuple[str, str], Any] = {}

    def __init__(
        self,
        device: str = "cpu",
        n_estimators: int = 8,
        tabfm_backend: str = "torch",
        **backend_kwargs: Any,
    ) -> None:
        if tabfm_backend not in ("torch", "jax"):
            raise ValueError(f"tabfm_backend must be 'torch' or 'jax', got {tabfm_backend!r}.")
        if n_estimators < 2:
            raise ValueError(
                "TabFMSurrogate needs n_estimators >= 2: its only uncertainty signal is "
                "the spread across ensemble members, which is identically zero for one."
            )
        super().__init__("tabfm", device=device, **backend_kwargs)
        self.tabfm_backend = tabfm_backend
        self.n_estimators = int(n_estimators)
        self._model: Any = None

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
        """Return the ensemble mean and the across-member spread.

        Args:
            X: Candidate coordinates, shape [N, D].

        Returns:
            ``(mean, std)``, each shape [N]; ``std`` is ensemble disagreement.
        """
        members = np.asarray(self._model._predict_internal(X), dtype=np.float64)   # [E, N]
        if members.ndim == 1:
            members = members[None, :]
        mean = members.mean(axis=0)                                                # [N]
        std = members.std(axis=0, ddof=1) if members.shape[0] > 1 else np.zeros_like(mean)
        # A zero spread would make EI and TS degenerate, so floor it at a small
        # fraction of the response scale rather than returning a fake certainty.
        floor = 1e-6 * max(float(np.ptp(mean)), 1.0)
        return mean, np.maximum(std, floor)


class PFNs4BOSurrogate(ExternalSurrogate):
    """PFNs4BO (HEBO prior) surrogate over a discrete candidate pool.

    Args:
        device: Torch device string.
        model_name: Which vendored checkpoint to load.
        **backend_kwargs: Forwarded to the backend.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "hebo_morebudget_9_unused_features_3",
        **backend_kwargs: Any,
    ) -> None:
        super().__init__("pfns4bo", device=device, **backend_kwargs)
        self.model_name = model_name

    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store the context for the PFN forward pass. **Not implemented.**

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].

        Raises:
            NotImplementedError: Task #8 Step 3.
        """
        raise NotImplementedError(
            "PFNs4BOSurrogate is not implemented yet (task #8 Step 3). The backend is "
            "installed; what remains is mapping our (X_pool, observations) layout onto "
            "pfns4bo's transformer input and reading back its bar distribution."
        )


class TabPFNv1Surrogate(BucketizedClassifierSurrogate):
    """TabPFN v1 as a bucketized regressor (classification-head adaptation).

    The adaptation is specified in ``docs/tabpfn_v1_adaptation.md``; what blocks it
    is packaging, not design: v1 needs ``tabpfn<2``, which cannot coexist with the
    pinned ``tabpfn==6.3.2`` used for TabPFN v2.5, so it requires its own
    environment.

    Args:
        device: Torch device string.
        n_bins: Number of response bins (v1 supports at most 10 classes, so this
            is capped at 10 by the adaptation spec).
        **backend_kwargs: Forwarded to ``TabPFNClassifier``.
    """

    def __init__(self, device: str = "cpu", n_bins: int = 10, **backend_kwargs: Any) -> None:
        super().__init__("tabpfn_v1", device=device, n_bins=n_bins, **backend_kwargs)

    def _make_classifier(self) -> Any:
        """Construct the v1 classifier. **Not implemented.**

        Returns:
            A ``tabpfn.TabPFNClassifier`` (v1 API) once the isolated env exists.

        Raises:
            NotImplementedError: Needs the ``tabpfn<2`` environment; see
                ``docs/tabpfn_v1_adaptation.md``.
        """
        raise NotImplementedError(
            "TabPFNv1Surrogate needs an environment with tabpfn<2, which cannot coexist "
            "with the pinned tabpfn 6.3.2. The adaptation itself is specified in "
            "docs/tabpfn_v1_adaptation.md; only the packaging is outstanding."
        )


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
