"""Wrappers for the external amortized surrogates of Hypothesis 0 (task #8).

Each benchmark model is reached through its **official implementation** — a pinned
pip dependency, a git submodule under ``libs/``, or an optional extra — and exposed
through one thin wrapper conforming to the package's surrogate interface. Nothing
here re-implements a model.

Availability is optional by design: these dependencies are heavy and some conflict
with the pinned Python 3.9 / torch 2.5.1 stack, so each wrapper imports its backend
lazily and, when it is missing, raises a message naming the extra to install rather
than failing at import time. ``availability()`` reports the whole table at once, so
a benchmark config can be validated before a cluster job starts.

Integration routes (web check 2026-09-16, recorded in the roadmap H0-1 table):

======================  =========================================  ==================================
Model                   Source                                     Regression route
======================  =========================================  ==================================
PFNs4BO (HEBO prior)    ``libs/PFNs4BO`` submodule + ``pfns4bo``   native, BO-specific
TabPFN v1               ``tabpfn<2`` in an isolated extra          classification-head adaptation
Google TabFM            ``google-research/tabfm``                  native
Mitra                   ``autogluon.tabular`` >= 1.4               native regressor
TabFlex                 ``microsoft/ticl`` submodule               classification-head adaptation
======================  =========================================  ==================================

The two "classification-head adaptation" rows go through
:mod:`~pfns4neurostim.models.pfn.bar_distribution` and **must be labelled as such**
in every table and caption: their resolution is bounded by the bin width, so they
are not like-for-like against TabPFN v2.5's native, learned bar distribution.
"""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bar_distribution import BarDistribution, quantile_borders

__all__ = [
    "ExternalSurrogate",
    "BucketizedClassifierSurrogate",
    "ExternalSpec",
    "EXTERNAL_SPECS",
    "availability",
    "require_backend",
]


@dataclass(frozen=True)
class ExternalSpec:
    """How to reach one external model.

    Attributes:
        key: Canonical model key.
        module: Python module that must be importable for the model to run.
        extra: ``pip install -e '.[<extra>]'`` group that provides it.
        route: ``'native'`` or ``'classification-head adaptation'``.
        source: Official implementation URL.
        dist: Installed distribution name, when it differs from ``module``.
        major_below: If set, the installed major version must be below this.
            TabPFN v1 and v2.5 share the module name ``tabpfn``, so importability
            alone would report v1 as available whenever v2.5 is installed.
        notes: Constraints worth knowing before installing.
    """

    key: str
    module: str
    extra: str
    route: str
    source: str
    dist: str = ""
    major_below: int | None = None
    notes: str = ""


EXTERNAL_SPECS: dict[str, ExternalSpec] = {
    "pfns4bo": ExternalSpec(
        key="pfns4bo",
        module="pfns4bo",
        extra="pfns4bo",
        route="native",
        source="https://github.com/automl/PFNs4BO",
        notes="Vendored weights in libs/PFNs4BO/pfns4bo/final_models/*.pt.gz; pip pkg installed (0.1.5).",
    ),
    "tabpfn_v1": ExternalSpec(
        key="tabpfn_v1",
        module="tabpfn",
        extra="tabpfn-v1",
        route="classification-head adaptation",
        source="https://github.com/automl/TabPFN",
        dist="tabpfn",
        major_below=2,
        notes="Requires tabpfn<2, which conflicts with the installed 6.3.2: use a separate env.",
    ),
    "tabfm": ExternalSpec(
        key="tabfm",
        module="tabfm",
        extra="tabfm",
        route="native",
        source="https://github.com/google-research/tabfm",
        notes="Released 2026-06-30; check its Python/JAX requirements against Python 3.9.25.",
    ),
    "mitra": ExternalSpec(
        key="mitra",
        module="autogluon.tabular",
        extra="mitra",
        route="native",
        source="https://huggingface.co/autogluon/mitra-regressor",
        notes="Heavy install (AutoGluon >= 1.4); verify predictive-distribution access.",
    ),
    "tabicl": ExternalSpec(
        key="tabicl",
        module="tabicl",
        extra="tabicl",
        route="classification-head adaptation",
        source="https://github.com/soda-inria/tabicl",
        notes=(
            "Added 2026-09-20 at the user's request; it was not in the original H0-1 "
            "table. Upstream is classification-focused (TabICLClassifier), so it is "
            "registered on the bucketized route; if a native regressor exists in the "
            "installed version, switch the wrapper to ExternalSurrogate and update "
            "this route before reporting any result."
        ),
    ),
    "tabflex": ExternalSpec(
        key="tabflex",
        module="ticl",
        extra="tabflex",
        route="classification-head adaptation",
        source="https://github.com/microsoft/ticl",
        notes="Add libs/ticl as a submodule; regression via naive binning upstream.",
    ),
}


def availability() -> dict[str, bool]:
    """Report which external backends are importable in this environment.

    Returns:
        Mapping model key -> whether its backend module can be imported.
    """
    return {key: _backend_ok(spec)[0] for key, spec in EXTERNAL_SPECS.items()}


def _backend_ok(spec: ExternalSpec) -> tuple[bool, str]:
    """Check whether one backend is importable *and* of the required version.

    Args:
        spec: The external model's spec.

    Returns:
        ``(ok, reason)``; ``reason`` is empty when ok.
    """
    try:
        # find_spec raises rather than returning None when a *parent* package is
        # missing (e.g. 'autogluon.tabular' with no 'autogluon' installed).
        if importlib.util.find_spec(spec.module) is None:
            return False, f"module {spec.module!r} not installed"
    except (ImportError, ValueError):
        return False, f"module {spec.module!r} not installed"

    if spec.major_below is not None:
        dist = spec.dist or spec.module
        try:
            from importlib.metadata import version as _dist_version  # noqa: PLC0415

            installed = _dist_version(dist)
        except Exception:  # noqa: BLE001 - absence or metadata failure both mean unknown
            return False, f"cannot read the installed version of {dist!r}"
        major = int(str(installed).split(".", 1)[0]) if str(installed)[:1].isdigit() else -1
        if major < 0 or major >= spec.major_below:
            return False, (
                f"{dist} {installed} is installed but this model needs major version "
                f"< {spec.major_below}; the two share the module name, so they cannot "
                "coexist in one environment"
            )
    return True, ""


def require_backend(key: str) -> Any:
    """Import an external model's backend or raise with install instructions.

    Args:
        key: Model key in :data:`EXTERNAL_SPECS`.

    Returns:
        The imported backend module.

    Raises:
        KeyError: If the key is unknown.
        ImportError: If the backend is not installed, naming the extra to add.
    """
    if key not in EXTERNAL_SPECS:
        raise KeyError(f"Unknown external model {key!r}. Known: {sorted(EXTERNAL_SPECS)}.")
    spec = EXTERNAL_SPECS[key]
    ok, reason = _backend_ok(spec)
    if not ok:
        raise ImportError(
            f"Model {key!r} is unavailable: {reason}. "
            f"Install it with: pip install -e '.[{spec.extra}]'  "
            f"(official implementation: {spec.source}). {spec.notes}"
        )
    return importlib.import_module(spec.module)


class ExternalSurrogate:
    """Base for external-model wrappers: fit on a context, predict marginals.

    Subclasses implement :meth:`_fit_backend` and :meth:`_predict_backend`. The
    base handles input validation and the fail-fast checks the rest of the package
    relies on.

    Args:
        key: Model key in :data:`EXTERNAL_SPECS`.
        device: Torch device string, where the backend accepts one.
        **backend_kwargs: Passed through to the backend constructor.
    """

    def __init__(self, key: str, device: str = "cpu", **backend_kwargs: Any) -> None:
        self.key = key
        self.device = device
        self.backend_kwargs = dict(backend_kwargs)
        self._backend = require_backend(key)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Condition the model on observed data.

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].

        Raises:
            RuntimeError: On non-finite inputs.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise RuntimeError(f"{self.key}.fit received non-finite inputs.")
        self._fit_backend(X, y)
        self._fitted = True

    def predict_marginals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean and standard deviation per candidate.

        Args:
            X: Candidate coordinates, shape [N, D].

        Returns:
            ``(mean, std)``, each shape [N].

        Raises:
            RuntimeError: If called before :meth:`fit`, or on non-finite output.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.key}.predict_marginals called before fit().")
        mean, std = self._predict_backend(np.asarray(X, dtype=np.float64))
        mean = np.asarray(mean, dtype=np.float64)   # [N]
        std = np.asarray(std, dtype=np.float64)     # [N]
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            raise RuntimeError(f"{self.key}.predict_marginals produced non-finite values.")
        return mean, std

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Alias of :meth:`predict_marginals`."""
        return self.predict_marginals(X)

    # --- subclass hooks -----------------------------------------------------
    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the backend model. Implemented by subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _fit_backend yet (task #8 Step 3)."
        )

    def _predict_backend(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with the backend model. Implemented by subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _predict_backend yet (task #8 Step 3)."
        )


class BucketizedClassifierSurrogate(ExternalSurrogate):
    """Regression through a classification head, via a bar distribution.

    Discretizes the observed responses into ``n_bins`` quantile bins, fits the
    backend classifier on the bin labels, and reads its class probabilities as a
    piecewise-uniform distribution over the response axis.

    Results from this wrapper must be labelled **"classification-head adaptation"**:
    no prediction can be sharper than one bin, which is a property of the
    adaptation, not of the model.

    Args:
        key: Model key.
        device: Torch device string.
        n_bins: Number of response bins K.
        **backend_kwargs: Passed to the backend constructor.
    """

    def __init__(self, key: str, device: str = "cpu", n_bins: int = 32, **backend_kwargs: Any) -> None:
        super().__init__(key, device=device, **backend_kwargs)
        self.n_bins = int(n_bins)
        self._bar: BarDistribution | None = None
        self._classifier: Any = None

    def _make_classifier(self) -> Any:
        """Construct the backend classifier. Implemented by subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _make_classifier yet (task #8 Step 3)."
        )

    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Bin the responses and fit the backend classifier on the labels.

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].
        """
        # With very few observations, K quantile bins would mostly be empty;
        # cap the resolution at the number of distinct observed values.
        n_bins = max(2, min(self.n_bins, int(np.unique(y).size)))
        self._bar = BarDistribution(quantile_borders(y, n_bins))
        labels = self._bar.digitize(y)                      # [n]
        self._classifier = self._make_classifier()
        self._classifier.fit(X, labels)

    def _predict_backend(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Read the classifier's class probabilities as a bar distribution.

        Args:
            X: Candidate coordinates, shape [N, D].

        Returns:
            ``(mean, std)``, each shape [N].
        """
        assert self._bar is not None and self._classifier is not None
        probs = np.asarray(self._classifier.predict_proba(X), dtype=np.float64)   # [N, C]
        classes = np.asarray(getattr(self._classifier, "classes_", np.arange(probs.shape[1])), dtype=int)
        full = self._bar.expand(probs, classes)                                   # [N, K]
        return self._bar.mean(full), self._bar.std(full)

    def sample_marginal(
        self, X: np.ndarray, rng: np.random.Generator, temperature: float = 1.0
    ) -> np.ndarray:
        """Draw per-site samples directly from the bar distribution.

        Args:
            X: Candidate coordinates, shape [N, D].
            rng: Seeded generator.
            temperature: Accepted for interface compatibility; a bar distribution
                is sampled as-is, so only ``1.0`` is exact.

        Returns:
            One sample per candidate, shape [N].
        """
        assert self._bar is not None and self._classifier is not None
        probs = np.asarray(self._classifier.predict_proba(X), dtype=np.float64)
        classes = np.asarray(getattr(self._classifier, "classes_", np.arange(probs.shape[1])), dtype=int)
        return self._bar.sample(self._bar.expand(probs, classes), rng)
