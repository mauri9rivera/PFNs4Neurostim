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

Integration routes (submodules vendored and APIs read 2026-09-20):

======================  =========================================  ==================================
Model                   Source                                     Regression route
======================  =========================================  ==================================
PFNs4BO (HEBO prior)    ``libs/PFNs4BO`` submodule + ``pfns4bo``   native, BO-specific
TabPFN-3.5              ``tabpfn>=9`` (env ``latest``)             native bar distribution
Causilo                 ``causilo`` (env ``latest``)               native quantile distribution
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
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bar_distribution import BarDistribution, quantile_borders

__all__ = [
    "availability_reasons",
    "ExternalSurrogate",
    "BucketizedClassifierSurrogate",
    "ExternalSpec",
    "EXTERNAL_SPECS",
    "availability",
    "conda_env_for",
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
        major_min: If set, the installed major version must be at least this. TabPFN-3.5 needs
            ``tabpfn>=9`` while the main env pins 6.3.2 (TabPFN-2.5), and both own the module name
            ``tabpfn``, so importability alone would report it available in the wrong environment.
        python_min: Minimum Python version the upstream project declares. Checked
            *before* the import, because the failure is otherwise a confusing
            SyntaxError from inside someone else's package.
        repo_subdir: Path under ``libs/`` holding the vendored submodule, and the
            sub-path within it that must go on ``sys.path`` (``'libs/tabicl/src'``).
            Set when the model is used from the submodule rather than from pip.
        env: Conda environment the model runs in, as a suffix of
            ``pfns4neurostim-<env>`` (``'main'`` means the plain env). Model -> env
            is data here and nowhere else, so a failed availability check can name
            the environment to switch to instead of leaving the caller guessing.
        weights: Path, relative to ``libs/``, that must exist for the model to run,
            or ``''`` when the checkpoint ships inside the wheel or is fetched by the
            backend itself. Checked by :func:`availability`, because an importable
            package with unreachable weights is not a runnable model: TabFlex
            reported available for months while its weight host was NXDOMAIN
            (#8 rule 2, finding N3).
        mem_per_lane_gb: Peak resident memory of one worker lane, in GB. The
            portfolio derives ``--mem`` from this instead of naming numbers by hand.
        max_lanes: Most lanes of this model that may share one job, from its memory.
        gpu: Whether the model needs a GPU to run at a usable speed.
        notes: Constraints worth knowing before installing.
    """

    key: str
    module: str
    extra: str
    route: str
    source: str
    dist: str = ""
    major_min: int | None = None
    python_min: tuple[int, int] | None = None
    repo_subdir: str = ""
    env: str = "main"
    weights: str = ""
    mem_per_lane_gb: float = 4.0
    max_lanes: int = 4
    gpu: bool = True
    notes: str = ""

    @property
    def conda_env(self) -> str:
        """Name of the conda environment this model runs in."""
        return "pfns4neurostim" if self.env == "main" else f"pfns4neurostim-{self.env}"

    @property
    def weights_path(self) -> str:
        """Absolute path of :attr:`weights`, or ``''`` when the model declares none."""
        if not self.weights:
            return ""
        return os.path.join(libs_root(), *self.weights.split("/"))

    def weights_present(self) -> bool:
        """Whether the declared weights exist on this machine (True when none are declared)."""
        path = self.weights_path
        return True if not path else os.path.exists(path)


EXTERNAL_SPECS: dict[str, ExternalSpec] = {
    "pfns4bo": ExternalSpec(
        key="pfns4bo",
        module="pfns4bo",
        extra="pfns4bo",
        route="native",
        source="https://github.com/automl/PFNs4BO",
        weights="PFNs4BO/pfns4bo/final_models/model_hebo_morebudget_9_unused_features_3.pt.gz",
        mem_per_lane_gb=1.0,   # measured 2.9 GB peak for the whole NHP job (Mila 2026-10-01) + headroom
        max_lanes=4,
        notes="Vendored weights in libs/PFNs4BO/pfns4bo/final_models/*.pt.gz; pip pkg installed (0.1.5).",
    ),
    "tabpfn_v3_5": ExternalSpec(
        key="tabpfn_v3_5",
        module="tabpfn",
        extra="tabpfn-v3-5",
        route="native",
        source="https://github.com/PriorLabs/TabPFN",
        dist="tabpfn",
        major_min=9,
        python_min=(3, 10),
        env="latest",
        weights="",   # tabpfn-v3.5-20260909.safetensors is fetched from Hugging Face (Prior-Labs/tabpfn_3_5) on first use
        mem_per_lane_gb=4.8,   # measured 3.82 GB peak RSS, 0.84 GB VRAM, one NHP cell on Narval a100_1g.5gb (2026-10-04) + 25 %
        max_lanes=4,
        notes=(
            "TabPFN-3.5 (tabpfn 9.1.0; non-commercial research licence). Needs the newer `tabpfn`, which "
            "cannot coexist with the 6.3.2 that provides TabPFN-2.5 (same module name), so it runs in the "
            "`latest` environment (environment.latest.yml). Same bar-distribution predictive as v2.5, read "
            "through the same TabPFNSurrogate. The weights are fetched from Hugging Face on first use: "
            "pre-fetch them on a login node (`setup weights latest`), compute nodes are offline."
        ),
    ),
    "causilo": ExternalSpec(
        key="causilo",
        module="causilo",
        extra="causilo",
        route="native (quantile predictive distribution)",
        source="https://github.com/nums-ai/causilo",
        python_min=(3, 10),
        env="latest",
        weights="",   # fetched from Hugging Face (nums-ai/causilo) on first use
        mem_per_lane_gb=2.3,   # measured 1.84 GB peak RSS, 0.15 GB VRAM, one NHP cell on Narval a100_1g.5gb (2026-10-04) + 25 %
        max_lanes=4,
        notes=(
            "Causilo 1.0.x (Nums AI, arXiv 2609.22866; Apache-2.0 code, non-commercial weights licence). "
            "Requires torch >= 2.13 and Python 3.10-3.14, so it runs in the `latest` environment. "
            "CausiloRegressor.predict(output_type='quantiles', quantiles=[...]) returns the predicted "
            "quantile function (999 native levels), which the wrapper integrates exactly as for TabICL. "
            "Upstream's default is n_estimators=8; the BO loops use 1, like every other model here."
        ),
    ),
    "tabfm": ExternalSpec(
        key="tabfm",
        module="tabfm",
        extra="tabfm",
        route="native (point prediction; uncertainty from ensemble spread)",
        source="https://github.com/google-research/tabfm",
        python_min=(3, 11),
        repo_subdir="tabfm",
        env="bench",
        mem_per_lane_gb=12.0,   # measured 9.9 GB RSS + headroom (task plan #8)
        max_lanes=2,
        notes=(
            "Vendored as libs/tabfm (2026-09-20). Requires Python >= 3.11, so it cannot "
            "run in the pinned 3.9 env. TabFMRegressor.predict returns point predictions "
            "only; the wrapper derives uncertainty from the spread across ensemble "
            "members, which is a proxy and not a calibrated predictive distribution."
        ),
    ),
    "mitra": ExternalSpec(
        key="mitra",
        module="autogluon.tabular",
        extra="mitra",
        route="native",
        source="https://huggingface.co/autogluon/mitra-regressor",
        env="mitra",
        mem_per_lane_gb=8.0,
        max_lanes=2,
        notes="Heavy install (AutoGluon >= 1.4); verify predictive-distribution access.",
    ),
    "tabicl": ExternalSpec(
        key="tabicl",
        module="tabicl",
        extra="tabicl",
        route="native",
        source="https://github.com/soda-inria/tabicl",
        python_min=(3, 10),
        repo_subdir="tabicl/src",
        env="bench",
        mem_per_lane_gb=2.0,   # measured 5.8 GB peak for 4 lanes with TabPFN-2.5 (Mila 2026-10-01) + headroom
        max_lanes=4,
        notes=(
            "Vendored as libs/tabicl at tag v2.2.0 (2026-09-20). **Route corrected**: "
            "v2 ships a native TabICLRegressor whose predict(output_type='quantiles') "
            "returns a full predictive distribution, so it is NOT a classification-head "
            "adaptation. Needs Python >= 3.10, so it cannot run in the pinned 3.9 env."
        ),
    ),
    "tabflex": ExternalSpec(
        key="tabflex",
        module="ticl",
        extra="tabflex",
        route="classification-head adaptation",
        source="https://github.com/microsoft/ticl",
        python_min=(3, 8),
        repo_subdir="ticl",
        weights="ticl/checkpoints/tabflex.ckpt",   # host is NXDOMAIN; declaring it makes availability() honest
        mem_per_lane_gb=4.0,
        max_lanes=4,
        notes=(
            "Vendored as libs/ticl (2026-09-20). Runs on the pinned Python 3.9. "
            "Classification-only upstream, so it goes through the bucketized adapter. "
            "Weights are fetched on first use into libs/ticl/ticl/models_diff/ "
            "(excluded locally, see scripts/mila_setup.sh submodules)."
        ),
    ),
}


def availability() -> dict[str, bool]:
    """Report which external backends are importable in this environment.

    Returns:
        Mapping model key -> whether its backend module can be imported.
    """
    return {key: _backend_ok(spec)[0] for key, spec in EXTERNAL_SPECS.items()}


def availability_reasons() -> dict[str, str]:
    """Why each external backend is unavailable here, or ``''`` when it is runnable.

    The reason string of :func:`availability`, kept separate so a preflight check can report *what* to fix
    (and which environment to activate) instead of only that something is wrong.

    Returns:
        Mapping model key -> reason, empty string when the model can run.
    """
    return {key: _backend_ok(spec)[1] for key, spec in EXTERNAL_SPECS.items()}


def conda_env_for(key: str) -> str:
    """Return the conda environment one external model runs in.

    Args:
        key: Model key in :data:`EXTERNAL_SPECS`.

    Returns:
        The environment name, e.g. ``'pfns4neurostim-bench'``.

    Raises:
        KeyError: If the key is unknown.
    """
    if key not in EXTERNAL_SPECS:
        raise KeyError(f"Unknown external model {key!r}. Known: {sorted(EXTERNAL_SPECS)}.")
    return EXTERNAL_SPECS[key].conda_env


def libs_root() -> str:
    """Return the absolute path of the repository's ``libs/`` directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", "..", "libs"))


def _ensure_repo_on_path(spec: ExternalSpec) -> str | None:
    """Put a vendored submodule on ``sys.path`` so it imports without pip install.

    Args:
        spec: The external model's spec.

    Returns:
        The path added, or None when the model has no vendored submodule or the
        submodule has not been initialised.
    """
    if not spec.repo_subdir:
        return None
    path = os.path.join(libs_root(), *spec.repo_subdir.split("/"))
    if not os.path.isdir(path):
        return None
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def _backend_ok(spec: ExternalSpec) -> tuple[bool, str]:
    """Check whether one backend is usable: right Python, importable, right version.

    The Python check comes first because the alternative failure mode is a
    confusing ``SyntaxError`` raised from inside somebody else's package.

    Args:
        spec: The external model's spec.

    Returns:
        ``(ok, reason)``; ``reason`` is empty when ok.
    """
    if spec.python_min is not None and sys.version_info[:2] < spec.python_min:
        need = ".".join(str(v) for v in spec.python_min)
        have = ".".join(str(v) for v in sys.version_info[:3])
        return False, (
            f"needs Python >= {need} but this environment is {have}; it runs in "
            f"{spec.conda_env} (`conda activate {spec.conda_env}`)"
        )
    _ensure_repo_on_path(spec)
    try:
        # find_spec raises rather than returning None when a *parent* package is
        # missing (e.g. 'autogluon.tabular' with no 'autogluon' installed).
        if importlib.util.find_spec(spec.module) is None:
            return False, f"module {spec.module!r} not installed"
    except (ImportError, ValueError):
        return False, f"module {spec.module!r} not installed"

    if not spec.weights_present():
        return False, (
            f"backend is importable but its weights are missing at {spec.weights_path!r}; "
            f"an importable package without weights is not a runnable model (see {spec.source})"
        )

    if spec.major_min is not None:
        dist = spec.dist or spec.module
        try:
            from importlib.metadata import version as _dist_version  # noqa: PLC0415

            installed = _dist_version(dist)
        except Exception:  # noqa: BLE001 - absence or metadata failure both mean unknown
            return False, f"cannot read the installed version of {dist!r}"
        major = int(str(installed).split(".", 1)[0]) if str(installed)[:1].isdigit() else -1
        if major < spec.major_min:
            return False, (
                f"{dist} {installed} is installed but this model needs major version "
                f">= {spec.major_min}; the two share the module name, so they cannot "
                f"coexist in one environment - it runs in {spec.conda_env} "
                f"(`conda activate {spec.conda_env}`)"
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
        hint = (
            f"Vendored at libs/{spec.repo_subdir.split('/')[0]} - run "
            "`git submodule update --init --recursive` if that directory is empty. "
            if spec.repo_subdir
            else f"Install it with: pip install -e '.[{spec.extra}]'. "
        )
        raise ImportError(
            f"Model {key!r} is unavailable: {reason}. {hint}"
            f"(official implementation: {spec.source}). {spec.notes}"
        )
    _ensure_repo_on_path(spec)
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

    Raises:
        ValueError: If ``n_bins`` is below 2.
    """

    def __init__(self, key: str, device: str = "cpu", n_bins: int = 32, **backend_kwargs: Any) -> None:
        # Validated before the backend import: a config asking for an impossible
        # resolution is wrong in every environment, so it must not need the right one
        # to say so.
        n_bins = int(n_bins)
        if n_bins < 2:
            raise ValueError(f"{type(self).__name__}: n_bins must be >= 2, got {n_bins}.")
        super().__init__(key, device=device, **backend_kwargs)
        self.n_bins = n_bins
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
