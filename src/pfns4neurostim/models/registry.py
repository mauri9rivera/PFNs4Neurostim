"""Surrogate-model registry.

Maps a canonical model key to a constructor and the version string that must
appear in every result row (**P0.1**: the model is reported as "TabPFN v2.5", and
the version is logged rather than assumed). Keys match
``visualization.style.MODEL_STYLES`` exactly, so a model name is enough to get
its colour, label and provenance.

Constructors import from ``models/gp/`` and ``models/pfn/`` lazily, so building a
GP does not pay for the TabPFN import and vice versa. As of task #1 Step 3 these
are package-native modules; nothing here reaches into the old flat ``src/`` tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "ModelSpec",
    "MODEL_REGISTRY",
    "build_surrogate",
    "model_version",
    "available_models",
    "STRESS_COMPARATORS",
]


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry for one surrogate.

    Attributes:
        key: Canonical key, matching ``visualization.style.MODEL_STYLES``.
        version: Version string written to every result row (P0.1).
        family: ``'pfn'``, ``'gp'`` or ``'baseline'``.
        factory: ``factory(device, **params) -> surrogate`` conforming to the
            ``SurrogateModel`` protocol (``fit``, ``predict``, ``predict_ucb``,
            ``predict_ts``).
        supports: Acquisition names this surrogate can serve.
        has_predictive_distribution: Whether the surrogate exposes a genuine
            predictive distribution. False for random search, whose "std" is a
            placeholder: calibration metrics are then reported as not computed
            rather than as a meaningless zero-variance posterior.
        native_policy: True for an end-to-end BO model that owns its query decision
            (it exposes ``policy_scores``): it serves only the ``native`` acquisition
            and that acquisition serves only such models. The runners read this flag,
            never a model name.
        notes: Short provenance note for tables and captions.
    """

    key: str
    version: str
    family: str
    factory: Callable[..., Any]
    supports: tuple[str, ...] = ("ei", "ucb", "ts_marginal")
    has_predictive_distribution: bool = True
    native_policy: bool = False
    notes: str = ""


def _build_tabpfn_v2_5(device: str = "cpu", n_estimators: int = 1, **kwargs: Any) -> Any:
    """Construct the vanilla TabPFN v2.5 surrogate.

    Args:
        device: ``'cpu'`` or ``'cuda'``.
        n_estimators: Ensemble members. 1 for BO loops (one transformer call
            per step, as every kept benchmark used).
        **kwargs: Forwarded to ``TabPFNRegressor``.

    Returns:
        A ``TabPFNSurrogate`` wrapping a fresh ``TabPFNRegressor``.
    """
    from tabpfn import TabPFNRegressor  # noqa: PLC0415 - heavy import, load on demand
    from .pfn.tabpfn import TabPFNSurrogate  # noqa: PLC0415 - heavy import, load on demand

    regressor = TabPFNRegressor(
        device=device,
        n_estimators=n_estimators,
        ignore_pretraining_limits=True,
        **kwargs,
    )
    return TabPFNSurrogate(regressor)


def _build_gp_mll(device: str = "cpu", n_opt_steps: int = 100, lr: float = 0.1, **kwargs: Any) -> Any:
    """Construct the MLL-tuned exact GP surrogate.

    Args:
        device: Torch device string.
        n_opt_steps: Marginal-likelihood optimisation steps per fit.
        lr: Adam learning rate.
        **kwargs: Forwarded to ``GPSurrogate``.

    Returns:
        A ``GPSurrogate``.
    """
    from .gp.surrogates import GPSurrogate  # noqa: PLC0415 - gpytorch import, load on demand

    return GPSurrogate(device=device, n_opt_steps=n_opt_steps, lr=lr, **kwargs)


def _build_gp_naive(device: str = "cpu", **kwargs: Any) -> Any:
    """Construct the fixed-hyperparameter (naive) GP surrogate.

    Args:
        device: Torch device string.
        **kwargs: ``lengthscale``, ``outputscale``, ``noise``.

    Returns:
        A ``NaiveGPSurrogate``.
    """
    from .gp.surrogates import NaiveGPSurrogate  # noqa: PLC0415 - gpytorch import, load on demand

    return NaiveGPSurrogate(device=device, **kwargs)


def _build_random(device: str = "cpu", **kwargs: Any) -> Any:
    """Construct the random-acquisition lower bound.

    Args:
        device: Ignored; kept for a uniform factory signature.
        **kwargs: Ignored.

    Returns:
        A ``RandomSearchSurrogate``.
    """
    from .baselines.random_search import RandomSearchSurrogate  # noqa: PLC0415 - load on demand

    return RandomSearchSurrogate()


def _build_external(key: str) -> Callable[..., Any]:
    """Build a factory for one external Hyp 0 surrogate (task #8).

    Args:
        key: External model key (``pfns4bo``, ``tabpfn_v1``, ``tabfm``,
            ``mitra``, ``tabflex``).

    Returns:
        A factory with the standard ``(device, **params)`` signature. Calling it
        raises ``ImportError`` naming the extra to install when the backend is
        absent, and ``NotImplementedError`` while the wrapper body is pending.
    """

    def factory(device: str = "cpu", **params: Any) -> Any:
        from .pfn import wrappers  # noqa: PLC0415 - optional backends, load lazily

        classes = {
            "pfns4bo": wrappers.PFNs4BOSurrogate,
            "tabpfn_v1": wrappers.TabPFNv1Surrogate,
            "tabfm": wrappers.TabFMSurrogate,
            "mitra": wrappers.MitraSurrogate,
            "tabflex": wrappers.TabFlexSurrogate,
            "tabicl": wrappers.TabICLSurrogate,
        }
        return classes[key](device=device, **params)

    return factory


def _build_gp_oracle(device: str = "cpu", **kwargs: Any) -> Any:
    """Oracle GP (hyperparameters from the dense noiseless map). **Not implemented.**

    Args:
        device: Torch device string.
        **kwargs: Unused.

    Raises:
        NotImplementedError: Roadmap S8; scheduled after the K2 sweep.
    """
    raise NotImplementedError(
        "gp_oracle (roadmap S8: MLL on the GT map plus true noise variance) is not "
        "implemented yet. The K2 sweep runs tabpfn_v2_5, gp_mll and gp_naive."
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "tabpfn_v2_5": ModelSpec(
        key="tabpfn_v2_5",
        version="TabPFN v2.5",
        family="pfn",
        factory=_build_tabpfn_v2_5,
        supports=("ei", "ucb", "ts_marginal"),
        notes="Pretrained, no finetuning; n_estimators=1; native bar distribution.",
    ),
    "gp_mll": ModelSpec(
        key="gp_mll",
        version="gpytorch ExactGP (RBF), MLL-tuned per step",
        family="gp",
        factory=_build_gp_mll,
        supports=("ei", "ucb", "ts_marginal", "ts_joint"),
        notes="Hyperparameters refit by marginal likelihood at every BO step.",
    ),
    "gp_naive": ModelSpec(
        key="gp_naive",
        version="gpytorch ExactGP (RBF), fixed hyperparameters",
        family="gp",
        factory=_build_gp_naive,
        supports=("ei", "ucb", "ts_marginal", "ts_joint"),
        notes="No tuning at all: fixed lengthscale 0.2, outputscale 1.0, noise 1e-2.",
    ),
    "gp_oracle": ModelSpec(
        key="gp_oracle",
        version="gpytorch ExactGP (RBF), oracle hyperparameters",
        family="gp",
        factory=_build_gp_oracle,
        supports=("ei", "ucb", "ts_marginal", "ts_joint"),
        notes="Declared (roadmap S8); not implemented.",
    ),
    # --- Hyp 0 external amortized surrogates (task #8) -----------------------
    # Registered so configs, tables and contract tests can name them; each raises
    # with the specific remaining work when constructed. "classification-head
    # adaptation" must appear in every table row for the two bucketized models.
    "pfns4bo": ModelSpec(
        key="pfns4bo",
        version="PFNs4BO (HEBO+ prior, vendored checkpoint, native EI policy)",
        family="pfn",
        factory=_build_external("pfns4bo"),
        supports=("native",),
        native_policy=True,
        notes=(
            "End-to-end BO model: the transformer scores the pool with its own criterion "
            "(acquisition 'native'). Predictive summary from its bar distribution."
        ),
    ),
    "tabpfn_v1": ModelSpec(
        key="tabpfn_v1",
        version="TabPFN v1 (classification-head adaptation, 10 quantile bins)",
        family="pfn",
        factory=_build_external("tabpfn_v1"),
        supports=("ei", "ucb", "ts_marginal"),
        notes=(
            "Classifier binned into a bar distribution (docs/tabpfn_v1_adaptation.md). v1 emits "
            "at most 10 classes, so its predictive resolution is bounded by the bin width: report "
            "calibration next to the bin count, never ranked against v2.5's native bar "
            "distribution. Runs in pfns4neurostim-v1 (environment.v1.yml); tabpfn<2 cannot "
            "coexist with the pinned 6.3.2."
        ),
    ),
    "tabfm": ModelSpec(
        key="tabfm",
        version="Google TabFM v1.0.0 (raw-scale point prediction; sd = ensemble spread + out-of-fold residual)",
        family="pfn",
        factory=_build_external("tabfm"),
        supports=("ei", "ucb", "ts_marginal"),
        notes=(
            "Native point prediction; the std is constructed (ensemble spread + out-of-fold "
            "residual RMSE), not a model predictive distribution (G3). Needs Python >= 3.11 (bench env)."
        ),
    ),
    "mitra": ModelSpec(
        key="mitra",
        version="Mitra (AutoGluon >= 1.4)",
        family="pfn",
        factory=_build_external("mitra"),
        supports=("ei", "ucb", "ts_marginal"),
        notes="Native regressor; predictive-distribution access to be confirmed.",
    ),
    "tabflex": ModelSpec(
        key="tabflex",
        version="TabFlex (classification-head adaptation)",
        family="pfn",
        factory=_build_external("tabflex"),
        supports=("ei", "ucb", "ts_marginal"),
        notes=(
            "Classifier binned into a bar distribution; vendored at libs/ticl and "
            "runnable on the pinned Python 3.9. Weights download on first use."
        ),
    ),
    "tabicl": ModelSpec(
        key="tabicl",
        version="TabICL v2.2.0",
        family="pfn",
        factory=_build_external("tabicl"),
        supports=("ei", "ucb", "ts_marginal"),
        notes=(
            "Native regressor with a quantile predictive distribution (vendored at "
            "libs/tabicl @ v2.2.0). Needs Python >= 3.10, so it runs in the py311 env."
        ),
    ),
    "random": ModelSpec(
        key="random",
        version="uniform random acquisition",
        family="baseline",
        factory=_build_random,
        supports=("random",),
        has_predictive_distribution=False,
        notes="Lower bound: queries uniformly at random.",
    ),
}

#: Comparator set for the Hyp B stress sweeps (user decision 2026-09-18).
STRESS_COMPARATORS: tuple[str, ...] = ("tabpfn_v2_5", "gp_mll", "gp_naive")


def build_surrogate(name: str, device: str = "cpu", **params: Any) -> Any:
    """Construct a registered surrogate, wrapped in the package's surrogate interface.

    Args:
        name: Canonical model key.
        device: Torch device string.
        **params: Model-specific constructor parameters from the config.

    Returns:
        A :class:`~pfns4neurostim.models.protocol.SurrogateAdapter` exposing
        ``fit``, ``predict_marginals``, ``sample_marginal``, ``sample_joint``
        and ``supports_joint``.

    Raises:
        KeyError: If the model key is unknown.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODEL_REGISTRY)}.")
    from .protocol import SurrogateAdapter  # noqa: PLC0415 - avoid an import cycle

    spec = MODEL_REGISTRY[name]
    return SurrogateAdapter(spec.factory(device=device, **params), key=name, family=spec.family)


def model_version(name: str) -> str:
    """Return the version string logged for a model (P0.1).

    Args:
        name: Canonical model key.

    Returns:
        Version string.

    Raises:
        KeyError: If the model key is unknown.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[name].version


def available_models() -> list[str]:
    """Return every registered model key, in registry order."""
    return list(MODEL_REGISTRY)
