"""Surrogate-model registry.

Maps a canonical model key to a constructor and the version string that must
appear in every result row (**P0.1**: the model is reported as "TabPFN v2.5", and
the version is logged rather than assumed). Keys match
``visualization.style.MODEL_STYLES`` exactly, so a model name is enough to get
its colour, label and provenance.

**Migration seam.** Constructors wrap the existing classes in
``src/models/regressors.py``. Task #1 Step 3 moves those classes into
``models/gp/`` and ``models/pfn/`` and only the bodies here change.
"""
from __future__ import annotations

import os
import sys
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


def _ensure_legacy_on_path() -> None:
    """Put the flat ``src/`` tree on ``sys.path`` so ``models.*`` imports resolve."""
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


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
        notes: Short provenance note for tables and captions.
    """

    key: str
    version: str
    family: str
    factory: Callable[..., Any]
    supports: tuple[str, ...] = ("ei", "ucb", "ts_marginal")
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
    _ensure_legacy_on_path()
    from tabpfn import TabPFNRegressor  # noqa: PLC0415 - heavy import, load on demand
    from models.regressors import TabPFNSurrogate  # noqa: PLC0415 - seam

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
    _ensure_legacy_on_path()
    from models.regressors import GPSurrogate  # noqa: PLC0415 - seam

    return GPSurrogate(device=device, n_opt_steps=n_opt_steps, lr=lr, **kwargs)


def _build_gp_naive(device: str = "cpu", **kwargs: Any) -> Any:
    """Construct the fixed-hyperparameter (naive) GP surrogate.

    Args:
        device: Torch device string.
        **kwargs: ``lengthscale``, ``outputscale``, ``noise``.

    Returns:
        A ``NaiveGPSurrogate``.
    """
    _ensure_legacy_on_path()
    from models.regressors import NaiveGPSurrogate  # noqa: PLC0415 - seam

    return NaiveGPSurrogate(device=device, **kwargs)


def _build_random(device: str = "cpu", **kwargs: Any) -> Any:
    """Construct the random-acquisition lower bound.

    Args:
        device: Ignored; kept for a uniform factory signature.
        **kwargs: Ignored.

    Returns:
        A ``RandomSearchSurrogate``.
    """
    _ensure_legacy_on_path()
    from models.regressors import RandomSearchSurrogate  # noqa: PLC0415 - seam

    return RandomSearchSurrogate()


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
    "random": ModelSpec(
        key="random",
        version="uniform random acquisition",
        family="baseline",
        factory=_build_random,
        supports=("random",),
        notes="Lower bound: queries uniformly at random.",
    ),
}

#: Comparator set for the Hyp B stress sweeps (user decision 2026-09-18).
STRESS_COMPARATORS: tuple[str, ...] = ("tabpfn_v2_5", "gp_mll", "gp_naive")


def build_surrogate(name: str, device: str = "cpu", **params: Any) -> Any:
    """Construct a registered surrogate.

    Args:
        name: Canonical model key.
        device: Torch device string.
        **params: Model-specific constructor parameters from the config.

    Returns:
        A surrogate conforming to the ``SurrogateModel`` protocol.

    Raises:
        KeyError: If the model key is unknown.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[name].factory(device=device, **params)


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
