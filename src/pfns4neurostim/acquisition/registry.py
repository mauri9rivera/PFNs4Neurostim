"""Acquisition registry: types, their parameter dataclasses, and their scoring functions.

Implements the **P0.2** config schema::

    acquisition:
      type: ucb                 # ts_marginal | ts_joint | ucb | ei | pi | greedy | random | native
      params: {kappa: 2.0}      # only this type's parameters; unknown keys raise
      schedules:                # optional; each key must be a param of this type
        kappa: {kind: cosine, start: 7.5, end: 0.6}

Every type declares a frozen params dataclass, so an unknown or misspelled
parameter fails at config-load time rather than being silently ignored. Each
params object can report its *resolved* value at a given step, which is what the
tidy CSV logs (``acq_param_<name>``).

Closed forms follow Jones et al. (1998) for EI and Kushner (1964) for PI, using
the surrogate's predictive marginals so GP and PFN get the identical rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable

import numpy as np
from scipy import stats

from ..models.protocol import marginals
from .base import BOState
from .schedules import Schedule, build_schedule

__all__ = [
    "AcquisitionSpec",
    "ACQUISITION_REGISTRY",
    "build_acquisition",
    "available_acquisitions",
    "AcqParams",
    "EIParams",
    "PIParams",
    "UCBParams",
    "TSMarginalParams",
    "TSJointParams",
    "GreedyParams",
    "RandomParams",
    "NativeParams",
]


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AcqParams:
    """Base for acquisition parameters, with optional per-parameter schedules.

    Attributes:
        schedules: Mapping parameter name -> :class:`~.schedules.Schedule`.
    """

    schedules: dict[str, Schedule] = field(default_factory=dict, repr=False)

    def value(self, name: str, state: BOState) -> float:
        """Return a parameter's value at this step, applying any schedule.

        Args:
            name: Parameter name.
            state: Loop state, supplying step index and dimensionality.

        Returns:
            The (possibly annealed) parameter value.
        """
        schedule = self.schedules.get(name)
        if schedule is None:
            return float(getattr(self, name))
        return schedule.value(state.step, state.n_steps, n_dims=state.n_dims)

    def resolved(self, state: BOState) -> dict[str, float]:
        """Return every parameter's value at this step, for logging.

        Args:
            state: Loop state.

        Returns:
            Mapping parameter name -> value at this step.
        """
        return {
            f.name: self.value(f.name, state)
            for f in fields(self)
            if f.name != "schedules"
        }


@dataclass(frozen=True)
class EIParams(AcqParams):
    """Expected Improvement. ``xi`` is the exploration offset added to the incumbent."""

    xi: float = 0.0


@dataclass(frozen=True)
class PIParams(AcqParams):
    """Probability of Improvement. ``xi`` is the exploration offset."""

    xi: float = 0.0


@dataclass(frozen=True)
class UCBParams(AcqParams):
    """Upper Confidence Bound. ``kappa`` weights the predictive standard deviation."""

    kappa: float = 2.0


@dataclass(frozen=True)
class TSMarginalParams(AcqParams):
    """Marginal Thompson sampling. ``temperature`` scales the predictive variance."""

    temperature: float = 1.0


@dataclass(frozen=True)
class TSJointParams(AcqParams):
    """Joint Thompson sampling (GP only). ``temperature`` scales the posterior variance."""

    temperature: float = 1.0


@dataclass(frozen=True)
class GreedyParams(AcqParams):
    """Pure exploitation: no parameters."""


@dataclass(frozen=True)
class RandomParams(AcqParams):
    """Uniform random acquisition: no parameters."""


@dataclass(frozen=True)
class NativeParams(AcqParams):
    """The model's own acquisition rule (end-to-end BO models): no parameters here.

    The rule's settings (e.g. PFNs4BO's criterion) belong to the model, are set in its
    ``model_params`` and enter the cache identity there.
    """


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def _score_ei(
    surrogate: Any, X_pool: np.ndarray, state: BOState, rng: np.random.Generator, params: EIParams
) -> np.ndarray:
    """Expected Improvement over the incumbent (Jones et al., 1998, Eq. 15).

    The incumbent is the best *predicted* value among queried sites, not the best
    noisy observation: with trial-to-trial noise the running max of raw
    observations is upward-biased, which makes EI far too conservative.

    Args:
        surrogate: Fitted surrogate.
        X_pool: Candidates, shape [N, D].
        state: Loop state.
        rng: Unused (EI is deterministic given the posterior).
        params: EI parameters.

    Returns:
        EI per candidate, shape [N].
    """
    mean, std = marginals(surrogate, X_pool)                       # [N], [N]
    xi = params.value("xi", state)
    observed = np.asarray(state.observed_indices, dtype=int)
    incumbent = float(np.max(mean[observed])) if observed.size else float(np.max(mean))
    gap = mean - incumbent - xi                                    # [N]
    safe = np.maximum(std, 1e-12)                                  # [N]
    z = gap / safe                                                 # [N]
    ei = gap * stats.norm.cdf(z) + safe * stats.norm.pdf(z)        # [N]
    # Where the posterior is deterministic, EI collapses to the raw improvement.
    return np.where(std > 1e-12, ei, np.maximum(gap, 0.0))


def _score_pi(
    surrogate: Any, X_pool: np.ndarray, state: BOState, rng: np.random.Generator, params: PIParams
) -> np.ndarray:
    """Probability of Improvement (Kushner, 1964).

    Args:
        surrogate: Fitted surrogate.
        X_pool: Candidates, shape [N, D].
        state: Loop state.
        rng: Unused.
        params: PI parameters.

    Returns:
        Improvement probability per candidate, shape [N].
    """
    mean, std = marginals(surrogate, X_pool)                       # [N], [N]
    xi = params.value("xi", state)
    observed = np.asarray(state.observed_indices, dtype=int)
    incumbent = float(np.max(mean[observed])) if observed.size else float(np.max(mean))
    safe = np.maximum(std, 1e-12)                                  # [N]
    return stats.norm.cdf((mean - incumbent - xi) / safe)


def _score_ucb(
    surrogate: Any, X_pool: np.ndarray, state: BOState, rng: np.random.Generator, params: UCBParams
) -> np.ndarray:
    """Upper Confidence Bound, ``mean + kappa * std``.

    Args:
        surrogate: Fitted surrogate.
        X_pool: Candidates, shape [N, D].
        state: Loop state (supplies the step for any kappa schedule).
        rng: Unused.
        params: UCB parameters.

    Returns:
        UCB per candidate, shape [N].
    """
    mean, std = marginals(surrogate, X_pool)                       # [N], [N]
    return mean + params.value("kappa", state) * std


def _score_ts_marginal(
    surrogate: Any,
    X_pool: np.ndarray,
    state: BOState,
    rng: np.random.Generator,
    params: TSMarginalParams,
) -> np.ndarray:
    """Marginal Thompson sampling: independent per-site predictive draws.

    The symmetric headline TS for both model families (task #4, 2026-09-17): a GP
    draws from N(mu_i, sigma_i^2 + sigma_n^2) and a PFN from its bar distribution,
    which is the same *family* of object — the per-site predictive.

    Args:
        surrogate: Fitted surrogate.
        X_pool: Candidates, shape [N, D].
        state: Loop state.
        rng: Seeded generator.
        params: TS parameters.

    Returns:
        One sampled value per candidate, shape [N].
    """
    return surrogate.sample_marginal(X_pool, rng, temperature=params.value("temperature", state))


def _score_ts_joint(
    surrogate: Any,
    X_pool: np.ndarray,
    state: BOState,
    rng: np.random.Generator,
    params: TSJointParams,
) -> np.ndarray:
    """Joint Thompson sampling: one draw from the joint latent posterior (GP only).

    Args:
        surrogate: Fitted surrogate with a tractable joint posterior.
        X_pool: Candidates, shape [N, D].
        state: Loop state.
        rng: Seeded generator.
        params: TS parameters.

    Returns:
        One sampled function per candidate, shape [N].

    Raises:
        NotImplementedError: For a surrogate without a joint posterior (every PFN).
    """
    return surrogate.sample_joint(X_pool, rng, temperature=params.value("temperature", state))


def _score_greedy(
    surrogate: Any, X_pool: np.ndarray, state: BOState, rng: np.random.Generator, params: GreedyParams
) -> np.ndarray:
    """Pure exploitation: the predictive mean.

    Args:
        surrogate: Fitted surrogate.
        X_pool: Candidates, shape [N, D].
        state: Unused.
        rng: Unused.
        params: Unused.

    Returns:
        Predictive mean per candidate, shape [N].
    """
    mean, _ = marginals(surrogate, X_pool)                         # [N]
    return mean


def _score_random(
    surrogate: Any, X_pool: np.ndarray, state: BOState, rng: np.random.Generator, params: RandomParams
) -> np.ndarray:
    """Uniform random acquisition, independent of the surrogate.

    Args:
        surrogate: Unused.
        X_pool: Candidates, shape [N, D].
        state: Unused.
        rng: Seeded generator.
        params: Unused.

    Returns:
        Uniform random scores, shape [N].
    """
    return rng.random(X_pool.shape[0])


def _score_native(
    surrogate: Any, X_pool: np.ndarray, state: BOState, rng: np.random.Generator, params: NativeParams
) -> np.ndarray:
    """Delegate to the model's own acquisition surface (end-to-end BO models).

    Args:
        surrogate: Fitted model that owns its query decision.
        X_pool: Candidates, shape [N, D].
        state: Unused.
        rng: Seeded generator, forwarded to the model.
        params: Unused.

    Returns:
        The model's preference per candidate, shape [N].

    Raises:
        NotImplementedError: For a plain surrogate with no native policy.
    """
    return surrogate.policy_scores(X_pool, rng)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AcquisitionSpec:
    """Registry entry for one acquisition type.

    Attributes:
        name: Registry key, used as ``acquisition.type`` and as ``acq_type``.
        params_cls: Frozen dataclass of this type's parameters.
        score_fn: Scoring function.
        needs_joint: Whether it requires a surrogate with a joint posterior.
        needs_surrogate: Whether the surrogate is consulted at all.
        needs_native_policy: Whether the model must own its query decision
            (:class:`~pfns4neurostim.models.protocol.NativePolicy`); such a model serves
            only this type, and this type only such models.
        description: One line for help output and captions.
    """

    name: str
    params_cls: type[AcqParams]
    score_fn: Callable[..., np.ndarray]
    needs_joint: bool = False
    needs_surrogate: bool = True
    needs_native_policy: bool = False
    description: str = ""


ACQUISITION_REGISTRY: dict[str, AcquisitionSpec] = {
    "ei": AcquisitionSpec("ei", EIParams, _score_ei, description="Expected Improvement"),
    "pi": AcquisitionSpec("pi", PIParams, _score_pi, description="Probability of Improvement"),
    "ucb": AcquisitionSpec("ucb", UCBParams, _score_ucb, description="Upper Confidence Bound"),
    "ts_marginal": AcquisitionSpec(
        "ts_marginal",
        TSMarginalParams,
        _score_ts_marginal,
        description="Marginal Thompson sampling (GP and PFN headline)",
    ),
    "ts_joint": AcquisitionSpec(
        "ts_joint",
        TSJointParams,
        _score_ts_joint,
        needs_joint=True,
        description="Joint Thompson sampling (GP-only reference row)",
    ),
    "greedy": AcquisitionSpec("greedy", GreedyParams, _score_greedy, description="Pure exploitation"),
    "random": AcquisitionSpec(
        "random",
        RandomParams,
        _score_random,
        needs_surrogate=False,
        description="Uniform random acquisition",
    ),
    "native": AcquisitionSpec(
        "native",
        NativeParams,
        _score_native,
        needs_native_policy=True,
        description="The model's own acquisition rule (end-to-end BO models, e.g. PFNs4BO)",
    ),
}


def available_acquisitions() -> list[str]:
    """Return every registered acquisition type name, sorted."""
    return sorted(ACQUISITION_REGISTRY)


def build_acquisition(
    acq_type: str,
    params: dict[str, Any] | None = None,
    schedules: dict[str, Any] | None = None,
) -> tuple[AcquisitionSpec, AcqParams]:
    """Validate and instantiate an acquisition type with its parameters.

    Args:
        acq_type: Registered type name.
        params: Parameter values; unknown keys raise (P0.2).
        schedules: Per-parameter schedule specs; each key must be a parameter of
            this type.

    Returns:
        ``(spec, params_instance)``.

    Raises:
        KeyError: If the type is not registered.
        ValueError: If a parameter or schedule key is not declared by the type.
    """
    if acq_type not in ACQUISITION_REGISTRY:
        raise KeyError(
            f"Unknown acquisition type {acq_type!r}. Registered: {available_acquisitions()}."
        )
    spec = ACQUISITION_REGISTRY[acq_type]
    declared = {f.name for f in fields(spec.params_cls) if f.name != "schedules"}

    given = dict(params or {})
    unknown = set(given) - declared
    if unknown:
        raise ValueError(
            f"acquisition.params has key(s) {sorted(unknown)} which {acq_type!r} does "
            f"not declare; allowed: {sorted(declared)}."
        )

    sched_specs = dict(schedules or {})
    unknown_sched = set(sched_specs) - declared
    if unknown_sched:
        raise ValueError(
            f"acquisition.schedules has key(s) {sorted(unknown_sched)} which {acq_type!r} "
            f"does not declare; allowed: {sorted(declared)}."
        )
    built = {name: build_schedule(spec_) for name, spec_ in sched_specs.items()}
    return spec, spec.params_cls(schedules=built, **given)
