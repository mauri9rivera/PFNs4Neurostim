"""The surrogate interface every model must satisfy, and an adapter that supplies it.

The acquisition layer talks to exactly three capabilities:

* ``fit(X, y)`` — condition on the observations so far.
* ``predict_marginals(X) -> (mean, std)`` — the per-site predictive marginals.
  Every surrogate has these; a PFN's come from its bar distribution, a GP's from
  its predictive posterior.
* ``sample_joint(X, rng, temperature) -> [N]`` — one draw from the **joint**
  posterior, keeping inter-site correlations. Only the GP family can do this:
  a PFN's query rows do not attend to each other, so it exposes marginals only
  (task #4 decision, 2026-09-17). ``supports_joint`` says which is which, and
  ``ts_joint`` raises a clear error rather than silently degrading to marginals.

:class:`SurrogateAdapter` wraps the legacy classes in ``src/models/regressors.py``
and presents this interface. It is the single seam between the new acquisition
code and the old model code; task #1 Step 3 moves the classes into ``models/gp/``
and ``models/pfn/`` and the adapter body shrinks to nothing.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["SurrogateModel", "LegacySurrogateModel", "SurrogateAdapter", "marginals"]


@runtime_checkable
class SurrogateModel(Protocol):
    """Structural interface required of every surrogate."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Condition the surrogate on observed data.

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].
        """
        ...

    def predict_marginals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean and standard deviation per candidate.

        Args:
            X: Candidate coordinates, shape [N, D].

        Returns:
            ``(mean, std)``, each shape [N].
        """
        ...

    @property
    def supports_joint(self) -> bool:
        """Whether :meth:`sample_joint` is available."""
        ...


class SurrogateAdapter:
    """Presents a legacy surrogate through the :class:`SurrogateModel` interface.

    Args:
        model: A legacy surrogate exposing ``fit``/``predict`` and optionally
            ``predict_ts`` (joint) and ``predict_ts_marginal``.
        key: Canonical model key, for error messages.
        family: ``'pfn'``, ``'gp'`` or ``'baseline'``.
    """

    def __init__(self, model: Any, key: str, family: str) -> None:
        self._model = model
        self.key = key
        self.family = family

    # --- core interface -----------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Condition the wrapped model on observed data.

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].
        """
        self._model.fit(X, y)

    def predict_marginals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean and standard deviation per candidate.

        Args:
            X: Candidate coordinates, shape [N, D].

        Returns:
            ``(mean, std)``, each shape [N], as float64 arrays.

        Raises:
            RuntimeError: If the model returns non-finite summaries.
        """
        mean, std = self._model.predict(X)
        mean = np.asarray(mean, dtype=np.float64)   # [N]
        std = np.asarray(std, dtype=np.float64)     # [N]
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            raise RuntimeError(
                f"{self.key}.predict_marginals returned non-finite values "
                f"({int((~np.isfinite(mean)).sum())} in mean, "
                f"{int((~np.isfinite(std)).sum())} in std)."
            )
        return mean, std

    @property
    def supports_joint(self) -> bool:
        """True for the GP family, whose posterior has a tractable joint draw."""
        return self.family == "gp" and hasattr(self._model, "predict_ts")

    # --- Thompson sampling --------------------------------------------------
    def sample_marginal(self, X: np.ndarray, rng: np.random.Generator, temperature: float = 1.0) -> np.ndarray:
        """Draw independent per-site samples from the predictive marginals.

        Uses the model's native sampler when it has one (a PFN samples its bar
        distribution, which is not Gaussian), and falls back to a Gaussian draw
        from ``predict_marginals`` otherwise.

        Args:
            X: Candidate coordinates, shape [N, D].
            rng: Seeded generator.
            temperature: Variance scaling; 1.0 is the exact predictive spread.

        Returns:
            One sampled value per candidate, shape [N].

        Raises:
            ValueError: If ``temperature`` is not positive.
            RuntimeError: If the sampler returns non-finite values.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        native = getattr(self._model, "predict_ts_marginal", None)
        if native is None and self.family != "gp":
            # PFN surrogates expose per-site draws under the plain name.
            native = getattr(self._model, "predict_ts", None)
        if native is not None:
            samples = np.asarray(native(X, temperature=temperature), dtype=np.float64)
        else:
            mean, std = self.predict_marginals(X)
            samples = rng.normal(mean, np.sqrt(temperature) * std)
        if not np.isfinite(samples).all():
            raise RuntimeError(f"{self.key}.sample_marginal produced non-finite samples.")
        return samples

    def sample_joint(self, X: np.ndarray, rng: np.random.Generator, temperature: float = 1.0) -> np.ndarray:
        """Draw one sample from the joint posterior over all candidates.

        Args:
            X: Candidate coordinates, shape [N, D].
            rng: Seeded generator (used to seed the model's torch generator).
            temperature: Variance scaling.

        Returns:
            One sampled function value per candidate, shape [N].

        Raises:
            NotImplementedError: If this surrogate has no joint posterior. PFNs
                land here by construction: their query rows do not attend to each
                other, so ``ts_joint`` is a GP-only reference row.
            ValueError: If ``temperature`` is not positive.
        """
        if not self.supports_joint:
            raise NotImplementedError(
                f"ts_joint requires a surrogate with a tractable joint posterior; "
                f"{self.key!r} (family {self.family!r}) exposes marginals only. "
                "PFN query rows do not attend to each other, so joint Thompson "
                "sampling is a GP-only reference row (task #4, 2026-09-17). "
                "Use acquisition type 'ts_marginal' for a symmetric comparison."
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        import torch  # noqa: PLC0415 - only needed on the GP path

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(rng.integers(0, 2**31 - 1)))
        samples = np.asarray(
            self._model.predict_ts(X, temperature=temperature, generator=generator),
            dtype=np.float64,
        )
        if not np.isfinite(samples).all():
            raise RuntimeError(f"{self.key}.sample_joint produced non-finite samples.")
        return samples

    # --- passthrough --------------------------------------------------------
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Alias of :meth:`predict_marginals`, for code still using the old name."""
        return self.predict_marginals(X)

    @property
    def wrapped(self) -> Any:
        """The underlying legacy surrogate (for tests and diagnostics)."""
        return self._model

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SurrogateAdapter({self.key!r}, family={self.family!r})"


def marginals(surrogate: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get predictive marginals from an adapter or a bare legacy surrogate.

    Args:
        surrogate: Adapter or legacy surrogate.
        X: Candidate coordinates, shape [N, D].

    Returns:
        ``(mean, std)``, each shape [N].
    """
    if hasattr(surrogate, "predict_marginals"):
        return surrogate.predict_marginals(X)
    mean, std = surrogate.predict(X)
    return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


# ---------------------------------------------------------------------------
# Legacy protocol (moved from src/models/regressors.py at task #1 Step 3)
# ---------------------------------------------------------------------------
# The pre-restructure loop checks a different surface: fit/predict plus the
# acquisition-specific predict_ucb / predict_ts / predict_ts_marginal. The new
# SurrogateModel above is narrower on purpose - acquisition lives in
# `pfns4neurostim.acquisition`, not on the model - so both are kept, with this
# one used only by `legacy_code.bo_loops`.

# ---------------------------------------------------------------------------
# SurrogateModel protocol — unified interface for all BO surrogate models
# ---------------------------------------------------------------------------

@runtime_checkable
class LegacySurrogateModel(Protocol):
    """Protocol for Bayesian optimisation surrogate models.

    Any surrogate (GP, vanilla TabPFN, finetuned TabPFN, LoRA TabPFN) must
    implement these two methods to be used with ``run_bo_loop()``.

    The ``predict_ucb`` method is optional: surrogates that implement native
    bar-distribution UCB (TabPFN variants) should override it; surrogates
    that do not (GP) will fall back to the ``mean + kappa * std`` formula
    computed by ``run_bo_loop`` from the ``predict`` return values.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit or update the surrogate on observed data.

        For GP surrogates this trains kernel hyperparameters via marginal
        likelihood. For TabPFN surrogates this stores in-context examples
        (no gradient updates).

        Args:
            X: Feature matrix of observed points, shape [N, D].
            y: Response vector of observed targets, shape [N].
        """
        ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean and standard deviation.

        Args:
            X: Query feature matrix, shape [M, D].

        Returns:
            Tuple of (mean, std), each shape [M].
        """
        ...

    def predict_ucb(
        self,
        X: np.ndarray,
        kappa: float,
        t: int,
        n_steps: int,
    ) -> np.ndarray:
        """Return UCB acquisition values for each candidate in X.

        Surrogates with native uncertainty representations (e.g. the TabPFN
        bar-distribution) should override this for a more accurate UCB.
        The default implementation (used by ``run_bo_loop`` when the surrogate
        does not override) computes ``mean + kappa * std``.

        Args:
            X: Candidate feature matrix, shape [M, D].
            kappa: Current UCB exploration coefficient.
            t: Current BO step index (0-indexed), used for annealing.
            n_steps: Total number of BO steps (``budget - n_init``).

        Returns:
            UCB values, shape [M].
        """
        ...

    def predict_ts(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Return one Thompson Sample value for each candidate in X.

        Draws a single function sample from the surrogate's predictive
        distribution.  For GP, this is an exact draw from the joint posterior
        MVN.  For TabPFN, this samples a bin from the temperature-scaled
        bar distribution at each candidate independently.

        Args:
            X: Candidate feature matrix, shape [M, D].
            temperature: Softmax temperature for bar-distribution sampling
                (TabPFN only).  ``1.0`` = exact predictive distribution;
                ``<1.0`` = sharper / greedier; ``>1.0`` = more uniform.
                Ignored for GP (posterior is uniquely determined by kernel).

        Returns:
            Thompson sample values, shape [M].
        """
        ...
