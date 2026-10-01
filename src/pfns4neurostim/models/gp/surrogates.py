"""Gaussian-process surrogates (task #1 Step 3).

Moved from the flat ``src/models/regressors.py``. ``GPSurrogate`` refits its
kernel hyperparameters by marginal likelihood at every BO step; ``NaiveGPSurrogate``
never tunes them (the "no tuning at all" lower bound); ``DeepKernelGPSurrogate``
puts a small MLP in front of the kernel.

``predict_ts`` is the **joint latent** draw (GP-only reference row) and
``predict_ts_marginal`` the per-site predictive draw that both model families
share as the headline TS (task #4, 2026-09-17).

``posterior_update`` is the GP side of the update-rule alignment experiment (task #9,
roadmap M10): the exact change of the posterior when one observation is added, with
hyperparameters frozen at their fit on the context (or refitted: the adaptive-GP arm).
"""
from __future__ import annotations

import copy
import math
from typing import Any, Optional

import gpytorch
import numpy as np
import torch
from linear_operator.utils.errors import NanError, NotPSDError

from .exact_gp import DeepKernelGP, ExactGP

__all__ = ["GPSurrogate", "DeepKernelGPSurrogate", "NaiveGPSurrogate", "GP_OPTIMIZERS", "GP_DTYPES"]

#: Marginal-likelihood optimisers ``GPSurrogate`` accepts. ``'lbfgs'`` is the converged-fit arm of
#: prerequisite P0.10 (guardrail G1): quasi-Newton with a strong-Wolfe line search, stopped by its own
#: gradient/step tolerances rather than by a fixed step count, so "the tuned GP" is a converged GP and the
#: latency comparison is not an artefact of running a fixed 100 Adam steps.
GP_OPTIMIZERS: tuple[str, ...] = ("adam", "lbfgs")

#: Floating-point precision of the GP model, its fit and its predictions. P0.10 names float32 alongside Adam
#: as a reason the MLL fit may not converge: a near-singular K in single precision makes the marginal
#: likelihood's gradient noise comparable to the gradient itself.
GP_DTYPES: dict[str, "torch.dtype"] = {"float32": torch.float32, "float64": torch.float64}

#: Loss returned for a parameter vector at which the marginal likelihood cannot be evaluated (a
#: non-positive-definite or NaN kernel matrix). L-BFGS's strong-Wolfe line search can extrapolate far
#: enough to leave the feasible region -- measured on an NHP channel at N = 5, where the likelihood is
#: nearly flat -- and it must be able to *see* that and contract, rather than crash the run or carry a
#: NaN gradient forward. Any value far above an attainable negative log marginal likelihood works; the
#: gradient is zeroed at such a point, and the best feasible parameters seen are restored afterwards.
INFEASIBLE_LOSS: float = 1e10

#: Log-uniform ranges the multi-start initialisations are drawn from. They are stated once, here, because
#: they are not a modelling choice: the preprocessing contract puts X in [0, 1]^D and the canonical online
#: scaler puts y in [0, 1], so one set of ranges covers every dataset. A start is only a place to begin --
#: the selection rule is the marginal likelihood, not the initialisation.
RESTART_INIT_RANGES: dict[str, tuple[float, float]] = {
    "lengthscale": (1e-2, 1e1),
    "outputscale": (1e-2, 1e1),
    "noise": (1e-3, 1e0),
}

#: Last-resort ladder when NO start produced a usable fit: multiply the noise floor by
#: :data:`NOISE_ESCALATION_FACTOR` at most this many times, then fail. Bounded on purpose -- gpytorch's own
#: jitter escalation is bounded the same way, and an unbounded "raise the noise until it factorises" loop
#: is exactly how this kind of safeguard turns into a hang.
MAX_NOISE_ESCALATIONS: int = 4
NOISE_ESCALATION_FACTOR: float = 10.0

#: A fit is flagged degenerate when it sits on the noise floor (within this relative tolerance), or its
#: signal variance has collapsed, or its mean lengthscale has left this band. Reported, never raised: the
#: flag is a column so the frequency can be read off a finished run, per dataset and per stress level.
DEGENERATE_NOISE_TOL: float = 0.01
DEGENERATE_OUTPUTSCALE: float = 1e-6
DEGENERATE_LENGTHSCALE_BAND: tuple[float, float] = (1e-3, 1e3)


# ---------------------------------------------------------------------------
# GPSurrogate — ExactGP wrapper conforming to SurrogateModel
# ---------------------------------------------------------------------------

class GPSurrogate:
    """ExactGP (RBF kernel) wrapper conforming to the ``SurrogateModel`` protocol.

    Trains kernel hyperparameters via marginal likelihood optimisation at each
    ``fit`` call.  ``predict`` returns the posterior mean and standard deviation
    from the trained likelihood.

    Args:
        device: PyTorch device string ('cpu' or 'cuda').
        n_opt_steps: Optimiser budget per fit. For ``'adam'`` this is the exact number of steps taken;
            for ``'lbfgs'`` it is ``max_iter``, an upper bound the tolerances may stop short of.
        lr: Learning rate / initial step size of the optimiser.
        optimizer: One of :data:`GP_OPTIMIZERS`. ``'adam'`` is the historical fixed-budget fit;
            ``'lbfgs'`` is the converged fit of P0.10.
        dtype: Key of :data:`GP_DTYPES`; the precision of the model, the fit and the predictions.
        tolerance_grad: Gradient-infinity-norm tolerance. L-BFGS stops on it, and both optimisers are
            *reported* against it through the ``gp_fit_converged`` diagnostic, so the two arms are
            judged converged by the same rule.
        tolerance_change: Parameter/loss change tolerance of L-BFGS (unused by Adam).
        noise_floor: Lower bound on the observation-noise **variance**. gpytorch's default is 1e-4, which
            a converged fit will sit on: the marginal likelihood is maximised by interpolating, and with
            re-queried sites (duplicate rows) or a near-interpolable context the resulting ridge is too
            small for ``K + noise I`` to factorise. Measured 2026-09-30: five of ninety NHP cells died
            there, and raising this to 1e-3 fixed all five while the fitted noise still settled at ~1e-2,
            i.e. the floor prevents a numerical pathology without doing modelling work.
        n_restarts: Multi-start count. Start 0 uses gpytorch's own initialisation; the rest are drawn from
            :data:`RESTART_INIT_RANGES`. The selected fit is the best marginal likelihood **among the
            starts that produced a usable model** -- the feasibility filter. Without that filter restarts
            make things worse, not better: the degenerate interpolating solution has by far the best
            likelihood (-3.49 against -0.93 on the cells that failed), so plain best-of-k picks it every
            time.
        restart_seed: Seed for the restart initialisations, so a cell is reproducible.

    Raises:
        ValueError: If ``optimizer`` or ``dtype`` is not a recognised key, or ``n_restarts`` < 1.
    """

    def __init__(
        self,
        device: str = 'cpu',
        n_opt_steps: int = 100,
        lr: float = 0.1,
        optimizer: str = 'adam',
        dtype: str = 'float32',
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        noise_floor: float = 1e-4,
        n_restarts: int = 1,
        restart_seed: int = 0,
    ) -> None:
        if optimizer not in GP_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {GP_OPTIMIZERS}, got {optimizer!r}.")
        if dtype not in GP_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(GP_DTYPES)}, got {dtype!r}.")
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}.")
        if noise_floor <= 0.0:
            raise ValueError(f"noise_floor must be > 0, got {noise_floor}.")
        self._device = device
        self._n_opt_steps = n_opt_steps
        self._lr = lr
        self._optimizer = optimizer
        self._dtype_name = dtype
        self._dtype = GP_DTYPES[dtype]
        self._tolerance_grad = tolerance_grad
        self._tolerance_change = tolerance_change
        self._noise_floor = noise_floor
        self._n_restarts = n_restarts
        self._restart_seed = restart_seed
        self._model: ExactGP | None = None
        self._likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self._train_X: np.ndarray | None = None   # [N, D] float64 context, for closed-form updates
        self._train_y: np.ndarray | None = None   # [N]
        self.last_refit_hyperparameters: list[dict[str, Any]] = []

    def _build_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: "gpytorch.likelihoods.GaussianLikelihood",
    ) -> ExactGP:
        """Construct the GP model for this surrogate (override point for subclasses).

        Args:
            train_x: Training inputs, shape [N, D].
            train_y: Training targets, shape [N].
            likelihood: The Gaussian likelihood instance.

        Returns:
            An initialised gpytorch ExactGP-derived model on the surrogate device.
        """
        return ExactGP(train_x, train_y, likelihood).to(self._device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP hyperparameters by marginal likelihood, over ``n_restarts`` starts.

        A start is kept only if it produced a *usable* model -- the fit ran and the posterior over the
        context factorises in evaluation mode, which is the operation that every later prediction needs.
        The selected fit is the best marginal likelihood among those. If no start is usable the noise floor
        is raised by :data:`NOISE_ESCALATION_FACTOR`, at most :data:`MAX_NOISE_ESCALATIONS` times, and the
        starts are retried; only then does the fit fail, with the escalation recorded.

        Args:
            X: Feature matrix of observed points, shape [N, D].  # [N, D]
            y: Response vector of observed targets, shape [N].   # [N]

        Raises:
            RuntimeError: On NaN inputs, or if no start yielded a usable model even at the highest floor.
        """
        if self._model is not None:
            del self._model, self._likelihood
            self._model, self._likelihood = None, None

        train_x = torch.tensor(X, dtype=self._dtype, device=self._device)  # [N, D]
        train_y = torch.tensor(y, dtype=self._dtype, device=self._device)  # [N]

        if torch.isnan(train_x).any() or torch.isnan(train_y).any():
            raise RuntimeError(
                "GPSurrogate.fit received NaN inputs. "
                f"X has {torch.isnan(train_x).sum()} NaNs, "
                f"y has {torch.isnan(train_y).sum()} NaNs."
            )

        self._train_X = np.asarray(X, dtype=np.float64).copy()
        self._train_y = np.asarray(y, dtype=np.float64).copy()

        rng = np.random.default_rng([self._restart_seed, int(train_x.shape[0])])
        inits = [None, *(self._sample_init(rng) for _ in range(self._n_restarts - 1))]
        floor = self._noise_floor
        best: tuple[float, ExactGP, Any, dict[str, float]] | None = None
        n_feasible, escalations, reasons = 0, 0, []

        while True:
            for init in inits:
                try:
                    candidate = self._fit_candidate(train_x, train_y, init, floor)
                except (NanError, NotPSDError, RuntimeError) as exc:
                    reasons.append(type(exc).__name__)
                    continue
                n_feasible += 1
                if best is None or candidate[0] < best[0]:
                    best = candidate
            if best is not None or escalations >= MAX_NOISE_ESCALATIONS:
                break
            floor *= NOISE_ESCALATION_FACTOR
            escalations += 1

        if best is None:
            raise RuntimeError(
                f"GPSurrogate.fit: no usable fit from {self._n_restarts} start(s) on "
                f"N={int(train_x.shape[0])} points after {escalations} noise-floor escalation(s) "
                f"(floor now {floor:.1e}, dtype {self._dtype_name}); failures: {sorted(set(reasons))}."
            )

        final_loss, self._model, self._likelihood, diagnostics = best
        self._last_fit = {
            **diagnostics,
            "gp_n_restarts": float(self._n_restarts),
            "gp_n_feasible": float(n_feasible),
            "gp_noise_escalations": float(escalations),
            "gp_noise_floor": float(floor),
        }
        self._last_fit["gp_fit_degenerate"] = self._degeneracy_flag(self._last_fit, floor)

    def _sample_init(self, rng: np.random.Generator) -> dict[str, float]:
        """Draw one multi-start initialisation log-uniformly from :data:`RESTART_INIT_RANGES`.

        Args:
            rng: Seeded generator, so a cell's starts are reproducible.

        Returns:
            Starting ``lengthscale``, ``outputscale`` and ``noise``.
        """
        return {
            name: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            for name, (lo, hi) in RESTART_INIT_RANGES.items()
        }

    def _fit_candidate(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        init: dict[str, float] | None,
        floor: float,
    ) -> tuple[float, ExactGP, Any, dict[str, float]]:
        """Fit one start and return it only if the resulting model is usable.

        Args:
            train_x: Context inputs, shape [N, D].  # [N, D]
            train_y: Context targets, shape [N].    # [N]
            init: Starting hyperparameters, or None for gpytorch's own initialisation.
            floor: Noise-variance lower bound for this attempt.

        Returns:
            ``(negative_mll, model, likelihood, diagnostics)``.

        Raises:
            NotPSDError, NanError, RuntimeError: If the fit or the evaluation-mode posterior fails; the
                caller treats that as an infeasible start rather than an error.
        """
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            device=self._device, dtype=self._dtype
        )
        if floor > 1e-4:
            # Only touched when the floor is raised above gpytorch's own default, so every surrogate that
            # keeps the default (gp_naive, the deep kernel, the mechanism engines) is numerically unchanged.
            likelihood.noise_covar.register_constraint(
                "raw_noise", gpytorch.constraints.GreaterThan(floor)
            )
            likelihood.initialize(noise=torch.tensor(max(10.0 * floor, 1e-2)))
        model = self._build_model(train_x, train_y, likelihood).to(dtype=self._dtype)
        if init is not None:
            with torch.no_grad():
                model.covar_module.base_kernel.lengthscale = init["lengthscale"]
                model.covar_module.outputscale = init["outputscale"]
                likelihood.noise = max(init["noise"], floor * 1.01)

        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # _optimize reads self._model, so point the surrogate at this candidate for the duration.
        self._model, self._likelihood = model, likelihood
        final_loss = float("nan")
        steps, n_evals, grad_max = 0, 0, float("nan")
        if self._n_opt_steps > 0:
            steps, n_evals, grad_max = self._optimize(train_x, train_y, mll)

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            if self._n_opt_steps > 0:
                # This is the feasibility test as well as the diagnostic: it is the evaluation-mode
                # factorisation of the context posterior, the exact operation that killed five cells on
                # 2026-09-30. A start that cannot do it is discarded instead of aborting the run.
                final_loss = float(-mll(model(train_x), train_y).detach())
                if not math.isfinite(final_loss):
                    raise NanError(f"non-finite marginal likelihood after the fit ({final_loss}).")
        diagnostics = self._read_hyperparameters(
            final_loss, int(train_x.shape[0]), steps=steps, n_evals=n_evals, grad_max=grad_max
        )
        return final_loss, model, likelihood, diagnostics

    def _degeneracy_flag(self, diagnostics: dict[str, float], floor: float) -> float:
        """1.0 when the selected fit sits on a bound rather than at an interior optimum.

        Reported, never raised: a fit on the noise floor is still usable, but a run in which it happens
        often is not a converged-GP comparison, and until this column existed there was no way to tell.

        Args:
            diagnostics: The selected fit's diagnostics.
            floor: Noise-variance floor in force.

        Returns:
            1.0 if degenerate, 0.0 otherwise, NaN when no fit was performed.
        """
        if not math.isfinite(diagnostics.get("gp_noise", float("nan"))):
            return float("nan")
        lo, hi = DEGENERATE_LENGTHSCALE_BAND
        return float(
            diagnostics["gp_noise"] <= floor * (1.0 + DEGENERATE_NOISE_TOL)
            or diagnostics["gp_outputscale"] < DEGENERATE_OUTPUTSCALE
            or not lo <= diagnostics["gp_lengthscale"] <= hi
        )

    def _optimize(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        mll: "gpytorch.mlls.ExactMarginalLogLikelihood",
    ) -> tuple[int, float]:
        """Maximise the exact marginal likelihood with the configured optimiser.

        Both arms evaluate the same objective through the same closure, so the only difference between
        them is the update rule: Adam takes exactly ``n_opt_steps`` first-order steps, L-BFGS takes at
        most ``n_opt_steps`` quasi-Newton steps with a strong-Wolfe line search and stops early on its
        own tolerances. Implements the P0.10 "converged MLL fit" arm.

        Args:
            train_x: Context inputs, shape [N, D].  # [N, D]
            train_y: Context targets, shape [N].    # [N]
            mll: The exact marginal log likelihood of the current model.

        Returns:
            ``(steps_taken, n_evaluations, grad_max)``: optimiser iterations actually performed,
            objective+gradient evaluations they cost (Adam: one per step; L-BFGS: more, because of the
            line search -- this is the number the latency comparison must use), and the infinity norm
            of the gradient at the final parameters.

        Raises:
            RuntimeError: If no feasible parameter vector was ever evaluated, or the final gradient is
                not finite.
        """
        params = [p for p in self._model.parameters() if p.requires_grad]
        best_loss = math.inf
        best_params: list[torch.Tensor] = []
        n_evals = 0

        def closure() -> torch.Tensor:
            """One objective + gradient evaluation of the negative marginal log likelihood.

            Returns :data:`INFEASIBLE_LOSS` with a zero gradient where the kernel matrix cannot be
            factorised, so a line search contracts instead of the fit dying on a NaN.
            """
            nonlocal best_loss, best_params, n_evals
            n_evals += 1
            optimizer.zero_grad(set_to_none=False)
            try:
                loss = -mll(self._model(train_x), train_y)
                if not torch.isfinite(loss):
                    raise NanError("non-finite marginal log likelihood")
                loss.backward()
            except (NanError, NotPSDError):
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
                return torch.as_tensor(INFEASIBLE_LOSS, dtype=self._dtype, device=self._device)
            value = float(loss.detach())
            if value < best_loss and all(
                p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params
            ):
                best_loss = value
                best_params = [p.detach().clone() for p in params]
            return loss

        if self._optimizer == "lbfgs":
            optimizer: torch.optim.Optimizer = torch.optim.LBFGS(
                params,
                lr=self._lr,
                max_iter=self._n_opt_steps,
                tolerance_grad=self._tolerance_grad,
                tolerance_change=self._tolerance_change,
                line_search_fn="strong_wolfe",
            )
            optimizer.step(closure)
            # LBFGS keeps its iteration count in the state of its first parameter.
            steps = int(optimizer.state[params[0]].get("n_iter", self._n_opt_steps))
            if not best_params:
                raise RuntimeError(
                    f"GPSurrogate.fit: the L-BFGS fit never reached a feasible point in {n_evals} "
                    f"evaluation(s) on N={int(train_x.shape[0])} points, dtype {self._dtype_name}."
                )
            # The line search may end on a worse (or infeasible) point than the best one it saw, so the
            # fit keeps the best feasible parameters and reports the gradient there.
            with torch.no_grad():
                for p, best in zip(params, best_params):
                    p.copy_(best)
            closure()
        else:
            optimizer = torch.optim.Adam(params, lr=self._lr)
            for _ in range(self._n_opt_steps):
                closure()
                optimizer.step()
            steps = self._n_opt_steps

        grads = [p.grad.detach().abs().max() for p in params if p.grad is not None]
        grad_max = float(torch.stack(grads).max()) if grads else float("nan")
        if not math.isfinite(grad_max):
            raise RuntimeError(
                f"GPSurrogate.fit: non-finite MLL gradient ({grad_max}) after {steps} "
                f"{self._optimizer} step(s) on N={int(train_x.shape[0])} points, "
                f"dtype {self._dtype_name}."
            )
        return steps, n_evals, grad_max

    def _read_hyperparameters(
        self,
        final_loss: float,
        n_train: int,
        *,
        steps: int = 0,
        n_evals: int = 0,
        grad_max: float = float("nan"),
    ) -> dict[str, float]:
        """Read the kernel hyperparameters back off the fitted model.

        Recorded per fit so a run is auditable: until 2026-09-27 no artefact stored what the MLL fit
        actually converged to, which made the open P0.10 question ("is the tuned GP converged?") and
        guardrail G1 (the latency claim) impossible to settle from the saved results. ``mll_final`` is the
        negative log marginal likelihood after the last step, and ``noise`` is a variance, so it is
        directly comparable with the empirical per-site trial variance.

        Args:
            final_loss: Negative log marginal likelihood after the final optimiser step.
            n_train: Context size this fit saw.
            steps: Optimiser iterations actually performed (Adam: the fixed budget; L-BFGS: however
                many it took before its tolerances stopped it).
            n_evals: Objective+gradient evaluations those steps cost; the cost-comparable quantity,
                since one L-BFGS iteration line-searches over several evaluations.
            grad_max: Infinity norm of the MLL gradient at the final parameters; NaN when no step
                was taken.

        ``gp_fit_converged`` is 1.0 when the optimiser stopped on a convergence criterion rather than on
        its budget: either the gradient is within ``tolerance_grad``, or it took fewer than
        ``n_opt_steps`` iterations, which for L-BFGS means one of its own tolerances fired. A fixed-budget
        Adam fit therefore reports 1.0 only if its gradient really is small, which is the P0.10 question.

        Returns:
            Scalar diagnostics of the fit. ``lengthscale`` is the mean over ARD dimensions and
            ``lengthscale_min`` / ``lengthscale_max`` bracket them, so an isotropic kernel reports all three
            equal.
        """
        ls = self._model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
        return {
            "gp_lengthscale": float(np.mean(ls)),
            "gp_lengthscale_min": float(np.min(ls)),
            "gp_lengthscale_max": float(np.max(ls)),
            "gp_outputscale": float(self._model.covar_module.outputscale.detach()),
            "gp_noise": float(self._likelihood.noise.detach().ravel()[0]),
            "gp_mll_final": final_loss,
            "gp_n_opt_steps": float(steps),
            "gp_n_opt_steps_max": float(self._n_opt_steps),
            "gp_n_obj_evals": float(n_evals),
            "gp_grad_max": grad_max,
            "gp_fit_converged": (
                float("nan")
                if not math.isfinite(grad_max)
                else float(grad_max <= self._tolerance_grad or steps < self._n_opt_steps)
            ),
            "gp_fit_n_train": float(n_train),
        }

    def fit_diagnostics(self) -> dict[str, float]:
        """Flat, tidy-row-ready diagnostics of the most recent :meth:`fit`.

        Distinct from :meth:`hyperparameters`, which returns the raw values (an ARD lengthscale is a [D]
        array) for closed-form conditioning and for tests. This one returns only ``gp_``-prefixed scalars,
        so it can be merged straight into a tidy row, and it never raises before the first fit.

        Returns:
            The diagnostics of :meth:`_read_hyperparameters`, or an empty mapping before the first fit.
        """
        return dict(getattr(self, "_last_fit", {}) or {})

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return GP posterior mean and standard deviation.

        Args:
            X: Query feature matrix, shape [M, D].  # [M, D]

        Returns:
            Tuple of (mean, std), each shape [M].   # [M], [M]

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if self._model is None or self._likelihood is None:
            raise RuntimeError(
                "GPSurrogate.predict called before fit. Call fit() first."
            )
        query_x = torch.tensor(X, dtype=self._dtype, device=self._device)  # [M, D]
        with torch.no_grad():
            posterior = self._likelihood(self._model(query_x))
            mean = posterior.mean.cpu().numpy()   # [M]
            std = posterior.stddev.cpu().numpy()  # [M]

        if np.isnan(mean).any() or np.isnan(std).any():
            raise RuntimeError(
                f"GPSurrogate.predict returned NaN values. "
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
        """Return UCB values using GP posterior mean and standard deviation.

        Uses cosine-annealed kappa internally (the ``kappa`` argument is the
        pre-annealed value computed by ``run_bo_loop``).

        Args:
            X: Candidate feature matrix, shape [M, D].  # [M, D]
            kappa: Current (already-annealed) UCB exploration coefficient.
            t: Current BO step index (unused; annealing done by caller).
            n_steps: Total BO steps (unused; annealing done by caller).

        Returns:
            UCB values, shape [M].  # [M]
        """
        mean, std = self.predict(X)  # [M], [M]
        return mean + kappa * std    # [M]

    def predict_ts(
        self,
        X: np.ndarray,
        temperature: float = 1.0,
        generator: "torch.Generator | None" = None,
        include_noise: bool = False,
    ) -> np.ndarray:
        """Draw one joint Thompson sample from the GP posterior.

        Default (``include_noise=False``) is textbook GP Thompson sampling: one
        draw from the joint latent posterior N(μ, Σ) over all candidates, so the
        sampled values keep the inter-point correlations dictated by the kernel.
        With ``include_noise=True`` the observation noise is added
        (N(μ, Σ + σ²_n I)); this is diagnostic only, because homoscedastic noise
        σ²_n swamps the posterior covariance once the surface is well observed
        and the "joint" draw then behaves like independent per-site draws.

        This is the GP-only reference row of the 2026-09-17 TS design decision
        (see ``.claude/research_design_log.md``): PFN surrogates expose per-site
        marginals only, so the symmetric GP-vs-PFN headline acquisition is
        ``predict_ts_marginal``.

        Only the GP supports this: PFN surrogates expose per-site marginals
        only, so the symmetric GP-vs-PFN headline acquisition is
        ``predict_ts_marginal``.

        Args:
            X: Candidate feature matrix, shape [M, D].  # [M, D]
            temperature: Variance scaling of the sampled distribution.  ``1.0``
                (default) samples the exact predictive posterior; ``>1`` inflates
                and ``<1`` shrinks its spread, matching the effect of the
                bar-distribution temperature in ``TabPFNSurrogate``.
            generator: Optional torch RNG for reproducible draws.  ``None`` uses
                the global torch RNG.
            include_noise: Sample the noisy predictive posterior instead of the
                latent one (diagnostic; see above).

        Returns:
            Thompson sample, shape [M].  # [M]

        Raises:
            RuntimeError: If ``fit`` has not been called yet, or the sample is
                not finite.
            ValueError: If ``temperature`` is not strictly positive.
        """
        if self._model is None or self._likelihood is None:
            raise RuntimeError(
                "GPSurrogate.predict_ts called before fit. Call fit() first."
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        query_x = torch.tensor(X, dtype=self._dtype, device=self._device)  # [M, D]
        with torch.no_grad():
            latent = self._model(query_x)                        # N(μ, Σ) — [M]
            posterior = self._likelihood(latent) if include_noise else latent
            mean = posterior.mean                                # [M]
            # rsample() has no generator argument: draw standard normals explicitly
            # and push them through the Cholesky factor of the covariance.
            # float64 + escalating jitter: the latent covariance of a well-fitted GP
            # over a dense grid is near-singular in float32, where Cholesky fails.
            cov = posterior.covariance_matrix.double()           # [M, M]
            eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
            scale = float(torch.diagonal(cov).mean())
            chol = None
            for exponent in range(-8, -2):                       # 1e-8 … 1e-3 × mean variance
                try:
                    chol = torch.linalg.cholesky(cov + scale * (10.0 ** exponent) * eye)
                    break
                except RuntimeError:
                    continue
            if chol is None:
                raise RuntimeError(
                    "GPSurrogate.predict_ts: latent covariance not positive definite "
                    f"even with 1e-3 relative jitter (M={cov.shape[0]})."
                )
            eps = torch.randn(
                cov.shape[0], dtype=cov.dtype, device=cov.device, generator=generator
            )                                                    # [M]
            sample = mean.double() + np.sqrt(temperature) * (chol @ eps)  # [M]

        out = sample.cpu().numpy()                               # [M]
        if not np.isfinite(out).all():
            raise RuntimeError(
                f"GPSurrogate.predict_ts produced {(~np.isfinite(out)).sum()} "
                "non-finite sample values."
            )
        return out                                               # [M]

    def predict_ts_marginal(
        self,
        X: np.ndarray,
        temperature: float = 1.0,
        generator: "torch.Generator | None" = None,
    ) -> np.ndarray:
        """Draw independent per-site Thompson samples from the GP predictive marginals.

        Samples each candidate independently from its own predictive marginal
        N(μ_i, temperature · (σ²_i + σ²_n)), discarding cross-site correlations.
        This is the symmetric counterpart of ``TabPFNSurrogate.predict_ts``,
        which can only sample per-site because PFN query rows do not attend to
        each other; it is the headline TS for both model families as of the
        2026-09-17 TS design decision.

        Args:
            X: Candidate feature matrix, shape [M, D].  # [M, D]
            temperature: Variance scaling of the per-site draws (``1.0`` = exact
                predictive marginals).
            generator: Optional torch RNG for reproducible draws.  ``None`` uses
                the global torch RNG.

        Returns:
            Thompson sample values, shape [M].  # [M]

        Raises:
            RuntimeError: If ``fit`` has not been called yet, or the sample is
                not finite.
            ValueError: If ``temperature`` is not strictly positive.
        """
        if self._model is None or self._likelihood is None:
            raise RuntimeError(
                "GPSurrogate.predict_ts_marginal called before fit. Call fit() first."
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        query_x = torch.tensor(X, dtype=self._dtype, device=self._device)  # [M, D]
        with torch.no_grad():
            posterior = self._likelihood(self._model(query_x))   # [M]
            mean = posterior.mean                                # [M]
            std = posterior.stddev                               # [M]
            eps = torch.randn(
                mean.shape[0], dtype=mean.dtype, device=mean.device, generator=generator
            )                                                    # [M]
            sample = mean + np.sqrt(temperature) * std * eps     # [M]

        out = sample.cpu().numpy()                               # [M]
        if not np.isfinite(out).all():
            raise RuntimeError(
                f"GPSurrogate.predict_ts_marginal produced "
                f"{(~np.isfinite(out)).sum()} non-finite sample values."
            )
        return out                                               # [M]


    # ------------------------------------------------------------------
    # Closed-form conditioning (task #9, M10)
    # ------------------------------------------------------------------
    def hyperparameters(self) -> dict[str, Any]:
        """Return the fitted hyperparameters of the stationary RBF GP.

        Returns:
            ``{'lengthscale': [D] array, 'outputscale': float, 'noise': float,
            'mean': float}`` in the (MinMax) input units the GP was fitted on.

        Raises:
            RuntimeError: Before ``fit``.
            NotImplementedError: For a non-stationary (deep-kernel) GP, whose kernel
                is not a plain RBF of the inputs.
        """
        if self._model is None or self._likelihood is None:
            raise RuntimeError("GPSurrogate.hyperparameters called before fit.")
        if type(self._model) is not ExactGP:
            raise NotImplementedError(
                "Closed-form conditioning needs the stationary ExactGP, got "
                f"{type(self._model).__name__}."
            )
        with torch.no_grad():
            kernel = self._model.covar_module
            ls = kernel.base_kernel.lengthscale.detach().cpu().double().numpy().reshape(-1)  # [D]
            return {
                "lengthscale": ls,
                "outputscale": float(kernel.outputscale.detach().cpu()),
                "noise": float(self._likelihood.noise.detach().cpu().reshape(-1)[0]),
                "mean": float(self._model.mean_module.constant.detach().cpu().reshape(-1)[0]),
            }

    @staticmethod
    def _rbf(A: np.ndarray, B: np.ndarray, hp: dict[str, Any]) -> np.ndarray:
        """ARD RBF prior covariance in float64.

        Args:
            A: Inputs, shape [P, D].
            B: Inputs, shape [Q, D].
            hp: Output of :meth:`hyperparameters`.

        Returns:
            ``outputscale * exp(-0.5 * ||(a - b) / ell||^2)``, shape [P, Q].
        """
        a = np.asarray(A, dtype=np.float64) / hp["lengthscale"]                    # [P, D]
        b = np.asarray(B, dtype=np.float64) / hp["lengthscale"]                    # [Q, D]
        sq = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T    # [P, Q]
        return hp["outputscale"] * np.exp(-0.5 * np.maximum(sq, 0.0))

    @classmethod
    def _conditioned(
        cls, X_train: np.ndarray, y_train: np.ndarray, hp: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cholesky factor and weights of a GP conditioned on ``(X_train, y_train)``.

        Args:
            X_train: Training inputs, shape [N, D].
            y_train: Training targets, shape [N].
            hp: Hyperparameters.

        Returns:
            ``(L, alpha)`` with ``L L^T = K + noise I`` ([N, N]) and
            ``alpha = (K + noise I)^{-1} (y - m)`` ([N]).
        """
        K = cls._rbf(X_train, X_train, hp) + hp["noise"] * np.eye(len(X_train))  # [N, N]
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train - hp["mean"]))     # [N]
        return L, alpha

    def predict_frozen(
        self,
        X: np.ndarray,
        *,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Float64 closed-form predictive mean and SD with the fitted hyperparameters.

        Args:
            X: Query inputs, shape [M, D].
            X_train: Conditioning inputs; defaults to the fitted context.
            y_train: Conditioning targets; defaults to the fitted context.

        Returns:
            ``(mean, predictive_sd)``, each shape [M]; the SD includes observation noise.

        Raises:
            RuntimeError: Before ``fit``.
        """
        if self._train_X is None or self._train_y is None:
            raise RuntimeError("GPSurrogate.predict_frozen called before fit.")
        hp = self.hyperparameters()
        Xt = self._train_X if X_train is None else np.asarray(X_train, dtype=np.float64)
        yt = self._train_y if y_train is None else np.asarray(y_train, dtype=np.float64)
        L, alpha = self._conditioned(Xt, yt, hp)
        Kq = self._rbf(X, Xt, hp)                                   # [M, N]
        mean = hp["mean"] + Kq @ alpha                              # [M]
        v = np.linalg.solve(L, Kq.T)                                # [N, M]
        var = np.maximum(hp["outputscale"] - (v ** 2).sum(0), 0.0)  # [M] latent
        return mean, np.sqrt(var + hp["noise"])                     # [M], [M] predictive

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        """Float64 latent posterior covariance over ``X`` given the fitted context.

        Used as the ``K_GP`` target kernel of the CKA analysis (task #5, P0.5).

        Args:
            X: Query inputs, shape [M, D].

        Returns:
            ``k(X, X) - k(X, C) (K_C + s2 I)^-1 k(C, X)``, shape [M, M].

        Raises:
            RuntimeError: Before ``fit``.
        """
        if self._train_X is None or self._train_y is None:
            raise RuntimeError("GPSurrogate.posterior_covariance called before fit.")
        hp = self.hyperparameters()
        L, _ = self._conditioned(self._train_X, self._train_y, hp)
        v = np.linalg.solve(L, self._rbf(self._train_X, X, hp))       # [N, M]
        return self._rbf(X, X, hp) - v.T @ v                          # [M, M]

    def _fresh_copy(self) -> "GPSurrogate":
        """An unfitted surrogate with this one's configuration (for the refit arm)."""
        other = copy.copy(self)
        other._model = None
        other._likelihood = None
        other._train_X = None
        other._train_y = None
        return other

    def posterior_update(
        self,
        x_star: np.ndarray,
        y_star: np.ndarray,
        X_query: np.ndarray,
        *,
        refit: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Change of the predictive mean and SD when one observation is added.

        Frozen hyperparameters (default) give the closed form of roadmap M10::

            dmu(x)  = k_C(x, x*) (y* - mu_C(x*)) / (k_C(x*, x*) + s2)
            dvar(x) = -k_C(x, x*)^2 / (k_C(x*, x*) + s2)

        where ``k_C`` is the *posterior* covariance given the context C. The variance
        update does not depend on ``y*`` (the H10.4 reference). ``refit=True`` instead
        refits the hyperparameters on ``C + {(x*, y*)}`` (the adaptive-GP comparator)
        and differences the two closed-form predictions; the refitted hyperparameters
        are stored in ``self.last_refit_hyperparameters``.

        Args:
            x_star: Probe inputs, shape [P, D] (or [D] for one probe).
            y_star: Probe values, shape [P] (or scalar).
            X_query: Readout sites, shape [M, D].
            refit: Refit hyperparameters per probe instead of freezing them.

        Returns:
            ``(d_mean, d_sd)``, each shape [P, M]; ``d_sd`` is the change of the
            predictive (noise-inclusive) SD.

        Raises:
            RuntimeError: Before ``fit`` or on a non-finite result.
        """
        if self._train_X is None or self._train_y is None:
            raise RuntimeError("GPSurrogate.posterior_update called before fit.")
        xs = np.atleast_2d(np.asarray(x_star, dtype=np.float64))    # [P, D]
        ys = np.atleast_1d(np.asarray(y_star, dtype=np.float64))    # [P]
        Q = np.asarray(X_query, dtype=np.float64)                   # [M, D]
        if refit:
            base_mean, base_sd = self.predict_frozen(Q)             # [M], [M]
            d_mean = np.empty((len(xs), len(Q)))                    # [P, M]
            d_sd = np.empty((len(xs), len(Q)))                      # [P, M]
            self.last_refit_hyperparameters = []
            for p in range(len(xs)):
                other = self._fresh_copy()
                other.fit(np.vstack([self._train_X, xs[p]]), np.append(self._train_y, ys[p]))
                mean_p, sd_p = other.predict_frozen(Q)
                d_mean[p], d_sd[p] = mean_p - base_mean, sd_p - base_sd
                self.last_refit_hyperparameters.append(other.hyperparameters())
        else:
            hp = self.hyperparameters()
            L, alpha = self._conditioned(self._train_X, self._train_y, hp)
            Kqx = self._rbf(Q, self._train_X, hp)                   # [M, N]
            Ksx = self._rbf(xs, self._train_X, hp)                  # [P, N]
            vq = np.linalg.solve(L, Kqx.T)                          # [N, M]
            vs = np.linalg.solve(L, Ksx.T)                          # [N, P]
            k_qs = self._rbf(Q, xs, hp) - vq.T @ vs                 # [M, P] posterior cov
            k_ss = hp["outputscale"] - (vs ** 2).sum(0)             # [P]
            mu_s = hp["mean"] + Ksx @ alpha                         # [P]
            denom = k_ss + hp["noise"]                              # [P]
            d_mean = (k_qs * ((ys - mu_s) / denom)[None, :]).T      # [P, M]
            var_q = np.maximum(hp["outputscale"] - (vq ** 2).sum(0), 0.0)            # [M]
            var_new = np.maximum(var_q[None, :] - (k_qs ** 2).T / denom[:, None], 0.0)  # [P, M]
            d_sd = np.sqrt(var_new + hp["noise"]) - np.sqrt(var_q + hp["noise"])[None, :]
        if not (np.isfinite(d_mean).all() and np.isfinite(d_sd).all()):
            raise RuntimeError("GPSurrogate.posterior_update produced non-finite values.")
        return d_mean, d_sd


# ---------------------------------------------------------------------------
# DeepKernelGPSurrogate — non-stationary GP baseline (§16, limitation L2/L9)
# ---------------------------------------------------------------------------

class DeepKernelGPSurrogate(GPSurrogate):
    """Deep-kernel (non-stationary) GP surrogate conforming to ``SurrogateModel``.

    Reuses the full :class:`GPSurrogate` fit/predict/acquisition machinery but
    swaps the stationary :class:`ExactGP` for a :class:`DeepKernelGP` whose MLP
    feature extractor is trained jointly with the GP hyperparameters. Serves as
    the "best-tuned, expressive GP" baseline: if TabPFN still matches or beats
    it, the on-par-but-cheaper claim is not an artefact of a weak (stationary)
    GP, and the GFS over-smoothing story (A5) is causal rather than a
    weak-baseline artefact.

    A higher default ``n_opt_steps`` and ``lr`` are used than the stationary GP
    because the extra MLP parameters need more optimisation to converge.

    Args:
        device: PyTorch device string ('cpu' or 'cuda').
        n_opt_steps: Adam steps for joint MLP + GP hyperparameter training.
        lr: Adam learning rate.
        feature_dim: Output dimensionality of the learned feature space.
        hidden: Hidden width of the feature-extractor MLP.
    """

    def __init__(
        self,
        device: str = 'cpu',
        n_opt_steps: int = 120,
        lr: float = 0.02,
        feature_dim: int = 2,
        hidden: int = 32,
    ) -> None:
        super().__init__(device=device, n_opt_steps=n_opt_steps, lr=lr)
        self._feature_dim = feature_dim
        self._hidden = hidden

    def _build_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: "gpytorch.likelihoods.GaussianLikelihood",
    ) -> DeepKernelGP:
        """Construct a :class:`DeepKernelGP` on the surrogate device."""
        return DeepKernelGP(
            train_x, train_y, likelihood,
            feature_dim=self._feature_dim, hidden=self._hidden,
        ).to(self._device)


# ---------------------------------------------------------------------------
# NaiveGPSurrogate — fixed-hyperparameter RBF GP lower-bound baseline (§16, L2)
# ---------------------------------------------------------------------------

class NaiveGPSurrogate(GPSurrogate):
    """Naive-default GP: fixed RBF kernel, *no* marginal-likelihood tuning.

    Sets ``n_opt_steps=0`` so kernel hyperparameters keep their default
    initialisation (a fixed lengthscale/outputscale/noise) and are never fit to
    the data. This is the "no tuning at all" lower bound: it isolates how much
    of a tuned GP's performance comes from hyperparameter optimisation, and
    quantifies the tuning that TabPFN avoids entirely.

    Args:
        device: PyTorch device string ('cpu' or 'cuda').
        lengthscale: Fixed RBF lengthscale applied to every input dimension.
        outputscale: Fixed signal variance (scale-kernel output scale).
        noise: Fixed Gaussian observation noise variance.
    """

    def __init__(
        self,
        device: str = 'cpu',
        lengthscale: float = 0.2,
        outputscale: float = 1.0,
        noise: float = 1e-2,
    ) -> None:
        # n_opt_steps=0 → fit() skips the optimisation loop entirely.
        super().__init__(device=device, n_opt_steps=0)
        self._lengthscale = lengthscale
        self._outputscale = outputscale
        self._noise = noise

    def _build_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: "gpytorch.likelihoods.GaussianLikelihood",
    ) -> ExactGP:
        """Construct an :class:`ExactGP` with fixed (unoptimised) hyperparameters."""
        model = ExactGP(train_x, train_y, likelihood).to(self._device)
        # Pin hyperparameters to the naive defaults; fit() runs 0 opt steps so
        # these values persist through prediction.
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = self._lengthscale
            model.covar_module.outputscale = self._outputscale
            likelihood.noise = self._noise
        return model


