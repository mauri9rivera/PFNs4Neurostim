"""Update-rule alignment and implicit kernel (task #9, roadmap M10 — primary Hyp C analysis).

Question: when one observation ``(x*, y*)`` is added to a fixed context C, does TabPFN change
its predictions across the grid the way a GP posterior update would?

For every probed model the readout over all N sites is::

    Delta(x) = m(x | C + {(x*, y*)}) - m(x | C)

and it is compared with the exact GP update (hyperparameters frozen at their fit on C)::

    Delta_GP(x) = k_C(x, x*) (y* - mu_C(x*)) / (k_C(x*, x*) + s2)

Probes: anchors ``x*`` stratified over high/low response, edge and centre sites; surprise
sweep ``y* = m(x*|C) + c s(x*|C)`` with ``c`` in a symmetric set (default +-{0.5, 1, 2, 3}),
``m, s`` being the *probed model's own* predictive mean and SD. The linear-regime profile is
the antisymmetrized smallest-surprise pair ``g = (Delta(+c0) - Delta(-c0)) / 2``, which cancels
any value-independent change (e.g. a context-size effect) and every even-order nonlinearity.

Metrics (roadmap M10 table): ``rho_shape`` (+ an off-anchor variant that is not dominated by
the anchor spike), ``rho_shape_null`` (against the GP update of a *different* anchor),
``gain_ratio``, ``lin_r2``, ``saturation_index``, ``ell_hat`` (implicit-kernel lengthscale:
the RBF lengthscale whose frozen GP update *given the same context* best reproduces the
normalized profile, see :func:`fit_conditioned_lengthscale`; a plain half-decay fit, kept as
``ell_half_decay``, measures the conditioned slice, which the context narrows, and failed the
positive control at 40% error), ``ell_cv_anchor`` (non-stationarity),
``sym_err`` and ``psd_neg_mass`` of the implied covariance, ``dsd_surprise_corr``
(value-dependent uncertainty; exactly 0 variation for a GP), ``decay_rho`` (distance decay).

Validation gates (research_design §Hyp C, M10 block) are functions here so that the tests
and the runner apply the same definitions: :func:`positive_control`,
:func:`negative_control`, :func:`seed_floor` and :func:`linear_regime_check`.

Implied covariance (``sym_err``, ``psd_neg_mass``). For a GP the unit-surprise update at an
anchor ``x_j`` read at anchor ``x_i`` is ``K_hat[i, j] = k_C(x_j, x_i) / (k_C(x_i, x_i) + s2)``;
its diagonal is the anchor gain ``gain_i``, and ``K_hat[i, j] / (1 - gain_i)`` equals
``k_C(x_j, x_i) / s2`` exactly, which is symmetric and PSD. The same transform applied to the
PFN's probes gives the covariance its updates *imply*; asymmetry or negative eigenvalues are
properties no GP can have. Undefined (``None``) when an anchor gain is >= 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

from ..data.channels import ChannelData

__all__ = [
    "DEFAULT_SURPRISES",
    "ANCHOR_STRATA",
    "Context",
    "UpdateEngine",
    "GPFrozenEngine",
    "GPRefitEngine",
    "PFNEngine",
    "ProbeResult",
    "draw_context",
    "select_anchors",
    "run_probes",
    "update_metrics",
    "fit_half_decay_lengthscale",
    "fit_conditioned_lengthscale",
    "icc_oneway",
    "gp_channel",
    "shuffle_coordinates",
    "positive_control",
    "negative_control",
    "seed_floor",
    "linear_regime_check",
    "layer_alignment",
]

#: Surprise multipliers c (in units of the model's own predictive SD at the anchor).
DEFAULT_SURPRISES: tuple[float, ...] = (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0)

#: Anchor strata, in the order anchors are drawn round-robin.
ANCHOR_STRATA: tuple[str, ...] = ("high", "low", "edge", "centre")

# ---------------------------------------------------------------------------
# Context and anchors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Context:
    """A fixed context of noisy observations shared by every probed model.

    Attributes:
        sites: Site indices of the observations, shape [t].
        X: Their coordinates, shape [t, D].
        y: One valid noisy trial per observation, shape [t].
    """

    sites: np.ndarray
    X: np.ndarray
    y: np.ndarray


def draw_context(channel: ChannelData, t: int, rng: np.random.Generator) -> Context:
    """Draw ``t`` distinct sites and one random valid trial at each.

    Args:
        channel: Source channel.
        t: Context size (``<= n_sites``).
        rng: Generator.

    Returns:
        The context.

    Raises:
        ValueError: If ``t`` exceeds the number of sites or is < 2.
    """
    if not 2 <= t <= channel.n_sites:
        raise ValueError(f"draw_context: t={t} must be in [2, {channel.n_sites}].")
    sites = np.sort(rng.choice(channel.n_sites, size=t, replace=False))     # [t]
    y = np.empty(t)
    for i, s in enumerate(sites):
        valid = np.flatnonzero(np.isfinite(channel.Y_trials[s]))
        y[i] = channel.Y_trials[s, rng.choice(valid)]
    return Context(sites=sites, X=channel.X_pool[sites].copy(), y=y)


def select_anchors(
    channel: ChannelData,
    n_anchors: int,
    rng: np.random.Generator,
    exclude: Sequence[int] = (),
) -> tuple[np.ndarray, list[str]]:
    """Stratified anchor sites: high / low response, edge, centre (round-robin).

    Strata: ``high`` = top quartile of ``y_gt``; ``low`` = bottom quartile; ``edge`` = a
    coordinate at its min or max; ``centre`` = the quartile of sites closest to the grid
    centre. Anchors are distinct and avoid ``exclude`` (the context sites) when possible.

    Args:
        channel: Source channel.
        n_anchors: Number of anchors.
        rng: Generator.
        exclude: Sites to avoid (context sites).

    Returns:
        ``(anchor_indices [A], stratum_per_anchor)``.
    """
    n = channel.n_sites
    q = max(1, n // 4)
    order = np.argsort(channel.y_gt)
    X = channel.X_pool
    lo, hi = X.min(axis=0), X.max(axis=0)
    is_edge = ((X <= lo) | (X >= hi)).any(axis=1)
    centre_dist = np.linalg.norm(X - (lo + hi) / 2.0, axis=1)
    pools = {
        "high": order[-q:],
        "low": order[:q],
        "edge": np.flatnonzero(is_edge),
        "centre": np.argsort(centre_dist)[:q],
    }
    excluded = set(int(e) for e in exclude)
    chosen: list[int] = []
    strata: list[str] = []
    k = 0
    while len(chosen) < min(n_anchors, n):
        stratum = ANCHOR_STRATA[k % len(ANCHOR_STRATA)]
        k += 1
        pool = [int(s) for s in pools[stratum] if int(s) not in chosen]
        preferred = [s for s in pool if s not in excluded] or pool
        if not preferred:
            if k > 4 * n:  # every stratum exhausted
                break
            continue
        chosen.append(int(rng.choice(preferred)))
        strata.append(stratum)
    return np.asarray(chosen, dtype=int), strata


# ---------------------------------------------------------------------------
# Engines: one fitted context, many one-observation probes
# ---------------------------------------------------------------------------
class UpdateEngine(Protocol):
    """A model that can be conditioned on a context and probed with one extra observation."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Condition on the context, shapes [t, D] and [t]."""

    def base(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and SD at ``Q`` given the context, each [M]."""

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and SD at ``Q`` after adding ``(xs[p], ys[p])``, each [P, M]."""


class GPFrozenEngine:
    """Exact GP update with hyperparameters frozen at their fit on the context.

    Args:
        surrogate: An unfitted :class:`~pfns4neurostim.models.gp.surrogates.GPSurrogate`
            (MLL-tuned) or ``NaiveGPSurrogate`` (fixed, e.g. the true hyperparameters).
        name: Arm label written into the tidy table.
    """

    def __init__(self, surrogate: Any, name: str = "gp_frozen") -> None:
        self.gp = surrogate
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit hyperparameters on the context."""
        self.gp.fit(X, y)

    def base(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Closed-form predictive mean and SD, each [M]."""
        return self.gp.predict_frozen(Q)

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Base plus the frozen closed-form update, each [P, M]."""
        m, s = self.base(Q)
        dm, ds = self.gp.posterior_update(xs, ys, Q)
        return m[None, :] + dm, s[None, :] + ds

    @property
    def lengthscale(self) -> float:
        """Geometric mean of the fitted ARD lengthscales."""
        ls = self.gp.hyperparameters()["lengthscale"]
        return float(np.exp(np.mean(np.log(ls))))

    @property
    def noise_ratio(self) -> float:
        """Fitted noise variance over outputscale (the kernel's noise-to-signal ratio)."""
        hp = self.gp.hyperparameters()
        return float(hp["noise"] / hp["outputscale"])


class GPRefitEngine(GPFrozenEngine):
    """Adaptive-GP comparator: hyperparameters refitted on ``C + {(x*, y*)}`` per probe."""

    def __init__(self, surrogate: Any, name: str = "gp_refit") -> None:
        super().__init__(surrogate, name)

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Base plus the refit update, each [P, M]."""
        m, s = self.base(Q)
        dm, ds = self.gp.posterior_update(xs, ys, Q, refit=True)
        return m[None, :] + dm, s[None, :] + ds


class PFNEngine:
    """TabPFN probed through the batched frozen-preprocessing forward.

    Args:
        device: Torch device.
        random_state: TabPFN preprocessing seed (the *inference seed* of the seed floor).
        max_batch_tokens: Forwarded to :class:`~pfns4neurostim.models.pfn.tabpfn.FrozenTabPFN`.
        name: Arm label.
    """

    def __init__(
        self,
        device: str = "cpu",
        random_state: int = 0,
        max_batch_tokens: int = 400_000,
        name: str = "tabpfn_v2_5",
    ) -> None:
        from ..models.pfn.tabpfn import FrozenTabPFN  # noqa: PLC0415 - heavy import

        self.engine = FrozenTabPFN(device=device, random_state=random_state, max_batch_tokens=max_batch_tokens)
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit (and freeze) preprocessing on the context."""
        self.engine.fit(X, y)

    def base(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and SD given the context alone, each [M]."""
        Qt = self.engine.transform_x(Q)                           # [M, F]
        m, s = self.engine.moments(self.engine.forward(None, None, Qt))  # [1, M]
        return m[0], s[0]

    def probe(self, xs: np.ndarray, ys: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One batched pass over every probe, each output [P, M]."""
        import torch  # noqa: PLC0415

        Qt = self.engine.transform_x(Q)                                    # [M, F]
        Xs = self.engine.transform_x(np.atleast_2d(xs)).unsqueeze(1)       # [P, 1, F]
        yt = self.engine.transform_y(
            torch.as_tensor(np.asarray(ys, dtype=np.float32), device=self.engine.device)
        ).unsqueeze(1)                                                     # [P, 1]
        return self.engine.moments(self.engine.forward(Xs, yt, Qt))        # [P, M] x2


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
@dataclass
class ProbeResult:
    """Raw probe readouts for one (engine, context, anchor set).

    Attributes:
        engine: Engine name.
        anchors: Anchor site indices, shape [A].
        strata: Stratum of each anchor.
        surprises: Surprise multipliers c, shape [S].
        base_mean: Predictive mean given C, shape [M].
        base_sd: Predictive SD given C, shape [M].
        y_star: Injected values, shape [A, S].
        d_mean: Mean change, shape [A, S, M].
        d_sd: SD change, shape [A, S, M].
        d_gp: Frozen-GP reference update for the same ``(x*, y*)``, shape [A, S, M].
        gp_lengthscale: Geometric-mean lengthscale of the reference GP.
        context_X: Context coordinates in readout geometry, shape [t, D].
        noise_ratio: Reference GP noise / outputscale, used by the conditioned
            lengthscale estimator.
        extra: Engine-specific provenance (e.g. refitted lengthscales).
    """

    engine: str
    anchors: np.ndarray
    strata: list[str]
    surprises: np.ndarray
    base_mean: np.ndarray
    base_sd: np.ndarray
    y_star: np.ndarray
    d_mean: np.ndarray
    d_sd: np.ndarray
    d_gp: np.ndarray
    gp_lengthscale: float
    context_X: np.ndarray | None = None
    noise_ratio: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def run_probes(
    engine: UpdateEngine,
    reference: GPFrozenEngine,
    context: Context,
    Q: np.ndarray,
    anchors: np.ndarray,
    strata: list[str],
    surprises: Sequence[float] = DEFAULT_SURPRISES,
) -> ProbeResult:
    """Probe ``engine`` at every (anchor, surprise) and compute the matching GP updates.

    ``reference`` must already be fitted on the same context (it is shared by every engine
    of a cell so all arms are compared against one GP).

    Args:
        engine: Model under test (fitted here on the context).
        reference: Fitted frozen-GP reference.
        context: The shared context.
        Q: Readout sites (the whole grid), shape [M, D].
        anchors: Anchor site indices into ``Q``, shape [A].
        strata: Stratum per anchor.
        surprises: Surprise multipliers, symmetric around 0.

    Returns:
        The raw readouts.

    Raises:
        RuntimeError: If a readout is non-finite.
    """
    c = np.asarray(surprises, dtype=np.float64)                           # [S]
    engine.fit(context.X, context.y)
    base_mean, base_sd = engine.base(Q)                                   # [M], [M]
    y_star = base_mean[anchors][:, None] + c[None, :] * base_sd[anchors][:, None]  # [A, S]
    xs = np.repeat(Q[anchors], len(c), axis=0)                            # [A*S, D]
    ys = y_star.reshape(-1)                                               # [A*S]
    m_new, s_new = engine.probe(xs, ys, Q)                                # [A*S, M]
    A, S, M = len(anchors), len(c), len(Q)
    d_mean = (m_new - base_mean[None, :]).reshape(A, S, M)
    d_sd = (s_new - base_sd[None, :]).reshape(A, S, M)
    d_gp, _ = reference.gp.posterior_update(xs, ys, Q)                    # [A*S, M]
    d_gp = d_gp.reshape(A, S, M)
    for name, arr in (("d_mean", d_mean), ("d_sd", d_sd), ("d_gp", d_gp)):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"run_probes({engine.name}): {name} is non-finite.")
    extra: dict[str, Any] = {}
    refits = getattr(getattr(engine, "gp", None), "last_refit_hyperparameters", None)
    if isinstance(engine, GPRefitEngine) and refits:
        ls = np.array([np.exp(np.mean(np.log(h["lengthscale"]))) for h in refits]).reshape(A, S)
        extra["refit_lengthscale"] = ls
    return ProbeResult(
        engine=engine.name, anchors=np.asarray(anchors), strata=list(strata), surprises=c,
        base_mean=base_mean, base_sd=base_sd, y_star=y_star, d_mean=d_mean, d_sd=d_sd,
        d_gp=d_gp, gp_lengthscale=reference.lengthscale, context_X=np.asarray(context.X),
        noise_ratio=reference.noise_ratio, extra=extra,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson r, or ``None`` when either vector is constant (undefined, not NaN)."""
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def fit_half_decay_lengthscale(g: np.ndarray, dist: np.ndarray) -> float | None:
    """Effective RBF lengthscale of a normalized update profile.

    Least-squares fit of ``g(d) = exp(-d^2 / (2 l^2))`` over sites with ``d > 0``, i.e. the
    lengthscale whose half-decay distance ``l sqrt(2 ln 2)`` matches the profile's. Reported in
    RBF-lengthscale units so it is directly comparable with a fitted GP lengthscale.

    Args:
        g: Profile normalized to 1 at the anchor, shape [M].
        dist: Distance of each site to the anchor, shape [M].

    Returns:
        The lengthscale, or ``None`` when no site has ``d > 0``.
    """
    mask = dist > 0
    if not mask.any():
        return None
    d, y = dist[mask], g[mask]
    lo, hi = float(d.min()) / 10.0, float(d.max()) * 10.0

    def loss(log_l: float) -> float:
        return float(np.sum((y - np.exp(-(d ** 2) / (2.0 * math.exp(2.0 * log_l)))) ** 2))

    res = minimize_scalar(loss, bounds=(math.log(lo), math.log(hi)), method="bounded")
    return float(math.exp(res.x))


def fit_conditioned_lengthscale(
    g: np.ndarray,
    Q: np.ndarray,
    anchor: int,
    context_X: np.ndarray,
    noise_ratio: float,
) -> float | None:
    """Implicit-kernel lengthscale: the RBF GP whose frozen update best explains ``g``.

    For an isotropic RBF GP with lengthscale ``l`` and noise-to-signal ratio ``r``, the
    normalized update of an observation at ``x*`` given context ``X_C`` is the normalized
    posterior-covariance slice ``k_C(x, x*) / k_C(x*, x*)``. It is narrower than the prior
    kernel wherever the context is dense, which is why a half-decay fit on it does not
    recover ``l``. This estimator conditions on the same context and fits ``l`` directly,
    so on data from a known GP it recovers the true lengthscale (the positive control).

    Args:
        g: Profile normalized to 1 at the anchor, shape [M].
        Q: Readout coordinates, shape [M, D].
        anchor: Anchor index into ``Q``.
        context_X: Context coordinates, shape [t, D].
        noise_ratio: ``noise / outputscale`` of the reference GP.

    Returns:
        The fitted lengthscale in coordinate units, or ``None`` for a degenerate grid.
    """
    dist = np.linalg.norm(Q - Q[anchor], axis=1)
    pos = dist[dist > 0]
    if pos.size == 0:
        return None
    lo, hi = float(pos.min()) / 10.0, float(pos.max()) * 10.0
    x_star = Q[anchor][None, :]                                       # [1, D]
    t = len(context_X)

    def profile(ell: float) -> np.ndarray:
        def k(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
            return np.exp(-0.5 * d2 / ell ** 2)

        L = np.linalg.cholesky(k(context_X, context_X) + noise_ratio * np.eye(t))  # [t, t]
        vq = np.linalg.solve(L, k(context_X, Q))                     # [t, M]
        vs = np.linalg.solve(L, k(context_X, x_star))                # [t, 1]
        kc = k(Q, x_star)[:, 0] - (vq * vs).sum(0)                   # [M]
        kss = 1.0 - float((vs ** 2).sum())
        return kc / kss

    def loss(log_l: float) -> float:
        return float(np.sum((g - profile(math.exp(log_l))) ** 2))

    res = minimize_scalar(loss, bounds=(math.log(lo), math.log(hi)), method="bounded")
    return float(math.exp(res.x))


def _antisym(d: np.ndarray, surprises: np.ndarray, c: float) -> np.ndarray:
    """``(d(+c) - d(-c)) / 2`` along the surprise axis (axis 1), shape [A, M]."""
    i_pos = int(np.flatnonzero(np.isclose(surprises, c))[0])
    i_neg = int(np.flatnonzero(np.isclose(surprises, -c))[0])
    return 0.5 * (d[:, i_pos] - d[:, i_neg])


def _symmetric_magnitudes(surprises: np.ndarray) -> list[float]:
    """Positive magnitudes present with both signs, ascending."""
    pos = sorted({float(abs(c)) for c in surprises if c > 0})
    return [c for c in pos if np.isclose(surprises, -c).any()]


def _implied_cov_props(K_hat: np.ndarray) -> tuple[float | None, float | None]:
    """``(sym_err, psd_neg_mass)`` of the implied covariance ``K_hat / (1 - gain)``.

    Args:
        K_hat: Unit-surprise updates between anchors, shape [A, A] (row = probed anchor).

    Returns:
        Both ``None`` when an anchor gain is >= 1 (the transform is undefined).
    """
    gain = np.diag(K_hat)
    if (gain >= 1.0).any() or (gain <= 0.0).any():
        return None, None
    K = K_hat / (1.0 - gain)[:, None]                                   # [A, A]
    sym = float(np.linalg.norm(K - K.T) / np.linalg.norm(K))
    eig = np.linalg.eigvalsh(0.5 * (K + K.T))
    neg = float(np.abs(eig[eig < 0]).sum() / np.abs(eig).sum())
    return sym, neg


def update_metrics(
    res: ProbeResult,
    Q: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-anchor and per-cell M10 metrics of one probe result.

    Args:
        res: Output of :func:`run_probes`.
        Q: Readout coordinates, shape [M, D].

    Returns:
        ``(anchor_rows, cell_row)``; values that are undefined for this input are ``None``.

    Raises:
        ValueError: If the surprise set has no symmetric pair.
    """
    mags = _symmetric_magnitudes(res.surprises)
    if not mags:
        raise ValueError("update_metrics: surprises need at least one +-c pair.")
    c0 = mags[0]
    g = _antisym(res.d_mean, res.surprises, c0)                         # [A, M]
    g_gp = _antisym(res.d_gp, res.surprises, c0)                        # [A, M]
    g1 = _antisym(res.d_mean, res.surprises, mags[1]) if len(mags) > 1 else None
    A = len(res.anchors)
    rows: list[dict[str, Any]] = []
    ell_hats: list[float] = []
    for i, a in enumerate(res.anchors):
        dist = np.linalg.norm(Q - Q[a], axis=1)                         # [M]
        off = dist > 0
        null = [r for j in range(A) if j != i for r in [_pearson(g[i], g_gp[j])] if r is not None]
        null_off = [
            r for j in range(A) if j != i for r in [_pearson(g[i][off], g_gp[j][off])] if r is not None
        ]
        u = res.surprises * res.base_sd[a]                               # [S]
        at_anchor = res.d_mean[i, :, a]                                  # [S]
        lin_r2 = None
        if np.std(at_anchor) > 0:
            coef = np.polyfit(u, at_anchor, 1)
            resid = at_anchor - np.polyval(coef, u)
            lin_r2 = float(1.0 - resid.var() / at_anchor.var())
        sat = None
        if 1.0 in mags and 3.0 in mags:
            d1 = _antisym(res.d_mean, res.surprises, 1.0)[i, a]
            d3 = _antisym(res.d_mean, res.surprises, 3.0)[i, a]
            sat = None if d1 == 0 else float(1.0 - (d3 / d1) / 3.0)
        def _ell(prof: np.ndarray) -> float | None:
            if prof[a] == 0:
                return None
            if res.context_X is None or res.noise_ratio is None:
                return fit_half_decay_lengthscale(prof / prof[a], dist)
            return fit_conditioned_lengthscale(prof / prof[a], Q, int(a), res.context_X, res.noise_ratio)

        ell = _ell(g[i])
        ell_half = None if g[i, a] == 0 else fit_half_decay_lengthscale(g[i] / g[i, a], dist)
        ell_gp_prof = _ell(g_gp[i])
        ell1 = _ell(g1[i]) if g1 is not None else None
        stab = None if ell is None or ell1 is None else float(abs(ell - ell1) / ell)
        dsd = _pearson(res.d_sd[i, :, a], np.abs(u))
        decay = None
        if off.sum() >= 3 and np.std(g[i][off]) > 0:
            decay = float(spearmanr(g[i][off], -dist[off]).correlation)
        if ell is not None:
            ell_hats.append(ell)
        row = {
            "engine": res.engine,
            "anchor": int(a),
            "stratum": res.strata[i],
            "surprise_c0": c0,
            "rho_shape": _pearson(g[i], g_gp[i]),
            "rho_shape_offanchor": _pearson(g[i][off], g_gp[i][off]),
            "rho_shape_null": float(np.mean(null)) if null else None,
            "rho_shape_null_offanchor": float(np.mean(null_off)) if null_off else None,
            "gain_ratio": None if g_gp[i, a] == 0 else float(g[i, a] / g_gp[i, a]),
            "gain": None if res.base_sd[a] <= 0 else float(g[i, a] / (c0 * res.base_sd[a])),
            "lin_r2": lin_r2,
            "saturation_index": sat,
            "ell_hat": ell,
            "ell_half_decay": ell_half,
            "ell_hat_gp_profile": ell_gp_prof,
            "ell_gp": res.gp_lengthscale,
            "ell_linear_stability": stab,
            "dsd_surprise_corr": dsd,
            "decay_rho": decay,
            "base_sd_anchor": float(res.base_sd[a]),
        }
        if "refit_lengthscale" in res.extra:
            row["ell_gp_refit"] = float(np.mean(res.extra["refit_lengthscale"][i]))
        rows.append(row)
    # Implied covariance over the anchor set (unit-surprise updates between anchors).
    u0 = c0 * res.base_sd[res.anchors]                                  # [A]
    K_hat = g[:, res.anchors] / u0[:, None]                              # [A, A]
    sym, neg = _implied_cov_props(K_hat)
    K_hat_gp = g_gp[:, res.anchors] / u0[:, None]
    sym_gp, neg_gp = _implied_cov_props(K_hat_gp)
    cell = {
        "engine": res.engine,
        "n_anchors": A,
        "rho_shape_median": _median([r["rho_shape"] for r in rows]),
        "rho_shape_null_median": _median([r["rho_shape_null"] for r in rows]),
        "rho_shape_offanchor_median": _median([r["rho_shape_offanchor"] for r in rows]),
        "rho_shape_null_offanchor_median": _median([r["rho_shape_null_offanchor"] for r in rows]),
        "ell_hat_median": _median(ell_hats),
        "ell_cv_anchor": float(np.std(ell_hats) / np.mean(ell_hats)) if len(ell_hats) >= 2 else None,
        "ell_gp": res.gp_lengthscale,
        "saturation_index_median": _median([r["saturation_index"] for r in rows]),
        "dsd_surprise_corr_median": _median([r["dsd_surprise_corr"] for r in rows]),
        "sym_err": sym,
        "psd_neg_mass": neg,
        "sym_err_gp": sym_gp,
        "psd_neg_mass_gp": neg_gp,
    }
    return rows, cell


def _median(values: Sequence[float | None]) -> float | None:
    """Median of the defined values, ``None`` if there are none."""
    vals = [float(v) for v in values if v is not None]
    return float(np.median(vals)) if vals else None


# ---------------------------------------------------------------------------
# Validation machinery
# ---------------------------------------------------------------------------
def icc_oneway(ratings: np.ndarray) -> float:
    """One-way random-effects ICC(1,1) (Shrout & Fleiss 1979).

    Args:
        ratings: Targets x raters (e.g. anchors x inference seeds), shape [n, k].

    Returns:
        ``(MSB - MSW) / (MSB + (k - 1) MSW)``; 1.0 when every rater agrees exactly.

    Raises:
        ValueError: With fewer than 2 targets or raters.
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    n, k = ratings.shape
    if n < 2 or k < 2:
        raise ValueError(f"icc_oneway needs >= 2 targets and raters, got {ratings.shape}.")
    grand = ratings.mean()
    msb = k * ((ratings.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msw = ((ratings - ratings.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    if msb + (k - 1) * msw == 0:
        return 1.0
    return float((msb - msw) / (msb + (k - 1) * msw))


def gp_channel(
    grid_side: int,
    lengthscale: float,
    noise_sd: float,
    n_trials: int,
    rng: np.random.Generator,
    outputscale: float = 1.0,
) -> ChannelData:
    """A channel drawn from a known isotropic RBF GP on a ``grid_side x grid_side`` grid.

    Args:
        grid_side: Sites per axis.
        lengthscale: True RBF lengthscale in [0, 1]-scaled coordinates.
        noise_sd: Trial noise SD.
        n_trials: Trials per site.
        rng: Generator.
        outputscale: True signal variance.

    Returns:
        A ``demo1`` channel whose ``meta['true_hyperparameters']`` records the truth.
    """
    ax = np.arange(grid_side)
    coords = np.stack(np.meshgrid(ax, ax, indexing="ij"), axis=-1).reshape(-1, 2)  # [N, 2]
    X = coords / (grid_side - 1)                                                   # [N, 2]
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)                            # [N, N]
    K = outputscale * np.exp(-0.5 * d2 / lengthscale ** 2) + 1e-8 * np.eye(len(X))
    f = np.linalg.cholesky(K) @ rng.normal(size=len(X))                            # [N]
    Y = f[:, None] + noise_sd * rng.normal(size=(len(X), n_trials))                # [N, R]
    return ChannelData(
        "synthetic_neurostim", 0, 0, X, Y, f, coords, (grid_side, grid_side), demo="demo1",
        meta={"true_hyperparameters": {
            "lengthscale": lengthscale, "outputscale": outputscale, "noise": noise_sd ** 2,
        }},
    )


def shuffle_coordinates(channel: ChannelData, rng: np.random.Generator) -> ChannelData:
    """Negative control: the model sees each site at another site's coordinates.

    Readouts are still indexed by the true site, so any model whose update decays with
    *presented* distance loses distance decay in the true geometry.

    Args:
        channel: Source channel.
        rng: Generator.

    Returns:
        A copy whose ``X_pool`` rows are permuted (``meta['coordinate_permutation']``).
    """
    from dataclasses import replace  # noqa: PLC0415

    perm = rng.permutation(channel.n_sites)
    meta = dict(channel.meta)
    meta["coordinate_permutation"] = perm
    return replace(channel, X_pool=channel.X_pool[perm], meta=meta)


EngineFactory = Callable[[int], UpdateEngine]


def _probe_cell(
    channel: ChannelData,
    engine: UpdateEngine,
    reference: GPFrozenEngine,
    t: int,
    n_anchors: int,
    surprises: Sequence[float],
    rng: np.random.Generator,
    readout_X: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], ProbeResult]:
    """Draw a context and anchors, fit the reference, probe the engine, compute metrics.

    Args:
        channel: Channel whose ``X_pool`` the engine sees.
        engine: Model under test.
        reference: Frozen-GP reference (fitted here on ``readout_X`` coordinates).
        t: Context size.
        n_anchors: Anchor count.
        surprises: Surprise multipliers.
        rng: Generator.
        readout_X: True coordinates for the reference GP and distances (defaults to
            ``channel.X_pool``; differs only in the coordinate-shuffle control).

    Returns:
        ``(anchor_rows, cell_row, probe_result)``.
    """
    true_X = channel.X_pool if readout_X is None else readout_X
    ctx = draw_context(channel, t, rng)
    anchors, strata = select_anchors(channel, n_anchors, rng, exclude=ctx.sites)
    reference.fit(true_X[ctx.sites], ctx.y)
    res = run_probes(engine, reference, ctx, channel.X_pool, anchors, strata, surprises)
    if readout_X is not None:
        # Engine readouts are indexed by site already; recompute the reference on true X.
        xs = np.repeat(true_X[anchors], len(res.surprises), axis=0)
        d_gp, _ = reference.gp.posterior_update(xs, res.y_star.reshape(-1), true_X)
        res.d_gp = d_gp.reshape(res.d_mean.shape)
    rows, cell = update_metrics(res, true_X)
    return rows, cell, res


def positive_control(
    engine: UpdateEngine,
    truth_reference: GPFrozenEngine,
    *,
    grid_side: int = 10,
    lengthscale: float = 0.2,
    noise_sd: float = 0.3,
    n_trials: int = 10,
    t: int = 25,
    n_anchors: int = 12,
    surprises: Sequence[float] = DEFAULT_SURPRISES,
    min_r: float = 0.7,
    lengthscale_tol: float = 0.25,
    seed: int = 0,
) -> dict[str, Any]:
    """Positive-control gate: probe ``engine`` on data from a known GP.

    Passes when the median shape alignment with the *true* GP's exact update exceeds
    ``min_r`` and the median recovered effective lengthscale lies within
    ``lengthscale_tol`` (relative) of the true lengthscale.

    Args:
        engine: Model under test.
        truth_reference: Frozen GP with the true hyperparameters (e.g. ``NaiveGPSurrogate``
            pinned to them).
        grid_side, lengthscale, noise_sd, n_trials: Known-GP channel parameters.
        t: Context size.
        n_anchors: Anchors.
        surprises: Surprise multipliers.
        min_r: Pre-declared alignment threshold (r > 0.7).
        lengthscale_tol: Pre-declared relative lengthscale tolerance (25%).
        seed: Seed of the channel, context and anchors.

    Returns:
        Gate record with ``passed`` and the measured quantities.
    """
    rng = np.random.default_rng(seed)
    ch = gp_channel(grid_side, lengthscale, noise_sd, n_trials, rng)
    _, cell, _ = _probe_cell(ch, engine, truth_reference, t, n_anchors, surprises, rng)
    r = cell["rho_shape_median"]
    ell = cell["ell_hat_median"]
    rel = None if ell is None else abs(ell - lengthscale) / lengthscale
    return {
        "gate": "positive_control",
        "engine": engine.name,
        "rho_shape_median": r,
        "ell_hat_median": ell,
        "ell_true": lengthscale,
        "ell_rel_error": rel,
        "passed": bool(r is not None and r > min_r and rel is not None and rel <= lengthscale_tol),
    }


def _site_permutation_null(
    res: ProbeResult, Q: np.ndarray, n_perm: int, rng: np.random.Generator
) -> np.ndarray:
    """Null distribution of the median off-anchor alignment under site permutation.

    Each draw permutes the GP profile's off-anchor entries, which destroys any spatial
    correspondence while keeping both value distributions.

    Args:
        res: Probe result.
        Q: Readout coordinates, shape [M, D].
        n_perm: Number of permutations.
        rng: Generator.

    Returns:
        Median-over-anchors correlations, shape [n_perm].
    """
    c0 = _symmetric_magnitudes(res.surprises)[0]
    g = _antisym(res.d_mean, res.surprises, c0)
    g_gp = _antisym(res.d_gp, res.surprises, c0)
    out = np.empty(n_perm)
    for b in range(n_perm):
        vals = []
        for i, a in enumerate(res.anchors):
            off = np.linalg.norm(Q - Q[a], axis=1) > 0
            r = _pearson(g[i][off], rng.permutation(g_gp[i][off]))
            if r is not None:
                vals.append(r)
        out[b] = np.median(vals) if vals else 0.0
    return out


def negative_control(
    engine: UpdateEngine,
    reference: GPFrozenEngine,
    channel: ChannelData,
    *,
    t: int = 25,
    n_anchors: int = 12,
    surprises: Sequence[float] = DEFAULT_SURPRISES,
    null_quantile: float = 0.95,
    n_perm: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Negative-control gate: a coordinate-shuffled channel falls to the null.

    The engine sees shuffled coordinates; alignment is measured against the GP update in
    the *true* geometry, off the anchor. Passes when the shuffled channel's median off-anchor
    alignment does not exceed the ``null_quantile`` of a site-permutation null (a proper
    "no spatial correspondence" reference; the anchor-permutation null of the intact channel
    is not, because distant anchors on a real map are anti-correlated and push it below 0).

    Args:
        engine: Model under test.
        reference: Frozen-GP reference (MLL-fitted on the true coordinates).
        channel: Intact channel.
        t: Context size.
        n_anchors: Anchors.
        surprises: Surprise multipliers.
        null_quantile: Upper quantile of the null the shuffled score must not exceed.
        n_perm: Site permutations of the null.
        seed: Seed.

    Returns:
        Gate record with ``passed`` and the measured quantities.
    """
    _, cell_i, _ = _probe_cell(
        channel, engine, reference, t, n_anchors, surprises, np.random.default_rng(seed)
    )
    shuffled = shuffle_coordinates(channel, np.random.default_rng(seed + 1))
    _, cell_s, res_s = _probe_cell(
        shuffled, engine, reference, t, n_anchors, surprises, np.random.default_rng(seed),
        readout_X=channel.X_pool,
    )
    null = _site_permutation_null(res_s, channel.X_pool, n_perm, np.random.default_rng(seed + 2))
    bound = float(np.quantile(null, null_quantile))
    score = cell_s["rho_shape_offanchor_median"]
    return {
        "gate": "negative_control",
        "engine": engine.name,
        "rho_offanchor_intact": cell_i["rho_shape_offanchor_median"],
        "rho_offanchor_shuffled": score,
        "null_upper": bound,
        "passed": bool(score is not None and score <= bound),
    }


def seed_floor(
    engine_factory: EngineFactory,
    reference: GPFrozenEngine,
    channel: ChannelData,
    *,
    n_seeds: int = 10,
    t: int = 25,
    n_anchors: int = 12,
    surprises: Sequence[float] = DEFAULT_SURPRISES,
    icc_min: float = 0.75,
    seed: int = 0,
) -> dict[str, Any]:
    """Seed-floor gate: ICC over inference seeds of ``ell_hat`` and ``rho_shape`` per anchor.

    The context and anchors are held fixed; only the engine's inference seed varies.

    Args:
        engine_factory: ``seed -> engine``.
        reference: Frozen-GP reference.
        channel: Channel.
        n_seeds: Number of inference seeds.
        t: Context size.
        n_anchors: Anchors.
        surprises: Surprise multipliers.
        icc_min: Pre-declared ICC threshold (0.75).
        seed: Seed of the context and anchors.

    Returns:
        Gate record; ``passed`` requires both ICCs >= ``icc_min``. The across-seed SD of the
        cell medians is the *seed floor* every reported effect must exceed.
    """
    ells, rhos, cells = [], [], []
    for s in range(n_seeds):
        rows, cell, _ = _probe_cell(
            channel, engine_factory(s), reference, t, n_anchors, surprises, np.random.default_rng(seed)
        )
        ells.append([np.nan if r["ell_hat"] is None else r["ell_hat"] for r in rows])
        rhos.append([np.nan if r["rho_shape"] is None else r["rho_shape"] for r in rows])
        cells.append(cell)
    E = np.asarray(ells).T                                               # [A, n_seeds]
    R = np.asarray(rhos).T
    keep_e = np.isfinite(E).all(axis=1)
    keep_r = np.isfinite(R).all(axis=1)
    icc_ell = icc_oneway(E[keep_e]) if keep_e.sum() >= 2 else None
    icc_rho = icc_oneway(R[keep_r]) if keep_r.sum() >= 2 else None
    meds = [c["rho_shape_median"] for c in cells if c["rho_shape_median"] is not None]
    ellm = [c["ell_hat_median"] for c in cells if c["ell_hat_median"] is not None]
    return {
        "gate": "seed_floor",
        "n_seeds": n_seeds,
        "icc_ell_hat": icc_ell,
        "icc_rho_shape": icc_rho,
        "seed_sd_rho_shape_median": float(np.std(meds)) if len(meds) > 1 else None,
        "seed_sd_ell_hat_median": float(np.std(ellm)) if len(ellm) > 1 else None,
        "passed": bool(
            icc_ell is not None and icc_rho is not None and icc_ell >= icc_min and icc_rho >= icc_min
        ),
    }


def linear_regime_check(anchor_rows: Sequence[dict[str, Any]], tol: float = 0.25) -> dict[str, Any]:
    """Linear-regime gate: the implicit kernel is stable across the two smallest surprises.

    Args:
        anchor_rows: Per-anchor rows (``ell_linear_stability`` = relative change of
            ``ell_hat`` between the two smallest symmetric surprise magnitudes).
        tol: Pre-declared tolerance on the median relative change.

    Returns:
        Gate record with ``passed``.
    """
    vals = [r["ell_linear_stability"] for r in anchor_rows if r.get("ell_linear_stability") is not None]
    med = float(np.median(vals)) if vals else None
    return {
        "gate": "linear_regime",
        "median_rel_change": med,
        "tol": tol,
        "passed": bool(med is not None and med <= tol),
    }


# ---------------------------------------------------------------------------
# Layer-wise arm (task #9 Step 6, secondary)
# ---------------------------------------------------------------------------
def layer_alignment(
    engine: PFNEngine,
    res: ProbeResult,
    Q: np.ndarray,
    *,
    ridge: float = 1.0,
    n_folds: int = 5,
) -> list[dict[str, Any]]:
    """Where in depth the update happens: per-layer representation change vs the GP update.

    For each anchor the antisymmetrized smallest-surprise representation change
    ``dZ_l(x) = (Z_l(x | C + (x*, +)) - Z_l(x | C + (x*, -))) / 2`` is read at every layer
    (one hooked forward per probe). Two readouts per layer: the correlation over sites of
    ``||dZ_l(x)||`` with ``|Delta_GP(x)|`` and the cross-validated ridge-probe R^2 from
    ``dZ_l`` to the PFN's own output change ``Delta_PFN``. ``layer_peak`` is the layer with
    the highest correlation.

    Args:
        engine: A :class:`PFNEngine` already fitted on the result's context (run_probes did).
        res: The PFN probe result (supplies anchors, y*, and both update profiles).
        Q: Readout coordinates, shape [M, D].
        ridge: Ridge penalty of the probe.
        n_folds: Cross-validation folds over sites.

    Returns:
        One row per (anchor, layer) with ``layer_corr``, ``layer_probe_r2`` and ``is_peak``.
    """
    from .embeddings import layer_embeddings  # noqa: PLC0415

    c0 = _symmetric_magnitudes(res.surprises)[0]
    i_pos = int(np.flatnonzero(np.isclose(res.surprises, c0))[0])
    i_neg = int(np.flatnonzero(np.isclose(res.surprises, -c0))[0])
    g = _antisym(res.d_mean, res.surprises, c0)                    # [A, M]
    g_gp = _antisym(res.d_gp, res.surprises, c0)                   # [A, M]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for i, a in enumerate(res.anchors):
        Zp = layer_embeddings(engine.engine, Q, X_extra=Q[a][None], y_extra=[res.y_star[i, i_pos]])
        Zn = layer_embeddings(engine.engine, Q, X_extra=Q[a][None], y_extra=[res.y_star[i, i_neg]])
        dZ = 0.5 * (Zp - Zn)                                        # [L, M, d]
        corrs = []
        for layer in range(dZ.shape[0]):
            norm = np.linalg.norm(dZ[layer], axis=1)               # [M]
            corrs.append(_pearson(norm, np.abs(g_gp[i])))
            folds = np.array_split(rng.permutation(len(Q)), n_folds)
            pred = np.empty(len(Q))
            for f in folds:
                tr = np.setdiff1d(np.arange(len(Q)), f)
                A_ = dZ[layer][tr]
                w = np.linalg.solve(A_.T @ A_ + ridge * np.eye(A_.shape[1]), A_.T @ g[i][tr])
                pred[f] = dZ[layer][f] @ w
            ss = float(((g[i] - g[i].mean()) ** 2).sum())
            r2 = None if ss == 0 else float(1.0 - ((g[i] - pred) ** 2).sum() / ss)
            rows.append({"engine": res.engine, "anchor": int(a), "layer": layer,
                         "layer_corr": corrs[-1], "layer_probe_r2": r2})
        defined = [(c, k) for k, c in enumerate(corrs) if c is not None]
        peak = max(defined)[1] if defined else None
        for r in rows[-dZ.shape[0]:]:
            r["is_peak"] = r["layer"] == peak
    return rows
