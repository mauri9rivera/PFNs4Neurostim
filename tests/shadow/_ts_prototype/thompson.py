"""Thompson-sampling acquisitions: joint (sequential fantasisation), marginal, ensemble.

Phase-3 target: ``src/pfns4neurostim/acquisition/thompson.py``.

Design A (headline ``ts`` for surrogates without an exact joint):
  1. predictive marginals over the pool (1 pass);
  2. candidate set C = {j : q_hi(y_j) >= max_i mean_i} U {argmax mean}, capped at k_max by
     upper quantile;
  3. aleatoric variance per candidate from the replicate-sensitivity probe (1 batched pass)
     or from empirical within-site replicate variance (ablation);
  4. random order over C, chunks of size c: sample y_chunk | context + previous fantasies
     by inverse CDF (ceil(k/c) passes) — chain rule => joint predictive sample over C;
  5. latent correction:
       * ``matheron_diag`` (default): one more pass predicts E[y_rep | context + all
         fantasies] at C and adds independent residual N(0, max(v_rep - sigma_eps^2, 0));
       * ``fantasy_mean``: E[y_rep | context + all fantasies] only (no residual; under-dispersed);
       * ``deflate``: f = m_j + sqrt(s_j) (y_j - m_j) with m_j the conditional mean when sampled;
       * ``none``: predictive joint (ablation).
  6. argmax over C.

Surrogates with ``has_exact_joint`` (GP) draw one exact joint latent sample over the full pool.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import numpy as np

from .protocol import AcqResult, BOState, Marginals, Space, Surrogate


def argmax_random_tiebreak(values: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax with uniform random tie-breaking.

    Args:
        values: Scores, shape [M].
        rng: Explicit Generator.

    Returns:
        Selected index.
    """
    if not np.isfinite(values).any():
        raise RuntimeError("argmax_random_tiebreak: no finite scores.")
    mx = np.nanmax(np.where(np.isfinite(values), values, -np.inf))
    ties = np.flatnonzero(values == mx)
    return int(ties[0] if ties.size == 1 else rng.choice(ties))


def replicate_sensitivity(
    surrogate: Surrogate, X: np.ndarray, pred: Marginals, delta: float = 1.0
) -> np.ndarray:
    """PFN-derived replicate sensitivity s_j = dE[y_rep|x_j, y_j]/dy_j (model-agnostic).

    For a Gaussian model s = v_f / (v_f + sigma_eps^2), so v_f = s * v_y and
    sigma_eps^2 = (1 - s) * v_y. Uses one batched conditional pass with 2k contexts.

    Args:
        surrogate: Fitted surrogate.
        X: Probe inputs, shape [k, D].
        pred: Predictive marginals at X (leading [k]).
        delta: Fantasy offset in predictive SDs.

    Returns:
        s, shape [k], clipped to [0, 1].
    """
    k = X.shape[0]
    m, sd = np.asarray(pred.mean), np.asarray(pred.std)                   # [k], [k]
    Xq = np.concatenate([X, X], 0)[:, None, :]                            # [2k, 1, D]
    ye = np.concatenate([m + delta * sd, m - delta * sd])[:, None]        # [2k, 1]
    mu = np.asarray(surrogate.conditional_marginals(Xq, Xq, ye).mean)[:, 0]  # [2k]
    s = (mu[:k] - mu[k:]) / (2.0 * delta * sd)
    if not np.isfinite(s).all():
        raise RuntimeError("replicate_sensitivity produced NaN/Inf.")
    return np.clip(s, 0.0, 1.0)


def empirical_noise_var(
    obs_idx: np.ndarray, y_obs: np.ndarray, fallback: np.ndarray
) -> np.ndarray | float:
    """Pooled within-site variance of the agent's own replicate observations (ablation).

    Args:
        obs_idx: Pool indices of observations, shape [n].
        y_obs: Observed values, shape [n].
        fallback: Returned when no site has >= 2 replicates.

    Returns:
        Scalar pooled variance, or ``fallback`` if unavailable.
    """
    ss, dof = 0.0, 0
    for j in np.unique(obs_idx):
        v = y_obs[obs_idx == j]
        if v.size >= 2:
            ss += float(((v - v.mean()) ** 2).sum())
            dof += v.size - 1
    return fallback if dof == 0 else ss / dof


def select_candidates(pred: Marginals, k_max: int, q_hi: float) -> np.ndarray:
    """Potentially-optimal candidate set from predictive marginals.

    Args:
        pred: Predictive marginals over the pool (leading [M]).
        k_max: Cap on |C| (None-like values <= 0 mean no cap).
        q_hi: Upper quantile level used for the optimism test.

    Returns:
        Sorted candidate indices, shape [k].
    """
    mean = np.asarray(pred.mean)                                          # [M]
    upper = pred.icdf(np.full(mean.shape, q_hi))                          # [M]
    keep = np.flatnonzero(upper >= mean.max())
    keep = np.union1d(keep, [int(np.argmax(mean))])
    if 0 < k_max < keep.size:
        keep = keep[np.argsort(upper[keep])[::-1][:k_max]]
    return np.sort(keep)


@dataclass
class JointTSConfig:
    """Parameters of the sequential joint sampler (all recorded in result metadata)."""

    # k_max=32 lost 26-42% (NHP, M=96) and ~89% (5d_rat, M=2048) of the PFN's own argmax
    # mass (test #6), so the cap is only a large-pool safeguard: 2D grids (M<=128) run the
    # full optimism set.
    k_max: int = 256
    # chunk 2/4 raise TV to P(optimal) 0.030 -> 0.042/0.048 (test #2); chunking is only used
    # when |C| exceeds max_chain_passes (chunk = ceil(|C| / max_chain_passes)).
    chunk_size: int = 1
    max_chain_passes: int = 96
    q_hi: float = 0.99
    latent_mode: Literal["matheron_diag", "fantasy_mean", "deflate", "none"] = "matheron_diag"
    noise_source: Literal["probe", "empirical"] = "probe"
    probe_delta: float = 1.0


def sequential_joint_sample(
    surrogate: Surrogate,
    X_cand: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    cfg: JointTSConfig,
    noise_var: Optional[np.ndarray] = None,
    sens: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Draw joint samples over candidates by sequential fantasisation (batched over samples).

    Args:
        surrogate: Fitted surrogate exposing ``conditional_marginals``.
        X_cand: Candidate inputs, shape [k, D].
        n_samples: Number of joint samples S (all chains share one random order).
        rng: Explicit Generator (order, uniforms, residual normals).
        cfg: Sampler configuration.
        noise_var: Aleatoric variance per candidate [k] (needed for latent modes).
        sens: Replicate sensitivity per candidate [k] (needed for ``deflate``).

    Returns:
        Dict with ``'y'`` [S, k] predictive joint sample, ``'f'`` [S, k] latent (or = y),
        ``'order'`` [k], ``'n_passes'`` scalar array.
    """
    k, D = X_cand.shape
    S = n_samples
    order = rng.permutation(k)
    y = np.zeros((S, k))
    cond_mean = np.zeros((S, k))
    Xf = np.zeros((S, 0, D))
    yf = np.zeros((S, 0))
    n_passes = 0
    c = max(1, cfg.chunk_size)
    for start in range(0, k, c):
        chunk = order[start:start + c]                                         # [c]
        Xq = np.broadcast_to(X_cand[chunk], (S, chunk.size, D))                # [S, c, D]
        marg = surrogate.conditional_marginals(Xq, Xf, yf)                     # [S, c]
        n_passes += 1
        draw = marg.icdf(rng.random((S, chunk.size)))                          # [S, c]
        y[:, chunk] = draw
        cond_mean[:, chunk] = np.asarray(marg.mean)
        Xf = np.concatenate([Xf, Xq], axis=1)                                  # [S, r+c, D]
        yf = np.concatenate([yf, draw], axis=1)                                # [S, r+c]
    if cfg.latent_mode == "none":
        f = y
    elif cfg.latent_mode == "deflate":
        if sens is None:
            raise ValueError("deflate latent mode requires sens.")
        f = cond_mean + np.sqrt(sens)[None, :] * (y - cond_mean)
    elif cfg.latent_mode in ("matheron_diag", "fantasy_mean"):
        Xq = np.broadcast_to(X_cand, (S, k, D))                                # [S, k, D]
        rep = surrogate.conditional_marginals(Xq, Xf, yf)                      # [S, k]
        n_passes += 1
        f = np.asarray(rep.mean)                                               # E[f | ctx, fantasies]
        if cfg.latent_mode == "matheron_diag":
            if noise_var is None:
                raise ValueError("matheron_diag latent mode requires noise_var.")
            resid = np.clip(np.asarray(rep.std) ** 2 - noise_var[None, :], 0.0, None)  # [S, k]
            f = f + np.sqrt(resid) * rng.standard_normal((S, k))
    else:
        raise ValueError(f"Unknown latent_mode {cfg.latent_mode!r}")
    if not (np.isfinite(y).all() and np.isfinite(f).all()):
        raise RuntimeError("sequential_joint_sample produced NaN/Inf.")
    return {"y": y, "f": f, "order": order, "n_passes": np.asarray(n_passes)}


class TSJoint:
    """Headline Thompson sampling: exact joint (if available) else sequential fantasisation.

    Args:
        cfg: Sampler configuration for non-exact surrogates.
        space: ``'latent'`` (headline) or ``'predictive'``.
        use_exact_if_available: Use ``surrogate.sample_joint`` over the full pool when exact.
    """

    name = "ts"

    def __init__(self, cfg: Optional[JointTSConfig] = None, space: Space = "latent",
                 use_exact_if_available: bool = True) -> None:
        self.cfg = cfg or JointTSConfig()
        self.space = space
        self.use_exact_if_available = use_exact_if_available

    def params(self) -> dict[str, Any]:
        return {"space": self.space, "use_exact_if_available": self.use_exact_if_available, **asdict(self.cfg)}

    def draw(
        self, surrogate: Surrogate, X_pool: np.ndarray, state: Optional[BOState],
        rng: np.random.Generator, n_samples: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Joint samples and the pool indices they cover.

        Returns:
            (samples [S, k], cand_idx [k], diagnostics).
        """
        t0 = time.perf_counter()
        if surrogate.has_exact_joint and self.use_exact_if_available:
            s = surrogate.sample_joint(X_pool, n_samples, rng, space=self.space)
            return s, np.arange(X_pool.shape[0]), {"k": X_pool.shape[0], "n_passes": 0,
                                                   "time_s": time.perf_counter() - t0}
        cfg = self.cfg
        pred = surrogate.predict_marginals(X_pool, space="predictive")        # [M]
        cand = select_candidates(pred, cfg.k_max, cfg.q_hi)                   # [k]
        Xc = X_pool[cand]
        pred_c_sd = np.asarray(pred.std)[cand]
        noise_var, sens = None, None
        n_passes = 1
        latent = self.space == "latent" and cfg.latent_mode != "none"
        if latent:
            pc = _subset_marginals(pred, cand)
            sens = replicate_sensitivity(surrogate, Xc, pc, cfg.probe_delta)  # [k]
            n_passes += 1
            noise_var = (1.0 - sens) * pred_c_sd ** 2
            if cfg.noise_source == "empirical":
                if state is None:
                    raise ValueError("empirical noise_source requires BOState.")
                pooled = empirical_noise_var(state.obs_idx, state.y_obs, fallback=np.nan)
                if np.isfinite(pooled):
                    noise_var = np.minimum(np.full(cand.size, pooled), pred_c_sd ** 2)
                    sens = 1.0 - noise_var / np.maximum(pred_c_sd ** 2, 1e-12)
        chunk = max(cfg.chunk_size, -(-cand.size // max(cfg.max_chain_passes, 1)))
        run_cfg = JointTSConfig(**{**asdict(cfg), "chunk_size": chunk,
                                   "latent_mode": cfg.latent_mode if latent else "none"})
        out = sequential_joint_sample(surrogate, Xc, n_samples, rng, run_cfg, noise_var, sens)
        n_passes += int(out["n_passes"])
        return out["f"], cand, {"k": int(cand.size), "chunk_size": chunk, "n_passes": n_passes,
                                "time_s": time.perf_counter() - t0}

    def __call__(self, surrogate: Surrogate, X_pool: np.ndarray, state: BOState,
                 rng: np.random.Generator) -> AcqResult:
        s, cand, diag = self.draw(surrogate, X_pool, state, rng, 1)
        scores = np.full(X_pool.shape[0], -np.inf)
        scores[cand] = s[0]
        return AcqResult(argmax_random_tiebreak(scores, rng), scores, diag)


class TSMarginal:
    """Ablation: independent per-candidate draws (the current ``src`` sampler, inverse-CDF within bin).

    Args:
        space: ``'predictive'`` (as currently implemented) or ``'latent'``.
    """

    name = "ts_marginal"

    def __init__(self, space: Space = "predictive") -> None:
        self.space = space

    def params(self) -> dict[str, Any]:
        return {"space": self.space}

    def draw(self, surrogate: Surrogate, X_pool: np.ndarray, rng: np.random.Generator,
             n_samples: int = 1) -> np.ndarray:
        """Independent samples, shape [S, M]."""
        marg = surrogate.predict_marginals(X_pool, space=self.space)          # [M]
        M = X_pool.shape[0]
        u = rng.random((M, n_samples))                                        # [M, S]
        return marg.icdf(u).T                                                 # [S, M]

    def __call__(self, surrogate: Surrogate, X_pool: np.ndarray, state: BOState,
                 rng: np.random.Generator) -> AcqResult:
        t0 = time.perf_counter()
        s = self.draw(surrogate, X_pool, rng, 1)[0]
        return AcqResult(argmax_random_tiebreak(s, rng), s, {"time_s": time.perf_counter() - t0})


@dataclass
class EnsembleTSConfig:
    """Design B parameters."""

    n_anchors: int = 4
    bootstrap: bool = True
    probe_delta: float = 1.0


class TSEnsemble:
    """Design B: perturbed-context ensemble TS (bootstrap + target noise + prior anchors).

    One function sample = predictive mean over the pool under a context that is
    (i) bootstrap-resampled, (ii) target-perturbed with the probe's aleatoric variance and
    (iii) augmented with ``n_anchors`` unobserved pool points whose y is drawn from their
    predictive marginals. Only valid if it passes the same calibration tests as design A.
    """

    name = "ts_ensemble"

    def __init__(self, cfg: Optional[EnsembleTSConfig] = None) -> None:
        self.cfg = cfg or EnsembleTSConfig()

    def params(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def draw(self, surrogate: Any, X_pool: np.ndarray, state: BOState, rng: np.random.Generator,
             n_samples: int = 1) -> np.ndarray:
        """Function samples over the pool, shape [S, M]."""
        cfg = self.cfg
        M, D = X_pool.shape
        n = state.X_obs.shape[0]
        pred_obs = surrogate.predict_marginals(state.X_obs, space="predictive")      # [n]
        s_obs = replicate_sensitivity(surrogate, state.X_obs, pred_obs, cfg.probe_delta)
        noise_sd = np.sqrt((1 - s_obs) * np.asarray(pred_obs.std) ** 2)             # [n]
        pred_pool = surrogate.predict_marginals(X_pool, space="predictive")          # [M]
        unobs = np.setdiff1d(np.arange(M), state.obs_idx)
        Xc = np.zeros((n_samples, n + cfg.n_anchors, D))
        yc = np.zeros((n_samples, n + cfg.n_anchors))
        for i in range(n_samples):
            rows = rng.integers(0, n, n) if cfg.bootstrap else np.arange(n)
            Xc[i, :n] = state.X_obs[rows]
            yc[i, :n] = state.y_obs[rows] + noise_sd[rows] * rng.standard_normal(n)
            if cfg.n_anchors > 0 and unobs.size > 0:
                a = rng.choice(unobs, size=min(cfg.n_anchors, unobs.size), replace=False)
                a = np.resize(a, cfg.n_anchors)
                Xc[i, n:] = X_pool[a]
                yc[i, n:] = pred_pool.icdf(rng.random(M))[a]
        Xq = np.broadcast_to(X_pool, (n_samples, M, D))
        if hasattr(surrogate, "predict_with_context"):
            marg = surrogate.predict_with_context(Xq, Xc, yc)
        else:
            marg = surrogate.conditional_marginals(Xq, Xc, yc, include_context=False)
        out = np.asarray(marg.mean)                                                  # [S, M]
        if not np.isfinite(out).all():
            raise RuntimeError("TSEnsemble produced NaN/Inf.")
        return out

    def __call__(self, surrogate: Surrogate, X_pool: np.ndarray, state: BOState,
                 rng: np.random.Generator) -> AcqResult:
        t0 = time.perf_counter()
        s = self.draw(surrogate, X_pool, state, rng, 1)[0]
        return AcqResult(argmax_random_tiebreak(s, rng), s, {"time_s": time.perf_counter() - t0})


def _subset_marginals(pred: Marginals, idx: np.ndarray) -> Marginals:
    """Marginals restricted to ``idx`` (leading [M] -> [k])."""
    from .protocol import BarMarginals, GaussianMarginals

    if isinstance(pred, GaussianMarginals):
        return GaussianMarginals(pred.mean[idx], pred.std[idx], pred.space)
    if isinstance(pred, BarMarginals):
        sh = None if pred.shrink is None else pred.shrink[idx]
        return BarMarginals(pred.logprobs[idx], pred.borders, pred.space, sh)
    raise TypeError(f"Unsupported marginals type {type(pred).__name__}")
