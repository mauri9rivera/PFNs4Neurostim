"""Synthetic neurostimulation maps (Demo 1, roadmap S0).

A synthetic channel is a map with **exact ground truth** built to look like a real one:

    drive(x) = sum_k a_k * exp(-0.5 * || R_k^T (x - c_k) / l_k ||^2)       K anisotropic hotspots
    mu(x)    = baseline + s * tanh(drive(x) / s)                           sigmoidal saturation
    y(x, r)  = mu(x) * exp(sigma * e - sigma^2 / 2),  e ~ N(0, 1)          multiplicative noise

with ``sigma = sqrt(log(1 + cv^2))``, so every trial is positive, its mean is ``mu(x)``
and its standard deviation is ``cv * mu(x)``: noise grows with the response, as it does
in recorded EMG (mean-SD correlation ~0.92). ``R_k`` rotates the first two axes. The
sites are the real electrode coordinates of the source channel (10x10, 8x4, 8x8 grids,
or the 5D condition set), so the geometry is never invented.

**Nominal anchor.** :func:`fit_generator_to_channel` fits the parameters to a real
channel (least squares on the raw ground-truth map, noise CV matched to its achieved SNR), so
a Demo 1 sweep starts from maps statistically matched to Demo 2. :func:`meta_features`
computes the statistics the roadmap uses to validate that match (Moran's I, skewness,
CV, mean-SD correlation).

Synthetic arrays go through :func:`~.preprocessing.preprocess_channel`, the same
preprocessing contract as the in-vivo data, so every model sees identically scaled
inputs whichever demo a channel comes from.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Iterator

import numpy as np
from scipy import optimize, stats

from ..seeding import rng_for
from .channels import ChannelData
from .preprocessing import DEFAULT_NORMALIZATION, preprocess_channel
from .snr import achieved_snr_db

__all__ = [
    "Hotspot",
    "GeneratorParams",
    "mean_map",
    "hotspot_drive",
    "generate_neurostim_map",
    "fit_generator_to_channel",
    "synthetic_channels",
    "meta_features",
    "morans_i",
]

#: Dataset-name prefix of Demo 1 channels; the source dataset follows (``synthetic_nhp``),
#: so synthetic channels from different sources never share a cache identity.
SYNTHETIC_PREFIX: str = "synthetic_"


@dataclass(frozen=True)
class Hotspot:
    """One anisotropic Gaussian hotspot.

    Attributes:
        center: Centre in electrode coordinates, shape [D].
        lengthscale: Width along each (rotated) axis, in electrode pitch, shape [D].
        amplitude: Peak drive before saturation, in raw response units.
        rotation: Rotation of the first two axes, in radians (ignored for D = 1).
    """

    center: np.ndarray
    lengthscale: np.ndarray
    amplitude: float
    rotation: float = 0.0


@dataclass(frozen=True)
class GeneratorParams:
    """Everything needed to draw one synthetic channel.

    Attributes:
        ch2xy: Electrode coordinates of the sites, shape [N, D].
        grid_shape: Shape of the electrode grid (recorded on the channel).
        hotspots: The hotspots, primary first.
        baseline: Response with no drive, raw units (> 0, so trials stay positive).
        saturation: Ceiling ``s`` of the ``tanh`` recruitment curve; ``inf`` is linear.
        noise_cv: Trial standard deviation over the site mean (heteroscedastic noise).
        n_trials: Trials per site R.
    """

    ch2xy: np.ndarray
    grid_shape: tuple[int, ...]
    hotspots: tuple[Hotspot, ...]
    baseline: float
    saturation: float
    noise_cv: float
    n_trials: int

    def __post_init__(self) -> None:
        """Fail fast on parameters that would yield non-positive or undefined trials."""
        if self.baseline <= 0.0:
            raise ValueError(f"GeneratorParams: baseline must be > 0, got {self.baseline}.")
        if self.saturation <= 0.0 or self.noise_cv < 0.0 or self.n_trials < 2:
            raise ValueError(
                f"GeneratorParams: need saturation > 0, noise_cv >= 0 and n_trials >= 2, got "
                f"{self.saturation}, {self.noise_cv}, {self.n_trials}."
            )
        if not self.hotspots:
            raise ValueError("GeneratorParams: at least one hotspot is required.")


def hotspot_drive(coords: np.ndarray, spot: Hotspot) -> np.ndarray:
    """Unsaturated drive of one hotspot at every site.

    Args:
        coords: Site coordinates, shape [N, D].
        spot: The hotspot.

    Returns:
        Drive, shape [N].
    """
    diff = coords - np.asarray(spot.center, dtype=np.float64)[None, :]    # [N, D]
    if diff.shape[1] >= 2 and spot.rotation != 0.0:
        c, s = np.cos(spot.rotation), np.sin(spot.rotation)
        x0, x1 = diff[:, 0].copy(), diff[:, 1].copy()
        diff[:, 0], diff[:, 1] = c * x0 + s * x1, -s * x0 + c * x1          # R^T (x - c)
    z = diff / np.asarray(spot.lengthscale, dtype=np.float64)[None, :]     # [N, D]
    return spot.amplitude * np.exp(-0.5 * np.sum(z * z, axis=1))          # [N]


def mean_map(params: GeneratorParams, coords: np.ndarray | None = None) -> np.ndarray:
    """Noise-free response ``mu(x)`` at every site (the Demo 1 ground truth, raw units).

    Args:
        params: Generator parameters.
        coords: Sites to evaluate, shape [N, D]; defaults to ``params.ch2xy``.

    Returns:
        ``mu``, shape [N].
    """
    coords = np.asarray(params.ch2xy if coords is None else coords, dtype=np.float64)  # [N, D]
    drive = sum(hotspot_drive(coords, spot) for spot in params.hotspots)              # [N]
    if np.isinf(params.saturation):
        return params.baseline + drive
    return params.baseline + params.saturation * np.tanh(drive / params.saturation)


def generate_neurostim_map(
    params: GeneratorParams,
    rng: np.random.Generator,
    *,
    dataset: str = "synthetic_neurostim",
    subject: int = 0,
    emg: int = 0,
    normalization: str = DEFAULT_NORMALIZATION,
) -> ChannelData:
    """Draw one synthetic channel (Demo 1) with exact ground truth.

    Args:
        params: Generator parameters.
        rng: Seeded generator (trial noise).
        dataset: Dataset label for the tidy schema.
        subject: Pseudo-subject index.
        emg: Pseudo-EMG index.
        normalization: Preprocessing mode (same contract as the in-vivo data).

    Returns:
        A :class:`ChannelData` with ``demo='demo1'``; ``y_gt`` is the noise-free map and
        ``meta['generator']`` holds ``params``.

    Raises:
        RuntimeError: If the map or the trials are non-finite.
    """
    mu = mean_map(params)                                                  # [N]
    sigma = float(np.sqrt(np.log1p(params.noise_cv ** 2)))
    eps = rng.standard_normal((mu.size, params.n_trials))                  # [N, R]
    trials = mu[:, None] * np.exp(sigma * eps - 0.5 * sigma ** 2)          # [N, R], mean mu, sd cv*mu
    if not (np.isfinite(mu).all() and np.isfinite(trials).all()):
        raise RuntimeError(f"generate_neurostim_map: non-finite map for {dataset}-s{subject}-e{emg}.")
    raw = {
        "ch2xy": np.asarray(params.ch2xy),
        "sorted_resp": trials[:, None, :],                                 # [N, 1, R]
        "sorted_respMean": mu[:, None],                                    # [N, 1]: exact GT
    }
    pre = preprocess_channel(raw, 0, normalization)
    return ChannelData(
        dataset=dataset,
        subject=subject,
        emg=emg,
        X_pool=pre.X_pool,
        Y_trials=pre.Y_trials,
        y_gt=pre.y_gt,
        ch2xy=np.asarray(params.ch2xy),
        grid_shape=tuple(params.grid_shape),
        demo="demo1",
        normalization=normalization,
        meta={"scaler_y": pre.scaler_y, "generator": params},
    )


def _raw_trials_and_gt(channel: ChannelData) -> tuple[np.ndarray, np.ndarray]:
    """Return a channel's trials and ground truth in raw units.

    Args:
        channel: A channel carrying ``meta['scaler_y']``.

    Returns:
        ``(trials [N, R], gt [N])`` in raw response units.

    Raises:
        ValueError: If the channel has no scaler to invert.
    """
    gt = channel.to_raw(channel.y_gt)
    if gt is None:
        raise ValueError(f"{channel.label}: no scaler_y in meta, so raw units are unavailable.")
    flat = channel.to_raw(channel.Y_trials.reshape(-1))
    return flat.reshape(channel.Y_trials.shape), gt                        # [N, R], [N]


def _is_collapsed(theta: np.ndarray, unpack: Callable[[np.ndarray], GeneratorParams]) -> bool:
    """Whether fitted parameters give no usable map.

    Args:
        theta: Fitted parameter vector (last entry is log-saturation).
        unpack: Maps ``theta`` to :class:`GeneratorParams`.

    Returns:
        True if the saturation underflows to 0 or the mean map is constant.
    """
    if not np.exp(theta[-1]) > 0.0:
        return True
    return not np.var(mean_map(unpack(theta))) > 0.0


def fit_generator_to_channel(
    channel: ChannelData,
    *,
    n_hotspots: int = 1,
    min_lengthscale: float = 0.3,
    min_saturation_frac: float = 0.5,
) -> GeneratorParams:
    """Fit generator parameters to a real channel (the Demo 1 nominal anchor).

    Hotspots are added greedily at the largest remaining residual, then all parameters
    (centres, log-lengthscales, log-amplitudes, rotations, log-baseline, log-saturation)
    are refined jointly by least squares against the raw ground-truth map. The noise CV
    is then set so the synthetic channel reproduces the real channel's **achieved SNR**
    (the K2 axis): with multiplicative noise the within-site variance is ``cv^2 mu^2``, so
    ``SNR = Var_s[mu] / (cv^2 * mean_s[mu^2])`` and ``cv`` follows in closed form.

    Args:
        channel: Real channel to imitate (must carry ``meta['scaler_y']``).
        n_hotspots: Number of hotspots K.
        min_lengthscale: Lower bound on a lengthscale, in electrode pitch, so a hotspot
            cannot collapse onto a single noisy site.
        min_saturation_frac: Lower bound on the saturation ceiling, as a fraction of the
            ground-truth range (peak - floor), applied only in a refit when the unbounded
            fit collapses: as ``s -> 0``, ``s * tanh(drive / s)`` vanishes and the map is
            constant (seen on 5d_rat s4-e2 with two hotspots).

    Returns:
        Fitted :class:`GeneratorParams` on the channel's own electrode coordinates.

    Raises:
        ValueError: If ``n_hotspots < 1`` or the channel has no raw scale.
        RuntimeError: If the fitted map is constant (no signal to match an SNR to).
    """
    if n_hotspots < 1:
        raise ValueError(f"fit_generator_to_channel: n_hotspots must be >= 1, got {n_hotspots}.")
    _, gt = _raw_trials_and_gt(channel)                                    # [N]
    coords = np.asarray(channel.ch2xy, dtype=np.float64)                   # [N, D]
    n, d = coords.shape
    span = np.ptp(coords, axis=0)                                          # [D]
    span = np.where(span > 0, span, 1.0)

    floor = max(float(np.percentile(gt, 5)), 1e-6 * max(float(np.max(np.abs(gt))), 1.0))
    peak = float(np.max(gt))

    # --- parameter vector: [c(D), log l(D), log a, theta] * K + [log b, log s] ----------
    per = 2 * d + 2

    def unpack(theta: np.ndarray) -> GeneratorParams:
        spots = []
        for k in range(len(theta[:-2]) // per):
            p = theta[k * per:(k + 1) * per]
            spots.append(Hotspot(p[:d], np.exp(p[d:2 * d]), float(np.exp(p[2 * d])), float(p[2 * d + 1])))
        return GeneratorParams(
            ch2xy=coords, grid_shape=tuple(channel.grid_shape), hotspots=tuple(spots),
            baseline=float(np.exp(theta[-2])), saturation=float(np.exp(theta[-1])),
            noise_cv=0.0, n_trials=max(channel.n_trials, 2),
        )

    def residual(theta: np.ndarray) -> np.ndarray:
        return mean_map(unpack(theta)) - gt                                # [N]

    theta = np.array([np.log(floor), np.log(2.0 * max(peak - floor, 1e-6))])
    for _ in range(n_hotspots):
        current = mean_map(unpack(theta)) if len(theta) > 2 else np.full(n, floor)
        centre = coords[int(np.argmax(gt - current))]
        amp = max(float(np.max(gt - current)), 1e-3 * max(peak - floor, 1e-6))
        spot = np.concatenate([centre, np.log(np.maximum(span / 4.0, min_lengthscale)), [np.log(amp), 0.0]])
        theta = np.concatenate([theta[:-2], spot, theta[-2:]])

    k = n_hotspots
    # Centres may sit up to one array span off the edge: a hotspot whose peak lies just
    # beyond the recorded electrodes is common, and bounding it to the array costs fit quality.
    lower = np.concatenate(
        [np.concatenate([coords.min(axis=0) - span, np.full(d, np.log(min_lengthscale)), [-np.inf, -np.pi]])] * k
        + [np.array([-np.inf, -np.inf])]
    )
    upper = np.concatenate(
        [np.concatenate([coords.max(axis=0) + span, np.log(2.0 * span), [np.inf, np.pi]])] * k
        + [np.array([np.inf, np.inf])]
    )
    theta = np.clip(theta, lower + 1e-9, upper - 1e-9)
    fit = optimize.least_squares(residual, theta, bounds=(lower, upper))

    if _is_collapsed(fit.x, unpack):
        # Fallback only: a finite bound changes the solver's variable scaling, so bounding
        # every fit would move every twin (and orphan the cells cached against them).
        lower[-1] = np.log(min_saturation_frac * max(peak - floor, 1e-6))
        fit = optimize.least_squares(residual, np.clip(theta, lower + 1e-9, upper - 1e-9), bounds=(lower, upper))
        if _is_collapsed(fit.x, unpack):
            raise RuntimeError(
                f"fit_generator_to_channel({channel.label}, n_hotspots={n_hotspots}): fitted map is "
                f"constant even with the saturation floor (log s = {fit.x[-1]:.3g})."
            )

    params = unpack(fit.x)
    mu = mean_map(params)                                                  # [N]
    target = 10.0 ** (achieved_snr_db(channel) / 10.0)
    cv = float(np.sqrt(np.var(mu) / (np.mean(mu ** 2) * target)))
    return replace(params, noise_cv=cv)


def synthetic_channels(
    real_channels: Iterable[ChannelData],
    *,
    n_hotspots: int,
    seed: int,
    normalization: str = DEFAULT_NORMALIZATION,
) -> Iterator[ChannelData]:
    """Yield one nominal Demo 1 channel per real channel, fitted to it.

    The synthetic channel keeps the real channel's subject/EMG indices and electrode
    geometry and is labelled ``synthetic_<source dataset>``, so it pairs one-to-one
    with its in-vivo counterpart (the S10 bridge) and never collides in the cell cache.

    Args:
        real_channels: In-vivo channels to anchor on.
        n_hotspots: Hotspots per fitted map.
        seed: Base seed; each channel's trial noise is seeded from it and its label.
        normalization: Preprocessing mode.

    Yields:
        Synthetic :class:`ChannelData` with ``meta['source_label']`` set.
    """
    for real in real_channels:
        params = fit_generator_to_channel(real, n_hotspots=n_hotspots)
        synth = generate_neurostim_map(
            params,
            rng_for(real.label, "demo1_nominal", base_seed=seed),
            dataset=f"{SYNTHETIC_PREFIX}{real.dataset}",
            subject=real.subject,
            emg=real.emg,
            normalization=normalization,
        )
        yield replace(synth, meta={**synth.meta, "source_label": real.label})


def morans_i(values: np.ndarray, coords: np.ndarray) -> float:
    """Moran's I spatial autocorrelation with rook (unit grid-distance) neighbours.

    Args:
        values: One value per site, shape [N].
        coords: Integer electrode coordinates, shape [N, D].

    Returns:
        Moran's I; NaN when no two sites are neighbours or the values are constant.
    """
    z = np.asarray(values, dtype=np.float64) - float(np.mean(values))     # [N]
    c = np.asarray(coords, dtype=np.float64)                               # [N, D]
    w = (np.abs(c[:, None, :] - c[None, :, :]).sum(axis=-1) == 1).astype(np.float64)  # [N, N]
    w_sum, denom = float(w.sum()), float(np.sum(z * z))
    if w_sum == 0.0 or denom == 0.0:
        return float("nan")
    return float(z.size / w_sum * (z @ w @ z) / denom)


def meta_features(channel: ChannelData) -> dict[str, float]:
    """Map statistics used to validate the generator against real channels (roadmap S0).

    Args:
        channel: Channel to describe (needs ``meta['scaler_y']`` for raw units).

    Returns:
        ``morans_i`` (of the GT map), ``skewness`` and ``cv`` (of the raw GT across
        sites), and ``mean_sd_corr`` (Pearson correlation across sites of the trial
        mean and trial SD, the heteroscedasticity signature).
    """
    trials, gt = _raw_trials_and_gt(channel)                               # [N, R], [N]
    means = np.nanmean(trials, axis=1)                                     # [N]
    sds = np.nanstd(trials, axis=1, ddof=1)                                # [N]
    ok = np.isfinite(sds)
    return {
        "morans_i": morans_i(gt, channel.ch2xy),
        "skewness": float(stats.skew(gt)),
        "cv": float(np.std(gt) / np.mean(gt)) if np.mean(gt) != 0 else float("nan"),
        "mean_sd_corr": float(np.corrcoef(means[ok], sds[ok])[0, 1]) if ok.sum() > 2 else float("nan"),
    }
