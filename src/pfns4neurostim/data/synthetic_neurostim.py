"""Synthetic neurostimulation map generator (Demo 1) — PLACEHOLDER.

Roadmap S0. Not implemented yet; scheduled for task #10 Step 9. The signatures
are fixed now so that the stress-sweep runner, the K1 knob and the S10
synthetic-to-real bridge can be written against them.

Design (roadmap S0): baseline plus K anisotropic Gaussian hotspots on the real
electrode-grid geometries (10x10, 8x4, 8x8), sigmoidal recruitment/saturation,
and heteroscedastic multiplicative noise (SD proportional to the mean). The
nominal anchor is fitted per real channel and must reproduce the measured
meta-features: Moran's I around 0.31, skewness 2.4, CV 0.86, mean-SD
correlation 0.92.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .channels import ChannelData

__all__ = ["GeneratorParams", "generate_neurostim_map", "fit_generator_to_channel", "meta_features"]


@dataclass(frozen=True)
class GeneratorParams:
    """Parameters of one synthetic neurostimulation map.

    Attributes:
        grid_shape: Electrode grid shape, e.g. ``(10, 10)``.
        n_hotspots: Number of Gaussian hotspots K.
        amplitudes: Hotspot peak amplitudes, length K.
        centers: Hotspot centres in grid coordinates, shape [K, D].
        lengthscales: Per-hotspot, per-axis widths, shape [K, D].
        rotations: Hotspot rotation angles in radians, length K.
        baseline: Additive baseline response.
        saturation: Sigmoid saturation ceiling of the recruitment curve.
        noise_cv: Multiplicative noise coefficient of variation (SD/mean).
        n_trials: Trials per site.
    """

    grid_shape: tuple[int, ...]
    n_hotspots: int = 1
    amplitudes: tuple[float, ...] = (1.0,)
    centers: np.ndarray | None = None
    lengthscales: np.ndarray | None = None
    rotations: tuple[float, ...] = (0.0,)
    baseline: float = 0.0
    saturation: float = 1.0
    noise_cv: float = 0.86
    n_trials: int = 15


def generate_neurostim_map(
    params: GeneratorParams,
    rng: np.random.Generator,
    *,
    dataset: str = "synthetic_neurostim",
    subject: int = 0,
    emg: int = 0,
) -> ChannelData:
    """Generate one synthetic channel (Demo 1). **Not implemented.**

    Args:
        params: Generator parameters.
        rng: Seeded generator.
        dataset: Dataset label for the tidy schema.
        subject: Pseudo-subject index.
        emg: Pseudo-EMG index.

    Returns:
        A :class:`ChannelData` with ``demo='demo1'``.

    Raises:
        NotImplementedError: Always; see task #10 Step 9.
    """
    raise NotImplementedError(
        "Demo 1 synthetic generator lands with task #10 Step 9 (roadmap S0)."
    )


def fit_generator_to_channel(channel: ChannelData) -> GeneratorParams:
    """Fit generator parameters to a real channel. **Not implemented.**

    Args:
        channel: Real in-vivo channel to imitate.

    Returns:
        Fitted parameters whose maps match the channel's meta-features.

    Raises:
        NotImplementedError: Always; see task #10 Step 9.
    """
    raise NotImplementedError(
        "fit_generator_to_channel lands with task #10 Step 9 (roadmap S0/S10)."
    )


def meta_features(channel: ChannelData) -> dict[str, float]:
    """Compute the map meta-features used to validate the generator. **Not implemented.**

    Args:
        channel: Channel to describe.

    Returns:
        Mapping with keys ``morans_i``, ``skewness``, ``cv``, ``mean_sd_corr``.

    Raises:
        NotImplementedError: Always; see task #10 Step 9.
    """
    raise NotImplementedError("Generator meta-features land with task #10 Step 9 (roadmap S0).")
