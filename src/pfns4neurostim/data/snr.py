"""Achieved signal-to-noise ratio of a channel.

The K2 knob is reported against **achieved SNR (dB)**, never against the raw
amplification factor alpha (roadmap S2). SNR is defined per channel as the ratio
of across-site ground-truth response variance (the signal the optimizer must
resolve) to the mean within-site trial variance (the noise it must see through):

    SNR_dB = 10 * log10( Var_s[y_gt(s)] / mean_s[ Var_r[y(s, r)] ] )

Scaling the residuals by alpha multiplies the noise power by alpha^2, so the
achieved SNR shifts by exactly -20*log10(alpha) dB. That identity is the unit
test for the knob.
"""
from __future__ import annotations

import numpy as np

from .channels import ChannelData

__all__ = ["achieved_snr_db", "per_site_snr_db", "noise_power", "signal_power"]


def signal_power(channel: ChannelData) -> float:
    """Across-site variance of the ground-truth response.

    Args:
        channel: The channel.

    Returns:
        Signal power in squared standardized response units.
    """
    return float(np.var(np.asarray(channel.y_gt, dtype=np.float64)))


def noise_power(channel: ChannelData) -> float:
    """Mean within-site trial variance, ignoring invalid (NaN) trials.

    Args:
        channel: The channel.

    Returns:
        Noise power in squared standardized response units.

    Raises:
        RuntimeError: If no site has at least two valid trials, so within-site
            variance is undefined everywhere.
    """
    Y = np.asarray(channel.Y_trials, dtype=np.float64)     # [N, R]
    n_valid = np.sum(~np.isnan(Y), axis=1)                 # [N]
    usable = n_valid >= 2
    if not usable.any():
        raise RuntimeError(
            f"noise_power({channel.label}): no site has >= 2 valid trials, so "
            "within-site variance is undefined. Achieved SNR cannot be computed."
        )
    with np.errstate(invalid="ignore"):
        within = np.nanvar(Y[usable], axis=1, ddof=1)      # [N_usable]
    within = within[np.isfinite(within)]
    if within.size == 0:
        raise RuntimeError(f"noise_power({channel.label}): all within-site variances non-finite.")
    return float(np.mean(within))


def achieved_snr_db(channel: ChannelData) -> float:
    """Achieved SNR of a channel in decibels.

    Args:
        channel: The channel (nominal or stressed).

    Returns:
        ``10 * log10(signal_power / noise_power)``.

    Raises:
        RuntimeError: If either power is non-positive or the result is
            non-finite (silent NaN is never returned; see CLAUDE.md section 4).
    """
    sig = signal_power(channel)
    noise = noise_power(channel)
    if sig <= 0.0 or noise <= 0.0:
        raise RuntimeError(
            f"achieved_snr_db({channel.label}): non-positive power "
            f"(signal={sig:.4g}, noise={noise:.4g})."
        )
    value = 10.0 * float(np.log10(sig / noise))
    if not np.isfinite(value):
        raise RuntimeError(f"achieved_snr_db({channel.label}): non-finite result {value}.")
    return value


def per_site_snr_db(channel: ChannelData) -> np.ndarray:
    """Per-site SNR in dB, for the faint per-channel traces of the S2 figures.

    Uses the shared across-site signal power with each site's own trial
    variance, so the values say where in the map the noise dominates.

    Args:
        channel: The channel.

    Returns:
        Array of shape [N]; sites with fewer than two valid trials are NaN.
    """
    Y = np.asarray(channel.Y_trials, dtype=np.float64)     # [N, R]
    sig = signal_power(channel)
    out = np.full(Y.shape[0], np.nan, dtype=np.float64)    # [N]
    n_valid = np.sum(~np.isnan(Y), axis=1)                 # [N]
    with np.errstate(invalid="ignore"):
        within = np.nanvar(Y, axis=1, ddof=1)              # [N]
    ok = (n_valid >= 2) & np.isfinite(within) & (within > 0)
    out[ok] = 10.0 * np.log10(sig / within[ok])
    return out
