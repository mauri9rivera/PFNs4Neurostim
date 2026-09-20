"""Gaussian-process surrogates and the exact-GP models behind them."""
from __future__ import annotations

from .exact_gp import DeepKernelGP, ExactGP
from .surrogates import DeepKernelGPSurrogate, GPSurrogate, NaiveGPSurrogate

__all__ = [
    "ExactGP",
    "DeepKernelGP",
    "GPSurrogate",
    "NaiveGPSurrogate",
    "DeepKernelGPSurrogate",
]
