"""Dataset access, ground truth and stress transformations.

``channels`` defines the single data object every experiment consumes
(:class:`~pfns4neurostim.data.channels.ChannelData`); ``stress`` holds the Hyp B
knobs; ``snr`` computes the achieved SNR used as the K2 x-axis;
``synthetic_neurostim`` is the Demo 1 generator placeholder.
"""
from __future__ import annotations

from . import channels, snr, splits, stress
from .channels import ChannelData, iter_channels, load_channel

__all__ = [
    "channels",
    "snr",
    "splits",
    "stress",
    "ChannelData",
    "load_channel",
    "iter_channels",
]
