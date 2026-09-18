"""Publication figures for PFNs4Neurostim.

``style`` is the canonical source of figure geometry, typography, model
colours/labels and axis wording (JNE/IOP specification). No other module may
hardcode a colour, label, figure size or axis string.
"""
from __future__ import annotations

from . import style

__all__: list[str] = ["style"]
