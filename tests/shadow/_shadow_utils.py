"""Measurement reporting shared by shadow tests."""
from __future__ import annotations

import numpy as np

MEASUREMENTS: list[str] = []


def report(name: str, **values: object) -> None:
    """Record and print a measured-value line (shown in the pytest terminal summary).

    Args:
        name: Test / quantity name.
        **values: Measured values.
    """
    body = " ".join(
        f"{k}={float(v):.4g}" if isinstance(v, (float, np.floating)) else f"{k}={v}"
        for k, v in values.items()
    )
    line = f"[MEASURE] {name}: {body}"
    MEASUREMENTS.append(line)
    print(line)
