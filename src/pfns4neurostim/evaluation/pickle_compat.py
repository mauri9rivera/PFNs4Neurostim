"""Cross-environment pickle loading for cached cells and trajectories.

The project runs in two environments (Python 3.9 with NumPy 1.26 for the main runs, Python 3.11 with
NumPy 2 for the bench models), and cells written under NumPy 2 reference ``numpy._core``, a module that
NumPy 1.26 lacks. Without this loader a laptop in the main env cannot read the bench env's cells or
``trajectories.pkl``, which would make ``--replot`` and ``--only-cached`` fail on exactly the D3 results.
"""
from __future__ import annotations

import pickle
from typing import IO, Any

__all__ = ["load"]

_NUMPY2_PREFIX = "numpy._core"
_NUMPY1_PREFIX = "numpy.core"


class _CompatUnpickler(pickle.Unpickler):
    """Unpickler that resolves NumPy-2 module paths on NumPy 1.x."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith(_NUMPY2_PREFIX):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                module = _NUMPY1_PREFIX + module[len(_NUMPY2_PREFIX):]
        return super().find_class(module, name)


def load(fh: IO[bytes]) -> Any:
    """Load a pickle written under either NumPy major version.

    Args:
        fh: Binary file object.

    Returns:
        The unpickled object.
    """
    return _CompatUnpickler(fh).load()
