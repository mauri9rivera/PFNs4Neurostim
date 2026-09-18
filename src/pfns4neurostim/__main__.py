"""Package CLI: ``python -m pfns4neurostim <experiment> [options]``.

Experiments are dispatched by name so a SLURM script never imports a module path::

    python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_nhp.yaml
    python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_nhp.yaml --replot

Each runner owns its own argument parser; unknown experiment names list what is
available rather than failing obscurely.
"""
from __future__ import annotations

import sys
from typing import Callable

__all__ = ["main", "EXPERIMENTS"]


def _stress_sweep(argv: list[str]) -> int:
    """Dispatch to the Hyp B stress sweep runner.

    Args:
        argv: Remaining CLI arguments.

    Returns:
        Process exit code.
    """
    from .experiments.stress_sweep import main as runner  # noqa: PLC0415 - lazy

    return runner(argv)


#: Registered experiment runners, keyed by CLI name.
EXPERIMENTS: dict[str, Callable[[list[str]], int]] = {
    "stress_sweep": _stress_sweep,
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch a subcommand.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print(f"Available experiments: {', '.join(sorted(EXPERIMENTS))}")
        return 0
    name, rest = args[0], args[1:]
    if name not in EXPERIMENTS:
        print(
            f"Unknown experiment {name!r}. Available: {', '.join(sorted(EXPERIMENTS))}",
            file=sys.stderr,
        )
        return 2
    return EXPERIMENTS[name](rest)


if __name__ == "__main__":
    raise SystemExit(main())
