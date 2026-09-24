"""Package CLI: ``python -m pfns4neurostim <experiment> [options]``.

Experiments are dispatched by name so a SLURM script never imports a module path::

    python -m pfns4neurostim bo_benchmark --config configs/experiment/hyp_a_nhp.yaml
    python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_channel_nhp.yaml
    python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_channel_nhp.yaml --replot

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


def _bo_benchmark(argv: list[str]) -> int:
    """Dispatch to the models x acquisitions benchmark runner (Hyp 0 / Hyp A).

    Args:
        argv: Remaining CLI arguments.

    Returns:
        Process exit code.
    """
    from .experiments.bo_benchmark import main as runner  # noqa: PLC0415 - lazy

    return runner(argv)


def _gt_sensitivity(argv: list[str]) -> int:
    """Dispatch to the full-mean vs split-half ground-truth check (task #7).

    Args:
        argv: Remaining CLI arguments.

    Returns:
        Process exit code.
    """
    from .experiments.gt_sensitivity import main as runner  # noqa: PLC0415 - lazy

    return runner(argv)


def _mechanism(argv: list[str]) -> int:
    """Dispatch to the Hyp C mechanism analyses (update rule, placement, CKA).

    Args:
        argv: Remaining CLI arguments.

    Returns:
        Process exit code.
    """
    from .experiments.mechanism import main as runner  # noqa: PLC0415 - lazy

    return runner(argv)


#: Registered experiment runners, keyed by CLI name.
EXPERIMENTS: dict[str, Callable[[list[str]], int]] = {
    "bo_benchmark": _bo_benchmark,
    "stress_sweep": _stress_sweep,
    "gt_sensitivity": _gt_sensitivity,
    "mechanism": _mechanism,
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
