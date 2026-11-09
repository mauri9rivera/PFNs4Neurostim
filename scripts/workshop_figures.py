"""Render the AAAI-27 NeuroAI workshop items (table + three figures) from the run directories in a workshop config.

    python scripts/workshop_figures.py --config configs/workshop/aaai27_neuroai.yaml
    python scripts/workshop_figures.py --config configs/workshop/aaai27_neuroai.yaml --only stress gp_like
"""
from __future__ import annotations

import argparse

from pfns4neurostim.visualization.workshop import FIGURE_NAMES, render_all


def main() -> None:
    """Parse the CLI and render."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="Workshop YAML (configs/workshop/*.yaml).")
    ap.add_argument("--only", nargs="+", choices=list(FIGURE_NAMES), default=None, help="Items to render.")
    args = ap.parse_args()
    for path in render_all(args.config, args.only):
        print(path)


if __name__ == "__main__":
    main()
