"""Split-half ground-truth sensitivity check (task #7 Step 6, P0.7).

Runs the same benchmark twice — once scored against the full-mean GT (primary), once
against the split-half GT (observed trials and GT disjoint) — and pairs the two per
channel, to size the leakage bias between observed trials and the GT mean (audit F8).

Both runs go through :func:`~pfns4neurostim.experiments.bo_benchmark.run_bo_benchmark`
(so they share the cell cache and tidy schema) in ``{run_dir}/full_mean`` and
``{run_dir}/split_half``. The pairing averages repetitions per (channel, model,
acquisition) first, because the two modes use different trial banks and cannot be
paired per repetition.

Deliverables:
    gt_sensitivity.csv   one row per channel x model x acquisition, both modes side by side,
                         with the split-half R^2 noise ceiling (``gt_r_half``)
    gt_identity.svg      identity scatter, full-mean (x) vs split-half (y)

CLI::

    python -m pfns4neurostim gt_sensitivity --config configs/experiment/gt_sensitivity_nhp.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from ..config import load_experiment_config
from ..data.ground_truth import GT_MODES
from .bo_benchmark import run_bo_benchmark

__all__ = ["run_gt_sensitivity", "pair_gt_modes", "main"]

#: Per-channel metrics compared between the two ground-truth modes.
PAIRED_METRICS: tuple[str, ...] = ("recommended_regret", "best_queried_regret", "cumulative_regret", "r2")

_CHANNEL_KEYS: list[str] = ["dataset", "subject", "emg", "model", "acq_label"]


def pair_gt_modes(full: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Pair per-channel means of the two modes.

    Args:
        full: Tidy frame of the full-mean run.
        split: Tidy frame of the split-half run.

    Returns:
        One row per (dataset, subject, emg, model, acq_label) with ``<metric>_full_mean``,
        ``<metric>_split_half`` and ``<metric>_delta`` (split minus full) columns, plus
        ``gt_r_half`` and ``gt_reliability`` of the channel.
    """
    metrics = [m for m in PAIRED_METRICS if m in full.columns and m in split.columns]
    f = full.groupby(_CHANNEL_KEYS, dropna=False)[metrics].mean()
    extra = [c for c in ("gt_r_half", "gt_reliability") if c in split.columns]
    s = split.groupby(_CHANNEL_KEYS, dropna=False)[metrics + extra].mean()
    paired = f.add_suffix("_full_mean").join(
        s[metrics].add_suffix("_split_half"), how="inner"
    ).join(s[extra])
    for m in metrics:
        paired[f"{m}_delta"] = paired[f"{m}_split_half"] - paired[f"{m}_full_mean"]
    return paired.reset_index()


def run_gt_sensitivity(
    config_path: str,
    overrides: list[str] | None = None,
    *,
    run_dir: str | None = None,
    replot: bool = False,
    use_cache: bool = True,
) -> str:
    """Run both GT modes of one benchmark config and pair them.

    Args:
        config_path: Experiment YAML (must carry a ``gt_mode`` key; it is overridden).
        overrides: Extra ``key=value`` overrides applied to both runs.
        run_dir: Output directory; defaults to
            ``{output_root}/gt_sensitivity/{dataset}/{family}-{tag}``.
        replot: Rebuild the pairing and figure from the two existing ``tidy.csv``.
        use_cache: Read/write the cell cache.

    Returns:
        The output directory.
    """
    cfg = load_experiment_config(config_path, overrides)
    target = run_dir or os.path.join(cfg.output_root, "gt_sensitivity", cfg.dataset.name, f"{cfg.family}-{cfg.tag}")
    frames: dict[str, pd.DataFrame] = {}
    for mode in GT_MODES:
        sub_dir = os.path.join(target, mode)
        if not replot:
            run_bo_benchmark(
                config_path, list(overrides or []) + [f"gt_mode={mode}"], run_dir=sub_dir,
                use_cache=use_cache,
            )
        frames[mode] = pd.read_csv(os.path.join(sub_dir, "tidy.csv"))
    paired = pair_gt_modes(frames["full_mean"], frames["split_half"])
    csv_path = os.path.join(target, "gt_sensitivity.csv")
    paired.to_csv(csv_path, index=False)
    print(f"[gt_sensitivity] wrote {csv_path}")
    for m in PAIRED_METRICS:
        if f"{m}_delta" in paired:
            print(
                f"[gt_sensitivity] {m}: mean split-full delta {paired[f'{m}_delta'].mean():+.4f} "
                f"over {len(paired)} channel x model rows"
            )

    from ..visualization.ground_truth import plot_gt_identity  # deferred: matplotlib

    for path in plot_gt_identity(paired, target):
        print(f"[gt_sensitivity] wrote {path}")
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m pfns4neurostim gt_sensitivity``.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="pfns4neurostim gt_sensitivity",
        description="Full-mean vs split-half ground truth on one benchmark config (task #7).",
    )
    parser.add_argument("--config", required=True, help="Path to an experiment YAML.")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None, metavar="KEY=VALUE")
    parser.add_argument("--run-dir", default=None, help="Override the output directory.")
    parser.add_argument("--replot", action="store_true", help="Re-pair from the existing tidy.csv files.")
    parser.add_argument("--no-cache", action="store_true", help="Neither read nor write the cell cache.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_gt_sensitivity(
        args.config, args.overrides, run_dir=args.run_dir, replot=args.replot,
        use_cache=not args.no_cache,
    )
    return 0
