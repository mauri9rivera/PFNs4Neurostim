"""Aggregate and plot LoRA ablation results from all nhp-lora-ablation-* runs.

Loads the three per-subject pkl files, merges them into a single results_dict,
and produces:
  1. R² barplot by subject + Average  (r2_barplot_by_subject)
  2. Regret curve traces by subject + Average  (regret_traces_by_subject)

Saved to output/aggregated/nhp-lora-ablation/optimization/.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.visualization import r2_barplot_by_subject, regret_traces_by_subject


def load_lora_runs(
    runs_dir: str,
    family: str = 'lora-ablation',
    dataset: str = 'nhp',
) -> dict[str, list[Any]]:
    """Load and merge all LoRA ablation result pkl files for a given dataset.

    Args:
        runs_dir: Path to the output/runs/ directory.
        family: Experiment family tag used in directory names.
        dataset: Dataset prefix to match (e.g. 'nhp').

    Returns:
        dict with keys 'GP' and 'LoRA TabPFN', each a list of result dicts
        aggregated across all matching run directories.

    Raises:
        FileNotFoundError: If no pkl files match the pattern.
    """
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, f'{dataset}-{family}-*')))
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found matching: {os.path.join(runs_dir, f'{dataset}-{family}-*')}"
        )

    # For each run directory pick only the most-recently-timestamped pkl file.
    pkl_files: list[str] = []
    for run_dir in run_dirs:
        candidates = sorted(
            glob.glob(os.path.join(run_dir, 'results', '*.pkl'))
        )
        if candidates:
            pkl_files.append(candidates[-1])  # lexicographic sort == chronological for YYYYMMDD_HHMMSS

    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found in matching run directories.")

    all_gp: list[Any] = []
    all_lora: list[Any] = []
    for path in pkl_files:
        with open(path, 'rb') as fh:
            data = pickle.load(fh)
        meta = data.get('_metadata', {})
        print(
            f"  Loaded: {os.path.basename(os.path.dirname(os.path.dirname(path)))} "
            f"(subj={meta.get('held_out_subj', '?')}, "
            f"GP={len(data.get('GP', []))}, "
            f"LoRA={len(data.get('TabPFN', []))} EMGs)"
        )
        all_gp.extend(data.get('GP', []))
        all_lora.extend(data.get('TabPFN', []))

    return {'GP': all_gp, 'LoRA TabPFN': all_lora}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot aggregated LoRA ablation results (R2 barplot + regret traces)"
    )
    parser.add_argument(
        '--runs_dir', default='output/runs',
        help='Path to the runs directory (default: output/runs)',
    )
    parser.add_argument(
        '--out_dir', default='output/aggregated/nhp-lora-ablation',
        help='Output directory for aggregated figures',
    )
    parser.add_argument(
        '--dataset', default='nhp',
        help='Dataset prefix to match run directories (default: nhp)',
    )
    args = parser.parse_args()

    print(f"Scanning {args.runs_dir} for nhp-lora-ablation-* runs...")
    results_dict = load_lora_runs(args.runs_dir, dataset=args.dataset)
    n_gp = len(results_dict['GP'])
    n_lora = len(results_dict['LoRA TabPFN'])
    print(f"Total: {n_gp} GP entries, {n_lora} LoRA TabPFN entries\n")

    os.makedirs(args.out_dir, exist_ok=True)
    split_tag = 'inter_subject_lora'

    print("Plotting R² barplot by subject...")
    r2_barplot_by_subject(
        results_dict,
        split_type=split_tag,
        save=True,
        output_dir=args.out_dir,
    )

    print("Plotting regret curve traces by subject...")
    regret_traces_by_subject(
        results_dict,
        split_type=split_tag,
        save=True,
        output_dir=args.out_dir,
    )

    print("\nDone. Figures saved to:", os.path.join(args.out_dir, 'optimization'))


if __name__ == '__main__':
    main()
