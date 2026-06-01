"""Re-generate optimization plots for existing run directories.

Loads the most recent pkl per run directory (by filename sort order) and
calls all updated visualization functions. Handles optimization and
vanilla-benchmark run types. Skips aug-sweep directories (DataFrame format,
visualized separately via augmentation_sweep_plot).

Usage:
    python scripts/replot_runs.py
    python scripts/replot_runs.py --runs_dir output/runs
    python scripts/replot_runs.py --runs_dir output/runs --run nhp-vanilla-benchmark-3a5d1
"""
import argparse
import glob
import os
import pickle
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.visualization import (
    r2_by_subject,
    regret_with_timing,
    regret_curve,
    regret_by_subject,
    regret_by_emg,
    exploration_by_subject,
    exploration_by_emg,
    visualize_representation,
)


def load_latest_pkl(run_dir: str) -> Optional[dict]:
    """Load the most recent timestamped pkl file from <run_dir>/results/."""
    pkls = sorted(glob.glob(os.path.join(run_dir, 'results', '*.pkl')))
    if not pkls:
        return None
    path = pkls[-1]
    print(f'  Loading {os.path.basename(path)}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def replot(run_dir: str) -> None:
    """Regenerate all optimization plots for a single run directory."""
    data = load_latest_pkl(run_dir)
    if data is None:
        print(f'[skip] No pkl files found in {run_dir}/results/')
        return

    results_dict = {k: v for k, v in data.items() if not k.startswith('_')}
    if not results_dict:
        print(f'[skip] No model keys in pkl (only metadata): {run_dir}')
        return

    models = list(results_dict.keys())
    n_experiments = sum(len(v) for v in results_dict.values())
    print(f'  Models: {models} | Experiments: {n_experiments}')

    kw = dict(save=True, output_dir=run_dir)
    errors = []

    for fn_name, fn in [
        ('regret_with_timing', regret_with_timing),
        ('regret_curve', regret_curve),
        ('regret_by_subject', regret_by_subject),
        ('regret_by_emg', regret_by_emg),
        ('exploration_by_subject', exploration_by_subject),
        ('exploration_by_emg', exploration_by_emg),
        ('r2_by_subject', r2_by_subject),
        ('visualize_representation', visualize_representation),
    ]:
        try:
            fn(results_dict, **kw)
        except Exception as exc:
            errors.append(f'{fn_name}: {exc}')
            print(f'  [WARNING] {fn_name} failed: {exc}')

    out_dir = os.path.join(run_dir, 'optimization')
    if errors:
        print(f'  Done with {len(errors)} warning(s) -> {out_dir}')
    else:
        print(f'  Done -> {out_dir}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Regenerate optimization plots for existing run directories.'
    )
    parser.add_argument(
        '--runs_dir', default='output/runs',
        help='Root directory containing run subdirectories (default: output/runs)',
    )
    parser.add_argument(
        '--run', default=None,
        help='Replot a single named run directory only (e.g. nhp-vanilla-benchmark-3a5d1)',
    )
    args = parser.parse_args()

    if args.run:
        targets = [os.path.join(args.runs_dir, args.run)]
    else:
        targets = sorted(
            os.path.join(args.runs_dir, d)
            for d in os.listdir(args.runs_dir)
            if os.path.isdir(os.path.join(args.runs_dir, d))
        )

    for full_path in targets:
        name = os.path.basename(full_path)
        if 'aug-sweep' in name:
            print(f'[skip] Aug-sweep run (DataFrame format): {name}')
            continue
        print(f'\nReplotting: {name}')
        replot(full_path)

    print('\nAll done.')


if __name__ == '__main__':
    main()
