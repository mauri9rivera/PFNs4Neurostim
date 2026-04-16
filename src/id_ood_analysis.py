"""
CLI entry point for ID/OOD analysis of neurostim data vs TabPFN's prior.

Usage:
  python src/id_ood_analysis.py --datasets rat nhp --analyses entropy mmd mahalanobis
  python src/id_ood_analysis.py --datasets nhp --prior_source tabpfn_prior --device cuda
  python src/id_ood_analysis.py --datasets rat nhp spinal --prior_source both --save
"""
import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import yaml

from analysis.id_ood import run_id_ood_analysis
from analysis.id_ood_visualization import (
    plot_entropy_distribution, plot_entropy_heatmap,
    plot_mmd_barplot, plot_mahalanobis_distribution,
    plot_summary_dashboard,
)


def _load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML experiment config file.

    Args:
        path: Filesystem path to a ``.yaml`` config file.

    Returns:
        Dict of key-value pairs from the YAML document.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the YAML document is not a mapping.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(cfg).__name__}: {path}")
    return cfg


def generate_id_ood_tag(datasets, analyses, prior_source):
    """Build a human-readable experiment tag with date.

    Tag format: id_ood_{datasets}_{prior_source}_{YYYYMMDD}
    Example:    id_ood_nhp_tabpfn_prior_20260319
    """
    date_str = datetime.now().strftime('%Y%m%d')
    ds_str = '_'.join(sorted(datasets))
    return f'id_ood_{ds_str}_{prior_source}_{date_str}'


def write_id_ood_config(output_dir, args):
    """Serialize all experiment parameters to config.json."""
    config = {
        'experiment': 'id_ood_analysis',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'datasets': args.datasets,
        'analyses': args.analyses,
        'prior_source': args.prior_source,
        'device': args.device,
        'n_synthetic': args.n_synthetic,
        'n_context': args.n_context,
        'seed': args.seed,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description='ID/OOD analysis: is neurostim data within TabPFN\'s prior?',
    )
    parser.add_argument('--config', type=str, default=None, metavar='PATH',
                        help='Path to a YAML config file.  All keys are used as defaults; '
                             'any CLI flag that is explicitly provided overrides the YAML value.')
    parser.add_argument('--datasets', nargs='+', default=None,
                        choices=['rat', 'nhp', 'spinal'],
                        help='Dataset types to analyze (default: rat nhp)')
    parser.add_argument('--analyses', nargs='+',
                        default=None,
                        choices=['entropy', 'mmd', 'mahalanobis'],
                        help='Which analyses to run (default: entropy mmd mahalanobis)')
    parser.add_argument('--prior_source', default=None,
                        choices=['gp', 'tabpfn_prior', 'noise', 'both', 'all'],
                        help='Synthetic reference distribution(s) (default: both)')
    parser.add_argument('--device', default=None,
                        help='Device for TabPFN inference (default: cpu)')
    parser.add_argument('--n_synthetic', type=int, default=None,
                        help='Number of synthetic datasets for reference (default: 500)')
    parser.add_argument('--n_context', type=int, default=None,
                        help='Context size for entropy and Mahalanobis (default: 50)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed.  If not provided, a fresh seed is '
                             'drawn from system entropy and logged to '
                             'config.json so priors differ across runs but '
                             'remain reproducible post-hoc.')
    parser.add_argument('--save', action='store_true',
                        help='Save results and plots to disk')

    args = parser.parse_args()

    # --- YAML config loading: load first, then let explicit CLI args override ---
    if args.config is not None:
        yaml_cfg = _load_yaml_config(args.config)
        _bool_flags = {'save'}
        for key, value in yaml_cfg.items():
            if key in _bool_flags:
                if not getattr(args, key, False):
                    setattr(args, key, value)
            elif getattr(args, key, None) is None:
                setattr(args, key, value)
        print(f"[config] Loaded YAML defaults from {args.config}")

    # Apply built-in defaults for any remaining None values
    _defaults: dict[str, Any] = {
        'datasets': ['rat', 'nhp'],
        'analyses': ['entropy', 'mmd', 'mahalanobis'],
        'prior_source': 'both',
        'device': 'cpu',
        'n_synthetic': 500,
        'n_context': 50,
    }
    for key, default in _defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default)

    # Seed: when unspecified, draw a fresh int so priors differ across runs.
    # The drawn value is logged to config.json for reproducibility.
    if getattr(args, 'seed', None) is None:
        args.seed = int(np.random.default_rng().integers(0, 2**31))
        print(f"[seed] No --seed provided — drew fresh seed: {args.seed}")
    else:
        print(f"[seed] Using seed: {args.seed}")

    # Build dated experiment tag and output directory
    exp_tag = generate_id_ood_tag(args.datasets, args.analyses, args.prior_source)
    output_dir = os.path.join('output', 'id_ood', exp_tag)

    # Write config.json with all experiment parameters
    if args.save:
        write_id_ood_config(output_dir, args)

    # Run analyses
    all_results = run_id_ood_analysis(
        dataset_types=args.datasets,
        analyses=args.analyses,
        prior_source=args.prior_source,
        device=args.device,
        n_synthetic=args.n_synthetic,
        n_context=args.n_context,
        seed=args.seed,
        save=args.save,
        output_dir=output_dir,
    )

    # Generate visualizations
    if 'entropy' in all_results:
        entropy_results = all_results['entropy']
        plot_entropy_distribution(entropy_results, save=args.save,
                                 output_dir=output_dir)
        for dt in args.datasets:
            plot_entropy_heatmap(entropy_results, dt, save=args.save,
                                output_dir=output_dir)

    if 'mmd' in all_results:
        plot_mmd_barplot(all_results['mmd'], save=args.save,
                         output_dir=output_dir)

    if 'mahalanobis' in all_results:
        plot_mahalanobis_distribution(all_results['mahalanobis'],
                                      save=args.save, output_dir=output_dir)

    # Summary dashboard (only if all three analyses ran)
    if all(k in all_results for k in ['entropy', 'mmd', 'mahalanobis']):
        plot_summary_dashboard(
            all_results['entropy'],
            all_results['mmd'],
            all_results['mahalanobis'],
            save=args.save, output_dir=output_dir,
        )

    print(f"\n[ID/OOD] Analysis complete. Tag: {exp_tag}")


if __name__ == '__main__':
    main()
