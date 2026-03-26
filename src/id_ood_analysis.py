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

from analysis.id_ood import run_id_ood_analysis
from analysis.id_ood_visualization import (
    plot_entropy_distribution, plot_entropy_heatmap,
    plot_mmd_barplot, plot_mahalanobis_distribution,
    plot_summary_dashboard,
)


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
    parser.add_argument('--datasets', nargs='+', default=['rat', 'nhp'],
                        choices=['rat', 'nhp', 'spinal'],
                        help='Dataset types to analyze')
    parser.add_argument('--analyses', nargs='+',
                        default=['entropy', 'mmd', 'mahalanobis'],
                        choices=['entropy', 'mmd', 'mahalanobis'],
                        help='Which analyses to run')
    parser.add_argument('--prior_source', default='both',
                        choices=['gp', 'tabpfn_prior', 'both'],
                        help='Synthetic reference distribution(s)')
    parser.add_argument('--device', default='cpu',
                        help='Device for TabPFN inference')
    parser.add_argument('--n_synthetic', type=int, default=500,
                        help='Number of synthetic datasets for reference')
    parser.add_argument('--n_context', type=int, default=50,
                        help='Context size for entropy and Mahalanobis')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save', action='store_true',
                        help='Save results and plots to disk')

    args = parser.parse_args()

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
