"""Post-hoc aggregation of per-run experiment results.

Driven by a canonical YAML config from ``configs/``, this script locates all
matching run directories under ``output/runs/``, merges their pkl files into
family-level DataFrames, and produces three aggregation views (per-EMG,
per-subject, family-level) saved as CSV.  Aggregate plots are also written for
``fit`` and ``optimization`` result types.

Usage::

    python src/aggregate.py --config configs/nhp_vanilla_benchmark.yaml
    python src/aggregate.py --config configs/nhp_optimization.yaml \\
        --result_types fit optimization
    python src/aggregate.py --config configs/nhp_vanilla_benchmark.yaml \\
        --runs_dir output/runs --output_dir output/aggregated

SLURM Family 8 invokes this script automatically.
"""
import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# --- Ensure src/ is on sys.path when invoked from project root ---
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.data_utils import aggregate_results, load_results
from utils.visualization import (
    r2_per_muscle,
    r2_by_subject,
    regret_with_timing,
    regret_curve,
)


# Result types that can be loaded as dict[str, list[dict]] pkl files
_DICT_RESULT_TYPES = frozenset({'fit', 'optimization'})
# Result types that correspond to DataFrame pkl files
_DF_RESULT_TYPES = frozenset({'optimization_budget'})
# Result types that come from mode names in YAML
_MODE_TO_RESULT_TYPE: Dict[str, str] = {
    'fit': 'fit',
    'optimization': 'optimization',
    'optimization_budget': 'optimization_budget',
}


# ============================================
#           YAML helpers
# ============================================


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a canonical YAML config file.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Dict of key-value pairs.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML is not a mapping.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config must be a YAML mapping, got {type(cfg).__name__}: {path}"
        )
    return cfg


def _infer_result_types(cfg: Dict[str, Any]) -> List[str]:
    """Infer aggregatable result types from the YAML ``mode`` key.

    Args:
        cfg: Parsed YAML config dict.

    Returns:
        List of result_type strings compatible with ``aggregate_results()``.
        Returns empty list if no ``mode`` key exists (e.g. id_ood configs).
    """
    raw_mode = cfg.get('mode', [])
    if isinstance(raw_mode, str):
        raw_mode = [raw_mode]

    result_types: List[str] = []
    for m in raw_mode:
        rt = _MODE_TO_RESULT_TYPE.get(m)
        if rt is not None:
            result_types.append(rt)
    return result_types


def _extract_datasets(cfg: Dict[str, Any]) -> List[str]:
    """Extract dataset(s) from the YAML config.

    Handles both single-dataset configs (``dataset: nhp``) and
    multi-dataset configs (``datasets: [rat, nhp]``).

    Args:
        cfg: Parsed YAML config dict.

    Returns:
        List of dataset strings.

    Raises:
        ValueError: If neither ``dataset`` nor ``datasets`` key is present.
    """
    if 'dataset' in cfg:
        return [cfg['dataset']]
    if 'datasets' in cfg:
        return list(cfg['datasets'])
    raise ValueError("YAML config must contain a 'dataset' or 'datasets' key.")


# ============================================
#           Aggregation views
# ============================================


def _compute_views(
    df: pd.DataFrame,
    result_type: str,
) -> Dict[str, pd.DataFrame]:
    """Compute the three standard aggregation views from a flat DataFrame.

    Args:
        df: Flat DataFrame produced by ``aggregate_results()``.
        result_type: ``'fit'`` or ``'optimization'`` (determines which
            metric columns are present).

    Returns:
        Dict with keys ``'per_emg'``, ``'per_subject'``, ``'family'``;
        each value is a grouped summary DataFrame.
    """
    agg_cols = ['mean_r2', 'std_r2']
    if result_type == 'optimization' and 'mean_final_regret' in df.columns:
        agg_cols += ['mean_final_regret', 'std_final_regret']

    # Per-EMG: (model, dataset, subject, emg) → mean/std across runs
    per_emg = (
        df.groupby(['model', 'dataset', 'subject', 'emg'])[agg_cols]
        .mean()
        .reset_index()
    )
    per_emg.columns = ['model', 'dataset', 'subject', 'emg'] + agg_cols

    # Per-subject: group per-EMG rows by (model, dataset, subject)
    per_subject = (
        per_emg.groupby(['model', 'dataset', 'subject'])[agg_cols]
        .mean()
        .reset_index()
    )

    # Family-level: group per-subject rows by model
    family = (
        per_subject.groupby(['model'])[agg_cols]
        .agg(['mean', 'std'])
        .reset_index()
    )
    # Flatten multi-level column names
    family.columns = [
        '_'.join(c).strip('_') if c[1] else c[0]
        for c in family.columns
    ]

    return {'per_emg': per_emg, 'per_subject': per_subject, 'family': family}


def _compute_views_budget(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute aggregation views for ``optimization_budget`` DataFrames.

    Args:
        df: Flat DataFrame with columns Budget|Model|Regret|R2|ID|tag|family.

    Returns:
        Dict with keys ``'per_emg'``, ``'per_subject'``, ``'family'``.
        ``'per_emg'`` here means per (Budget, Model, ID) grain.
    """
    per_emg = (
        df.groupby(['Budget', 'Model', 'ID'])[['Regret', 'R2']]
        .mean()
        .reset_index()
    )

    # Parse ID (format: "{subject}_{emg}") into subject/emg columns
    per_emg[['subject', 'emg']] = per_emg['ID'].str.split('_', expand=True)

    per_subject = (
        per_emg.groupby(['Budget', 'Model', 'subject'])[['Regret', 'R2']]
        .mean()
        .reset_index()
    )

    family = (
        per_subject.groupby(['Budget', 'Model'])[['Regret', 'R2']]
        .agg(['mean', 'std'])
        .reset_index()
    )
    family.columns = [
        '_'.join(c).strip('_') if c[1] else c[0]
        for c in family.columns
    ]

    return {'per_emg': per_emg, 'per_subject': per_subject, 'family': family}


# ============================================
#           Plot helpers
# ============================================


def _load_combined_results_dict(
    family: str,
    dataset: str,
    result_type: str,
    runs_dir: str,
    tags: Optional[List[str]] = None,
) -> Optional[Dict[str, list]]:
    """Reconstruct a combined ``results_dict`` from all matching run pkl files.

    Loads every matching pkl file and merges the per-model result lists into
    one combined dict, suitable for the existing visualization functions.

    Args:
        family: Experiment family string.
        dataset: Dataset type.
        result_type: ``'fit'`` or ``'optimization'``.
        runs_dir: Root directory for runs.
        tags: Optional list of 5-char hash suffixes to restrict loading.
            ``None`` loads all runs matching the family prefix.

    Returns:
        Combined ``{'ModelName': [list of result dicts], ...}`` or ``None``
        if no pkl files matched.
    """
    prefix = f"{dataset}-{family}-"
    tag_set = set(tags) if tags is not None else None
    if not os.path.isdir(runs_dir):
        return None

    combined: Dict[str, list] = {}
    # Pkl files from save_results() have the pattern:
    #   {dataset}_{evaluation_type}_{tag}_{timestamp}.pkl
    # Match by checking for f'_{result_type}_' in the filename.
    marker = f'_{result_type}_'

    for name in sorted(os.listdir(runs_dir)):
        if not name.startswith(prefix):
            continue
        if tag_set is not None and name[len(prefix):] not in tag_set:
            continue
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        results_subdir = os.path.join(run_dir, 'results')
        if not os.path.isdir(results_subdir):
            continue

        for fname in sorted(os.listdir(results_subdir)):
            if 'budget' in fname:
                continue
            if marker not in fname:
                continue
            pkl_path = os.path.join(results_subdir, fname)
            try:
                data = load_results(pkl_path)
            except Exception:
                continue
            for model_name, result_list in data.items():
                if model_name == '_metadata':
                    continue
                combined.setdefault(model_name, []).extend(result_list)

    return combined if combined else None


# ============================================
#           Main aggregation routine
# ============================================


def run_aggregation(
    config_path: str,
    result_types: Optional[List[str]] = None,
    runs_dir: str = './output/runs',
    output_dir: str = './output/aggregated',
    tags: Optional[List[str]] = None,
) -> None:
    """Run full aggregation pipeline for a canonical YAML config.

    Args:
        config_path: Path to a canonical YAML file from ``configs/``.
        result_types: Result types to aggregate.  Inferred from YAML ``mode``
            key when ``None``.
        runs_dir: Root directory containing per-run subdirectories.
        output_dir: Root directory for aggregated output.
    """
    cfg = _load_yaml_config(config_path)

    family: str = cfg.get('family', '')
    if not family:
        raise ValueError(
            f"YAML config at {config_path!r} has no 'family' key.  "
            "Run Step 4a to add it."
        )

    datasets = _extract_datasets(cfg)

    if result_types is None:
        result_types = _infer_result_types(cfg)

    if not result_types:
        print(
            f"[aggregate] No aggregatable result types found in config "
            f"(family={family!r}).  Supported modes: "
            f"{sorted(_MODE_TO_RESULT_TYPE.keys())}. Exiting."
        )
        return

    total_runs = 0
    total_experiments = 0

    for dataset in datasets:
        out_subdir = os.path.join(output_dir, f"{dataset}-{family}")
        os.makedirs(out_subdir, exist_ok=True)

        for result_type in result_types:
            print(
                f"\n[aggregate] {dataset}/{family}/{result_type} "
                f"(runs_dir={runs_dir})"
            )

            df = aggregate_results(family, dataset, result_type, runs_dir, tags=tags)

            if df.empty:
                print(
                    f"  [WARNING] No matching runs found for "
                    f"{dataset}-{family}-* / {result_type}.  Skipping."
                )
                continue

            n_runs = df['tag'].nunique() if 'tag' in df.columns else 0
            n_exp = (
                df.groupby(['model', 'subject', 'emg']).ngroups
                if result_type in _DICT_RESULT_TYPES
                else len(df)
            )
            print(f"  Loaded {n_runs} run(s), {n_exp} experiment rows.")
            total_runs += n_runs
            total_experiments += n_exp

            # --- Compute and save views ---
            if result_type in _DICT_RESULT_TYPES:
                views = _compute_views(df, result_type)
            else:
                views = _compute_views_budget(df)

            for view_name, view_df in views.items():
                csv_path = os.path.join(
                    out_subdir, f"{result_type}_{view_name}.csv"
                )
                view_df.to_csv(csv_path, index=False)
                print(f"  Saved {view_name} -> {csv_path} ({len(view_df)} rows)")

            # --- Generate plots ---
            if result_type in _DICT_RESULT_TYPES:
                combined = _load_combined_results_dict(
                    family, dataset, result_type, runs_dir, tags=tags
                )
                if combined is not None:
                    tag_label = f"{dataset}_{family}_aggregated"
                    try:
                        if result_type == 'fit':
                            r2_per_muscle(
                                combined, mode=f'_{tag_label}',
                                save=True, output_dir=out_subdir,
                            )
                            r2_by_subject(
                                combined, split_type=tag_label,
                                save=True, output_dir=out_subdir,
                            )
                        elif result_type == 'optimization':
                            regret_with_timing(
                                combined, split_type=tag_label,
                                save=True, output_dir=out_subdir,
                            )
                            regret_curve(
                                combined, split_type=tag_label,
                                save=True, output_dir=out_subdir,
                            )
                            r2_by_subject(
                                combined, split_type=tag_label,
                                save=True, output_dir=out_subdir,
                                output_subdir='optimization',
                            )
                    except Exception as exc:
                        print(f"  [WARNING] Plot generation failed: {exc}")

    print(
        f"\n[aggregate] Done. "
        f"Aggregated {total_runs} run(s), {total_experiments} experiment rows "
        f"-> {output_dir}/"
    )


# ============================================
#           CLI Entry Point
# ============================================


def main() -> None:
    """CLI entry point for post-hoc result aggregation.

    Reads ``--config`` (canonical YAML), derives family+dataset, calls
    ``aggregate_results()`` for each requested result type, produces 3 CSV
    views and aggregate plots, and saves them under
    ``output/aggregated/{dataset}-{family}/``.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Aggregate per-run experiment results into family-level summaries.\n'
            'Driven by a canonical YAML config (e.g. configs/nhp_vanilla_benchmark.yaml).'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--config', type=str, required=True, metavar='PATH',
        help='Path to a canonical YAML config in configs/.\n'
             'Provides family and dataset; mode keys determine result_types.',
    )
    parser.add_argument(
        '--result_types', type=str, nargs='+', default=None,
        metavar='TYPE',
        help='Result types to aggregate: fit, optimization, optimization_budget.\n'
             'Inferred from the YAML mode key when omitted.',
    )
    parser.add_argument(
        '--runs_dir', type=str, default='./output/runs',
        help='Root directory containing per-run subdirectories (default: ./output/runs)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='./output/aggregated',
        help='Root directory for aggregated output (default: ./output/aggregated)',
    )
    parser.add_argument(
        '--tags', type=str, nargs='+', default=None,
        metavar='TAG',
        help='Restrict aggregation to runs with these 5-char hash suffixes.\n'
             'Example: --tags 32c2b 15h5p\n'
             'Default: aggregate all runs matching the family prefix.',
    )

    args = parser.parse_args()

    run_aggregation(
        config_path=args.config,
        result_types=args.result_types,
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        tags=args.tags,
    )


if __name__ == '__main__':
    main()
