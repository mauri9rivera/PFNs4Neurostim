"""
Dedicated plots for ID/OOD analysis results.

Follows conventions from src/utils/visualization.py:
matplotlib/seaborn, SVG output, save flag, output_dir.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PALETTE = {
    'Neurostim': 'royalblue',
    'Synthetic GP': 'sandybrown',
    'TabPFN Prior': 'seagreen',
    'Noise (OOD)': 'firebrick',
}


def _save_dir(output_dir, subdir):
    """Return output directory, creating if needed."""
    base = os.path.join(output_dir, subdir) if output_dir else \
           os.path.join('output', 'id_ood', subdir)
    os.makedirs(base, exist_ok=True)
    return base


def plot_entropy_distribution(entropy_results, save=False, output_dir=None):
    """Violin/box plot: entropy distributions per dataset type.

    One panel per dataset. X-axis: subjects. Overlaid reference bands
    from GP and/or Prior Bag synthetic data.

    Args:
        entropy_results: dict from entropy_analysis()
        save: whether to save figure
        output_dir: base output directory
    """
    _ref_keys = {'synthetic_gp', 'synthetic_prior', 'noise'}
    dataset_types = [k for k in entropy_results if k not in _ref_keys]
    n_panels = max(1, len(dataset_types))

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)

    for ax_idx, dataset_type in enumerate(dataset_types):
        ax = axes[0, ax_idx]
        ds_data = entropy_results[dataset_type]

        # Collect per-subject entropy values
        plot_data = []
        for subj_idx, subj_data in sorted(ds_data.items()):
            for emg_idx, entropy_arr in sorted(subj_data.items()):
                for val in entropy_arr:
                    plot_data.append({
                        'Subject': f'S{subj_idx}',
                        'Entropy': float(val),
                    })

        if not plot_data:
            continue

        import pandas as pd
        df = pd.DataFrame(plot_data)

        sns.violinplot(data=df, x='Subject', y='Entropy', ax=ax,
                       color=PALETTE['Neurostim'], alpha=0.7, inner='box')

        # Reference bands
        y_bottom, y_top = ax.get_ylim()
        if 'synthetic_gp' in entropy_results:
            gp_ent = entropy_results['synthetic_gp']
            if len(gp_ent) > 0:
                gp_mean = np.mean(gp_ent)
                gp_std = np.std(gp_ent)
                ax.axhspan(gp_mean - gp_std, gp_mean + gp_std,
                           color=PALETTE['Synthetic GP'], alpha=0.15,
                           label=f'GP ref (mean={gp_mean:.2f})')
                ax.axhline(gp_mean, color=PALETTE['Synthetic GP'],
                           linestyle='--', linewidth=1)

        if 'synthetic_prior' in entropy_results:
            pr_ent = entropy_results['synthetic_prior']
            if len(pr_ent) > 0:
                pr_mean = np.mean(pr_ent)
                pr_std = np.std(pr_ent)
                ax.axhspan(pr_mean - pr_std, pr_mean + pr_std,
                           color=PALETTE['TabPFN Prior'], alpha=0.15,
                           label=f'Prior ref (mean={pr_mean:.2f})')
                ax.axhline(pr_mean, color=PALETTE['TabPFN Prior'],
                           linestyle='--', linewidth=1)

        if 'noise' in entropy_results:
            noise_ent = entropy_results['noise']
            if len(noise_ent) > 0:
                noise_mean = np.mean(noise_ent)
                noise_std = np.std(noise_ent)
                ax.axhspan(noise_mean - noise_std, noise_mean + noise_std,
                           color=PALETTE['Noise (OOD)'], alpha=0.15,
                           label=f'Noise OOD (mean={noise_mean:.2f})')
                ax.axhline(noise_mean, color=PALETTE['Noise (OOD)'],
                           linestyle='--', linewidth=1)

        ax.set_title(f'{dataset_type.upper()}')
        ax.set_ylabel('Shannon Entropy')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Bar-Distribution Entropy: Neurostim vs Synthetic', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        base = _save_dir(output_dir, 'entropy')
        path = os.path.join(base, 'entropy_distribution.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


def plot_entropy_heatmap(entropy_results, dataset_type, save=False, output_dir=None):
    """Heatmap: mean entropy per (subject x EMG) for one dataset type.

    Args:
        entropy_results: dict from entropy_analysis()
        dataset_type: which dataset to plot
        save: whether to save figure
        output_dir: base output directory
    """
    ds_data = entropy_results.get(dataset_type, {})
    if not ds_data:
        return

    # Build matrix
    subjects = sorted(ds_data.keys())
    all_emgs = set()
    for subj_data in ds_data.values():
        all_emgs.update(subj_data.keys())
    emgs = sorted(all_emgs)

    matrix = np.full((len(subjects), len(emgs)), np.nan)
    for i, subj in enumerate(subjects):
        for j, emg in enumerate(emgs):
            if emg in ds_data[subj]:
                matrix[i, j] = np.mean(ds_data[subj][emg])

    fig, ax = plt.subplots(figsize=(max(6, len(emgs) * 0.8),
                                     max(3, len(subjects) * 0.6)))
    sns.heatmap(matrix, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f',
                xticklabels=[f'EMG {e}' for e in emgs],
                yticklabels=[f'S{s}' for s in subjects])
    ax.set_title(f'Mean Entropy | {dataset_type.upper()}')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('Subject')

    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'entropy')
        path = os.path.join(base, f'entropy_heatmap_{dataset_type}.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


def plot_mmd_barplot(mmd_results, save=False, output_dir=None):
    """Grouped bar chart: MMD vs GP ref and MMD vs Prior Bag ref.

    P-value annotations: * p<0.05, ** p<0.01, ns.

    Args:
        mmd_results: dict from mmd_analysis()
        save: whether to save figure
        output_dir: base output directory
    """
    import pandas as pd

    plot_data = []
    for dataset_type, ds_data in mmd_results.items():
        for subj_idx, subj_data in ds_data.items():
            for emg_idx, emg_data in subj_data.items():
                label = f'{dataset_type} S{subj_idx}\nEMG{emg_idx}'
                ref_labels = {
                    'gp': 'Synthetic GP',
                    'prior': 'TabPFN Prior',
                    'noise': 'Noise (OOD)',
                }
                for ref_name in ['gp', 'prior', 'noise']:
                    mmd_key = f'mmd2_{ref_name}'
                    p_key = f'p_{ref_name}'
                    if mmd_key in emg_data:
                        plot_data.append({
                            'Experiment': label,
                            'MMD²': emg_data[mmd_key],
                            'p-value': emg_data[p_key],
                            'Reference': ref_labels[ref_name],
                        })

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)

    # Aggregate per dataset-subject (mean across EMGs)
    df['DS_Subj'] = df['Experiment'].str.split('\n').str[0]
    agg = df.groupby(['DS_Subj', 'Reference']).agg(
        MMD2_mean=('MMD²', 'mean'),
        MMD2_sem=('MMD²', 'sem'),
        p_mean=('p-value', 'mean'),
    ).reset_index()

    n_groups = agg['DS_Subj'].nunique()
    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.5), 5))

    ref_types = agg['Reference'].unique()
    x = np.arange(n_groups)
    n_refs = len(ref_types)
    width = 0.25
    ds_subjects = sorted(agg['DS_Subj'].unique())

    for i, ref in enumerate(ref_types):
        ref_data = agg[agg['Reference'] == ref].set_index('DS_Subj')
        vals = [ref_data.loc[ds, 'MMD2_mean'] if ds in ref_data.index else 0
                for ds in ds_subjects]
        errs = [ref_data.loc[ds, 'MMD2_sem'] if ds in ref_data.index else 0
                for ds in ds_subjects]
        p_vals = [ref_data.loc[ds, 'p_mean'] if ds in ref_data.index else 1
                  for ds in ds_subjects]

        color = PALETTE.get(ref, 'gray')
        offset = (i - (n_refs - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs,
                      label=ref, color=color, alpha=0.8, capsize=3)

        # P-value annotations
        for j, (bar, p) in enumerate(zip(bars, p_vals)):
            if p < 0.01:
                annot = '**'
            elif p < 0.05:
                annot = '*'
            else:
                annot = 'ns'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    annot, ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_subjects, rotation=45, ha='right')
    ax.set_ylabel('MMD²')
    ax.set_title('MMD² vs Synthetic References')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'mmd')
        path = os.path.join(base, 'mmd_barplot.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


def plot_mahalanobis_distribution(mahalanobis_results, save=False, output_dir=None):
    """Violin plot: Mahalanobis distance distributions per dataset.

    Overlays held-out synthetic reference distances for calibration.

    Args:
        mahalanobis_results: dict from mahalanobis_analysis()
        save: whether to save figure
        output_dir: base output directory
    """
    import pandas as pd

    dataset_types = [k for k in mahalanobis_results
                     if k != 'ref_stats']
    if not dataset_types:
        return

    plot_data = []
    for dataset_type in dataset_types:
        ds_data = mahalanobis_results[dataset_type]
        for subj_idx, subj_data in ds_data.items():
            for emg_idx, emg_data in subj_data.items():
                for ref_name in ['gp', 'prior', 'noise']:
                    dist_key = f'distances_{ref_name}'
                    if dist_key in emg_data:
                        ref_label_map = {
                            'gp': 'vs GP', 'prior': 'vs Prior',
                            'noise': 'vs Noise',
                        }
                        for d in emg_data[dist_key]:
                            plot_data.append({
                                'Dataset': dataset_type.upper(),
                                'Distance': float(d),
                                'Source': 'Neurostim',
                                'Reference': ref_label_map[ref_name],
                            })

    # Add reference self-distances
    ref_label_map = {
        'gp': ('Synthetic GP', 'vs GP'),
        'prior': ('TabPFN Prior', 'vs Prior'),
        'noise': ('Noise (OOD)', 'vs Noise'),
    }
    ref_stats = mahalanobis_results.get('ref_stats', {})
    for ref_name, stats in ref_stats.items():
        label, ref_col = ref_label_map.get(
            ref_name, (ref_name, f'vs {ref_name}'),
        )
        for d in stats['self_distances']:
            for dataset_type in dataset_types:
                plot_data.append({
                    'Dataset': dataset_type.upper(),
                    'Distance': float(d),
                    'Source': label,
                    'Reference': ref_col,
                })

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)

    refs = df['Reference'].unique()
    n_panels = len(refs)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)

    source_palette = {
        'Neurostim': PALETTE['Neurostim'],
        'Synthetic GP': PALETTE['Synthetic GP'],
        'TabPFN Prior': PALETTE['TabPFN Prior'],
        'Noise (OOD)': PALETTE['Noise (OOD)'],
    }

    for ax_i, ref in enumerate(refs):
        ax = axes[0, ax_i]
        ref_df = df[df['Reference'] == ref]
        sns.violinplot(data=ref_df, x='Dataset', y='Distance', hue='Source',
                       palette=source_palette, ax=ax, inner='box', alpha=0.7)
        ax.set_title(f'Mahalanobis Distance ({ref})')
        ax.set_ylabel('D_M')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Mahalanobis Distance in TabPFN Representation Space', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        base = _save_dir(output_dir, 'mahalanobis')
        path = os.path.join(base, 'mahalanobis_distribution.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


def plot_summary_dashboard(entropy_results, mmd_results, mahalanobis_results,
                           save=False, output_dir=None):
    """3-panel paper-ready summary.

    Panel A: Entropy | Panel B: MMD | Panel C: Mahalanobis

    Args:
        entropy_results: dict from entropy_analysis()
        mmd_results: dict from mmd_analysis()
        mahalanobis_results: dict from mahalanobis_analysis()
        save: whether to save figure
        output_dir: base output directory
    """
    import pandas as pd

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel A: Entropy summary ---
    ax = axes[0]
    _ref_keys = {'synthetic_gp', 'synthetic_prior', 'noise'}
    dataset_types = [k for k in entropy_results if k not in _ref_keys]
    box_data = []
    for dt in dataset_types:
        for subj_data in entropy_results[dt].values():
            for entropy_arr in subj_data.values():
                for val in entropy_arr:
                    box_data.append({'Dataset': dt.upper(), 'Entropy': float(val)})

    # Add synthetic
    for synth_key, label in [('synthetic_gp', 'GP Ref'),
                              ('synthetic_prior', 'Prior Ref'),
                              ('noise', 'Noise OOD')]:
        if synth_key in entropy_results:
            for val in entropy_results[synth_key]:
                box_data.append({'Dataset': label, 'Entropy': float(val)})

    if box_data:
        df_ent = pd.DataFrame(box_data)
        palette_a = {dt.upper(): PALETTE['Neurostim'] for dt in dataset_types}
        palette_a['GP Ref'] = PALETTE['Synthetic GP']
        palette_a['Prior Ref'] = PALETTE['TabPFN Prior']
        palette_a['Noise OOD'] = PALETTE['Noise (OOD)']
        sns.boxplot(data=df_ent, x='Dataset', y='Entropy', palette=palette_a,
                    ax=ax, fliersize=2)
    ax.set_title('A) Bar-Distribution Entropy')
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel B: MMD summary ---
    ax = axes[1]
    mmd_data = []
    for dt, ds_data in mmd_results.items():
        for subj_data in ds_data.values():
            for emg_data in subj_data.values():
                ref_label_map = {
                    'gp': 'vs GP', 'prior': 'vs Prior', 'noise': 'vs Noise',
                }
                for ref_name in ['gp', 'prior', 'noise']:
                    mmd_key = f'mmd2_{ref_name}'
                    if mmd_key in emg_data:
                        mmd_data.append({
                            'Dataset': dt.upper(),
                            'MMD²': emg_data[mmd_key],
                            'Reference': ref_label_map[ref_name],
                        })

    if mmd_data:
        df_mmd = pd.DataFrame(mmd_data)
        sns.barplot(data=df_mmd, x='Dataset', y='MMD²', hue='Reference',
                    palette=[PALETTE['Synthetic GP'], PALETTE['TabPFN Prior'],
                             PALETTE['Noise (OOD)']],
                    ax=ax, capsize=0.1, errorbar=('ci', 95))
    ax.set_title('B) MMD² vs References')
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel C: Mahalanobis summary ---
    ax = axes[2]
    mah_data = []
    m_dataset_types = [k for k in mahalanobis_results if k != 'ref_stats']
    for dt in m_dataset_types:
        for subj_data in mahalanobis_results[dt].values():
            for emg_data in subj_data.values():
                ref_label_map = {
                    'gp': 'vs GP', 'prior': 'vs Prior', 'noise': 'vs Noise',
                }
                for ref_name in ['gp', 'prior', 'noise']:
                    dist_key = f'distances_{ref_name}'
                    if dist_key in emg_data:
                        for d in emg_data[dist_key]:
                            mah_data.append({
                                'Dataset': dt.upper(),
                                'D_M': float(d),
                                'Reference': ref_label_map[ref_name],
                            })

    if mah_data:
        df_mah = pd.DataFrame(mah_data)
        sns.boxplot(data=df_mah, x='Dataset', y='D_M', hue='Reference',
                    palette=[PALETTE['Synthetic GP'], PALETTE['TabPFN Prior'],
                             PALETTE['Noise (OOD)']],
                    ax=ax, fliersize=2)
    ax.set_title('C) Mahalanobis Distance')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('ID/OOD Analysis: Is Neurostim Data Within TabPFN\'s Prior?',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        base = output_dir if output_dir else os.path.join('output', 'id_ood')
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, 'summary_dashboard.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()
