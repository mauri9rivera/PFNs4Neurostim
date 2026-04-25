import hashlib
import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os
import json
import yaml
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import Any, Dict, List, Optional
import pandas as pd


# ============================================
#           Data Loader and Preprocessing
# ============================================


def _topographic_reorder(sorted_resp, sorted_respMean, sorted_respSD,
                         sorted_isvalid, ch2xy, maps):
    """Reorder channel-indexed arrays to row-major grid order, padding grounds.

    The Utah microelectrode array has a scrambled pin-to-electrode mapping.
    ch2xy[i] = [row, col] gives the physical grid position of channel i.
    maps.shape gives the full grid dimensions (e.g. 10x10 for NHP).

    After reordering, channel k in the output corresponds to grid position
    (k // n_cols, k % n_cols), so array.reshape(grid_shape) gives the correct
    topographic map. Ground positions (grid cells with no electrode) are filled
    with 0.

    Returns:
        (sorted_resp, sorted_respMean, sorted_respSD, sorted_isvalid,
         ch2xy, grid_shape) — all reordered/padded.
    """
    grid_shape = maps.shape[:2]
    n_rows, n_cols = grid_shape
    n_grid = n_rows * n_cols

    # Map grid position → original channel index
    grid_to_chan = {}
    for i in range(ch2xy.shape[0]):
        r, c = int(round(ch2xy[i, 0])), int(round(ch2xy[i, 1]))
        grid_to_chan[(r, c)] = i

    # Row-major ordering: grid position 0 = (0,0), 1 = (0,1), ...
    topo_order = []
    topo_coords = []
    for r in range(n_rows):
        for c in range(n_cols):
            topo_order.append(grid_to_chan.get((r, c), -1))
            topo_coords.append([r, c])

    def _reorder(arr):
        """Reorder first axis from channel order to row-major grid order."""
        out = np.zeros((n_grid,) + arr.shape[1:], dtype=arr.dtype)
        for g, ch in enumerate(topo_order):
            if ch >= 0:
                out[g] = arr[ch]
        return out

    return (
        _reorder(sorted_resp),
        _reorder(sorted_respMean),
        _reorder(sorted_respSD),
        _reorder(sorted_isvalid),
        np.array(topo_coords, dtype=ch2xy.dtype),
        grid_shape,
    )


def load_data(dataset_type, m_i):
    '''
    Input: 
        - dataset_type: str characterizing the modality of the experiment
        - m_i: int of subject

    Output:
        - dictionary of neurostimulation data
    
        Important: sorted response shape: (nChan, nEmgs, nReps)
    '''
    
    path_to_dataset = f'./data'
    if dataset_type=='nhp':
        if m_i==0:
            data = scipy.io.loadmat(path_to_dataset+'/monkeys/Cebus1_M1_190221.mat')['Cebus1_M1_190221'][0][0]
        elif m_i==1:
            data = scipy.io.loadmat(path_to_dataset+'/monkeys/Cebus2_M1_200123.mat')['Cebus2_M1_200123'][0][0]
        elif m_i==2:    
            data = scipy.io.loadmat(path_to_dataset+'/monkeys/Macaque1_M1_181212.mat')['Macaque1_M1_181212'][0][0]
        elif m_i==3:
            data  = scipy.io.loadmat(path_to_dataset+'/monkeys/Macaque2_M1_190527.mat')['Macaque2_M1_190527'][0][0]

        if m_i >= 2:
            #macaques
            mapping = {
                'emgs': 0, 'emgsabr': 1, 'nChan': 2, 'stimProfile': 3, 'stim_channel': 4, 
                'evoked_emg': 5, 'response': 6, 'isvalid': 7, 'sorted_isvalid': 8, 'sorted_resp': 9, 
                'sorted_evoked': 10, 'sampFreqEMG': 11, 'resp_region': 12, 'map': 13, 'ch2xy': 14, 
                'sorted_respMean': 15, 'sorted_respSD': 16
            }
        else:
            # cebus
            mapping = {
                'emgs': 0, 'emgsabr': 1, 'nChan': 2, 'stimProfile': 3, 'stim_channel': 4, 
                'evoked_emg': 5, 'response': 6, 'isvalid': 7, 'sorted_isvalid': 8, 'sorted_resp': 9, 
                'sorted_respMean': 10, 'sorted_respSD': 11, 'sorted_evoked': 12, 'sampFreqEMG': 13, 
                'resp_region': 14, 'map': 15, 'ch2xy': 16
            }

        nChan = data[mapping['nChan']][0][0]
        evoked_emg = np.stack(data[mapping['evoked_emg']][0], axis=0)

        rN = data[mapping['sorted_isvalid']]
        j1, j2, j3 = rN.shape[0], rN.shape[1], rN[0][0].shape[0]
        sorted_isvalid = np.stack([np.squeeze(rN[i, j]) for i in range(j1) for j in range(j2)], axis=0)
        sorted_isvalid = sorted_isvalid.reshape(j1, j2, j3)

        emgs = {
            'emgs': [name[0] for name in data[mapping['emgs']][0]],
            'emgsabr': [name[0] for name in data[mapping['emgsabr']][0]]
        }

        ch2xy = data[mapping['ch2xy']] - 1
        se = data[mapping['sorted_evoked']]
        i1, i2, i3, i4 = se.shape[0], se.shape[1], se[0][0].shape[0], se[0][0].shape[1]
        sorted_evoked = np.stack([np.squeeze(se[i, j]) for i in range(i1) for j in range(i2)], axis=0)
        sorted_evoked = sorted_evoked.reshape(i1, i2, i3, i4)
        sorted_filtered = sorted_evoked

        stim_channel = data[mapping['stim_channel']]
        if stim_channel.shape[0] == 1:
            stim_channel = stim_channel[0]

        fs = data[mapping['sampFreqEMG']][0][0]
        parameters = {'c': nChan, 'j': stim_channel.shape[0]}
        n_muscles = data[mapping['emgs']].shape[1]
        maps = data[mapping['map']]
        resp_region = data[mapping['resp_region']][0]

        stimProfile = data[mapping['stimProfile']][0]
        
        # compute baseline
        where_zero = np.where(abs(stimProfile) > 10**(-50))[0][0]
        window_size = int(fs * 30 * 10**(-3))
        baseline = []
        for iChan in range(nChan):
            reps = np.where(stim_channel == iChan + 1)[0]
            n_rep = len(reps)
            # Compute mean over the last dimension (time), across those repetitions
            mean_baseline = np.mean(sorted_filtered[iChan, :, :n_rep, where_zero - window_size : where_zero], axis=-1)
            baseline.append(mean_baseline)
        
        baseline = np.stack(baseline, axis=0)  # shape: (nChan, nSamples)
        
        sorted_filtered = sorted_filtered - baseline[..., np.newaxis]
        sorted_resp = np.max(sorted_filtered[:,:,:n_rep,resp_region[0]:resp_region[1]], axis=-1)

        # Create a masked array where invalid points are masked
        masked_resp = np.ma.masked_where(sorted_isvalid == 0, sorted_resp)

        # Compute the mean and std over the last axis, ignoring masked (invalid) values
        sorted_respMean = masked_resp.mean(axis=-1)
        sorted_respSD = masked_resp.std(axis=-1)
        sorted_respSD = np.ma.filled(sorted_respSD, fill_value=0.0)
        sorted_respMean = np.ma.filled(sorted_respMean, fill_value=0.0)

        emgs = data[0][0]

        # Topographic reorder: place channels at correct grid positions
        sorted_resp, sorted_respMean, sorted_respSD, sorted_isvalid, ch2xy, grid_shape = \
            _topographic_reorder(sorted_resp, sorted_respMean, sorted_respSD,
                                 sorted_isvalid, ch2xy, maps)

        return {
        'correspondance': mapping,
        'emgs': emgs,
        'evoked_emg': evoked_emg,
        'nChan': grid_shape[0] * grid_shape[1],
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
        'sorted_respMean': sorted_respMean,
        'sorted_respSD': sorted_respSD,
        'sorted_evoked': sorted_evoked,
        'sorted_filtered': sorted_filtered,
        'ch2xy': ch2xy,
        'parameters': parameters, 'n_muscles': n_muscles, 'maps': maps,
        'DimSearchSpace': grid_shape[0] * grid_shape[1],
        }

    elif dataset_type=='rat':  # rat dataset has 6 subjects
        if m_i==0:
            data = scipy.io.loadmat(path_to_dataset+'/rat/rat1_M1_190716.mat')['rat1_M1_190716'][0][0]
        elif m_i==1:
            data = scipy.io.loadmat(path_to_dataset+'/rat/rat2_M1_190617.mat')['rat2_M1_190617'][0][0]     
        elif m_i==2:
            data = scipy.io.loadmat(path_to_dataset+'/rat/rat3_M1_190728.mat')['rat3_M1_190728'][0][0]                  
        elif m_i==3:
            data = scipy.io.loadmat(path_to_dataset+'/rat/rat4_M1_191109.mat')['rat4_M1_191109'][0][0]                  
        elif m_i==4:
            data = scipy.io.loadmat(path_to_dataset+'/rat/rat5_M1_191112.mat')['rat5_M1_191112'][0][0]                  
        elif m_i==5:
            data = scipy.io.loadmat(path_to_dataset+'/rat/rat6_M1_200218.mat')['rat6_M1_200218'][0][0]   

        mapping = {
                'emgs': 0, 'emgsabr': 1, 'nChan': 2, 'stimProfile': 3, 'stim_channel': 4, 
                'evoked_emg': 5, 'response': 6, 'isvalid': 7, 'sorted_isvalid': 8, 'sorted_resp': 9, 
                'sorted_evoked': 10, 'sampFreqEMG': 11, 'resp_region': 12, 'map': 13, 'ch2xy': 14, 
                'sorted_respMean': 15, 'sorted_respSD': 16
            }
        
        nChan = data[mapping['nChan']][0][0]

        rN = data[mapping['sorted_isvalid']]
        j1, j2, j3 = rN.shape[0], rN.shape[1], rN[0][0].shape[0]
        sorted_isvalid = np.stack([np.squeeze(rN[i, j]) for i in range(j1) for j in range(j2)], axis=0)
        sorted_isvalid = sorted_isvalid.reshape(j1, j2, j3)

        ch2xy = data[mapping['ch2xy']] - 1
        se = data[mapping['sorted_evoked']]
        i1, i2, i3, i4 = se.shape[0], se.shape[1], se[0][0].shape[0], se[0][0].shape[1]
        sorted_evoked = np.stack([np.squeeze(se[i, j]) for i in range(i1) for j in range(i2)], axis=0)
        sorted_evoked = sorted_evoked.reshape(i1, i2, i3, i4)
        sorted_filtered = sorted_evoked

        stim_channel = data[mapping['stim_channel']]
        if stim_channel.shape[0] == 1:
            stim_channel = stim_channel[0]

        fs = data[mapping['sampFreqEMG']][0][0]
        maps = data[mapping['map']]
        resp_region = data[mapping['resp_region']][0]

        stimProfile = data[mapping['stimProfile']][0]

        # compute baseline
        where_zero = np.where(abs(stimProfile) > 10**(-50))[0][0]
        window_size = int(fs * 30 * 10**(-3))
        baseline = []
        for iChan in range(nChan):
            reps = np.where(stim_channel == iChan + 1)[0]
            n_rep = len(reps)
            # Compute mean over the last dimension (time), across those repetitions
            mean_baseline = np.mean(sorted_filtered[iChan, :, :n_rep, where_zero - window_size : where_zero], axis=-1)
            baseline.append(mean_baseline)

        baseline = np.stack(baseline, axis=0)  # shape: (nChan, nSamples)

        sorted_filtered = sorted_filtered - baseline[..., np.newaxis]
        sorted_resp = np.max(sorted_filtered[:,:,:n_rep,resp_region[0]:resp_region[1]], axis=-1)
        # Create a masked array where invalid points are masked
        masked_resp = np.ma.masked_where(sorted_isvalid == 0, sorted_resp)

        # Compute the mean and std over the last axis, ignoring masked (invalid) values
        sorted_respMean = masked_resp.mean(axis=-1)
        sorted_respSD = masked_resp.std(axis=-1)
        sorted_respSD = np.ma.filled(sorted_respSD, fill_value=0.0)
        sorted_respMean = np.ma.filled(sorted_respMean, fill_value=0.0)

        emgs = data[0][0]

        # Topographic reorder: place channels at correct grid positions
        sorted_resp, sorted_respMean, sorted_respSD, sorted_isvalid, ch2xy, grid_shape = \
            _topographic_reorder(sorted_resp, sorted_respMean, sorted_respSD,
                                 sorted_isvalid, ch2xy, maps)

        return {
        'emgs': emgs,
        'nChan': grid_shape[0] * grid_shape[1],
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
        'sorted_respMean': sorted_respMean,
        'sorted_respSD': sorted_respSD,
        'ch2xy': ch2xy,
        'DimSearchSpace': grid_shape[0] * grid_shape[1],
        }
    elif dataset_type =='spinal':

        subject_map = {
            0: 'rat0_C5_500uA.pkl', 1: 'rat1_C5_500uA.pkl', 2: 'rat1_C5_700uA.pkl', 3: 'rat1_midC4_500uA.pkl',
            4: 'rat2_C4_300uA.pkl', 5: 'rat2_C5_300uA.pkl', 6: 'rat2_C6_300uA.pkl', 7: 'rat3_C4_300uA.pkl',
            8: 'rat3_C5_200uA.pkl', 9: 'rat3_C5_350uA.pkl', 10: 'rat3_C6_300uA.pkl' 
        }
        
        #load data
        with open(f'{path_to_dataset}/spinal/{subject_map[m_i]}', "rb") as f:
            data = pickle.load(f)
        
        ch2xy, emgs = data['ch2xy'], data['emgs']
        evoked_emg, filtered_emg = data['evoked_emg'], data['filtered_emg']
        maps = data['map']
        parameters = data['parameters']
        resp_region = data['resp_region']
        fs = data['sampFreqEMG']
        sorted_evoked = data['sorted_evoked']
        sorted_filtered = data['sorted_filtered']
        sorted_resp = data['sorted_resp']
        sorted_isvalid = data['sorted_isvalid']
        sorted_respMean = data['sorted_respMean']
        sorted_respSD = data['sorted_respSD']
        stim_channel = data['stim_channel']
        stimProfile=data['stimProfile']
        n_muscles = emgs.shape[0]

        #?# We are removing lots of reps here print(f'sorted response: {sorted_resp.shape}') 
        #Computing baseline for filtered signal
        nChan = parameters['nChan'][0]
        where_zero = np.where(abs(stimProfile) > 10**(-50))[0][0]
        window_size = int(fs * 35 * 10**(-3))
        baseline = []
        n_rep = 10000 # First, determine n_reps global
        for iChan in range(nChan):
            reps= np.where(stim_channel == iChan + 1)[0]
            if len(reps) < n_rep:
                n_rep = len(reps)
        for iChan in range(nChan):
            mean_baseline = np.mean(sorted_filtered[iChan, :, :n_rep, 0 : where_zero], axis=-1)
            baseline.append(mean_baseline)
        
        baseline = np.stack(baseline, axis=0)

        #remove baseline from filtered signal
        sorted_filtered[:, :, :n_rep, :] = sorted_filtered[:, :, :n_rep, :] - baseline[..., np.newaxis]
        sorted_resp = np.nanmax(sorted_filtered[:, :, :n_rep, int(resp_region[0]): int(resp_region[1])], axis=-1)
        masked_resp = np.ma.masked_where(sorted_isvalid[:, :, :n_rep] == 0, sorted_resp)
        sorted_respMean = masked_resp.mean(axis=-1)

         # compute baseline for evoked signal
        baseline = []
        for iChan in range(nChan):
            # Compute mean over the last dimension (time), across those repetitions
            mean_baseline = np.mean(sorted_evoked[iChan, :, :n_rep, 0 : where_zero], axis=-1)
            baseline.append(mean_baseline)
        baseline = np.stack(baseline, axis=0)  # shape: (nChan, nSamples)
        
        #remove baseline from evoked signal
        sorted_evoked[:, :, :n_rep, :] = sorted_evoked[:, :, :n_rep, :] - baseline[..., np.newaxis]
        sorted_resp = np.nanmax(sorted_evoked[:,:,:n_rep,int(resp_region[0]) :int(resp_region[1])], axis=-1)
        masked_resp = np.ma.masked_where(sorted_isvalid[:,:,:n_rep] == 0, sorted_resp)

        #mask sorted_isvalid by n_rep
        sorted_isvalid = sorted_isvalid[:, :, :n_rep]

        # Topographic reorder: place channels at correct grid positions
        sorted_resp, sorted_respMean, sorted_respSD, sorted_isvalid, ch2xy, grid_shape = \
            _topographic_reorder(sorted_resp, sorted_respMean, sorted_respSD,
                                 sorted_isvalid, ch2xy, maps)
        n_grid = grid_shape[0] * grid_shape[1]

        subject = {
            'emgs': emgs,
            'nChan': n_grid,
            'DimSearchSpace': n_grid,
            'sorted_respMean': sorted_respMean,
            'ch2xy': ch2xy,
            'evoked_emg': evoked_emg, 'filtered_emg':filtered_emg, 'sorted_resp': sorted_resp,
            'sorted_isvalid': sorted_isvalid, 'sorted_respSD': sorted_respSD,
            'sorted_filtered': sorted_filtered, 'stim_channel': stim_channel, 'fs': fs,
        'parameters': parameters, 'n_muscles': n_muscles, 'maps': maps,
        'resp_region': resp_region, 'stimProfile': stimProfile,  'baseline' : baseline
        }

        return subject
    else:
        raise ValueError('The dataset type should be 5d_rat, nhp, rat or spinal' )


# ============================================
#      Held-Out / Train Subject Splits
# ============================================

HELD_OUT_SUBJECTS = {'rat': [0, 5], 'nhp': [1], 'spinal': [0, 2, 5, 9]}
TRAIN_SUBJECTS = {'rat': [1, 2, 3, 4], 'nhp': [0, 3], 'spinal': [1, 3, 4, 6, 7, 8, 10]}
ALL_SUBJECTS = {'rat': [0, 1, 2, 3, 4, 5], 'nhp': [0, 1, 3], 'spinal': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}


def generate_experiment_tag(
    dataset: str,
    family: str,
    config: dict[str, Any],
) -> str:
    """Build a short deterministic experiment tag.

    Tag format: ``{dataset}-{family}-{5char_hash}``

    The hash is derived from the full config dict so that the same
    hyperparameter combination always produces the same tag.  Different
    configs — even differing by a single value — produce different hashes,
    making every experiment uniquely addressable without verbose filenames.

    Args:
        dataset: Dataset identifier, e.g. ``'nhp'`` or ``'rat'``.
        family: Experiment family name, e.g. ``'optimization'`` or
            ``'lora-ablation'``.
        config: Full hyperparameter dict for this run.  All values must be
            JSON-serialisable.  The dict is sorted by key before hashing so
            insertion order does not affect the result.

    Returns:
        A tag string of the form ``{dataset}-{family}-{5char_hash}``, e.g.
        ``nhp-optimization-a3f9c``.

    Example:
        >>> tag = generate_experiment_tag(
        ...     'nhp', 'optimization',
        ...     {'epochs': 50, 'lr': 1e-5, 'n_augmentations': 25},
        ... )
        >>> assert len(tag.split('-')) == 3
    """
    serialised = json.dumps(config, sort_keys=True, default=str)
    digest = hashlib.md5(serialised.encode()).hexdigest()[:5]
    return f"{dataset}-{family}-{digest}"


def create_run_dir(
    exp_tag: str,
    base_dir: str = './output/runs',
    tag: Optional[str] = None,
) -> str:
    """Create a run directory and its standard subdirectories.

    The directory is placed at ``{base_dir}/{tag}/`` when *tag* is provided,
    or at ``{base_dir}/{exp_tag}_{timestamp}/`` for backwards compatibility
    when *tag* is ``None``.

    All standard subdirectories (``fitness``, ``optimization``, ``results``,
    ``diagnostics``, …) are created unconditionally.

    Args:
        exp_tag: Legacy experiment tag string used when *tag* is not provided.
            Still required for the backwards-compatible path.
        base_dir: Root directory for all run outputs.  Defaults to
            ``'./output/runs'``.
        tag: Short experiment tag from :func:`generate_experiment_tag`.
            When provided, the run directory is ``{base_dir}/{tag}/`` — no
            timestamp suffix is added.

    Returns:
        Absolute-or-relative path to the newly created run directory.
    """
    if tag is not None:
        run_dir = os.path.join(base_dir, tag)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(base_dir, f'{exp_tag}_{timestamp}')
    for sub in ('fitness', 'fitness/emg_maps', 'optimization', 'optimization/emg_maps', 'results', 'diagnostics'):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def write_run_config(run_dir: str, config: dict) -> str:
    """Serialize config dict to {run_dir}/config.yaml. Returns the file path."""
    path = os.path.join(run_dir, 'config.yaml')
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Saved config  -> {path}")
    return path


# ============================================
#      Data Augmentation for Fine-Tuning
# ============================================

def augment_maps(subject_data, emg_idx, n_augmentations=25, seed=42):
    """
    Generate augmented (X, y) training pairs by adding per-channel Gaussian noise.

    For each augmentation, sample noise ~ N(0, std_map[channel]) per channel
    and add it to the mean map. Also perturb individual repetitions similarly.

    Args:
        subject_data: dict returned by load_data
        emg_idx: int, which EMG channel to augment
        n_augmentations: int, number of augmented maps to produce
        seed: int, random seed for reproducibility

    Returns:
        List of (X, y) tuples where X = MinMax-scaled coords, y = perturbed response
    """
    rng = np.random.RandomState(seed)

    coords = subject_data['ch2xy']
    mean_map = subject_data['sorted_respMean'][:, emg_idx]  # (nChan,)
    std_map = subject_data['sorted_respSD'][:, emg_idx]     # (nChan,)

    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(coords)
    n_channels = mean_map.shape

    scaler_y = StandardScaler()
    scaler_y.fit(mean_map.reshape(-1, 1))

    augmented_pairs = []
    for _ in range(n_augmentations):
        noise = rng.randn(len(mean_map)) * std_map
        y_aug = scaler_y.transform((mean_map + noise).reshape(-1, 1)).ravel()
        augmented_pairs.append((X_scaled.copy(), y_aug))

    return augmented_pairs


def plot_augmented_maps(subject_data, emg_idx, dataset_type, subj_idx,
                        n_show=6, n_augmentations=25, seed=42):
    """
    Visualize the original EMG map alongside augmented versions (debug only).

    Inverse-transforms augmented y values back to the original response scale
    so all maps share the same colorbar for direct comparison.

    Args:
        subject_data: dict returned by load_data
        emg_idx: int, which EMG channel to visualize
        dataset_type: str, e.g. 'nhp' or 'rat' (used in title only)
        subj_idx: int, subject index (used in title only)
        n_show: number of augmented maps to display (default 6)
        n_augmentations: total augmentations to generate before selecting n_show
        seed: random seed passed to augment_maps
    """


    mean_map = subject_data['sorted_respMean'][:, emg_idx]  # (nChan,)
    nChan = len(mean_map)
    if nChan == 100:
        grid_shape = (10, 10)
    elif nChan == 64:
        grid_shape = (8, 8)
    elif nChan == 32:
        grid_shape = (4, 8)
    else:
        grid_shape = (1, nChan)

    # Generate augmented pairs and inverse-transform y back to response scale
    scaler_y = StandardScaler()
    scaler_y.fit(mean_map.reshape(-1, 1))

    pairs = augment_maps(subject_data, emg_idx,
                         n_augmentations=n_augmentations, seed=seed)
    n_show = min(n_show, len(pairs))
    aug_maps = [
        scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()
        for _, y in pairs[:n_show]
    ]

    # Shared color scale across original + all augmented maps
    all_vals = np.concatenate([mean_map] + aug_maps)
    vmin, vmax = all_vals.min(), all_vals.max()
    heatmap_kw = dict(cmap='viridis', vmin=vmin, vmax=vmax,
                      cbar=False, xticklabels=False, yticklabels=False)

    n_total = 1 + n_show
    n_cols = min(4, n_total)
    n_rows = math.ceil(n_total / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    # Original map
    sns.heatmap(mean_map.reshape(grid_shape), ax=axes_flat[0], **heatmap_kw)
    axes_flat[0].set_title(f'Original\n{dataset_type} S{subj_idx} EMG{emg_idx}',
                           fontsize=9)

    # Augmented maps
    for i, y_map in enumerate(aug_maps):
        sns.heatmap(y_map.reshape(grid_shape), ax=axes_flat[i + 1], **heatmap_kw)
        axes_flat[i + 1].set_title(f'Aug {i + 1}\n{dataset_type} S{subj_idx} EMG{emg_idx}',
                                   fontsize=9)

    # Hide unused axes
    for j in range(n_total, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared colorbar on the right
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax)

    fig.suptitle(f'Data Augmentation | {dataset_type} Subject {subj_idx} EMG {emg_idx}',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.89, 0.95])
    plt.show()


def build_finetuning_dataset(dataset_type, subject_indices=None,
                              held_out_emg_idx=None,
                              n_augmentations=10, seed=42):
    """
    Build a large (X_all, y_all) dataset for fine-tuning TabPFN by augmenting
    training subjects across all EMG channels.

    Args:
        dataset_type: 'rat' or 'nhp'
        subject_indices: list of subject ints to use (defaults to TRAIN_SUBJECTS)
        held_out_emg_idx: int or None. If set, this EMG index is skipped for all
            subjects (intra-EMG holdout). Passing None preserves existing behavior.
        n_augmentations: augmentations per subject-EMG pair
        seed: random seed

    Returns:
        X_all: np.ndarray of shape (N, 2), MinMax-scaled coordinates
        y_all: np.ndarray of shape (N,), response values
    """
    if subject_indices is None:
        subject_indices = TRAIN_SUBJECTS[dataset_type]

    X_parts, y_parts = [], []

    for subj_idx in subject_indices:
        data = load_data(dataset_type, subj_idx)
        n_emgs = data['sorted_respMean'].shape[1]

        for emg_idx in range(n_emgs):
            if held_out_emg_idx is not None and emg_idx == held_out_emg_idx:
                continue
            pairs = augment_maps(data, emg_idx,
                                 n_augmentations=n_augmentations,
                                 seed=seed + subj_idx * 100 + emg_idx)
            for X, y in pairs:
                X_parts.append(X)
                y_parts.append(y)

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    return X_all, y_all


def preprocess_neural_data(subject_data, emg_idx=0, normalization='pfn'):
    """
    Ideally, you would preprocess neural data so that Y_test is the resp_mean,
    but we train (Y_train) on a randomly selected sample of sorted_resp.
    """

    coords = subject_data['ch2xy'] # Shape (96, 2)
    resp_all = subject_data['sorted_resp'][:, emg_idx, :] # Shape (96, repetitions)
    resp_flat = resp_all.reshape(-1, 1) # Shape (96*repetitions, )
    n_channels, n_reps = resp_all.shape
    resp_mean = subject_data['sorted_respMean'][:, emg_idx] # Shape (96,)

    if normalization == 'pfn':
    
        scaler_x = MinMaxScaler()
        X_train = scaler_x.fit_transform(coords)
        
        scaler_y = StandardScaler()
        y_scaled_flat = scaler_y.fit_transform(resp_flat)
        Y_train = y_scaled_flat.reshape(n_channels, n_reps) 
        
        Y_test_scaled = scaler_y.transform(resp_mean.reshape(-1, 1))
        Y_test = Y_test_scaled.flatten()

    else:

        X_train = coords

        scaler_y = MinMaxScaler()
        y_scaled_flat = scaler_y.fit_transform(resp_flat)
        Y_train = y_scaled_flat.reshape(n_channels, n_reps)

        Y_test_scaled = scaler_y.transform(resp_mean.reshape(-1,1))
        Y_test = Y_test_scaled.flatten()


    return X_train, Y_train, X_train, resp_mean, scaler_y


# ============================================
#           Results Persistence
# ============================================


def save_results(
    results_dict: dict,
    evaluation_type: str,
    output_dir: str = './output/results',
    tag: str = '',
    metadata: Optional[dict] = None,
) -> tuple:
    """Persist experiment results as a full-fidelity pickle and a scalar summary CSV.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
                      (as returned by run_experiment() or the evaluation loops).
        evaluation_type: 'fit' or 'optimization'.
        output_dir: Directory to write into (created if absent).
        tag: Optional suffix for the filename (e.g. 'finetuned_vs_gp').
        metadata: Optional dict injected as ``_metadata`` key in the pickle.
            Recommended keys: ``family``, ``dataset``, ``tag``, ``date``,
            ``run_type``, ``held_out_subj``.

    Returns:
        (pickle_path, csv_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Infer dataset name from the first result dict
    first_results = next(iter(results_dict.values()))
    dataset = first_results[0].get('dataset', 'unknown') if first_results else 'unknown'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parts = [dataset, evaluation_type]
    if tag:
        parts.append(tag)
    parts.append(timestamp)
    base = '_'.join(parts)

    pkl_path = os.path.join(output_dir, f'{base}.pkl')
    csv_path = os.path.join(output_dir, f'{base}_summary.csv')

    # --- Pickle (full fidelity) ---
    if metadata is not None:
        results_dict['_metadata'] = metadata
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Summary CSV ---
    rows = []
    for model_name, result_list in results_dict.items():
        if model_name == '_metadata':
            continue
        for res in result_list:
            r2_arr = np.asarray(res['r2'])
            row = {
                'model': model_name,
                'dataset': res.get('dataset', ''),
                'subject': res.get('subject', ''),
                'emg': res.get('emg', ''),
                'mean_r2': float(np.mean(r2_arr)),
                'std_r2': float(np.std(r2_arr)),
                'n_reps': len(r2_arr),
                'mean_time_s': float(np.mean(res['times'])),
            }

            if evaluation_type == 'optimization' and 'values' in res:
                values = np.asarray(res['values'])
                best_so_far = np.maximum.accumulate(values, axis=1)
                optimal = float(res['y_test'].max())
                final_regret = optimal - best_so_far[:, -1]
                row['mean_final_regret'] = float(np.mean(final_regret))
                row['budget'] = values.shape[1]

            rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        # Ensure optimization-only columns appear even when mixed
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved pickle  -> {pkl_path}")
    print(f"Saved summary -> {csv_path}")
    return pkl_path, csv_path


def load_results(pickle_path: str) -> dict:
    """Reload a results_dict from a pickle saved by save_results().

    Args:
        pickle_path: Absolute or relative path to a ``.pkl`` file produced by
            ``save_results()``.

    Returns:
        The exact ``dict[str, list[dict]]`` for direct use with
        ``r2_per_muscle``, ``regret_with_timing``, etc.
    """
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def aggregate_results(
    family: str,
    dataset: str,
    result_type: str,
    runs_dir: str = './output/runs',
    tags: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Find all run directories matching ``{dataset}-{family}-*`` and merge results.

    Scans ``runs_dir`` for subdirectories whose names start with
    ``{dataset}-{family}-``, loads their pkl result files, and concatenates
    them into a single flat DataFrame.

    Args:
        family: Experiment family string, e.g. ``'vanilla-benchmark'`` or
            ``'optimization'``.
        dataset: Dataset type — ``'rat'``, ``'nhp'``, or ``'spinal'``.
        result_type: Which pkl type to load.  One of:

            * ``'fit'`` — ``*_fit.pkl`` files (dict[str, list[dict]])
            * ``'optimization'`` — ``*_optimization.pkl`` files (same schema)
            * ``'optimization_budget'`` — ``*_optimization_budget.pkl``
              (DataFrame pkl, columns: Budget|Model|Regret|R2|ID)

        runs_dir: Root directory that contains per-run subdirectories.
        tags: Optional list of 5-char hash suffixes (e.g. ``['32c2b', '15h5p']``).
            When provided only run directories whose suffix matches are loaded.
            ``None`` means load all directories matching the family prefix.

    Returns:
        Concatenated DataFrame.  Schema for ``fit`` / ``optimization``:

        .. code-block::

            model | dataset | subject | emg | mean_r2 | std_r2 | n_reps |
            mean_time_s [| mean_final_regret | budget] | tag | family

        Schema for ``optimization_budget``:

        .. code-block::

            Budget | Model | Regret | R2 | ID | tag | family

        Returns an empty DataFrame if no matching runs or pkl files are found.

    Raises:
        ValueError: If ``result_type`` is not one of the recognised values.
    """
    valid_types = {'fit', 'optimization', 'optimization_budget'}
    if result_type not in valid_types:
        raise ValueError(
            f"result_type must be one of {sorted(valid_types)}, got {result_type!r}"
        )

    prefix = f"{dataset}-{family}-"

    if not os.path.isdir(runs_dir):
        return pd.DataFrame()

    # Collect matching run directories, optionally filtered to specific hash tags
    tag_set = set(tags) if tags is not None else None
    matching_dirs: List[str] = [
        os.path.join(runs_dir, name)
        for name in os.listdir(runs_dir)
        if name.startswith(prefix)
        and os.path.isdir(os.path.join(runs_dir, name))
        and (tag_set is None or name[len(prefix):] in tag_set)
    ]

    if not matching_dirs:
        return pd.DataFrame()

    all_frames: List[pd.DataFrame] = []

    for run_dir in sorted(matching_dirs):
        tag = os.path.basename(run_dir)
        results_dir = os.path.join(run_dir, 'results')
        if not os.path.isdir(results_dir):
            continue

        pkl_files = [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if f.endswith('.pkl')
        ]

        for pkl_path in sorted(pkl_files):
            fname = os.path.basename(pkl_path)

            # --- Route by result_type ---
            # Pkl filenames from save_results() follow the pattern:
            #   {dataset}_{evaluation_type}_{tag}_{timestamp}.pkl
            # Budget pks saved directly via df.to_pickle() follow:
            #   {tag}_optimization_budget.pkl
            if result_type == 'optimization_budget':
                if '_optimization_budget.pkl' not in fname:
                    continue
                try:
                    df = pd.read_pickle(pkl_path)
                except Exception:
                    continue
                df = df.copy()
                df['tag'] = tag
                df['family'] = family
                all_frames.append(df)

            else:
                # 'fit' or 'optimization': match files that contain
                # f'_{result_type}_' (the evaluation_type component).
                # Exclude budget files (contain 'budget').
                marker = f'_{result_type}_'
                if 'budget' in fname:
                    continue
                if marker not in fname:
                    continue
                try:
                    data = load_results(pkl_path)
                except Exception:
                    continue

                rows: List[Dict[str, Any]] = []
                for model_name, result_list in data.items():
                    if model_name == '_metadata':
                        continue
                    for res in result_list:
                        r2_arr = np.asarray(res['r2'])
                        row: Dict[str, Any] = {
                            'model': model_name,
                            'dataset': res.get('dataset', dataset),
                            'subject': res.get('subject', ''),
                            'emg': res.get('emg', ''),
                            'mean_r2': float(np.mean(r2_arr)),
                            'std_r2': float(np.std(r2_arr)),
                            'n_reps': int(len(r2_arr)),
                            'mean_time_s': float(np.mean(res['times'])),
                            'tag': tag,
                            'family': family,
                        }
                        if result_type == 'optimization' and 'values' in res:
                            values = np.asarray(res['values'])
                            best_so_far = np.maximum.accumulate(values, axis=1)
                            optimal = float(res['y_test'].max())
                            final_regret = optimal - best_so_far[:, -1]
                            row['mean_final_regret'] = float(np.mean(final_regret))
                            row['std_final_regret'] = float(np.std(final_regret))
                            row['budget'] = int(values.shape[1])
                        rows.append(row)

                if rows:
                    all_frames.append(pd.DataFrame(rows))

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


if __name__ == '__main__':
    DATASET  = 'spinal'
    SUBJ_IDX = 1
    EMG_IDX  = 0
    N_SHOW   = 1

    data = load_data(DATASET, SUBJ_IDX)
    #data = load_matlab_data(DATASET, SUBJ_IDX)
    plot_augmented_maps(data, EMG_IDX, DATASET, SUBJ_IDX, n_show=N_SHOW)

