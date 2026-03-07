import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os
import json
from datetime import datetime
import csv


# ============================================
#           Data Loader and Preprocessing
# ============================================


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

        return {
        'emgs': emgs,
        'nChan': nChan,
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
        'sorted_respMean': sorted_respMean,
        'sorted_respSD': sorted_respSD,
        'ch2xy': ch2xy,
        'DimSearchSpace': 96
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

        return {
        'emgs': emgs,
        'nChan': nChan,
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
        'sorted_respMean': sorted_respMean,
        'sorted_respSD': sorted_respSD,
        'ch2xy': ch2xy,
        'DimSearchSpace': 32
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

        subject = {
            'emgs': emgs,
            'nChan': 64,
            'DimSearchSpace': 64,
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

HELD_OUT_SUBJECTS = {'rat': [0, 5], 'nhp': [1]}
TRAIN_SUBJECTS = {'rat': [1, 2, 3, 4], 'nhp': [0, 2, 3]}
ALL_SUBJECTS = {'rat': [0, 1, 2, 3, 4, 5], 'nhp': [0, 1, 3]}


def generate_experiment_tag(dataset_type, split_type, epochs, lr, n_augmentations,
                             held_out_subj_idx=None, held_out_emg_idx=None):
    """
    Build a deterministic, human-readable tag encoding all training hyper-params.

    Tag format:
        {dataset}_{split}[_subj{N}][_emg{N}]_ep{E}_lr{LR}_aug{A}

    Examples:
        nhp_inter_subject_ep20_lr1.00e-06_aug25
        rat_intra_emg_emg3_ep20_lr1.00e-06_aug25
        nhp_inter_subject_subj0_ep30_lr1.00e-05_aug50
    """
    lr_str = f"{lr:.2e}"
    parts = [dataset_type, split_type]
    if held_out_subj_idx is not None:
        parts.append(f'subj{held_out_subj_idx}')
    if held_out_emg_idx is not None:
        parts.append(f'emg{held_out_emg_idx}')
    parts += [f'ep{epochs}', f'lr{lr_str}', f'aug{n_augmentations}']
    return '_'.join(parts)


def create_run_dir(exp_tag: str, base_dir='./output/runs') -> str:
    """Create output/runs/{exp_tag}_{timestamp}/ and return its path."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f'{exp_tag}_{timestamp}')
    for sub in ('fitness', 'optimization', 'results'):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def write_run_config(run_dir: str, config: dict) -> str:
    """Serialize config dict to {run_dir}/config.json. Returns the file path."""
    path = os.path.join(run_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
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
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math

    mean_map = subject_data['sorted_respMean'][:, emg_idx]  # (nChan,)
    nChan = len(mean_map)
    if nChan == 96:
        grid_shape = (8, 12)
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


def save_results(results_dict, evaluation_type, output_dir='./output/results', tag=''):
    """
    Persist experiment results as a full-fidelity pickle and a scalar summary CSV.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
                      (as returned by main() or the finetuning evaluation loops).
        evaluation_type: 'fit' or 'optimization'
        output_dir: directory to write into (created if absent).
        tag: optional suffix for the filename (e.g. 'finetuned_vs_gp').

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
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Summary CSV ---
    rows = []
    for model_name, result_list in results_dict.items():
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


def load_results(pickle_path):
    """
    Reload a results_dict from a pickle saved by save_results().

    Returns the exact dict[str, list[dict]] for direct use with
    r2_comparison, regret_curve, show_emg_map, plot_runtime_trajectory, etc.
    """
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    DATASET  = 'nhp'
    SUBJ_IDX = 1
    EMG_IDX  = 4
    N_SHOW   = 12

    data = load_data(DATASET, SUBJ_IDX)
    plot_augmented_maps(data, EMG_IDX, DATASET, SUBJ_IDX, n_show=N_SHOW)

