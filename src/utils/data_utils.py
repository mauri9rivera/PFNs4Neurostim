from __future__ import annotations

import hashlib
import numpy as np
import scipy.io
from scipy.stats import spearmanr
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


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank-correlation fit metric, robust to near-flat targets.

    Spearman's ρ measures monotonic agreement between predicted and true
    responses by rank, so unlike R² it does not blow up when the response
    variance (and thus the R² denominator) collapses on non-responsive EMG
    channels — exactly the case that makes spinal R² explode to large
    negatives. Bounded in ``[-1, 1]`` and aligned with what the EMG-map
    argmax/ranking actually shows.

    Args:
        y_true: Ground-truth responses, shape ``[M]``.
        y_pred: Predicted responses, shape ``[M]``.

    Returns:
        Spearman ρ as a float, or ``nan`` if it is undefined (e.g. a constant
        input, where rank correlation has zero variance).
    """
    rho = spearmanr(y_true, y_pred).statistic
    return float(rho) if np.isfinite(rho) else float('nan')


# ============================================
#           Data Loader and Preprocessing
# ============================================


def _topographic_metadata(ch2xy: np.ndarray, maps: np.ndarray) -> tuple:
    """Return integer-cast ``ch2xy`` and grid shape — no array reordering.

    Channel-indexed arrays (``sorted_resp``, ``sorted_respMean``,
    ``sorted_isvalid``, …) stay at length ``nChan_real``.  Ground/empty grid
    positions are *never* materialised, so they cannot pollute ``Y_test`` /
    ``r2_score`` downstream.  ``ch2xy[i] = [row, col]`` keeps mapping the
    *i-th real electrode* to its physical grid position; visualisation code
    scatters values into a NaN-padded display grid using ``ch2xy`` +
    ``grid_shape`` (see ``utils.visualization._to_grid``).

    Args:
        ch2xy: (nChan_real, 2) electrode → (row, col) coords, 0-indexed.
        maps: Full grid layout from the .mat file, shape (n_rows, n_cols, …).

    Returns:
        Tuple ``(ch2xy_int, grid_shape)``.
    """
    grid_shape = (int(maps.shape[0]), int(maps.shape[1]))
    ch2xy_int = np.rint(ch2xy).astype(np.int64)  # [nChan_real, 2]
    return ch2xy_int, grid_shape


def _sort_valid_5drat_reps(resp: np.ndarray) -> np.ndarray:
    """Flag outlier repetitions for the 5-D rat dataset.

    For each (condition, EMG) pair, a repetition is valid if it is finite and
    within 2 standard deviations of the per-pair mean across repetitions.
    Mathematically equivalent to the reference ``sort_valid_5drats`` helper —
    that implementation re-buckets responses through a ``dim_sizes``-derived
    grid before applying the same per-condition threshold, but since each row
    of ``stim_combinations`` is unique, the bucketing is an identity remap and
    can be skipped entirely.

    Args:
        resp: Per-repetition responses, shape ``[n_cond, n_emgs, n_reps]``.

    Returns:
        Binary validity mask, shape ``[n_cond, n_emgs, n_reps]``, dtype int64.
    """
    mean = np.nanmean(resp, axis=-1, keepdims=True)  # [n_cond, n_emgs, 1]
    std = np.nanstd(resp, axis=-1, keepdims=True)  # [n_cond, n_emgs, 1]
    valid = (~np.isnan(resp)) & (np.abs(resp - mean) <= 2 * std)  # [n_cond, n_emgs, n_reps]
    sorted_isvalid = valid.astype(np.int64)

    all_invalid = ~sorted_isvalid.any(axis=-1)  # [n_cond, n_emgs]
    n_all_invalid = int(all_invalid.sum())
    if n_all_invalid > 0:
        print(f"[_sort_valid_5drat_reps] {n_all_invalid} (condition, EMG) pairs have all repetitions flagged invalid.")

    return sorted_isvalid


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

        # Channels remain in their native (real-electrode) order — no padding.
        # grid_shape is exposed only so visualisation can place values into
        # a NaN-padded display grid via ch2xy.
        ch2xy, grid_shape = _topographic_metadata(ch2xy, maps)
        n_real_channels = sorted_resp.shape[0]

        return {
        'correspondance': mapping,
        'emgs': emgs,
        'evoked_emg': evoked_emg,
        'nChan': n_real_channels,
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
        'sorted_respMean': sorted_respMean,
        'sorted_respSD': sorted_respSD,
        'sorted_evoked': sorted_evoked,
        'sorted_filtered': sorted_filtered,
        'ch2xy': ch2xy,
        'grid_shape': grid_shape,
        'parameters': parameters, 'n_muscles': n_muscles, 'maps': maps,
        'DimSearchSpace': n_real_channels,
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

        # Channels remain in their native (real-electrode) order — no padding.
        ch2xy, grid_shape = _topographic_metadata(ch2xy, maps)
        n_real_channels = sorted_resp.shape[0]

        return {
        'emgs': emgs,
        'nChan': n_real_channels,
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
        'sorted_respMean': sorted_respMean,
        'sorted_respSD': sorted_respSD,
        'ch2xy': ch2xy,
        'grid_shape': grid_shape,
        'DimSearchSpace': n_real_channels,
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

        # Channels remain in their native (real-electrode) order — no padding.
        ch2xy, grid_shape = _topographic_metadata(ch2xy, maps)
        n_real_channels = sorted_resp.shape[0]

        subject = {
            'emgs': emgs,
            'nChan': n_real_channels,
            'DimSearchSpace': n_real_channels,
            'sorted_respMean': sorted_respMean,
            'ch2xy': ch2xy,
            'grid_shape': grid_shape,
            'evoked_emg': evoked_emg, 'filtered_emg':filtered_emg, 'sorted_resp': sorted_resp,
            'sorted_isvalid': sorted_isvalid, 'sorted_respSD': sorted_respSD,
            'sorted_filtered': sorted_filtered, 'stim_channel': stim_channel, 'fs': fs,
        'parameters': parameters, 'n_muscles': n_muscles, 'maps': maps,
        'resp_region': resp_region, 'stimProfile': stimProfile,  'baseline' : baseline
        }

        return subject
    elif dataset_type == '5d_rat':
        # 5-D rat motor-cortex stimulation: search space is
        # (pulse-width, frequency, duration, x-channel, y-channel) — no 2D
        # electrode grid, so ch2xy holds raw physical coordinates and
        # grid_shape is None.
        # Each entry: (filename, emgs_raw, valid_emg_idx). `emgs_raw` lists EMG
        # channels in the order documented by each subject's README (rCer1.12/
        # 1.14/1.15: ECR, FCU, Biceps, Triceps, Deltoid). `valid_emg_idx` selects
        # the channels to keep — excludes channels flagged as artifact-prone in
        # the README (e.g. rCer1.15 drops Triceps/Deltoid) or undocumented
        # (BCI00's 5th "unknown" channel). rCer1.14 keeps all 5 despite the
        # README's blanket recommendation, per user override.
        subject_map = {
            0: ('rData03_5D.mat', ['left extensor carpi radialis', 'left flexor carpi ulnaris', 'left triceps', 'left pectoralis'], [1]),
            1: ('rCer1.5_5D.mat', ['left extensor carpi radialis', 'left flexor carpi ulnaris', 'left triceps', 'left biceps'], [0]),
            2: ('BCI00_5D.mat', ['left extensor carpi radialis', 'biceps', 'triceps', 'left flexor carpi ulnaris', 'unknown'], [0, 1, 2, 3]),
            3: ('rCer1.12_5D.mat', ['left extensor carpi radialis', 'left flexor carpi ulnaris', 'left biceps', 'left triceps', 'deltoid'], [0, 1, 3, 4]),
            4: ('rCer1.14_5D.mat', ['left extensor carpi radialis', 'left flexor carpi ulnaris', 'left biceps', 'left triceps', 'deltoid'], [0, 1, 2, 3, 4]),
            5: ('rCer1.15_5D.mat', ['left extensor carpi radialis', 'left flexor carpi ulnaris', 'left biceps', 'left triceps', 'deltoid'], [0, 1, 2]),
            # Subject 6: placeholder, intentionally excluded from ALL_SUBJECTS.
            6: ('5D_step4_noartrej.mat', ['left extensor carpi radialis', 'left flexor carpi ulnaris', 'left biceps', 'left triceps', 'deltoid'], [0, 1, 2, 3, 4]),
        }
        filename, emgs_raw, valid_emg_idx = subject_map[m_i]
        data = scipy.io.loadmat(f'{path_to_dataset}/5d_rat/{filename}')

        resp = data['emg_response']  # [8 reps, n_emgs_raw, n_cond, 4 metrics]
        resp = resp[:, valid_emg_idx, :, :]  # [8 reps, n_emgs, n_cond, 4 metrics]
        emgs = [emgs_raw[i] for i in valid_emg_idx]
        param = data['stim_combinations']  # [n_cond, 7] = [PW, freq, dur, count, chan, x_ch, y_ch]

        # [PW, freq, duration, x_ch, y_ch] in raw physical units.
        ch2xy = param[:, [0, 1, 2, 5, 6]].astype(np.float64)  # [n_cond, 5]

        peak_resp = resp[:, :, :, 0]  # [8 reps, n_emgs, n_cond] — peak-EMG metric
        sorted_resp = peak_resp.transpose(2, 1, 0)  # [n_cond, n_emgs, n_reps]

        sorted_isvalid = _sort_valid_5drat_reps(sorted_resp)

        masked_resp = np.ma.masked_where(sorted_isvalid == 0, sorted_resp)
        sorted_respMean = masked_resp.mean(axis=-1)
        sorted_respSD = masked_resp.std(axis=-1)
        sorted_respSD = np.ma.filled(sorted_respSD, fill_value=0.0)
        sorted_respMean = np.ma.filled(sorted_respMean, fill_value=0.0)

        n_cond = sorted_resp.shape[0]

        return {
            'emgs': emgs,
            'nChan': n_cond,
            'sorted_isvalid': sorted_isvalid,
            'sorted_resp': sorted_resp,
            'sorted_respMean': sorted_respMean,
            'sorted_respSD': sorted_respSD,
            'ch2xy': ch2xy,
            'grid_shape': None,
            'DimSearchSpace': n_cond,
        }
    else:
        raise ValueError('The dataset type should be 5d_rat, nhp, rat or spinal' )


# ============================================
#      Held-Out / Train Subject Splits
# ============================================

# 5d_rat: TRAIN = rData03, BCI00, rCer1.12 (0, 2, 3); HELD_OUT = rCer1.5, rCer1.14,
# rCer1.15 (1, 4, 5). Subject 6 (5D_step4_noartrej) remains a placeholder, excluded.
HELD_OUT_SUBJECTS = {'rat': [0, 5], 'nhp': [1], 'spinal': [0, 2, 5, 9], '5d_rat': [1, 4, 5]}
TRAIN_SUBJECTS = {'rat': [1, 2, 3, 4], 'nhp': [0, 3], 'spinal': [1, 3, 4, 6, 7, 8, 10], '5d_rat': [0, 2, 3]}
ALL_SUBJECTS = {'rat': [0, 1, 2, 3, 4, 5], 'nhp': [0, 1, 3], 'spinal': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], '5d_rat': [0, 1, 2, 3, 4, 5]}


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
        ...     {'epochs': 100, 'lr': 1e-4, 'aug_pct': 2.5},
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
    for sub in ('optimization', 'optimization/emg_maps', 'results', 'diagnostics'):
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
#         Subject-Oriented Result Cache
# ============================================


def _subject_cache_dir(
    dataset: str,
    subject_idx: int,
    model_type: str,
    emg_idx: int,
    base_dir: str = './output/subjects',
) -> str:
    """Return the per-EMG cache directory for a (dataset, subject, model_type, emg) tuple."""
    return os.path.join(
        base_dir, dataset, f'subject_{subject_idx}', model_type, f'emg_{emg_idx}'
    )


def _cache_hash(cache_params: dict[str, Any]) -> str:
    """Compute a 5-char alphanumeric content hash from ``cache_params``.

    Uses the same MD5-based scheme as :func:`generate_experiment_tag` so the
    hash is deterministic: identical params always produce the same key.

    Args:
        cache_params: Hyperparams dict to hash.  All values must be
            JSON-serialisable.

    Returns:
        5-character hex digest string.
    """
    serialised = json.dumps(cache_params, sort_keys=True, default=str)
    return hashlib.md5(serialised.encode()).hexdigest()[:5]


def load_subject_result(
    dataset: str,
    subject_idx: int,
    emg_idx: int,
    model_type: str,
    cache_params: dict[str, Any],
    base_dir: str = './output/subjects',
) -> Optional[dict]:
    """Return cached result dict if a matching entry exists; else ``None``.

    Layout: ``output/subjects/{dataset}/subject_{idx}/{model_type}/emg_{emg_idx}/``
    contains one ``{hash}.pkl`` + ``{hash}.yaml`` pair per distinct ``cache_params``
    combination.  The hash is a content-addressed key derived from ``cache_params``;
    the YAML is the authoritative source of truth and is verified on load to guard
    against the (astronomically rare) case of a hash collision.

    Multiple ``cache_params`` combinations for the same (dataset, subject, emg,
    model_type) coexist as separate hash-named file pairs — no overwriting occurs
    unless the exact same params are re-saved.

    Args:
        dataset: Dataset identifier, e.g. ``'nhp'`` or ``'rat'``.
        subject_idx: Integer subject index.
        emg_idx: Integer EMG channel index.
        model_type: Model identifier used as subdirectory name, e.g.
            ``'gp'``, ``'vanilla_tabpfn'``, ``'finetuned_tabpfn'``.
        cache_params: Dict of hyperparams that must match for a cache hit.
        base_dir: Root of the subject cache tree.

    Returns:
        Loaded result dict on a cache hit, or ``None`` on a miss.
    """
    cache_dir = _subject_cache_dir(dataset, subject_idx, model_type, emg_idx, base_dir)
    h = _cache_hash(cache_params)
    pkl_path = os.path.join(cache_dir, f'{h}.pkl')
    cfg_path = os.path.join(cache_dir, f'{h}.yaml')
    if not os.path.exists(pkl_path) or not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        stored = yaml.safe_load(f)
    if stored != cache_params:
        return None
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_subject_result(
    result: dict,
    dataset: str,
    subject_idx: int,
    emg_idx: int,
    model_type: str,
    cache_params: dict[str, Any],
    base_dir: str = './output/subjects',
) -> None:
    """Persist a result dict and its cache key config for future lookup.

    Writes two files to the per-EMG cache directory:
    ``{hash}.pkl`` (full result) and ``{hash}.yaml`` (``cache_params``).
    The hash is derived from ``cache_params`` so different hyperparam
    combinations produce distinct file pairs and never overwrite each other.

    Args:
        result: Result dict from any evaluation function.
        dataset: Dataset identifier.
        subject_idx: Integer subject index.
        emg_idx: Integer EMG channel index.
        model_type: Model identifier used as subdirectory name.
        cache_params: Hyperparams dict written to the companion YAML and used
            to derive the content-addressed filename.
        base_dir: Root of the subject cache tree.
    """
    cache_dir = _subject_cache_dir(dataset, subject_idx, model_type, emg_idx, base_dir)
    os.makedirs(cache_dir, exist_ok=True)
    h = _cache_hash(cache_params)
    with open(os.path.join(cache_dir, f'{h}.pkl'), 'wb') as f:
        pickle.dump(result, f)
    with open(os.path.join(cache_dir, f'{h}.yaml'), 'w') as f:
        yaml.dump(cache_params, f, sort_keys=True)
    print(
        f"[SUBJECT CACHE] Saved {dataset}/subject_{subject_idx}"
        f"/{model_type}/emg_{emg_idx}/{h}"
    )


# ============================================
#      Data Augmentation for Fine-Tuning
# ============================================

# Supported topology-preserving augmentation transforms.
# Spatial transforms (h_flip, v_flip, d_flip) operate on MinMax-scaled [0,1]²
# electrode coordinates; y_shift operates on the StandardScaler-normalised response.
_AUG_TRANSFORMS: frozenset[str] = frozenset({'none', 'h_flip', 'v_flip', 'd_flip', 'y_shift'})


def _apply_aug_transform(
    X: np.ndarray,
    y: np.ndarray,
    transform: str,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one topology-preserving augmentation to a (X, y) map pair.

    All transforms preserve the spatial organisation of the EMG response map.
    Coordinate transforms (h_flip, v_flip, d_flip) operate on MinMax-scaled
    [0, 1]² coords and remain in-bounds for any grid shape.  The response
    transform (y_shift) adds a global baseline shift to the standardized y,
    simulating inter-session DC offset without altering electrode ordering.

    Args:
        X: MinMax-scaled coordinates, shape [N, D]. For the 2D coordinate
            transforms (``h_flip``, ``v_flip``, ``d_flip``), D must be 2
            (column 0 = row, column 1 = col), values in [0, 1].
        y: StandardScaler-normalised response values, shape [N].
        transform: One of ``'none'``, ``'h_flip'``, ``'v_flip'``,
            ``'d_flip'``, ``'y_shift'``.
        rng: Random state used by stochastic transforms (``'y_shift'``).
            Deterministic transforms ignore it.

    Returns:
        Tuple ``(X_out, y_out)`` with the same shapes as the inputs.

    Raises:
        ValueError: If ``transform`` is not in :data:`_AUG_TRANSFORMS`, or if
            a 2D-only coordinate transform (``h_flip``, ``v_flip``,
            ``d_flip``) is requested for a non-2D search space (e.g. the
            5-D rat dataset).
    """
    if transform == 'none':
        return X, y
    if transform in ('h_flip', 'v_flip', 'd_flip'):
        if X.shape[1] != 2:
            raise ValueError(
                f"Aug transform {transform!r} is 2D-only (assumes a 2D "
                f"electrode grid), but X has {X.shape[1]} dimensions."
            )
        if transform == 'h_flip':                   # left-right mirror
            X = X.copy(); X[:, 1] = 1.0 - X[:, 1]
            return X, y
        if transform == 'v_flip':                   # top-bottom mirror
            X = X.copy(); X[:, 0] = 1.0 - X[:, 0]
            return X, y
        return X[:, [1, 0]], y                       # d_flip: diagonal / transpose
    if transform == 'y_shift':                  # global baseline shift
        beta = rng.randn() * 0.15               # β ~ N(0, 0.15) in standardised space
        return X, y + beta
    raise ValueError(
        f"Unknown aug transform {transform!r}. Valid: {sorted(_AUG_TRANSFORMS)}"
    )


def augment_maps(
    subject_data: dict,
    emg_idx: int,
    n_augmentations: int = 10,
    seed: int = 42,
    aug_transforms: tuple[str, ...] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate augmented (X, y) training pairs from one EMG channel.

    For each augmentation:

    1. Sample a noisy response map: ``noise ~ N(0, std_map)`` added to
       ``mean_map`` per channel.
    2. Uniformly draw one transform from ``aug_transforms`` and apply it.
       Coordinate transforms flip the MinMax-scaled electrode layout; the
       response transform (``'y_shift'``) adds a global baseline shift.

    Args:
        subject_data: Dict returned by :func:`load_data`.
        emg_idx: Which EMG channel to augment.
        n_augmentations: Number of augmented maps to produce.
        seed: Random seed for reproducibility.
        aug_transforms: Tuple of transform names to sample uniformly from.
            Supported: ``'none'``, ``'h_flip'``, ``'v_flip'``, ``'d_flip'``,
            ``'y_shift'``.  ``None`` → ``('none',)`` (noise-only, default).

    Returns:
        List of ``(X, y)`` tuples — MinMax-scaled (possibly transformed)
        electrode coords and StandardScaler-normalised perturbed response.

    Raises:
        ValueError: If any value in ``aug_transforms`` is not in
            :data:`_AUG_TRANSFORMS`.
    """
    if aug_transforms is None:
        aug_transforms = ('none',)
    unknown = set(aug_transforms) - _AUG_TRANSFORMS
    if unknown:
        raise ValueError(
            f"Unknown aug transforms: {sorted(unknown)}. "
            f"Valid: {sorted(_AUG_TRANSFORMS)}"
        )

    rng = np.random.RandomState(seed)

    coords = subject_data['ch2xy']                            # [nChan, D]
    mean_map = subject_data['sorted_respMean'][:, emg_idx]   # [nChan]
    std_map = subject_data['sorted_respSD'][:, emg_idx]      # [nChan]

    # Drop sites with zero valid reps — their respMean is the np.ma.filled
    # fill value (0.0), not a real measurement, and would bias the scaler.
    if 'sorted_isvalid' in subject_data:
        valid_site_mask = (
            subject_data['sorted_isvalid'][:, emg_idx, :] != 0
        ).any(axis=-1)
        coords = coords[valid_site_mask]
        mean_map = mean_map[valid_site_mask]
        std_map = std_map[valid_site_mask]

    scaler_x = MinMaxScaler()
    X_base = scaler_x.fit_transform(coords)                  # [N_valid, D]

    scaler_y = StandardScaler()
    scaler_y.fit(mean_map.reshape(-1, 1))

    transforms_list = list(aug_transforms)
    augmented_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_augmentations):
        noise = rng.randn(len(mean_map)) * std_map
        y_aug = scaler_y.transform((mean_map + noise).reshape(-1, 1)).ravel()
        chosen = transforms_list[rng.randint(len(transforms_list))]
        X_aug, y_aug = _apply_aug_transform(X_base, y_aug, chosen, rng)
        augmented_pairs.append((X_aug, y_aug))

    return augmented_pairs


def plot_augmented_maps(
    subject_data: dict,
    emg_idx: int,
    dataset_type: str,
    subj_idx: int,
    n_show: int = 6,
    aug_pct: float = 2.5,
    aug_transforms: tuple[str, ...] | None = None,
    seed: int = 42,
) -> None:
    """Visualize the original EMG map alongside augmented versions (debug only).

    Inverse-transforms augmented y values back to the original response scale
    so all maps share the same colorbar for direct comparison.

    Args:
        subject_data: Dict returned by :func:`load_data`.
        emg_idx: Which EMG channel to visualize.
        dataset_type: E.g. ``'nhp'``, ``'rat'``, or ``'5d_rat'`` (used in title only).
        subj_idx: Subject index (used in title only).
        n_show: Number of augmented maps to display (default 6).
        aug_pct: Augmentation percentage; 1.0 = 100% = 10 maps/EMG.
            Total maps generated = ``max(1, round(aug_pct * 10))``.
        aug_transforms: Transforms to sample from (passed to
            :func:`augment_maps`).  ``None`` → noise-only.
        seed: Random seed passed to :func:`augment_maps`.
    """


    from utils.visualization import _to_grid  # local import to avoid cycle

    if subject_data.get('grid_shape') is None:
        print(f"[plot_augmented_maps] No 2D grid_shape for {dataset_type} (e.g. 5D dataset) — skipping.")
        return

    mean_map_full = subject_data['sorted_respMean'][:, emg_idx]   # (nChan,)
    grid_shape = subject_data['grid_shape']
    ch2xy_full = subject_data['ch2xy']

    if 'sorted_isvalid' in subject_data:
        valid_site_mask = (
            subject_data['sorted_isvalid'][:, emg_idx, :] != 0
        ).any(axis=-1)
    else:
        valid_site_mask = np.ones(len(mean_map_full), dtype=bool)

    mean_map = mean_map_full[valid_site_mask]                    # (n_valid,)
    ch2xy = ch2xy_full[valid_site_mask]

    # Generate augmented pairs and inverse-transform y back to response scale
    scaler_y = StandardScaler()
    scaler_y.fit(mean_map.reshape(-1, 1))

    n_augmentations = max(1, round(aug_pct * 10))
    pairs = augment_maps(subject_data, emg_idx,
                         n_augmentations=n_augmentations,
                         seed=seed,
                         aug_transforms=aug_transforms)
    n_show = min(n_show, len(pairs))
    aug_maps = [
        scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()
        for _, y in pairs[:n_show]
    ]

    # Shared color scale across original + all augmented maps
    all_vals = np.concatenate([mean_map] + aug_maps)
    vmin, vmax = all_vals.min(), all_vals.max()
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('lightgrey')
    heatmap_kw = dict(cmap=cmap, vmin=vmin, vmax=vmax,
                      cbar=False, xticklabels=False, yticklabels=False)

    n_total = 1 + n_show
    n_cols = min(4, n_total)
    n_rows = math.ceil(n_total / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    # Original map
    sns.heatmap(_to_grid(mean_map, ch2xy, grid_shape), ax=axes_flat[0], **heatmap_kw)
    axes_flat[0].set_title(f'Original\n{dataset_type} S{subj_idx} EMG{emg_idx}',
                           fontsize=9)

    # Augmented maps
    for i, y_map in enumerate(aug_maps):
        sns.heatmap(_to_grid(y_map, ch2xy, grid_shape), ax=axes_flat[i + 1], **heatmap_kw)
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


def build_finetuning_dataset(
    dataset_type: str,
    subject_indices: list[int] | None = None,
    held_out_emg_idx: int | None = None,
    aug_pct: float = 1.0,
    seed: int = 42,
    aug_transforms: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (X_all, y_all) dataset for fine-tuning TabPFN.

    Iterates over all (subject, EMG) training pairs and generates augmented
    maps via :func:`augment_maps`.  The number of maps per EMG is controlled
    by ``aug_pct``, a percentage relative to a canonical reference of 10 maps
    per EMG channel:

        ``n_aug_per_emg = max(1, round(aug_pct * 10))``

    Percentage → count mapping for common sweep values:

    ============  ================
    ``aug_pct``   maps per EMG
    ============  ================
    0.1  (10 %)   1
    0.2  (20 %)   2
    0.5  (50 %)   5
    0.7  (70 %)   7
    1.0 (100 %)  10
    2.5 (250 %)  25
    ============  ================

    Args:
        dataset_type: ``'rat'``, ``'nhp'``, ``'spinal'``, or ``'5d_rat'``.
        subject_indices: Subject indices to include in training.  Defaults to
            ``TRAIN_SUBJECTS[dataset_type]``.
        held_out_emg_idx: If set, this EMG index is excluded from all subjects
            (intra-EMG holdout).  ``None`` → include all EMGs.
        aug_pct: Augmentation percentage.  1.0 = 100 % = 10 maps per EMG.
        seed: Base random seed; per-EMG seeds are derived as
            ``seed + subj_idx * 100 + emg_idx``.
        aug_transforms: Tuple of transform names passed to
            :func:`augment_maps`.  ``None`` → ``('none',)`` (noise-only).

    Returns:
        ``(X_all, y_all)`` — concatenated MinMax-scaled coordinates
        ``[N, D]`` and StandardScaler-normalised responses ``[N]``.
    """
    if subject_indices is None:
        subject_indices = TRAIN_SUBJECTS[dataset_type]

    n_aug_per_emg = max(1, round(aug_pct * 10))

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for subj_idx in subject_indices:
        data = load_data(dataset_type, subj_idx)
        n_emgs = data['sorted_respMean'].shape[1]

        for emg_idx in range(n_emgs):
            if held_out_emg_idx is not None and emg_idx == held_out_emg_idx:
                continue
            pairs = augment_maps(
                data, emg_idx,
                n_augmentations=n_aug_per_emg,
                seed=seed + subj_idx * 100 + emg_idx,
                aug_transforms=aug_transforms,
            )
            for X, y in pairs:
                X_parts.append(X)
                y_parts.append(y)

    X_all = np.concatenate(X_parts, axis=0)   # [N, D]
    y_all = np.concatenate(y_parts, axis=0)   # [N]
    return X_all, y_all


def preprocess_neural_data(subject_data, emg_idx=0, normalization='pfn'):
    """
    Build the per-EMG (X_pool, y_pool) tensors consumed by the BO loops.

    Sites with zero valid reps at this EMG are dropped entirely — they would
    otherwise enter ``Y_test`` as standardized-zero artefacts (from
    ``np.ma.filled(..., 0.0)`` in :func:`load_data`) and pollute
    ``r2_score(y_test, y_pred)``.  Combined with the no-padding policy in
    :func:`_topographic_metadata`, this guarantees every row of the returned
    tensors corresponds to a real, observable electrode-EMG pair.

    y_pool exposes the full noisy single-trial response bank (one column per
    repetition) so the BO loop can draw one stochastic observation per query,
    matching the real neurostim experimental protocol.  Repetitions flagged
    invalid by ``sorted_isvalid`` are masked to NaN in y_pool; downstream
    samplers must skip NaN draws (see ``bo_loops._draw_valid_rep``).

    Returns:
        X_train: MinMax-scaled coordinates [N_valid, D].
        Y_train: Per-trial responses [N_valid, n_reps], standardized; NaN at
            invalid trials.  Used as ``y_pool``.
        X_test: Same as X_train (passed through for API compatibility).
        Y_test: Per-site ground-truth means [N_valid], standardized in the
            *same* space as Y_train (so ``r2_score(y_test, y_pred)`` and
            regret ``y_test.max() - max(real_values)`` are both unit-consistent).
            To recover raw EMG units, apply ``scaler_y.inverse_transform``.
        scaler_y: Fitted scaler used for Y_train (and Y_test).
    """
    coords = subject_data['ch2xy']                              # [N, D]
    resp_all = subject_data['sorted_resp'][:, emg_idx, :].astype(np.float32).copy()  # [N, n_reps]
    resp_mean = subject_data['sorted_respMean'][:, emg_idx]     # [N]

    # Mask invalid trials with NaN so the BO sampler can redraw past them.
    if 'sorted_isvalid' in subject_data:
        valid_trial_mask = subject_data['sorted_isvalid'][:, emg_idx, :]  # [N, n_reps]
        resp_all[valid_trial_mask == 0] = np.nan
        # A site is usable only if at least one rep is valid; otherwise
        # ``sorted_respMean`` at that site is the np.ma.filled fill value
        # (0.0) — never a real measurement.
        valid_site_mask = (valid_trial_mask != 0).any(axis=-1)            # [N]
    else:
        valid_site_mask = np.ones(resp_all.shape[0], dtype=bool)

    # Drop unobservable sites *before* anything else touches Y_test.
    coords = coords[valid_site_mask]                            # [N_valid, D]
    resp_all = resp_all[valid_site_mask]                        # [N_valid, n_reps]
    resp_mean = resp_mean[valid_site_mask]                      # [N_valid]
    n_channels, n_reps = resp_all.shape

    # Fit y-scaler on valid trials only (NaN propagates through .fit otherwise).
    valid_flat = resp_all[~np.isnan(resp_all)].reshape(-1, 1)
    if valid_flat.size == 0:
        raise RuntimeError(
            f"preprocess_neural_data: no valid trials for emg_idx={emg_idx}."
        )

    if normalization == 'pfn':
        scaler_x = MinMaxScaler()
        X_train = scaler_x.fit_transform(coords)                # [N_valid, D]
        scaler_y = StandardScaler()
    else:
        X_train = coords                                         # [N_valid, D]
        scaler_y = MinMaxScaler()

    scaler_y.fit(valid_flat)
    Y_train = scaler_y.transform(resp_all.reshape(-1, 1)).reshape(n_channels, n_reps)  # [N_valid, n_reps], NaN preserved
    Y_test = scaler_y.transform(resp_mean.reshape(-1, 1)).flatten()                    # [N_valid], same space as Y_train

    return X_train, Y_train, X_train, Y_test, scaler_y


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
        evaluation_type: 'optimization' (only supported value; kept as a
            filename component for back-compat with existing pkl naming).
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
                'median_r2': float(np.median(r2_arr)),
                'std_r2': float(np.std(r2_arr)),
                'n_reps': len(r2_arr),
                'mean_time_s': float(np.mean(res['times'])),
            }

            # Rank-correlation fit metric (robust to flat/non-responsive
            # channels where R² explodes). May be absent in legacy pkls.
            spearman = res.get('spearman')
            if spearman is not None:
                sp_arr = np.asarray(spearman, dtype=float)
                row['mean_spearman'] = float(np.nanmean(sp_arr))
                row['median_spearman'] = float(np.nanmedian(sp_arr))

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
        ``r2_by_subject``, ``regret_with_timing``, etc.
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
        dataset: Dataset type — ``'rat'``, ``'nhp'``, ``'spinal'``, or ``'5d_rat'``.
        result_type: Which pkl type to load.  One of:

            * ``'optimization'`` — ``*_optimization.pkl`` files (dict[str, list[dict]])
            * ``'optimization_budget'`` — ``*_optimization_budget.pkl``
              (DataFrame pkl, columns: Budget|Model|Regret|R2|ID)

        runs_dir: Root directory that contains per-run subdirectories.
        tags: Optional list of 5-char hash suffixes (e.g. ``['32c2b', '15h5p']``).
            When provided only run directories whose suffix matches are loaded.
            ``None`` means load all directories matching the family prefix.

    Returns:
        Concatenated DataFrame.  Schema for ``optimization``:

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
    valid_types = {'optimization', 'optimization_budget'}
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
                # 'optimization': match files that contain f'_{result_type}_'
                # (the evaluation_type component).
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

