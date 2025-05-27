# -*- coding: utf-8 -*-
"""
Core feature extraction functions for sleepstim project.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats # For welch, stats.iqr, stats.skew, stats.kurtosis
import antropy as ant
import yasa
import fooof
import neurokit2 as nk
from neurodsp.timefrequency import amp_by_time # Used in feature_extraction

# Import bandpower from the new dsp module
from sleepstim.core.dsp import bandpower

# Copied functions

def lziv(x):
    """Binarize the EEG signal and calculate the Lempel-Ziv complexity.
    """
    # antropy import is at the top level now
    return ant.lziv_complexity(x > np.median(x), normalize=True)

def feature_extraction(epochs, sf, line_noise_freq=(59, 61),
                       ch_names=['C3', 'C4', 'M1', 'M2', 'EOG', 'EMG']):
    """
    **1. Features extraction**

    For each 30-seconds epoch and each channel, the following features are calculated:

    * Standard deviation
    * Interquartile range
    * Skewness and kurtosis
    * Number of zero crossings
    * Hjorth mobility and complexity
    * Absolute total power in the 0.4-30 Hz band.
    * Relative power in the main frequency bands (for EEG and EOG only)
    * Power ratios (e.g. delta / beta)
    * Permutation entropy
    * Higuchi and Petrosian fractal dimension

    In addition, the algorithm also calculates a smoothed and normalized version of these features.
    Specifically, a 7.5 min centered triangular-weighted rolling average and a 2 min past rolling
    average are applied. The resulting smoothed features are then normalized using a robust
    z-score.
    """

    bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
           (8, 12, 'Alpha'), (12, 16, 'Sigma'), 
           (16, 30, 'Beta'), (line_noise_freq[0], line_noise_freq[1],
                              'Line noise')]
    dfs = []
    for idx, ch in enumerate(ch_names):     
        # Calculate standard descriptive statistics
        hmob, hcomp = ant.hjorth_params(epochs[:,idx,:], axis=-1)
    
        feat = {
            "std": np.std(epochs[:,idx,:], ddof=1, axis=-1),
            "iqr": stats.iqr(epochs[:,idx,:], rng=(25, 75), axis=-1),
            "skew": stats.skew(epochs[:,idx,:], axis=-1),
            "kurt": stats.kurtosis(epochs[:,idx,:], axis=-1),
            "nzc": ant.num_zerocross(epochs[:,idx,:], axis=-1),
            "hmob": hmob,
            "hcomp": hcomp,
            }
    
        # Calculate spectral power features using bandpower from dsp module
        # Assuming bandpower function expects (n_samples) or (n_chans, n_samples)
        # If epochs is (n_epochs, n_chans, n_samples), then epochs[:, idx, :] is (n_epochs, n_samples)
        # The original bandpower in sleep_funs took (epochs, fs, bands, relative), 
        # and yasa.bandpower_from_psd_ndarray expects (psd_array, freqs, bands, relative)
        # The bandpower function moved to dsp.py computes welch and then calls yasa.bandpower_from_psd_ndarray
        
        # The bandpower function in dsp.py takes `epochs` as (n_epochs, n_chans, n_samples) or (n_chans, n_samples)
        # Here, we are iterating per channel, so `epochs[:, idx, :]` is (n_epochs, n_samples)
        # The dsp.bandpower function expects the last axis to be samples.
        power = bandpower(epochs[:, idx, :], fs=sf, bands=bands, relative=True) 
        
        # append power values to feature space
        # power should now be (n_bands, n_epochs) if bandpower was called with (n_epochs, n_samples)
        # or (n_bands) if bandpower was called with (n_samples) and averaged over epochs implicitly
        # Based on dsp.bandpower, it returns (n_bands, n_epochs) if input is 2D (like (n_epochs, n_samples))
        # and psd from welch is (n_epochs, n_freqs).
        # So, power[j] would be power for band j across all epochs.
        for j, (_, _, b) in enumerate(bands):
            feat[b] = power[j, :] # Assuming power is [n_bands, n_epochs]
            
        # extract and add spectral fit
        sp_exp = []
        sp_exp_fooof = []
        for i in range(epochs.shape[0]): # Iterate over epochs for this channel
            epoch_data_ch = epochs[i, idx, :]
            try:
                ## IRASA-ing
                s = yasa.irasa(epoch_data_ch, sf=sf, ch_names=[ch] if isinstance(ch, str) else ch, band=(1, 30),
                               return_fit=True, win_sec=2/1, verbose='error')[-1]['Slope'].to_numpy()[0] 
                
                ## Fooof-ing
                nperseg = (2 / 1) * sf # Assuming min freq for FOOOF is 1 Hz for this nperseg

                freqs_fooof, psd_fooof = signal.welch(epoch_data_ch, sf, nperseg=int(nperseg), average='median')
                               
                fg = fooof.FOOOF(max_n_peaks=5, verbose=False) # verbose=False
                fg.fit(freqs_fooof, psd_fooof, freq_range=(1, 30)) 
                
                s_fooof = fg.get_params('aperiodic_params', 'exponent')
                
            except Exception: # Catching generic Exception is broad, but okay for now
                s = np.nan
                s_fooof = np.nan
                
            sp_exp.append(s)
            sp_exp_fooof.append(s_fooof)
            
        feat["Spectral_exp_irasa"] = np.array(sp_exp)
        feat["Spectral_exp_fooof"] = np.array(sp_exp_fooof)
            
        # Add power ratios for EEG
        delta = feat["Delta"] # This will be an array of delta powers for each epoch
        feat["DT_ratio"] = delta / feat["Theta"]
        feat["DS_ratio"] = delta / feat["Sigma"]
        feat["DB_ratio"] = delta / feat["Beta"]
        feat["AT_ratio"] = feat["Alpha"] / feat["Theta"]
    
        # Calculate entropy and fractal dimension features
        feat["Perm"] = np.apply_along_axis(ant.perm_entropy, axis=-1, arr=epochs[:, idx, :],
                                           normalize=True)
        feat["Higuchi"] = np.apply_along_axis(lambda x_h: ant.higuchi_fd(x_h.flatten(), kmax=10), 
                                              axis=-1, arr=epochs[:, idx, :])
        feat["Petrosian"] = ant.petrosian_fd(epochs[:, idx, :], axis=-1)
        
        # Analytical transformation 
        amp_epochs_ch = np.array([amp_by_time(epochs[j, idx, :], fs=sf) for j in range(epochs.shape[0])])

        # Apply Lempel-Ziv
        feat['lziv'] = np.apply_along_axis(lziv, axis=-1, arr=epochs[:, idx, :])
        feat['lziv_power'] = np.apply_along_axis(lziv, axis=-1, arr=amp_epochs_ch)

        # Create a pandas DataFrame from the dictionary
        df_features = pd.DataFrame.from_dict(feat) # Each value in feat dict is now a 1D array (n_epochs)
        df_features['epoch'] = np.arange(epochs.shape[0]) # Add epoch numbers
        df_features['chan'] = ch
        
        dfs.append(df_features)
        
    df = pd.concat(dfs).reset_index(drop=True)
    
    return df

def ecg_feature_extraction(ecg_epochs, sf, stages=None, method='neurokit2'):
    """
    **1. ECG/HRV features extraction**
    For each 30-seconds epoch and each channel, the following features are calculated:
    see: https://neuropsychology.github.io/NeuroKit/functions/hrv.html#
    """
    if method=='neurokit2':
        cols = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2',
                'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD', 'HRV_SDSD',
                'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN',
                'HRV_IQRNN', 'HRV_Prc20NN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20',
                'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN', 'HRV_ULF', 'HRV_VLF',
                'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn',
                'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S', 'HRV_CSI',
                'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS',
                'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a',
                'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a',
                'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_DFA_alpha1',
                'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
                'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
                'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry',
                'HRV_MFDFA_alpha1_Fluctuation', 'HRV_MFDFA_alpha1_Increment',
                'HRV_ApEn', 'HRV_SampEn', 'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn',
                'HRV_CMSEn', 'HRV_RCMSEn', 'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
        hrv_dfs = []
        # Iterate through epoch data
        # Assuming ecg_epochs is (n_epochs, n_samples) if single channel ECG, or (n_epochs, 1, n_samples)
        # If ecg_epochs is (n_epochs, n_chans_ecg, n_samples_ecg), take the first ECG channel:
        if ecg_epochs.ndim == 3:
            ecg_epochs_proc = ecg_epochs[:, 0, :] 
        else: # Already (n_epochs, n_samples)
            ecg_epochs_proc = ecg_epochs

        for idx, epoch_data in enumerate(ecg_epochs_proc):
            hrv = pd.DataFrame(np.nan, index=[0], columns=cols) # Initialize with NaNs
            try:
                peaks, _ = nk.ecg_peaks(epoch_data, sampling_rate=sf)
                if len(peaks['ECG_R_Peaks']) > 1: # Need at least 2 peaks for HRV
                    hrv_temp = nk.hrv(peaks, sampling_rate=sf, show=False) # show=False
                    # Ensure only columns that exist in hrv_temp are assigned
                    valid_cols = [col for col in cols if col in hrv_temp.columns]
                    hrv[valid_cols] = hrv_temp[valid_cols].iloc[0]
            except Exception as e:
                # print(f"HRV calculation failed for epoch {idx}: {e}")
                pass # hrv remains NaNs
            
            hrv['epoch'] = idx
            hrv_dfs.append(hrv)           
                
        hrv_df = pd.concat(hrv_dfs).reset_index(drop=True)
        
    elif method=='sleepecg':
        from sleepecg import SleepRecord, extract_features, detect_heartbeats
        
        hbeats = []
        # Assuming ecg_epochs is (n_epochs, n_chans_ecg, n_samples_ecg)
        if ecg_epochs.ndim == 3:
            ecg_epochs_proc = ecg_epochs[:, 0, :]
        else:
            ecg_epochs_proc = ecg_epochs

        for idx, ep_data in enumerate(ecg_epochs_proc):
            hbeat_times = detect_heartbeats(ep_data, sf) 
            # Adjust heartbeat times to be relative to the start of the recording, not just the epoch
            hbeat_times_global = hbeat_times + (idx * ecg_epochs_proc.shape[1]) # Assumes epochs are contiguous
            hbeats.append(hbeat_times_global)
            
        heartbeat_times_concat = np.concatenate(hbeats) / sf   
        
        if stages is None:
            raise ValueError("Sleep stages must be provided for sleepecg method.")

        rec = SleepRecord(
            sleep_stages=stages, # stages should be 1D array matching number of epochs
            sleep_stage_duration=ecg_epochs_proc.shape[1]/sf, # duration of each epoch in seconds
            heartbeat_times=heartbeat_times_concat,
        )
        
        # This part might need adjustment based on how sleepecg handles feature extraction
        # across multiple concatenated epochs vs. per-epoch features.
        # The original code implies per-epoch HRV, so this might need to be adapted.
        features_extracted, stages_out, feature_ids = extract_features(
            [rec], # Expects a list of SleepRecord objects
            lookback=0, # Adjust as needed for features dependent on previous epochs
            lookforward=ecg_epochs_proc.shape[1]/sf 
        )
        # feature_ids might not align directly with nk.hrv columns. Manual mapping or selection needed.
        # For now, creating a placeholder DataFrame
        if features_extracted:
            hrv_df = pd.DataFrame(features_extracted[0], columns=feature_ids)
            # This line was problematic, stage array should be aligned with epochs
            # hrv_df['stage'] = stages_out.reshape(-1,1).astype(int)
            hrv_df['epoch'] = np.arange(len(hrv_df)) # Add epoch numbers
        else:
            hrv_df = pd.DataFrame(columns=feature_ids + ['epoch'])


    # Apply to all 
    # drop columns with too many nans 
    if not hrv_df.empty:
        null_counts = hrv_df.isnull().sum()
        null_percentages = null_counts / len(hrv_df)
        columns_to_drop = null_percentages[null_percentages > .10].index.tolist()
        hrv_df.drop(columns=columns_to_drop, axis=1, inplace=True)
        
    return hrv_df
