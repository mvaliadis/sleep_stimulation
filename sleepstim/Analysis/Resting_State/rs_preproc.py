#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:06:12 2021

@author: administrator
"""

from sleepstim.sleep_funs import (bfr_butter_filt, bandpower, unravel_hypnogram_visbrain, 
                                  downsample_scaled, load_xdf, channel_parser, thresholdcrossings,
                                  hjorth_complexity, hjorth_mobility)                        
from sleepstim.Analysis.snr import snr_spectrum
import numpy as np
import matplotlib.pyplot as plt
import yasa 
import mne
from autoreject import Ransac
import logging 
from sklearn.ensemble import IsolationForest
from sklearn.covariance import ShrunkCovariance
import pandas as pd 
from collections import OrderedDict
import scipy.stats as stats
from statsmodels.api import tsa
import statsmodels
from PCIst.PCIst import pci_st

#%%
files = '/media/administrator/data/Study_1_data/Resting_state_data/RVQL2MRD/cond_0_resting_state_pre_R001.xdf'

def _pre_process_rs_data(files, reference='surface_laplacian'):
    ## load and parse data
    stream_dict = load_xdf(files) 
    data = stream_dict['eego']['time_series']
    breath_data = stream_dict['GDX-RB_0K1000F2']['time_series'].squeeze()
    breath_times = stream_dict['GDX-RB_0K1000F2']['time_stamps'].squeeze()
    eego_times = stream_dict['eego']['time_stamps']
    marker = stream_dict['reiz-marker']
    eyes_open_ts = [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if 
                  marker['time_series'][ix][0] == 'augen_auf']
    eyes_close_ts = [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if 
                   marker['time_series'][ix][0] == 'augen_zu']
    info = stream_dict['eego']['info'] 
    ch_names, ch_types = channel_parser(info, data)
    sf = int(float(info['nominal_srate'][0]))
    ch_names, ch_types = channel_parser(info, data) 
    # select only non empty channel data
    data = data[:,np.asarray(ch_types) != 'misc']*1e6
    
    ## channel indices
    ch_names = [ch_names[i] for i in [j for j, x in enumerate(ch_types) if x != "misc"]]
    ch_types = [ch_types[i] for i in [j for j, x in enumerate(ch_types) if x != "misc"]]
    eeg_index = [i for i, x in enumerate(ch_types) if x == "eeg"]  
    emg_index = [i for i, x in enumerate(ch_types) if x == "emg"]  
    ecg_index = [i for i, x in enumerate(ch_types) if x == "ecg"] 
    
    ## apply bandpass filter  
    data_eeg = mne.filter.filter_data(data[:,eeg_index].astype(np.float64).T, sfreq=sf, l_freq=1, h_freq=45).T
    data_emg = mne.filter.filter_data(data[:,emg_index].astype(np.float64).T, sfreq=sf, l_freq=20, h_freq=200).T
    data_ecg = mne.filter.filter_data(data[:,ecg_index].astype(np.float64).T, sfreq=sf, l_freq=0.3, h_freq=70).T
    data = np.concatenate([data_eeg, data_emg, data_ecg], axis=-1)
    
    ## apply notch filter to combined data
    data_filt = mne.filter.notch_filter(data.T, Fs=sf, method='spectrum_fit', freqs=np.arange(50,50*4+1,50), 
                                        mt_bandwidth=2, p_value=0.01, filter_length='10s').T
    
    ## synchronize clocks for later epoching
    eyes_open_times = [np.argmin(np.abs(eego_times - ts)) for ts in eyes_open_ts]
    eyes_close_times = [np.argmin(np.abs(eego_times - ts)) for ts in eyes_close_ts]
    
    #%%
    # =============================================================================
    # Bad channel detection
    # =============================================================================
    ### RANSAC
    ## apply RANSAC algorithm to find bad channels and subsequently interolate them in later step
    _, epochs = yasa.sliding_window(data_filt.T, sf=1000., window=2)
    mne_info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)
    mne_epochs = mne.EpochsArray(epochs, mne_info, tmin = 0, 
                                 baseline=(None), proj=False) 
    mne_epochs.set_montage(mne.channels.make_standard_montage('standard_1005'))    
    picks = mne.pick_types(mne_epochs.info, eeg=True)
    ransac = Ransac(verbose=False, picks=picks, n_jobs=-1)
    mne_epochs_clean = ransac.fit_transform(mne_epochs)
    logging.warning(f'The RANSAC algorithm detected the following as bad channels: {ransac.bad_chs_}!')
    
    ### Isolation Forest
    ## Create dataframe with outlier detection variables
    # mean rectified amplitude 
    MeanAbsAmplitude = np.abs(data_filt[:,eeg_index]).mean(0)
    
    # mean zscore recifified amplitude 
    MeanZscoreAmplitude = np.abs(stats.zscore(data_filt[:,eeg_index])).mean(0) # more a metric of within channel variability
    
    # Normalized kurtosis
    norm_kurtosis = [stats.kurtosis(data_filt[:,i], axis=-1)for i in eeg_index]
    
    # PSD in bands of interest
    bp = yasa.bandpower(data_filt[:,eeg_index].T, sf=sf, ch_names=ch_names[0:len(eeg_index)], win_sec=2, relative=True, 
                        bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'), 
                               (30, 45, 'Gamma'), (48, 52, 'Line Noise')], 
                        kwargs_welch={'average': 'median', 'window': 'hamming'})
    
    # Signal Complexity
    h_complex = [hjorth_complexity(data_filt[:,i]) for i in eeg_index] # weird result
    
    # Cross-correlation
    norm_cov = np.corrcoef(data_filt[:,eeg_index].T)
    # regularize normalized covariance matrix
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=norm_cov.mean(0),
                                cov=norm_cov,
                                size=500)
    cov = ShrunkCovariance().fit(X)
    
    # Autocorrelation
    auto_corr = [tsa.stattools.acf(data_filt[:,i].squeeze(), fft=True, nlags=100).mean() for i in eeg_index]
    # plot autocorrelation 
    # statsmodels.api.graphics.tsa.plot_acf(data_filt[:,0:20].mean(1).squeeze(), lags=100)
    
    # SNR based on PSD
    psds, freqs = mne.time_frequency.psd_array_welch(data_filt[:,eeg_index].T,
                                                     sfreq=sf, n_fft=int(sf * 2),
                                                     n_overlap=0, fmin=1, fmax=100,
                                                     window='boxcar', verbose=False)
    
    snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3,
                        noise_skip_neighbor_freqs=1)
    alpha_freqs, ln_freqs = np.logical_and(freqs > 8, freqs < 12), np.logical_and(freqs > 48, freqs < 52)
    snrs_ratio = snrs[:, ln_freqs].mean(1)/ snrs[:, alpha_freqs].mean(1)
    
    # Create a dictionary of features
    outlier_params = OrderedDict({
        'MeanAbsAmplitude': MeanAbsAmplitude,
        # 'MeanZscoreAmplitude': MeanZscoreAmplitude, 
        'NormalizedKurtosis': norm_kurtosis,
        'LineNoisePSD': bp['Line Noise']/bp['Alpha'],
        # 'HjorthComplexity': h_complex, 
        # 'NormalizedMeanCrossCov': norm_cov.mean(0),
        'RegularizedMeanCrossCov': cov.covariance_.mean(0),
        'MeanAutoCorrelation': auto_corr, 
        'PSDbasedSNR' : snrs_ratio
        })
    
    # Convert to dataframe, keeping only good events
    df_chan = pd.DataFrame(outlier_params)
    
    ## Isolation forest for bad channel detection
    ilf = IsolationForest(contamination='auto', max_samples='auto',
                          verbose=0, random_state=42)
    good = ilf.fit_predict(df_chan)
    good[good == -1] = 0
    logging.warning(f'Isolation forest detected the following as bad channels: {list(df_chan[good==0].index)}!')
                  
    ## Common bad channels for interpolation
    bads = [element for element in list(df_chan[good==0].index) if element in ransac.bad_chs_]
    if 'M1' in bads:
        bads.remove('M1')
    if 'M2' in bads:
        bads.remove('M2')
        
    #%%

    ## rereference data
    if reference=='surface laplacian':    
        # ignore non-EEG channels! 
        raw.pick_types(eeg=True)
        EEG = mne.preprocessing.compute_current_source_density(raw).get_data()
    elif reference == 'common average':
        ref_data = EEG[..., :].mean(-1, keepdims=True)
        EEG -= ref_data 
        
    ## Create eyes open/eyes closed epochs based on corrected timestamps 
    eyes_open_idx, _  = yasa.get_centered_indices(data_filt[:,0], np.asarray(eyes_open_times), 
                                                 npts_before = 0, npts_after = sf*30.)
    eyes_close_idx, _  = yasa.get_centered_indices(data_filt[:,0], np.asarray(eyes_close_times), 
                                                 npts_before = 0, npts_after = sf*30.)
    
    data_eyes_open = np.swapaxes(data_filt[eyes_open_idx, :],1,2)
    data_eyes_close = np.swapaxes(data_filt[eyes_close_idx, :],1,2)

        

