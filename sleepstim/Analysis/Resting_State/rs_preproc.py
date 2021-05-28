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
import scipy.signal as signal
from statsmodels.api import tsa
import statsmodels
from PCIst.PCIst import pci_st
from meegkit.detrend import detrend
import pyprep


class Resting_State_Data_Struct:
    def __init__(self, data_eyes_open, data_eyes_close, sf, eego_times, 
                 ch_names, ch_types, eeg_index, emg_index, ecg_index):
        self.data_eyes_open = data_eyes_open
        self.data_eyes_close = data_eyes_close
        self.sf = sf
        self.times = eego_times - eego_times[0]
        self.ch_names = ch_names
        self.ch_types = ch_types
        self.eeg_index = eeg_index
        self.emg_index = emg_index
        self.ecg_index = ecg_index
        self.bad_chans = []
        print('Creating Data Structure with %s data, of length %s minutes, containing %s channels'
              % (self.data.dtype, round(data.shape[0]/data.sf/60, 2), data.shape[1]))

#%%
files = '/media/administrator/data/Study_1_data/Resting_state_data/RVQL2MRD/cond_0_resting_state_pre_R001.xdf'

def _pre_process_rs_data(files, csd = True):
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
    
    ## function below is in testing mode, instead use PREP pipeline below...
    def bad_channels_isoforest_ransac(data, sf, ch_names, ch_types):
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
        
        # return bads if they were detected in both methods
        bads = [element for element in list(df_chan[good==0].index) if element in ransac.bad_chs_]
        
        return bads
    
    ## PREP pipeline
    ## bad channel detection, interpolation and subsequent robust rereferencing 
    mne_info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)
    raw = mne.io.RawArray(data_filt.T/1e6, mne_info) 
    raw = raw.set_montage(mne.channels.make_standard_montage('standard_1005')) 

    prep_params = {"ref_chs": ch_names[0:63],
                   "reref_chs": ch_names[0:63]
                   }

    raw_ref = pyprep.reference.Reference(raw, prep_params)
    raw_ref.perform_reference()
    
    ## Perform ICA to remove eye blinks/muscle artifacts
    # Calculate ICA
    print('\n***** Perform ICA...\n')
    ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(raw_ref.raw)
    
    # Check and remove components
    ica.plot_sources(raw_ref.raw, show_scrollbars=True); ica.plot_components()
    # ica.plot_properties(raw_ref.raw)
    ica.exclude = [int(x) for x in input('Enter components to remove: ').split()]
    ica.plot_overlay(raw_ref.raw, exclude=ica.exclude)

    # Apply ICA
    fit_ica = ica.apply(raw_ref.raw)
  
    ##  Extract data and compute Surface Laplacian transform to reduce affect of volume conduction
    if csd:
        print(f'Ignoring status of bad channels: {fit_ica.info["bads"]} for the purposes of computing CSD. ')
        fit_ica.info['bads'] = []
        data_filt_filt = mne.preprocessing.compute_current_source_density(fit_ica).get_data().T*1e3
    else:
        data_filt_filt = fit_ica.get_data().T*1e6    
    
    # recombine data once more...
    data_filt_filt = np.append(data_filt_filt, data_filt[:, emg_index + ecg_index], axis=1)

    ## Create eyes open/eyes closed epochs based on corrected timestamps 
    # synchronize clocks for later epoching
    eyes_open_times = [np.argmin(np.abs(eego_times - ts)) for ts in eyes_open_ts]
    eyes_close_times = [np.argmin(np.abs(eego_times - ts)) for ts in eyes_close_ts]
    # index eyes open/eyes closed times
    eyes_open_idx, _  = yasa.get_centered_indices(data_filt_filt[:,0], np.asarray(eyes_open_times), 
                                                 npts_before = 0, npts_after = sf*2.)
    eyes_close_idx, _  = yasa.get_centered_indices(data_filt_filt[:,0], np.asarray(eyes_close_times), 
                                                 npts_before = 0, npts_after = sf*2.)
    # create epochs
    data_eyes_open = np.swapaxes(data_filt_filt[eyes_open_idx, :],1,2)
    data_eyes_close = np.swapaxes(data_filt_filt[eyes_close_idx, :],1,2)

    # create data objects
    epoched_data = Data_Struct(data_eyes_open, data_eyes_close, sf, eego_times, 
                               ch_names, ch_types, eeg_index, emg_index, ecg_index)
  
    return epoched_data


#%%
# =============================================================================
# Compute measures of connectivity 
# =============================================================================

## Phase Slope Index 
psi, _, _, _, _ = mne.connectivity.phase_slope_index(data_eyes_open[:, eeg_index], mode='multitaper', 
                                                     sfreq=1000, fmin=8, fmax=12)

## Envelope correlation
env_corr = mne.connectivity.envelope_correlation(data_eyes_open[:, eeg_index], combine='mean', orthogonalize='pairwise',
                                                 log=False, absolute=True, verbose=None)

## Spectral connectivity includes - Preferable to use a seed region, e.g., Motor Cortex (see MNE documentation): 
    # Phase Locking Value (PLV)
    # Phase Lag Index (PLI) / Weighted PLI 
    # Pairwise Phase Consistency (PPC), an unbiased estimator of squared PLV 
    # Coherence
    # Imaginary Coherence 
    # Coherency 
    # etc...
