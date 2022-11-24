#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:40:22 2021

@author: administrator
"""

# =============================================================================
# Bad channel detection - comparing RANSAC, Bandpower ratio (my approach), and 
# Fickling et al., 2019 method 
# =============================================================================

from scipy.fft import fft, fftfreq
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import seaborn as sns
import os 
import yasa
import mne
from autoreject import Ransac
from autoreject.utils import interpolate_bads
from os import chdir as cd
from os import listdir
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, preprocess_sleep_data, compare_hypnograms, check_match_data_hypno_elements, 
                                                     label_artifacts, load_preprocessed_data, Data_SW)

#%%
path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded/'
def bad_channel_detection(path):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    for i, data_files in enumerate(files_list):
        #pass
        Data = pickle.load(open(data_files,"rb")) 
     
        # epoched data
        _, epoched_data = yasa.sliding_window(Data.data.T, sf=Data.sfreq, window=5)
        
        ## RANSAC method
        # mne-ize 
        info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)
        epochs = mne.EpochsArray(epoched_data/1e6, info, tmin = 0, baseline=(None)) 
        epochs.set_montage(mne.channels.make_standard_montage('standard_1005'))
    
        picks = mne.pick_types(epochs.info, eeg=True, stim=False, eog=False,
                               include=[], exclude=[])
        ransac = Ransac(verbose='progressbar', picks=picks, n_jobs=-1)
        epochs_clean = ransac.fit_transform(epochs)
        
        print(f'The following dataset: {data_files.split("/")[-1].split(".")[0]} has the following bad channels: {ransac.bad_chs_} ')
        print(f'The aforementioned dataset has a rectified 99th perentile amplitude of: {np.percentile(abs(Data.data[:,[Data.chans.index("M1"), Data.chans.index("M2")]].mean(-1)), 99)}')
        
        #del epochs, Data, epoched_data, epochs_clean, ransac
        
        evoked = epochs.average()
        evoked_clean = epochs_clean.average()
        
        # detected and repaired channels
        evoked.info['bads'] = ransac.bad_chs_
        evoked_clean.info['bads'] = ransac.bad_chs_
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        
        for ax in axes:
            ax.tick_params(axis='x', which='both', bottom='off', top='off')
            ax.tick_params(axis='y', which='both', left='off', right='off')
        
        evoked.plot(exclude=[], axes=axes[0], show=False)
        axes[0].set_title('Before RANSAC')
        evoked_clean.plot(exclude=[], axes=axes[1])
        axes[1].set_title('After RANSAC')
        fig.tight_layout()
        print('\n'.join(ransac.bad_chs_))
        
        ##
        ch_names = [epochs.ch_names[ii] for ii in ransac.picks][0]
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.imshow(ransac.bad_log, cmap='Reds',
                  interpolation='nearest')
        ax.grid(False)
        ax.set_xlabel('Sensors')
        ax.set_ylabel('Trials')
        plt.setp(ax, xticks=range(7, len(ransac.picks), 10),
                 xticklabels=ch_names)
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.tick_params(axis=u'both', which=u'both', length=0)
        fig.tight_layout(rect=[None, None, None, 1.1])
        plt.show()

#%%


## function below is in testing mode
def bad_channels_isoforest_ransac(data_filt, sf, ch_names, ch_types):
    ### RANSAC
    ## apply RANSAC algorithm to find bad channels and subsequently interolate them in later step
    _, epochs = yasa.sliding_window(data_filt.T, sf=sf, window=2)
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
    norm_kurtosis = [stats.kurtosis(data_filt[:,i], axis=-1) for i in eeg_index]
    
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
  
#%%
# =============================================================================
# The below EEG quality assessment is almost entirely based on Dr. Tianlu Wang's
# code implementation of Fickling et al., 2019
# =============================================================================

"""

Summary of approach:
    
Sliding window approach, where windows of 1 second in length are repeatedly, advancing by one frame/sample at a time. 
The EQI values for each 1 second window are compared to a normative database of artifact-free “clean” EEG.
The percentage of “clean” data (frames with individual score equal to 0 or summed score less than 3) over the duration 
of the scan then represents how sustained the noise is throughout the recording.

1. ASSAS - Average Single-Sided Amplitude Spectrum (1-50Hz range) - The absolute value of the FFT is then averaged across
the frequency range of 1-50Hz.
2. LN - Line Noise - Average Single-Sided Amplitude Spectrum (59-61Hz range) we compute the average single sided amplitude 
spectrum of the signal -> 49-51 Hz in Europe
3. RMS - The root-mean-square (RMS) amplitude of the EEG signal is a general measure of the magnitude of the signal 
throughout the window, irrespective of frequency. x_rms = sqrt(sum(abs(x)**2)/N)
4. MG - The maximum gradient of the EEG signal is the largest difference between all adjacent samples within the window. 
MG = max(x(n)-x(n-1))
5. ZCR - The Zero-Crossing Rate is the rate at which the signal changes signs from positive to negative - 
ZCR = sum( sgn(x(n) - sgn(x(n-1)) )/N
6. Kurtosis is a standard statistical measure of the heaviness of the tails of a distribution of samples, i.e. 
it indicates how likely the sample is to contain an outlier. 
K = sum( (x(n) - mean(x))**4 )/N / (sum( (x(n) - mean(x))**2 )/N)**2

"""

def qc_segment(xch,sf,window = 2):
    """
    Divide into segments of n s according to sampling frequency, selecting the middle part of the data
    ----------
    x : eeg data (1 x timepoints)
    sf : sampling rate
    window : length of epochs in seconds

    Returns
    -------
    xs : eeg data in segments x timepoints .

    """
    
    pps = sf*window
    nseg      = int(len(xch)/pps)    
    remainder = len(xch) - nseg*pps
    if remainder>0:
        part2     = int(remainder/2)
        part1     = remainder - part2
        if part2 == 0: xs = np.reshape(xch[part1:],(nseg, pps))
        else:          xs = np.reshape(xch[part1:-part2],(nseg, pps))
    else:
        xs = np.reshape(xch,(nseg, pps))
    return xs

def qc_assas(xs, frange, fs = 1000):
    # Calculate FFT
    yf   = fft(xs)
    bins = fftfreq(len(xs), 1/fs)
    # Check fft:
    # plt.plot(bins[:int(fs/2)],yf[:int(fs/2)])
    assas = np.mean(np.abs(yf[np.argwhere(bins == frange[0])[0][0]:np.argwhere(bins == frange[1])[0][0]+1]))
    return assas
    
def qc_rms(xs):
    # Calculate root mean square of the EEG (NB after de-meaning?)
    rms = (np.sum(np.abs(xs)**2)/len(xs))**0.5
    return rms

def qc_mg(xs):
    # Calculate maximum gradient
    mg = max(xs[1:]-xs[:-1])
    return mg

def qc_zcr(xs):
    # Calculate zero-crossing rate
    zcr = np.sum(np.abs( np.sign(xs[1:]) - np.sign(xs[:-1]) ))/len(xs)
    return zcr

def qc_kurt(xs):
    # Calculate kurtosis
    # kurt = np.sum((xs - np.mean(xs))**4) / np.sum((xs - np.mean(xs))**2)**2
    kurt = stats.kurtosis(xs, axis=-1)
    return kurt

def qc_calcEQI(data2, fs, frange=(1,45)):
    """
    Combines 6 quality index functions above and returns qc_arr matrix in the shape of
    number of measures x channels x segments
    ----- input -----
    eeg : channels x timepoints
    fs  : sampling frequency

    """
    nch    = data2.shape[0]
    nsegs  = qc_segment(data2[0,:],fs).shape[0]
    qc_arr = np.zeros((6,nch,nsegs))
    
    # Loop over channels and segments and calculate quality indices
    for ich, xch in enumerate(data2):
        xsegs = qc_segment(xch,fs,window=2)
        for idxs, xs in enumerate(xsegs):
            # 1. Average Single-Sided Amplitude Spectrum in range given range
            qc_arr[0,ich,idxs] = qc_assas(xs, frange, fs)
            # 2. Average Single-Sided Amplitude Spectrum in range 49-51 Hz
            qc_arr[1,ich,idxs] = qc_assas(xs, [49,51], fs)
            # 3. root mean square
            qc_arr[2,ich,idxs] = qc_rms(xs)
            # 4. maximum gradient
            qc_arr[3,ich,idxs] = qc_mg(xs)
            # 5. Zero-Crossing Rate 
            qc_arr[4,ich,idxs] = qc_zcr(xs)
            # 6. Kurtosis 
            qc_arr[5,ich,idxs] = qc_kurt(xs)
        
    return qc_arr