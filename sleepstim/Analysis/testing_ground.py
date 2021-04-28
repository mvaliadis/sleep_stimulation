#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:47:28 2021

@author: administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import pandas as pd
import pyxdf
import yasa 
import mne
import pingouin as pg
import logging
import joblib
import time
import wonambi
import seaborn as sns
from scipy.signal import welch, butter, filtfilt, hilbert 
from scipy.stats import zscore
from scipy.fftpack import next_fast_len
from scipy.special import erf
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from os import chdir as cd
from os import listdir
import os
from autoreject import Ransac
from autoreject.utils import interpolate_bads
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, preprocess_sleep_data, compare_hypnograms, check_match_data_hypno_elements, 
                                                     label_artifacts, load_preprocessed_data, save_preprocess_sleep_data, Data_SW)
from sleepstim.sleep_funs import (bfr_butter_filt, bandpower, unravel_hypnogram_visbrain, 
                        downsample_scaled, load_xdf, channel_parser, thresholdcrossings, plot_confusion_matrix)

sns.set(style='darkgrid', font_scale=1.2)

#%%

path = '/media/administrator/data/Somno_test_data'
files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
for i, files in enumerate(files_list):
    print(i, files)
    data = mne.io.read_raw_edf(files, eog = ('EOG1','EOG2','EOG1:A2','EOG2:A1','EOG2:A2'), preload = True)
    
    eeg_chans = list(['Fp1','Fpz','Fp2','F3','Fz','F4','F7','F8','C3','Cz','C4',
                      'T3','T4','T5','T6','O1','Oz','O2','P3','Pz','P4','A1','A2'])
    eog_chans = list(['EOG1:A2','EOG2:A2'])
    emg_chans = list(['EMG1', 'EMG2'])
    
    eeg_mapping = dict(zip(eeg_chans, list(len(eeg_chans)*('eeg',))))
    eog_mapping = dict(zip(eog_chans, list(len(eog_chans)*('eog',))))
    emg_mapping = dict(zip(emg_chans, list(len(emg_chans)*('emg',))))
    
    mapping = {**eeg_mapping, **eog_mapping, **emg_mapping}

    data.pick_channels(ch_names=list(mapping), ordered = True)
    data.set_channel_types(mapping) 
    montage = mne.channels.make_standard_montage('standard_1020')
    data.set_montage(montage)
    
    data.set_eeg_reference(ref_channels = ['A1','A2'])
    data.filter(0.3, 35)
    data.notch_filter(freqs=(49,51), picks = list(mapping))

    raw_data = data.get_data()*1e6
    times = data.times
    
    for average in (False, True):
        data.plot_psd(average = average, n_fft= int((2/0.5)*data.info['sfreq']), dB=True, fmin=0.5, fmax=30)


    #%%
    ## Bad channel detection
    
     # epoched data
    _, epoched_data = yasa.sliding_window(raw_data, sf=256, window=30)
    
    ## RANSAC method
    # mne-ize 
    epochs = mne.EpochsArray(epoched_data/1e6, data.info, tmin = 0, baseline=(None, None)) 

    picks = mne.pick_types(epochs.info, eeg=True, stim=False, eog=False,
                           include=[], exclude=[])
    ransac = Ransac(verbose='progressbar', picks=picks, n_jobs=-1)
    epochs_clean = ransac.fit_transform(epochs)
    
    print(f'The following dataset: {files.split("/")[-1].split(".")[0]} has the following bad channels: {ransac.bad_chs_} ')
    print(f'The aforementioned dataset has a rectified 99th perentile amplitude of: {np.percentile(abs(raw_data[[list(mapping).index("A1"), list(mapping).index("A2")],:].mean(-1)), 99)}')
    
    
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
    ch_names = [epochs.ch_names[ii] for ii in ransac.picks][7::10]
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
    
    # create a new event_id that unifies stages 3 and 4
    event_id = {'Artifact': -1,
                'Wake': 0,
                'NREM1': 1,
                'NREM2': 2,
                'NREM3': 3,
                'REM': 4}
    
    # plot events
    fig = mne.viz.plot_events(events_train, event_id=event_id,
                              sfreq=raw_train.info['sfreq'],
                              first_samp=events_train[0, 0])
    
    # keep the color-code for further plotting
    stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    #%% 
    ## IRASA fun

    win = int(4 * 256)  # Window size is set to 4 seconds
    freqs, psd = welch(raw_data[0:10,:], 256, nperseg=win)  # Works with single or multi-channel data
    
    print(freqs.shape, psd.shape)  # psd has shape (n_channels, n_frequencies)
    
    # Plot
    plt.plot(freqs, psd[1, :], 'k', lw=2)
    plt.fill_between(freqs, psd[1, :], cmap='Spectral')
    plt.xlim(1, 30)
    plt.yscale('log')
    sns.despine()
    plt.title(list(mapping)[0])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)');
        
    # Apply the IRASA technique
    freqs, psd_aperiodic, psd_osc = yasa.irasa(raw_data, sf=256, ch_names=list(mapping), band=(1, 30), win_sec=4, return_fit=False)
    
    
    # Plot the aperiodic component on a linear-log scale
    plt.plot(freqs, psd_aperiodic[1, :], 'k', lw=2)
    plt.xlim(1, 30)
    plt.yscale('log')
    sns.despine()
    plt.title('Aperiodic component, chan = ' + chan[1])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)');
    
    # Plot the oscillatory component on a linear-linear scale
    plt.plot(freqs, psd_osc[1, :], 'k', lw=2)
    plt.xlim(1, 30)
    sns.despine()
    plt.title('Oscillatory component, chan = ' + chan[1])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)');



