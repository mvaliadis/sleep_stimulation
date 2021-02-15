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
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, preprocess_sleep_data, compare_hypnograms, check_match_data_hypno_elements, 
                                  label_artifacts, load_preprocessed_data, Data_SW)
from sleepstim.sleep_funs import (bfr_butter_filt, bandpower, unravel_hypnogram_visbrain, 
                        downsample_scaled, load_xdf, channel_parser, thresholdcrossings, plot_confusion_matrix)

sns.set(style='darkgrid', font_scale=1.2)

#%%
files = '/media/administrator/data/Study_1_data/Raw_data/Experimental/YIOYSRPX_1/sleepstim_R001.xdf'
stream_dict = load_xdf(files) 
# select data and marker streams
try:
    rec_type = 'eeg_replay'
    stream_dict[rec_type]
except:
    rec_type = 'eego'
    stream_dict[rec_type]

data = stream_dict[rec_type]['time_series']
eego_times = stream_dict[rec_type]['time_stamps']
marker = stream_dict['reiz-marker']
classifier_predict = [marker['time_series'][ix] for ix in range(len(marker['time_series'])) if 'rf' in marker['time_series'][ix][0]]
classifier_timestamps = [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if 'rf' in marker['time_series'][ix][0]]
pinknoise_timestamps = [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if marker['time_series'][ix][0] == 'pinknoise']
# load hdr
info = stream_dict[rec_type]['info'] 
# delete stream_dict for memory sake
del stream_dict

stageing = False

sf = int(info['nominal_srate'][0])
# find channel names and types
ch_names, ch_types = channel_parser(info, data) 
# chans of interest for stageing
if stageing:
    EEG_index = np.r_[ch_names.index('F3'), ch_names.index('Fz'), ch_names.index('F4'), 
                      ch_names.index('C3'), ch_names.index('Cz'), ch_names.index('C4'), 
                      ch_names.index('P3'), ch_names.index('P4'), ch_names.index('O1'), 
                      ch_names.index('O2')]
else:
    EEG_index = [i for i, x in enumerate(ch_types) if x == "eeg"]
    
# other relevant indices
mastoids_index = np.r_[ch_names.index('M1'), ch_names.index('M2')]



filtparams = butter(4, (30), btype='low', fs = 512)
EEG = filtfilt(*filtparams, data[:,EEG_index], axis=0, padtype='odd')
EOG_L = bfr_butter_filt(data[:,[ch_names.index('EOG_L')]], sf, lfreq=0.5, hfreq=35)
EOG_R = bfr_butter_filt(data[:,[ch_names.index('EOG_R')]], sf, lfreq=0.5, hfreq=35)
EMG_L = bfr_butter_filt(data[:,[ch_names.index('EMG_L')]], sf, lfreq=10, hfreq=100)
EMG_R = bfr_butter_filt(data[:,[ch_names.index('EMG_R')]], sf, lfreq=10, hfreq=100)
ref_data = EEG[:,mastoids_index][..., :].mean(-1, keepdims=True)
EEG -= ref_data  

if 'bipECG' in ch_names:
    ECG = bfr_butter_filt(data[:,[ch_names.index('bipECG')]], sf, lfreq=0.5, hfreq=70)
    
data = np.concatenate([EEG, EOG_L, EOG_R, ECG], axis=1)
new_chans = [ch_names[i] for i in [j for j, x in enumerate(ch_types) if x == "eeg" or x == "eog" or x=="emg" or x=="ecg"]]
new_chtypes = [ch_types[i] for i in [j for j, x in enumerate(ch_types) if x == "eeg" or x == "eog" or x=="emg" or x=="ecg"]]
data = np.concatenate([data, EMG_L, EMG_R], axis=1)*1e6
Data = Data_Struct(data, new_chans, new_chtypes, eego_times, pinknoise_timestamps, classifier_predict, classifier_timestamps, sf)


eeg_index = np.r_[Data.chans.index('F3'), Data.chans.index('Fz'), Data.chans.index('F4'), 
                  Data.chans.index('C3'), Data.chans.index('Cz'), Data.chans.index('C4'), 
                  Data.chans.index('P3'), Data.chans.index('Pz'), Data.chans.index('P4')]
# hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/YIOYSRPX_1_hypno.txt'
# Data.hypno_with_art, Data.hypno = artifact_detect_hypno(Data.data[:,eeg_index], hypno_path = hypno_path, 
#                                                         downsample=False, sf=Data.sfreq, method='covar', window=2)

# filtparams = butter(6, (0.1,2.0), btype='band', fs = 512)
# EEG_sw_filt = filtfilt(*filtparams, Data.data[:,EEG_index], axis=0, padtype='odd')

EEG_sw_filt = Data.data[:,EEG_index]

Data.pinknoise_timestamps_sync = [Data.times[np.abs(Data.times - Data.pinknoise_times[ix]).argmin()] for ix in range(len(Data.pinknoise_times))]
Data.pinknoise_timestamps_sync_adj = np.asarray(Data.pinknoise_timestamps_sync)*Data.sfreq
first_bursts = Data.pinknoise_timestamps_sync_adj[::2]
insp_ix = [np.where(Data.times*Data.sfreq == first_bursts[i])[0][0] for i in range(len(first_bursts))]
center, _ = yasa.get_centered_indices(EEG_sw_filt[:,Data.chans.index('C3')], np.asarray(insp_ix), 
                                      npts_before = Data.sfreq*1.5, npts_after = Data.sfreq*2)
Data.chans = Data.chans[0:23]; Data.chtypes = Data.chtypes[0:23]
info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
epochs = mne.EpochsArray(np.swapaxes(EEG_sw_filt[center]/1e6, 1, 2), info, tmin = -1.5, baseline=(-1.5, -1.0), proj=False)      
epochs.filter(l_freq = None, h_freq = 4.0)
epochs.average(method='mean', picks='C3').plot()

C3 = epochs.get_data()[:,Data.chans.index('C3'),:]*1e6
times = epochs.times

# Compute instantaneous phase from a signal
n_samples = max(C3.shape)
nfast = next_fast_len(n_samples)
sw_pha = [np.angle(hilbert(C3[i,:], N=nfast)[:n_samples]) for i in range(C3.shape[0])]
pn_phase = [sw_pha[i][int(Data.sfreq*1.5)] for i in range(len(sw_pha))]
plt.figure()
plt.hist(np.rad2deg(pn_phase))
