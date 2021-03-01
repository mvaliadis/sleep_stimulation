#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:41:47 2021

@author: administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import seaborn as sns
import os 
import yasa
import mne
from os import chdir as cd
from os import listdir
import logging 
from scipy.signal import welch, butter, filtfilt, hilbert 
from scipy.stats import zscore
from scipy.fftpack import next_fast_len
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, preprocess_sleep_data, compare_hypnograms, check_match_data_hypno_elements,
                                                     label_artifacts, load_preprocessed_data, Data_SW)
from sleepstim.sleep_funs import (load_xdf, channel_parser, bfr_butter_filt, thresholdcrossings)

#%%
path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded/'
files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
for i, data_files in enumerate(files_list):
    # if data_files.endswith('_sham.p'):
    pass
    Data = pickle.load(open(data_files,"rb")) 
    Data.pinknoise_timestamps_sync = [Data.times[np.abs(Data.times - Data.pinknoise_times[ix]).argmin()] for ix in range(len(Data.pinknoise_times))]
    Data.pinknoise_timestamps_sync_adj = np.asarray(Data.pinknoise_timestamps_sync)*Data.sfreq
    first_bursts = Data.pinknoise_timestamps_sync_adj[::2]
    insp_ix = [np.where(Data.times*Data.sfreq == first_bursts[i])[0][0] for i in range(len(first_bursts))]
    center, _ = yasa.get_centered_indices(Data.data[:,Data.chans.index('C3')], np.asarray(insp_ix), 
                                          npts_before = Data.sfreq*1.5, npts_after = Data.sfreq*2)
    info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
    epochs = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = -1.5, baseline=(-1.5, -1.0), proj=False) 
    # epochs.set_montage(mne.channels.make_standard_montage('standard_1005'))
    
    # ## compute CSD 
    # epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
    # evoked_csd = epochs_csd.average()
    
    ## plotting without CSD
    # epochs.plot_image(combine='mean', picks='C3', title = f'{data_files.split("/")[-1].split(".")[0]} - C3')
    epochs.average(method='mean', picks='C3').plot()
    plt.title(f'{data_files.split("/")[-1].split(".")[0]} - C3')
    
    ## plotting with CSD
    # epochs_csd.plot_image(picks='C3')
    epochs_csd.average(method='mean', picks='C3').plot()
    plt.title(f'{data_files.split("/")[-1].split(".")[0]} - C3 - CSD')
    
    ## compartive plotting
    epochs.average().plot_joint(title='Average Mastoids', show=False)
    evoked_csd.plot_joint(title='Current Source Density')

#%%

path = '/media/administrator/data/Study_1_data/Raw_data/Experimental/'
files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])

for i, files in enumerate(files_list):
    print(i, files.split('/')[-2])
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
    filtparams = butter(4, 4, fs = 512)
    #filtparams = butter(1, (0.3, 4.0), btype = 'bandpass', fs = 512)
    EEG = filtfilt(*filtparams, data[:,EEG_index], axis=0, padtype='odd')
    EOG_L = bfr_butter_filt(data[:,[ch_names.index('EOG_L')]], sf, lfreq=0.5, hfreq=35)
    EOG_R = bfr_butter_filt(data[:,[ch_names.index('EOG_R')]], sf, lfreq=0.5, hfreq=35)
    EMG_L = bfr_butter_filt(data[:,[ch_names.index('EMG_L')]], sf, lfreq=10, hfreq=100)
    EMG_R = bfr_butter_filt(data[:,[ch_names.index('EMG_R')]], sf, lfreq=10, hfreq=100)
    #mastoids_index = np.r_[ch_names.index('M1'), ch_names.index('M2')]
    #ref_data = EEG[:,mastoids_index][..., :].mean(-1, keepdims=True)
    #EEG -= ref_data  
    
    if 'bipECG' in ch_names:
        ECG = bfr_butter_filt(data[:,[ch_names.index('bipECG')]], sf, lfreq=0.5, hfreq=70)
        data = np.concatenate([EEG, EOG_L, EOG_R, ECG], axis=1)
    else:
        data = np.concatenate([EEG, EOG_L, EOG_R], axis=1)
    
    new_chans = [ch_names[i] for i in [j for j, x in enumerate(ch_types) if x == "eeg" or x == "eog" or x=="emg" or x=="ecg"]]
    new_chtypes = [ch_types[i] for i in [j for j, x in enumerate(ch_types) if x == "eeg" or x == "eog" or x=="emg" or x=="ecg"]]
    data = np.concatenate([data, EMG_L, EMG_R], axis=1)*1e6
    Data = Data_Struct(data, new_chans, new_chtypes, eego_times, pinknoise_timestamps, classifier_predict, classifier_timestamps, sf)
    
    Data.pinknoise_timestamps_sync = [Data.times[np.abs(Data.times - Data.pinknoise_times[ix]).argmin()] for ix in range(len(Data.pinknoise_times))]
    Data.pinknoise_timestamps_sync_adj = np.asarray(Data.pinknoise_timestamps_sync)*Data.sfreq
    first_bursts = Data.pinknoise_timestamps_sync_adj[::2]
    insp_ix = [np.where(Data.times*Data.sfreq == first_bursts[i])[0][0] for i in range(len(first_bursts))]
    if insp_ix != []:
        center, _ = yasa.get_centered_indices(Data.data[:,Data.chans.index('C3')], np.asarray(insp_ix), 
                                              npts_before = Data.sfreq*1.5, npts_after = Data.sfreq*2)
        info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
        epochs = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = -1.5, baseline=(-1.5, 2.0), proj=False) 
    
        epochs.average(method='mean', picks='C3').plot()
        plt.title(f'{files.split("/")[-2]} - C3')
        logging.warning(f'The dataset: {files.split("/")[-2]} has {len(insp_ix)*2} pinknoise bursts! ')  
        
        
        epochs1 = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = -1.5, 
                          baseline=(-1.5, 2.0), proj=False) 
        epochs2 = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = -1.5,
                          baseline=(-1.5, 2.0), proj=False) 
        epochs2.set_eeg_reference(['M1','M2'])
        
        mne.viz.plot_compare_evokeds(dict(online_ref=epochs1.average(method='mean', picks='C3'), 
                                          offline_ref=epochs2.average(method='mean', picks='C3')),
                                          legend='upper left', show_sensors='upper right')

    else: 
        logging.warning(f'The dataset: {files.split("/")[-2]} has no pinknoise epochs! ')    

#%%
## Phase analysis 
C3 = epochs.get_data()[:,Data.chans.index('C3'),:]*1e6

# Identify slow waves in data
sws = [yasa.sw_detect(data = C3[i,:], sf=512, ch_names=['C3'], freq_sw=(None, None), 
                      amp_neg=(35, 300), amp_pos=(5, 200), amp_ptp=(75, 500)) for i in range(min(np.shape(C3)))]
sws_idx = []
for i in range(len(sws)):
    if sws[i] != None:
        sws_idx.append(1)
    else:
        sws_idx.append(0)
        
sws_idx = np.asarray(sws_idx)

# Compute instantaneous phase from a signal
n_samples = max(C3.shape)
nfast = next_fast_len(n_samples)
#sw_pha = [np.angle(hilbert(C3[i,:], N=nfast)[:n_samples]) for i in range(C3.shape[0])]
sw_pha = [np.angle(hilbert(C3[i,:], N=nfast)[:n_samples]) for i in np.where(sws_idx == 1)[0]]
pn_phase = [sw_pha[i][int(Data.sfreq*1.5)] for i in range(len(sw_pha))]

half1, half2, half3, half4 = [],[],[],[]
for i in np.where(sws_idx == 1)[0]:
    half1.append(np.linspace(sws[i].summary()['Start'][0]*Data.sfreq, 
                             sws[i].summary()['NegPeak'][0]*Data.sfreq, num=90, endpoint=True).astype(int))
    half2.append(np.linspace(sws[i].summary()['NegPeak'][0]*Data.sfreq + 1, 
                             sws[i].summary()['MidCrossing'][0]*Data.sfreq, num=90).astype(int))
    half3.append(np.linspace(sws[i].summary()['MidCrossing'][0]*Data.sfreq + 1, 
                             sws[i].summary()['PosPeak'][0]*Data.sfreq, num=90).astype(int))
    half4.append(np.linspace(sws[i].summary()['PosPeak'][0]*Data.sfreq + 1, 
                             sws[i].summary()['End'][0]*Data.sfreq, num=90).astype(int))

# Plot some figures
plt.figure() 
plt.plot(C3[np.where(sws_idx == 0)[0],:].mean(0), labels = 'PN modulation - No SW')
plt.plot(C3[np.where(sws_idx == 1)[0],:].mean(0), labels = 'PN modulation - SW')
plt.plot(C3.mean(0), labels = 'PN modulation - All')
plt.legend()

# for i in np.where(sws_idx == 0)[0]:
#     plt.figure()
#     plt.plot(C3[i,:])    
#     plt.vlines(thresholdcrossings(C3[i,:], 0)[0:5], ymin=-100, ymax=40)
    
# takes radians 
ax = plt.subplot(111, projection='polar')
ax.hist(pn_phase)


ptps = [np.ptp(C3[i,0:Data.sfreq*2]) for i in range(np.shape(C3)[0])]
loc = np.where(np.asarray(ptps)>75)[0]

for i in loc:
    plt.figure()
    plt.plot(C3_pn_narrow[i,0:Data.sfreq*3])
    plt.vlines(Data.sfreq*1.5, ymin=-75, ymax=40)

#%%
## Compare online and offline referenced data stimulation     
epochs1 = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = -1.5, 
                          baseline=(-1.5, 2.0), proj=False) 
epochs2 = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = -1.5,
                          baseline=(-1.5, 2.0), proj=False) 
epochs2.set_eeg_reference(['M1','M2'])

mne.viz.plot_compare_evokeds(dict(online_ref=epochs1.average(method='mean', picks='C3'), 
                                  offline_ref=epochs2.average(method='mean', picks='C3')),
                             legend='upper left', show_sensors='upper right')


