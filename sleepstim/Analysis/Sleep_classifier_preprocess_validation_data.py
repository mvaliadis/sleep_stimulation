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
                                                     label_artifacts, load_preprocessed_data, save_preprocess_sleep_data, Data_SW)
from sleepstim.sleep_funs import (bfr_butter_filt, bandpower, unravel_hypnogram_visbrain, 
                                  downsample_scaled, load_xdf, channel_parser, thresholdcrossings, plot_confusion_matrix)

sns.set(style='darkgrid', font_scale=1.2)

#%%

path = '/media/administrator/data/Study_1_data/Raw_data/Experimental/'
def sleep_data_validation(path, save=True, validation='classsifier'):
    """
    
    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    save : TYPE, optional
        DESCRIPTION. The default is True.
    validation : TYPE, {'classifier' or 'auditory'}
        DESCRIPTION. Choose between respective data preprocessing validation methods.

    Returns
    -------
    Data : TYPE
        DESCRIPTION.

    """
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    for i, files in enumerate(files_list):
        if validation == 'classifier':
            new_path='/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_classifier_validation/' + files.split('/')[-2] + '_preproc_data_cv.p'
        elif validation == 'auditory':
            new_path='/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/' + files.split('/')[-2] + '_preproc_data_av.p'
        if not os.path.exists(new_path):
            print(f'Pre-processing the dataset for subject and condition: {files.split("/")[-2]} !')
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
            
            
            # set filtering params
            if validation=='classsifier':
                filtparams = butter(4, (0.5, 35), btype='bandpass', fs = 512)
                mastoids_index = np.r_[ch_names.index('M1'), ch_names.index('M2')]
            elif validation=='auditory':
                filtparams = butter(4, 4, fs = 512)
        
            EEG = filtfilt(*filtparams, data[:,EEG_index], axis=0, padtype='odd')
            EOG_L = bfr_butter_filt(data[:,[ch_names.index('EOG_L')]], sf, lfreq=0.5, hfreq=35)
            EOG_R = bfr_butter_filt(data[:,[ch_names.index('EOG_R')]], sf, lfreq=0.5, hfreq=35)
            EMG_L = bfr_butter_filt(data[:,[ch_names.index('EMG_L')]], sf, lfreq=10, hfreq=100)
            EMG_R = bfr_butter_filt(data[:,[ch_names.index('EMG_R')]], sf, lfreq=10, hfreq=100)
            
            if validation=='classsifier':
                ref_data = EEG[:,mastoids_index][..., :].mean(-1, keepdims=True)
                EEG -= ref_data  
            
            if 'bipECG' in ch_names:
                ECG = bfr_butter_filt(data[:,[ch_names.index('bipECG')]], sf, lfreq=0.5, hfreq=70)
                data = np.concatenate([EEG, EOG_L, EOG_R, ECG], axis=1)
            else:
                data = np.concatenate([EEG, EOG_L, EOG_R], axis=1)
                
            new_chans = [ch_names[i] for i in [j for j, x in enumerate(ch_types) if x == "eeg" or x == "eog" or x=="emg" or x=="ecg"]]
            new_chtypes = [ch_types[i] for i in [j for j, x in enumerate(ch_types) if x == "eeg" or x == "eog" or x=="emg" or x=="ecg"]]
            data = np.concatenate([data, EMG_L, EMG_R], axis=1)*1e6
            Data = Data_Struct(data, new_chans, new_chtypes, eego_times, pinknoise_timestamps, classifier_predict, classifier_timestamps, sf)
            
            # save data
            if save:
                save_preprocess_sleep_data(Data, new_path + files.split('/')[-2] + '_preproc_data_cv.p') 
            else:
                return Data
        else:
            print(f'The dataset for subject and condition: {files.split("/")[-2]} has already been pre-processed! ')
            continue