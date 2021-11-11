#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:06:12 2021

@author: administrator
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import yasa 
import mne
import seaborn as sns
import pickle
from sleepstim.Analysis.Resting_State.rs_preproc import (Resting_State_Data_Struct, _pre_process_rs_data, plot_psd,  
                                                         norm_wavelet_power, subject_cond_parser, check_match_prepost_data_elements)
from sleepstim.Analysis.analysis_pre_process import load_preprocessed_data
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from sleepstim.sleep_funs import bandpower
import scipy.signal as signal 
import seaborn as sns

#%%
## Step 1 -- Preprocess resting state datasets and save into data structure with epochs for eyes open/eyes closed  
path = '/media/administrator/data/Study_1_data/Pre_post_data/'
files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files if 'resting_state' in i])
for i, files in enumerate(files_list):
    subj_cond = files.split("/")[-2] + "_" + files.split("/")[-1].split("_")[1] + "_" + files.split("/")[-1].split("_")[-2]
    save_path = '/media/administrator/data/Study_1_data/Pre-processed_data_resting_state/' + subj_cond + '_rs_preprocessed.p'
    print(i, files.split('/')[-2] + '_' + files.split('/')[-1])
    if not os.path.exists(save_path):
        _pre_process_rs_data(files = files, csd = True, prep_pipeline = False, art_method = 'covar', save = True)
            
#%%
## Step 2 -- Load in epoched data and compute connectivity/power analysis/etc.
path = '/media/administrator/data/Study_1_data/Pre-processed_data_resting_state/'
pre_files, post_files = check_match_prepost_data_elements(path, files=None, dtype='rs')
for j, (pre_f, post_f) in enumerate(zip(pre_files, post_files)):
    if pre_f.split('/')[-1].split('_')[0:2] == post_f.split('/')[-1].split('_')[0:2]:
        print(j, pre_f.split('/')[-1], post_f.split('/')[-1])
        # load pre/post
        Data_pre = load_preprocessed_data(pre_f)
        Data_post = load_preprocessed_data(post_f)
        epochs_eyes_open_pre = mne.EpochsArray(Data_pre.data_eyes_open/1e3, Data_pre.mne_info, 
                                                tmin = 0, baseline=(None), verbose=0) 
        epochs_eyes_closed_pre = mne.EpochsArray(Data_pre.data_eyes_close/1e3, Data_pre.mne_info, 
                                                  tmin = 0, baseline=(None), verbose=0) 
        epochs_eyes_open_post = mne.EpochsArray(Data_post.data_eyes_open/1e3, Data_post.mne_info, 
                                                tmin = 0, baseline=(None), verbose=0) 
        epochs_eyes_closed_post = mne.EpochsArray(Data_post.data_eyes_close/1e3, Data_post.mne_info, 
                                                  tmin = 0, baseline=(None), verbose=0) 
        # decode night with stimulation condition (pre or post will work)
        if subject_cond_parser(pre_f) == subject_cond_parser(post_f):
            cond_night = subject_cond_parser(pre_f)
        else:
            raise NameError('Pre and Post files do not match! ')
        # wavelet power analysis
        epochs_eyes_open_pre_pw = norm_wavelet_power(Data_pre.data_eyes_open[:,0:64,:], Data_pre.sf, foi=(4,8), 
                                                     wlt_params={'nc': 4, 'cf': 'auto'})
        epochs_eyes_close_pre_pw = norm_wavelet_power(Data_pre.data_eyes_close[:,0:64,:], Data_pre.sf, foi=(4,8), 
                                                      wlt_params={'nc': 4, 'cf': 'auto'})
        open_pre_pw = epochs_eyes_open_pre_pw['Normalized wavelet magnitude (channel/trial avg)'].mean(-1)
        close_pre_pw = epochs_eyes_close_pre_pw['Normalized wavelet magnitude (channel/trial avg)'].mean(-1)
        
        epochs_eyes_open_post_pw = norm_wavelet_power(Data_post.data_eyes_open[:,0:64,:], Data_post.sf, foi=(4,8), 
                                                      wlt_params={'nc': 4, 'cf': 'auto'})
        epochs_eyes_close_post_pw = norm_wavelet_power(Data_post.data_eyes_close[:,0:64,:], Data_post.sf, foi=(4,8), 
                                                       wlt_params={'nc': 4, 'cf': 'auto'})
        open_post_pw = epochs_eyes_open_post_pw['Normalized wavelet magnitude (channel/trial avg)'].mean(-1)
        close_post_pw = epochs_eyes_close_post_pw['Normalized wavelet magnitude (channel/trial avg)'].mean(-1)
        
        # PSD analysis 
        nperseg = (2 / 1) * 1000
        # Compute the modified periodogram (Welch)
        freqs, psd = signal.welch(Data_pre.data_eyes_open[:,0:64,:], 1000, nperseg=nperseg, average='median')
        # extract relative or absolute spectral density values for frequency bands of interest
        bp = yasa.bandpower_from_psd_ndarray(psd, freqs, bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
                                                                (12, 30, 'Beta'), (30, 40, 'Gamma')], relative=True)
        
        psd_eyes_open_pre = bandpower(Data_pre.data_eyes_open[:,0:64,:], Data_pre.sf, 
                                      bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
                                             (12, 30, 'Beta'), (30, 40, 'Gamma')], relative=True)
        
        plot_psd(psd, freqs, freq_range = (1,30), foi= (4,8), dB=True, ci=True)
        






mutual_info = mutual_info_score(open_post_pw, close_post_pw)
adj_mutual_info = adjusted_mutual_info_score(open_post_pw, close_post_pw, average_method='geometric')

#%%
