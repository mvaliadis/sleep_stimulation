#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:27:37 2022

@author: administrator
"""

from tqdm import tqdm
import neurokit2 as nk
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, load_preprocessed_data)
from sleepstim.Analysis.Resting_State.rs_preproc import subject_cond_parser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import seaborn as sns
from pymer4.models import Lmer
import yasa
import liesl

def hrv_sleep_results(maindir):
    hrv_results_all = pd.DataFrame()
    for i, files in tqdm(enumerate(maindir)):
        print(f' Computing HRV for file: {files.split("/")[-1]} ')
        # load data 
        Data = load_preprocessed_data(files)[0]
        # check number of stimuli 
        if len(Data.pinknoise_times_sync) < 1:
            pass
        else:  
            # check data length
            data_minutes = (Data.pinknoise_times_sync[-1] - Data.pinknoise_times_sync[0])/Data.sfreq/60
            if data_minutes <= 5:
                pass
            ## HRV
            ecg = nk.ecg_clean(Data.data[:,-1], sampling_rate=Data.sfreq, method='neurokit') 
            ecg_stim = ecg[Data.pinknoise_times_sync[0]:Data.pinknoise_times_sync[-1]]
            ecg_stim = ecg[0:210*Data.sfreq]
            _, epochs = yasa.sliding_window(ecg_stim, sf = Data.sfreq, window = 150)
            # compute HRV epoch-wise
            for idx, epoch in enumerate(epochs):
                epoch = nk.signal.signal_resample(epoch, sampling_rate=Data.sfreq,
                                                  desired_sampling_rate=Data.sfreq/4, 
                                                  method="interpolation")
                peaks, info = nk.ecg_peaks(epoch, sampling_rate=Data.sfreq/4, 
                                           correct_artifacts=True)
                quality = nk.ecg_quality(epoch, rpeaks=None, sampling_rate=Data.sfreq/4, 
                                         method="zhao2018", approach="fuzzy")
                hrv_t = nk.hrv_time(peaks, sampling_rate=Data.sfreq/4)
                hrv_f = nk.hrv_frequency(peaks, sampling_rate=Data.sfreq/4, psd_method="lomb")
                hrv = pd.concat([hrv_t, hrv_f], axis=1)
                hrv.insert(0, 'Subject', files.split("/")[-1].split('_')[0])
                hrv.insert(1, 'Session', subject_cond_parser(files, study_phase = 'sleep'))
                hrv.insert(2, 'Stimuli', len(Data.pinknoise_times_sync))
                hrv.insert(3, 'Stimulation_period', data_minutes)
                hrv.insert(4, 'Signal_quality', quality)
                hrv_results_all = hrv_results_all.append(hrv)
        
    hrv_results_all.reset_index(inplace=True)
    del hrv_results_all['index']
    
    return hrv_results_all

def hrv_resting_results(restdir):
    hrv_results_all = pd.DataFrame()
    for i, files in tqdm(enumerate(restdir)):
        print(f' Computing HRV for file: {files.split("/")[-1]} ')
        if i == 87:
            break
        # load data 
        data = liesl.XDFFile(files, verbose=1)['eego']
        markers = liesl.XDFFile(files, verbose=1)['reiz-marker']
        marker_times = markers.time_stamps
        # add another time stamp to determine end of resting state
        marker_times = np.append(marker_times, marker_times[-1] + np.mean(np.diff(marker_times)))
        eye_ts = [np.argmin(np.abs(data.time_stamps - ts)) for ts in marker_times]
        # data_breath = liesl.XDFFile(files)
        ch_names = data.channel_labels
        ch_names[data.channel_labels.index('chan_5')] = 'ECG'
        ecg_data = data.time_series[:, ch_names.index('ECG')][eye_ts[0]:eye_ts[-1]]*1e6
        # check data length
        data_minutes = len(ecg_data)/data.nominal_srate/60
        if data_minutes <= 5:
            pass
        else:
        ## HRV
            ecg = nk.ecg_clean(ecg_data, sampling_rate=data.nominal_srate, method='neurokit') 
            _, epochs = yasa.sliding_window(ecg, sf=data.nominal_srate, window = 60*5)
            # compute HRV epoch-wise
            for idx, epoch in enumerate(epochs):
                epoch = nk.signal.signal_resample(epoch, sampling_rate=data.nominal_srate,
                                                  desired_sampling_rate=data.nominal_srate/4, 
                                                  method="interpolation")
                peaks, info = nk.ecg_peaks(epoch, sampling_rate=data.nominal_srate/4, 
                                           correct_artifacts=True)
                quality = nk.ecg_quality(epoch, rpeaks=None, sampling_rate=data.nominal_srate/4, 
                                         method="zhao2018", approach="fuzzy")
                hr = nk.ecg_rate(peaks, sampling_rate=data.nominal_srate/4, 
                                 desired_length=len(epoch))
                hrv = nk.hrv(peaks, sampling_rate=data.nominal_srate/4, **dict(psd_method="lomb", silent=True))
                hrv.insert(0, 'Subject', files.split("/")[-2])
                hrv.insert(1, 'Condition', subject_cond_parser(files, study_phase = 'rs_hrv'))
                hrv.insert(2, 'Session', files.split('/')[-1].split('_')[-2])
                hrv.insert(3, 'Data_length', data_minutes)
                hrv.insert(4, 'Signal_quality', quality)
                hrv.insert(5, 'Heart_rate', hr.mean())
                hrv_results_all = hrv_results_all.append(hrv)
            
        hrv_results_all.reset_index(inplace=True)
        del hrv_results_all['index']
    
    return hrv_results_all

def hrv_stats(df):
    # params = df.columns.to_list()[5::]
    params = ['HRV_LFHF','HRV_SDNN', 'HRV_HF', 'HRV_LF']
    for param in params:
        print(f'Running LMM for the follow HRV parameter: {param} ')
        model = Lmer(f"{param} ~ Session + (1|Subject)",
                     data=df, family='gaussian')
    
        # Using dummy-coding; suppress summary output
        model.fit(factors={"Session": ["sham", "up", "down"]}, 
                  ordered=True, summarize=True)
        
        # Get ANOVA table, but this time force orthogonality for valid SS III inferences
        # In this case the data are balanced so nothing changes
        print(model.anova(force_orthogonal=True))
    
        ## Post-hoc tests 
        marginal_estimates, comparisons = model.post_hoc(p_adjust="fdr",
                                                         marginal_vars='Session',
                                                         # grouping_vars='Block',
                                                         )
        
        print(marginal_estimates)
        print(comparisons)
        
#%%
if __name__ == "__main__":
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files if 'av.p' in i])
    path_rs = '/media/administrator/data/Study_1_data/Pre_post_data/'
    restdir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path_rs) for i in files if 'resting_state' in i])
    if run:
        hrv_df_sleep = hrv_sleep_results(maindir)
        hrv_df_sleep.to_csv('/media/administrator/data/Study_1_data/Statistics/HRV/HRV_sleep.csv')
        hrv_df_rs = hrv_resting_results(restdir)
        hrv_df_rs.to_csv('/media/administrator/data/Study_1_data/Statistics/HRV/HRV_rs.csv')
    else:
        hrv_df = pd.read_csv('/media/administrator/data/Study_1_data/Statistics/HRV/HRV_sleep.csv')
    hrv_df = hrv_df[hrv_df.Signal_quality!='Unnacceptable']
    # do stats 
    stats = hrv_stats(hrv_df)
    
    