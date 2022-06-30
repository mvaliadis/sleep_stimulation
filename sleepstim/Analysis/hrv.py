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
#import pingouin as pg

def outlier_detection(hrv_df, session='sleep'):
    from sklearn.ensemble import IsolationForest
    # initialize IF predictor object
    ilf = IsolationForest(contamination='auto', max_samples='auto',
                          verbose=0, random_state=42)
    # exclude categorical and string variables
    if session=='sleep':
        excl = ['Subject','Condition','Stimuli', 
                'Stimulation_period','Signal_quality']
    else:
        excl = ['Subject','Session','Condition','Stimuli', 
                'Stimulation_period','Signal_quality']  
    stripped_df = hrv_df.loc[:, ~hrv_df.columns.isin(excl)]
    # predict inliers versus outliers
    good = ilf.fit_predict(
        stripped_df.fillna(value=int(1e3)).replace([-np.inf, np.inf], int(1e3))
        )
    good[good == -1] = 0
    hrv_df['Isolation forest score'] = good
    # drop bad sessions
    hrv_df_clean = hrv_df.copy()[good==1].reset_index()
    del hrv_df_clean['index']

    return hrv_df_clean

def hrv_sleep_results():
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files if 'av.p' in i])
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
            #ecg_stim = ecg[Data.pinknoise_times_sync[0]:Data.pinknoise_times_sync[-1]]
            ecg_stim = ecg[Data.pinknoise_times_sync[0]:210*Data.sfreq*60]
            _, epochs = yasa.sliding_window(ecg_stim, sf = Data.sfreq, window = 60*5)
            # compute HRV epoch-wise
            for idx, epoch in enumerate(epochs):
                ep = nk.signal.signal_resample(epoch, sampling_rate=Data.sfreq,
                                               desired_sampling_rate=Data.sfreq/2, 
                                               method="interpolation")
                peaks, info = nk.ecg_peaks(ep, sampling_rate=Data.sfreq/2, 
                                           correct_artifacts=True)
                quality = nk.ecg_quality(ep, rpeaks=None, sampling_rate=Data.sfreq/2, 
                                         method="zhao2018", approach="fuzzy")
                #hrv_t = nk.hrv_time(peaks, sampling_rate=Data.sfreq/4)
                #hrv_f = nk.hrv_frequency(peaks, sampling_rate=Data.sfreq/4, psd_method="lomb")
                #hrv = pd.concat([hrv_t, hrv_f], axis=1)
                try:
                    hrv = nk.hrv(peaks, sampling_rate=Data.sfreq/2, 
                                 **dict(psd_method="lomb", silent=True))
             
                    hr = nk.ecg_rate(peaks, sampling_rate=Data.sfreq/2, 
                                     desired_length=len(ep))
                    
                    hrv.insert(0, 'Subject', files.split("/")[-1].split('_')[0])
                    hrv.insert(1, 'Condition', subject_cond_parser(files, study_phase = 'sleep'))
                    hrv.insert(2, 'Stimuli', len(Data.pinknoise_times_sync))
                    hrv.insert(3, 'Stimulation_period', data_minutes)
                    hrv.insert(4, 'Signal_quality', quality)
                    hrv.insert(5, 'Heart_rate', hr.mean())
                    hrv_results_all = pd.concat([hrv_results_all, hrv])
                except:
                    pass
        
    hrv_results_all.reset_index(inplace=True)
    del hrv_results_all['index']
    
    return hrv_results_all

def hrv_resting_results():
    path_rs = '/media/administrator/data/Study_1_data/Pre_post_data/'
    restdir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path_rs) for i in files if 'resting_state' in i])
    hrv_results_all = pd.DataFrame()
    for i, files in tqdm(enumerate(restdir)):
        print(f' Computing HRV for file: {files.split("/")[-1]} ')
        # if i == 87:
        #     break
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
                hrv_results_all = pd.concat([hrv_results_all, hrv])
            
        hrv_results_all.reset_index(inplace=True)
        del hrv_results_all['index']
    
    return hrv_results_all

def hrv_stats(df):
    params = df.columns.to_list()[5::]
    #params = ['HRV_LFHF','HRV_SDNN', 'HRV_HF', 'HRV_LF']
    for param in params:
        if all(np.isnan(df[param])):
            continue
        else:
            print(f'Running LMM for the follow HRV parameter: {param} ')
            model = Lmer(f"{param} ~ Condition*Session + (1|Subject)",
                         data=df, family='gaussian')
        
            # Using dummy-coding; suppress summary output
            model.fit(factors={"Condition": ["sham", "up", "down"],
                               "Session": ["pre", "post"]}, 
                      ordered=True, summarize=False)
            
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
    run = input('Do you wish to restart the HRV analysis? ')
    if run == 'yes':
        hrv_df_sleep = hrv_sleep_results()
        hrv_df_sleep.to_csv('/media/administrator/data/Study_1_data/Statistics/HRV/HRV_sleep.csv')
        hrv_df_rs = hrv_resting_results()
        hrv_df_rs.to_csv('/media/administrator/data/Study_1_data/Statistics/HRV/HRV_rs.csv')
    else:
        hrv_df_sleep = pd.read_csv('/media/administrator/data/Study_1_data/Statistics/HRV/HRV_sleep.csv', index_col=0)
        hrv_df_rs = pd.read_csv('/media/administrator/data/Study_1_data/Statistics/HRV/HRV_rs.csv', index_col=0)
    #hrv_df_sleep = hrv_df_sleep[hrv_df_sleep.Signal_quality!='Unnacceptable']
    hrv_df_sleep_clean = outlier_detection(hrv_df_sleep, session='sleep')
    
    hrv_df_rs = outlier_detection(hrv_df_rs, session='rs')
    #hrv_df_rs = hrv_df_rs[hrv_df_rs.Heart_rate > np.percentile(hrv_df_rs.Heart_rate, 1)]
    
    # do stats 
    #stats = hrv_stats(hrv_df_rs)  
    
    
    # take mean of hrv
    hrv_df_sleep_mean = hrv_df_sleep_clean.groupby(['Condition','Subject']).mean().reset_index()
  
    # keep only if 2 sessions
    for subject in hrv_df_sleep_mean['Subject']:
        occurences = len(np.where((hrv_df_sleep_mean['Subject'] == subject).to_numpy())[0])
        if occurences < 2:
            hrv_df_sleep_mean.drop(hrv_df_sleep_mean.loc[hrv_df_sleep_mean['Subject']==subject].index, inplace=True)
            hrv_df_sleep_clean.drop(hrv_df_sleep_clean.loc[hrv_df_sleep_clean['Subject']==subject].index, inplace=True)
            
#%%
def plot_rs_params(hrv_df_rs):
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 3, figsize=(6 * 6, 3*2), sharey=False)
    fig.suptitle("Change in Resting State HRV Parameters")
    params = ['Heart_rate','HRV_LFHF','HRV_SDNN', 'HRV_RMSSD','HRV_HF', 'HRV_LF']
    #params = list(hrv_df_rs.columns)[5:]
    for ind, (ax, param) in enumerate(zip(axes.flat,params)):
            
        # Create a violin plot for the current parameter
        ax = sns.violinplot(data=hrv_df_rs, x='Condition', #inner="quart",
                            y=f'{param}', hue='Session', hue_order=["pre","post"],
                            ax=ax, palette="Set3", bw=.2, linewidth=1, split=True); 
             
        # despine layout
        sns.despine(left=True, bottom=True)
        
        # tighten layout
        plt.tight_layout()

plot_rs_params(hrv_df_rs)

#%%
def plot_sleep_params(hrv_df_sleep_mean):
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 3, figsize=(6 * 6, 8*2), sharey=False)
    fig.suptitle("Change in HRV Parameters")
    params = ['Heart_rate','HRV_LFHF','HRV_SDNN', 'HRV_RMSSD','HRV_HF', 'HRV_LF']
    #params = list(hrv_df_sleep_mean.columns)[5:]
    for ind, (ax, param) in enumerate(zip(axes.flat,params)):
            
        # Create a violin plot for the current parameter
        ax = sns.violinplot(data=hrv_df_sleep_mean, x='Condition',
                            y=f'{param}', hue='Condition',
                            ax=ax, palette="Set3", bw=.2, linewidth=1); 
             
        # despine layout
        sns.despine(left=True, bottom=True)
        
        # tighten layout
        plt.tight_layout()

plot_sleep_params(hrv_df_sleep_mean)    

#%%
params = ['Heart_rate','HRV_LFHF','HRV_SDNN', 
          'HRV_RMSSD','HRV_HF','HRV_LF']
for param in params:
    if all(np.isnan(hrv_df_sleep_clean[param])):
        continue
    else:
        print(f'Running LMM for the follow HRV parameter: {param} ')
        model = Lmer(f"{param} ~ Condition + (Condition|Subject)",
                     data=hrv_df_sleep_clean, family='gaussian')
    
        # Using dummy-coding; suppress summary output
        model.fit(factors={"Condition": ["sham", "up", "down"]}, 
                  ordered=True, summarize=False)
        
        # Get ANOVA table, but this time force orthogonality for valid SS III inferences
        # In this case the data are balanced so nothing changes
        print(model.anova(force_orthogonal=True))
        
#%%
## delete all subjects with less than 4 sessions
for subject in hrv_df_rs['Subject']:
    occurences = len(np.where((hrv_df_rs['Subject'] == subject).to_numpy())[0])
    if occurences < 4:
        hrv_df_rs.drop(hrv_df_rs.loc[hrv_df_rs['Subject']==subject].index, inplace=True)
    
## create HRV resting state contrasts 
hrv_df_rs_f = hrv_df_rs.groupby(['Subject','Condition']).mean().reset_index()
for param in hrv_df_rs.columns.to_list()[5::]:
    hrv_df_rs_f[param + '_diff'] = np.nan
for (sub, cond), df_s in hrv_df_rs.groupby(['Subject','Condition']):
    for param in df_s.columns.to_list()[5::]:
        if all(np.isnan(df_s[param])):
            continue               
        else:
            if len(df_s) == 2:
                diff = (df_s.query("Session == 'post'")[param].to_numpy() - 
                        df_s.query("Session == 'pre'")[param].to_numpy())[0]
                loc = hrv_df_rs_f[np.logical_and(hrv_df_rs_f.Subject==sub, 
                                                 hrv_df_rs_f.Condition==cond)]
                hrv_df_rs_f[param + '_diff'][loc.index[0]] = diff

#%%
## Do stats
#params = hrv_df_rs_f.columns.to_list()[5::]
params = ['Heart_rate_diff','HRV_LFHF_diff','HRV_SDNN_diff', 
          'HRV_RMSSD_diff','HRV_HF_diff','HRV_LF_diff']
for param in params:
    if all(np.isnan(hrv_df_rs_f[param])):
        continue
    else:
        #hrv_df_rs_f[hrv_df_rs_f[param].dropna(inplace=True)]
        print(f'Running LMM for the follow HRV parameter: {param} ')
        model = Lmer(f"{param} ~ Condition + (1|Subject)",
                     data=hrv_df_rs_f, family='gaussian')
    
        # Using dummy-coding; suppress summary output
        model.fit(factors={"Condition": ["sham", "up", "down"]}, 
                  ordered=True, summarize=False)
        
        # Get ANOVA table, but this time force orthogonality for valid SS III inferences
        # In this case the data are balanced so nothing changes
        print(model.anova(force_orthogonal=True))

#%%

x = sws.groupby(['Condition','Subject']).mean().reset_index()
df_c = pd.merge(x, hrv_df_sleep_mean)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=df_c['ndPAC'].to_numpy(),
                                                                     y=df_c['HRV_HF'].to_numpy())

import pingouin as pg
r, pval = pg.circ_corrcl(x = df_c['PhaseAtSigmaPeak'].to_numpy(),
                         y = df_c['HRV_LF'].to_numpy())