#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:02:20 2024

@author: administrator
"""

import numpy as np
from glob import glob 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mne 
mne.set_log_level('ERROR')

def process_pvt(file):
    # extract subject info
    subject_night = file.split('/')[5]
    print(subject_night)
    subject, night = subject_night.split('_')
    
    # get stim mode (clas vs. clnmes)
    try:  
        events = mne.read_events(f'/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/{subject}_{night}_csd-epo.fif', 
                                 return_event_id=True)
        if 'pn_sham_c3' in events[-1]:
            mode = 'pn'
        elif 'nmes_sham_c3' in events[-1]:
            mode = 'nmes'
    
    except Exception as e:
        print(f"Error reading events: {e}")
        return None
    
    # If mode is not set, return None or skip further processing
    if mode is None:
        print("No relevant events found.")
        return None
       
    # extract pvt csv file
    pvt = pd.read_csv(file)
    pvt = pvt[['sender', 'sender_type', 'response', 'response_action', 'ended_on', 'duration', 'time_run', 
               'time_render', 'time_show', 'time_end', 'time_commit', 'timestamp', 'time_switch',
               'openLabId', 'type', 'task', 'project', 'status', 'code', 'timeout', 'Trial_Number']]
    if pvt['type'].unique()[0] == 'incremental':
        df = pvt[pvt['type']=='incremental'].reset_index(drop=True)
    else:
        df = pvt[pvt['type']=='full'].reset_index(drop=True)
    
    # extract reaction times, trials, and subject info
    rt = np.asarray([df['duration'][idx] for idx, trial in enumerate(df['ended_on']) if trial=='response' and df['Trial_Number'][idx] >= 0])
    # correct ~50 ms lag
    rt = rt - 50
    subjects = np.asarray([df['code'][idx] for idx, trial in enumerate(df['ended_on']) if trial=='response' and df['Trial_Number'][idx] >= 0])
    if np.unique(subjects) == subject:
        trials = np.asarray([df['Trial_Number'][idx] for idx, trial in enumerate(df['ended_on']) if trial=='response' and df['Trial_Number'][idx] >= 0]).astype(int)
    
        # extract and correct stats: response speed, lapses, lapse probability.
        validity_rt = (rt >= 100)
        rt = rt[validity_rt]
        trials = trials[validity_rt]
        
        validity_lapses = (rt >= 500)
        speed = (1/rt)
        try:
            lapses = pd.Series(validity_lapses).value_counts().loc[True]
            lapse_prob = pd.Series(validity_lapses).value_counts(normalize=True).loc[True]
        except:
            lapses = 0
            lapse_prob = 0.0
        
        # create dataframe and add RT, Subject, and trials as column
        df_rt = pd.DataFrame({'RT': rt, 
                              'Speed': speed, 
                              'Lapses': lapses, 
                              'Lapses_Transformed' : np.sqrt(lapses) + (np.sqrt(lapses + 1)), 
                              'Lapse_Probability': lapse_prob,
                              'Trial': trials,
                              'Subject': subjects, 
                              'Night': night, 
                              'Mode': mode})
        
        # drop RTs over 5 seconds
        df_rt = df_rt[df_rt.RT <= 5000]
        
        return df_rt
    
    else:
        print(f'{subject} is not the same as {np.unique(subjects)}')

#%%
path = '/media/administrator/Sleep_Data/Raw/*/*PVT.csv'
stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'
pvts = []
if __name__ == '__main__':
    for file in glob(path):
        print(file)
        pvt_df = process_pvt(file)
        pvts.append(pvt_df)
        
    # concatenate pvts
    pvt_all_df = pd.concat(pvts).reset_index(drop=True)
    pvt_all_df.to_csv(stats_path + 'df_pvt.csv')
        
    # delete first 10 trials for each subject in dataframe
    # pvt_all_df = pvt_all_df.groupby('Subject').apply(lambda x: x.iloc[10:]).reset_index(drop=True)
        
    # plot RTs across subjects
    # sns.boxplot(pvt_all_df, x='Subject', y='RT', showfliers=False)
    # sns.boxplot(pvt_all_df, x='Mode', y='RT', hue='Night', showfliers=False)
    
    # # # plot distribution of RT per subject
    # # sns.displot(pvt_all_df, x='RT', hue='Subject', kind='hist')
    # # plt.xlim(pvt_all_df.RT.min(), pvt_all_df.RT.max())
    # # plt.vlines(np.median(pvt_all_df['RT']), 0, 20, color='red', linestyles='--')
    
    # # extract rhytmicity of RT across trials with line plot in separate plots per subject
    # pvt_all_df.groupby('Subject').plot(x='Trial', y='RT', kind='line', legend=False)
    
    # # plot autocorrelation of RT per subject
    # from statsmodels.graphics.tsaplots import plot_acf
    # for subject in pvt_all_df['Subject'].unique():
    #     plot_acf(pvt_all_df[pvt_all_df['Subject']==subject]['RT'], title=subject)
        
    # for sub, df in pvt_all_df.groupby('Subject'): 
    #     print(sub, len(df))
    #     plt.plot(df.RT.diff().to_numpy()[1:], '--o')
        
