#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:26:36 2021

@author: administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os 
import pandas as pd
import pingouin as pg
from scipy.signal import hilbert
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sleepstim.Analysis.Resting_State.rs_preproc import check_match_prepost_data_elements
sns.set_theme(style="whitegrid")
path = r'/media/administrator/data/Study_1_data/Pre_post_data'

#%%
def slalom_results(path, plot=False):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files if '.csv' in i])  
    sheet = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
    subj_cond = np.loadtxt(sheet, delimiter=',', dtype='str', skiprows=1) 
    cond_dict = {0:'sham', 1:'up', 2:'down'}
    # sort files to make sure blocks are consistent between subject nights
    pre, post = check_match_prepost_data_elements(path, files=files_list, dtype='slalom')
    slalom_dict = []
    for j, (pre_f, post_f) in enumerate(zip(pre, post)):
        if pre_f.split('/')[-1][-16::] == post_f.split('/')[-1][-16::]:
            # decode night with stimulation condition
            subj = pre_f.split('/')[-1].split('_')[0][-8::]
            sub_pos = np.where(subj == subj_cond[:,0])[0][0]
            if int(pre_f.split('/')[-1].split('_')[-2]) + 1 == 1:
                cond = int(subj_cond[sub_pos,:][1])
            elif int(pre_f.split('/')[-1].split('_')[-2]) + 1 == 2:
                cond = int(subj_cond[sub_pos,:][2])
            elif int(pre_f.split('/')[-1].split('_')[-2]) + 1 == 3:
                cond = int(subj_cond[sub_pos,:][3])
            
            print('Generating report for the following pre/post subject + condition: ' + pre_f.split('/')[-1][-16::])
            data_pre, data_post = np.genfromtxt(pre_f, delimiter=','), np.genfromtxt(post_f, delimiter=',')
                       
            # Calculate RMSE for pre/post with zscores & zscores + amplitude envelope
            error_pre_rms = mean_squared_error(y_true = stats.zscore(data_pre[0,:]), 
                                               y_pred = stats.zscore(data_pre[1,:]), squared=False)

            error_post_rms = mean_squared_error(y_true = stats.zscore(data_post[0,:]), 
                                                y_pred = stats.zscore(data_post[1,:]), squared=False)
            amplitude_envelope_pre_rms = mean_squared_error(y_true = np.abs(hilbert(stats.zscore(data_pre[0,:]))), 
                                                            y_pred = np.abs(hilbert(stats.zscore(data_pre[1,:]))), squared=False)            
            amplitude_envelope_post_rms = mean_squared_error(y_true = np.abs(hilbert(stats.zscore(data_post[0,:]))), 
                                                             y_pred = np.abs(hilbert(stats.zscore(data_post[1,:]))), squared=False)  
            
            # # Calculation of error per time point 
            # error_pre = stats.zscore(data_pre[0,:]) - stats.zscore(data_pre[1,:])
            # error_post = stats.zscore(data_post[0,:]) - stats.zscore(data_post[1,:])
            # amplitude_envelope_pre_diff = np.abs(hilbert(stats.zscore(data_pre[0,:]))) - np.abs(hilbert(stats.zscore(data_pre[1,:])))
            # amplitude_envelope_post_diff = np.abs(hilbert(stats.zscore(data_post[0,:]))) - np.abs(hilbert(stats.zscore(data_post[1,:])))

            # create dictionary with info
            s_dict = {
                "Absolute RMSE (Pre)": error_pre_rms,
                "Absolute RMSE (Post)": error_post_rms,
                "Amplitude envelope RMSE (Pre)": amplitude_envelope_pre_rms,
                "Amplitude envelope RMSE (Post)": amplitude_envelope_post_rms,
                "Absolute RMSE difference": error_pre_rms - error_post_rms,
                "Amplitude envelope RMSE difference": amplitude_envelope_pre_rms - amplitude_envelope_post_rms,
                "Subject": pre_f.split('/')[-1].split('_')[0][-8::],
                "Night": int(pre_f.split('/')[-1].split('_')[1]),
                "Stimulation night": cond_dict[cond], 
                "Block": int(pre_f.split('/')[-1].split('_')[2][0]),
                }
            
            slalom_dict.append(s_dict)
            
            if plot:
                plt.figure()
                plt.plot(stats.zscore(data_pre[0,:]), label = 'Slalom correct trajectory (pre)')
                plt.plot(stats.zscore(data_pre[1,:]), label = 'Slalom attempted trajectory (pre)')
                plt.plot(stats.zscore(data_post[0,:]), label = 'Slalom correct trajectory (post)')
                plt.plot(stats.zscore(data_post[1,:]), label = 'Slalom attempted trajectory (post)')
                plt.legend()
               
        else:
            raise NameError('The given pre- and post- file names do not correspond! ')
        
    # covert to dataframe    
    df = pd.DataFrame(slalom_dict)     
    
    return df

#%%

# =============================================================================
# TO DO- LINEAR MODEL WITH FACTORS INCLUDING BLOCK/CONDITION/NIGHT + ADD STIMULATION COND
# =============================================================================

def slalom_stats(df): 
    # ttests
    stats.ttest_ind(df[df['Stimulation night']=='sham']['Absolute RMSE difference'], df[df['Stimulation night']=='up']['Absolute RMSE difference'])
    stats.ttest_ind(df[df['Stimulation night']=='down']['Absolute RMSE difference'], df[df['Stimulation night']=='up']['Absolute RMSE difference'])
    # descriptive difference
    print(df.groupby(['Stimulation night'])['Absolute RMSE difference'].agg(['mean', 'std']).round(2))  
    # Compute repeated measures ANOVA
    rmanova = pg.rm_anova(dv='Absolute RMSE difference', within='Stimulation night', 
                          subject='Subject', detailed = True, data=df)
    # Pretty printing of ANOVA summary
    pg.print_table(rmanova)
    # Post hoc analysis
    # posthocs = pg.pairwise_ttests(dv='Absolute RMSE difference', within='Stimulation night',
    #                               subject='Subject', data=df)
    # pg.print_table(posthocs)
    

def violin_plot(df):     
    up_idx = (df['Stimulation night'] == 'up').to_numpy()  
    sham_idx = (df['Stimulation night'] == 'sham').to_numpy()
    down_idx = (df['Stimulation night'] == 'down').to_numpy() 
    vals = df['Absolute RMSE difference']
    ax = sns.violinplot(data=[vals[up_idx], vals[sham_idx], vals[down_idx]], palette="Set3", bw=.2, cut=1, linewidth=1)
    # Finalize the figure
    ax.set_xticklabels(['up','sham','down'])
    ax.set_ylabel("\u0394 Root Mean Square Error")
    ax.set_title("Slalom Performance")
    sns.despine(left=True, bottom=True)
    
#%%
path = r'/media/administrator/data/Study_1_data/Pre_post_data'
if __name__ == '__main__':
    df = slalom_results(path) 
    slalom_stats(df)
    violin_plot(df)

