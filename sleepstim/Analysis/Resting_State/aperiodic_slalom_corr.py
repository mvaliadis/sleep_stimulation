#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:35:31 2024

@author: administrator
"""

import yasa
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import pingouin as pg

good_subs = ['0RCB4IRJ', '3LFLTILW', '6QJ3ITMT', '7XVWEVOK', 'A4VCLD2I', 'CWESJCNJ', 
             'D1BOI2AY', 'FUOPOVNF', 'HTXEYPW6', 'IYPJJ2KE', 'KEQB5AWM', 'RVQL2MRD', 
             'UDLD86TO', 'W6AX3IMN', 'Y9VJUA9F', 'YIOYSRPX', 'EQDORXF6'] #'886MCPKG', 'IBYYXKMB']

save_path = '/media/administrator/data/Study_1_data/Statistics/Resting_state/' 
df_psd = pd.read_csv(save_path + 'rs_results_psd_new.csv', index_col=0)
df_psd = df_psd.loc[df_psd['Subject'].isin(good_subs)]
    

df_slalom = pd.read_csv('/media/administrator/data/Study_1_data/Statistics/Slalom/Slalom_results.csv', index_col=0)
for i in zip(['9PJZ8Z8F','475MQ9BL','5LNKD1MG','CWESJCNJ','6QF3HOJC','IBYYXKMB','NW2JV7YA']): #'RVQL2MRD'
    df_slalom.drop(df_slalom.loc[df_slalom['Subject']==i[0]].index, inplace=True)
df_slalom = df_slalom[df_slalom.Subject.isin(good_subs)]
df_slalom = df_slalom.set_index(['Subject','Condition','Night','Block'])

#%%
# yasa.topoplot(df_psd.groupby(['Condition','Channel']).mean().Spectral_exponent.loc['up'])
# yasa.topoplot(df_psd.groupby(['Condition','Channel']).mean().Spectral_exponent.loc['sham'])
# yasa.topoplot(df_psd.groupby(['Condition','Channel']).mean().Spectral_exponent.loc['down'])



merged_df = pd.merge(df_psd, df_slalom, on=['Subject', 'Condition'], 
                     how='inner').reset_index(drop=True)

merged_df2 = merged_df.groupby(['Condition','Channel','Subject'])[['Spectral_exponent','RMSE_difference_ratio']].mean()

# Define a function to compute correlations
def compute_correlations(group):
    from scipy.stats import pearsonr
    corr_slope_sp, p_value_sp = pearsonr(group['Spectral_exponent'], 
                                         group['RMSE_difference_ratio'])
    
    return pd.Series({'Corr': corr_slope_sp, 'P_Value': p_value_sp})


# Group the data by Subject, Condition, Stage, and Chan
grouped_data = merged_df2.groupby(['Condition', 
                                   'Channel']).apply(compute_correlations).reset_index()


yasa.topoplot(grouped_data.set_index(['Condition','Channel']).Corr.loc['up'], mask=
              grouped_data.set_index(['Condition','Channel']).P_Value.loc['up'] < 0.05)
yasa.topoplot(grouped_data.set_index(['Condition','Channel']).Corr.loc['sham'], mask=
              grouped_data.set_index(['Condition','Channel']).P_Value.loc['sham'] < 0.05)
yasa.topoplot(grouped_data.set_index(['Condition','Channel']).Corr.loc['down'], mask=
              grouped_data.set_index(['Condition','Channel']).P_Value.loc['down'] < 0.05)