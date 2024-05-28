#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:28:21 2024

@author: administrator
"""

import yasa
import pandas as pd

# Load data 
stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'
df_hrv = pd.read_csv(stats_path + 'df_sleep_hrv.csv', index_col=0)


df_sws = pd.read_csv(stats_path + 'df_sw.csv', index_col=0)

df_spindle = pd.read_csv(stats_path + 'df_spindles.csv', index_col=0)

#%%
# Plot data
# Loop over each combination of stimulation type and channel
for stim in df_ndpac.Stim.unique():
    for chan in df_ndpac.Target_Chan.unique():
        # Calculate the mean ndPAC for the specified conditions
        data = df_ndpac.groupby(['Mode', 'Session', 'Target_Chan', 'Chan']).mean().loc['nmes', stim, 'Post', chan].ndPAC
        
        # Generate the topoplot for each condition
        yasa.topoplot(data, cmap='Spectral_r', vmin=0.2, vmax=0.3)


# Pivot table to restructure data for difference calculation
df_pivot = df_ndpac.pivot_table(index=['Mode','Subject','Night','Chan'], columns='Session',
                                values=['SigmaPeakTime', 'PhaseAtSigmaPeak', 'ndPAC'])

# Calculate the difference (post - pre)
df_contrasts = df_pivot.copy()
df_contrasts['SigmaPeakTime_diff'] =  df_pivot['SigmaPeakTime']['Post'] - df_pivot['SigmaPeakTime']['Pre']
df_contrasts['PhaseAtSigmaPeak_diff'] = df_pivot['PhaseAtSigmaPeak']['Post'] - df_pivot['PhaseAtSigmaPeak']['Pre']
df_contrasts['ndPAC_diff'] = df_pivot['ndPAC']['Post'] - df_pivot['ndPAC']['Pre']
df_contrasts.columns = df_contrasts.columns.droplevel('Session')
df_contrasts = df_contrasts[['SigmaPeakTime_diff','PhaseAtSigmaPeak_diff', 'ndPAC_diff']].reset_index()

# Plot ndPAC difference plot
yasa.topoplot(df_contrasts.groupby(['Condition','Chan']).mean().loc['nmes_sham_fz'].ndPAC_diff, 
              cmap = 'Spectral_r', vmin=-0.01, vmax=0.1)
yasa.topoplot(df_contrasts.groupby(['Condition','Chan']).mean().loc['nmes_stim_fz'].ndPAC_diff,
              cmap = 'Spectral_r', vmin=-0.01, vmax=0.1)
yasa.topoplot(df_contrasts.groupby(['Condition', 'Chan']).mean().loc['nmes_sham_c3'].ndPAC_diff,
              cmap = 'Spectral_r', vmin=-0.01, vmax=0.1)
yasa.topoplot(df_contrasts.groupby(['Condition','Chan']).mean().loc['nmes_stim_c3'].ndPAC_diff,
              cmap = 'Spectral_r', vmin=-0.01, vmax=0.1)

#%%
def hrv_sleep_correlations(hrv):
    # Correlation in NREM sleep
    cols = hrv.columns #["hr_mean", "hrv_rmssd", "delta", "theta", "alpha", "sigma"]
    nrem_corr = hrv.xs(2)[cols].corr(method="spearman").round(2).iloc[:2]
    
    # Correlation in REM sleep
    rem_corr = hrv.xs(4)[cols].corr(method="spearman").round(2).iloc[:2]
    
    return nrem_corr, rem_corr