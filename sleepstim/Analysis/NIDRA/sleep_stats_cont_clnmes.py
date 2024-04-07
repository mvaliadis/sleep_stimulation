#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:28:21 2024

@author: administrator
"""

import yasa
import pandas as pd

# Load data 
stats_path = '/media/administrator/data/Study_2_data/NIDRA/CLNMES/Statistics/'
df_ndpac = pd.read_csv(stats_path + 'df_ndpac.csv')

#%%
# Plot data
yasa.topoplot(df_ndpacs_all.groupby(['Condition','Session','Chan']).mean().loc['nmes_sham_fz','Post'].ndPAC, 
              cmap = 'Spectral_r', vmin=0.2, vmax=0.3)
yasa.topoplot(df_ndpacs_all.groupby(['Condition','Session','Chan']).mean().loc['nmes_stim_fz','Post'].ndPAC,
              cmap = 'Spectral_r', vmin=0.2, vmax=0.3)
yasa.topoplot(df_ndpacs_all.groupby(['Condition','Session','Chan']).mean().loc['nmes_sham_c3','Post'].ndPAC,
              cmap = 'Spectral_r', vmin=0.2, vmax=0.3)
yasa.topoplot(df_ndpacs_all.groupby(['Condition','Session','Chan']).mean().loc['nmes_stim_c3','Post'].ndPAC,
              cmap = 'Spectral_r', vmin=0.2, vmax=0.3)

# Pivot table to restructure data for difference calculation
df_pivot = df_ndpacs_all.pivot_table(index=['Condition','Subject','Night','Recording','Chan'], columns='Session',
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