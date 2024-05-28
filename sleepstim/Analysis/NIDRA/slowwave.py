#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:32:50 2024

@author: administrator
"""


import pandas as pd
import seaborn as sns
import numpy as np
import pingouin as pg
import yasa

df_sw_old = pd.read_csv('/media/administrator/Sleep_Data/Processed/Statistics/df_sw_old.csv', index_col=0)
df_sp_old = pd.read_csv('/media/administrator/Sleep_Data/Processed/Statistics/df_spindles_old.csv', index_col=0)

df_sw = pd.read_csv('/media/administrator/Sleep_Data/Processed/Statistics/df_sw.csv', index_col=0)
df_sp = pd.read_csv('/media/administrator/Sleep_Data/Processed/Statistics/df_spindles.csv', index_col=0)

df_sw_lm = pd.read_csv('/media/administrator/Sleep_Data/Processed/Statistics/df_sw_lm.csv', index_col=0)
df_sp_lm = pd.read_csv('/media/administrator/Sleep_Data/Processed/Statistics/df_spindles_lm.csv', index_col=0)

df_stats = pd.read_csv('/media/administrator/Sleep_Data/Processed/Statistics/df_sleep_stats.csv', index_col=0)

sns.violinplot(df_sw, x='Mode', y='75_Thresh')
sns.violinplot(df_sp, x='Mode', y='Amplitude')

from scipy.stats import trim_mean

sw_amplitudes = trim_mean(df_sw['PTP'], proportiontocut=0.05, axis=0)

df_sw.groupby(['Subject','Mode'])['PTP'].apply(trim_mean, proportiontocut=0.05)
df_sw.groupby(['Subject','Mode'])['PTP'].apply(np.median)

df_sp.groupby(['Subject','Mode'])['Amplitude'].apply(trim_mean, proportiontocut=0.05)

for stat in df_stats.columns[3:]:
    print(stat)
    print(df_stats.rm_anova(dv=stat, within='Mode', subject='Subject'))