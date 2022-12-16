#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:42:20 2022

@author: administrator
"""

## Import packages
import mne 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pingouin as pg
import seaborn as sns
import time
import pickle
from pymer4.models import Lmer
from functools import reduce
from matplotlib.colors import Normalize
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       permutation_cluster_1samp_test)
#mne.set_log_level("CRITICAL")
#sns.set_theme(color_codes=True) 
good_subs = ['0RCB4IRJ', '3LFLTILW', '6QJ3ITMT', '7XVWEVOK', 'A4VCLD2I', 'CWESJCNJ', 
             'D1BOI2AY', 'FUOPOVNF', 'HTXEYPW6', 'IYPJJ2KE', 'KEQB5AWM', 
             'RVQL2MRD', 'UDLD86TO', 'W6AX3IMN', 'Y9VJUA9F', 'YIOYSRPX', 'EQDORXF6'] #'886MCPKG', 'IBYYXKMB']

# Load results dataframe
phase_df = pd.read_csv('/media/administrator/data/Study_1_data/Statistics/Auditory_validation/phase_report.csv', index_col=0)
phase_df.groupby(['Condition']).agg([pg.circ_mean, scipy.stats.circstd]).round(2)

# Load results dataframe
phase_b_df = pd.read_csv('/media/administrator/data/Study_1_data/Statistics/Auditory_validation/phase_binned_report.csv', index_col=0)
phase_b_df.groupby(['Condition']).agg([pg.circ_mean, scipy.stats.circstd]).round(2)

# keep only good subjects
phase_b_df = phase_b_df.loc[phase_b_df['Subject'].isin(good_subs)]
phase_df = phase_df.loc[phase_df['Subject'].isin(good_subs)]

def group_phase_targeting_plot(phase_df, bins=12):
    kwargs_arrow={'fc': 'tab:red', 'ec': 'tab:red'}
    with plt.style.context('seaborn'):
        sns.set_palette("flare")
        meth = 'Analytical - Hilbert'
        #meth = 'Binned'
        subject_phi = phase_df.groupby(['Condition','Subject']).agg(pg.circ_mean).reset_index()       
        subject_rvs = phase_df.groupby(['Condition','Subject']).agg(pg.circ_r).reset_index()       
        subject_stds = phase_df.groupby(['Condition','Subject']).agg(scipy.stats.circstd).reset_index()
        g = sns.FacetGrid(phase_df, col='Condition', hue='Condition',
                          subplot_kws=dict(projection='polar'), height=4.5,
                          sharex=True, sharey=True, despine=False, ylim=[0, 0.8])
        g.map_dataframe(sns.histplot, f'Target Phase ({meth})', 
                        stat='density', bins = bins)
        for ax, con in zip(range(len(g.axes[0])), phase_df.Condition.unique()): 
            rv = np.float64(np.mean(subject_rvs[subject_rvs.Condition==con][f'Target Phase ({meth})']))
            phi = pg.circ_mean(subject_phi[subject_phi.Condition==con][f'Target Phase ({meth})'])
            circ_std = np.float64(np.mean(subject_stds[subject_stds.Condition==con][f'Target Phase ({meth})']))
            z, pval = pg.circ_rayleigh(subject_phi[subject_phi.Condition==con][f'Target Phase ({meth})'])
            g.axes[0, ax].arrow(0, 0, phi, rv, **kwargs_arrow)       
            plt.tight_layout()
            print(f'The circular mean for {con.upper()} - {meth} is {phi.round(4)} with a stand deviation of {circ_std.round(4)}, and a resultant vector length of : {rv.round(4)}')
            print(f'The Rayleigh Z-statistic for non-uniformity of circular data for {con.upper()} - {meth} : {z.round(4)} with a p-value : {pval.round(4)}')
        plt.show()
