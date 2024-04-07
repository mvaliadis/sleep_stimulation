#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:37:37 2024

@author: administrator
"""

import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

stats_path = '/media/administrator/data/Study_2_data/NIDRA/CLNMES/Statistics/'
df_tct = pd.read_csv(stats_path + 'df_tct.csv')

def plot_erp_consistency(df_tct):
    for col in ['Consistency_GFP','Consistency', 'GFP_Trial',
                'GFP_Evoked','ERP']:
        # Plotting
        g = sns.catplot(data=df_tct, kind="bar", x="Stim",
                        y=col, col="Chan", palette="pastel")
        
        # Adjusting the plot
        g.set_titles("{col_name} Channel")
        g.set_axis_labels("", f"{col}", fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.show()

def group_sleep_epoch_stats(gavs, obj, title=None):
    # prepare adjacency matrix, must load non-csd mne epochs info
    adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/Final/Epochs/LuPf_1_1-epo.fif')
    adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
    # extract evoked data 
    eps = [list(obj[1].Epochs)[i].get_data(tmin=-0.01).mean(1)*1e3 for i in range(len(obj[1]))]
    contrast = np.concatenate([eps])
        
    # spatial permuation cluster test
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        contrast,                           # numpy array for contrast [n_subjects, n_voltage, n_channels]
        n_permutations=1024,                # 1000 is the minimum
        threshold=dict(start=0, step=0.2),  # TFCE, starting at 0, in 0.2 steps (in t-values)
        tail=0,                             # two-tailed test (1 or -1 for one-tailed)
        n_jobs=-1,                          # increase value to speed up computations
        adjacency=adjacency,                # sparse matrix for channel adjacency as computed above
        buffer_size=None,
        out_type='mask',                    # returns a mask map instead of indices of sig. points
        seed=1503
    )
    
    df_stats = pd.DataFrame({'Chan': ch_names, 'T-Stat': t_obs, 
                             'Pval': cluster_p_values}) 
    df_stats = df_stats.set_index("Chan")
    df_stats['Sig'] = (df_stats['Pval'] < 0.05)

    

    # Plot
    unit='mV/m2'
    fig, ax = plt.subplots(1, 2, figsize=(8,6))
    im1, _ = mne.viz.plot_topomap(contrast.mean(0), 
                                  pos=gavs.info,
                                  axes=ax[0], show=0, cmap='RdBu_r',
                                  names=None, show_names=False)
    cbar1 = fig.colorbar(im1, fraction=0.05, ax=ax[0])   
    cbar1.ax.set_ylabel(unit, rotation=270)
    plt.tight_layout()
    im2, _ = mne.viz.plot_topomap(t_obs, 
                                  pos=gavs.info, mask=df_stats['Sig'],
                                  axes=ax[1], show=0, cmap='RdBu_r',
                                  names=None, show_names=False, 
                                  mask_params=dict(markersize=8, markerfacecolor='y'))
    cbar2 = fig.colorbar(im2, fraction=0.05, ax=ax[1])   
    cbar2.ax.set_ylabel('t-stat', rotation=270)
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

def stats_targeting(gav_double, double_contrast, title=' '):
    gav = gav_double.loc[6].GAV
    # prepare adjacency matrix, must load non-csd mne epochs info
    adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/LuPf_1_1-epo.fif')
    adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
         
    # Cluster test 
    # spatial permuation cluster test
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        double_contrast,                    # numpy array for contrast [n_subjects, n_voltage, n_channels]
        n_permutations=1024,                # 1000 is the minimum
        threshold=dict(start=0, step=0.2),  # TFCE, starting at 0, in 0.2 steps (in t-values)
        tail=0,                             # two-tailed test (1 or -1 for one-tailed)
        n_jobs=-1,                          # increase value to speed up computations
        adjacency=adjacency,                # sparse matrix for channel adjacency as computed above
        buffer_size=None,
        out_type='mask',                    # returns a mask map instead of indices of sig. points
        seed=1503
    )
    
    df_stats = pd.DataFrame({'Chan': ch_names, 'T-Stat': t_obs, 
                             'Pval': cluster_p_values}) 
    df_stats = df_stats.set_index("Chan")
    df_stats['Sig'] = (df_stats['Pval'] < 0.05)
    unit='uV'
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(8,6))
    im1, _ = mne.viz.plot_topomap(double_contrast.mean(0), 
                                  pos=gav.info,
                                  axes=ax[0], show=0, cmap='RdBu_r',
                                  names=None)
    cbar1 = fig.colorbar(im1, fraction=0.05, ax=ax[0])   
    cbar1.ax.set_ylabel(unit, rotation=270)
    #plt.tight_layout()
    im2, _ = mne.viz.plot_topomap(t_obs, 
                                  pos=gav.info, mask=df_stats['Sig'],
                                  axes=ax[1], show=0, cmap='RdBu_r',
                                  names=None,
                                  mask_params=dict(markersize=8, markerfacecolor='y'))
    cbar2 = fig.colorbar(im2, fraction=0.05, ax=ax[1])   
    cbar2.ax.set_ylabel('T-stat', rotation=270)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig