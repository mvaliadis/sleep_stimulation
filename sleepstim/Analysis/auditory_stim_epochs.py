#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:55:28 2024

@author: administrator
"""

import mne
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm
import pandas as pd
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.channels import find_ch_adjacency
import scipy
    
def erp_sleep_analysis():
    sns.set_theme(color_codes=True) 
    # configure paths
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    #report_path = '/media/administrator/data/Study_1_data/Statistics/'
    #fig_path = '/media/administrator/data/Study_1_data/Figures/Pinknoise_epochs/'
    eps = []
    for i, file in enumerate(tqdm(maindir)):
        subject = file.split("/")[-1].split("_")[0]
        if '_mast_' not in file:
            pass
        else:
            if '_sham_down_' in file:
                cond = 'sham down'
            else:
                cond = file.split("/")[-1].split("_")[1]
            print(i, file.split('/')[-1])

            # load data & create evoked object
            epoch = mne.read_epochs(file, preload=True).apply_baseline((-4, -1.5)).resample(128)
            eps.append([subject, cond, epoch])
            
            eps_df = pd.DataFrame(eps, columns=['Subject','Condition','Epochs'])
            
    return eps_df

def tfr_sleep_analysis():
    sns.set_theme(color_codes=True) 
    # configure paths
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    #report_path = '/media/administrator/data/Study_1_data/Statistics/'
    #fig_path = '/media/administrator/data/Study_1_data/Figures/Pinknoise_epochs/'
    eps = []
    for i, file in enumerate(tqdm(maindir)):
        subject = file.split("/")[-1].split("_")[0]
        if '_mast_' not in file:
            pass
        else:
            if '_sham_down_' in file:
                cond = 'sham down'
            else:
                cond = file.split("/")[-1].split("_")[1]
            print(i, file.split('/')[-1])

            # load data & create evoked object
            epoch = mne.read_epochs(file, preload=True).apply_baseline((-4, -1.5)).resample(128)
            
            # Define parameters
            l_freq = 5  # Lower bound of frequency range
            h_freq = 25  # Upper bound of frequency range
            freqs = np.arange(l_freq, h_freq, 0.5)  # Frequency range
            n_cycles = freqs / 2.  # Different number of cycle per frequency
            baseline = (-3, -2)  # Baseline period in seconds
            chan = 'all'  # Channels to include
            vmin, vmax = [-3, 3]  # Define the color range for the plot
            
            # Compute the TFR using Morlet wavelets
            power, itc = mne.time_frequency.tfr_morlet(epoch.pick('eeg'), freqs=freqs, n_cycles=n_cycles, 
                                                       use_fft=True, return_itc=True, average=True)

            # Apply baseline correction
            power.apply_baseline(baseline=baseline, mode='zlogratio')
            
            power.plot(epoch.ch_names, baseline=baseline, mode='logratio', 
                       title='Time-Frequency Power', cmap=cmap)

            
            eps.append([subject, cond, epoch])
            
            eps_df = pd.DataFrame(eps, columns=['Subject','Condition','Epochs'])
            
    return eps_df
   
def create_gavs(up_contrast_evokeds, down_contrast_evokeds, double_contrast_evokeds):
    # Calculate grand averages if there are valid evokeds
    grand_average_down = mne.grand_average(down_contrast_evokeds) if down_contrast_evokeds else None
    grand_average_up = mne.grand_average(up_contrast_evokeds) if up_contrast_evokeds else None
    grand_average_double = mne.grand_average(double_contrast_evokeds) if double_contrast_evokeds else None
    
    return grand_average_up, grand_average_down, grand_average_double

def stats(up_evk, down_evk, double_evk,
          gav_up, gav_down, gav_double):
        
    # Prepare the data
    # Convert Evoked objects to numpy arrays and then transpose to (n_epochs, n_channels, n_times)
    data_down = np.concatenate([[down_evk[i].get_data(units='uV', 
                                                      tmin=-3, tmax=3)] for i in range(len(down_evk))])
    data_up = np.concatenate([[up_evk[i].get_data(units='uV', 
                                                  tmin=-3, tmax=3)] for i in range(len(up_evk))])
    data_double = np.concatenate([[double_evk[i].get_data(units='uV',
                                                          tmin=-3, tmax=3)] for i in range(len(double_evk))])
    
    # Define the adjacency matrix
    adjacency, _ = find_ch_adjacency(up_evk[0].info, ch_type='eeg') 
    
    # Hat variance adjustment 
    from functools import partial
    from mne.stats import ttest_1samp_no_p
    sigma = 1e-3  # sigma for the "hat" method
    stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)

    # Run stats and plots for each contrast
    for con, gav in zip([data_up, data_down, data_double],
                        [gav_up, gav_down, gav_double]):
        contrast = con.swapaxes(1,2)
        # compute threshold
        pval = 0.05  # arbitrary
        df = contrast.shape[0] - 1  # degrees of freedom for the test
        thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
        # Perform the cluster-based permutation test
        # Repeat for each contrast
        t_obs, clusters, cluster_p_values, h0 = spatio_temporal_cluster_1samp_test(
            contrast,  # Replace with the appropriate data for each contrast
            adjacency=adjacency,
            n_permutations=1024,  
            threshold=thresh,
            tail=0,                             # two-tailed test (1 or -1 for one-tailed)
            out_type='mask', 
            buffer_size=None,
            n_jobs=-1,
            stat_fun=stat_fun_hat, 
        )
        
        # return mask
        mask = cluster_p_values < 0.05
        
        # channel ROI
        selections = mne.channels.make_1020_channel_selections(gav.info, midline="z")
        
        # Visualize the results
        fig, axes = plt.subplots(nrows=3, figsize=(8, 8), dpi=78.38)
        axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
        gav.copy().crop(tmin=-2.99, tmax=3).plot_image(mask=clusters[np.argmin(cluster_p_values)].T,                                              
                                                       axes=axes,
                                                       show_names="all", 
                                                       mask_cmap='Spectral_r',
                                                       titles=None,
                                                       cmap='Spectral_r', 
                                                       colorbar=True, 
                                                       mask_alpha=0.80, 
                                                       group_by=selections, 
                                                       show=False)
        plt.tight_layout()
        plt.show()

def create_evoked(eps_df):
    # Assuming eps_df is your DataFrame
    subjects = eps_df['Subject'].unique()
    conditions = eps_df['Condition'].unique()
    
    # Lists to hold evoked responses for each condition
    evoked_up = []
    evoked_down = []
    evoked_sham_up = []
    evoked_sham_down = []

    # Dictionary to hold evoked responses
    evoked_dict = {subj: {cond: None for cond in conditions} for subj in subjects}
    
    # Initialize the contrast dictionaries
    contrast_dict = {subj: {'down_contrast': None, 'up_contrast': None} for subj in subjects}
    double_contrast_dict = {subj: None for subj in subjects}
    
    # Step 1 & 2: Extract and average epochs for each subject and condition
    for subj in subjects:
        for cond in conditions:
            if not eps_df[(eps_df['Subject'] == subj) & (eps_df['Condition'] == cond)].empty:
                epochs = eps_df[(eps_df['Subject'] == subj) & (eps_df['Condition'] == cond)]['Epochs'].iloc[0]
                evoke = epochs.average()
                evoked_dict[subj][cond] = evoke
                
                if cond == 'up':
                    evoked_up.append(evoke)
                elif cond == 'down':
                    evoked_down.append(evoke)
                elif cond == 'sham':
                    evoked_sham_up.append(evoke)
                elif cond == 'sham down':
                    evoked_sham_down.append(evoke)
    
    # Step 3: Compute individual and double contrasts for each subject
    for subj in subjects:
        if evoked_dict[subj]['down'] is not None and evoked_dict[subj]['sham down'] is not None:
            contrast_dict[subj]['down_contrast'] = mne.combine_evoked(
                [evoked_dict[subj]['down'], evoked_dict[subj]['sham down']], weights=[1, -1])
    
        if evoked_dict[subj]['up'] is not None and evoked_dict[subj]['sham'] is not None:
            contrast_dict[subj]['up_contrast'] = mne.combine_evoked(
                [evoked_dict[subj]['up'], evoked_dict[subj]['sham']], weights=[1, -1])
    
        # Compute double contrast if both individual contrasts exist
        if contrast_dict[subj]['down_contrast'] is not None and contrast_dict[subj]['up_contrast'] is not None:
            double_contrast_dict[subj] = mne.combine_evoked(
                [contrast_dict[subj]['up_contrast'], contrast_dict[subj]['down_contrast']], weights=[1, -1])

    # Step 4: Compute grand averages for each contrast
    down_contrast_evokeds = [contrast_dict[subj]['down_contrast'] for subj in subjects if contrast_dict[subj]['down_contrast'] is not None]
    up_contrast_evokeds = [contrast_dict[subj]['up_contrast'] for subj in subjects if contrast_dict[subj]['up_contrast'] is not None]
    double_contrast_evokeds = [double_contrast_dict[subj] for subj in subjects if double_contrast_dict[subj] is not None]
    
    return evoked_up, evoked_down, evoked_sham_up, evoked_sham_down, up_contrast_evokeds, down_contrast_evokeds, double_contrast_evokeds

def stats_targeting(gav_double, double_contrast0, title=' '):
    # prepare adjacency matrix
    adjacency, ch_names = mne.channels.find_ch_adjacency(gav_double.info, 'eeg')
         
    # Cluster test 
    # spatial permuation cluster test
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        double_contrast0,                           # numpy array for contrast [n_subjects, n_voltage, n_channels]
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
    im1, _ = mne.viz.plot_topomap(double_contrast0.mean(0), 
                                  pos=gav_double.info,
                                  axes=ax[0], show=0, cmap='RdBu_r',
                                  names=None)
    cbar1 = fig.colorbar(im1, fraction=0.05, ax=ax[0])   
    cbar1.ax.set_ylabel(unit, rotation=270)
    #plt.tight_layout()
    im2, _ = mne.viz.plot_topomap(t_obs, 
                                  pos=gav_double.info, mask=df_stats['Sig'],
                                  axes=ax[1], show=0, cmap='RdBu_r',
                                  names=None,
                                  mask_params=dict(markersize=8, markerfacecolor='y'))
    cbar2 = fig.colorbar(im2, fraction=0.05, ax=ax[1])   
    cbar2.ax.set_ylabel('T-stat', rotation=270)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig
    
#%%
if __name__ == '__main__':
    path = '/media/administrator/data/Study_1_data/Statistics/Sleep/'
    df = erp_sleep_analysis()
    good_subs = ['0RCB4IRJ', '3LFLTILW', '6QJ3ITMT', '7XVWEVOK', 'A4VCLD2I', 
                 'CWESJCNJ', 'D1BOI2AY', 'FUOPOVNF', 'HTXEYPW6', 'IYPJJ2KE', 
                 'KEQB5AWM', 'RVQL2MRD', 'UDLD86TO', 'W6AX3IMN', 'Y9VJUA9F', 
                 'YIOYSRPX', 'EQDORXF6'] 
    df = df[df['Subject'].isin(good_subs)]
    up, down, sham_up, sham_down, up_evk, down_evk, double_evk = create_evoked(df)
    gav_up, gav_down, gav_double = create_gavs(up_evk, down_evk, double_evk)
   
    # for c_evk, cond_names in zip([up, down, sham_up, sham_down], 
    #                              ['up','down','sham up', 'sham down']):
    #     t3 = np.asarray([0, 0.25, 0.5, .75, 1.0, 1.5, 1.75, 2.0])
    #     t = np.asarray([-0.75, -0.25, .25, 0.75, 1.25, 1.75])
    #     ts_args = dict(spatial_colors = True, gfp=True, time_unit='s')
    #     topomap_args = dict(outlines = 'head', time_unit='s', cmap = 'Spectral_r', 
    #                         time_format = "%0.2f s",  nrows=2, ncols='auto',
    #                         res=64, image_interp="cubic")
    #     g = mne.grand_average(c_evk)
    #     # gav.plot_joint(times = t, title=f'Evoked Response Difference - {contrast}', 
    #     #                ts_args=ts_args, topomap_args=topomap_args)
    #     fig = g.plot_topomap(times=t3, **topomap_args)
    #     fig.suptitle(f"Auditory response {cond_names}")
    #     fig.savefig(path + f'topomap_{cond_names}')
    #     plt.close('all')
        
    # for c_evk, cond_names in zip([gav_up, gav_down, gav_double], 
    #                              ['up contrast','down contrast','double contrast']):
    #     t3 = np.asarray([0, 0.25, 0.5, .75, 1.0, 1.5, 1.75, 2.0])
    #     t2 = np.asarray([-0.75, -0.25, .25, 0.75, 1.25, 1.75])
    #     t = np.asarray([0, 0.25, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0])
    #     ts_args = dict(spatial_colors = True, gfp=True, time_unit='s')
    #     topomap_args = dict(outlines = 'head', time_unit='s', cmap = 'Spectral_r', 
    #                         time_format = "%0.2f s",  nrows=2, ncols='auto',
    #                         res=64, image_interp="cubic")
    #     topomap_args2 = dict(outlines = 'head', time_unit='s', cmap = 'Spectral_r', 
    #                         time_format = "%0.2f s",  nrows=2, ncols='auto',
    #                         res=64, image_interp="cubic", average=0.5)
    #     g = c_evk
    #     fig1 = g.copy().crop(0, 2).plot_joint(times = t, title=f'Evoked Response Difference ({cond_names})', 
    #                                           ts_args=ts_args, topomap_args=topomap_args)
    #     fig1.savefig(path + f'Joint_plot_{cond_names}')
    #     fig2 = g.plot_topomap(times=t3, **topomap_args)
    #     fig2.suptitle(f"Auditory response ({cond_names})")
    #     fig2.savefig(path + f'topomap_{cond_names}')
    #     plt.close('all')
         
    ##### stats
    #%%
    down_contrast0 = np.concatenate([np.expand_dims(down_evk[i].get_data('eeg', 'uV', 
                                                                            tmin=-0.01, tmax=.01), 0) 
                                      for i in range(len(down_evk))], 0).mean(-1)   
    fig1 = stats_targeting(gav_double, down_contrast0, title='Stimulus 1 State - Down Contrast')  
    fig1.savefig(path + 'DC_1')
    
    down_contrast1 = np.concatenate([np.expand_dims(down_evk[i].get_data('eeg', 'uV', 
                                                                            tmin=1.065, tmax=1.085), 0) 
                                      for i in range(len(down_evk))], 0).mean(-1)   
    fig2 = stats_targeting(gav_double, down_contrast1, title='Stimulus 2 State - Down Contrast')
    fig2.savefig(path + 'DC_2')
    
    up_contrast0 = np.concatenate([np.expand_dims(up_evk[i].get_data('eeg', 'uV', 
                                                                            tmin=-0.01, tmax=.01), 0) 
                                      for i in range(len(up_evk))], 0).mean(-1)   
    fig3 = stats_targeting(gav_double, up_contrast0, title='Stimulus 1 State - Up Contrast')
    fig3.savefig(path + 'UC_1')
    
    up_contrast1 = np.concatenate([np.expand_dims(up_evk[i].get_data('eeg', 'uV', 
                                                                            tmin=1.065, tmax=1.085), 0) 
                                      for i in range(len(up_evk))], 0).mean(-1)   
    fig4 = stats_targeting(gav_double, up_contrast1, title='Stimulus 2 State - Up Contrast')
    fig4.savefig(path + 'UC_2')
    
    double_contrast0 = np.concatenate([np.expand_dims(double_evk[i].get_data('eeg', 'uV', 
                                                                            tmin=-0.01, tmax=.01), 0) 
                                      for i in range(len(double_evk))], 0).mean(-1)    
    fig5 = stats_targeting(gav_double, double_contrast0, title='Stimulus 1 State - Double Contrast') 
    fig5.savefig(path + 'DDC_1')
    
    double_contrast1 = np.concatenate([np.expand_dims(double_evk[i].get_data('eeg', 'uV', 
                                                                            tmin=1.065, tmax=1.085), 0) 
                                      for i in range(len(double_evk))], 0).mean(-1)   
    fig6 = stats_targeting(gav_double, double_contrast1, title='Stimulus 2 State - Double Contrast')
    fig6.savefig(path + 'DDC_2')
    
    plt.close('all')