#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:06:12 2021

@author: administrator
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import yasa 
import mne
import seaborn as sns
import pickle
from sleepstim.Analysis.Resting_State.rs_preproc import (Resting_State_Data_Struct, _pre_process_rs_data, plot_psd,  
                                                         norm_wavelet_power, subject_cond_parser, check_match_prepost_data_elements)
from sleepstim.Analysis.analysis_pre_process import load_preprocessed_data
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from sleepstim.sleep_funs import bandpower
import scipy.signal as signal 
import seaborn as sns
from neurodsp.rhythm import compute_lagged_coherence
from neurodsp.plts.rhythm import plot_lagged_coherence
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pingouin as pg

#%%
## Step 1 -- Preprocess resting state datasets and save into data structure with epochs for eyes open/eyes closed  
path = '/media/administrator/data/Study_1_data/Pre_post_data/'
files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files if 'resting_state' in i])
for i, files in enumerate(files_list):
    subj_cond = files.split("/")[-2] + "_" + files.split("/")[-1].split("_")[1] + "_" + files.split("/")[-1].split("_")[-2]
    save_path = '/media/administrator/data/Study_1_data/Pre-processed_data_resting_state/' + subj_cond + '_rs_preprocessed.p'
    print(i, files.split('/')[-2] + '_' + files.split('/')[-1])
    if not os.path.exists(save_path):
        _pre_process_rs_data(files = files, csd = True, prep_pipeline = False, art_method = 'covar', save = True)
            
#%%
## Step 2 -- Load in epoched data and compute connectivity/power analysis/etc.
path = '/media/administrator/data/Study_1_data/Pre-processed_data_resting_state/'
def resting_state_power_analysis(path):
    pre_files, post_files = check_match_prepost_data_elements(path, files=None, dtype='rs')
    results = defaultdict(lambda: [])
    for j, (pre_f, post_f) in tqdm(enumerate(zip(pre_files, post_files))):
        if pre_f.split('/')[-1].split('_')[0:2] == post_f.split('/')[-1].split('_')[0:2]:
            print(j, pre_f.split('/')[-1], post_f.split('/')[-1])
            subjname = " ".join(str(pre_f).split("/")[-1].split('_')[0:2])         
            night = int(subjname[-1]) + 1
            session = str(pre_f).split("/")[-1].split('_')[2]
            # decode night with stimulation condition (pre or post will work)
            if subject_cond_parser(pre_f) == subject_cond_parser(post_f):
                cond = subject_cond_parser(pre_f)
            else:
                raise NameError('Pre and Post files do not match! ')     
      
            # load pre/post
            Data_pre = load_preprocessed_data(pre_f)
            Data_post = load_preprocessed_data(post_f)
            epochs_eyes_open_pre = mne.EpochsArray(Data_pre.data_eyes_open/1e3, Data_pre.mne_info, 
                                                    tmin = 0, baseline=(None), verbose=0) 
            # epochs_eyes_closed_pre = mne.EpochsArray(Data_pre.data_eyes_close/1e3, Data_pre.mne_info, 
            #                                           tmin = 0, baseline=(None), verbose=0) 
            epochs_eyes_open_post = mne.EpochsArray(Data_post.data_eyes_open/1e3, Data_post.mne_info, 
                                                    tmin = 0, baseline=(None), verbose=0) 
            # epochs_eyes_closed_post = mne.EpochsArray(Data_post.data_eyes_close/1e3, Data_post.mne_info, 
            #                                           tmin = 0, baseline=(None), verbose=0) 
    
            ## Wavelet power analysis
            epochs_eyes_open_pre_pw = norm_wavelet_power(Data_pre.data_eyes_open[:,0:64,:], Data_pre.sf, foi=(4,8), 
                                                         wlt_params={'nc': 4, 'cf': 'auto'})
            # epochs_eyes_close_pre_pw = norm_wavelet_power(Data_pre.data_eyes_close[:,0:64,:], Data_pre.sf, foi=(4,8), 
            #                                               wlt_params={'nc': 4, 'cf': 'auto'})
            open_pre_pw = epochs_eyes_open_pre_pw['Normalized wavelet magnitude (channel/trial avg)'].mean(-1)
            # close_pre_pw = epochs_eyes_close_pre_pw['Normalized wavelet magnitude (channel/trial avg)'].mean(-1)
            
            epochs_eyes_open_post_pw = norm_wavelet_power(Data_post.data_eyes_open[:,0:64,:], Data_post.sf, foi=(4,8), 
                                                          wlt_params={'nc': 4, 'cf': 'auto'})
            # epochs_eyes_close_post_pw = norm_wavelet_power(Data_post.data_eyes_close[:,0:64,:], Data_post.sf, foi=(4,8), 
            #                                                wlt_params={'nc': 4, 'cf': 'auto'})
            open_post_pw = epochs_eyes_open_post_pw['Normalized wavelet magnitude (channel/trial avg)'].mean(-1)
            # close_post_pw = epochs_eyes_close_post_pw['Normalized wavelet magnitude (channel/trial avg)'].mean(-1)
            
            wavelet_diff = open_post_pw - open_pre_pw
                    
            ## PSD analysis 
            nperseg = (2 / 1) * 1000
            # Compute the modified periodogram (Welch)
            freqs_pre, psd_pre  = signal.welch(Data_pre.data_eyes_open[:,0:64,:], 1000, nperseg=nperseg, average='median')
            freqs_post, psd_post = signal.welch(Data_post.data_eyes_open[:,0:64,:], 1000, nperseg=nperseg, average='median')
            # extract relative or absolute spectral density values for frequency bands of interest
            bp_pre = yasa.bandpower_from_psd_ndarray(psd_pre, freqs_pre, bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
                                                                    (12, 30, 'Beta'), (30, 40, 'Gamma')], relative=True)
            bp_post = yasa.bandpower_from_psd_ndarray(psd_post, freqs_post, bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
                                                                    (12, 30, 'Beta'), (30, 40, 'Gamma')], relative=True)
            bp_mean_pre = bp_pre.mean(1)   
            bp_mean_post = bp_post.mean(1)
            
            bp_diff = np.asarray(bp_mean_post) - np.asarray(bp_mean_pre)
            
            ## lagged theta coherence
            lagged_theta_post = [compute_lagged_coherence(sig = epochs_eyes_open_post.get_data()[i,0:64,:]*1e3, fs=1000, freqs=(4, 8),
                                                      return_spectrum=False, n_cycles=2) for i in range(epochs_eyes_open_post.get_data().shape[0])]
            
            lagged_theta_pre = [compute_lagged_coherence(sig = epochs_eyes_open_pre.get_data()[i,0:64,:]*1e3, fs=1000, freqs=(4, 8),
                                                      return_spectrum=False, n_cycles=2) for i in range(epochs_eyes_open_pre.get_data().shape[0])]
            
            theta_diff = np.asarray(lagged_theta_post).mean(0) - np.asarray(lagged_theta_pre).mean(0)
            
            ## lagged alpha coherence
            lagged_alpha_post = [compute_lagged_coherence(sig = epochs_eyes_open_post.get_data()[i,0:64,:]*1e3, fs=1000, freqs=(8, 12),
                                                      return_spectrum=False, n_cycles=3) for i in range(epochs_eyes_open_post.get_data().shape[0])]
            
            lagged_alpha_pre = [compute_lagged_coherence(sig = epochs_eyes_open_pre.get_data()[i,0:64,:]*1e3, fs=1000, freqs=(8, 12),
                                                      return_spectrum=False, n_cycles=3) for i in range(epochs_eyes_open_pre.get_data().shape[0])]
            
            alpha_diff = np.asarray(lagged_alpha_post).mean(0) - np.asarray(lagged_alpha_pre).mean(0)
            
            ## Create dataframe with post - pre differences 
            results["Subject"].extend([subjname]*64)
            results["Night"].extend([night]*64)
            results["Condition"].extend([cond]*64)
            results["Session"].extend([session]*64)
            
            results["Normalized theta (wavelet) power - diff"].extend(wavelet_diff)
            results["Lagged theta coherence - diff"].extend(theta_diff)
            results["Lagged alpha coherence - diff"].extend(alpha_diff)
            results["PSD delta (1 - 4 Hz) - post"].extend(bp_post[0].mean(0))
            results["PSD theta (4 - 8 Hz) - post"].extend(bp_post[1].mean(0))
            results["PSD alpha (8 - 12 Hz) - post"].extend(bp_post[2].mean(0))
            results["PSD beta (12 - 30 Hz) - post"].extend(bp_post[3].mean(0))
            results["PSD gamma (30 - 40 Hz) - post"].extend(bp_post[4].mean(0))
            results["PSD delta (1 - 4 Hz) - pre"].extend(bp_pre[0].mean(0))
            results["PSD theta (4 - 8 Hz) - pre"].extend(bp_pre[1].mean(0))
            results["PSD alpha (8 - 12 Hz) - pre"].extend(bp_pre[2].mean(0))
            results["PSD beta (12 - 30 Hz) - pre"].extend(bp_pre[3].mean(0))
            results["PSD gamma (30 - 40 Hz) - pre"].extend(bp_pre[4].mean(0))
            results["PSD delta (1 - 4 Hz) - diff"].extend(bp_diff[0])
            results["PSD theta (4 - 8 Hz) - diff"].extend(bp_diff[1])
            results["PSD alpha (8 - 12 Hz) - diff"].extend(bp_diff[2])
            results["PSD beta (12 - 30 Hz) - diff"].extend(bp_diff[3])
            results["PSD gamma (30 - 40 Hz) - diff"].extend(bp_diff[4])
    
    return results 


#%%
run = input('Do you wish to restart the resting state power analysis? ')
if run == 'yes':
    results = resting_state_power_analysis(path)
    df = pd.DataFrame(results)
    save_path = '/media/administrator/data/Study_1_data/Statistics/Resting_state/rs_results.p'
    pickle.dump(df, open(save_path, "wb"))  
else:
    df = pickle.load(open('/media/administrator/data/Study_1_data/Statistics/Resting_state/rs_results.p', 'rb'))
    df.to_csv(r'/media/administrator/data/Study_1_data/Statistics/Resting_state/rs_results.csv')

sub = df['Subject'].to_numpy()
subs = [sub[i][0:8] for i in range(len(sub))]
df['Subject'] = subs

stacks = []
for i in range(int(len(df)/64)):
    stacks.append(np.arange(0,64,1))
chans = np.hstack(stacks)
df['Channels'] = chans

subjects = df['Subject'].unique()
for i in zip(['9PJZ8Z8F','475MQ9BL', '5LNKD1MG', 'CWESJCNJ']):
    df.drop(df.loc[df['Subject']==i[0]].index, inplace=True)
    

sham = df[df['Condition']=='sham']['Normalized theta (wavelet) power - diff'].to_numpy()
sham = sham.reshape(64, int(len(sham)/64))
    
up = df[df['Condition']=='up']['Normalized theta (wavelet) power - diff'].to_numpy()
up = up.reshape(64, int(len(up)/64))
    
down = df[df['Condition']=='down']['Normalized theta (wavelet) power - diff'].to_numpy()
down = down.reshape(64, int(len(down)/64))

# rmanova = pg.rm_anova(dv='Normalized theta (wavelet) power - diff', within=['Condition','Channels'], 
#                       subject='Subject', detailed = True, data=df)
# # Pretty printing of ANOVA summary
# pg.print_table(rmanova)
# # Post hoc analysis
# posthocs = pg.pairwise_ttests(dv='Normalized theta (wavelet) power - diff', within='Condition',
#                               subject='Subject', data=df)
# pg.print_table(posthocs)


#t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test([norm_up_diff,norm_down_diff])
# t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test([down.T, up.T, sham.T], 
#                                                                      n_permutations=1000,
#                                                                      tail=1, n_jobs=1,
#                                                                      out_type='mask')

t_obs, clusters, cluster_pv, H0 = mne.stats.spatio_temporal_cluster_test([down.T, up.T, sham.T], 
                                                                         n_permutations=1000,
                                                                         tail=1, n_jobs=1,
                                                                         out_type='mask')
# from mne.channels import find_ch_adjacency
# adjacency, ch_names = find_ch_adjacency(raw.info, ch_type='eeg')
# threshold = 50.0  # very high, but the test is quite sensitive on this data
# # set family-wise p-value
# p_accept = 0.01
# t_obs, clusters, cluster_pv, H0 = mne.stats.spatio_temporal_cluster_test(X, n_permutations=1000,
#                                              threshold=threshold, tail=1,
#                                              n_jobs=1, buffer_size=None,
#                                              adjacency=adjacency)

#%%
## plot wavelet power diff (group level analysis)
fig, ax = plt.subplots()
im, cm = mne.viz.plot_topomap(up.mean(1), 
                              pos = epochs_eyes_open_pre.info, vmin = np.percentile(up.mean(1), 5),
                              vmax = np.percentile(up.mean(1), 95), cmap='Spectral_r', axes=ax, show=True)
fig.colorbar(im, ax=ax)      
plt.title('Up - Normalized theta (wavelet) power - diff')
plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/Resting_state/theta_up.jpg')

fig2, ax2 = plt.subplots()
im2, cm2 = mne.viz.plot_topomap(down.mean(1), 
                              pos = epochs_eyes_open_pre.info, vmin = np.percentile(down.mean(1), 5),
                              vmax = np.percentile(down.mean(1), 95), cmap='Spectral_r', axes=ax2, show=True)
fig2.colorbar(im2, ax=ax2)      
plt.title('Down - Normalized theta (wavelet) power - diff')
plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/Resting_state/theta_down.jpg')

fig3, ax3 = plt.subplots()
im3, cm3 = mne.viz.plot_topomap(sham.mean(1), 
                              pos = epochs_eyes_open_pre.info, vmin = np.percentile(sham.mean(1), 5),
                              vmax = np.percentile(sham.mean(1), 95), cmap='Spectral_r', axes=ax3, show=True)
fig3.colorbar(im3, ax=ax3)      
plt.title('Sham - Normalized theta (wavelet) power - diff')
plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/Resting_state/theta_sham.jpg')

#%%
times = np.arange(0,64,1)
# plt.subplot(211)
plt.plot(times, up.mean(axis=1) - sham.mean(axis=1),
         label="Contrast (Up - Sham)")
plt.plot(times, up.mean(axis=1) - down.mean(axis=1),
      label="Contrast (Up - Down)")
plt.plot(times, down.mean(axis=1) - sham.mean(axis=1),
   label="Contrast (Down - Sham)")

plt.ylabel("CMC difference")
plt.legend()
# plt.xlim(CMC_freqs[0], CMC_freqs[-1])
plt.subplot(212)
for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_pv[i_c] <= 0.05:
        h = plt.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
    else:
        plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                    alpha=0.3)
hf = plt.plot(times, t_obs, 'g')
plt.legend((h, ), ('cluster p-value < 0.05', ))
plt.xlabel("Channels")
plt.ylabel("f-values")
plt.show()

#%%

# # Plot PSD (group level analysis)
# plot_psd(psd, freqs, freq_range = (1,40), foi= (4,8), dB=True, ci=True)


# ## plot wavelet power diff (group level analysis)
# plt.figure()
# fig, ax = plt.subplots()
# im, cm = mne.viz.plot_topomap(open_post_pw - open_pre_pw, 
#                               pos = epochs_eyes_open_pre.info, vmin = np.percentile(open_post_pw - open_pre_pw, 5),
#                               vmax = np.percentile(open_post_pw - open_pre_pw, 95), cmap='Spectral_r', axes=ax, show=True)
# fig.colorbar(im, ax=ax)      
        
#%%

## Plot lagged coherence
# fig, ax = plt.subplots()
# im, cn = mne.viz.plot_topomap(np.asarray(lagged_theta_pre).mean(0), 
#                               pos = epochs_eyes_open_pre.info, vmin=np.percentile(np.asarray(lagged_theta).mean(0), 5), 
#                               vmax = np.percentile(np.asarray(lagged_theta).mean(0), 95), cmap='Spectral_r', axes=ax, show=True)
# fig.colorbar(im, ax=ax)


# fig2, ax2 = plt.subplots()
# im2, cm2 = mne.viz.plot_topomap(np.asarray(lagged_alpha).mean(0), 
#                                 pos = epochs_eyes_open_pre.info, vmin = np.percentile(np.asarray(lagged_alpha).mean(0), 5),
#                                 vmax = np.percentile(np.asarray(lagged_alpha).mean(0), 95), cmap='Spectral_r', axes=ax2, show=True)
# fig2.colorbar(im2, ax=ax2)

#%%        
# mutual_info = mutual_info_score(open_post_pw, close_post_pw)
# adj_mutual_info = adjusted_mutual_info_score(open_post_pw, close_post_pw, average_method='geometric')
