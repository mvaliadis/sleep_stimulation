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
import scipy.stats as stats 
import seaborn as sns
from neurodsp.rhythm import compute_lagged_coherence
from neurodsp.plts.rhythm import plot_lagged_coherence
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pingouin as pg
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from pymer4.utils import get_resource_path
from pymer4.models import Lmer
from functools import reduce
epo_example_obj = mne.read_epochs('/media/administrator/data/Study_1_data/ex_epo.fif')
   
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
    psd_results = defaultdict(lambda: [])
    # topo_results = defaultdict(lambda: [])
    for j, (pre_f, post_f) in tqdm(enumerate(zip(pre_files, post_files))):
        if pre_f.split('/')[-1].split('_')[0:2] == post_f.split('/')[-1].split('_')[0:2]:
            print(j, pre_f.split('/')[-1], post_f.split('/')[-1])
            subjname = " ".join(str(pre_f).split("/")[-1].split('_')[0:2])         
            night = int(subjname[-1]) + 1
            # session = str(pre_f).split("/")[-1].split('_')[2]
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
            epochs_eyes_open_pre.save('/media/administrator/data/Study_1_data/ex_epo.fif', overwrite=True)
            epochs_eyes_open_post = mne.EpochsArray(Data_post.data_eyes_open/1e3, Data_post.mne_info, 
                                                    tmin = 0, baseline=(None), verbose=0) 

            # from sleepstim.Analysis.foof import periodic_fit, channel_epoch_psd, oscillatory_plot_psd_map
            # post_psd = channel_epoch_psd(epochs_eyes_open_post, foi=(1, 40), tmin=0, tmax=2,
            #                              session='prepost')

            # pre_psd = channel_epoch_psd(epochs_eyes_open_pre, foi=(1, 40), tmin=0, tmax=2,
            #                             session='prepost')
            
            
            from mne.time_frequency import psd_multitaper
            bands=[(1, 4, 'Delta'), (4, 8, 'Theta'),(8, 13, 'Alpha'), 
                   (13, 30, 'Beta'), (30, 40, 'Gamma')]
            # bp_ = pd.DataFrame(columns=['Subject', 'Night', 'Condition', 'Session', 
            #                             'Chan', 'Delta', 'Theta', 'Alpha', 'Beta', 
            #                             'Gamma', 'TotalAbsPow', 'FreqRes', 'Relative'])
            for idx, (session, sess_name) in enumerate(zip([epochs_eyes_open_pre, epochs_eyes_open_post],['pre','post'])):
                spectra, freqs = psd_multitaper(session, fmin=1, fmax=40,
                                                tmin=0, tmax=2)
                if sess_name == 'pre':
                    bp_pre = yasa.bandpower_from_psd(spectra.mean(0), freqs, ch_names = session.ch_names[0:64],
                                                     bands=bands, relative=True)
                elif sess_name == 'post':
                    bp_post = yasa.bandpower_from_psd(spectra.mean(0), freqs, ch_names = session.ch_names[0:64],
                                                      bands=bands, relative=True)
                                        
                    
                # if j == 0 and idx == 0:
                #     bp = yasa.bandpower_from_psd(spectra.mean(0), freqs, ch_names = session.ch_names[0:64],
                #                                  bands=bands, relative=True)
                #     bp.insert(0, "Subject", subjname.split(" ")[0])
                #     bp.insert(1, "Night", night)
                #     bp.insert(2, "Condition", cond)
                #     bp.insert(3, "Session", sess_name)
                #     bp = bp_.append(bp, 1)
                # else:
                #     bp2 = yasa.bandpower_from_psd(spectra.mean(0), freqs, ch_names = session.ch_names[0:64],
                #                                   bands=bands, relative=True)
                #     bp2.insert(0, "Subject", subjname.split(" ")[0])
                #     bp2.insert(1, "Night", night)
                #     bp2.insert(2, "Condition", cond)
                #     bp2.insert(3, "Session", sess_name)
                #     bp = bp.append(bp2, 1)

            # fm_post = periodic_fit(epochs_eyes_open_post, foi=(1, 40), tmin=0, tmax=2)
            # fm_pre = periodic_fit(epochs_eyes_open_pre, foi=(1, 40), tmin=0, tmax=2)
            
            # post_freqs = np.unique([fm_post[i].freqs for i in range(len(fm_post))])
            # pre_freqs = np.unique([fm_pre[i].freqs for i in range(len(fm_pre))])
            
            # post_peak_fit = np.asarray([fm_post[i]._spectrum_flat for i in range(len(fm_post))])#.mean(0)
            # pre_peak_fit = np.asarray([fm_pre[i]._spectrum_flat for i in range(len(fm_pre))])#.mean(0)
            
            # # from fooof.plts.spectra import plot_spectrum
            # # plot_spectrum(pre_freqs, post_peak_fit - pre_peak_fit, color='green', label='Final Periodic Fit - Difference')
            # # plt.close('all')
            
            # Create dataframe with post - pre differences 
            psd_results["Subject"].extend([subjname.split(' ')[0]]*len(bp_post))
            psd_results["Night"].extend([night]*len(bp_post))
            psd_results["Condition"].extend([cond]*len(bp_post))
            psd_results["Channel"].extend(bp_post.Chan)
            psd_results["Delta"].extend(bp_post['Delta'] - bp_pre['Delta'])
            psd_results["Theta"].extend(bp_post['Theta'] - bp_pre['Theta'])
            psd_results["Alpha"].extend(bp_post['Alpha'] - bp_pre['Alpha'])
            psd_results["Beta"].extend(bp_post['Beta'] - bp_pre['Beta'])
            psd_results["Gamma"].extend(bp_post['Gamma'] - bp_pre['Gamma'])
            
            ###
            # post_psd_df = pd.DataFrame(post_psd, 
            #                            columns=['Epochs','Bands','Chans','PSD']).groupby(['Bands','Chans']).mean()['PSD']
            # pre_psd_df = pd.DataFrame(pre_psd, 
            #                           columns=['Epochs','Bands','Chans','PSD']).groupby(['Bands','Chans']).mean()['PSD']
            
            # topo_results["Subject"].extend([subjname.split(' ')[0]]*64)
            # topo_results["Night"].extend([night]*64)
            # topo_results["Condition"].extend([cond]*64)
            # # topo_results["Channels"].extend(post_psd['Chans'])
            # topo_results["Delta_PSD"].extend(post_psd_df.delta - pre_psd_df.delta)
            # topo_results["Theta_PSD"].extend(post_psd_df.theta - pre_psd_df.theta)
            # topo_results["Alpha_PSD"].extend(post_psd_df.alpha - pre_psd_df.alpha)
            # topo_results["Beta_PSD"].extend(post_psd_df.beta - pre_psd_df.beta)
            # topo_results["Gamma_PSD"].extend(post_psd_df.gamma - pre_psd_df.gamma)            
    
    return psd_results

def remove_subs(df):    
    for i in zip(['9PJZ8Z8F','475MQ9BL', '5LNKD1MG', 'CWESJCNJ']):
        df.drop(df.loc[df['Subject']==i[0]].index, inplace=True)
        
#%%
run = input('Do you wish to restart the resting state power analysis? ')
save_path = '/media/administrator/data/Study_1_data/Statistics/Resting_state/' 
if run == 'yes':
    psd_results = resting_state_power_analysis(path)
    df_psd = pd.DataFrame(psd_results)
    #df_psd = remove_subs(df_psd)
    df_psd.to_csv(save_path + 'rs_results_psd.csv')
else:
    df_psd = pd.read_csv(save_path + 'rs_results_psd.csv', index_col=0)

#%%

def plot_psd_diff(df_psd):
    common_subj = reduce(np.intersect1d, [df_psd[df_psd.Condition=='sham'].Subject, 
                                          df_psd[df_psd.Condition=='up'].Subject, 
                                          df_psd[df_psd.Condition=='down'].Subject])

    common_df = df_psd.loc[df_psd['Subject'].isin(common_subj)]
    common_df.sort_values(['Condition','Subject','Channel'], inplace=True) 
    fig, axes = plt.subplots(3, 5, figsize=(2 * 5, 3*1.5))
    grk = ['\u03B4', '\u03B8', '\u03B1', '\u03B2', '\u03B3']
    for idx, cond in enumerate(np.unique(df_psd.reset_index().Condition)):
        #print(cond.capitalize())
        #fig, axes = plt.subplots(1, 5, figsize=(2 * 5, 1.5))
        for ind, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']):
            #print(ind, band)
    
            # Get the PSD diff across channels for the current band
            bp_contrast = df_psd.groupby(['Condition','Channel'])[f'{band}'].mean()[cond]
            bp_contrast = bp_contrast.reindex(list(df_psd.Channel.unique()))
                   
            # Create a topomap for the current oscillation band
            im, cn = mne.viz.plot_topomap(bp_contrast, epo_example_obj.info, cmap='Spectral_r', contours=0,
                                          axes=axes[idx, ind], show=False, names=df_psd.Channel.unique());
           
            # add color bar
            mne.viz.topomap._add_colorbar(axes[idx, ind], im, cmap = 'Spectral_r', side='right', pad=0.05, 
                                          title=None, format=None, size='5%')
           
            # Set the plot title
            axes[idx, ind].set_title('\u0394 ' + grk[ind] + f' ({cond.capitalize()})')
           
            # tighten layout
            plt.tight_layout()
            
    # save fig
    plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/Resting_state/Topo_PSD.jpg')
  

#%%
def rs_topo_stats(df_topo, adjacency=None, plot=True, 
                  info=epo_example_obj, tfce=True):
    '''
    H1: Effect of condition on RS network
    df_topo = pd.read_csv(path + 'rs_results_topo.csv', index_col=0)
    rs_topo_stats(df_topo,adjacency)
    '''
    mne.set_log_level("CRITICAL")
    info.drop_channels(['EDC_L','ECR_L','FCR_L','FDS_L','ECG',
                        'EDC_R','ECR_R','FCR_R','FDS_R'])
            
    for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
        # parse dataframe by condition 
        
        ### Take only common subjects and convert to numpy accordingly
        common_subj = reduce(np.intersect1d, [df_topo[df_topo.Condition=='sham'].Subject, 
                                              df_topo[df_topo.Condition=='up'].Subject, 
                                              df_topo[df_topo.Condition=='down'].Subject])
    
        common_df = df_topo.loc[df_topo['Subject'].isin(common_subj)]
        common_df.sort_values(['Condition','Subject'], inplace=True) #Chan too?
        
        # parse conditions
        sham = common_df[common_df.Condition=='sham']
        up = common_df[common_df.Condition=='up']
        down = common_df[common_df.Condition=='down']

        # create contrast
        reshape_factor = len(common_df[common_df.Condition=='sham'].Subject.unique()), 1, 64
        up = up[f'{band}'].to_numpy().reshape(reshape_factor)
        down = down[f'{band}'].to_numpy().reshape(reshape_factor)
        sham = sham[f'{band}'].to_numpy().reshape(reshape_factor)
        contrast = np.concatenate([sham, up, down], 1)
        
        def stat_fun(*args):
            return mne.stats.f_mway_rm(np.swapaxes(args, 0, 0), factor_levels=[3],
                                       effects='all', return_pvals=False)[0]
    
        adjacency, ch_names = mne.channels.find_ch_adjacency(info.info, None)
        
        clus_kwargs = {'n_permutations' : 1024,  # 1000 is the minimum
                       'threshold' : dict(start=0, step=0.2), # None
                       'tail' : 1,               # two-tailed test (1 or -1 for one-tailed)
                       'n_jobs' : -1,            # increase value to speed up computations
                       'buffer_size' : None,
                       'out_type' : 'mask',      # returns a mask map instead of indices of sig. points
                       'seed' : 1503}
        
        F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
            contrast, stat_fun=stat_fun, adjacency=adjacency, **clus_kwargs)
        
        good_clusters = np.where(cluster_p_values < .05)[0]
        mask = cluster_p_values < .05
        pmin = np.min(cluster_p_values)
        if len(good_clusters) > 0:
            sig_chan = ch_names[good_clusters]
        else:
            sig_chan = 'None'
        
        print('Min p-val: {}\nSignificant Chans: {}'.format(pmin, sig_chan))
    
        # plot t-contrast with significance mask
        if plot:
            fig, ax = plt.subplots(figsize=(6,4))
            im, _ = mne.viz.plot_topomap(F_obs.squeeze(), pos=info.info, 
                                         mask=mask, axes=ax, show=0, 
                                         cmap='Spectral_r', names=ch_names, 
                                         show_names=False, contours = 0,
                                         mask_params=dict(markersize=10, markerfacecolor='y'))
            cbar = fig.colorbar(im, ax=ax)   
            cbar.ax.set_ylabel('f-stat',rotation=270)
            ax.set_title(f'Resting State Power - {band} Power')
            plt.tight_layout()
            plt.show()
            
#%%
sns.set_theme(color_codes=True)
# df_psd.dropna(inplace=True) 
for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']): 
    print(f'Running LMM for frequency band: {band}')
    
    model = Lmer(f"{band} ~ Condition*Channel + (1|Subject)", 
                 data=df_psd)
    
    # Using dummy-coding; suppress summary output
    model.fit(factors={"Condition": ["sham", "up", "down"],
                       "Channel" : list(df_psd.Channel.unique())},
              ordered=True, summarize=True)
    
    # Get ANOVA table, but force orthogonality for valid SS III inferences
    # In this case the data is unbalaced, otherwise nothing changes
    print(model.anova(force_orthogonal=True))
    
    ## Post-hoc tests 
    marginal_estimates, comparisons = model.post_hoc(p_adjust="fdr",
                                                     marginal_vars=['Condition','Channel'],
                                                     grouping_vars=None
                                                     )
    
    print(marginal_estimates)
    print(comparisons)
 
# Plot
for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']):
    plt.figure()
    sns.violinplot(data = df_psd, y=f'{band}',
                   x='Condition', dodge=True)
    plt.tight_layout()
    sns.despine()      
    plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/Resting_state/{band}_diff.jpg')
    plt.close('all')


#%%

# freqs_pre = df.groupby('Frequencies (pre)').mean()['Frequencies (post)'].to_numpy()
# fit_pre = df.groupby('Frequencies (pre)').mean()['Periodic fit (pre)'].to_numpy()

# freqs_post = df.groupby('Frequencies (post)').mean()['Frequencies (pre)'].to_numpy()
# fit_post = df.groupby('Frequencies (post)').mean()['Periodic fit (post)'].to_numpy()

# plt.plot(freqs_pre, fit_pre, label='pre')
# plt.plot(freqs_post, fit_post, label='post')
# plt.legend()

# sns.lineplot(data = df, x = 'Frequencies (post)', y = 'Periodic fit (post)', hue = 'Condition',
#              ci = None)
# sns.despine()
# plt.figure()
# sns.lineplot(data = df, x = 'Frequencies (pre)', y = 'Periodic fit (pre)', hue = 'Condition',
#              ci = None)
# sns.despine()

# #%%

# info = mne.create_info(ch_names=epochs_eyes_open_pre.info.ch_names[0:64], sfreq=epochs_eyes_open_pre.info['sfreq'], ch_types=['eeg']*64)
# info.set_montage(mne.channels.make_standard_montage('standard_1005'))
# sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type = 'eeg')

# # X = np.vstack([df_bp[df_bp.band=='Delta'][df_bp.cond=='down']['band_power'].to_numpy()[0], 
# #                df_bp[df_bp.band=='Delta'][df_bp.cond=='up']['band_power'].to_numpy()[0], 
# #                df_bp[df_bp.band=='Delta'][df_bp.cond=='sham']['band_power'].to_numpy()[0]]).T

# t_obs, clusters, cluster_pv, H0 = mne.stats.spatio_temporal_cluster_test(X, n_permutations=1024, 
#                                                                          adjacency=sensor_adjacency)

#%%
# sham = df[df['Condition']=='sham']['Normalized theta (wavelet) power - diff'].to_numpy()
# sham = sham.reshape(64, int(len(sham)/64))
    
# up = df[df['Condition']=='up']['Normalized theta (wavelet) power - diff'].to_numpy()
# up = up.reshape(64, int(len(up)/64))
    
# down = df[df['Condition']=='down']['Normalized theta (wavelet) power - diff'].to_numpy()
# down = down.reshape(64, int(len(down)/64))

# # rmanova = pg.rm_anova(dv='Normalized theta (wavelet) power - diff', within=['Condition','Channels'], 
# #                       subject='Subject', detailed = True, data=df)
# # # Pretty printing of ANOVA summary
# # pg.print_table(rmanova)
# # # Post hoc analysis
# # posthocs = pg.pairwise_ttests(dv='Normalized theta (wavelet) power - diff', within='Condition',
# #                               subject='Subject', data=df)
# # pg.print_table(posthocs)

# t_obs, clusters, cluster_pv, H0 = mne.stats.spatio_temporal_cluster_test([down.T, up.T, sham.T], 
#                                                                           n_permutations=1000,
#                                                                           tail=1, n_jobs=1,
#                                                                           out_type='mask')

# #%%
# ## plot wavelet power diff (group level analysis)
# fig, ax = plt.subplots()
# im, cm = mne.viz.plot_topomap(up.mean(1), 
#                               pos = epochs_eyes_open_pre.info, vmin = np.percentile(up.mean(1), 5),
#                               vmax = np.percentile(up.mean(1), 95), cmap='Spectral_r', axes=ax, show=True)
# fig.colorbar(im, ax=ax)      
# plt.title('Up - Normalized theta (wavelet) power - diff')
# plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/Resting_state/theta_up.jpg')

# fig2, ax2 = plt.subplots()
# im2, cm2 = mne.viz.plot_topomap(down.mean(1), 
#                               pos = epochs_eyes_open_pre.info, vmin = np.percentile(down.mean(1), 5),
#                               vmax = np.percentile(down.mean(1), 95), cmap='Spectral_r', axes=ax2, show=True)
# fig2.colorbar(im2, ax=ax2)      
# plt.title('Down - Normalized theta (wavelet) power - diff')
# plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/Resting_state/theta_down.jpg')

# fig3, ax3 = plt.subplots()
# im3, cm3 = mne.viz.plot_topomap(sham.mean(1), 
#                               pos = epochs_eyes_open_pre.info, vmin = np.percentile(sham.mean(1), 5),
#                               vmax = np.percentile(sham.mean(1), 95), cmap='Spectral_r', axes=ax3, show=True)
# fig3.colorbar(im3, ax=ax3)      
# plt.title('Sham - Normalized theta (wavelet) power - diff')
# plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/Resting_state/theta_sham.jpg')

#%%
## Plot PSD (group level analysis)
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