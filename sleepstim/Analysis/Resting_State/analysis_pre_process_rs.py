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
    topo_results = defaultdict(lambda: [])
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
            # epochs_eyes_closed_pre = mne.EpochsArray(Data_pre.data_eyes_close/1e3, Data_pre.mne_info, 
            #                                           tmin = 0, baseline=(None), verbose=0) 
            epochs_eyes_open_post = mne.EpochsArray(Data_post.data_eyes_open/1e3, Data_post.mne_info, 
                                                    tmin = 0, baseline=(None), verbose=0) 
            # epochs_eyes_closed_post = mne.EpochsArray(Data_post.data_eyes_close/1e3, Data_post.mne_info, 
            #                                           tmin = 0, baseline=(None), verbose=0) 
    
    
    
            from sleepstim.Analysis.foof import oscillatory_plot_psd_map, periodic_fit
            post_psd = oscillatory_plot_psd_map(epochs_eyes_open_post, foi=(1, 40), tmin=0, tmax=2,
                                    session='prepost')

            pre_psd = oscillatory_plot_psd_map(epochs_eyes_open_pre, foi=(1, 40), tmin=0, tmax=2,
                                    session='prepost')
            
            fm_post = periodic_fit(epochs_eyes_open_post, foi=(1, 40), tmin=0, tmax=2)
            fm_pre = periodic_fit(epochs_eyes_open_pre, foi=(1, 40), tmin=0, tmax=2)
            
            post_freqs = np.unique([fm_post[i].freqs for i in range(len(fm_post))])
            pre_freqs = np.unique([fm_pre[i].freqs for i in range(len(fm_pre))])
            
            post_peak_fit = np.asarray([fm_post[i]._peak_fit for i in range(len(fm_post))])
            pre_peak_fit = np.asarray([fm_pre[i]._peak_fit for i in range(len(fm_pre))])
            
            plt.close('all')
            
            from fooof.plts.spectra import plot_spectrum
            #plot_spectrum(post_freqs, post_peak_fit, color='green', label='Final Periodic Fit - Post Session')
            #plot_spectrum(pre_freqs, pre_peak_fit, color='green', label='Final Periodic Fit - Pre Session')
        
            ## Compute relative PSD per band
            bp_pre = yasa.bandpower_from_psd(pre_peak_fit, pre_freqs, 
                                             bands=[(1, 4, 'Delta'), (4, 8, 'Theta'),
                                                    (8, 12, 'Alpha'), (12, 30, 'Beta'), 
                                                    (30, 40, 'Gamma')], relative=False)
            bp_pre = bp_pre.rename(columns={'Chan': 'Epoch'})
            bp_pre['Epoch'] = np.arange(0, len(pre_peak_fit), 1)
            
            bp_post = yasa.bandpower_from_psd(post_peak_fit, post_freqs, 
                                              bands=[(1, 4, 'Delta'), (4, 8, 'Theta'),
                                                     (8, 12, 'Alpha'), (12, 30, 'Beta'), 
                                                     (30, 40, 'Gamma')], relative=False)
            bp_post = bp_post.rename(columns={'Chan': 'Epoch'})
            bp_post['Epoch'] = np.arange(0, len(post_peak_fit), 1)
            
            
            ## Create dataframe with post - pre differences 
            psd_results["Subject"].extend([subjname]*len(bp_pre))
            psd_results["Night"].extend([night]*len(bp_pre))
            psd_results["Condition"].extend([cond]*len(bp_pre))
            psd_results["Session"].extend(['pre']*len(bp_pre))
            psd_results["Delta_pre"].extend(stats.zscore(bp_pre['Delta']))
            psd_results["Theta_pre"].extend(stats.zscore(bp_pre['Theta']))
            psd_results["Alpha_pre"].extend(stats.zscore(bp_pre['Alpha']))
            psd_results["Beta_pre"].extend(stats.zscore(bp_pre['Beta']))
            psd_results["Gamma_pre"].extend(stats.zscore(bp_pre['Gamma']))
            # psd_results["Subject"].extend([subjname]*len(bp_post))
            # psd_results["Night"].extend([night]*len(bp_post))
            # psd_results["Condition"].extend([cond]*len(bp_post))
            # psd_results["Session"].extend(['post']*len(bp_post))
            psd_results["Delta_DV"].extend(bp_post['Delta'] - bp_pre['Delta'])
            psd_results["Theta_DV"].extend(bp_post['Theta'] - bp_pre['Theta'])
            psd_results["Alpha_DV"].extend(bp_post['Alpha'] - bp_pre['Alpha'])
            psd_results["Beta_DV"].extend(bp_post['Beta'] - bp_pre['Beta'])
            psd_results["Gamma_DV"].extend(bp_post['Gamma'] - bp_pre['Gamma'])
            
            ###
            topo_results["Subject"].extend([subjname]*64)
            topo_results["Night"].extend([night]*64)
            topo_results["Condition"].extend([cond]*64)
            topo_results["Session"].extend(['pre']*64)
            topo_results["Delta_topography"].extend(pre_psd[0])
            topo_results["Theta_topography"].extend(pre_psd[1])
            topo_results["Alpha_topography"].extend(pre_psd[2])
            topo_results["Beta_topography"].extend(pre_psd[3])
            topo_results["Gamma_topography"].extend(pre_psd[4])
            
            topo_results["Subject"].extend([subjname]*64)
            topo_results["Night"].extend([night]*64)
            topo_results["Condition"].extend([cond]*64)
            topo_results["Session"].extend(['post']*64)
            topo_results["Delta_topography"].extend(post_psd[0])
            topo_results["Theta_topography"].extend(post_psd[1])
            topo_results["Alpha_topography"].extend(post_psd[2])
            topo_results["Beta_topography"].extend(post_psd[3])
            topo_results["Gamma_topography"].extend(post_psd[4])
            
    
    return topo_results, psd_results


#%%
run = input('Do you wish to restart the resting state power analysis? ')
if run == 'yes':
    topo_results, psd_results = resting_state_power_analysis(path)
    df_topo, df_psd = pd.DataFrame(topo_results), pd.DataFrame(psd_results)
    save_path = '/media/administrator/data/Study_1_data/Statistics/Resting_state/rs_results.p'
    pickle.dump(df, open(save_path, "wb"))  
else:
    df = pickle.load(open('/media/administrator/data/Study_1_data/Statistics/Resting_state/rs_results.p', 'rb'))
    df.to_csv(r'/media/administrator/data/Study_1_data/Statistics/Resting_state/rs_results.csv')


#%%
sub = df_topo['Subject'].to_numpy()
subs = [sub[i][0:8] for i in range(len(sub))]
df_topo['Subject'] = subs

chan = epochs_eyes_open_pre.info['ch_names'][0:64]
# dup_chans = list(itertools.chain(*zip(chan,chan)))
dup_chans = chan + chan
stacks = []
for i in range(int(len(df_topo)/128)):
    stacks.append(dup_chans)
    # stacks.append(np.arange(0,64,1))
chans = np.hstack(stacks)
df_topo['Channels'] = chans

subjects = df_topo['Subject'].unique()
for i in zip(['9PJZ8Z8F','475MQ9BL', '5LNKD1MG', 'CWESJCNJ']):
    df_topo.drop(df_topo.loc[df_topo['Subject']==i[0]].index, inplace=True)
    
#%%
sub = df_psd['Subject'].to_numpy()
subs = [sub[i][0:8] for i in range(len(sub))]
df_psd['Subject'] = subs

subjects = df_psd['Subject'].unique()
for i in zip(['9PJZ8Z8F','475MQ9BL', '5LNKD1MG', 'CWESJCNJ']):
    df_psd.drop(df_psd.loc[df_psd['Subject']==i[0]].index, inplace=True)

#%%
freqs_pre = df.groupby('Frequencies (pre)').mean()['Frequencies (post)'].to_numpy()
fit_pre = df.groupby('Frequencies (pre)').mean()['Periodic fit (pre)'].to_numpy()

freqs_post = df.groupby('Frequencies (post)').mean()['Frequencies (pre)'].to_numpy()
fit_post = df.groupby('Frequencies (post)').mean()['Periodic fit (post)'].to_numpy()

plt.plot(freqs_pre, fit_pre, label='pre')
plt.plot(freqs_post, fit_post, label='post')
plt.legend()

sns.lineplot(data = df, x = 'Frequencies (post)', y = 'Periodic fit (post)', hue = 'Condition',
             ci = None)
sns.despine()
plt.figure()
sns.lineplot(data = df, x = 'Frequencies (pre)', y = 'Periodic fit (pre)', hue = 'Condition',
             ci = None)
sns.despine()

#%%

results = defaultdict(lambda: [])
for cond in np.unique(df_topo.Condition):
    # print(cond)
    fig, axes = plt.subplots(1, 5, figsize=(2 * 5, 1.5))
    for ind, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']):
        # print(ind, band)

        # Get the power values across channels for the current band
        band_power_pre = df_topo.groupby(['Session','Condition','Channels'])[f'{band}_topography'].mean()['pre'][cond].to_numpy()
        band_power_post = df_topo.groupby(['Session','Condition','Channels'])[f'{band}_topography'].mean()['post'][cond].to_numpy()
        band_power = band_power_post - band_power_pre

        results["cond"].extend([cond])
        results["band"].extend([band])
        results["band_power"].extend([band_power])
        
        # Create a topomap for the current oscillation band
        im, cn = mne.viz.plot_topomap(band_power, epochs_eyes_open_pre.info, cmap='Spectral_r', contours=0,
                                      axes=axes[ind], show=False);
       
        # add color bar
        mne.viz.topomap._add_colorbar(axes[ind], im, cmap = 'Spectral_r', side='right', pad=0.05, 
                                      title=None, format=None, size='5%')
       
        # Set the plot title
        axes[ind].set_title(band + ' power')
       
        # tighten layout
        plt.tight_layout()

df_bp = pd.DataFrame(results)

#%%

lmm_ch_power = []
for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']): 
    print(i, band)
    for chan in df.Channels.unique():
        print(f'Running LMM for frequency band: {band} and channel: {chan}')
        
        ## Statsmodels based LMM
        # warnings.simplefilter("ignore", ConvergenceWarning)
        # md = smf.mixedlm(f"{band}_topography_diff ~ Condition", df[df.Channels==chan],
        #                  groups=df[df.Channels==chan]["Subject"])
        # mdf = md.fit()   
        # print(mdf.summary())
        
        
        # We're going to fit a multi-level regression using the
        # categorical predictor which has 3 levels
        # di = {"sham" : 1.0, "up" : 0.5, "down" : 1.5}
        # df = df.replace({"Condition": di})
        
        ## Pymer based LMM
        # data frame by channel
        df_dummy = df_topo #[df_topo.Channels==chan]
        
        model = Lmer(f"{band}_topography ~ Condition*Channels*Session + (1|Subject)", 
                     data=df_dummy)
        
        # Using dummy-coding; suppress summary output
        model.fit(factors={"Condition": ["sham", "up", "down"],
                           "Channels" : list(df_dummy['Channels'].unique()),
                           "Session" : ["pre", "post"]}, 
                  ordered=True, summarize=False)
        
        # Get ANOVA table, but this time force orthogonality for valid SS III inferences
        # In this case the data are balanced so nothing changes
        print(model.anova(force_orthogonal=True))
        
    
        ## Post-hoc tests 
        marginal_estimates, comparisons = model.post_hoc(p_adjust="fdr",
                                                         marginal_vars='Session',
                                                         grouping_vars='Condition',
                                                         )
        
        print(marginal_estimates)
        print(comparisons)
   

#%%
sns.set_theme(color_codes=True)
lmm_power = []
df_psd.dropna(inplace=True) 
for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']): 
    print(f'Running LMM for frequency band: {band}')
    
    model = Lmer(f"{band}_DV ~ Condition*{band}_pre + (1|Subject)", 
                 data=df_psd)
    
    # Using dummy-coding; suppress summary output
    model.fit(factors={"Condition": ["sham", "up", "down"],
                       # "Session" : ["pre", "post"]}, 
                       },
              ordered=True, summarize=False)
    
    # Get ANOVA table, but this time force orthogonality for valid SS III inferences
    # In this case the data are balanced so nothing changes
    print(model.anova(force_orthogonal=True))
    
    # Plot
    ax = sns.pointplot(data = df_psd, x='Session', y=f'Relative_{band}', 
                       hue='Condition', estimator=np.mean, ci=95, dodge=True)
    plt.tight_layout()
    sns.despine()
    
    ## Post-hoc tests 
    marginal_estimates, comparisons = model.post_hoc(p_adjust="fdr",
                                                     marginal_vars='Session',
                                                     grouping_vars='Condition',
                                                     )
    
    print(marginal_estimates)
    print(comparisons)
             

#%%

info = mne.create_info(ch_names=epochs_eyes_open_pre.info.ch_names[0:64], sfreq=epochs_eyes_open_pre.info['sfreq'], ch_types=['eeg']*64)
info.set_montage(mne.channels.make_standard_montage('standard_1005'))
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type = 'eeg')

# X = np.vstack([df_bp[df_bp.band=='Delta'][df_bp.cond=='down']['band_power'].to_numpy()[0], 
#                df_bp[df_bp.band=='Delta'][df_bp.cond=='up']['band_power'].to_numpy()[0], 
#                df_bp[df_bp.band=='Delta'][df_bp.cond=='sham']['band_power'].to_numpy()[0]]).T

t_obs, clusters, cluster_pv, H0 = mne.stats.spatio_temporal_cluster_test(X, n_permutations=1024, 
                                                                         adjacency=sensor_adjacency)

#%%

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

t_obs, clusters, cluster_pv, H0 = mne.stats.spatio_temporal_cluster_test([down.T, up.T, sham.T], 
                                                                         n_permutations=1000,
                                                                         tail=1, n_jobs=1,
                                                                         out_type='mask')

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

#%%        
# mutual_info = mutual_info_score(open_post_pw, close_post_pw)
# adj_mutual_info = adjusted_mutual_info_score(open_post_pw, close_post_pw, average_method='geometric')
