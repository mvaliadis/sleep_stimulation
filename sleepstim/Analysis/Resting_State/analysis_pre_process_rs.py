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
from sleepstim.core.dsp import bandpower # Changed import
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
from tqdm import tqdm
from pymer4.utils import get_resource_path
from pymer4.models import Lmer
from functools import reduce
import psutil
# Import the FOOOF object
from fooof.objs.utils import combine_fooofs
from fooof import FOOOF, FOOOFGroup, fit_fooof_3d
from fooof.plts.spectra import plot_spectrum
from fooof.plts.annotate import plot_annotated_peak_search
epo_example_obj = mne.read_epochs('/media/administrator/data/Study_1_data/ex_epo.fif')
#%matplotlib 
good_subs = ['0RCB4IRJ', '3LFLTILW', '6QJ3ITMT', '7XVWEVOK', 'A4VCLD2I', 'CWESJCNJ', 
             'D1BOI2AY', 'FUOPOVNF', 'HTXEYPW6', 'IYPJJ2KE', 'KEQB5AWM', 'RVQL2MRD', 
             'UDLD86TO', 'W6AX3IMN', 'Y9VJUA9F', 'YIOYSRPX', 'EQDORXF6'] #'886MCPKG', 'IBYYXKMB']

#%%
## Step 1 -- Preprocess resting state datasets and save into data structure with epochs for eyes open/eyes closed  
path = '/media/administrator/data/Study_1_data/Pre_post_data/'
files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files if 'resting_state' in i])
for i, files in enumerate(tqdm(files_list)):
    subj_cond = files.split("/")[-2] + "_" + files.split("/")[-1].split("_")[1] + "_" + files.split("/")[-1].split("_")[-2]
    save_path = '/media/administrator/data/Study_1_data/Pre-processed_data_resting_state/' + subj_cond + '_rs_preprocessed.p'
    print(i, files.split('/')[-2] + '_' + files.split('/')[-1])
    if not os.path.exists(save_path):
        if psutil.virtual_memory().percent < 85:
            _pre_process_rs_data(files = files, csd = True, prep_pipeline = False, art_method = 'covar', save = True)
        else:
            break
        
#%%
## Step 2 -- Load in epoched data and compute connectivity/power analysis/etc.
path = '/media/administrator/data/Study_1_data/Pre-processed_data_resting_state/'
def resting_state_power_analysis(path):
    pre_files, post_files = check_match_prepost_data_elements(path, files=None, dtype='rs')
    psd_results = defaultdict(lambda: [])
    # topo_results = defaultdict(lambda: [])
    bp_pre_open = pd.DataFrame(columns=['Chan', 'Delta', 'Theta', 'Alpha', 'Beta', 
                                       'Gamma', 'TotalAbsPow','FreqRes', 'Relative',
                                       'Session', 'Eyes'])
    bp_pre_close = pd.DataFrame(columns=['Chan', 'Delta', 'Theta', 'Alpha', 'Beta', 
                                       'Gamma', 'TotalAbsPow','FreqRes', 'Relative',
                                       'Session', 'Eyes'])
    bp_post_open = pd.DataFrame(columns=['Chan', 'Delta', 'Theta', 'Alpha', 'Beta', 
                                    'Gamma', 'TotalAbsPow','FreqRes', 'Relative',
                                    'Session', 'Eyes'])
    bp_post_close = pd.DataFrame(columns=['Chan', 'Delta', 'Theta', 'Alpha', 'Beta', 
                                    'Gamma', 'TotalAbsPow','FreqRes', 'Relative',
                                    'Session', 'Eyes'])
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
            # eyes open epochs
            epochs_eyes_open_pre = mne.EpochsArray(Data_pre.data_eyes_open/1e3, Data_pre.mne_info, 
                                                   tmin = 0, baseline=(None), verbose=0) 
            #epochs_eyes_open_pre.save('/media/administrator/data/Study_1_data/ex_epo.fif', overwrite=True)
            epochs_eyes_open_post = mne.EpochsArray(Data_post.data_eyes_open/1e3, Data_post.mne_info, 
                                                    tmin = 0, baseline=(None), verbose=0) 
            # eyes closed epochs
            epochs_eyes_close_pre = mne.EpochsArray(Data_pre.data_eyes_close/1e3, Data_pre.mne_info, 
                                                    tmin = 0, baseline=(None), verbose=0) 
            epochs_eyes_close_post = mne.EpochsArray(Data_post.data_eyes_close/1e3, Data_post.mne_info, 
                                                    tmin = 0, baseline=(None), verbose=0) 

            # from sleepstim.Analysis.foof import periodic_fit, channel_epoch_psd, oscillatory_plot_psd_map
            # post_psd = channel_epoch_psd(epochs_eyes_open_post, foi=(1, 40), tmin=0, tmax=2,
            #                              session='prepost')

            # pre_psd = channel_epoch_psd(epochs_eyes_open_pre, foi=(1, 40), tmin=0, tmax=2,
            #                             session='prepost')
            
            # from mne.time_frequency import psd_multitaper -> deprecated
            bands=[(1, 4, 'Delta'), (4, 7, 'Theta'),(8, 14, 'Alpha'), 
                   (14, 30, 'Beta'), (30, 40, 'Gamma')]
            # Create dataframe 
            for idx, (post, pre, eyes) in enumerate(zip([epochs_eyes_open_post, epochs_eyes_close_post],
                                                        [epochs_eyes_open_pre, epochs_eyes_close_pre],
                                                        ['Open','Close'])):
                print(idx, pre, post, eyes)
                spectra_post, freqs_post = post.compute_psd(method="multitaper", fmin=1, fmax=40,
                                                            tmin=0, tmax=2).get_data(return_freqs=True)
                bp_post = yasa.bandpower_from_psd(spectra_post.mean(0), freqs_post, 
                                                  ch_names = post.ch_names[0:64],
                                                  bands=bands, relative=True)
                bp_post2 = yasa.bandpower_from_psd(spectra_post.mean(0), freqs_post, 
                                                   ch_names = post.ch_names[0:64],
                                                   bands=bands, relative=False)
                
                # fooof it!
                fg = FOOOFGroup(peak_width_limits=[1, 8], min_peak_height=0.1, 
                                max_n_peaks=6, peak_threshold=1.5)
                fgs = fit_fooof_3d(fg, freqs_post, spectra_post.mean(0, keepdims=True))[0]
                peak_fits = []
                chs = []
                for idx, f in enumerate(zip(fgs, post.ch_names[0:64])):
                    #print(idx, f[1])
                    fooofy = fgs.get_fooof(ind=idx, regenerate=True)
                    # peak_fit = fooofy._peak_fit
                    # peak_fits.append(peak_fit.reshape(1,-1))
                    spectral_slope = fooofy.get_params('aperiodic_params', col='exponent')
                    peak_fits.append(spectral_slope)
                    chs.append(f[1])
                    # init_flat_spec = fm.power_spectrum - fm._ap_fit
                    # plts.plot_spectrum(fm.freqs, fm._peak_fit, color='green', label='Final Periodic Fit')
                    
                # bp_post3 = yasa.bandpower_from_psd(np.concatenate(peak_fits, 0), 
                #                                    freqs_post, ch_names=chs,
                #                                    bands=bands, relative=True)
                bp_post3 = pd.DataFrame({'Chan' : chs,
                                         'Exponent' : np.asarray(peak_fits)})
     
                
                spectra_pre, freqs_pre = pre.compute_psd(method="multitaper", fmin=1, fmax=40,
                                                         tmin=0, tmax=2).get_data(return_freqs=True)
                bp_pre = yasa.bandpower_from_psd(spectra_pre.mean(0), freqs_pre, 
                                                 ch_names = post.ch_names[0:64],
                                                 bands=bands, relative=True)
                bp_pre2 = yasa.bandpower_from_psd(spectra_pre.mean(0), freqs_pre, 
                                                  ch_names = post.ch_names[0:64],
                                                  bands=bands, relative=False)
                
                # fooof it!
                fg = FOOOFGroup(peak_width_limits=[1, 8], min_peak_height=0.1, 
                                max_n_peaks=6, peak_threshold=1.5)
                fgs = fit_fooof_3d(fg, freqs_pre, spectra_pre.mean(0, keepdims=True))[0]
                peak_fits = []
                chs = []
                for idx, f in enumerate(zip(fgs, pre.ch_names[0:64])):
                    #print(idx, f[1])
                    fooofy = fgs.get_fooof(ind=idx, regenerate=True)
                    # peak_fit = fooofy._peak_fit
                    # peak_fits.append(peak_fit.reshape(1,-1))
                    spectral_slope = fooofy.get_params('aperiodic_params', col='exponent')
                    peak_fits.append(spectral_slope)
                    chs.append(f[1])
                    # init_flat_spec = fm.power_spectrum - fm._ap_fit
                    # plts.plot_spectrum(fm.freqs, fm._peak_fit, color='green', label='Final Periodic Fit')
                    # report 
                    
                # bp_pre3 = yasa.bandpower_from_psd(np.concatenate(peak_fits, 0), 
                #                                   freqs_post, ch_names=chs,
                #                                   bands=bands, relative=True)
                bp_pre3 = pd.DataFrame({'Chan' : chs,
                                        'Exponent' : np.asarray(peak_fits)})
                
                # append results
                psd_results["Subject"].extend([subjname.split(' ')[0]]*len(bp_pre))
                psd_results["Night"].extend([night]*len(bp_pre))
                psd_results["Condition"].extend([cond]*len(bp_pre))
                psd_results["Channel"].extend(bp_pre.Chan)
                psd_results["Session"].extend([eyes]*len(bp_pre))
                psd_results["Delta"].extend(bp_post.Delta.to_numpy() - 
                                            bp_pre.Delta.to_numpy())
                psd_results["Delta_log"].extend(np.log10(bp_post2.Delta.to_numpy()) -
                                                np.log10(bp_pre2.Delta.to_numpy()))
                # psd_results["Delta_fooof"].extend(bp_post3.Delta.to_numpy() -
                #                                   bp_pre3.Delta.to_numpy())
                psd_results["Theta"].extend(bp_post.Theta.to_numpy() - 
                                            bp_pre.Theta.to_numpy())
                psd_results["Theta_log"].extend(np.log10(bp_post2.Theta.to_numpy()) -
                                                np.log10(bp_pre2.Theta.to_numpy()))
                # psd_results["Theta_fooof"].extend(bp_post3.Theta.to_numpy() -
                #                                   bp_pre3.Theta.to_numpy())
                psd_results["Alpha"].extend(bp_post.Alpha.to_numpy() - 
                                            bp_pre.Alpha.to_numpy())
                psd_results["Alpha_log"].extend(np.log10(bp_post2.Alpha.to_numpy()) -
                                                np.log10(bp_pre2.Alpha.to_numpy()))
                # psd_results["Alpha_fooof"].extend(bp_post3.Alpha.to_numpy() -
                #                                   bp_pre3.Alpha.to_numpy())
                psd_results["Beta"].extend(bp_post.Beta.to_numpy() - 
                                           bp_pre.Beta.to_numpy())
                psd_results["Beta_log"].extend(np.log10(bp_post2.Beta.to_numpy()) -
                                               np.log10(bp_pre2.Beta.to_numpy()))
                # psd_results["Beta_fooof"].extend(bp_post3.Beta.to_numpy() -
                #                                   bp_pre3.Beta.to_numpy())
                psd_results["Gamma"].extend(bp_post.Gamma.to_numpy() - 
                                            bp_pre.Gamma.to_numpy())
                psd_results["Gamma_log"].extend(np.log10(bp_post2.Gamma.to_numpy()) -
                                                np.log10(bp_pre2.Gamma.to_numpy()))
                # psd_results["Gamma_fooof"].extend(bp_post3.Gamma.to_numpy() -
                #                                   bp_pre3.Gamma.to_numpy())
                psd_results['Spectral_exponent'].extend(bp_post3.Exponent.to_numpy() - 
                                                        bp_pre3.Exponent.to_numpy())

            
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
            
            # ###
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

# def remove_subs(df):    
#     for i in zip(['9PJZ8Z8F','475MQ9BL','5LNKD1MG','CWESJCNJ']):
#         df.drop(df.loc[df['Subject']==i[0]].index, inplace=True)
        
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
    df_psd = df_psd.loc[df_psd['Subject'].isin(good_subs)]
    
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
            im, cn = mne.viz.plot_topomap(bp_contrast, epo_example_obj.info, cmap='Spectral_r', 
                                          contours=0, axes=axes[idx, ind], 
                                          show=False, names=None); #df_psd.Channel.unique());
           
            # add color bar
            mne.viz.topomap._add_colorbar(axes[idx, ind], im, cmap = 'Spectral_r', side='right', pad=0.05, 
                                          title=None, format=None, size='5%')
           
            # Set the plot title
            axes[idx, ind].set_title('\u0394 ' + grk[ind] + f' ({cond.capitalize()})')
           
            # tighten layout
            plt.tight_layout()
            
    # save fig
    plt.savefig('f/media/administrator/data/Study_1_data/Statistics/Resting_state/Topo_PSD.jpg')
  

#%%           
def rs_topo_stats(df_topo, adjacency=None, plot=True, 
                  info=epo_example_obj.copy(), tfce=True):
    '''
    H1: Effect of condition on RS network
    df_topo = pd.read_csv(path + 'rs_results_topo.csv', index_col=0)
    rs_topo_stats(df_topo,adjacency)
    '''
    mne.set_log_level("CRITICAL")
    info.drop_channels(['EDC_L','ECR_L','FCR_L','FDS_L','ECG',
                        'EDC_R','ECR_R','FCR_R','FDS_R'])
    
    sess = 'Open'
    grk = ['\u03B8', '\u03B1', '\u03B2']
    
    #pairwise_tests = []
    # for sess in list(df_topo.Session.unique()): 
    # for band in ['Delta','Delta_fooof','Delta_log','Theta','Theta_fooof','Theta_log',
    #              'Alpha','Alpha_fooof','Alpha_log', 'Beta','Beta_fooof', 'Beta_log',
    #              'Gamma', 'Gamma_fooof', 'Gamma_log']:
    for i, band in enumerate(['Theta','Alpha','Beta']):
        grk_now = grk[i]
        # parse dataframe by condition 
        
        ### Take only common subjects and convert to numpy accordingly
        common_subj = reduce(np.intersect1d, [df_topo[df_topo.Condition=='sham'].Subject, 
                                              df_topo[df_topo.Condition=='up'].Subject, 
                                              df_topo[df_topo.Condition=='down'].Subject])
    
        common_df = df_topo.loc[df_topo['Subject'].isin(common_subj)]
        #common_df.sort_values(['Condition','Subject'], inplace=True) #Chan too?
        common_df = common_df[common_df.Session=='Open']
        
        # parse conditions
        sham = common_df[common_df.Condition=='sham'].fillna(0)
        up = common_df[common_df.Condition=='up'].fillna(0)
        down = common_df[common_df.Condition=='down'].fillna(0)

        # create contrast
        reshape_factor = len(common_df[common_df.Condition=='sham'].Subject.unique()), 1, 64
        up = up[f'{band}'].to_numpy().reshape(reshape_factor)
        down = down[f'{band}'].to_numpy().reshape(reshape_factor)
        sham = sham[f'{band}'].to_numpy().reshape(reshape_factor)
        contrast = np.concatenate([sham, up, down], 1)
        
        from mne.stats import f_threshold_mway_rm
        # f_thresh = f_threshold_mway_rm(n_subjects=contrast.shape[0],
        #                                factor_levels=[3], effects='A',
        #                                pvalue=0.05)
        
        def stat_fun(*args):
            return mne.stats.f_mway_rm(np.swapaxes(args, 0, 0), factor_levels=[3],
                                       effects='A', return_pvals=False)[0]
    
        adjacency, ch_names = mne.channels.find_ch_adjacency(info.info, None)
        
        clus_kwargs = {'n_permutations' : 1024,  # 1000 is the minimum
                       'threshold' : dict(start=0, step=0.2), # None
                       #'threshold' : f_thresh, 
                       'tail' : 1,               # one-tailed test (0 for two-tailed)
                       'n_jobs' : -1,            # increase value to speed up computations
                       'buffer_size' : None,
                       'out_type' : 'mask',      # returns a mask map instead of indices of sig. points
                       'seed' : 1503}
        
        F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
            contrast, stat_fun=stat_fun, adjacency=adjacency, **clus_kwargs)
        
        good_clusters = np.where(cluster_p_values < .05)[0]
        mask = cluster_p_values < .05
        try:
            pmin = np.min(cluster_p_values)
        except:
            pmin = np.inf
        if len(good_clusters) > 0:
            sig_chan = np.asarray(ch_names)[good_clusters]
            #common_df_ttests = common_df[common_df.Session==sess]
            #t = [common_df_ttests.loc[common_df_ttests['Channel'].isin([sig_chan[i]])].pairwise_tests(dv=f'{band}', within=['Condition'], subject='Subject') for i in range(len(sig_chan))]
            #t2 = pd.concat(t).reset_index(drop=True)
            #_, p_corrected =statsmodels.stats.multitest.fdrcorrection(t2['p-unc'])
            #_, p_corrected,_,_ =statsmodels.stats.multitest.multipletests(t2['p-unc'], method='sidak')
            #print(f'Min p-value: {p_corrected.min().round(4)}')
            #pairwise_tests.append(t)
        else:
            sig_chan = 'None'
        
        print('Min p-val: {}\nSignificant Chans: {}'.format(pmin, sig_chan))
    
        # plot t-contrast with significance mask
        if plot:
            # fig, ax = plt.subplots(figsize=(6,4))
            # im, _ = mne.viz.plot_topomap(F_obs.squeeze(), pos=info.info, 
            #                              mask=mask, axes=ax, show=0, cmap='RdBu_r',
            #                              names=ch_names, show_names=False, contours = 0,
            #                              mask_params=dict(markersize=8, markerfacecolor='y'))
            # cbar = fig.colorbar(im, ax=ax)   
            # cbar.ax.set_ylabel('f-stat',rotation=270)
            # ax.set_title(f'Resting State Power - {band} Power')
            # plt.tight_layout()
            # plt.show()
            
            fig, axs = plt.subplots(1, 4, figsize=(15, 6))
            for j, (ax, label) in enumerate(zip(axs,['(Sham)', '(Up)','(Down)'])):
                if j < 3:
                    im1, _ = mne.viz.plot_topomap(contrast[:,j,:].mean(0), 
                                                  pos=info.info,
                                                  axes=ax, show=0, cmap='Spectral_r',
                                                  names=None)
                    ax.set_title('\u0394 ' + grk_now + ' ' +label, fontsize=16)
                    #ax.set_title(label)
                    cbar1 = fig.colorbar(im1, fraction=0.05, ax=ax)   
                    cbar1.ax.set_ylabel('\u03BC' + 'V', rotation=270, fontsize=16)
                    plt.tight_layout()
                
            im2, _ = mne.viz.plot_topomap(F_obs.squeeze(), 
                                          pos=info.info, mask=mask,
                                          axes=axs[-1], show=0, cmap='Spectral_r',
                                          names=None,  
                                          mask_params=dict(markersize=6, markerfacecolor='y'))
            axs[-1].set_title(f'Spatial Cluster Test {grk_now}', fontsize=16)
            cbar2 = fig.colorbar(im2, fraction=0.05, ax=axs[-1])   
            cbar2.ax.set_ylabel('F-stat', rotation=270, fontsize=16)
            
            plt.suptitle(f'Eyes {sess} {band} stats')
            plt.tight_layout()
            import time 
            time.sleep(0.5)
            plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/Resting_state/{sess.capitalize()}_{band}_stats_new.jpg')
            plt.close('all')
            
#%%
sns.set_theme(color_codes=True)
# df_psd.dropna(inplace=True) 
for i, band in enumerate(['Delta_fooof', 'Theta_fooof', 'Alpha_fooof', 'Beta_fooof', 'Gamma_fooof']): 
    print(f'Running LMM for frequency band: {band}')
    
    model = Lmer(f"{band} ~ Condition*Channel + (Condition+1|Subject)", 
                 data=df_psd)
    
    # Using dummy-coding; suppress summary output
    model.fit(factors={"Condition": ["sham", "up", "down"],
                       "Channel" : list(df_psd.Channel.unique())},
              ordered=True, summarize=False)
    
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


#%%
from statannotations.Annotator import Annotator
import seaborn as sns
import pandas as pd
sns.set(style="whitegrid")

test=df_psd.loc[df_psd['Channel'].isin(['Cz','Fz','F3','F4','FCz','Fpz'])].reset_index(drop=True)
test=test[test.Session=='Open']

for s in test.Session.unique():  
    test_s = test.loc[test.Session==s].dropna()
    for b in ['Theta']:
        for c in test_s.Channel.unique():      
            #p = test_s.loc[test_s.Channel==c].rm_anova(dv=b, within='Condition', subject='Subject')['p-unc']
            model = Lmer(f"{b} ~ Condition + (1|Subject)", 
                         data= test_s.loc[test_s.Channel==c])
            model.fit(factors={"Condition": ["sham", "up", "down"]},
                      ordered=True, summarize=False)
            p = model.anova(force_orthogonal=True)['P-val'][0]
            print(b, c, p.round(4))
    
for band in ['Delta','Theta','Alpha','Beta','Gamma']:
    x = "Channel"
    y = band
    hue = "Condition"
    hue_order=["sham","down","up"]
    order = ["Fpz","F3","Fz","F4","Cz","FCz"]
    # pairs=[
    #     # (("Fpz", "up"), ("Fpz", "sham")),
    #     # (("Fpz", "up"), ("Fpz", "down")),
    #     # (("Fpz", "down"), ("Fpz", "sham")), 
    #     # (("F3", "up"), ("F3", "sham")),
    #     # (("F3", "up"), ("F3", "down")),
    #     # (("F3", "down"), ("F3", "sham")), 
    #     (("Fz", "up"), ("Fz", "sham")),
    #     (("Fz", "up"), ("Fz", "down")),
    #     (("Fz", "down"), ("Fz", "sham")), 
    #     # (("F4", "up"), ("F4", "sham")),
    #     # (("F4", "up"), ("F4", "down")),
    #     # (("F4", "down"), ("F4", "sham")), 
    #     # (("Cz", "up"), ("Cz", "sham")),
    #     # (("Cz", "up"), ("Cz", "down")),
    #     # (("Cz", "down"), ("Cz", "sham")), 
    #     # (("FCz", "up"), ("FCz", "sham")),
    #     # (("FCz", "up"), ("FCz", "down")),
    #     # (("FCz", "down"), ("FCz", "sham")) 
    #     ]
    
    pairs=[(('Fpz','sham'),('FCz','up'))]
    # pairs=[(("Fz", "up"), ("Fz", "sham")),
    #         (("Fz", "up"), ("Fz", "down")),
    #         (("Fz", "down"), ("Fz", "sham")) ]
    
    # Initialize the figure with a wider aspect ratio
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed
    sns.boxplot(data=test, x=x, y=y, order=order, hue=hue, hue_order=hue_order, ax=ax)
    ax.set_ylabel(f'\u0394 {band} ', weight='bold', fontsize=16)
    ax.set_xlabel('Frontocentral Channels ', weight='bold', fontsize=16)
    annot = Annotator(ax, pairs, data=test, x=x, y=y, order=order, hue=hue, hue_order=hue_order)
    annot.configure(comparisons_correction="fdr_by", test='t-test_paired', verbose=2)
    annot.apply_test()
    annot.annotate()
    plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))  
    plt.tight_layout()
    plt.savefig(save_path + f'{band}_post-hoc.png', dpi=1000, bbox_inches='tight')
    del ax
    
#%%
## RS-Sleep correlations 
stat_path = '/media/administrator/data/Study_1_data/Statistics/SW_spindle_summary/'
sws_density = pd.read_csv(stat_path + 'sws_density.csv', index_col=0)
sp_density = pd.read_csv(stat_path + 'sp_density.csv', index_col=0)

# Change dfs to merge
df_psd = df_psd[df_psd.Session=='Open'].groupby(['Subject', 'Night', 'Condition', 'Channel']).mean()
sws_density = sws_density[sws_density.Stage==3]

# Merge the two dataframes on the shared variable
merged_df = pd.merge(df_psd, sws_density, on=['Condition', 'Subject', 'Channel'])
merged_df1 = merged_df.loc[merged_df['Condition'].isin(['up', 'sham'])]
merged_df2 = merged_df.loc[merged_df['Condition'].isin(['down', 'sham'])]
  
# Group the merged dataframe by Subject, Session, Condition, and Channel
grouped_df1 = merged_df1.groupby(['Condition', 'Subject', 'Channel'], sort=False)
grouped_df2 = merged_df2.groupby(['Condition', 'Subject', 'Channel'], sort=False)

# Define a function to subtract values for two conditions
def subtract_conditions1(group):
    x_values = group.loc[group.Condition=='up', group.columns[3:]].sum()
    y_values = group.loc[group.Condition=='sham', group.columns[3:]].sum() 
    return x_values - y_values
def subtract_conditions2(group):
    x_values = group.loc[group.Condition=='down', group.columns[3:]].sum()
    y_values = group.loc[group.Condition=='sham', group.columns[3:]].sum() 
    return x_values - y_values

# Apply the function to the groupby object
result1 = grouped_df1.apply(subtract_conditions1).reset_index()
result1['Condition'] = ['up']*len(result1)
result2 = grouped_df2.apply(subtract_conditions2).reset_index()
result2['Condition'] = ['down']*len(result2)

# recombined dataframes
remerge = pd.concat([result1, result2])

avg_df = merged_df.loc[merged_df['Channel'].isin(['Cz','Fz','F3','F4','FCz','Fpz'])].reset_index(drop=True)
avg_df = avg_df.groupby(['Condition','Subject']).mean().reset_index()
#avg_df = merged_df[merged_df.Channel=='C3'].groupby(['Condition','Subject']).mean().reset_index()
#avg_df = merged_df.groupby(['Condition','Subject']).mean().reset_index()


## Stat fun
# for i, metric in enumerate(['Delta_fooof', 'Theta_fooof', 'Alpha_fooof', 
#                           'Beta_fooof', 'Gamma_fooof']): 
metric = 'Theta'
print(f'Running LMM for frequency band: {metric}')

model = Lmer(f"{metric} ~ Condition*Density + (1|Subject)", 
             data=avg_df)

# Using dummy-coding; suppress summary output
model.fit(factors={"Condition": ["sham", "up", "down"]},
                   #"Channel" : list(merged_df.Channel.unique())},
          ordered=True, summarize=True)

# Get ANOVA table, but force orthogonality for valid SS III inferences
# In this case the data is unbalaced, otherwise nothing changes
print(model.anova(force_orthogonal=True))

## Post-hoc tests 
marginal_estimates, comparisons = model.post_hoc(p_adjust="fdr",
                                                 marginal_vars='Density',
                                                 grouping_vars='Condition' #None
                                                 )
print(marginal_estimates)
print(comparisons)
 
sns.set_theme(style="darkgrid")
g = sns.lmplot(data=avg_df, x='Density', y='Theta', 
               col='Condition', hue='Condition', **dict(sharex=False))
# Change the labels
g.set_ylabels('\u0394 ' + 'Relative ' + '\u03B8 ' + 'power', fontsize=16)
g.set_xlabels('\u0394 ' + 'SW Density', fontsize=16)
plt.tight_layout()
plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/Resting_state/{metric}_SW.jpg', dpi=1000)
  
# corr
#pg.normality(avg_df, method='normaltest')
ps = []
for c in avg_df.Condition.unique():
    res = pg.corr(avg_df[avg_df.Condition==c].Density, 
                  avg_df[avg_df.Condition==c].Theta)
    ps.append(res['p-val'].values[0])

reject, pvals_corr = pg.multicomp(ps, method='fdr_bh')

# recode condition and perform lm 
mapping = {'sham': 1, 'up': 2, 'down': 3}
avg_df['Condition'] = avg_df['Condition'].replace(mapping)
lm = pg.linear_regression(avg_df[['Density', 'Condition']], avg_df['Theta'])

#%%
# Calculate the frontocentral average for theta activity
frontocentral_channels = ['Cz', 'Fz', 'F3', 'F4', 'FCz', 'Fpz']
fc_avg_theta = merged_df[merged_df['Channel'].isin(frontocentral_channels)].groupby(['Condition', 'Subject']).mean().reset_index()
fc_avg_theta = fc_avg_theta[['Condition', 'Subject', 'Theta']]  # Keep only relevant columns

# Extract SW density for C3
sw_density_c3 = sws_density[sws_density['Channel'] == 'C3']
sw_density_c3 = sw_density_c3[['Condition', 'Subject', 'Density']]  # Keep only relevant columns

# Merge the two datasets on Condition and Subject
avg_df = pd.merge(fc_avg_theta, sw_density_c3, on=['Condition', 'Subject'])

# Perform the LMM analysis
print('Running LMM for frequency band: Theta with SW density at C3')
model = Lmer("Theta ~ Condition*Density + (1|Subject)", data=avg_df)
model.fit(factors={"Condition": ["sham", "up", "down"]}, ordered=True, summarize=True)
print(model.anova(force_orthogonal=True))

# Post-hoc tests to assess differences due to Condition and SW Density
marginal_estimates, comparisons = model.post_hoc(p_adjust="fdr", 
                                                 marginal_vars='Density', 
                                                 grouping_vars='Condition')

# Print the results
print("Marginal Estimates:")
print(marginal_estimates)
print("\nPairwise Comparisons:")
print(comparisons)

# Plotting
sns.set_theme(style="darkgrid")
g = sns.lmplot(data=avg_df, x='Density', y='Theta', col='Condition', hue='Condition', sharex=False)
g.set_ylabels('\u0394 Relative Frontocentral \u03B8 Power', fontsize=16)
g.set_xlabels('\u0394 SW Density at C3', fontsize=16)
plt.tight_layout()
plt.savefig('/media/administrator/data/Study_1_data/Statistics/Resting_state/Theta_SW.jpg', dpi=1000)
  
#%%
## Sleep TMS correlations
param_df = pd.read_csv('/media/administrator/data/Study_1_data/Statistics/TMS/TMS_results_params.csv', index_col=0)
param_df_session = pd.read_csv('/media/administrator/data/Study_1_data/Statistics/TMS/TMS_results_params2.csv', index_col=0)

# swa_tms_rs_df = pd.merge(avg_df, param_df, on=['Condition', 'Subject'])

load_path = '/media/administrator/data/Study_1_data/Statistics/Sleep/Inst_power.p'
ip_df = pickle.load(open(load_path, "rb"))
ip_df = ip_df.loc[ip_df['Subject'].isin(good_subs)]
#swa_df = ip_df[ip_df.Band=="Delta"]

# prepare data for permutation cluster test
subject_vals = ip_df['Subject'].unique()
condition_vals = ip_df['Condition'].unique()
band_vals = ip_df['Band'].unique()
chan_vals = ip_df['Channel'].unique()
num_subjects = len(subject_vals)
num_conditions = len(condition_vals)
num_bands = len(band_vals)
num_channels = len(chan_vals)

for timepoint in ('Post_Pre','IPI', 'Post2'):
    merged_array = np.zeros((num_subjects, num_conditions, num_bands, num_channels))*np.nan
    for i, subject in enumerate(subject_vals):
        for j, condition in enumerate(condition_vals):
            for b, band in enumerate(band_vals):
                for c, channel in enumerate(chan_vals):
                    mask = (ip_df['Subject'] == subject) & \
                        (ip_df['Condition'] == condition) & \
                        (ip_df['Band'] == band) & \
                        (ip_df['Channel'] == channel)
                    if timepoint == 'IPI':
                        diff_values = ip_df.loc[mask, 'IPI']
                    elif timepoint == 'Post_Pre':
                        diff_values = ip_df.loc[mask, 'Post'] - ip_df.loc[mask, 'Pre']
                    elif timepoint == 'Post2':
                        diff_values = ip_df.loc[mask, 'Post2']
                    if len(diff_values) == 0:
                        merged_array[i, j, b, c] = np.nan
                    else:
                        merged_array[i, j, b, c] = diff_values.values[0]
    
    merged_array_down = np.expand_dims(np.subtract(merged_array[:,0,:,:],
                                                   merged_array[:,1,:,:]), 1)
    merged_array_up = np.expand_dims(np.subtract(merged_array[:,-1,:,:],
                                                 merged_array[:,-2,:,:]), 1)
    
    merged_stats = np.concatenate([merged_array_up, merged_array_down], axis=1)
    
    
    # adjacency 
    info = mne.create_info(ch_names=list(ip_df.Channel.unique()),
                           sfreq=512, ch_types=len(ip_df.Channel.unique())*['eeg'])
       
    # Add channel positions to the info object
    info.set_montage(mne.channels.make_standard_montage('standard_1005'))
    adjacency, ch_names = mne.channels.find_ch_adjacency(info, 'eeg')
       
    # threshold
    dof = merged_stats.shape[0] - 1  # degrees of freedom
    t_thresh = stats.distributions.t.ppf(1 - 0.05 / 2, df=dof)
       
    ## do stats
    for inst in ('up','down','double_contrast'):
        if inst == 'up':
            merged_array = merged_array_up
        elif inst == 'down':
            merged_array = merged_array_down
        elif inst == 'double_contrast':
            merged_array = merged_stats
        for idx, band in enumerate(band_vals):
            if inst != 'double_contrast': 
                contrast = merged_array[:,:,idx,:] 
            else:
                contrast = merged_array[:,1,idx,:] - merged_array[:,0,idx,:]
            t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
                contrast,
                n_permutations=1024,                                             
                threshold=dict(start=0, step=0.2), # => performs tfce
                #threshold=t_thresh,
                tail=0,                            # two-tailed
                n_jobs=-1,
                adjacency=adjacency,
                buffer_size=None,                                             
                out_type='mask',
                seed=1503) 
            
            if np.any(cluster_p_values < 0.05):
                pmin = max(cluster_p_values[cluster_p_values < 0.05])
                mask = cluster_p_values < 0.05
                sig_chan = np.asarray(ch_names)[mask]
                print(f'{timepoint}_{inst}_{band}')
                print('Min p-val: {}\nSignificant Chans: {}'.format(pmin, sig_chan))
            else:
                mask = None
        
            # plot t-contrast with significance mask
            fig, ax = plt.subplots(figsize=(10,5), dpi=100)
            im, _ = mne.viz.plot_topomap(t_obs.squeeze(), pos=info, 
                                         mask=mask, axes=ax, show=0, #cmap='RdBu_r', 
                                         names=None, show_names=False, contours = 0,
                                         vlim=(np.percentile(t_obs.squeeze(), 10),
                                               np.percentile(t_obs.squeeze(), 90)), 
                                         mask_params=dict(markersize=15, markerfacecolor='y'))
            cbar = fig.colorbar(im, ax=ax)   
            cbar.ax.set_ylabel('T Statistic', rotation=270)
            ax.set_title(f'{band}')
            plt.tight_layout()
            plt.show()
            plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/Sleep/Instantaneous_power/{timepoint}_{inst}_{band}.jpg')
            plt.close('all')

#%%
load_path = '/media/administrator/data/Study_1_data/Statistics/Sleep/Inst_power.p'
ip_df = pickle.load(open(load_path, "rb"))
ip_df = ip_df.loc[ip_df['Subject'].isin(good_subs)]

C3 = ip_df[ip_df.Channel=='C3'].reset_index(drop=True)
C3 = C3[C3.Band=='Delta']
# for b, band in enumerate(band_vals):
#     for i, subject in enumerate(subject_vals):
#         for j, condition in enumerate(condition_vals):
#             for interval in ('Post_Pre','IPI'):
#                 mask = (C3['Band'] == band) 
#                 if timepoint == 'IPI':
#                     swa_tms_df = pd.merge(C3.loc[mask, 'IPI'], 
#                                           param_df, on=['Condition', 'Subject'])
#                 elif timepoint == 'Post_Pre':
#                     diff_values = pd.merge(C3.loc[mask, 'Post'] - C3.loc[mask, 'Pre'],
#                                            param_df, on=['Condition', 'Subject'])

# Merge the two dataframes on the shared variable
merged_df1 = C3.loc[C3['Condition'].isin(['up', 'sham'])]
merged_df2 = C3.loc[C3['Condition'].isin(['down', 'sham down'])]

# Group the merged dataframe by Subject, Session, Condition, and Channel
grouped_df1 = merged_df1.groupby('Subject', sort=False)
grouped_df2 = merged_df2.groupby('Subject', sort=False)

# Define a function to subtract values for two conditions
def subtract_conditions1(group):
    if 'up' in group['Condition'].values and 'sham' in group['Condition'].values:
        x_values = group.loc[group.Condition=='up', 'Post'].sum() - group.loc[group.Condition=='up', 'Pre'].sum()
        y_values = group.loc[group.Condition=='sham', 'Post'].sum() - group.loc[group.Condition=='sham', 'Pre'].sum()
        return x_values - y_values
    else:
        return np.nan
        
def subtract_conditions2(group):
    if 'down' in group['Condition'].values and 'sham down' in group['Condition'].values:
        x_values = group.loc[group.Condition=='down', 'Post'].sum() - group.loc[group.Condition=='down', 'Pre'].sum()
        y_values = group.loc[group.Condition=='sham down', 'Post'].sum() - group.loc[group.Condition=='sham down', 'Pre'].sum()
        return x_values - y_values
    else:
        return np.nan

# Apply the function to the groupby object
result1 = grouped_df1.apply(subtract_conditions1).reset_index()
result1['Condition'] = ['Up'] * len(result1)
result1 = result1.rename(columns={0: '\u0394SWA'})

result2 = grouped_df2.apply(subtract_conditions2).reset_index()
result2['Condition'] = ['Down'] * len(result2)
result2 = result2.rename(columns={0: '\u0394SWA'})

# recombined dataframes
remerge = pd.concat([result1, result2]).reset_index(drop=True)


# now fix FacIDx
# Merge the two dataframes on the shared variable
merged_df3 = param_df.loc[param_df['Condition'].isin(['up', 'sham'])]
merged_df4 = param_df.loc[param_df['Condition'].isin(['down', 'sham'])]

# Group the merged dataframe by Subject
grouped_df3 = merged_df3.groupby('Subject', sort=False)
grouped_df4 = merged_df4.groupby('Subject', sort=False)

# Define a function to subtract values for two conditions
def subtract_conditions3(group):
    if 'up' in group['Condition'].values and 'sham' in group['Condition'].values:
        x_values = group.loc[group.Condition=='up', ['FacIdx']].sum().values[0] 
        y_values = group.loc[group.Condition=='sham', ['FacIdx']].sum().values[0]
        return x_values - y_values
    else:
        return np.nan
        
def subtract_conditions4(group):
    if 'down' in group['Condition'].values and 'sham' in group['Condition'].values:
        x_values = group.loc[group.Condition=='down', ['FacIdx']].sum().values[0] 
        y_values = group.loc[group.Condition=='sham', ['FacIdx']].sum().values[0]
        return x_values - y_values
    else:
        return np.nan

# Apply the function to the groupby object
result3 = grouped_df3.apply(subtract_conditions3).reset_index()
result3 = result3.rename(columns={0: '\u0394''FacIdx'})
result3['Condition'] = ['Up']*len(result3)
result4 = grouped_df4.apply(subtract_conditions4).reset_index()
result4 = result4.rename(columns={0: '\u0394''FacIdx'})
result4['Condition'] = ['Down']*len(result4)

# recombined dataframes
remerge2 = pd.concat([result3, result4]).reset_index(drop=True)

# merge dataframes
merged_df = pd.merge(remerge, remerge2, on=['Condition', 'Subject'])
# # merged_df = merged_df.rename(columns={: '\u0394''SWA'})

# Create FacetGrid with lmplot
sns.set_theme(color_codes=True)
sns.lmplot(merged_df, x='ΔSWA', y='ΔFacIdx', col='Condition', hue='Condition',
           col_wrap=1, ci=95, scatter_kws={"s": 50})
plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/TMS/SWA_FacIdx1.jpg')
           
# do stats
res_u = pg.corr(merged_df[merged_df.Condition=='Up']['ΔSWA'].values, 
              merged_df[merged_df.Condition=='Up']['ΔFacIdx'].values,
              method='spearman')
res_d = pg.corr(merged_df[merged_df.Condition=='Down']['ΔSWA'].values, 
              merged_df[merged_df.Condition=='Down']['ΔFacIdx'].values,
              method='spearman')
reject, pvals_corr = pg.multicomp([res_u['p-val'].values[0], 
                                   res_d['p-val'].values[0]], method='fdr_bh')


#%% 
model = Lmer(f"ΔFacIdx ~ Condition*ΔSWA + (1|Subject)", 
             data=merged_df)

# Using dummy-coding; suppress summary output
model.fit(factors={"Condition": ["Up", "Down"]},
          ordered=True, summarize=True)

# Get ANOVA table, but force orthogonality for valid SS III inferences
# In this case the data is unbalaced, otherwise nothing changes
print(model.anova(force_orthogonal=True))

## Post-hoc tests 
marginal_estimates, comparisons = model.post_hoc(p_adjust="fdr",
                                                 marginal_vars='ΔSWA',
                                                 grouping_vars='Condition'
                                                 )

print(marginal_estimates)
print(comparisons)
