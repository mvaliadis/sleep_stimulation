#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:06:12 2021

@author: administrator
"""

import os
from sleepstim.sleep_funs import (bfr_butter_filt, bandpower, unravel_hypnogram_visbrain, 
                                  downsample_scaled, load_xdf, channel_parser, thresholdcrossings,
                                  hjorth_complexity, hjorth_mobility)                        
from sleepstim.Analysis.snr import snr_spectrum
from sleepstim.Analysis import bss, bad_channel_detection
import numpy as np
import matplotlib.pyplot as plt
import yasa 
import mne
from autoreject import Ransac
import logging 
from sklearn.ensemble import IsolationForest
from sklearn.covariance import ShrunkCovariance
import pandas as pd 
from collections import OrderedDict
import scipy.stats as stats
from scipy.stats import skewnorm
import scipy.signal as signal
#from statsmodels.api import tsa
import statsmodels
# import easyEEG.structure as eeg_stats
# from PCIst.PCIst import pci_st
from meegkit.detrend import detrend
import pyprep
from mne.preprocessing import ICA
from pyriemann.estimation import Covariances, Shrinkage
from pyriemann.clustering import Potato
import seaborn as sns
import pickle

class Resting_State_Data_Struct:
    def __init__(self, data_eyes_open, data_eyes_close, breath_data, 
                 breath_times, sf, eego_times, ch_names, ch_types,
                 bad_chans, eeg_index, emg_index, ecg_index, mne_info):
        self.data_eyes_open = data_eyes_open
        self.data_eyes_close = data_eyes_close
        self.breath_data = breath_data
        self.breath_times = breath_times
        self.sf = sf
        self.times = eego_times
        self.ch_names = ch_names
        self.ch_types = ch_types
        self.eeg_index = eeg_index
        self.emg_index = emg_index
        self.ecg_index = ecg_index
        self.bad_chans = bad_chans
        self.mne_info = mne_info
        print('Creating resting state data structure with %s eyes open data, with %s epochs, of length %s minutes, containing %s channels'
              % (self.data_eyes_open.dtype, data_eyes_open.shape[0], round(data_eyes_open.shape[2]/sf,2), data_eyes_open.shape[1]))
        print('and eyes closed data structure with %s eyes closed data, with %s epochs, of length %s minutes, containing %s channels'
              % (self.data_eyes_close.dtype, data_eyes_close.shape[0], round(data_eyes_close.shape[2]/sf,2), data_eyes_close.shape[1]))

    def save_rs_data(self, path): 
        # save as pickle file
        pickle.dump(self, open(path, "wb")) 
        
#%%
def _pre_process_rs_data(files, csd = True, prep_pipeline = False, art_method = 'covar', save = True):
    # create log file 
    log_path = '/media/administrator/data/Study_1_data/Data_tracking/resting_state_reports/'
    subj_cond = files.split("/")[-2] + "_" + files.split("/")[-1].split("_")[1] + "_" + files.split("/")[-1].split("_")[-2]
    out = open(log_path + subj_cond + '_resting_state_report.txt', "w")
    # plotting & ica object path
    fig_path = '/media/administrator/data/Study_1_data/Figures/Resting_state/'
    ica_path = '/media/administrator/data/Study_1_data/Resting_state_ica_data/'
    ## load and parse data
    stream_dict = load_xdf(files) 
    data = stream_dict['eego']['time_series']
    breath_name = ','.join([item for item in list(stream_dict) if str(item).startswith('GDX')])
    if breath_name == '':
        breath_data, breath_times = [], []
    else:
        breath_data = stream_dict[breath_name]['time_series'].squeeze()
        breath_times = stream_dict[breath_name]['time_stamps'].squeeze()
    eego_times = stream_dict['eego']['time_stamps'] - stream_dict['eego']['time_stamps'][0]
    marker = stream_dict['reiz-marker']
    eyes_open_ts = [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if 
                  marker['time_series'][ix][0] == 'augen_auf'] - stream_dict['eego']['time_stamps'][0]
    eyes_close_ts = [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if 
                   marker['time_series'][ix][0] == 'augen_zu'] - stream_dict['eego']['time_stamps'][0]
    info = stream_dict['eego']['info'] 
    ch_names, ch_types = channel_parser(info, data)
    sf = int(float(info['nominal_srate'][0]))
    ch_names, ch_types = channel_parser(info, data) 
    # select only non empty channel data
    data = data[:,np.asarray(ch_types) != 'misc']*1e6
    
    ## channel indices
    ch_names = [ch_names[i] for i in [j for j, x in enumerate(ch_types) if x != "misc"]]
    ch_types = [ch_types[i] for i in [j for j, x in enumerate(ch_types) if x != "misc"]]
    eeg_index = [i for i, x in enumerate(ch_types) if x == "eeg"]  
    emg_index = [i for i, x in enumerate(ch_types) if x == "emg"]  
    ecg_index = [i for i, x in enumerate(ch_types) if x == "ecg"] 
    
    ## apply bandpass filter
    data_eeg = mne.filter.filter_data(data[:,eeg_index].astype(np.float64).T, sfreq=sf, l_freq=1, h_freq=45, verbose=0).T
    data_emg1 = mne.filter.filter_data(data[:,[emg_index[0:4]]].astype(np.float64).T, sfreq=sf, l_freq=10, h_freq=200, verbose=0).T
    data_emg2 = mne.filter.filter_data(data[:,[emg_index[4:8]]].astype(np.float64).T, sfreq=sf, l_freq=10, h_freq=200, verbose=0).T
    data_ecg = mne.filter.filter_data(data[:,ecg_index].astype(np.float64).T, sfreq=sf, l_freq=0.3, h_freq=70, verbose=0).T
    data = np.concatenate([data_eeg, data_emg1.squeeze(), data_ecg, data_emg2.squeeze()], axis=-1)
    
    ## apply notch filter to combined data
    data_filt = mne.filter.notch_filter(data.T, Fs=sf, method='spectrum_fit', freqs=np.arange(50,50*4+1,50), 
                                        mt_bandwidth=2, p_value=0.01, filter_length='10s', verbose=0).T
    
    ## calculate EEG quality index based on 6 discrete measurements, including:   
    # 1. Average Single-Sided Amplitude Spectrum in range 1-50Hz
    # 2. Average Single-Sided Amplitude Spectrum in range 49-51 Hz
    # 3. root mean square
    # 4. maximum gradient
    # 5. Zero-Crossing Rate 
    # 6. Kurtosis 
    EEG_qi = bad_channel_detection.qc_calcEQI(data_filt[:, eeg_index].T, sf)
    
    # # plot metrics 
    # plt.hist(EEG_qi[0,:,:],histtype="stepfilled", bins=25, alpha=0.8)
    # plt.legend(ch_names[0:64],ncol=3, fontsize='x-small', bbox_to_anchor=(0.75, 0.5),loc='center left')
    
    # create dataframe for EQI results 
    eqi_lab = ['avgspec_1-48','line_noise','root_mean_sq','max_gradient','zero-crossing_rate','kurtosis']
    df_comb = [pd.DataFrame(data = EEG_qi[:,:,i].T, columns = eqi_lab, index = ch_names[0:64]) for i in 
               range(EEG_qi.shape[-1])]
    for i in range(EEG_qi.shape[-1]): df_comb[i]['Epoch'] = i
    df_comb2 = pd.concat(df_comb)
    ilf = IsolationForest(contamination='auto', max_samples='auto',
                          verbose=0, random_state=42)
    good = ilf.fit_predict(df_comb2.drop(['Epoch'], axis=1))
    good[good == -1] = 0
    df_comb2['Isolation forest score'] = good
    
    # isolation forest scores by channel
    df_if_scores = pd.DataFrame([df_comb2.loc[(ch_names[i])]['Isolation forest score'].value_counts(normalize=True) 
                                 for i in range(len(ch_names[0:64]))])
    df_if_scores['Channel'] = ch_names[0:64]
    ch_to_interp = df_if_scores['Channel'][df_if_scores[0] > 0.25].to_list()
    
    # log bad channels
    out.write(f'The dataset: {subj_cond} has the following bad channels: {ch_to_interp}' + '\n')
        
    ## bad channel detection and interpolation
    mne_info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)
    raw = mne.io.RawArray(data_filt.T/1e6, mne_info) 
    raw = raw.set_montage(mne.channels.make_standard_montage('standard_1005')) 
    raw.info['bads'] = ch_to_interp
    raw = raw.interpolate_bads()
    
    ## PREP pipeline bad channel detection, interpolation and subsequent robust rereferencing 
    if prep_pipeline:
        prep_params = {"ref_chs": ch_names[0:64],
                       "reref_chs": ch_names[0:64]
                        }
    
        raw_ref = pyprep.reference.Reference(raw, prep_params, ransac=False)
        raw_ref.perform_reference()
       
    ## Perform ICA to remove eye blinks/muscle artifacts
    # Calculate ICA with faster picard method which converges faster than fastICA or infomax methods
    ica = ICA(n_components=15, method='picard', random_state=42, fit_params = dict(ortho=True, extended=True))
    print('\n***** Performing ICA ...\n')
    ica.fit(raw, picks='eeg')
    
    # Determine ecg component to remove
    ecg_idx, ecg_scores = ica.find_bads_ecg(raw, method = 'ctps', threshold= 'auto')
    ecg_score_plot = ica.plot_scores(ecg_scores)
    ecg_score_plot.savefig(fig_path + subj_cond + '_ecg_component_score.png')
    
    # Automated component detection assistance 
    from mne_icalabel import label_components
    ic_labels = label_components(raw, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    print(f"Excluding these ICA components: {exclude_idx}")
    #ica.detect_artifacts(raw) -> now deprecated in mne 
     
    # Plot sources separated by ICA
    ic_source_plot = ica.plot_sources(raw, show_scrollbars=True, title='EEG sources estimated by ICA')
    ic_source_plot.savefig(fig_path + subj_cond + '_ic_component_source.png')

    # Plot topographic maps of sources separated by ICA
    ic_comp_plot = ica.plot_components(title='Topographic maps of EEG sources estimated by ICA')
    ic_comp_plot[0].savefig(fig_path + subj_cond + '_ic_topo_plot.png')
    
    # components to exclude 
    ica.exclude = exclude_idx + ecg_idx
     
    # save ICA object
    ica.save(ica_path + subj_cond + '_ica_obj.fif')
    
    # log excluded ICA components
    out.write(f'The dataset: {subj_cond} had the following components removed: {ica.exclude}' + '\n')
    
    # Add to components
    #ica.exclude = ica.exclude + [int(x) for x in input('Enter any additional components to remove: ').split()] + ecg_idx
    
    # Plot signal with removed components
    overlay_plot = ica.plot_overlay(raw, exclude=ica.exclude)
    overlay_plot.savefig(fig_path + subj_cond + '_ic_removed_overlay.png')
    
    # Apply ICA
    fit_ica = ica.apply(raw)
    
    ##  Extract data and compute Surface Laplacian transform to reduce affect of volume conduction
    if csd:
        # print(f'Ignoring status of bad channels: {fit_ica.info["bads"]} for the purposes of computing CSD. ')
        # fit_ica.info['bads'] = []
        fit_ica = mne.preprocessing.compute_current_source_density(fit_ica)
    else:
        # perform common average reref
        fit_ica = fit_ica.set_eeg_reference(ref_channels='average')

    ## Label remaining artifact window
    
    # Covariance based approach
    if art_method == 'covar':
        # Calculate the covariance matrices,
        # shape (n_epochs, n_chan, n_chan)
        _, epochs = yasa.sliding_window(fit_ica.get_data()[eeg_index, :]*1e6, window = 2, sf = sf)  
        covmats = Covariances().fit_transform(epochs)
        # Shrink the covariance matrix (ensure positive semi-definite)
        covmats = Shrinkage().fit_transform(covmats)
        # Define Potato instance: 0 = clean, 1 = art
        # To increase speed we set the max number of iterations from 10 to 100
        potato = Potato(metric='riemann', threshold=3, pos_label=0,
                        neg_label=1, n_iter_max=100)
    
        # Apply Potato algorithm, extract z-scores and labels
        zs = potato.fit_transform(covmats)
        art = potato.predict(covmats).astype(int)
       
    # STD version 
    elif art_method == 'std':
        std_epochs = np.log(np.nanstd(epochs, axis=-1) + 1)
        # Calculate z-scores of STD for each channel x stage
        c_mean = np.nanmean(std_epochs, axis=0, keepdims=True)
        c_std = np.nanstd(std_epochs, axis=0, keepdims=True)
        zs = (std_epochs - c_mean) / c_std
        
        # Any epoch with at least 5 channels above or below threshold (3 z-scores)
        n_chan_supra = (np.abs(zs) > 3).sum(axis=1)  # >
        art = (n_chan_supra >= 5).astype(int)  # >= !
        
    # percent of epochs rejected
    perc_reject = 100 * (art.sum() / art.size)
    
    # Print total rejected epochs 
    print(f"Total: {art.sum()} / {art.size} "
          f"epochs rejected ({perc_reject:.2f}%)")
    out.write(f"The dataset: {subj_cond} had Total: {art.sum()} / {art.size} "
              f"epochs rejected ({perc_reject:.2f}%)" + '\n')
    
    # Convert epoch_is_art to boolean [0, 0, 1] -- > [False, False, True]
    epoch_is_art = art.astype(bool)
    
    # set annotations in mne object 
    annot = mne.Annotations(onset=np.where(epoch_is_art)[0]*2,
                            duration = len(np.where(epoch_is_art)[0])*[2],
                            description=len(np.where(epoch_is_art)[0])*['bad']
                            )
    fit_ica_annot = fit_ica.copy().set_annotations(annot)
    
    # plot to confirm 
    bad_epochs_plot = fit_ica_annot.plot(duration=max(fit_ica.get_data().shape), start=0, color={'eeg': 'steelblue'},
                                         n_channels = 64, title='Cleaned Signal with Artifacted Epochs labelled', 
                                         show_scalebars=True, show = False)
    bad_epochs_plot.savefig(fig_path + subj_cond + '_cleaned_signal.png')
    
    # extract 
    if csd:
        data_filt_filt = fit_ica_annot.get_data().T*1e3
    else:
        data_filt_filt = fit_ica_annot.get_data().T*1e6    
    
    ## Create eyes open/eyes closed epochs based on corrected timestamps
    
    # synchronize clocks for later epoching
    eyes_open_times = [np.argmin(np.abs(eego_times - ts)) for ts in eyes_open_ts]
    eyes_close_times = [np.argmin(np.abs(eego_times - ts)) for ts in eyes_close_ts]
    # index eyes open/eyes closed times
    eyes_open_idx, _  = yasa.get_centered_indices(data_filt_filt[:,0], np.asarray(eyes_open_times), 
                                                 npts_before = 0, npts_after = sf*30.)
    eyes_close_idx, _  = yasa.get_centered_indices(data_filt_filt[:,0], np.asarray(eyes_close_times), 
                                                 npts_before = 0, npts_after = sf*30.)
    
    # cut segments into 2 second windows 
    cut_idx = np.linspace(0,max(eyes_close_idx.shape)-1, 15, endpoint=False).astype(int)
    reshape = []
    for j in range(min(eyes_close_idx.shape)):
        for i in range(len(cut_idx)-1):
            reshape.append(eyes_close_idx[j ,0:-1][cut_idx[i]:cut_idx[i+1]]) 
    eyes_close_idx_all = np.concatenate([reshape])
    
    cut_idx = np.linspace(0,max(eyes_open_idx.shape)-1, 15, endpoint=False).astype(int)
    reshape = []
    for j in range(min(eyes_open_idx.shape)):
        for i in range(len(cut_idx)-1):
            reshape.append(eyes_open_idx[j ,0:-1][cut_idx[i]:cut_idx[i+1]]) 
    eyes_open_idx_all = np.concatenate([reshape])
       
    # index of detected artifacts 
    if len(annot.onset) > 0:
        annot_idx, _ = yasa.get_centered_indices(data_filt_filt[:,0], np.asarray(annot.onset)*sf, 
                                                 npts_before = 0, npts_after = sf*2.)

        # compare annotation time indices and eyes open/close time indices
        epoch_match_open, eye_open, annot_open = np.intersect1d(eyes_open_idx_all, annot_idx, return_indices=True)
        epoch_match_close, eye_close, annot_close = np.intersect1d(eyes_close_idx_all, annot_idx, return_indices=True)
        
        # indices are performed in an unravelled fashion so you need to floor divide to ascertain row to remove
        annot_open //= 2000; eye_open //= 2000
        annot_close //= 2000; eye_close //= 2000
        if len(annot_open) > 0:
            eyes_open_idx_final = np.delete(eyes_open_idx_all, np.unique(eye_open), axis=0)
        else:
            eyes_open_idx_final = eyes_close_idx_all
        if len(annot_close) > 0:
            eyes_close_idx_final = np.delete(eyes_close_idx_all, np.unique(eye_close), axis=0)  
        else:
            eyes_close_idx_final = eyes_close_idx_all
            
        # create epochs
        data_eyes_open = np.swapaxes(data_filt_filt[eyes_open_idx_final, :],1,2)
        data_eyes_close = np.swapaxes(data_filt_filt[eyes_close_idx_final, :],1,2)
    else:
        # create epochs
        data_eyes_open = np.swapaxes(data_filt_filt[eyes_open_idx_all, :],1,2)
        data_eyes_close = np.swapaxes(data_filt_filt[eyes_close_idx_all, :],1,2)
        
    # create data objects
    RS_data_struct = Resting_State_Data_Struct(data_eyes_open, data_eyes_close, breath_data, breath_times,
                                               sf, eego_times, ch_names, ch_types, ch_to_interp, 
                                               eeg_index, emg_index, ecg_idx, fit_ica_annot.info)
      
    # close log file
    out.close()
    
    # close all plots
    plt.close('all')
    
    if save:
        # save RS object
        save_path = '/media/administrator/data/Study_1_data/Pre-processed_data_resting_state/' + subj_cond + '_rs_preprocessed.p'
        RS_data_struct.save_rs_data(save_path) 
    else:
        return RS_data_struct

  
#%%
def subject_cond_parser(file, study_phase='resting state'):
    # data sheet with true subject condition nights
    sheet = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
    subj_cond = np.loadtxt(sheet, delimiter=',', dtype='str', skiprows=1) 
    cond_dict = {0:'sham', 1:'up', 2:'down'}
    # find condition by night
    if study_phase=='resting state':
        sc = file.split('/')[-1].split('_')[0] + '_' + str(int(file.split('/')[-1].split('_')[1]))
    elif study_phase=='rs_hrv':
        sc =  file.split('/')[-2] + '_' + str(int(file.split('/')[-1].split('_')[1]))
    elif study_phase == 'sleep':
        #sc = file.split('/')[-2].split('_')[0] + '_' + str(int(file.split('/')[-2].split('_')[1]) - 1)
        sc = file.split('/')[-1].split('_')[0] + '_' + str(int(file.split('/')[-1].split('_')[1]) - 1)
    elif study_phase=='tms':
        sc = file.split('/')[-2] + '_' + str(int(file.split('/')[-1].split('_')[1]))
    index_name = list(subj_cond[:,0]).index(sc.split('_')[0])
    cond = int(subj_cond[index_name,1::][int(sc[-1])])
    # true condition night name
    condition_night = cond_dict[cond]  
    
    return condition_night

#%%
def check_match_prepost_data_elements(path, files=None, dtype='slalom'):
    if files is not None:
        files_list = files
    else:
        files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    if dtype == 'rs':
        pre_post = [files_list[idx].split('/')[-1].split('_')[2] for idx, file in enumerate(files_list)]
        pre_idx, post_idx = np.where(np.asarray(pre_post) == 'pre')[0], np.where(np.asarray(pre_post) == 'post')[0]
    elif dtype == 'slalom':
        pre_post = [files_list[idx].split('/')[-1].split('_')[0] for idx, file in enumerate(files_list)]
        pre_idx = np.where([pre_post[i].startswith('pre') for i in range(len(pre_post))])[0]
        post_idx = np.where([pre_post[i].startswith('post') for i in range(len(pre_post))])[0]
    pre, post = np.asarray(files_list)[pre_idx], np.asarray(files_list)[post_idx] 
    all_elem = np.asarray([(x,y) for x in pre for y in post])
    if dtype == 'rs':
        exist = [elem[0].split('/')[-1].split('_')[0:2] == elem[1].split('/')[-1].split('_')[0:2]
                 for elem in all_elem]   
    elif dtype == 'slalom':
        exist = [elem[0].split('/')[-1][-16::] == elem[1].split('/')[-1][-16::] for elem in all_elem]
             
    exist_pre_files, exist_post_files = all_elem[exist][:,0], all_elem[exist][:,1]
    
    if len(exist_pre_files) > 0 and len(exist_post_files) > 0:
        return exist_pre_files, exist_post_files
    else:
        raise TypeError('No subject entries align! Please check whether the paths have any corresponding files')
        
#%%
## Plot PSD function 
def plot_psd(psd, freqs, freq_range = (1,30), foi= (4,8), dB=True, ci=True):
    plt.figure()
    idx_all = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
    idx_theta = np.logical_and(freqs >= foi[0], freqs <= foi[1])
    line = sns.lineplot(freqs[idx_all], psd.mean(0)[:,idx_all].mean(0).T, color='k')
    if dB:
        plt.ylabel(r'PSD [(mV/$\mathregular{m^2})^2$/Hz (dB)]')
        line.set_yscale("log")
    else:
        plt.ylabel(r'PSD [(mV/$\mathregular{m^2})^2$]')
    if ci:
        ch_psd = psd[:,:,idx_all].mean(1)
        limits = (np.percentile(ch_psd, 5, axis=0), np.percentile(ch_psd, 95, axis=0))
        plt.fill_between(freqs[idx_all], y1=limits[0], y2=limits[1], alpha=0.5, color='gray')
        plt.ylim([limits[0].min(), limits[1].max()*1.2])
        plt.vlines(foi, ymin=limits[0].min(), ymax=limits[1].max()*1.2, linestyles='dashed', 
                   linewidth=0.7, label='Theta') 
    else:
        plt.ylim([0, psd.mean(0)[:,idx_all].mean(0).max()*1.2])
        plt.fill_between(freqs[idx_theta], psd.mean(0)[:,idx_theta].mean(0).T, 
                         where=idx_theta[idx_theta], color='skyblue', label='Theta')
    plt.xlim(freq_range)
    plt.xlabel('Frequency [Hz]')
    plt.show()
    plt.legend()
    sns.despine()
    
#%%
# =============================================================================
# Wavelet power analysis 
# =============================================================================
def norm_wavelet_power(x, sf, foi=(4,7), wlt_params={'nc': 4, 'cf': 'auto'}):
    """Normalized power based on Morlet wavelet convolution.

    Parameters
    ----------
    x : 1D-array or 3D-array
        EEG/epoch signal 
    sf : float
        Sampling frequency
    foi : tuple
        Canonical frequency range of interest
    wlt_params : dict
        Morlet wavelet parameters ::

        'nc' : number of oscillations
        'cf' : central frequency (int or 'auto')

    Returns
    -------
    sp_params : dict
        Spindles parameters dictionary.
    """
    from mne.time_frequency import morlet, psd_array_multitaper

    # if using singular channel data, need to convert from 2D to 3D epochs
    if np.ndim(x) < 3:
        x = np.expand_dims(x, 1)
        
    if wlt_params['cf'] == 'auto':
        # Compute the power spectrum and find the peak beween selected frequencies of interest.
        psd, freqs = psd_array_multitaper(x, sf, fmin=foi[0], fmax=foi[1], verbose=0)
        # wlt_params['cf'] = freqs[np.argmax(psd)] # for one dimensional data 
        wlt_params['cf'] = freqs[np.asarray(np.where(psd == np.amax(psd))).squeeze()[-1]]
        print('Central frequency: %.2f Hz' % wlt_params['cf'])

    # Compute the wavelet and convolve with data
    wlt = morlet(sf, [wlt_params['cf']], n_cycles=wlt_params['nc'])[0]
    # wlt = signal.morlet(M = sf*2, w=wlt_params['nc'], s=1.0, complete=True)
    # analytic = np.convolve(x, wlt, mode='same')
    analytics = []
    for epoch in x:
        for channel in epoch:
            analytics.append(np.convolve(channel, wlt, mode='same'))
        
    analytics = np.asarray(analytics).reshape(x.shape)
    phase = np.angle(analytics)

    # Square and normalize the magnitude from 0 to 1 (using the min and max)
    power = np.square(np.abs(analytics))
    norm_power = (power - power.min()) / (power.max() - power.min())
    
    wavelet_stats = {'Normalized wavelet magnitude (channel/trial avg)' : norm_power.mean(0), 
                     'Analytical wavelet phase (channel/trial avg)' : phase.mean(0), 
                     'Wavelet magnitude power (channel/trial avg)' : norm_power.mean(0)}

    return wavelet_stats


#%%
# =============================================================================
# Compute measures of connectivity 
# =============================================================================
## Create seed and target area for M1 area
def null_connectivity():
    if seed:
        picks = mne.pick_types(epochs.info, eeg=True)
        seed_ch = 'C3'
        targets = ['FC5','FC3','FC1',
                   'C5','C3','C1',
                   'CP5','CP3','CP1']
        picks_ch_names = [epochs.ch_names[i] for i in picks if epochs.ch_names[i] in target]
        targets = [epochs.ch_names.index(i) for i in picks_ch_names]
        seed = epochs.ch_names.index(seed_ch)
        indices = mne.connectivity.seed_target_indices(seed, targets=targets)
    else:
        indices = None
        
    ## Phase locking value 
    con, freqs, times, _, n_tapers = mne.connectivity.spectral_connectivity(epochs.get_data()[:,0:64,:]*1e3, 
                                                                            method='plv', indices=None, sfreq=sf,
                                                                            mode='multitaper', fmin=13, fmax=30,
                                                                            faverage=False, cwt_freqs=None, cwt_n_cycles=7,
                                                                            block_size=1000, n_jobs=1)
    if seed:
        # Mark the seed channel with a value of 1.0, so we can see it in the plot
        con[np.where(indices[1] == seed)] = 1.0
    else:
        for i in range(len(con)):
            con[i,i] = 1.0
    
    # Plot adjacency matrix
    sns.heatmap(con.mean(-1), xticklabels=RS_data_struct.ch_names[0:64], 
                yticklabels=RS_data_struct.ch_names[0:64], cmap='Spectral_r')
    
    ## Phase Slope Index 
    psi, _, _, _, _ = mne.connectivity.phase_slope_index(data_eyes_open[:, eeg_index], mode='multitaper', 
                                                         sfreq=1000, fmin=8, fmax=12)
    
    ## Envelope correlation
    env_corr = mne.connectivity.envelope_correlation(data_eyes_open[:, eeg_index], combine='mean', orthogonalize='pairwise',
                                                     log=False, absolute=True, verbose=None)
    
    ## Spectral connectivity includes - Preferable to use a seed region, e.g., Motor Cortex (see MNE documentation): 
        # Phase Locking Value (PLV)
        # Phase Lag Index (PLI) / Weighted PLI 
        # Pairwise Phase Consistency (PPC), an unbiased estimator of squared PLV 
        # Coherence
        # Imaginary Coherence 
        # Coherency 
        # etc...
    
    return psi 

