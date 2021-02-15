#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:35:47 2020

@author: administrator
"""

# =============================================================================
# Preprocess data for stageing/analysis
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle 
import pandas as pd
import pyxdf
import yasa 
import mne
import pingouin as pg
from mne.stats import permutation_cluster_test
import logging
import time
import wonambi
import seaborn as sns
from scipy.signal import welch, butter, filtfilt
from scipy.stats import zscore
from scipy.special import erf
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
from meegkit import dss, star, asr
from meegkit.utils import demean, normcol
from os import chdir as cd
from os import listdir
import os, shutil
cd('/home/administrator/sleep_stimulation-development')
from sleep_funs import (bfr_butter_filt, bandpower, unravel_hypnogram_visbrain, 
                        downsample_scaled, load_xdf, channel_parser, thresholdcrossings, plot_confusion_matrix)
sns.set(style='darkgrid', font_scale=1.2)

class Data_Struct:
    def __init__(self, data, chans, chtypes, times, pinknoise_times, classif_predict, classif_times, sfreq):
        self.data = data
        self.chans = chans
        self.chtypes = chtypes
        self.times = times - times[0]
        self.pinknoise_times = pinknoise_times - times[0]
        self.classif_times = classif_times - times[0]
        self.classif_predict = classif_predict
        self.sfreq = sfreq
        print('Creating RawArray with %s data, of length %s minutes, containing %s channels'
              % (self.data.dtype, round(data.shape[0]/sfreq/60, 2), data.shape[1]))
        
    def downsample(self, new_sfreq):
        self.data = downsample_scaled(self.data, self.sfreq, new_sfreq)
        self.times = np.arange(len(self.data))/new_sfreq
        self.sfreq = new_sfreq
        
    def fir_filter(self, l_freq, h_freq):
        eeg_index = [i for i, x in enumerate(self.chtypes) if x == "eeg"]
        self.data = mne.filter.filter_data(self.data[:, eeg_index].T, self.sfreq, l_freq, h_freq).T
    
        
#%%

path = '/media/administrator/data/Study_1_data/Raw_data/'
cd(path)
         
def _pre_process_sleep_data(files, simulation=False):
    # parse data
    stream_dict = load_xdf(files) 
    # select data and marker streams
    if simulation:
        rec_type = 'eeg_replay'
    else:
        rec_type = 'eego'
    data = stream_dict[rec_type]['time_series']
    eego_times = stream_dict[rec_type]['time_stamps']
    marker = stream_dict['reiz-marker']
    classifier_predict = [marker['time_series'][ix] for ix in range(len(marker['time_series'])) if 'rf' in marker['time_series'][ix][0]]
    classifier_timestamps = [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if 'rf' in marker['time_series'][ix][0]]
    pinknoise_timestamps = [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if marker['time_series'][ix][0] == 'pinknoise']
    # load hdr
    info = stream_dict[rec_type]['info'] 
    # delete stream_dict for memory sake
    del stream_dict
    
    # sampling rate
    sf = int(info['nominal_srate'][0])
    # find channel names and types
    ch_names, ch_types = channel_parser(info, data) 
    # chans of interest for stageing
    EEG_index = np.r_[ch_names.index('F3'), ch_names.index('Fz'), ch_names.index('F4'), 
                      ch_names.index('C3'), ch_names.index('Cz'), ch_names.index('C4'), 
                      ch_names.index('P3'), ch_names.index('P4'), ch_names.index('O1'), 
                      ch_names.index('O2')]
    mastoids_index = np.r_[ch_names.index('M1'), ch_names.index('M2')]
    
    # filter data 
    EEG = bfr_butter_filt(data[:,EEG_index], sf, lfreq=0.5, hfreq=35)
    # EEG, _ = dss.dss_line(EEG, fline=50, sfreq=sf, nfft=4*sf)
    # EEG, _ = dss.dss_line(EEG, fline=100, sfreq=sf, nfft=4*sf)
    Mastoids = bfr_butter_filt(data[:,mastoids_index], sf, lfreq=0.5, hfreq=35)
    EOG_L = bfr_butter_filt(data[:,[ch_names.index('EOG_L')]], sf, lfreq=0.5, hfreq=35)
    EOG_L, _ = dss.dss_line(EOG_L, fline=50, sfreq=sf, nfft=4*sf)
    EOG_L, _ = dss.dss_line(EOG_L, fline=100, sfreq=sf, nfft=4*sf) 
    EOG_R = bfr_butter_filt(data[:,[ch_names.index('EOG_R')]], sf, lfreq=0.5, hfreq=35)
    EOG_R, _ = dss.dss_line(EOG_R, fline=50, sfreq=sf, nfft=4*sf)
    EOG_R, _ = dss.dss_line(EOG_R, fline=100, sfreq=sf, nfft=4*sf) 
    
    #EMG processing
    EMG_L = bfr_butter_filt(data[:,[ch_names.index('EMG_L')]], sf, lfreq=10, hfreq=100)
    EMG_R = bfr_butter_filt(data[:,[ch_names.index('EMG_R')]], sf, lfreq=10, hfreq=100)
  
    # re-reference EEG channels to average of mastoids
    ref_data = Mastoids[..., :].mean(-1, keepdims=True)
    EEG -= ref_data
    if 'bipECG' in ch_names:
        ECG = bfr_butter_filt(data[:,[ch_names.index('bipECG')]], sf, lfreq=0.5, hfreq=70)
        ECG, _ = dss.dss_line(ECG, fline=50, sfreq=sf, nfft=4*sf)
        ECG, _ = dss.dss_line(ECG, fline=100, sfreq=sf, nfft=4*sf)
        # re-combine data
        data = np.concatenate([EEG, EOG_L, EOG_R, ECG], axis=1)
        # edit channel names and types based on new selection 
        new_chans = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4', 'O1', 'O2', 'EOG_L', 'EOG_R', 'ECG', 'EMG_L', 'EMG_R']
        new_chtypes = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eog', 'eog', 'ecg', 'emg', 'emg']
    else: 
        # re-combine data
        data = np.concatenate([EEG, EOG_L, EOG_R], axis=1)
        # edit channel names and types based on new selection 
        new_chans = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4', 'O1', 'O2', 'EOG_L', 'EOG_R', 'EMG_L', 'EMG_R']
        new_chtypes = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eog', 'eog', 'emg', 'emg']
        
    #EMG epoching for time complexity reduction  
    if 'calibration' in files:
        EMG_L = mne.filter.notch_filter(np.squeeze(EMG_L), Fs=sf, method='spectrum_fit', freqs=np.arange(50,50*3+1,50))
        EMG_R = mne.filter.notch_filter(np.squeeze(EMG_R), Fs=sf, method='spectrum_fit', freqs=np.arange(50,50*3+1,50))
    else:
        cutindx = np.linspace(0,len(EMG_L), 100).astype(int)
        
        EMG_L_cut = [EMG_L[cutindx[i]:cutindx[i+1]] for i in range(len(cutindx)-1)] 
        epoched_data_EMG_L = []
        for i in range(len(EMG_L_cut)):  
            EMG_L_chunks = mne.filter.notch_filter(np.squeeze(EMG_L_cut[i]), Fs=sf, method='spectrum_fit', freqs=np.arange(50,50*3+1,50))
            epoched_data_EMG_L.append(EMG_L_chunks)
        EMG_L = np.concatenate(epoched_data_EMG_L, axis=0)
        
        EMG_R_cut = [EMG_R[cutindx[i]:cutindx[i+1]] for i in range(len(cutindx)-1)]
        epoched_data_EMG_R = []
        for i in range(len(EMG_L_cut)):  
            EMG_R_chunks = mne.filter.notch_filter(np.squeeze(EMG_R_cut[i]), Fs=sf, method='spectrum_fit', freqs=np.arange(50,50*3+1,50))
            epoched_data_EMG_R.append(EMG_R_chunks)
        EMG_R = np.concatenate(epoched_data_EMG_R, axis=0)
    
     # add EMG data back
    data = np.concatenate([data, np.expand_dims(EMG_L,1), np.expand_dims(EMG_R,1)], axis=1)*1e6
        
    # create class to save as object
    #Data = Data_Struct(data, new_chans, new_chtypes, eego_times, pinknoise_timestamps, classifier_predict, classifier_timestamps, sf)

    return data, new_chans, new_chtypes, eego_times, pinknoise_timestamps, classifier_predict, classifier_timestamps, sf

def save_preprocess_sleep_data(*args, path): 
    # iterate over args
    var = [arg for arg in args]
    # save as pickle file
    pickle.dump(var, open(path, "wb"))  
      
def preprocess_sleep_data(path, simulation=False, save=True):
    files_list = [os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files]
    for i, files in enumerate(files_list):
        # select subject ID + cond identifier
        subjID_cond = files.split('/')[-2]
        if 'Experimental' in files:
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental/'
        elif 'adaption_calibration' in files:
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Calibration/'
        elif 'Calibration' in files:
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Calibration/'
        elif 'Adaption' in files:
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Adaption/'
        else:
            raise NameError('Please verify that the file name contains an appropriate recording type! ')
        # check if file exists
        new_path = save_path + subjID_cond + '_preproc_data.p'
        if not os.path.exists(new_path):
            data, new_chans, new_chtypes, eego_times, pinknoise_timestamps, \
                classifier_predict, classifier_timestamps, sf = _pre_process_sleep_data(files, simulation=False)
            # check if a second recording file exists, then combine after preprocessing
            if int(files.split('/')[-1].split('.')[0][-1])!= 1:
                new_path = save_path + subjID_cond + '_preproc_data_' + files.split('/')[-1].split('.')[0][-1] + '.p'
            if save:
                save_preprocess_sleep_data(data, new_chans, new_chtypes, eego_times, pinknoise_timestamps, classifier_predict, classifier_timestamps, sf, path=new_path)
            else:
                return data, new_chans, new_chtypes, eego_times, pinknoise_timestamps, classifier_predict, classifier_timestamps, sf
                        
        else:
            logging.warning(f'The requested dataset for subject and condition: {subjID_cond} has already been pre-processed!')
            continue

def load_preprocessed_data(file):
    # load as pickle file
    var = pickle.load(open(file,"rb")) 
    return var

def to_do_combine_recordings():
    return combined

#%%
## SNR computation

snr_asr = asr.ASR(sfreq = Data.sfreq, method='euclid')
snr_asr.fit(Data.data[:, eeg_index].T)

#%%

def plot_zscore_signal(data, zscore_val=3, sf=512, chan='C3'):
    """
    Returns plotted signal with z_score deviations
    TO:DO: Need to revise for object implementation as no new_chans index exists.
    
    """
    times = np.arange(len(data))/sf
    plt.plot(times, data[:,new_chans.index(chan)])
    plt.vlines(times[zscore(np.abs(data[:,new_chans.index(chan)]))>zscore_val], ymin=np.min(data), ymax=np.max(data))
    
def plot_multitaper_spectrogram(data, sf, ch_names, hypno=None, coi ='F3', fmin=0.5, fmax=30):
    if hypno:
        plot = yasa.plot_spectrogram(data[:,[ch_names.index(coi)]], sf, hypno, fmin, fmax, trimperc=5, cmap='Spectral_r');
    else:
        plot = yasa.plot_spectrogram(data[:,[ch_names.index(coi)]], sf,  fmin, fmax, trimperc=5, cmap='Spectral_r');
    return plot;

def artifact_detect_hypno(data, hypno_path, sf, downsample=False, method='covar', window=2, inspection=False): 
    
    # epoch data 
    _, epochs = yasa.sliding_window(data.T, sf=512, window=30)
            
    # load hypnogram
    hypnogram = unravel_hypnogram_visbrain(hypno_path, epochs)
    
    # optional downsamling
    if downsample:
        data = downsample_scaled(data, old_sf=512, new_sf=128, nint_method='resample_poly')
        sf = 128
        
    # sample hypnogram to full dataset
    hypno_unsampled = yasa.hypno_upsample_to_data(hypno = hypnogram, sf_hypno=1/30, data=data.T, sf_data=sf)
      
    # Covariance based artifact rejection
    if method=='covar':
        art, zscores = yasa.art_detect(data.T, sf=sf, window=2, hypno=hypno_unsampled, include=(0, 1, 2, 3, 4), 
                                   method='covar', threshold=3, n_chan_reject=3, verbose='info')
    # Standard deviation based artifact rejection
    elif method=='std':
        art, zscores = yasa.art_detect(data.T, sf=sf, window=2, hypno=hypno_unsampled, include=(0, 1, 2, 3, 4), 
                                      method='std', threshold=3, n_chan_reject=3, verbose='info')
   # Calculate both for the sake of comparison...     
    elif method=='both':
        art, zscores = yasa.art_detect(data.T, sf=sf, window=2, hypno=hypno_unsampled, include=(0, 1, 2, 3, 4), 
                           method='covar', threshold=3, n_chan_reject=3, verbose='info')
        art_std, zscores_std = yasa.art_detect(data.T, sf=sf, window=2, hypno=hypno_unsampled, include=(0, 1, 2, 3, 4), 
                                       method='std', threshold=3, n_chan_reject=3, verbose='info')
        # Correlation between covariance and std based artifact rejection
        print(f'The correlation between the two methods is r = {np.corrcoef(art, art_std)[0, 1]:.2f}')
    
    else:
        raise ValueError('Please select either ''covar'' or ''std'' based artifact rejection')
       
    # Art is an aray of 0 and 1, where 0 indicates a clean (or good epoch)  and 1 indicates an artifact epoch
    # print(f'{art.sum()} / {art.size} epochs rejected.')
    
    # Plot the artifact vector
    # plt.plot(art);
    # plt.yticks([0, 1], labels=['Good (0)', 'Art (1)']);
       
    # threshold = 3
    # perc_expected_rejected = (1 - erf(threshold / np.sqrt(2))) * 100
    # print(f'{perc_expected_rejected:.2f}% of all epochs are expected to be rejected.')
    
    # Actual
    # print(f'{(art.sum() / art.size) * 100:.2f}% of all epochs were actually rejected.')
    
    # The resolution of art is 2 seconds, so its sampling frequency is 1/2 = 0.5 Hz)
    sf_art = 1 / window
           
    if method == 'both':
        art_up = yasa.hypno_upsample_to_data(art, sf_art, data.T, sf)       
        art_up_std = yasa.hypno_upsample_to_data(art_std, sf_art, data.T, sf)
        # Add -1 to hypnogram where artifacts were detected
        hypno_with_art = hypno_unsampled.copy()
        hypno_with_art[art_up] = -1
        hypno_with_art_std = hypno_unsampled.copy()
        hypno_with_art_std[art_up_std] = -1
        
        # Proportion of each stage in ``hypno_with_art``
        # pd.Series(hypno_with_art).value_counts(normalize=True)
        # pd.Series(hypno_with_art_std).value_counts(normalize=True)
        
        return hypno_with_art, hypno_with_art_std, hypno_unsampled
        
    else:
        art_up = yasa.hypno_upsample_to_data(art, sf_art, data.T, sf)
        hypno_with_art = hypno_unsampled.copy()
        hypno_with_art[art_up] = -1
     
        return hypno_with_art, hypno_unsampled   
        
    # inspection of the data
    if inspection:
        # Plot the distribution of z-scores   
        if method=='covar':
            sns.distplot(zscores, label='covar')
        elif method=='std':
            # take mean across channels
            sns.distplot(zscores_std.mean(-1), label='std')
        elif method=='both':
            sns.distplot(zscores, label='covar')
            # take mean across channels
            sns.distplot(zscores_std.mean(-1), label='std')
        plt.title('Histogram of z-scores')
        plt.xlabel('Z-scores')
        plt.ylabel('Density')
        plt.axvline(3, color='r', label='Threshold')
        plt.axvline(-3, color='r')
        plt.legend(frameon=False);
        
        # plot data
        plt.figure()
        plt.plot(np.arange(len(data))/sf, data[...,:].mean(-1), label='Average of selected data channels')
        plt.vlines(np.where(hypno_with_art==-1)[0]/sf, ymin=-5000, ymax=5000, label='Detected artifacts')
        plt.legend(frameon=False);
        

#%%
    
def compare_hypnograms(subj_night : str, base_rater : str, comp_rater : str, save_hypno=False):
    path = '/media/administrator/data/Study_1_data/Hypnograms/Scoring_comparison/'

    # open a (new) file to write
    out = open(path + subj_night + '_report.txt', "w")
    
    # unravel hypnograms and compare overall agreement 
    base_hypno = unravel_hypnogram_visbrain(path + subj_night + '_hypno_' + base_rater + '.txt')
    comp_hypno = unravel_hypnogram_visbrain(path + subj_night + '_hypno_' + comp_rater + '.txt')
    agreement = np.mean(base_hypno == comp_hypno).round(3)*100
    out.write(f'Overall agreement between {base_rater} and {comp_rater} for {subj_night} hypnogram is {agreement} %' + '\n')
    
    # Matrix of stage counts
    # counts_base = np.vstack(np.unique(base_hypno, return_counts=True))
    # counts_comp = np.vstack(np.unique(comp_hypno, return_counts=True))
    
    # Sleep statistics based on hypnogram
    out.write(f'Sleep statistics for {base_rater} : {yasa.sleep_statistics(base_hypno, 1/30)}' + '\n')
    out.write(f'Sleep statistics for {comp_rater} : {yasa.sleep_statistics(comp_hypno, 1/30)}' + '\n')
    
    # Compute interrater reliability
    inter_agreement = cohen_kappa_score(base_hypno, comp_hypno).round(2)
    out.write(f'The inter-rate agreement between {base_rater} and {comp_rater} for {subj_night} hypnogram is K = {inter_agreement}' + '\n')
       
    # Compute accuracy by stage
    for idx, stage in enumerate(['Wake','N1','N2','N3','REM']):
        stage_acc = globals()[stage + '_acc'] = np.mean(comp_hypno[np.where(base_hypno==idx)] == idx).round(3)*100
        out.write(f'{comp_rater} has an accuracy of {stage_acc} % for stage {stage}' + '\n')
    
    # Compute where disgareement lies
    disagreement = ((np.where(base_hypno!=comp_hypno)[0] + 1)*30/60)
    out.write(f'{base_rater} and {comp_rater} disagree on the epochs at the given minute intervals: \n {disagreement}' + '\n')
 
   
    # Compute confusion matrix
    event_id ={'Wake':0,'Stage 1':1,'Stage 2':2,'Stage 3':3,'REM':4}
    target_names=event_id.keys()
    cm = confusion_matrix(base_hypno, comp_hypno, normalize='true')
    np.set_printoptions(precision=1)
    out.write(f'Confusion matrix: \n {cm}' + '\n')
    plt.figure()
    plot_confusion_matrix(cm, target_names)
    plt.savefig(path + 'Normalized_confusion_matrix_' + subj_night)
    
    # close txt file
    out.close()
    
    # save 'flattened' hypnograms
    if save_hypno:
        np.savetxt(path + subj_night + '_' + base_rater + '_hypno_flattened.txt', base_hypno, fmt='%d')
        np.savetxt(path + subj_night + '_' + comp_rater + '_hypno_flattened.txt', comp_hypno, fmt='%d')

save = [compare_hypnograms(base_rater='Mike', comp_rater='Lea', subj_night = '7XVWEVOK_' + str(i), save_hypno=True) for i in range(1,4)]

#%%

def unblind_files(sheet, recording_file=None):
    
    # =============================================================================
    # USE SHUTIL PACKAGE TO COPY AND MOVE FILES THEN OS RENAME NEW FILES BASED ON CONDITION
    # =============================================================================
    
    subj_cond = np.loadtxt(sheet, delimiter=',', dtype='str', skiprows=1) 
    cond_dict = {0:'sham', 1:'up', 2:'down'}
    if recording_file:
        path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental/' + recording_file
        hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/' + recording_file
    else:
        path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental/'
        hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/'
    target_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded'
    target_hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Unblinded'
    exp_files = sorted(os.listdir(path))
    hyp_files = sorted(os.listdir(hypno_path))
    
    all_elem = np.asarray([(x,y) for x in exp_files for y in hyp_files])
    exist = [elem[0].split('_')[0] + '_' + elem[0].split('_')[1] == 
             elem[1].split('_')[0] + '_' + elem[1].split('_')[1] for elem in all_elem]
    exist_exp_files, exist_hyp_files = all_elem[exist][:,0], all_elem[exist][:,1]
    
    for j, (d_files, h_files) in enumerate(zip(exist_exp_files, exist_hyp_files)):
        print(j, d_files, h_files)
        d_subj, h_subj = d_files.split('_')[0], h_files.split('_')[0]
        if d_subj != h_subj:
            raise TypeError('Subject entries do not align! Please check whether the paths have corresponding data and hypnogram files')
        subj = d_subj
        sub_pos = np.where(subj == subj_cond[:,0])[0][0]
        
        if '1' in d_files.split('_')[1]:
            cond = int(subj_cond[sub_pos,:][1])
        elif '2' in d_files.split('_')[1]:
            cond = int(subj_cond[sub_pos,:][2])
        elif '3' in d_files.split('_')[1]:
            cond = int(subj_cond[sub_pos,:][3])
        
        new_data_path = target_path + '/' + subj + '_' + cond_dict[cond] + '.p' 
        new_hypno_path = target_hypno_path + '/' + subj + '_' + cond_dict[cond] + '_hypno.txt'
        if not os.path.exists(new_data_path) and not os.path.exists(new_hypno_path):
            shutil.copy(path + d_files, new_data_path)
            print(f'Copying file: {d_files} into target path: {target_path} and renaming file: {d_files} with condition: {cond_dict[cond]} as {new_data_path}' )
            time.sleep(2)
            shutil.copy(hypno_path + h_files, new_hypno_path)
            print(f'Copying file: {h_files} into target path: {target_hypno_path} and renaming file: {h_files} with condition: {cond_dict[cond]} as {new_hypno_path}' )
            time.sleep(1)
        else:
            continue

#%%

def SW_summary(Data, subj_cond, sw_detection_method='abs', sp_coupling=False, sp_freq=(9,16), save=True): 
    """
    TO:DO - coupling only implemented for yasa sw detection methods

    """
    # channels to perform sw detection
    eeg_index = [i for i, x in enumerate(Data.chtypes) if x == "eeg"]
     
    # SW detection method
    if sw_detection_method!='ngo':
        if sp_coupling:
            sp_freq, coupling = sp_freq, True
        else:
            sp_freq, coupling = None, False
        # SW detection approach #1 - see Massimini et al., 2004 for more detail
        if sw_detection_method=='abs':
            sw = yasa.sw_detect(Data.data[:,eeg_index].T, Data.sfreq, ch_names = Data.chans[0:len(eeg_index)], 
                                hypno = Data.hypno_with_art, include=(2,3), freq_sw=(0.5, 2.0), dur_neg=(0.3, 1.5), 
                                dur_pos=(0.1, 1), amp_neg=(35, 300), amp_pos=(10, 200), amp_ptp=(75, 400), 
                                coupling=coupling, freq_sp=sp_freq, remove_outliers=True)
        
        # SW detection approach #2 - see Muehlroth & Werkle-Bergner, 2020 for more detail 
        elif sw_detection_method=='rel_zscore':
            thresh = np.percentile(np.abs(Data.data[:,eeg_index][Data.hypno_with_art==3]), 75)
            print('75th percentile threshold: %.2f uV' % thresh)
            sw = yasa.sw_detect(Data.data[:,eeg_index].T, Data.sfreq, ch_names = Data.chans[0:len(eeg_index)], 
                                hypno = Data.hypno_with_art, include=(2,3), freq_sw=(0.5, 2.0),
                                amp_neg=(None, None), amp_pos=(None, None), amp_ptp=(thresh, np.inf), 
                                coupling=coupling, freq_sp=sp_freq, remove_outliers=True)
        
        # SW detection approach #3 - see Helfrich et al., 2018 for more detail
        elif sw_detection_method=='rel_percentile': 
            data_zscored = zscore(Data.data[:,eeg_index].T)
            # Detect all events with a relative peak-to-peak amplitude between 3 to 10 z-scores, 
            # and positive/negative peaks amplitude > 1 standard deviations
            sw = yasa.sw_detect(data_zscored, Data.sfreq, ch_names = Data.chans[0:len(eeg_index)], 
                                hypno = Data.hypno_with_art, include=(2,3), freq_sw=(0.5, 2.0),
                                amp_neg=(1, None), amp_pos=(1, None), amp_ptp=(3, 10), 
                                coupling=coupling, freq_sp=sp_freq, remove_outliers=True)

        ## plot average SW
        # plt.figure; sw.plot_average(center = 'NegPeak', time_before=1, time_after=1.5)

    # SW detection approach #4 - see Ngo et al., 2015 for more detail
    elif sw_detection_method=='ngo':
        opts = wonambi.detect.DetectSlowWave(method='Ngo2015')
        n2_n3 = np.logical_or([Data.hypno_with_art==3],[Data.hypno_with_art==3]).ravel()
        sw = []
        for i in range(len(eeg_index)):
            sw_df = pd.DataFrame(wonambi.detect.slowwave.detect_Ngo2015(
                Data.data[:, eeg_index[i]][n2_n3], s_freq = Data.sfreq, 
                time = Data.times[n2_n3], opts = opts))
            sw_df['channel'] = Data.chans[0:len(eeg_index)][i] 
            sw.append(sw_df)
        Data.sw_summary = pd.concat(sw, axis=0)
        Data.sw_summary['sw_index'] = np.arange(0, len(Data.sw_summary))
        
    if save:
        # save as pickle file
        if sw_detection_method != 'ngo':
            Data.sw_summary = sw.summary().round(2)
        pickle.dump(Data.sw_summary, open(f'/media/administrator/data/Study_1_data/Statistics/' + subj_cond + '_sw_summary_' + sw_detection_method + '.p',"wb"))
        return Data.sw_summary
    else:
        if sw_detection_method != 'ngo':
            Data.sw_summary = sw.summary().round(2)
        return Data.sw_summary

#%%
path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded/'
hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Unblinded/'

def Data_SW(path, hypno_path):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    hypno_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(hypno_path) for i in files])
    for i, (data_files, hypno_files) in enumerate(zip(files_list, hypno_list)):
        #pass
        # load pre-processed data
        data, ch_names, ch_types, time_stamps, pinknoise_timestamps, classifier_predict, classifier_timestamps, sfreq = load_preprocessed_data(file=data_files)
        # convert to object
        Data = Data_Struct(data, ch_names, ch_types, time_stamps, pinknoise_timestamps, classifier_predict, classifier_timestamps, sfreq)
        
        # apply artifact detection for EEG channels only!
        eeg_index = [i for i, x in enumerate(Data.chtypes) if x == "eeg"]
        Data.hypno_with_art, Data.hypno = artifact_detect_hypno(Data.data[:,eeg_index], hypno_path = hypno_files, 
                                                                downsample=False, sf=Data.sfreq, method='covar', 
                                                                window=2)   
        Data.sw_summary.abs = SW_summary(Data, subj_cond=data_files.split('/')[-1].split('.')[0], 
                                         sw_detection_method='abs', sp_coupling=True, sp_freq=(9,16), save=True)
        Data.sw_summary.rel_zscore = SW_summary(Data, subj_cond=data_files.split('/')[-1].split('.')[0],
                                                sw_detection_method='rel_zscore', sp_coupling=True, sp_freq=(9,16), save=True)
        Data.sw_summary.rel_percentile = SW_summary(Data, subj_cond=data_files.split('/')[-1].split('.')[0],
                                                    sw_detection_method='rel_percentile', sp_coupling=True, sp_freq=(9,16), save=True)
        Data.sw_summary.ngo = SW_summary(Data, subj_cond=data_files.split('/')[-1].split('.')[0],
                                         sw_detection_method='ngo', sp_coupling=False, save=True)
        #print(f'The slow wave detection method calculated {Data.sw_summary.m}')
        
        # save data
        pickle.dump(Data, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded_cleaned/' + data_files.split('/')[-1],"wb"))
        pickle.dump(Data.hypno_with_art, open('/media/administrator/data/Study_1_data/Hypnograms/Unblinded_clean/' + hypno_files.split('/')[-1], "wb"))
       
        #return Data

Data_SW(path, hypno_path)

#%%    
path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded_cleaned/'         
def SW_ERPs(path, filter_data=False, downsample=False, inspection=False):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    for i, data_files in enumerate(files_list):
        # load pre-processed data
        Data = pickle.load(open(data_files,"rb")) 

        # filter data
        if filter_data:
            Data.fir_filter(0.5, 2)
            logging.warning('Filtering the data will result in the deletion of the non-EEG channels!')    
            
        # downsample data
        if downsample:
            Data.downsample(128)  
        
        # eeg index
        eeg_index = [i for i, x in enumerate(Data.chtypes) if x == "eeg"]
    
        # get a mask of detected SW
        mask = np.zeros(Data.data[:,eeg_index].T.shape, dtype=int)
        if data_files.endswith('ngo.p'):
            start, end = 'start', 'end'
        else:
            start, end = 'Start', 'End'
        idx_ev = yasa.others._index_to_events(Data.sw_summary[[start, end]].to_numpy() * Data.sfreq)
        mask[:, idx_ev] = 1
        sw_highlight = np.squeeze(Data.data[:,Data.chans.index('C3')] * mask)
        sw_highlight[sw_highlight == 0] = np.nan  
                    
        # time point synchronization
        Data.pinknoise_timestamps_sync = [Data.times[np.abs(Data.times - Data.pinknoise_times[ix]).argmin()] for ix in range(len(Data.pinknoise_times))]
        Data.pinknoise_timestamps_sync_adj = np.asarray(Data.pinknoise_timestamps_sync)*Data.sfreq
        first_bursts = Data.pinknoise_timestamps_sync_adj[::2]
        
        # sw times in bool, index, and raw times
        sw_times_bool = ~np.isnan(sw_highlight[Data.chans.index('C3'),:])
        sw_times_idx = np.where(~np.isnan(sw_highlight[Data.chans.index('C3'),:]))[0]
        sw_times_times = Data.times[np.argwhere(~np.isnan(sw_highlight[Data.chans.index('C3'),:])).ravel()]
        
        # find pinknoise burst times imposed onto signal 
        insp_ix = [np.where(Data.times*Data.sfreq == first_bursts[i])[0][0] for i in range(len(first_bursts))]
        
        ## for mne implementation
        # events = np.asarray([[insp_ix[i],1,1] for i in range(len(insp_ix))])
        # event_id = dict(pinknoise=1)
        
        # center data around pinknoise bursts
        if len(insp_ix) > 0:
            center, _ = yasa.get_centered_indices(Data.data[:,Data.chans.index('C3')], 
                                                  np.asarray(insp_ix), npts_before = Data.sfreq*1, npts_after = Data.sfreq*1)
            
            # check if slow wave is contained in pinknoise ERP epoch
            sync_sw_noise_idx = [np.where(sw_times_bool[center[idx,:]])[0] for idx, times in enumerate(center)]  
            
            # update center events where it is 
            center = center[[i for i in range(len(sync_sw_noise_idx)) if len(sync_sw_noise_idx[i])>1],:]
            
            # # mask for pinknoise times
            # mask = np.zeros(Data.data[:,[Data.chans.index('C3')]].T.shape, dtype=int)
            # mask[:, insp_ix] = 1
            # pn_highlight = np.squeeze(Data.data[:,Data.chans.index('C3')] * mask)
            # pn_highlight[pn_highlight == 0] = np.nan  
            # pn_highlight_bool = ~np.isnan(pn_highlight) 
            # pn_highlight_idx = np.where(~np.isnan(pn_highlight))[0]
            # center2, _ = yasa.get_centered_indices(data[:,Data.chans.index('C3')], np.where(np.asarray(pn_highlight_bool[center])), 
            #                                       npts_before = Data.sfreq*2, npts_after = Data.sfreq*4)
            
            # neg peaks of detected slow waves 
            mask = np.zeros(Data.data[:,[Data.chans.index('C3')]].T.shape, dtype=int)
            idx_ev = yasa.others._index_to_events(Data.sw_summary[['NegPeak', 'NegPeak']][Data.sw_summary['Channel']=='C3'].to_numpy() * Data.sfreq)
            mask[:, idx_ev] = 1
            sw_highlight_negpeaks = np.squeeze(Data.data[:,Data.chans.index('C3')] * mask)
            sw_highlight_negpeaks[sw_highlight_negpeaks == 0] = np.nan  
            sw_highlight_negpeaks_bool = ~np.isnan(sw_highlight_negpeaks)
            sw_highlight_negpeaks_idx = np.where(~np.isnan(sw_highlight_negpeaks))[0]
            sw_highlight_negpeaks_times = Data.times[np.argwhere(~np.isnan(sw_highlight_negpeaks)).ravel()]
            sync_sw_negpeak_idx = [np.where(sw_highlight_negpeaks_bool[center[idx,:]])[0] for idx, times in enumerate(center)] 
            # center_sw_neg, _ = yasa.get_centered_indices(data[:,Data.chans.index('C3')], sw_highlight_negpeaks_idx,
            #                                           npts_before = Data.sfreq*2, npts_after = Data.sfreq*4)
            
            # # update center events where it is 
            # center = center[[i for i in range(len(sync_sw_negpeak_idx)) if len(sync_sw_negpeak_idx[i])>1],:]
            
            # a = [np.where(center_sw_neg[idx,:]==center[idx,:])[0] for idx, times in enumerate(center)]
            
            # center3, _ = yasa.get_centered_indices(data[:,Data.chans.index('C3')],pn_highlight_bool[sw_highlight_negpeaks_idx]),
            #                                         npts_before = Data.sfreq*2, npts_after = Data.sfreq*4)
        
        
            what, which = np.where(sw_highlight_negpeaks_bool[center])
            where = pd.DataFrame(which, what)
            a = [np.asarray(where[0][lar]) for idx, lar in enumerate(np.unique(what))]
            b = [a[idx].ravel()[0] for idx in range(len(a))]
            blap = pd.DataFrame(b,np.unique(what))
            c = [center[lar,:][blap[0][lar]] for idx, lar in enumerate(np.unique(what))]
            
            # new center around closest negative peak to first pinknoise time burst
            new_center, _ = yasa.get_centered_indices(Data.data[:,Data.chans.index('C3')], np.asarray(c), 
                                                      npts_before = Data.sfreq*2, npts_after = Data.sfreq*4)
            
            
                
            info = mne.create_info(ch_names=['C3'], sfreq=512, ch_types=['eeg'])
            epochs = mne.EpochsArray(np.expand_dims(Data.data[:, Data.chans.index('C3')][new_center]/1e6, 1), info, tmin = -2, baseline=None)
            
            # info = mne.create_info(ch_names=Data.chans[0:len(eeg_index)], sfreq=512, ch_types=Data.chtypes[0:len(eeg_index)])
            # epochs = mne.EpochsArray(Data.data[:, eeg_index][new_center].swapaxes(1,2)/1e6, 1, info, tmin = -2, baseline=None)
            
            if inspection: 
                sns.set(style='darkgrid', font_scale=1.2)
                plt.figure(figsize=(16, 4.5))
                plt.plot(Data.times, Data.data[:,Data.chans.index('C3')], 'k')
                plt.plot(Data.times, sw_highlight, 'indianred')
                plt.plot(summary['NegPeak'], sw_highlight[(summary['NegPeak'] * Data.sfreq).astype(int)], 'bo', label='Negative peaks')
                plt.plot(summary['PosPeak'], sw_highlight[(summary['PosPeak'] * Data.sfreq).astype(int)], 'go', label='Positive peaks')
                plt.plot(summary['Start'], Data.data[:,Data.chans.index('C3')][(summary['Start'] * Data.sfreq).astype(int)], 'ro', label='Start')
                plt.vlines(Data.pinknoise_times, ymin=-500, ymax=500, colors='y', label='Pinknoise bursts') 
                plt.xlabel('Time (seconds)')
                plt.ylabel('Amplitude (uV)')
                plt.xlim([0, Data.times[-1]])
                plt.title('Detected Slow Waves - C3')
                plt.legend()
                sns.despine()   
            
        else:
            logging.warning(f'The following subject condition file: {data_files[-15:-2]}, has no pinknoise bursts!')
         
        # Save epochs as numpy arrays
        pickle.dump(epochs, open('/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/' + data_files.split('/')[-1].split('.')[0] + '_epochs.p',"wb"))

#%%

path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/' 
def group_SW_ERPs(path):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    Up, Down, Sham = [], [], []
    for i, data_files in enumerate(files_list):
        if data_files.endswith('up_epochs.p'):
            epochs = pickle.load(open(data_files,"rb"))
            Up.append(epochs)
        elif data_files.endswith('down_epochs.p'):
            epochs = pickle.load(open(data_files,"rb"))
            Down.append(epochs)
        elif data_files.endswith('sham_epochs.p'):
            epochs = pickle.load(open(data_files,"rb"))
            Sham.append(epochs)
   
    down = [Down[idx].to_data_frame() for idx in range(len(Down))]
    up = [Up[idx].to_data_frame() for idx in range(len(Up))]
    sham = [Sham[idx].to_data_frame() for idx in range(len(Sham))]
    
    for idx in range(len(down)):
        down[idx]['condition']=idx
    for idx in range(len(up)):
        up[idx]['condition']=idx
    for idx in range(len(sham)):
        sham[idx]['condition']=idx
        
    df_down = pd.concat(down, axis=0)
    df_up = pd.concat(up, axis=0)
    df_sham = pd.concat(sham, axis=0)

    df = pd.concat([df_down, df_up, df_sham], keys=['Down', 'Up', 'Sham'],
                   names=['Condition'], axis=0)
    df['time'] = df['time']/1000
    df = df.rename(columns={'C3':'Amplitude (uV)','time': 'Time (s)', 'condition': 'Subject', 'epoch':'Epoch'})
   
    return df

def plot_SW_ERPs(df, title='Pinknoise SWA Acute Modulation', ci=95):    
    # df.loc[('Sham')]; df.loc[('Up')]; df.loc[('Down')]
    
    sns.set(style='darkgrid', font_scale=1.2)
    # plot = sns.relplot(x="Times", y="Amplitude", kind="line",
    #                    hue=df.index.get_level_values(0), data=df);
    plot = sns.lineplot(x="Time (s)", y="Amplitude (uV)", hue=df.index.get_level_values(0), 
              data=df, ci=ci);
    sns.despine()
    plt.legend(frameon=False);
    plt.title(title)

    return plot

df = group_SW_ERPs(path)
# fig = plot_SW_ERPs(df)

#%%
## Permutation Cluster Testing

def permutation_cluster_test_mne(epochs : list):
    threshold = 6.0
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test(epochs, n_permutations=1024, threshold=threshold, 
                                 tail=1, n_jobs=1, out_type='mask')
    
    return T_obs, clusters, cluster_p_values, H0

T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test_mne([epochs_up.get_data()*1e6, 
                                                                      epochs_down.get_data()*1e6, 
                                                                      epochs_sham.get_data()*1e6])

def permutation_cluster_test_plot(epochs : list, times, T_obs, clusters, cluster_p_values, H0):
    """
    TO:DO: Fix plotting of conditional contrasts....
    """
    plt.figure()
    plt.subplot(211)
    plt.title("Permutation Clustering F-test for Slow Wave Condition Comparison")
    # plt.title('Channel : ' + epochs_up.info['ch_names'][0])
    plt.plot(times, np.squeeze(epochs[0][0].mean(axis=0)) - np.squeeze(epochs[1][0].mean(axis=0)),
             label="ERP Contrast (Up - Down)")
    plt.plot(times, np.squeeze(epochs[0][0].mean(axis=0)) - np.squeeze(epochs[2][0].mean(axis=0)),
             label="ERP Contrast (Up - Sham)")
    plt.plot(times, np.squeeze(epochs[1][0].mean(axis=0)) - np.squeeze(epochs[2][0].mean(axis=0)),
             label="ERP Contrast (Down - Sham)")
    plt.ylabel("Amplitude (uV)")
    plt.legend(frameon=False);
    plt.subplot(212)
    for i_c, c in enumerate(clusters):
        if cluster_p_values[i_c] <= 0.01:
            h = plt.axvspan(times[c[0]][0], times[c[0]][-1], 
                            color='r', alpha=0.3)
        else:
            plt.axvspan(times[c[0]][0], times[c[0]][-1], 
                        color=(0.3, 0.3, 0.3), alpha=0.3)
    hf = plt.plot(times, np.squeeze(T_obs), 'g')
    plt.legend((h, ), ('cluster p-value < 0.01', ), frameon=False)
    plt.xlabel("time (s)")
    plt.ylabel("f-values")
    plt.show()

permutation_cluster_test_plot([epochs_up.get_data()*1e6, epochs_down.get_data()*1e6, epochs_sham.get_data()*1e6], 
                              epochs_up.times, T_obs, clusters, cluster_p_values, H0)

#%%
## plot spectrogram

peak = PeakLockedTF(np.squeeze(epochs_up.get_data())*1e6, sf=512, cue=0., times=epochs_up.times,
                    f_pha=[0.5, 30],f_amp=(0.5, 30, 0.3, 0.3), n_jobs=16, verbose=True)

plt.figure(figsize=(8, 8))
ax_1, ax_2 = peak.plot(zscore=True, baseline=None, cmap='Spectral_r',
                            vmin=-1, vmax=1)
# add_motor_condition(135, color='black', ax=ax_1)
plt.tight_layout()
plt.show()

# altered plot
plt.figure(figsize=(8, 8))
plt.title('Up-state targeted spectral modulation')
ax_1, ax_2 = peak.plot_altered(zscore=True, baseline=None, cmap='Spectral_r',
                            vmin=-1, vmax=2)
plt.title('Up-state targeted spectral modulation')
# add_motor_condition(135, color='black', ax=ax_1)
plt.tight_layout()
plt.show()

mean_epochs = np.squeeze(epochs_up.get_data())[0:500,:].mean(0)*1e6
min_peak_time = np.where(mean_epochs == np.min(mean_epochs))[0][0]
max_peak_time = np.where(mean_epochs == np.max(mean_epochs))[0][0]

#%%
## plot spectrogram - Daniela method 

def mne_wavelet_tutorial():
    freqs = np.arange(0, 256.5,.5)
    n_cycles = freqs / 2.
    vmin, vmax = -3., 3.  # Define our color limits.
    power = mne.time_frequency.tfr_array_morlet(epochs_up.get_data(), sfreq=epochs.info['sfreq'],
                             freqs=freqs, n_cycles=n_cycles,
                             output='avg_power')
    mne.baseline.rescale(power, epochs.times, (0., 0.1), mode='mean', copy=False)
    fig, ax = plt.subplots()
    x, y = mne.viz.centers_to_edges(epochs.times * 1000, freqs)
    mesh = ax.pcolormesh(x, y, power[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title('TFR calculated on a numpy array')
    ax.set(ylim=freqs[[0, -1]], xlabel='Time (ms)')
    fig.colorbar(mesh)
    plt.tight_layout()
    plt.show()

def get_band_power_wavelet(win_sig, win_size_s, low, high, fs, scale = ''):
      
    win_sig = np.array(win_sig)
    f = np.logspace(-1, 3, 150)
    f = f[f>low]
    f = f[f<high]
    n_cyc = np.linspace(.5,17, num = len(f))
    # n_cyc = n_cyc[np.mod(n_cyc,2)!=0]
    # n_cyc = n_cyc[range(len(f))]
    Sxx = cwt_morlet(win_sig, fs, f, use_fft=False, n_cycles=n_cyc)
    t=np.arange(0,win_size_s,1/fs)
    Sxx = np.abs(Sxx)
    Sxx = np.median(Sxx, axis=0)
    
    if scale == 'dB':
        Sxx = 10 * np.log10(Sxx)
    # # plt.plot(freqs,psd)
    # Sxx_mean = Sxx_sum/c
    # Sxx_std = np.sqrt(Sxx_sum2/c)
    
    return f, t, Sxx

def plot_spectrogram(ax, f, t, Sxx, win_size_s, low, high, title, clim):
    from matplotlib import cm
    cmap = cm.Spectral_r
    Sxx = np.asarray(Sxx)
    if len(Sxx.shape) > 1:
        i = np.logical_and(f >= low, f <= high)
        try:
            fig, ax = plt.subplots()
            CM = ax.pcolormesh(t-win_size_s/2, f[i], Sxx[i], shading='gouraud', cmap=cmap, rasterized = True, vmin=clim[0], vmax=clim[1])
        except AttributeError:
            print('creating new figure ...')
            fig, ax = plt.subplots()
            CM = ax.pcolormesh(t-win_size_s/2, f[i], Sxx[i], shading='gouraud', cmap=cmap, rasterized = True, vmin=clim[0], vmax=clim[1])
       
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        cbar = plt.colorbar(CM, ax=ax)
    else:
        i = np.logical_and(f >= low, f <= high)
        ax.plot(f[i],Sxx[i])
        ax.set_ylabel('signal power')
        ax.set_ylabel('Frequency [Hz]')
   
    ax.set_yscale('log')
    ax.set_title(str(title))
 
f, t, Sxx3 = scipy.signal.spectrogram(np.squeeze(epochs_up.get_data())[0,:]*1e6, fs=512, nperseg=1024)

Sxx = np.squeeze(Sxx)

plot_spectrogram(ax=[], f=f, t=t, Sxx=Sxx, win_size_s=0, low=0.5, high=30, title='nothing', clim=[np.min(Sxx), np.max(Sxx)])
 

#%%        
# plot artifacts
avg_data = data[..., eeg_index].mean(-1, keepdims=True)
times = np.arange(0, len(data))/sf
plt.plot(times, data[:,0:7])
plt.vlines(np.where(hypno_with_art==-1)[0]/128, ymin=-1000, ymax=1000, colors='k', linestyles='dotted')
plt.vlines(np.where(hypno_with_art_std==-1)[0]/128, ymin=-750, ymax=750, colors='r',linestyles='dotted')

# Plot multitaper spectrogram with hypno (no artifact)
yasa.plot_spectrogram(data[:,1], sf, hypno = hypno_unsampled, fmax=30, cmap='Spectral_r');

# Plot new hypnogram and spectrogram on Fz (with artifact)
yasa.plot_spectrogram(data[:,0], sf=128, hypno = hypno_with_art, fmax=30, cmap='Spectral_r');


#%%

def load_mult_xdf(path: str):
    data=[]
    marker=[]
    for file in sorted(os.listdir(path)):
        if file.endswith(".xdf"):
            streams, fileheader = pyxdf.load_xdf(file)
            data.append(streams[0]['time_series'])
            marker.append(streams[0]['time_stamps'])

 
data = np.concatenate(data)
times = np.concatenate(marker)
C3 = mne.filter.filter_data(data[:,4].astype('float64'), sfreq=500, l_freq=0.5, h_freq=35)*1e6
C3_notched = mne.filter.notch_filter(C3, Fs=500, freqs=(50,100,150))

#%%
path = '/media/administrator/data/Study_1_data/Pre-processed_data/Adaption/6QJ3ITMT_adaption_preproc_data.p'
hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Adaption/6QJ3ITMT_adaption_hypno.txt'

# load pre-processed data
data, ch_names, ch_types, time_stamps, pinknoise_timestamps, classifier_predict, classifier_timestamps, sfreq = pickle.load(open(path,"rb"))
        
def peak2peak_SW_duration(data, hypno_path, ch_names, chan='C3', sf=512, data_len=210):
    # epoch data
    _, epochs = yasa.sliding_window(data[:,ch_names.index(chan)].T, sf, window=30)
    # load hypnogram
    hypno = unravel_hypnogram_visbrain(hypno_path)
    # unsampled hypnogram
    hypno = yasa.hypno_upsample_to_data(hypno=hypno[0:data_len*2], sf_hypno=1/30, data=data[0:sf*60*210,:].T, sf_data=sf)
    # calculate sw dataframe for detected SWs
    sw = yasa.sw_detect(data[0:sf*60*data_len,ch_names.index(chan)].T, sf, ch_names = [chan], hypno = hypno, include=(2,3), freq_sw=(0.5, 3.0),
                   dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(35, 300), amp_pos=(10, 200), 
                   amp_ptp=(75, 400), remove_outliers=True)
    # print summary
    sw2 = sw.summary()
    
    # return mean peak to peak time duration
    return sw2['PosPeak'].mean() - sw2['NegPeak'].mean()


p2p = peak2peak_SW_duration(data, hypno_path, ch_names)

#%%

## create mne info
 
# data = raw.get_data()*1e6
# data = data.T
   
    
#%%

## Artifact rejection fun
# ECG rejection
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
ecg_epochs.plot_image(combine='mean')

eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
eog_epochs.plot_image(combine='mean')
eog_epochs.average().plot_joint()
            
#%%
## IRASA

# # Apply the IRASA technique
# freqs, psd_aperiodic, psd_osc = yasa.irasa(data2.T, sf, band=(1, 30), win_sec=4, return_fit=False)

# # Plot the aperiodic component on a linear-log scale
# plt.plot(freqs2, psd_aperiodic[0,:], 'k', lw=2)
# plt.xlim(10, 100)
# plt.yscale('log')
# sns.despine()
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD log($uV^2$/Hz)');

# # Plot the oscillatory component on a linear-linear scale
# plt.plot(freqs2, psd_osc[0,:], 'k', lw=2)
# plt.xlim(10, 100)
# sns.despine()
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD log($uV^2$/Hz)');

# # Plot the oscillatory + aperiodic component on a linear-log scale
# psd_combined = psd_aperiodic[0, :] + psd_osc[0, :]
# plt.plot(freqs2, psd_combined, 'k', lw=2)
# plt.fill_between(freqs2, psd_combined, cmap='Spectral')
# plt.xlim(10, 100)
# plt.yscale('log')
# sns.despine()
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('PSD log($uV^2$/Hz)');

# freqs, psd_aperiodic, psd_osc, fit_params = yasa.irasa(data2.T, sf)
# fit_params

#%%   
     
# PSD-based signal quality check
freqs, psd = welch(EEG[:,Data.chans.index('C3')].T, fs=512, nperseg=1024, average='median')
plt.semilogy(freqs, psd.T)       
# plt.ylim([0.5e-3, 1e5])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
               
## Initialize all channels as functioning -> 1
# channel_failureArray = np.ones(len(new_chans))
  
# ## Baseline spectral density ratio levels
# # calculate relative alpha power for all channels
# rel_alpha = bp[2::6,:]
# # calculate relative line noise power for all channels
# rel_ln = bp[5::6,:]
# # calculate alpha/line ratio for all channels
# ln_alpha = rel_ln/rel_alpha
# channel_failure_feat = list(ln_alpha)
# epoch_check = channel_failure_feat[-1] > 5
# if any(epoch_check):
#     channel_failureArray[np.where(epoch_check)[0]] = 0
# else:
#     # channel failure array remains the same
#     channel_failureArray = channel_failureArray
        
        
        
# def channel_failure_test_offline(data, fs=512, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'),
#                                               (8, 12, 'Alpha'), (12, 16, 'Sigma'),
#                                               (16, 30, 'Beta'), (49, 51, 'Line noise')], relative=True):
#     """Offline channel failure detection

#     This function assess the stability of a given input of spectral density features,
#     where the input corresponds to an array, typically consisting of a [1 x 30] array
#     with spectral features for a given epoched. These are then parsed and made into a 
#     ratio of line noise (49 - 51 Hz) to alpha (8 - 12 Hz) 

#     Parameters
#     ----------
#     epochs :  numpy array of [n_epochs x n_chans*n_spectral_feats]
#         Spectral density feature array.  
              
#     Returns
#     -------
#     stage_predictArrays : list of integer values
#         A list containing the status of selected channels, either if they are intact --> 1,
#         or whether they have failed --> 0. They reflect the status of C3, Cz, C4, EOG_average,
#         and EMG_average. For example, if all channels are functional --> [1,1,1,1,1], if all
#         have failed --> [0,0,0,0,0]    
#     """
#     # epoch data
#     _, epochs = yasa.sliding_window(data, sf=fs, window=30, axis=0)
    
#     # extract relative or absolute spectral density values for frequency bands of interest
#     bp = bandpower(epochs, fs=fs)
    
#     # reshape data 
#     chans, epoch, samples = epochs.shape
#     #bp = bp.reshape(1, chans*epoch*6, order='F') # could check to see if 'len(bands)' works
    
#     # initialize channels as functional --> 1
#     channel_failureArray = np.hstack(np.ones(chans))
      
#     ## Baseline spectral density ratio levels
#     # calculate relative alpha power for all channels
#     rel_alpha = bp[:,2::6]
#     # calculate relative line noise power for all channels
#     rel_ln = bp[:,5::6]
#     # calculate alpha/line ratio for all channels
#     ln_alpha = rel_ln/rel_alpha
#     channel_failure_feat = list(ln_alpha)
#     failure = np.where(channel_failure_feat[0]> 5)[0]
    
#     alpha_stack = []
#     for i in range(np.size(bp, axis=-1)):
#         alpha_stack.append(np.hstack(bp[:,:,i][2]))
#     rel_alpha = np.concatenate(alpha_stack)
    
#     ln_stack = []
#     for i in range(np.size(bp, axis=-1)):
#         ln_stack.append(np.hstack(bp[:,:,i][5]))
#     rel_ln = np.concatenate(ln_stack)
    
#     ln_alpha = rel_ln/rel_alpha
#     failure_2 = np.where(ln_alpha > 5)[0]
    
#     epoch_check = channel_failure_feat[-1] > 5
#     if any(epoch_check):
#         channel_failureArray[np.where(epoch_check)[0]] = 0
#     else:
#         # channel failure array remains the same
#         channel_failureArray = channel_failureArray
                
    
#     channel_failure_featArray = []
#     channel_failureArray = []
#     ## Initialize all channels as functioning -> 1
#     channel_failureArray = np.ones(len(new_chans))
        
#     ## Baseline spectral density ratio levels
#     # calculate relative alpha power for all channels
#     rel_alpha = epochs[:,2::6]
#     # calculate relative line noise power for all channels
#     rel_ln = epochs[:,5::6]
#     # calculate alpha/line ratio for all channels
#     ln_alpha = rel_ln/rel_alpha
#     # combine the above into a list to be able to do baseline/ongoing comparisons
#     channel_failure_feat = list(ln_alpha) 
#     # append into list
#     channel_failure_featArray.append(channel_failure_feat[0])
    
#     ## Ongoing check to see if threshold is exceeded
#     # individual epoch check (change in ln_alpha ratio)
#     epoch_check = channel_failure_feat[-1] > 5
#     if any(epoch_check): 
#         channel_failureArray[np.where(epoch_check)[0]] = 0
#     else:
#         # channel failure array remains the same
#         channel_failureArray = channel_failureArray
#     # epoch to epoch change in ln_alpha ratio, can only occur 
#     # if list of channel_failure_feats contains more than one index
#     if len(channel_failure_featArray) < 1:
#         epoch_diff = channel_failure_featArray[-2] - channel_failure_featArray[-1] > 2                                                    
#         if any(epoch_diff): 
#             channel_failureArray[np.where(epoch_diff)[0]] = 0
#         else:
#             # channel failure array remains the same
#             channel_failureArray = channel_failureArray
#     # compare baseline and present epoch (ln_alpha ratio)
#     baseline_check = ln_alpha[-1] - channel_failure_featArray[0] > channel_failure_featArray[0]*1.3
#     if any(baseline_check):
#         channel_failureArray[np.where(baseline_check)[0]] = 0
#     else:
#         # channel failure array remains the same
#         channel_failureArray = channel_failureArray

#     ## Return updated channel_failureArray
#     return channel_failureArray 

    
    
#%%
# second_sound_delay_diff = np.diff(pinknoise_timestamps)[::2]   
# difference_sorted_2 = np.sort(second_sound_delay_diff) 
    
    
data_narrow = mne.filter.filter_data(data.T, sfreq=512, l_freq=0.5, h_freq=4)

filtparams_lp = butter(4, 4, fs = 128)
data_narrow = filtfilt(*filtparams_lp, Data.data.T)
crossings = thresholdcrossings(data_narrow[3,:], -35)
# plt.plot(eego_times, data_narrow[2,:])
# # plt.plot(eego_times, data[:,2])
# plt.vlines(pinknoise_timestamps, ymin=-2000, ymax=2000, colors='r')
# plt.vlines(data_narrow[2,:][crossings], ymin=-100, ymax=100, colors='o')
    
# plt.vlines([eego_times, crossings], ymin=-2000, ymax=2000, colors='r')


plt.plot(time, data_narrow[3,:])
plt.vlines(time[crossings], ymin=-1000, ymax=1000)
plt.vlines(pinknoise_timestamps - time_stamps[0], ymin=-750, ymax=750, colors='r')
