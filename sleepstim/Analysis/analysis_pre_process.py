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
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import pickle 
import pandas as pd
import pyxdf
import yasa 
import mne
import pingouin as pg
from mne.stats import permutation_cluster_test
from pyriemann.estimation import Covariances, Shrinkage
from pyriemann.clustering import Potato
import logging
import time
import wonambi
import seaborn as sns
import Levenshtein as lev
from scipy.signal import welch, butter, filtfilt
from scipy.stats import zscore, skewnorm
from scipy.special import erf
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
from meegkit import dss, star, asr
from meegkit.utils import demean, normcol
from os import chdir as cd
from os import listdir
import os, shutil
from sklearn.ensemble import IsolationForest
from sleepstim.sleep_funs import (bfr_butter_filt, bandpower, unravel_hypnogram_visbrain, 
                                  downsample_scaled, load_xdf, channel_parser, thresholdcrossings, 
                                  plot_confusion_matrix)
#sns.set(style='darkgrid', font_scale=1.2)

class Data_Struct:
    def __init__(self, data, chans, chtypes, times, pinknoise_times, classif_predict, classif_times, sfreq):
        self.data = data
        self.chans = chans
        self.chtypes = chtypes
        self.times = times - times[0]
        self.pinknoise_times = pinknoise_times - times[0]
        if np.all(np.diff(self.pinknoise_times[0:3]) < 0.5):
            self.pinknoise_times_sync = [np.argmin(np.abs(self.times - ts)) for ts in self.pinknoise_times[3::]] 
        else:
            self.pinknoise_times_sync = [np.argmin(np.abs(self.times - ts)) for ts in self.pinknoise_times] 
        self.classif_times = classif_times - times[0]
        self.classif_times_sync = [np.argmin(np.abs(self.times - ts)) for ts in self.classif_times]  
        self.classif_predict = classif_predict
        self.sfreq = sfreq
        self.hypno = []
        self.hypno_with_art = []
        self.bad_chans = []
        self.sws = []
        self.spindles = []
        print('Creating RawArray with %s data, of length %s minutes, containing %s channels'
              % (self.data.dtype, round(data.shape[0]/sfreq/60, 2), data.shape[1]))
        
    def detect_bad_chans(self, method='EQI'):
        if method == 'EQI':
            from sleepstim.Analysis import bad_channel_detection
            eeg_index = [i for i, x in enumerate(self.chtypes) if x == "eeg"]
            eeg_chans = list(np.asarray(self.chans)[eeg_index])
            EEG_qi = bad_channel_detection.qc_calcEQI(self.data[:,eeg_index].T, self.sfreq, frange=(0.5,30))
            # create dataframe for EQI results 
            eqi_lab = ['avgspec_0.5-30','line_noise','root_mean_sq','max_gradient','zero-crossing_rate','kurtosis']
            df_comb = [pd.DataFrame(data = EEG_qi[:,:,i].T, columns = eqi_lab, index = eeg_chans)
                       for i in range(EEG_qi.shape[-1])]
            for i in range(EEG_qi.shape[-1]): df_comb[i]['Epoch'] = i
            df_comb2 = pd.concat(df_comb)
            ilf = IsolationForest(contamination='auto', max_samples='auto',
                                  verbose=0, random_state=42)
            good = ilf.fit_predict(df_comb2.drop(['Epoch'], axis=1))
            good[good == -1] = 0
            df_comb2['Isolation forest score'] = good
            # isolation forest scores by channel
            df_if_scores = pd.DataFrame([df_comb2.loc[(self.chans[i])]['Isolation forest score'].value_counts(normalize=True) 
                                         for i in range(len(eeg_chans))])
            df_if_scores['Channel'] = eeg_chans
            bads = df_if_scores['Channel'][df_if_scores[0] > 0.25].to_list()
            logging.warning(f'The EQI method detected the following as bad channels: {bads}!')
            self.bad_chans = bads
        
        ## RANSAC
        elif method == 'RANSAC':
            from autoreject import Ransac
            _, epoched_data = yasa.sliding_window(self.data.T, sf=self.sfreq, window=2)
            info = mne.create_info(ch_names=self.chans, sfreq=self.sfreq, ch_types=self.chtypes)
            epochs = mne.EpochsArray(epoched_data/1e6, info, tmin = 0, baseline=(None), verbose=0) 
            epochs.set_montage(mne.channels.make_standard_montage('standard_1005'), verbose=0)
            picks = mne.pick_types(epochs.info, eeg=True, stim=False, eog=False,
                                   include=[], exclude=[])
            ransac = Ransac(verbose=False, picks=picks, n_jobs=-1)
            ransac.fit_transform(epochs)
            logging.warning(f'The RANSAC algorithm detected the following as bad channels: {ransac.bad_chs_}!')
            self.bad_chans = ransac.bad_chs_
        
    def downsample(self, new_sfreq):
        self.data = downsample_scaled(self.data, self.sfreq, new_sfreq)
        self.times = np.arange(len(self.data))/new_sfreq
        self.sfreq = new_sfreq
        
    def fir_filter(self, l_freq, h_freq):
        eeg_index = [i for i, x in enumerate(self.chtypes) if x == "eeg"]
        self.data = mne.filter.filter_data(self.data[:, eeg_index].T, self.sfreq, l_freq, h_freq).T
        
    def notch_filter(self, freqs):
         self.data = mne.filter.notch_filter(x = self.data.T, Fs = self.sfreq, freqs=freqs).T
             
    def art_detection(self, method='covar', window=2, downsample=False, hypno_path = ''):
        # apply artifact detection for a selct number of core EEG channels
        eeg_index = np.r_[self.chans.index('F3'), self.chans.index('Fz'), self.chans.index('F4'), 
                          self.chans.index('C3'), self.chans.index('Cz'), self.chans.index('C4'), 
                          self.chans.index('P3'), self.chans.index('Pz'), self.chans.index('P4')]
        # epoch data
        _, epoched_data = yasa.sliding_window(self.data.T, sf=self.sfreq, window=30)
        # eliminate bad channels from covariance artifact detection approach
        bad_idx = np.asarray([np.r_[self.chans.index(self.bad_chans[i])][0] for i in range(len(self.bad_chans))])
        good_idx = np.asarray([item for item in eeg_index if item not in bad_idx])
        
        # apply artifact detection
        self.hypno_with_art, self.hypno = artifact_detect_hypno(self.data[:,good_idx], hypno_path = hypno_path, 
                                                                downsample=downsample, sf=self.sfreq, method=method, 
                                                                window=window)  
        
    def plot_zscore_signal(self, zscore_val=3, chan='C3'):
        """
        Returns plotted signal with z_score deviations
        
        """
        plt.plot(self.times, self.data[:,self.chans.index(chan)])
        plt.vlines(self.times[zscore(np.abs(self.data[:,self.chans.index(chan)]))>zscore_val], ymin=np.min(self.data), ymax=np.max(self.data))
        
    def sw_detection(self):
        eeg_index = [i for i, x in enumerate(self.chtypes) if x == "eeg"]
        if self.hypno_with_art != []:
            hypno = self.hypno_with_art
        else:
            hypno = None
        self.sws = yasa.sw_detect(self.data[:,eeg_index].T, self.sfreq, ch_names = np.asarray(self.chans)[eeg_index], 
                                  hypno = hypno, include=(2,3), freq_sw=(0.5, 1.5), dur_neg=(0.3, 1.5), 
                                  dur_pos=(0.1, 1), amp_neg=(35, 300), amp_pos=(10, 200), amp_ptp=(75, 400), 
                                  coupling=False, freq_sp=None, remove_outliers=True)
        
    def sp_detection(self):
        eeg_index = [i for i, x in enumerate(self.chtypes) if x == "eeg"]
        if self.hypno_with_art != []:
            hypno = self.hypno_with_art
        else:
            hypno = None
        self.spindles = yasa.spindles_detect(self.data[:,eeg_index].T, sf=self.sfreq, ch_names=np.asarray(self.chans)[eeg_index], 
                                             hypno=hypno, include=(2, 3), freq_sp=(9, 16), freq_broad=(1, 30),
                                             duration=(0.5, 2), min_distance=500,
                                             thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5},
                                             multi_only=False, remove_outliers=False, verbose=False)
        
#%%
         
def _pre_process_sleep_data(files, low_density = False, reference='mastoids', validation=None, stageing=False, line_noise_removal='spectrum_fit'):
    """

    Parameters
    ----------
    files : TYPE
        DESCRIPTION.
    reference : TYPE, {'mastoids', 'surface laplacian', 'common average', or None}
        DESCRIPTION. The default is 'mastoids'.
    validation : TYPE, {None, 'auditory', 'classifier'}
        DESCRIPTION. The default is None.
    stageing : TYPE, optional
        DESCRIPTION. The default is False.
    line_noise_removal : TYPE, {'spectrum_fit' or 'dss'}
        DESCRIPTION. The default is 'spectrum_fit'.
    Returns
    -------
    Data : TYPE
        DESCRIPTION.

    """
    # parse data
    stream_dict = load_xdf(files) 
    # select data and marker streams
    try:
        rec_type = 'eeg_replay'
        stream_dict[rec_type]
    except:
        rec_type = 'eego'
        stream_dict[rec_type]
        
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
    if 'Cpz' in ch_names:
        # ch_types[ch_names.index('Cpz')] = 'misc'
        data = np.delete(data, ch_names.index('Cpz'), axis=-1)
        ch_types.pop(ch_names.index('Cpz'))
        ch_names.remove('Cpz')
    if 'OZ' in ch_names:
        ch_names[ch_names.index('OZ')] = 'Oz'
    if 'EOG' in ch_names:
        ch_types[ch_names.index('EOG')] = 'misc'
    # chans of interest for stageing
    if stageing:
        EEG_index = np.r_[ch_names.index('F3'), ch_names.index('Fz'), ch_names.index('F4'), 
                          ch_names.index('C3'), ch_names.index('Cz'), ch_names.index('C4'), 
                          ch_names.index('P3'), ch_names.index('P4'), ch_names.index('O1'), 
                          ch_names.index('O2'), ch_names.index('M1'), ch_names.index('M2')]
    else:
        EEG_index = [i for i, x in enumerate(ch_types) if x == "eeg"]
    # to reduce amount of channels in high density 
    if low_density:
        EEG_index = np.r_[ch_names.index('Fp1'), ch_names.index('Fpz'), ch_names.index('Fp2'), 
                          ch_names.index('F7'), ch_names.index('F3'), ch_names.index('Fz'), 
                          ch_names.index('F4'), ch_names.index('F8'), ch_names.index('Cz'), 
                          ch_names.index('C3'), ch_names.index('C4'), ch_names.index('T7'), 
                          ch_names.index('T8'), ch_names.index('P3'), ch_names.index('Pz'), 
                          ch_names.index('P4'), ch_names.index('P7'), ch_names.index('P8'),
                          ch_names.index('O1'), ch_names.index('Oz'), ch_names.index('O2'), 
                          ch_names.index('M1'), ch_names.index('M2')]          
    
    # other relevant indices
    mastoids_index = np.r_[ch_names.index('M1'), ch_names.index('M2')]

    EEG = data[:,EEG_index].astype(np.float64)
    # re-reference EEG channels to average of mastoids/common average/surface laplacian
    if reference=='surface laplacian':
        mne_info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)
        raw = mne.io.RawArray(data.T, mne_info)
        raw.set_montage(mne.channels.make_standard_montage('standard_1005'))
        # ignore non-EEG channels! 
        raw.pick_types(eeg=True) 
        EEG = mne.preprocessing.compute_current_source_density(raw).get_data().T
    elif reference!='surface laplacian' and validation!='auditory':
        if stageing==False and reference=='mastoids':
            ref_data = EEG[:,mastoids_index][..., :].mean(-1, keepdims=True)
        elif stageing==True and reference=='mastoids':
            ref_data = EEG[:,-2:-1][..., :].mean(-1, keepdims=True)
        elif reference=='common average':
            ref_data = EEG[..., :].mean(-1, keepdims=True)
        EEG -= ref_data 
    elif reference==None or validation=='auditory':
        EEG = EEG
            
    # filter data
    EEG = mne.filter.filter_data(EEG.T, sfreq=sf, l_freq=0.3, h_freq=35, verbose=0).T
    EOG_L = mne.filter.filter_data(data[:,[ch_names.index('EOG_L')]].astype(np.float64).T, sf, l_freq=0.3, h_freq=35, verbose=0).T
    EOG_R = mne.filter.filter_data(data[:,[ch_names.index('EOG_R')]].astype(np.float64).T, sf, l_freq=0.3, h_freq=35, verbose=0).T
    EMG_L = mne.filter.filter_data(data[:,[ch_names.index('EMG_L')]].astype(np.float64).T, sf, l_freq=10, h_freq=100, verbose=0).T
    EMG_R = mne.filter.filter_data(data[:,[ch_names.index('EMG_R')]].astype(np.float64).T, sf, l_freq=10, h_freq=100, verbose=0).T
    
    if 'bipECG' in ch_names:
        ECG = mne.filter.filter_data(data[:,[ch_names.index('bipECG')]].astype(np.float64).T, sf, l_freq=0.3, h_freq=70, verbose=0).T
        if reference=='surface laplacian':
            data = np.concatenate([EEG*1e3, EOG_L*1e6, EOG_R*1e6, EMG_L*1e6, EMG_R*1e6, ECG*1e6], axis=-1)
        else:
            data = np.concatenate([EEG, EOG_L, EOG_R, EMG_L, EMG_R, ECG], axis=-1)*1e6
    else:
        if reference=='surface laplacian':
            data = np.concatenate([EEG*1e3, EOG_L*1e6, EOG_R*1e6, EMG_L*1e6, EMG_R*1e6], axis=-1)
        else:
            data = np.concatenate([EEG, EOG_L, EOG_R, EMG_L, EMG_R], axis=-1)*1e6
        
    if validation is None:
        if line_noise_removal == 'dss':
            data = np.concatenate([dss.dss_line(data[:,i], fline=50, sfreq=sf, nfft=4*sf)[0] for i in range(min(np.shape(data)))], axis=-1)
            data = np.concatenate([dss.dss_line(data[:,i], fline=100, sfreq=sf, nfft=4*sf)[0] for i in range(min(np.shape(data)))], axis=-1)
            data = np.concatenate([dss.dss_line(data[:,i], fline=150, sfreq=sf, nfft=4*sf)[0] for i in range(min(np.shape(data)))], axis=-1)
            data = np.concatenate([dss.dss_line(data[:,i], fline=200, sfreq=sf, nfft=4*sf)[0] for i in range(min(np.shape(data)))], axis=-1)
        else:
            data = mne.filter.notch_filter(data[:,:].T, Fs=sf, method='spectrum_fit', freqs=np.arange(50,50*4+1,50)).T
       
    # edit channel names and types based on new selection
    if low_density==True or stageing==True:
        new_chans = list(np.asarray(ch_names)[EEG_index]) + [ch_names[i] for i in [j for j, x in enumerate(ch_types) if x == "eog" or x=="emg" or x=="ecg"]]
        new_chtypes = list(np.asarray(ch_types)[EEG_index]) + [ch_types[i] for i in [j for j, x in enumerate(ch_types) if x == "eog" or x=="emg" or x=="ecg"]]
    else:
        new_chans = [ch_names[i] for i in [j for j, x in enumerate(ch_types) if x == "eeg" or x == "eog" or x=="emg" or x=="ecg"]]
        new_chtypes = [ch_types[i] for i in [j for j, x in enumerate(ch_types) if x == "eeg" or x == "eog" or x=="emg" or x=="ecg"]]     
    
    # create data object
    Data = Data_Struct(data, new_chans, new_chtypes, eego_times, pinknoise_timestamps, classifier_predict, classifier_timestamps, sf)
    
    return Data

def save_preprocess_sleep_data(*args, path): 
    # iterate over args
    var = [arg for arg in args]
    # save as pickle file
    pickle.dump(var, open(path, "wb"))  
      
def preprocess_sleep_data(path, low_density = False, save=True, reference='mastoids', validation=None, stageing=False):
    files_list = [os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files]
    for i, files in enumerate(files_list):
        # select subject ID + cond identifier
        subjID_cond = files.split('/')[-2]
        number = int(files.split('/')[-1].split('.')[0][-1])
        if 'Experimental' in files:
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental/'
            if validation == 'classifier':
                if number > 1:
                    new_path='/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_classifier_validation/' + subjID_cond + '_preproc_data_cv_' + str(number) + '.p'
                else:
                    new_path='/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_classifier_validation/' + subjID_cond + '_preproc_data_cv.p' 
        elif 'adaption_calibration' in files:
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Calibration/'
        elif 'Calibration' in files:
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Calibration/'
        elif 'Adaption' in files:
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Adaption/'
            if validation == 'classifier':
                if number > 1:
                    new_path='/media/administrator/data/Study_1_data/Pre-processed_data/Adaption_classifier_validation/' + subjID_cond + '_preproc_data_cv_' + str(number) + '.p'
                else:
                    new_path='/media/administrator/data/Study_1_data/Pre-processed_data/Adaption_classifier_validation/' + subjID_cond + '_preproc_data_cv.p'                
        else:
            raise NameError('Please verify that the file name contains an appropriate recording type! ')
        # check if file exists
        if validation == 'classifier':
            new_path = new_path
        elif validation is None:
            new_path = save_path + subjID_cond + '_preproc_data.p'
        elif validation == 'auditory':
            new_path ='/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/' + subjID_cond + '_preproc_data_av.p'
        if not os.path.exists(new_path):
            # check if a second recording file exists, then combine after preprocessing
            if number == 1:
                print(f'Pre-processing the dataset for subject, condition, and recording #: {subjID_cond + "_" + str(number)} !')
                Data = _pre_process_sleep_data(files, low_density=low_density, reference=reference, validation=validation, stageing=stageing)
            elif number > 1:
                if validation == None:
                    new_path = save_path + subjID_cond + '_preproc_data_' + files.split('/')[-1].split('.')[0][-1] + '.p'
                print(f'Pre-processing an additional file for the following dataset: {subjID_cond} which will be contained in the following path: {new_path} !')
                Data = _pre_process_sleep_data(files, low_density=low_density, reference=reference, validation=validation, stageing=stageing)
            if save:
                save_preprocess_sleep_data(Data, path=new_path)
            else:
                return Data
                        
        else:
            logging.warning(f'The requested dataset for subject and condition: {subjID_cond} has already been pre-processed!')
            continue

def load_preprocessed_data(file):
    # load as pickle file
    var = pickle.load(open(file,"rb")) 
    return var

def to_do_combine_recordings():
    combined = []
    return combined

     
#%%

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
def plot_multitaper_spectrogram(data, sf, ch_names, hypno=None, coi ='F3', fmin=0.5, fmax=30):
    if hypno:
        plot = yasa.plot_spectrogram(data[:,[ch_names.index(coi)]], sf, hypno, fmin, fmax, trimperc=5, cmap='Spectral_r');
    else:
        plot = yasa.plot_spectrogram(data[:,[ch_names.index(coi)]], sf,  fmin, fmax, trimperc=5, cmap='Spectral_r');
    return plot;

"""
TO-DO: test the below out...
"""
def sleep_statistics_reports(path, Data):
    subj_night = path ### + Data str info 
    out = open(path + '/Sleep_stats/' + subj_night + '.txt', "w")
    out.write(f'Sleep statistics for: {yasa.sleep_statistics(base_hypno, 1/30)}' + '\n')
    base_hypno = unravel_hypnogram_visbrain(path + subj_night + '_hypno_' + base_rater + '.txt')
    # close txt file
    out.close()
    print(f'Generated sleep statistics report for {subj_night}.')
    # generate hypnogram spectrograms
    plot_multitaper_spectrogram(Data.data, Data.sfreq, hypno=True, coi='Cz', fmin=0.5, fmax=30)


#%%    
def compare_hypnograms(subj_night : str, group : int, base_rater : str, comp_rater : str, save_hypno=False):
    path = '/media/administrator/data/Study_1_data/Hypnograms/Scoring_comparison/' 

    # open a (new) file to write
    out = open(path + 'Stat_reports/' + subj_night + '_report.txt', "w")
    
    # unravel hypnograms and compare overall agreement 
    base_hypno = unravel_hypnogram_visbrain(path + 'Group' + str(group) + '/' + subj_night + '_hypno_' + base_rater + '.txt')
    comp_hypno = unravel_hypnogram_visbrain(path + 'Group' + str(group) + '/' + subj_night + '_hypno_' + comp_rater + '.txt')
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
    disagreement = ((np.where(base_hypno!=comp_hypno)[0])*30/60)
    out.write(f'{base_rater} and {comp_rater} disagree on the epochs at the given minute intervals: \n {disagreement}' + '\n')
 
   
    # Compute confusion matrix
    event_id ={'Wake':0,'Stage 1':1,'Stage 2':2,'Stage 3':3,'REM':4}
    target_names=event_id.keys()
    cm = confusion_matrix(base_hypno, comp_hypno, normalize='true')
    np.set_printoptions(precision=1)
    out.write(f'Confusion matrix: \n {cm}' + '\n')
    plt.figure()
    plot_confusion_matrix(cm=cm, target_names=target_names, path=None)
    plt.savefig(path + 'Stat_reports/' + 'Normalized_confusion_matrix_' + subj_night)
    
    # close txt file
    out.close()
    
    # save 'flattened' hypnograms
    if save_hypno:
        np.savetxt(path + 'Flattened/' + subj_night + '_' + base_rater + '_hypno_flattened.txt', base_hypno, fmt='%d')
        np.savetxt(path + 'Flattened/' + subj_night + '_' + comp_rater + '_hypno_flattened.txt', comp_hypno, fmt='%d')

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
        path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_cleaned/'
        hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental_cleaned/'
    target_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded'
    target_hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental_unblinded'
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
def check_file_exists(target_file_path):
    if not os.path.exists(target_file_path):
        print(f"Pre-processing the following dataset: {target_file_path.split('/')[-1]} ! ")
        return False
    else:
        logging.warning(f"The requested dataset: {target_file_path.split('/')[-1]} has already been pre-processed! ")
        return True

#%%
def check_match_data_hypno_elements(data_path, hypno_path, hypno_path2):
    data_elem = sorted(os.listdir(data_path))
    try:
        hypno_elem = sorted(os.listdir(hypno_path)) + sorted(os.listdir(hypno_path2))
    except:
        hypno_elem = sorted(os.listdir(hypno_path))
    all_elem = np.asarray([(x,y) for x in data_elem for y in hypno_elem])
    ################### -- WORKS! -- ###################  
    exist = [elem[0].split('_')[0] + '_' + elem[0].split('_')[1] + '_' + elem[0].split('_')[-1].split('.')[0] == 
             elem[1].split('_')[0] + '_' + elem[1].split('_')[1] + '_' + elem[1].split('_')[-1].split('.')[0]
             if str(2) in elem[0].split("_")[-1] or str(2) in elem[1].split("_")[-1] else 
             elem[0].split('_')[0] + '_' + elem[0].split('_')[1]  == 
             elem[1].split('_')[0] + '_' + elem[1].split('_')[1]  
             for elem in all_elem]
    exist_exp_files, exist_hyp_files = all_elem[exist][:,0], all_elem[exist][:,1]
    
    if len(exist_exp_files) > 0 and len(exist_hyp_files) > 0:
        return exist_exp_files, exist_hyp_files
    else:
        raise TypeError('No subject entries align! Please check whether the paths have any corresponding data and hypnogram files')
 
#%%
def art_detect(epochs, raw=False): 
    if raw:
        _, epochs = yasa.sliding_window(epochs.get_data('eeg')*1e6, window=2, 
                                       sf = epochs.info['sfreq'])
    else:
        epochs = epochs.get_data('eeg')*1e6
    # Calculate the covariance matrices
    covmats = Covariances().fit_transform(epochs)
    # Shrink the covariance matrix (ensure positive semi-definite)
    covmats = Shrinkage().fit_transform(covmats)
    # Define Potato instance: 0 = clean, 1 = art
    potato = Potato(metric='riemann', threshold=3, pos_label=0,
                    neg_label=1, n_iter_max=100)
     
    # Apply Potato algorithm, extract z-scores and labels
    zs = potato.fit_transform(covmats)
    art = potato.predict(covmats).astype(int)
    
    return art.astype(bool)
    
#%%
def label_artifacts(path, hypno_path, save=True): 
    exist_exp_files, exist_hyp_files = check_match_data_hypno_elements(path, hypno_path)
    for i, (data_files, hypno_files) in enumerate(zip(exist_exp_files, exist_hyp_files)):
        print(i, data_files, hypno_files)
        # check if file has been preprocessed already
        data_save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_cleaned/' + data_files.split('/')[-1].split('.')[0] + '_clean.p'
        hypno_save_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental_cleaned/' + hypno_files.split('/')[-1].split('.')[0] + '_clean.txt'
        if not check_file_exists(data_save_path) and not check_file_exists(hypno_save_path):
            # load pre-processed data
            Data = load_preprocessed_data(file=path + data_files)[0]
            
            # apply artifact detection for a selct number of core EEG channels
            eeg_index = np.r_[Data.chans.index('F3'), Data.chans.index('Fz'), Data.chans.index('F4'), 
                              Data.chans.index('C3'), Data.chans.index('Cz'), Data.chans.index('C4'), 
                              Data.chans.index('P3'), Data.chans.index('Pz'), Data.chans.index('P4')]
            
            # eliminate bad channels from covariance artifact detection approach
            _, epoched_data = yasa.sliding_window(Data.data.T, sf=Data.sfreq, window=30)
            info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)
            epochs = mne.EpochsArray(epoched_data/1e6, info, tmin = 0, baseline=(None)) 
            epochs.set_montage(mne.channels.make_standard_montage('standard_1005'))
        
            picks = mne.pick_types(epochs.info, eeg=True, stim=False, eog=False,
                                    include=[], exclude=[])
            from autoreject import Ransac
            ransac = Ransac(verbose='progressbar', picks=picks, n_jobs=-1)
            ransac.fit_transform(epochs)
            logging.warning(f'The RANSAC algorithm detected the following as bad channels: {ransac.bad_chs_}!')
            
            # detected and repaired channels
            bad_idx = np.asarray([np.r_[Data.chans.index(ransac.bad_chs_[i])][0] for i in range(len(ransac.bad_chs_))])
            good_idx = np.asarray([item for item in eeg_index if item not in bad_idx])
            
            # apply artifact detection
            Data.hypno_with_art, Data.hypno = artifact_detect_hypno(Data.data[:,good_idx], hypno_path = hypno_path + hypno_files, 
                                                                    downsample=False, sf=Data.sfreq, method='covar', window=2)  
            # save data
            if save:
                pickle.dump(Data, open(data_save_path,"wb"))
                pickle.dump(Data.hypno_with_art, open(hypno_save_path, "wb"))
            else:
                return Data
        else:
            continue

#%%
def SW_summary(Data, subj_cond, sw_detection_method='abs', sp_coupling=False, sp_freq=(9,16), save=True): 
    """
    TO:DO - coupling only implemented for yasa sw detection methods

    """
    # channels to perform sw detection
    eeg_index = [i for i, x in enumerate(Data.chtypes) if x == "eeg"]
    eeg_index_names = [Data.chans[i] for i in [j for j, x in enumerate(Data.chtypes) if x == "eeg"]]
    
    # SW detection method
    if sw_detection_method!='ngo':
        if sp_coupling:
            sp_freq, coupling = sp_freq, True
        else:
            sp_freq, coupling = None, False
        # SW detection approach #1 - see Massimini et al., 2004 for more detail
        if sw_detection_method=='abs':
            sw = yasa.sw_detect(Data.data[:,eeg_index[0::3]].T, Data.sfreq, ch_names = eeg_index_names[0::3], 
                                hypno = Data.hypno_with_art, include=(2,3), freq_sw=(0.5, 2.0), dur_neg=(0.3, 1.5), 
                                dur_pos=(0.1, 1), amp_neg=(35, 300), amp_pos=(10, 200), amp_ptp=(75, 400), 
                                coupling=coupling, freq_sp=sp_freq, remove_outliers=True)
            sw_summary1 = sw.summary().round(2)
            sw2 = yasa.sw_detect(Data.data[:,eeg_index[1::3]].T, Data.sfreq, ch_names = eeg_index_names[1::3], 
                                hypno = Data.hypno_with_art, include=(2,3), freq_sw=(0.5, 2.0), dur_neg=(0.3, 1.5), 
                                dur_pos=(0.1, 1), amp_neg=(35, 300), amp_pos=(10, 200), amp_ptp=(75, 400), 
                                coupling=coupling, freq_sp=sp_freq, remove_outliers=True)
            sw_summary2 = sw2.summary().round(2)
            sw_summary2['IdxChannel'] += len(np.unique(sw.summary()['IdxChannel']))
            
            sw3 = yasa.sw_detect(Data.data[:,eeg_index[2::3]].T, Data.sfreq, ch_names = eeg_index_names[2::3], 
                                hypno = Data.hypno_with_art, include=(2,3), freq_sw=(0.5, 2.0), dur_neg=(0.3, 1.5), 
                                dur_pos=(0.1, 1), amp_neg=(35, 300), amp_pos=(10, 200), amp_ptp=(75, 400), 
                                coupling=coupling, freq_sp=sp_freq, remove_outliers=True)
            sw_summary3 = sw3.summary().round(2)
            sw_summary3['IdxChannel'] += max(sw_summary2['IdxChannel']) + 1
            
            Data.sw_summary = pd.concat([sw_summary1, sw_summary2, sw_summary3], axis=0)
            sw_summary_stats_chan_stage = pd.concat([sw.summary(grp_chan=True, grp_stage=True), sw2.summary(grp_chan=True, grp_stage=True), sw2.summary(grp_chan=True, grp_stage=True)])
            sw_summary_stats_chan = pd.concat([sw.summary(grp_chan=True), sw2.summary(grp_chan=True), sw2.summary(grp_chan=True)])
            sw_summary_stats_stage = pd.concat([sw.summary(grp_stage=True), sw2.summary(grp_stage=True), sw2.summary(grp_stage=True)])
            
        # SW detection approach #2 - see Muehlroth & Werkle-Bergner, 2020 for more detail 
        elif sw_detection_method=='rel_percentile':
            thresh = np.percentile(np.abs(Data.data[:,eeg_index][Data.hypno_with_art==3]), 75)
            print('75th percentile threshold: %.2f uV' % thresh)
            sw = yasa.sw_detect(Data.data[:,eeg_index].T, Data.sfreq, ch_names = eeg_index_names, 
                                hypno = Data.hypno_with_art, include=(2,3), freq_sw=(0.5, 2.0),
                                amp_neg=(None, None), amp_pos=(None, None), amp_ptp=(thresh, np.inf), 
                                coupling=coupling, freq_sp=sp_freq, remove_outliers=True)
            Data.sw_summary_rel_percentile = sw.summary().round(2)
        
        # SW detection approach #3 - see Helfrich et al., 2018 for more detail
        elif sw_detection_method=='rel_zscore': 
            data_zscored = zscore(Data.data[:,eeg_index].T)
            # Detect all events with a relative peak-to-peak amplitude between 3 to 10 z-scores, 
            # and positive/negative peaks amplitude > 1 standard deviations
            sw = yasa.sw_detect(data_zscored, Data.sfreq, ch_names = eeg_index_names, 
                                hypno = Data.hypno_with_art, include=(2,3), freq_sw=(0.5, 2.0),
                                amp_neg=(1, None), amp_pos=(1, None), amp_ptp=(3, 10), 
                                coupling=coupling, freq_sp=sp_freq, remove_outliers=True)
            Data.sw_summary_rel_zscore = sw.summary().round(2)

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
        # if sw_detection_method != 'ngo':
        #     """
            
        #     To-do: correct below as there are now 2 summaries  
            
        #     """
        #     Data.sw_summary = sw.summary().round(2)
        pickle.dump(Data.sw_summary, open(f'/media/administrator/data/Study_1_data/Statistics/SW_summary/' + subj_cond + '_sw_summary_' + sw_detection_method + '.p',"wb"))
        pickle.dump(sw_summary_stats_chan_stage, open(f'/media/administrator/data/Study_1_data/Statistics/SW_summary/' + subj_cond + '_sw_summary_stats_' + sw_detection_method + '_chan_stage.p',"wb"))
        pickle.dump(sw_summary_stats_chan, open(f'/media/administrator/data/Study_1_data/Statistics/SW_summary/' + subj_cond + '_sw_summary_stats_' + sw_detection_method + '_chan.p',"wb"))
        pickle.dump(sw_summary_stats_stage, open(f'/media/administrator/data/Study_1_data/Statistics/SW_summary/' + subj_cond + '_sw_summary_stats_' + sw_detection_method + '_stage.p',"wb"))
    else:
        if sw_detection_method != 'ngo':
            Data.sw_summary = sw.summary().round(2)
    
    return Data.sw_summary

#%%
def Data_SW(path, hypno_path, save=True):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    hypno_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(hypno_path) for i in files])
    for i, (data_files, hypno_files) in enumerate(zip(files_list, hypno_list)):
        data_save_path = '/media/administrator/data/Study_1_data/Statistics/SW_summary/' + data_files.split('/')[-1].split('.')[0] + '_sw_summary_abs.p'
        if not check_file_exists(data_save_path):
            # load pre-processed data
            Data = load_preprocessed_data(file=data_files)
          
            """
            WARNING: Please note that the following is only saved into sw summary file and 
                     not into cleaned data stuct.
            
            """
            # write in SW summary info for different approaches
            Data.sw_summary.abs = SW_summary(Data, subj_cond=data_files.split('/')[-1].split('.')[0], 
                                             sw_detection_method='abs', sp_coupling=True, sp_freq=(9,16), save=True)
            # Data.sw_summary.rel_zscore = SW_summary(Data, subj_cond=data_files.split('/')[-1].split('.')[0],
            #                                         sw_detection_method='rel_zscore', sp_coupling=True, sp_freq=(9,16), save=True)
            # Data.sw_summary.rel_percentile = SW_summary(Data, subj_cond=data_files.split('/')[-1].split('.')[0],
            #                                             sw_detection_method='rel_percentile', sp_coupling=True, sp_freq=(9,16), save=True)
            # Data.sw_summary.ngo = SW_summary(Data, subj_cond=data_files.split('/')[-1].split('.')[0],
            #                                  sw_detection_method='ngo', sp_coupling=False, save=True)
            #print(f'The slow wave detection method calculated {Data.sw_summary}')
            # if save:
            #     pickle.dump(Data, open(f''/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded/' + subj_cond + '.p',"wb"))
            # else:
            #     return Data
        else:
            continue

#%% 
def plot_ndPAC(df_sw, epochs):
    fig, axes = plt.subplots()
    # Create a topomap for ndPAC values
    ndPAC = df_sw.groupby(['IdxChannel']).agg(np.nanmean)['ndPAC'].to_numpy()
    im, cn = mne.viz.plot_topomap(data = ndPAC, names = df_sw['Channel'].unique()[0:-1], 
                                  pos = epochs.info, cmap='Spectral_r', show_names = True, 
                                  contours=0, axes=axes, show=False,                               
                                  vmin=np.percentile(ndPAC, 5), vmax=np.percentile(ndPAC, 95));
    
    # add color bar
    mne.viz.topomap._add_colorbar(axes, im, cmap = 'Spectral_r', side='right', pad=0.05, 
                                  title=None, format=None, size='5%')
    
    # Set the plot title
    axes.set_title('ndPAC values')
    
    # tighten layout
    plt.tight_layout()
    
    return fig, axes

#path = '/media/administrator/data/Study_1_data/Statistics/SW_summary/'
def SW_spindle_PAC(path):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    Up, Down, Sham = [], [], []
    for i, data_files in enumerate(files_list):
        if data_files.endswith('_abs.p'):
            if data_files.split('/')[-1].split('_')[1] == 'up':
                PAC_df_up = load_preprocessed_data(file=data_files)
                Up.append(pd.concat([PAC_df_up['PhaseAtSigmaPeak'], PAC_df_up['ndPAC']], axis=1))
            elif data_files.split('/')[-1].split('_')[1] == 'down':
                PAC_df_down = load_preprocessed_data(file=data_files)
                Down.append(pd.concat([PAC_df_down['PhaseAtSigmaPeak'], PAC_df_down['ndPAC']], axis=1))
            elif data_files.split('/')[-1].split('_')[1] == 'sham':
                PAC_df_sham = load_preprocessed_data(file=data_files)
                Sham.append(pd.concat([PAC_df_sham['PhaseAtSigmaPeak'], PAC_df_sham['ndPAC']], axis=1))
    Up_PAC = pd.concat(Up)
    Down_PAC = pd.concat(Down)
    Sham_PAC = pd.concat(Sham)
    
    # for iterables in ([Up_PAC, Down_PAC, Sham_PAC]):
    #     pg.plot_circmean(iterables['PhaseAtSigmaPeak'])
    #     print('Circular mean: %.3f rad' % pg.circ_mean(iterables['PhaseAtSigmaPeak']))
    #     print('Vector length: %.3f' % pg.circ_r(iterables['PhaseAtSigmaPeak']))
        
    pg.plot_circmean(Up_PAC['PhaseAtSigmaPeak'])
    print('Circular mean: %.3f rad' % pg.circ_mean(Up_PAC['PhaseAtSigmaPeak']))
    print('Vector length: %.3f' % pg.circ_r(Up_PAC['PhaseAtSigmaPeak']))
          
    pg.plot_circmean(Down_PAC['PhaseAtSigmaPeak'])
    print('Circular mean: %.3f rad' % pg.circ_mean(Down_PAC['PhaseAtSigmaPeak']))
    print('Vector length: %.3f' % pg.circ_r(Down_PAC['PhaseAtSigmaPeak']))
    
    pg.plot_circmean(Sham_PAC['PhaseAtSigmaPeak'])
    print('Circular mean: %.3f rad' % pg.circ_mean(Sham_PAC['PhaseAtSigmaPeak']))
    print('Vector length: %.3f' % pg.circ_r(Sham_PAC['PhaseAtSigmaPeak']))
    
    # Rayleigh test
    z, pval = pg.circ_rayleigh(Up_PAC['PhaseAtSigmaPeak'])
    print(round(z, 3), round(pval, 6))

    return 

#%%    
# path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded/'         
def SW_ERPs(path, filter_data=False, downsample=False, inspection=False):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    for i, data_files in enumerate(files_list):
        # load pre-processed data
        Data = load_preprocessed_data(file = data_files) 

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
            
        # load sw summaries 
        sw_summary_path = '/media/administrator/data/Study_1_data/Statistics/SW_summary/' + data_files.split('/')[-1].split('.')[0] + '_sw_summary_abs.p'
        Data.sw_summary = load_preprocessed_data(file=sw_summary_path)
        
        # update time series index to include SW occurences
        idx_ev = yasa.others._index_to_events(Data.sw_summary[[start, end]].to_numpy() * Data.sfreq)
        mask[:, idx_ev] = 1
        sw_highlight = Data.data[:,eeg_index].T * mask
        sw_highlight[sw_highlight == 0] = np.nan  
                    
        # time point synchronization
        Data.pinknoise_timestamps_sync = [Data.times[np.abs(Data.times - Data.pinknoise_times[ix]).argmin()] for ix in range(len(Data.pinknoise_times))]
        Data.pinknoise_timestamps_sync_adj = np.asarray(Data.pinknoise_timestamps_sync)*Data.sfreq
        first_bursts = Data.pinknoise_timestamps_sync_adj[::2]
        
        # sw times in bool, index, and raw times
        sw_times_bool = ~np.isnan(sw_highlight[eeg_index,:])
        sw_times_idx = np.where(~np.isnan(sw_highlight[eeg_index,:]))[0]
        sw_times_times = Data.times[np.argwhere(~np.isnan(sw_highlight[eeg_index,:])).ravel()]
        
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
            sync_sw_noise_idx = [np.where(sw_times_bool[Data.chans.index('C3'),:][center[idx,:]])[0] for idx, times in enumerate(center)]  
            
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
            
            # neg peaks of detected slow waves - only for C3!
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
            
            # create mne epochs object for all eeg channels
            eeg_chans = [Data.chans[i] for i in [j for j, x in enumerate(Data.chtypes) if x == "eeg"]]
            eeg_chtypes = [Data.chtypes[i] for i in [j for j, x in enumerate(Data.chtypes) if x == "eeg"]]
            info = mne.create_info(ch_names=eeg_chans, sfreq=512, ch_types=eeg_chtypes)
            epochs = mne.EpochsArray(Data.data[:, eeg_index][new_center].swapaxes(1,2)/1e6, info, tmin = -2, 
                                     baseline=(-2.0, 4.0), proj=False)
            
            
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

# path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/' 
def group_SW_ERPs(path, output='df'):
    """
    
    Warning: supports only singular channel implementation!
    
    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    output : TYPE, optional
        DESCRIPTION. The default is 'df'. For mne epoch output, select: 'mne' and for mixed models 
        statistical implementation please select 'mixed_model'

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    Up, Down, Sham = [], [], []
    Up_sub, Down_sub, Sham_sub = [], [], []
    for i, data_files in enumerate(files_list):
        if data_files.endswith('up_epochs.p'):
            epochs = pickle.load(open(data_files,"rb"))
            Up.append(np.expand_dims(epochs.get_data()[:,epochs.ch_names.index('C3'),:],1)*1e6)
            Up_sub.append(Up[-1].shape[0]*[data_files.split('/')[-1].split('_')[0]]*epochs.get_data().shape[-1])
        elif data_files.endswith('down_epochs.p'):
            epochs = pickle.load(open(data_files,"rb"))
            Down.append(np.expand_dims(epochs.get_data()[:,epochs.ch_names.index('C3'),:],1)*1e6)
            Down_sub.append(Down[-1].shape[0]*[data_files.split('/')[-1].split('_')[0]]*epochs.get_data().shape[-1])
        elif data_files.endswith('sham_epochs.p'):
            epochs = pickle.load(open(data_files,"rb"))
            Sham.append(np.expand_dims(epochs.get_data()[:,epochs.ch_names.index('C3'),:],1)*1e6)
            Sham_sub.append(Sham[-1].shape[0]*[data_files.split('/')[-1].split('_')[0]]*epochs.get_data().shape[-1])
            
    #info = mne.create_info(ch_names=epochs.ch_names, sfreq=512, ch_types=np.asarray(len(chans)*('eeg',)))
    info = mne.create_info(ch_names=['C3'], sfreq=512, ch_types=['eeg'])
    epochs_up = mne.EpochsArray(np.concatenate(Up)/1e6, info, tmin = -2, baseline=None)
    epochs_down = mne.EpochsArray(np.concatenate(Down)/1e6, info, tmin = -2, baseline=None)
    epochs_sham = mne.EpochsArray(np.concatenate(Sham)/1e6, info, tmin = -2, baseline=None)
      
    df = pd.concat([epochs_down.to_data_frame(picks='C3'), epochs_up.to_data_frame(picks='C3'),
                    epochs_sham.to_data_frame(picks='C3')], keys=['Down', 'Up', 'Sham'],
                    names=['Condition'], axis=0)
    
    df['time'] = df['time']/1000
    df['condition'] = np.concatenate([np.concatenate(Down_sub), np.concatenate(Up_sub), np.concatenate(Sham_sub)])
    df = df.rename(columns={'C3':'Amplitude (uV)','time': 'Time (s)', 'condition': 'Subject', 'epoch':'Epoch'})
    
    if output=='df':
        return df
    elif output=='mne':
        return epochs_up, epochs_down, epochs_sham
    elif output=='mixed_model':
        return None
    
def df_to_epochs(df, condition='Down'):
    epochs = [df.loc[condition]['Amplitude (uV)'].to_numpy()[df.loc[condition]['Epoch'].to_numpy()==i] for i in range(len(df.loc[condition]['Epoch'].unique()))]
    return np.vstack(epochs)

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

# df = group_SW_ERPs(path)
# fig = plot_SW_ERPs(df)

#%%
## Permutation Cluster Testing

def permutation_cluster_test_mne(epochs : list):
    threshold = 6.0
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test(epochs, n_permutations=1024, threshold=threshold, 
                                 tail=1, n_jobs=1, out_type='mask')
    
    return T_obs, clusters, cluster_p_values, H0

# T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test_mne([epochs_up.get_data()*1e6, 
#                                                                       epochs_down.get_data()*1e6, 
#                                                                       epochs_sham.get_data()*1e6])

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

# permutation_cluster_test_plot([epochs_up.get_data()*1e6, epochs_down.get_data()*1e6, epochs_sham.get_data()*1e6], 
#                               epochs_up.times, T_obs, clusters, cluster_p_values, H0)

#%%

def spectral_conn(epochs, ch_names, sf, tmin=-4, tmax=4, foi = (0.5, 2), method='ciplv'):
    indices = (np.ones(len(ch_names))*ch_names.index('C3'),           # row indices
               np.arange(0,len(ch_names),1))                          # col indices
    indices = indices[0].astype(int), indices[1]
    if method == 'psi':
        from mne_connectivity import phase_slope_index
        conn = phase_slope_index(epochs, indices=indices,
                                 sfreq=sf, mode='multitaper', fmin=foi[0], fmax=foi[1],
                                 verbose=0, tmin=tmin, tmax=tmax)
    else:
        conn = spectral_connectivity(epochs, method=method, indices=indices,
                                     sfreq=sf, mode='multitaper', fmin=foi[0], fmax=foi[1],
                                     fskip=0, faverage=True,
                                     verbose=0, tmin=tmin, tmax=tmax)
        
    return conn.freqs, conn._data.squeeze()

#%%
## plot spectrogram

# peak = PeakLockedTF(np.squeeze(epochs_up.get_data())*1e6, sf=512, cue=0., times=epochs_up.times,
#                     f_pha=[0.5, 30],f_amp=(0.5, 30, 0.3, 0.3), n_jobs=16, verbose=True)

# plt.figure(figsize=(8, 8))
# ax_1, ax_2 = peak.plot(zscore=True, baseline=None, cmap='Spectral_r',
#                             vmin=-1, vmax=1)
# # add_motor_condition(135, color='black', ax=ax_1)
# plt.tight_layout()
# plt.show()

# # altered plot
# plt.figure(figsize=(8, 8))
# plt.title('Up-state targeted spectral modulation')
# ax_1, ax_2 = peak.plot_altered(zscore=True, baseline=None, cmap='Spectral_r',
#                             vmin=-1, vmax=2)
# plt.title('Up-state targeted spectral modulation')
# # add_motor_condition(135, color='black', ax=ax_1)
# plt.tight_layout()
# plt.show()

# mean_epochs = np.squeeze(epochs_up.get_data())[0:500,:].mean(0)*1e6
# min_peak_time = np.where(mean_epochs == np.min(mean_epochs))[0][0]
# max_peak_time = np.where(mean_epochs == np.max(mean_epochs))[0][0]

#%%
       
def peak2peak_SW_duration(data, hypno_path, ch_names, chan='C3', sf=512, data_len=210):
    # epoch data
    _, epochs = yasa.sliding_window(data[:,ch_names.index(chan)].T, sf, window=30)
    # load hypnogram
    hypno = unravel_hypnogram_visbrain(hypno_path)
    # unsampled hypnogram
    hypno = yasa.hypno_upsample_to_data(hypno=hypno[0:data_len*2], sf_hypno=1/30, data=data[0:sf*60*data_len,:].T, sf_data=sf)
    # calculate sw dataframe for detected SWs
    sw = yasa.sw_detect(data[0:sf*60*data_len,ch_names.index(chan)].T, sf, ch_names = [chan], hypno = hypno, include=(2,3), freq_sw=(0.5, 3.0),
                   dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(35, 300), amp_pos=(10, 200), 
                   amp_ptp=(75, 400), remove_outliers=True)
    # print summary
    sw2 = sw.summary()
    print(f'Channel {chan} contains {len(sw2)} detected slow waves in the first {data_len} minutes of data.')
    
    # return mean peak to peak time duration
    return sw2['PosPeak'].mean() - sw2['NegPeak'].mean()


# p2p = peak2peak_SW_duration(Data.data, hypno_path, Data.chans)
  
           
#%%   
     
# # PSD-based signal quality check
# freqs, psd = welch(EEG[:,Data.chans.index('C3')].T, fs=512, nperseg=1024, average='median')
# plt.semilogy(freqs, psd.T)       
# # plt.ylim([0.5e-3, 1e5])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()
               
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
    
    
# data_narrow = mne.filter.filter_data(data.T, sfreq=512, l_freq=0.5, h_freq=4)

# filtparams_lp = butter(4, 4, fs = 128)
# data_narrow = filtfilt(*filtparams_lp, Data.data.T)
# crossings = thresholdcrossings(data_narrow[3,:], -35)
# # plt.plot(eego_times, data_narrow[2,:])
# # # plt.plot(eego_times, data[:,2])
# # plt.vlines(pinknoise_timestamps, ymin=-2000, ymax=2000, colors='r')
# # plt.vlines(data_narrow[2,:][crossings], ymin=-100, ymax=100, colors='o')
    
# # plt.vlines([eego_times, crossings], ymin=-2000, ymax=2000, colors='r')


# plt.plot(time, data_narrow[3,:])
# plt.vlines(time[crossings], ymin=-1000, ymax=1000)
# plt.vlines(pinknoise_timestamps - time_stamps[0], ymin=-750, ymax=750, colors='r')
