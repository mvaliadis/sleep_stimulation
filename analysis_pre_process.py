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
import pickle 
import pandas as pd
import pyxdf
import yasa 
import mne
import logging
import seaborn as sns
from scipy.signal import welch, butter, filtfilt
from scipy.stats import zscore
from meegkit import dss, star
from meegkit.utils import demean, normcol
from os import chdir as cd
from os import listdir
import os
cd('/home/administrator/sleep_stimulation-development')
from sleep_funs import bfr_butter_filt, process_raw_EDF_cfs, bandpower, unravel_hypnogram_visbrain, downsample_scaled, load_xdf, channel_parser, thresholdcrossings
sns.set(style='darkgrid', font_scale=1.2)

class Data_Struct:
    def __init__(self, data, chans, chtypes, times, pinknoise_times, classif_predict, classif_times, sfreq):
        self.data = data
        self.chans = chans
        self.chtypes = chtypes
        self.times = times #-times[0]
        #if 'calibration' not in files:
        self.pinknoise_times = pinknoise_times #-times[0]
        self.classif_times = classif_times  #-times[0]
        #else:
            # self.pinknoise_times = pinknoise_times
            # self.classif_times = classif_times
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

path = '/media/administrator/data/Study_1_data/Raw_data/Experimental/9PJZ8Z8F_3'
cd(path)

def preprocess_sleep_data(path):
    files_list = [os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files]
    for i, files in enumerate(files_list):
        # parse data
        stream_dict = load_xdf(files) 
        # select subject ID + cond identifier
        if 'Experimental' in files:
            subjID_cond = files[61:71]
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental/'
        elif 'adaption_calibration' in files:
            subjID_cond = files[60:89]
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Calibration/'
        elif 'Calibration' in files:
            subjID_cond = files[60:82]
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Calibration/'
        elif 'Adaption' in files:
            subjID_cond = files[57:74]
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Adaption/'
        else:
            raise NameError('Please verify that the file name contains an appropriate recording type! ')
            
        # select data and marker streams
        data = stream_dict['eego']['time_series']
        marker = stream_dict['reiz-marker']
        classifier_predict = [marker['time_series'][ix] for ix in range(len(marker['time_series'])) if 'rf' in marker['time_series'][ix][0]]
        classifier_timestamps = [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if 'rf' in marker['time_series'][ix][0]]
        pinknoise_timestamps = [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if marker['time_series'][ix][0] == 'pinknoise']
        eego_times = stream_dict['eego']['time_stamps']
        # load hdr
        info = stream_dict['eego']['info'] 
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
        #Data = (data, new_chans, new_chtypes, eego_times, pinknoise_timestamps, classifier_predict, classifier_timestamps, sf)
        
        # save as pickle file
        pickle.dump([data, new_chans, new_chtypes, eego_times, pinknoise_timestamps, classifier_predict, classifier_timestamps, sf], open(save_path + subjID_cond + '_preproc_data.p', "wb"))

    
#%%
 
def plot_multitaper_spectrogram(data, sf, ch_names, hypno=None, coi ='F3', fmin=0.5, fmax=30):
    if hypno:
        plot = yasa.plot_spectrogram(data[:,[ch_names.index(coi)]], sf, hypno, fmin, fmax, trimperc=5, cmap='Spectral_r');
    else:
        plot = yasa.plot_spectrogram(data[:,[ch_names.index(coi)]], sf,  fmin, fmax, trimperc=5, cmap='Spectral_r');
    return plot;

def artifact_detect_hypno(data, hypno_path, sf, downsample=False, method='covar', window=2): 
    
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
        art_std, zscores_std = yasa.art_detect(data.T, sf=sf, window=2, hypno=hypno_unsampled, include=(0, 1, 2, 3, 4), 
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
    
    # Plot the distribution of z-scores
    # if method=='covar':
    #     sns.distplot(zscores, label='covar')
    # elif method=='std':
    #     # take mean across channels
    #     sns.distplot(zscores_std.mean(-1), label='std')
    # elif method=='both':
    #     sns.distplot(zscores, label='covar')
    #     # take mean across channels
    #     sns.distplot(zscores_std.mean(-1), label='std')
    # plt.title('Histogram of z-scores')
    # plt.xlabel('Z-scores')
    # plt.ylabel('Density')
    # plt.axvline(3, color='r', label='Threshold')
    # plt.axvline(-3, color='r')
    # plt.legend(frameon=False);
    
    # threshold = 3
    # perc_expected_rejected = (1 - erf(threshold / np.sqrt(2))) * 100
    # print(f'{perc_expected_rejected:.2f}% of all epochs are expected to be rejected.')
    
    # Actual
    # print(f'{(art.sum() / art.size) * 100:.2f}% of all epochs were actually rejected.')
    
    # The resolution of art is 2 seconds, so its sampling frequency is 1/2 = 0.5 Hz)
    sf_art = 1 / window
        
    if method == 'covar':
        art_up = yasa.hypno_upsample_to_data(art, sf_art, data.T, sf)
        hypno_with_art = hypno_unsampled.copy()
        hypno_with_art[art_up] = -1
     
        return hypno_with_art, hypno_unsampled
    
    elif method == 'std':
        art_up_std = yasa.hypno_upsample_to_data(art_std, sf_art, data.T, sf)
        hypno_with_art_std = hypno_unsampled.copy()
        hypno_with_art_std[art_up_std] = -1
        
        return hypno_with_art_std, hypno_unsampled
    
    elif method == 'both':
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

#%%

path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental/9PJZ8Z8F_3_preproc_data.p'
hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/9PJZ8Z8F_3_hypno.txt'
def SW_analysis(path, hypno_path, filter_data=True, downsample=True, sw_detection_method='abs', save=True):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    hypno_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(hypno_path) for i in files])
    Dat = []
    for i, (data_files, hypno_files) in enumerate(zip(files_list, hypno_list)):
        pass 
        # load pre-processed data
        data, ch_names, ch_types, time_stamps, pinknoise_timestamps, classifier_predict, classifier_timestamps, sf = pickle.load(open(data_files,"rb"))
        # convert to object
        Data = Data_Struct(data, ch_names, ch_types, time_stamps, pinknoise_timestamps, classifier_predict, classifier_timestamps, sfreq=512)
        
        # apply artifact detection for EEG channels only!
        eeg_index = [i for i, x in enumerate(Data.chtypes) if x == "eeg"]
        Data.hypno_with_art, Data.hypno = artifact_detect_hypno(Data.data[:,eeg_index], hypno_path = hypno_files, downsample=False, sf=Data.sfreq, method='covar', window=2)   
        
        # filter data
        if filter_data:
            Data.fir_filter(0.5, 2)
            logging.warning('Filtering the data will result in the deletion of the non-EEG channels!')    
            
        # downsample data
        if downsample:
            Data.downsample(128)    
        
        Dat.append(Data)
        
        # # combine data sets
        # if data_files.endswith('down.p'):
        #     down = Dat.append(Data)
                
        # elif data_files.endswith('up.p'):
        #     up = Dat.append(Data)
            
        # elif data_files.endswith('sham.p'):
        #     sham = Dat.append(Data)
        
        # SW detection approach #1 - see Massimini et al., 2004 for more detail
        if sw_detection_method=='abs':
            sw = yasa.sw_detect(Data.data[:,eeg_index].T, Data.sfreq, ch_names = Data.chans[0:len(eeg_index)], hypno = Data.hypno, include=(2,3), freq_sw=(0.5, 2.0),
               dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(35, 300), amp_pos=(10, 200), amp_ptp=(75, 400), remove_outliers=True)
        
        # SW detection approach #2 - see Muehlroth & Werkle-Bergner, 2020 for more detail 
        if sw_detection_method=='rel_zscore':
            thresh = np.percentile(np.abs(Data.data[:,Data.chans.index('C3')][Data.hypno_with_art==3]), 75)
            print('75th percentile threshold: %.2f uV' % thresh)
            sw = yasa.sw_detect(Data.data[:,Data.chans.index('C3')].T, Data.sfreq, ch_names = ['C3'], hypno = Data.hypno, include=(2,3), freq_sw=(0.5, 2.0),
               amp_neg=(None, None), amp_pos=(None, None), amp_ptp=(thresh, np.inf), remove_outliers=True)
        
        # SW detection approach #3 - see Helfrich et al., 2018 for more detail
        if sw_detection_method=='rel_percentile': 
            data_zscored = zscore(Data.data[:,Data.chans.index('C3')])
            # Detect all events with a relative peak-to-peak amplitude between 3 to 10 z-scores, 
            # and positive/negative peaks amplitude > 1 standard deviations
            sw = yasa.sw_detect(data_zscored, sf = Data.sfreq, ch_names = ['C3'], hypno = Data.hypno, include=(2,3), freq_sw=(0.5, 2.0),
                           amp_neg=(1, None), amp_pos=(1, None), amp_ptp=(3, 10))
        
        # SW summary
        summary = sw.summary().round(2) 
        
        # get a mask of detected SW
        mask = sw.get_mask()
        sw_highlight = Data.data[:,Data.chans.index('C3')] * mask
        sw_highlight[sw_highlight == 0] = np.nan
        
        # plot SW mask
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
        
        # # plot average SW
        # plt.figure; sw.plot_average(center = 'NegPeak', time_before=1, time_after=1.5)
        
        if save:
            # save as pickle file
            pickle.dump(sw, open('C:/Users/neuro/sleep_stimulation/Statistics/' + data_files[-25:-15] + '_SW_summary.p',"wb"))
        else:
            return sw


#%%

Data.pinknoise_timestamps_sync = [Data.times[np.abs(Data.times - Data.pinknoise_times[ix]).argmin()] for ix in range(len(Data.pinknoise_times)) if sw_times]
Data.pinknoise_timestamps_sync_adj = np.asarray(Data.pinknoise_timestamps_sync)*Data.sfreq
first_bursts = Data.pinknoise_timestamps_sync_adj[::2] 


Data.sw_sync_C3 = [Data.times[np.abs(Data.times - Data.pinknoise_times[ix]).argmin()] for ix in range(len(Data.pinknoise_times))]
sw_times = Data.times[np.argwhere(~np.isnan(sw_highlight[Data.chans.index('C3'),:])).ravel()]


insp_ix = []
for i in range(len(first_bursts)):
    insp_indx = np.where(Data.times*Data.sfreq == first_bursts[i])[0][0]
    insp_ix.append(insp_indx) 

events = []
for i in range(len(insp_ix)):
    event = [insp_ix[i],1,1]
    events.append(event)
    
event_id = dict(pinknoise=1)
events = np.asarray(events)

center, _ = yasa.get_centered_indices(Data.data[:,Data.chans.index('C3')], np.asarray(insp_ix), 
                                      npts_before=Data.sfreq*2, npts_after=Data.sfreq*4)
C3_filtered_sync = C3_filtered[center]
C3_unfiltered_sync = C3_unfiltered[center]

filtparams_lp = signal.butter(4, 4, fs = Data.sfreq)
C3_bandpass_low = signal.filtfilt(*filtparams_lp, C3_unfiltered)
            
C3_bandpass_low_sync = C3_bandpass_low[center]

info = mne.create_info(ch_names=Data.chans, sfreq=512, ch_types=Data.chtypes)
raw = mne.io.RawArray(data.T/1e6, info)
raw.filter(0.5, 2)

epochs = mne.Epochs(raw, events, event_id, tmin = -2, tmax = 4, baseline= None, event_repeated = 'drop').get_data()*1e6
C3_mne = epochs[:,Data.chans.index('C3'),:]
C3_avg = C3_mne[...,:].mean(0, keepdims=False)

#%%        
# plot artifacts
avg_data = data[..., eeg_index].mean(-1, keepdims=True)
times = np.arange(0, len(data))/sf
plt.plot(times, data[:,0:7])
plt.vlines(np.where(hypno_with_art==-1)[0]/128, ymin=-1000, ymax=1000, colors='k', linestyles='dotted')
plt.vlines(np.where(hypno_with_art_std==-1)[0]/128, ymin=-750, ymax=750, colors='r',linestyles='dotted')

# Plot multitaper spectrogram with hypno (no artifact)
yasa.plot_spectrogram(data[:,1], sf, hypno_unsampled, fmax=30, cmap='Spectral_r');

# Plot new hypnogram and spectrogram on Fz (with artifact)
yasa.plot_spectrogram(data[:,0], sf=128, hypno_with_art, fmax=30, cmap='Spectral_r');

#%%

def peak2peak_SW_duration(data, hypno_path, ch_names, chan='C3', sf=512, data_len=210):
    # epoch data
    _, epochs = yasa.sliding_window(data[:,ch_names.index(chan)].T, sf, window=30)
    # load hypnogram
    hypno = unravel_hypnogram_visbrain(hypno_path, epochs)
    # unsampled hypnogram
    hypno = yasa.hypno_upsample_to_data(hypno=hypno[0:data_len*2], sf_hypno=1/30, data=data[0:sf*60*210,:].T, sf_data=sf)
    # calculate sw dataframe for detected SWs
    sw = sw_detect(data[0:sf*60*data_len,ch_names.index(chan)].T, sf, ch_names = [chan], hypno = hypno, include=(2,3), freq_sw=(0.5, 3.0),
                   dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(35, 300), amp_pos=(10, 200), 
                   amp_ptp=(75, 400), remove_outliers=True)
    # print summary
    sw2 = sw.summary()
    
    # return mean peak to peak time duration
    return sw2['PosPeak'].mean() - sw2['NegPeak'].mean()


#%%

## create mne info
info = mne.create_info(ch_names=ch_names, sfreq=512, ch_types=ch_types)
# # create mne object 
raw = mne.io.RawArray(data.T/1e6, info)
raw.set_eeg_reference(ref_channels=['M1','M2'])
 
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

time = time_stamps - time_stamps[0]
plt.plot(time, data_narrow[3,:])
plt.vlines(time[crossings], ymin=-1000, ymax=1000)
plt.vlines(pinknoise_timestamps - time_stamps[0], ymin=-750, ymax=750, colors='r')






