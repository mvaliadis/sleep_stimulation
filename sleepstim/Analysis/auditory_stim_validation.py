#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:41:47 2021

@author: administrator
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os 
import yasa
import mne
import pingouin as pg
from os import chdir as cd
from os import listdir
import logging 
from tqdm import tqdm
from scipy.signal import welch, butter, filtfilt, hilbert, detrend
from scipy.stats import zscore, circmean, circstd, circvar
from scipy.fftpack import next_fast_len
from scipy.linalg import eigh
import scipy.signal as signal
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
from matplotlib import cm
import emd
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, _pre_process_sleep_data, preprocess_sleep_data, 
                                                     compare_hypnograms, check_match_data_hypno_elements,
                                                     label_artifacts, load_preprocessed_data, Data_SW, spectral_conn, 
                                                     art_detect, plot_ndPAC)
from sleepstim.sleep_funs import (load_xdf, channel_parser, bfr_butter_filt, thresholdcrossings)
from sleepstim.Analysis.pac import unit_root_test, add_stimulus_onset, ERPAC, extract_pha_amp, SO_spindle_coupling, imf_analytical_transform
from sleepstim.Analysis.time_frequency import tfr_analysis
from sleepstim.Analysis.Resting_State.rs_preproc import (plot_psd, norm_wavelet_power, subject_cond_parser)
import tensorpac.methods as tpm
from sleepstim.Analysis.foof import oscillatory_plot_psd_map, periodic_fit
from fooof.plts.spectra import plot_spectrum
from sleepstim.Analysis.hilbert_huang import hilbert_huang_spectrum
from sleepstim.Analysis.utils import coincidence_matrix, find_nearest, plot_cm, get_mask, _coincidence
import neurokit2 as nk
from PCIst.PCIst import pci_st
import scipy
sns.set_theme(color_codes=True) 
mne.set_log_level("CRITICAL")
      
def find_cooccurring_rrpeaks(sws, spindles, lookaround=1.2):
    """Given a spindles detection summary dataframe, find slow-waves that co-occur with
    sleep spindles.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    spindles : :py:class:`pandas.DataFrame`
        Output dataframe of :py:meth:`yasa.SpindlesResults.summary`.
    lookaround : float
        Lookaround window, in seconds. The default is +/- 1.2 seconds around the
        negative peak of the slow-wave, as in [1]_. This means that YASA will look for a
        spindle in a 2.4 seconds window centered around the downstate of the slow-wave.

    Returns
    -------
    _events : :py:class:`pandas.DataFrame`
        The slow-wave detection is modified IN-PLACE (see Notes). To see the updated dataframe,
        call the :py:meth:`yasa.SWResults.summary` method.

    Notes
    -----
    From [1]_:

        "SO–spindle co-occurrence was first determined by the number of spindle centers
        occurring within a ±1.2-sec window around the downstate peak of a SO, expressed as
        the ratio of all detected SO events in an individual channel."

    This function adds three columns to the output detection dataframe:

    * `CooccurringSpindle`: a boolean column (True / False) that indicates whether the given
      slow-wave co-occur with a sleep spindle.

    * `CooccurringSpindlePeak`: the timestamp of the peak of the co-occurring,
      in seconds from beginning of recording. Values are set to np.nan when no co-occurring
      spindles were found.

    * `DistanceSpindleToSW`: The distance in seconds from the center peak of the spindles and
      the negative peak of the slow-waves. Negative values indicate that the spindles occured
      before the negative peak of the slow-waves. Values are set to np.nan when no co-occurring
      spindles were found.

    References
    ----------
    .. [1] Kurz, E. M., Conzelmann, A., Barth, G. M., Renner, T. J., Zinke, K., & Born, J.
           (2021). How do children with autism spectrum disorder form gist memory during sleep?
           A study of slow oscillation–spindle coupling. Sleep, 44(6), zsaa290.
    """
    assert isinstance(signals, pd.DataFrame), "spindles must be a detection dataframe."
    distance_rrpeak_to_sw_peak = []
    cooccurring_rr_peaks = []

    # Loop across channels
    rr_peaks = info['ECG_R_Peaks']
    for chan in sws._events['Channel'].unique():
        sw_chan_peaks = sws._events[sws._events["Channel"] == chan]["NegPeak"].to_numpy()
        # Loop across individual slow-waves
        for sw_negpeak in sw_chan_peaks:
            start = sw_negpeak - lookaround
            end = sw_negpeak + lookaround
            mask = np.logical_and(start < rr_peaks, rr_peaks < end)
            if any(mask):
                # If multiple spindles are present, take the last one
                rr_peak = rr_peaks[mask][-1]
                cooccurring_rr_peaks.append(rr_peak)
                distance_rrpeak_to_sw_peak.append(rr_peak - sw_negpeak)
            else:
                cooccurring_rr_peaks.append(np.nan)
                distance_rrpeak_to_sw_peak.append(np.nan)

    # Add columns to self._events: IN-PLACE MODIFICATION!
    sws._events["CooccurringRR"] = ~np.isnan(distance_rrpeak_to_sw_peak)
    sws._events["CooccurringRRPeak"] = cooccurring_rr_peaks
    sws._events['DistanceRRPeakToSW'] = distance_rrpeak_to_sw_peak

def cooccuring_fun(sws, coi='C3'):
    # create mask for channel-wise cooccurence analysis 
    mask = np.logical_and(sws.summary().Channel==coi, 
                          sws.summary().CooccurringSpindle == True)
    try:
        df_sync = sws.get_sync_events(center='NegPeak', time_after=1.2, 
                                      time_before=1.2, filt=(None, 2.0),
                                      mask=mask)
    except:
        df_sync = []
                   
    return mask, df_sync
   
def plot_co_occurence(sws, mask, df_sync, fig_path, subject, cond, save=True):   
    plt.figure()
    ax = sns.histplot(sws.summary()[mask]['DistanceSpindleToSW'], 
                      binwidth=0.1, stat='probability')
    ax.set_xlim(-1.2,1.2)
    plt.axvline(0., lw=1.5, color='k', linestyle='--')
    ax2 = ax.twinx()
    sns.lineplot(x='Time',y='Amplitude',data=df_sync, ci=False, 
                 axes=ax2, linewidth=5, color='k')
    plt.tight_layout()
    if save:
        plt.savefig(fig_path + f'event_peri_hist_{subject}_{cond}')
        plt.close('all')
                    
def sw_phase_binning(epoch):
    # filter data to perform slow wave detection
    data_lp = epoch.copy().filter(l_freq=None, h_freq=2).get_data('eeg')*1e6
    sws = [yasa.sw_detect(data = data_lp[i,:,:], sf=epoch.info['sfreq'], 
                          ch_names=epoch.info.ch_names[0:23], 
                          freq_sw=(None, None), dur_neg=(0.1, 2), dur_pos=(0.1, 2),
                          amp_neg=(None, None), coupling = False, amp_pos=(None, None),
                          amp_ptp=(20, 350), verbose='CRITICAL') 
           for i in range(np.shape(data_lp)[0])]

    ## remove all empty sequences without slow waves 
    sws_idx = []
    for i in range(len(sws)):
        if sws[i] != None:
            if len(sws[i].summary()[sws[i].summary()['Channel']=='C3'].to_numpy()) > 0:
                sws_idx.append(1)
        else:
            sws_idx.append(0)
       
    # get sw idx
    sws_idx = np.asarray(sws_idx)
    
    # space 90 degree bins for slow waves representing negative/postive half wave durations
    half1, half2, half3, half4 = [],[],[],[]
    for i in np.where(sws_idx == 1)[0]:
        if sws[i] != None:
            C3_df = sws[i].summary()[sws[i].summary().Channel=='C3'].reset_index()
            if len(C3_df) > 0:
                idx_neg_nearest = find_nearest(C3_df['MidCrossing'], 4)
                idx_not = np.where(np.arange(0, len(C3_df)) != idx_neg_nearest)[0]
                C3_df = C3_df.reset_index().drop(idx_not, inplace=False)
            
                half1.append(np.linspace(C3_df['Start']*epoch.info['sfreq'], 
                                         C3_df['NegPeak']*epoch.info['sfreq'], num=90, endpoint=True).astype(int))
                half2.append(np.linspace(C3_df['NegPeak']*epoch.info['sfreq'] + 1, 
                                         C3_df['MidCrossing']*epoch.info['sfreq'], num=90).astype(int))
                half3.append(np.linspace(C3_df['MidCrossing']*epoch.info['sfreq'] + 1, 
                                         C3_df['PosPeak']*epoch.info['sfreq'], num=90).astype(int))
                half4.append(np.linspace(C3_df['PosPeak']*epoch.info['sfreq'] + 1, 
                                         C3_df['End']*epoch.info['sfreq'], num=90).astype(int))                    
    
    # determine pinknoise targeted slow wave phase 
    crossings = []
    for i in range(len(half1)):
        crossing = thresholdcrossings(np.concatenate([half1[i], half2[i], half3[i], half4[i]]), 
                                      epoch.info['sfreq']*4)
        if len(crossing) != 0:
            crossings.append(crossing[0])
        else:
            crossings.append(np.nan)
    
    # invert sign in radians to make comparable with analytical analysis
    return -np.deg2rad(crossings) 
            
def SO_spindle_analysis(epoch, reference, subject, condition, fig_path):  
    if 'Mastoids' in reference:       
        # Identify slow waves in data
        data_lp = epoch.copy().filter(l_freq=None, h_freq=2).get_data('eeg')*1e6
        sws = [yasa.sw_detect(data = data_lp[i,:,:], sf=128, 
                              ch_names=epoch.info.ch_names[0:23], 
                              freq_sw=(None, None), dur_neg=(0.1, 2), dur_pos=(0.1, 2),
                              amp_neg=(None, None), coupling = False, amp_pos=(None, None),
                              amp_ptp=(40, 350), verbose='CRITICAL') 
                for i in range(np.shape(data_lp)[0])]
        
        ## add ndPAC info and remove all empty sequences without slow waves 
        coupling_summaries = []
        for i in range(len(sws)):
            if sws[i] != None:
                coupling_summaries.append(SO_spindle_coupling(sws[i], data_broad = epoch.get_data(picks='eeg')*1e6, 
                                                              idx=i, sf=128, target='stim_onset'))
        
        sw_summaries, sws_idx = [], []
        for i in range(len(sws)):
            if sws[i] != None:
                sw_summaries.append(sws[i].summary(grp_chan=True))
                if len(sws[i].summary()[sws[i].summary()['Channel']=='C3'].to_numpy()) > 0:
                    sws_idx.append(1)
            else:
                sws_idx.append(0)
                    
        sws_idx = np.asarray(sws_idx)  
    
    
        # Identify spindles in data   
        sp = []
        for i in range(len(epoch)):
            try:
                sp.append(yasa.spindles_detect(data = epoch.get_data(picks='eeg')[i,:,:]*1e6, sf=128,
                                               ch_names=epoch.info.ch_names[0:23], freq_sp=(11, 16),
                                               freq_broad=(1, 30), duration=(0.5, 3), min_distance=500, 
                                               thresh={'rel_pow': None, 'corr': None, 'rms': 1.5},
                                               multi_only=False, verbose='CRITICAL'))
            except:
                ValueError
        
        # leave only epochs with spindles
        sp_summaries = []
        for idx, spindle in enumerate(sp):
            if spindle != None:
                sp_summaries.append(spindle.summary(grp_chan=True)) 
                   
        # save and append sp summaries 
        sp_summaries = pd.concat(sp_summaries)
        sp_summaries = sp_summaries.reset_index()
        count = sp_summaries.groupby(['Channel']).agg({'Count': 'sum'})
        df_sp = sp_summaries.groupby(['Channel'], as_index=False).mean()
        df_sp['Count'] = count.Count.to_numpy()
        df_sp['Subject'] = subject
        df_sp['Condition'] = condition
        
        # save and append coupling summaries 
        df_couple = pd.concat([pd.concat(coupling_summaries[i]) for i in range(len(coupling_summaries))]).reset_index()
        del df_couple['index'], df_couple['level_0']
        df_couple = df_couple.groupby(['Channel'], as_index=False).mean()
        df_couple['Subject'] = subject
        df_couple['Condition'] = condition
        
        # save and append slow wave summaries
        df_sw = pd.concat(sw_summaries)
        df_sw = df_sw.reset_index()
        count = df_sw.groupby(['Channel']).agg({'Count': 'sum'})
        df_sw = df_sw.groupby(['Channel'], as_index=False).mean()
        df_sw['Count'] = count.Count.to_numpy()
        df_sw['Subject'] = subject
        df_sw['Condition'] = condition
    else:
        df_sw, df_sp, df_couple = [], [], []
             
    return df_sw, df_sp, df_couple
    
def auditory_stim_epochs(maindir, epoch_time = (-4, 4)):
    # create log file 
    log_path = '/media/administrator/data/Study_1_data/Data_tracking/epoch_reports.csv'
    for i, files in tqdm(enumerate(maindir)):
        print(i, files.split('/')[-1])
        if i == 0:
            df = pd.DataFrame(columns = ['Subject', 'Condition', 'Stimuli', 'First bursts', 
                                         'Bad Channels', 'Epochs included'])
        else:
            df = pd.read_csv(log_path, usecols = ['Subject', 'Condition', 'Stimuli', 'First bursts', 
                                                  'Bad Channels', 'Epochs included']) 
        # load data instead of pre-processing data 
        Data = load_preprocessed_data(files)[0]
        # Data = _pre_process_sleep_data(files, low_density=True, reference=None, validation='auditory', stageing=False)
        
        # take first (adjusted) pinknoise bursts as center point
        first_bursts = Data.pinknoise_times_sync[::2]
        if first_bursts != []:
            # check to see if condition is sham, if so, create second instance of center points to generate down sham condition
            cond = subject_cond_parser(files, study_phase = 'sleep')
            subject = files.split("/")[-1].split('_')[0]
            # create down sham instance subtracting first bursts by respective p2p duration difference
            reject_criteria = dict(eeg=550e-6)
            # EQI method determines bad channels before epoching
            Data.detect_bad_chans(method='EQI')
            if cond == 'sham':
                subject_table = pd.DataFrame(pd.read_csv('/media/administrator/data/Study_1_data/Data_tracking/Subject_table.csv', 
                                                         delimiter=',', dtype='str', header=0))
                p2p_delay = int(subject_table[subject_table['Subject ID'] == subject]['Peak to Peak duration'].to_list()[0][0:3])/1000
                sham_down_idx = np.asarray(first_bursts) - round(p2p_delay*Data.sfreq)
    
                # generate center of stimulation index points
                center_down, _ = yasa.get_centered_indices(Data.data[:,Data.chans.index('C3')], sham_down_idx, 
                                                           npts_before = Data.sfreq*abs(epoch_time[0]), 
                                                           npts_after = Data.sfreq*abs(epoch_time[0]))
                
                # create mne epoch object with online reference + offline re-reference
                info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
                info['bads'] = Data.bad_chans
            
                # create epochs with online reference
                epochs_down = mne.EpochsArray(np.swapaxes(Data.data[center_down]/1e6, 1, 2), info, tmin = epoch_time[0], 
                                              baseline=None, proj=False, reject = None)
                epochs_down.set_montage(mne.channels.make_standard_montage('standard_1005')) 
                epochs_down.interpolate_bads()
                art_idx = art_detect(epochs_down)
                epochs_down.drop(art_idx)
                epochs_down.drop_bad(reject = reject_criteria) 
                
                # mastoid referenced
                epochs_down_mastoids = epochs_down.copy().set_eeg_reference(['M1','M2'])
            else:
                epochs_down = []
                epochs_down_mastoids = []
                
            # generate center of stimulation index points
            center, _ = yasa.get_centered_indices(Data.data[:,Data.chans.index('C3')], np.asarray(first_bursts), 
                                                  npts_before = Data.sfreq*abs(epoch_time[0]), 
                                                  npts_after = Data.sfreq*abs(epoch_time[0]))
                
            info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
            info['bads'] = Data.bad_chans
            
            # create epochs with online reference
            epochs = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = epoch_time[0], 
                                     baseline=None, proj=False, reject = None)
            epochs.set_montage(mne.channels.make_standard_montage('standard_1005')) 
            epochs.interpolate_bads()
            if cond != 'sham':
                art_idx = art_detect(epochs)
            epochs.drop(art_idx)
            epochs.drop_bad(reject = reject_criteria)
            
            # =============================================================================
            ## CREATES CHANNEL
            #  EEG_index = np.r_[Data.chans.index('F3'), Data.chans.index('C3'), Data.chans.index('P3')]
            #  raw = mne.channels.combine_channels(epochs, groups=dict(Left_hemispher = EEG_index))       
            # =============================================================================
                    
            # create epochs with mastoids reference
            epochs_mastoids = epochs.copy().set_eeg_reference(['M1','M2'])
            
            # log info
            new_row = {'Subject': subject, 
                       'Condition': cond, 
                       'Stimuli': len(first_bursts)*2,
                       'First bursts': len(first_bursts), 
                       'Bad Channels': Data.bad_chans,
                       'Epochs included': len(epochs_mastoids.events)
                       }
        
            # update dataframe 
            df = df.append(new_row, ignore_index=True)
            
            ## compute CSD 
            # epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
            
            ## compute GED 
            # see GED script
            
            ## GFP
            #GFP = np.std(epochs.get_data(picks='eeg'),axis=0)
            
            # dump epochs
            save_path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/'
            if epochs_down != []:
                epochs_down.save(save_path + f'{subject}_{cond}_down_epo.fif', overwrite=True)
                epochs_down_mastoids.save(save_path + f'{subject}_{cond}_down_mast_epo.fif', overwrite=True)
            epochs.save(save_path + f'{subject}_{cond}_epo.fif', overwrite=True)
            epochs_mastoids.save(save_path + f'{subject}_{cond}_mast_epo.fif', overwrite=True)
            #epochs_csd.save(save_path + f'{subject}_{cond}_epo.fif', overwrite=True)
            
            # save dataframe
            df.to_csv(log_path)

def sw_spindle_detection_stim_auditory(save=True):
    # configure paths 
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    report_path = '/media/administrator/data/Study_1_data/Data_tracking/[].csv'
    fig_path = '/media/administrator/data/Study_1_data/Figures/Pinknoise_epochs/'
    results = defaultdict(lambda: [])
    sw_df, sp_df, couple_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i, file in enumerate(tqdm(maindir)):
        # print(i, file.split('/')[-1])
        subject = file.split("/")[-1].split("_")[0]
        if '_sham_down_' in file:
            cond = 'sham down'
        else:
            cond = file.split("/")[-1].split("_")[1]
        if '_mast_' in file:
            reference = 'Linked Mastoids'
            
            # load data 
            epoch = mne.read_epochs(file, preload=True).apply_baseline((-4, -1.5)).resample(128)
                  
            # summaries
            sw_summary, sp_summary, couple_summary = SO_spindle_analysis(epoch, reference,
                                                                         subject, cond, fig_path) 
            
            # save as dataframes 
            sw_df = pd.concat([sw_df, sw_summary])
            sp_df = pd.concat([sp_df, sp_summary])
            couple_df = pd.concat([couple_df, couple_summary])
            
            del epoch
            
        else:
            reference = 'Cpz'
            pass
    
    if save:
        stat_path = '/media/administrator/data/Study_1_data/Statistics/SW_spindle_summary/'
        sw_df.to_csv(stat_path + 'sw_summary.csv')
        sp_df.to_csv(stat_path + 'sp_summary.csv')
        couple_df.to_csv(stat_path + 'coupling_summary.csv')
    else:
        return sw_df, sp_df, couple_df
      
def sw_spindle_detection_half_night(plot=True):
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files if 'av.p' in i])
    for i, files in enumerate(tqdm(maindir)):
        print(i, files.split('/')[-1])
        fig_path = '/media/administrator/data/Study_1_data/Figures/Half_night/'
        log_path = '/media/administrator/data/Study_1_data/Statistics/SW_spindle_summary/'
        # load data instead of pre-processing data 
        Data = load_preprocessed_data(files)[0]
        # take first (adjusted) pinknoise bursts as center point
        burst_range = Data.pinknoise_times_sync[0:-1]
        if len(burst_range) != 0:
            # check condition 
            cond = subject_cond_parser(files, study_phase = 'sleep')
            subject = files.split("/")[-1].split('_')[0]
            # load bad channel info before epoching
            data_sheet = pd.read_csv('/media/administrator/data/Study_1_data/Data_tracking/epoch_reports.csv')
            mask = np.logical_and(data_sheet['Subject']==subject, 
                                  data_sheet['Condition']==cond)
            bad_items = data_sheet[mask]['Bad Channels'].to_numpy()
            if len(bad_items) > 0:
                Data.bad_chans = bad_items[0].split("'")[1::2]
            else:
                Data.bad_chans = []
            
            ## Epoch based 
            # # generate epochs based on stimulation index points
            # _, eps = yasa.sliding_window(Data.data[burst_range[0]-Data.sfreq:burst_range[-1]+-Data.sfreq, :].T, 
            #                              sf = Data.sfreq, window = 30)
            # # create mne epoch object
            # info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
            # info['bads'] = Data.bad_chans        
            # # create epochs with online reference
            # epochs = mne.EpochsArray(eps/1e6, info, tmin = 0, 
            #                          baseline=None, proj=False, reject = None)
            
            ## Data length based
            # create mne epoch object
            info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
            info['bads'] = Data.bad_chans        
            # create epochs with online reference
            epochs = mne.io.RawArray(Data.data.T/1e6, info)
            # cut data to length
            # epochs.crop(tmin=(burst_range[0]/512)-2, tmax=(burst_range[-1]/512)+2)
            if epochs.times.shape[0]/epochs.info['sfreq']/60 >= 210:
                epochs.crop(tmin=0, tmax=210*60)
            epochs.set_montage(mne.channels.make_standard_montage('standard_1005')) 
            epochs.interpolate_bads()
            # mastoid referencing
            epochs.set_eeg_reference(['M1','M2'])  
            epochs.resample(128)
            # create sudo hypno for now
            sls = yasa.SleepStaging(epochs, eeg_name='Cz', eog_name='EOG_R', emg_name='EMG_L')
            hypno_pred = yasa.hypno_upsample_to_data(hypno = yasa.hypno_str_to_int(sls.predict()),
                                                     data = epochs, sf_hypno = 1/30, sf_data = 128)
            hypno_pred = scipy.ndimage.median_filter(hypno_pred, size=128*10)
            # sw detection
            sws = yasa.sw_detect(data = epochs.pick('eeg'), coupling = True, 
                                 remove_outliers = True, hypno = hypno_pred, 
                                 coupling_params={"freq_sp": (12, 16), "time": 2, "p": 0.05}) 
            # spindle detection
            sp = yasa.spindles_detect(data = epochs.pick('eeg'), freq_sp=(12, 16), remove_outliers = True,
                                      duration=(0.5, 3), thresh={'rel_pow': None, 'corr': None, 'rms': 1.5},
                                      hypno = hypno_pred)
            # coupling fun
            sws.find_cooccurring_spindles(sp.summary(), lookaround=1.2)
            
            # save co-occurence df with mask
            mask, df_sync = cooccuring_fun(sws)
            
            # plot co-occurence
            if len(df_sync) > 0:
                if plot:
                    plot_co_occurence(sws, mask, df_sync, fig_path, subject, 
                                      cond, save=True)
                df_sync['Condition'] = [cond]*len(df_sync)
                df_sync['Subject'] = [subject]*len(df_sync)
                df_sync.to_csv(log_path + f'half_co_mask_{subject}_{cond}.csv')
            
            # save sw df
            sws_summary = sws.summary()
            sws_summary['Condition'] = [cond]*len(sws_summary)
            sws_summary['Subject'] = [subject]*len(sws_summary)
            sws_summary.to_csv(log_path + f'half_sw_{subject}_{cond}.csv')
            # save sp df
            sp_summary = sp.summary()
            sp_summary['Condition'] = [cond]*len(sp_summary)
            sp_summary['Subject'] = [subject]*len(sp_summary)
            sp_summary.to_csv(log_path + f'half_sp_{subject}_{cond}.csv')
            
            del Data, epochs, sws, sp, df_sync
    
def auditory_stim_results(maindir):
    # configure paths
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    report_path = '/media/administrator/data/Study_1_data/Data_tracking/[].csv'
    fig_path = '/media/administrator/data/Study_1_data/Figures/Pinknoise_epochs/'
    # initialize dictionary for results
    results = defaultdict(lambda: [])
    # params for single subject analysis
    par = {'baseline_window':(-3000,-1500),'response_window':(0,2000), 'k':1.2,
           'min_snr':1.1, 'max_var':99, 'embed':False,'n_steps':100}
    for i, file in enumerate(tqdm(maindir)):
        print(i, file.split('/')[-1])
        subject = file.split("/")[-1].split("_")[0]
        if '_sham_down_' in file:
            cond = 'sham down'
        else:
            cond = file.split("/")[-1].split("_")[1]
        if '_mast_' in file:
            reference = 'Linked Mastoids'
        else:
            reference = 'Cpz'
        
        # load data & create evoked object
        evk = mne.read_epochs(file, preload=True).apply_baseline((-4, -1.5)).average().resample(128)
        
        # compute subject wise PCI
        pci = pci_st.calc_PCIst(signal_evk = evk.get_data('eeg')*1e6, times = evk.times*1000, **par)
        
        # update results dictionary
        results['Subject'].append(subject)
        results['Condition'].append(cond)
        results['Reference'].append(reference)
        results['PCI'].append(pci)
        results['Data'].append(evk)
        
    # convert results dict to dataframe
    df = pd.DataFrame(results)
    
    # create contrast within subjects, only!
    def create_subj_comp_evoked(df):
        df_lm = df.copy().drop(df.loc[df['Reference']=='Cpz'].index, inplace=False)
        subject = list(dict.fromkeys(list(df_lm.Subject)))
        comp_df = pd.DataFrame(columns=['Subject', 'Evoked Difference Up', 
                                        'Evoked Difference Down'])
        for sub in subject:
            sub_res = df_lm[df_lm.Subject == sub]
            ## Create comparision object
            # up comparison  
            try:
                gav_comp_up = mne.combine_evoked([sub_res[sub_res.Condition=='up']['Data'].squeeze(), 
                                                  sub_res[sub_res.Condition=='sham']['Data'].squeeze()],
                                                 weights=[1, -1])
            except:
                gav_comp_up = 0
                
            # down comparison  
            try:
                gav_comp_down = mne.combine_evoked([sub_res[sub_res.Condition=='down']['Data'].squeeze(), 
                                                    sub_res[sub_res.Condition=='sham down']['Data'].squeeze()],
                                                   weights=[1, -1])
            except:
                gav_comp_down = 0
                
            # log subject/bad channels
            new_row = {'Subject': sub, 
                       'Evoked Difference Up': gav_comp_up,
                       'Evoked Difference Down': gav_comp_down}
                
            # update dataframe 
            comp_df = comp_df.append(new_row, ignore_index=True)
            
        return comp_df
     
    comp_df = create_subj_comp_evoked(df)
    
    gav_down_comp = mne.grand_average(list(comp_df.copy().drop(comp_df.loc[comp_df['Evoked Difference Down']==0].index)
                                           ['Evoked Difference Down']))
    gav_up_comp = mne.grand_average(list(comp_df.copy().drop(comp_df.loc[comp_df['Evoked Difference Up']==0].index)
                                         ['Evoked Difference Up']))

    ## Plotting fun
    ts_args = dict(spatial_colors = True, gfp=True, time_unit='s')
    topomap_args = dict(outlines = 'head', time_unit='s', cmap = 'Spectral_r', time_format = "%0.2f s")
    times = np.asarray([-0.5, 0, 0.25, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0])
    all_times = np.arange(-3.0, 3.25, 0.25)
    results_gav = defaultdict(lambda: [])
    for (cond, ref), evok in df.groupby(['Condition', 'Reference'])['Data']:
        print(cond, ref)
        gav = mne.grand_average(list(evok))
        gav.plot_joint(times, ts_args = ts_args, topomap_args = topomap_args,
                       title = cond.capitalize() + ' ' + ref.capitalize())     
        gav.plot_topomap(all_times, ch_type='eeg', ncols='auto', nrows='auto',
                         title=f'Amplitude distributions - {cond.capitalize()} - {ref.capitalize()}',
                         **topomap_args) 
        
        results_gav['Condition'].append(cond)
        results_gav['Reference'].append(ref)
        results_gav['Data'].append(gav)
        
    # convert results dict to dataframe
    df_gav = pd.DataFrame(results_gav)
    
def auditory_phase_analysis(plot=True, save=True):
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    fig_path = '/media/administrator/data/Study_1_data/Figures/Pinknoise_epochs/'
    results = defaultdict(lambda: [])
    results_bin = defaultdict(lambda: [])
    for i, file in enumerate(tqdm(maindir)):
        subject = file.split("/")[-1].split("_")[0]
        if '_sham_down_' in file:
            cond = 'sham down'
        else:
            cond = file.split("/")[-1].split("_")[1]
        if '_mast_' not in file:
            # load data 
            epoch = mne.read_epochs(file, preload=True).apply_baseline((-4, -1.5)).resample(128)
            # extract phase & such
            sw_pha, _, _ = imf_analytical_transform(epoch) 
            epo_len = len(epoch.times)
            pn_phase = [sw_pha[i][int(epo_len/2)] for i in range(len(sw_pha))]
            # binned phases
            crossings = sw_phase_binning(epoch)
            if plot:
                ## polar plot for phase targets  
                ax = plt.subplot(111, projection='polar')
                ax.hist(pn_phase, density=True)
                ax.set_title(f'Phase targeting accuracy (Analytical): {subject} {cond}', va='bottom')
                plt.savefig(f'{fig_path}{subject}_{cond}_phase_targeting.png')
                plt.close('all')
            
                ## polar plot for phase targets 
                if ~np.all(np.isnan(crossings)):
                    plt.figure()
                    ax2 = plt.subplot(111, projection='polar')
                    ax2.hist(crossings, density = True)
                    ax2.set_title(f'Phase targeting accuracy (SW): {subject} {cond}', va='bottom')
                    plt.savefig(f'{fig_path}{subject}_{cond}_phase_targeting_sw.png')         
                    plt.close('all')

            # update results dictionaries
            results['Subject'].extend([subject]*len(pn_phase))
            results['Condition'].extend([cond]*len(pn_phase))
            results['Target Phase (Analytical)'].extend(np.asarray(pn_phase))
            
            results_bin['Subject'].extend([subject]*len(crossings))
            results_bin['Condition'].extend([cond]*len(crossings))
            results_bin['Target Phase (Binned)'].extend(np.asarray(crossings))
            
    # create df
    phase_df = pd.DataFrame(results)
    phase_b_df = pd.DataFrame(results_bin)
    
    if save:
        phase_df.to_csv('/media/administrator/data/Study_1_data/Statistics/Auditory_validation/phase_report.csv')
        phase_b_df.to_csv('/media/administrator/data/Study_1_data/Statistics/Auditory_validation/phase_binned_report.csv')
    
    return phase_df, phase_b_df

def group_phase_targeting_plot(phase_df, phase_b_df):
    ## Circular mean + resultant vector length plot  
    kwargs_arrow={'fc': 'tab:red', 'ec': 'tab:red'}
    with plt.style.context('seaborn'):
        sns.set_palette("flare")
        for df, meth in zip([phase_df, phase_b_df],['Analytical', 'Binned']):
            g = sns.FacetGrid(df, col='Condition', hue='Condition',
                      subplot_kws=dict(projection='polar'), height=4.5,
                      sharex=True, sharey=True, despine=False, ylim=0)
            g.map_dataframe(sns.histplot, f'Target Phase ({meth})', 
                    stat='density', bins = 12)
            # with plt.style.context('sns'):
            #     sns.set(font_scale=1.0, style='white')
            #     for cond in df.Condition.unique():
            #         pg.plot_circmean(df[df.Condition==cond][f'Target Phase ({meth})'], 
            #                          kwargs_arrow = kwargs_arrow_pg,
            #                          kwargs_markers=dict(marker="None"))
            for ax, con in zip(range(len(g.axes[0])), df.Condition.unique()):
                rv = pg.circ_r(df[df.Condition==con][f'Target Phase ({meth})'])
                phi = pg.circ_mean(df[df.Condition==con][f'Target Phase ({meth})'])
                circ_std = scipy.stats.circstd(df[df.Condition==con][f'Target Phase ({meth})'], nan_policy='omit')
                g.axes[0, ax].arrow(0, 0, phi, rv, **kwargs_arrow)       
                plt.tight_layout()
                print(f'The circular mean for {con.upper()} - {meth} is {phi.round(4)}, with a stand deviation of {circ_std.round(4)}, and a resultant vector length of : {rv.round(4)}')
            plt.show()      
  
def foo_power():
    # blah blah        
    for idx, ep in enumerate(tqdm(df['Data'])):
        reference, subject, condition = df.Reference[idx], df.Subject[idx], df.Condition[idx]
        epochs = ep.load_data().apply_baseline((-4, -1.5)).resample(128)
        sw_summary, sp_summary, sws_idx = SO_spindle_phase_analysis(epochs, reference,
                                                                    subject, condition, 
                                                                    fig_path)
        sw_summaries.append(sw_summary)
        sp_summaries.append(sp_summary)
        
        # Power distributions (Pre/Post stim onset)
    for x,y in zip([-4.0, 0.],[0., 4.0]):
        epoch.plot_psd_topomap(tmin=x, tmax=y, bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Slow Sigma'),
                                                        (12, 16, 'Fast Sigma'), (16, 30, 'Beta')], ch_type='eeg',
                                dB=False, normalize = True, cmap='Spectral_r')

    # Power with fooof 1/f component 
    for x,y in zip([-4., 0.],[0., 4.]):
        band_powers = oscillatory_plot_psd_map(epoch.copy().pick('eeg'), tmin=x, tmax=y, plot=True)
    
    fm = periodic_fit(epoch.copy().pick('eeg'), foi=(0.5, 30), tmin=0, tmax=3)
    spectra = np.mean([np.asarray(fm[i]._peak_fit) for i in range(len(fm))], 0)
    plot_spectrum(fm[0].freqs, spectra, color='green', label='Final Periodic Fit')
    
    # TF analysis 
    _, Sxx = tfr_analysis(Data=epoch, l_freq=5, h_freq=20, steps=0.25, 
                          method = 'wavelet', baseline=[-3, 3], chan = 'C3', 
                          itc_calculation = None, plot=True, output='avg', 
                          cmap = 'Spectral_r', save_path=fig_path + subject)
    
    return Sxx
   
def group_so_sp_analysis():
    log_path = '/media/administrator/data/Study_1_data/Statistics/SW_spindle_summary/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(log_path) for i in files])
    fig_path
    df_sync, sws, sp = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i, files in enumerate(tqdm(maindir)):
        if 'half_co_mask_' in files:
            df_sync = pd.concat([df_sync,pd.read_csv(files, index_col=0)])
        elif 'half_sw_' in files:
            sws = pd.concat([sws,pd.read_csv(files, index_col=0)])
        elif 'half_sp_' in files:
            sp = pd.concat([sp,pd.read_csv(files, index_col=0)])
          
    ## Coincidence matrices for spindles and SOs- Group level
    cm_sp = coincidence_matrix(sp, sf=128, plot=True, 
                               window='group', standardize=True)
    plt.savefig(fig_path + 'group_coincidence_matrix_sp_.png')
    cm_sw = coincidence_matrix(sws, sf=128, plot=True, 
                               window='group', standardize=True)
    plt.savefig(fig_path + 'group_coincidence_matrix_sw_.png')
    
    ## Plot ndPAC- Group level
    plt.figure()
    for cond in 
    sws.groupby(['Condition','Subject']).mean()['ndPAC'].sham.plot.kde();
    plt.savefig(fig_path + files.split('/')[-1] + '_ndPAC_' + epoch_name + '.png')
    print(f'ndPAC means: {sws.groupby("Channel")["ndPAC"].mean()}')
    pg.plot_circmean(df_sw['PhaseAtSigmaPeak'])
    plt.savefig(fig_path + files.split('/')[-1] + '_circular_phase_' + epoch_name + '.png')
    print('Circular mean: %.3f rad' % pg.circ_mean(df_sw['PhaseAtSigmaPeak']))
    print('Vector length: %.3f' % pg.circ_r(df_sw['PhaseAtSigmaPeak'])) 


#%%
## Run things here dumb dumb
# auditory_stim_epochs(maindir, epoch_time = (-4, 4))
# sw_spindle_detection_half_night()
# phase_df, phase_b_df = auditory_phase_analysis(plot=True, save=True)  
# phase_df = pd.read_csv('/media/administrator/data/Study_1_data/Statistics/Auditory_validation/phase_report.csv')
# phase_b_df = pd.read_csv('/media/administrator/data/Study_1_data/Statistics/Auditory_validation/phase_binned_report.csv')
# sw_df, sp_df, couple_df = sw_spindle_detection_stim_auditory()

#%%       
## plot difference for C3
# mne.viz.plot_compare_evokeds(dict(online_ref=epochs.average(method='mean', picks='C3'), 
#                                   offline_ref=epochs_mastoids.average(method='mean', picks='C3')),
#                                   legend='upper left', show_sensors='upper right',
#                                   title=f'{files.split("/")[-1]} - C3')


# ## ERPAC plots
# erpac = ERPAC(data = epochs_mastoids, f_pha=[0.3, 4], f_amp=(4, 20, .5, .5), n_perm=None, 
#               smooth=200, method = 'gc', edges=0.5, stationarity_t=False, plot=True, 
#               save_path=fig_path + files.split('/')[-1])

# #%% PLV analysis 
# _, plv = spectral_conn(epoch.get_data(picks ='eeg'), epoch.ch_names[0:23], 
#                        sf = epoch.info['sfreq'], foi = (0.5, 2), 
#                        method='ciplv', tmin=0, tmax=4)

# ## Plot PLV 
# fig, ax = plt.subplots()
# im, cm = mne.viz.plot_topomap(plv, pos = epoch.info, vmin = np.percentile(plv, 5),
#                               vmax = np.percentile(plv, 95), cmap='Spectral_r', 
#                               axes=ax, show=True)
# fig.colorbar(im, ax=ax)   
# plt.title(f'PLV seeded to C3 - {files.split("/")[-1][0:10]} - {epoch_name}')

# pickle.dump(plv, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + epoch_name + '_plv.p',"wb"))
# plt.savefig(fig_path + files.split('/')[-1] + epoch_name + '_plv.png')
# plt.close('all')
# del plv 

# ## Coincidence matrices for spindles and SOs- Group level
# cm_sp = coincidence_matrix(sp, plot = True, window='whole')
# plt.savefig(fig_path + files.split('/')[-1] + '_coincidence_matrix_sp_' + epoch_name + '.png')
# cm_sw = coincidence_matrix(sws, plot = True, window='whole')
# plt.savefig(fig_path + files.split('/')[-1] + '_coincidence_matrix_sw_' + epoch_name + '.png')

# # Distribution of ndPAC value [ADD TO GROUP LEVEL ANALYSIS] 
# plt.figure()
# df_sw['ndPAC'].hist();
# plt.savefig(fig_path + files.split('/')[-1] + '_ndPAC_' + epoch_name + '.png')
# print(f'ndPAC means: {df_sw.groupby("Channel")["ndPAC"].mean()}')
# pg.plot_circmean(df_sw['PhaseAtSigmaPeak'])
# plt.savefig(fig_path + files.split('/')[-1] + '_circular_phase_' + epoch_name + '.png')
# print('Circular mean: %.3f rad' % pg.circ_mean(df_sw['PhaseAtSigmaPeak']))
# print('Vector length: %.3f' % pg.circ_r(df_sw['PhaseAtSigmaPeak']))      
  


#%%
# ## Group level phase analysis
# ## ADD ndPAC, ERPAC, PLV, GFP group analysis 

# ## Plot PLV 
# fig, ax = plt.subplots()
# im, cm = mne.viz.plot_topomap(np.vstack(up_plv).mean(0), pos = epochs.info, vmin = np.percentile(np.vstack(up_plv).mean(0), 5),
#                               vmax = np.percentile(np.vstack(up_plv).mean(0), 95), cmap='Spectral_r', axes=ax, show=True)
# fig.colorbar(im, ax=ax)   
# plt.title(f'PLV seeded to C3 - Up')



# mne.viz.plot_compare_evokeds(dict(up_state=epochs_up_mastoids.average(method='mean', picks='C3'),
#                                   down_state=epochs_down_mastoids.average(method='mean', picks='C3'),
#                                   sham_up_state=epochs_sham_mastoids.average(method='mean', picks='C3'),
#                                   sham_down_state=epochs_sham_mastoids_down.average(method='mean', picks='C3')), 
#                               legend='upper left', show_sensors='upper right',
#                               title='Group level average pinknoise epochs - C3 (mastoids)')


# times = epochs_up_mastoids.times
# plt.plot(times, epochs_down_mastoids.average(method='mean', picks='C3').get_data().squeeze()*1e6 - epochs_sham_mastoids_down.average(method='mean', picks='C3').get_data().squeeze()*1e6, 
#           label = 'Down - Sham Down')
# plt.plot(times, epochs_up_mastoids.average(method='mean', picks='C3').get_data().squeeze()*1e6 - epochs_sham_mastoids.average(method='mean', picks='C3').get_data().squeeze()*1e6,
#           label = 'Up - Sham Up')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage difference (uV)')
# plt.show()

# # plt.savefig(fig_path + files.split('/')[-2] + '_average_epoch.png')
# mne.viz.plot_compare_evokeds(dict(up_state=epochs_up.average(method='mean', picks='C3'),
#                                   down_state=epochs_down.average(method='mean', picks='C3'),
#                                   sham_up_state=epochs_sham.average(method='mean', picks='C3'),
#                                   sham_down_state=epochs_sham_down.average(method='mean', picks='C3')),
#                               legend='upper left', show_sensors='upper right',
#                               title='Group level average pinknoise epochs - C3')
# # plt.savefig(fig_path + files.split('/')[-2] + '_average_epoch_mastoids.png')
# # mne.viz.plot_compare_evokeds(dict(up_state=epochs_up_csd.average(method='mean', picks='C3'),
# #                                   down_state=epochs_down_csd.average(method='mean', picks='C3'),
# #                                   sham_state=epochs_sham_csd.average(method='mean', picks='C3')), 
# #                              legend='upper left', show_sensors='upper right',
# #                              title='Group level average pinknoise epochs - C3 (CSD)')
# # plt.savefig(fig_path + files.split('/')[-2] + '_average_epoch_csd.png')

# #%%
# ## TF/ITC plots 
# itc_data, Sxx = tfr_analysis(Data = epochs_up_mastoids, l_freq = 5, h_freq = 20, steps = 0.25, method = 'wavelet', 
#                               baseline=[-3, -1.25], mode='zlogratio', chan = 'C3', itc_calculation = None,
#                               plot=True, output='avg', zscore=False, cmap = 'Spectral_r', 
#                       save_path=None)

# itc_data, Sxx = tfr_analysis(Data = epochs_down_mastoids, l_freq = 5, h_freq = 20, steps = 0.25, method = 'wavelet', 
#                               baseline=[-3, -1.25], mode='zlogratio', chan = 'C3', itc_calculation = None,
#                               plot=True, output='avg', zscore=False, cmap = 'Spectral_r', 
#                       save_path=None)

# itc_data, Sxx = tfr_analysis(Data = epochs_sham_mastoids, l_freq = 5, h_freq = 20, steps = 0.25, method = 'wavelet', 
#                               baseline=[-3, -1.25], mode='zlogratio', chan = 'C3', itc_calculation = None,
#                               plot=True, output='avg', zscore=False, cmap = 'Spectral_r', 
#                       save_path=None)

# # freqs = np.arange(5, 25, 0.25)
# # print('creating new figure ...')
# # fig, ax = plt.subplots()
# # CM = ax.pcolormesh(np.linspace(-2.5, 2.5, max(np.shape(Sxx))), freqs, 
# #                Sxx, 
# #                shading='gouraud', cmap='Spectral_r', rasterized = True, 
# #                vmin=np.percentile(Sxx.min(),10),
# #                vmax=np.percentile(Sxx.max(),90))
# # ax.set_ylabel('Frequency [Hz]')
# # ax.set_xlabel('Time [sec]')
# # cbar = plt.colorbar(CM, ax=ax)
# # cbar.set_label(mode + ' (dB)', rotation=270)
# # # plt.savefig(save_path + '_tf_plot.png')
            
# ## ERPAC plots
# erpac = ERPAC(data = epochs_up_mastoids, f_pha=[0.3, 4], f_amp=(4, 20, .5, .5), n_perm=None, 
#       smooth=200, method = 'gc', edges=0.5, stationarity_t=False, plot=True, 
#       save_path=None)

# erpac = ERPAC(data = epochs_down_mastoids, f_pha=[0.3, 4], f_amp=(4, 20, .5, .5), n_perm=None, 
#       smooth=200, method = 'gc', edges=0.5, stationarity_t=False, plot=True, 
#       save_path=None)

# erpac = ERPAC(data = epochs_sham_mastoids, f_pha=[0.3, 4], f_amp=(4, 20, .5, .5), n_perm=None, 
#       smooth=200, method = 'gc', edges=0.5, stationarity_t=False, plot=True, 
#       save_path=None)

# #%%
# times = np.asarray([-0.5, 0, 0.25, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0])
# ts_args = dict(spatial_colors = True, gfp=True, time_unit='s')
# topomap_args = dict(outlines = 'head', time_unit='s', cmap = 'Spectral_r', time_format = "%0.2f s")
# epochs_up_mastoids.average().plot_joint(times = times, title='GAV - Up', 
#                                         ts_args=ts_args, topomap_args=topomap_args)