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
from sleepstim.sleep_funs import unravel_hypnogram_visbrain
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, _pre_process_sleep_data, preprocess_sleep_data, 
                                                     compare_hypnograms, check_match_data_hypno_elements,
                                                     label_artifacts, load_preprocessed_data, Data_SW, spectral_conn, 
                                                     art_detect, plot_ndPAC)
from sleepstim.sleep_funs import (load_xdf, channel_parser, bfr_butter_filt, thresholdcrossings)
from sleepstim.Analysis.pac import unit_root_test, add_stimulus_onset, ERPAC, extract_pha_amp, SO_spindle_coupling, imf_analytical_transform
from sleepstim.Analysis.time_frequency import tfr_analysis, analytical_transform
from sleepstim.Analysis.Resting_State.rs_preproc import (plot_psd, norm_wavelet_power, subject_cond_parser)
from sleepstim.Analysis.swa_decay import swa_decay, _decay_func
import tensorpac.methods as tpm
from sleepstim.Analysis.foof import oscillatory_plot_psd_map, periodic_fit
from fooof.plts.spectra import plot_spectrum
from sleepstim.Analysis.hilbert_huang import hilbert_huang_spectrum
from sleepstim.Analysis.utils import (coincidence_matrix, find_nearest, plot_cm, 
                                      get_mask, _coincidence, spectral_sorting_func_mne)
import neurokit2 as nk
from PCIst.PCIst import pci_st
import scipy
import pickle
from functools import reduce
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       permutation_cluster_1samp_test)
from sleepstim.Analysis.transitional_probabilities import (transition_matrix, 
                                                           transition_matrix_plot, 
                                                           transition_matrix_prob)
# sns.set_theme(color_codes=True) 
mne.set_log_level("CRITICAL")

#%%
def subject_night_parser(subject, cond, study_phase='sleep'):
    # data sheet with true subject condition nights
    sheet = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
    subj_cond = np.loadtxt(sheet, delimiter=',', dtype='str', skiprows=1) 
    cond_dict = {'sham' : 0, 'up' : 1, 'down' : 2} 
    # find condition by night
    if study_phase == 'sleep':
        index_name = list(subj_cond[:,0]).index(subject)
        # true condition night name
        condition_night = np.where(subj_cond[index_name, 1:].astype(int)==cond_dict[cond])[0][0] 
    
    return condition_night

def df_sw_sp_density(df, spindle=True):
    hypno_path1 = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/'
    hypno_path2 = '/media/administrator/data/Study_1_data/Pre-processed_data/EDFs/Auto_hypnograms/'     
    df = df.groupby(['Condition','Subject','Channel',
                     'Stage']).agg('count')['Start'].reset_index()
    df.rename(columns = {'Start':'Count'}, inplace = True)
    if spindle:
        df = df[np.logical_or(df['Stage']==2, df['Stage']==3)]
    df_ = pd.DataFrame(columns = ['Condition', 'Subject', 'Channel', 'Stage', 
                                  'Count', 'Density'])
    for i, subject in enumerate(df.Subject.unique()):
        df_s = df[df.Subject==subject]
        for cond in df_s.Condition.unique():
            df_s_c = df_s[df_s.Condition==cond]
            night = subject_night_parser(subject, cond, study_phase='sleep')
            try:
                hypno_pred_s = unravel_hypnogram_visbrain(hypno_path1 + f'{subject}_{night + 1}_hypno.txt')
            except:
                hypno_pred_s = np.load(hypno_path2 + f'{subject}_{night + 1}_hypnogram.npy')
            
            hypno_pred_s = hypno_pred_s[0:420]
            N2_counts = pd.Series(hypno_pred_s).value_counts()[2]
            N3_counts = pd.Series(hypno_pred_s).value_counts()[3]
            
            df_s_c['Density'] = np.ones(len(df_s_c))
            df_s_c['Density'].loc[df_s_c.Stage==2] = df_s_c.loc[df_s_c.Stage==2]['Count']/N2_counts
            df_s_c['Density'].loc[df_s_c.Stage==3] = df_s_c.loc[df_s_c.Stage==3]['Count']/N3_counts
            
            df_ = df_.append(df_s_c, ignore_index=True)
            
    return df_

def swa_dissipation_plot(sws_decay):
    sns.set_theme(color_codes=True) 
    pal = sns.color_palette("Blues_d", n_colors=len(sws_decay))
    for idx in range(len(sws_decay)):
        xdata = sws_decay['X-data'][idx]
        ydata = sws_decay['Y-data'][idx]
        popt = np.asarray([sws_decay['Asym'][idx],
                            sws_decay['Intercept'][idx],
                            sws_decay['Tau'][idx]])
 
        plt.plot(xdata, _decay_func(xdata, *popt), marker="o",
                  color=pal[idx]);
        #plt.plot(xdata, ydata, linewidth=0, marker="o");
        plt.ylim(ydata.min()-0.2, 1)
        plt.xlim(0, None)
        plt.xlabel("Time (hours)")
        plt.ylabel("Relative SWA (0.5-4 Hz) power")
        #plt.legend(frameon=True, loc="lower right");
        plt.title(f"{subject} - {cond} - Exponential decline", fontweight="bold")
                        
def find_cooccurring_rrpeaks(sws, spindles, lookaround=1.2):
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
    if len(mask) < 0:
        ax = sns.histplot(sws[sws.Channel=='C3']['DistanceSpindleToSW'], 
                          binwidth=0.1, stat='probability')
    else:
        ax = sns.histplot(sws.summary()[mask]['DistanceSpindleToSW'], 
                          binwidth=0.1, stat='probability')
    ax.set_xlim(-1.2,1.2)
    plt.axvline(0., lw=1.5, color='k', linestyle='--')
    ax2 = ax.twinx()
    sns.lineplot(x='Time', y='Amplitude',data=df_sync, ci=False, 
                 axes=ax2, linewidth=5, color='k')
    plt.tight_layout()
    if save:
        plt.savefig(fig_path + f'event_peri_hist_{subject}_{cond}')
        plt.close('all')
        
def group_co_occurence(sws):
    # ## Plot ridge plots --> push back 
    # sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # g = sns.FacetGrid(sws, row="Channel", col="Condition", aspect=9, height=1.2)
    # g.map_dataframe(sns.histplot, binwidth=0.1, stat='probability', 
    #                 x="DistanceSpindleToSW", fill=True, alpha=1)
    # g.map_dataframe(sns.kdeplot, x="DistanceSpindleToSW", fill=True, alpha=1)
    # g.fig.subplots_adjust(hspace=-.5)
    # g.set_titles("")
    # g.set(yticks=[])
    # g.despine(left=True)
    # plt.tight_layout()

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(23, rot=-.25, light=.7)
    g = sns.FacetGrid(sws, row="Channel", col="Condition", hue="Channel", 
                      aspect=15, height=.5, palette=pal)
    g.map_dataframe(sns.kdeplot, "DistanceSpindleToSW",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "DistanceSpindleToSW", clip_on=False, 
          color="w", lw=2, bw_adjust=.5)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    g.map(label, "DistanceSpindleToSW")
    
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)
    
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="", 
          #xticks=[], xlabel="",
          xticks=np.arange(-1.5, 2.0, 0.5), xlabel="",
          xlim = (-1.5, 1.5))
    g.despine(bottom=True, left=True)
                        
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
    
def auditory_stim_epochs(epoch_time = (-4, 4)):
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
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
                                     baseline=None, proj=False, reject=None)
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
            
    return df

def sw_spindle_detection_stim_auditory(save=True):
    # configure paths 
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    #report_path = '/media/administrator/data/Study_1_data/Data_tracking/[].csv'
    fig_path = '/media/administrator/data/Study_1_data/Figures/Pinknoise_epochs/'
    #results = defaultdict(lambda: [])
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
        try:
            Data = load_preprocessed_data(files)[0]
        except:
            Data = load_preprocessed_data(files)    
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
            else:
                pass
            # if epochs.times.shape[0]/epochs.info['sfreq']/60 <= 210:
            #     pass
            epochs.set_montage(mne.channels.make_standard_montage('standard_1005')) 
            epochs.interpolate_bads()
            # mastoid referencing
            epochs.set_eeg_reference(['M1','M2']) 
            # resample data
            epochs.resample(128)
            # create sudo hypno for now
            try:
                hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/' + subject + '_' + files.split('/')[-1].split('_')[1]
                hypno_pred_s =  unravel_hypnogram_visbrain(hypno_path + '_hypno.txt')
                hypno_pred = yasa.hypno_upsample_to_data(hypno_pred_s, sf_hypno = 1/30, 
                                                         sf_data = 128, data = epochs)
                print('Self-Scored Hypnogram Utilized! ')
            except:
                hypno_path = '/media/administrator/data/Study_1_data/Pre-processed_data/EDFs/Auto_hypnograms/' + subject + '_' + files.split('/')[-1].split('_')[1]  + '_hypnogram.npy'
                hypno_pred_s = np.load(hypno_path)
                hypno_pred = yasa.hypno_upsample_to_data(hypno = hypno_pred_s, data = epochs, 
                                                         sf_hypno = 1/30, sf_data = 128)
                print('U-Sleep Classfier Hypnogram Utilized! ')
            # artifact detection
            art_idx, _ = yasa.art_detect(epochs, hypno = hypno_pred, 
                                         window = 2)
            #art_idx = art_detect(epochs, raw=True)
            # The resolution of art is 2 seconds, so its sampling frequency is 1/2 (= 0.5 Hz)
            sf_art = 1 / 2
            art_up = yasa.hypno_upsample_to_data(art_idx, sf_art, epochs)
            # Add -1 to hypnogram where artifacts were detected
            hypno_with_art = hypno_pred.copy()
            hypno_with_art[art_up] = -1            
            # Proportion of each stage in ``hypno_with_art``
            pd.Series(hypno_with_art).value_counts(normalize=True).round(4)*100
            
            # sw detection
            try:
                sws = yasa.sw_detect(data = epochs.pick('eeg'), coupling = True, 
                                     remove_outliers = True, hypno = hypno_with_art, 
                                     coupling_params={"freq_sp": (12, 16), "time": 2, "p": 0.05}) 
            except:
                sws = yasa.sw_detect(data = epochs.pick('eeg'), coupling = False, 
                                     remove_outliers = True, hypno = hypno_with_art) 
            # spindle detection
            sp = yasa.spindles_detect(data = epochs.pick('eeg'), freq_sp=(12, 16), remove_outliers = True,
                                      duration=(0.5, 3), thresh={'rel_pow': None, 'corr': None, 'rms': 1.5},
                                      hypno = hypno_with_art)
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
  
def sleep_statistics_decay_stability(save=True):
    fig_path = '/media/administrator/data/Study_1_data/Figures/Full_night_sleep/'
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files if 'av.p' in i])
    sws_decay_ = pd.DataFrame(columns=['Subject','Condition','Channel','Intercept',
                                       'Asym','Tau','Decay', 'MAE', 'X-data', 'Y-data'])
    sleep_stats = pd.DataFrame()
    for i, files in enumerate(tqdm(maindir)):
        print(i, files.split('/')[-1])
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
                
            ## Data length based
            # create mne epoch object
            info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
            info['bads'] = Data.bad_chans        
            # create epochs with online reference
            epochs = mne.io.RawArray(Data.data.T/1e6, info)
            # cut data to length
            # if epochs.times.shape[0]/epochs.info['sfreq']/60 >= 210:
            #     #epochs.crop(tmin=0, tmax=210*60)
            if epochs.times.shape[0]/epochs.info['sfreq']/60 <= 210:
                pass
            epochs.set_montage(mne.channels.make_standard_montage('standard_1005')) 
            epochs.interpolate_bads()
            # mastoid referencing
            epochs.set_eeg_reference(['M1','M2']) 
            # resample data
            epochs.resample(128)
            # hypnograms
            try:
                hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/' + subject + '_' + files.split('/')[-1].split('_')[1]
                hypno_pred_s =  unravel_hypnogram_visbrain(hypno_path + '_hypno.txt')
                hypno_pred_s[0] = 0
                probs = transition_matrix_prob(transition_matrix(hypno_pred_s))
                transition_matrix_plot(probs, save=True, path=fig_path + subject + '_' + cond)
                plt.close('all')
                stability = np.diag(probs[2:, 2:]).mean().round(4)
                hypno_pred = yasa.hypno_upsample_to_data(hypno_pred_s, sf_hypno = 1/30, 
                                                         sf_data = 128, data = epochs)
                print('Self-Scored Hypnogram Utilized! ')
            except:
                hypno_path = '/media/administrator/data/Study_1_data/Pre-processed_data/EDFs/Auto_hypnograms/' + subject + '_' + files.split('/')[-1].split('_')[1]  + '_hypnogram.npy'
                hypno_pred_s = np.load(hypno_path)
                hypno_pred_s[0] = 0
                probs = transition_matrix_prob(transition_matrix(hypno_pred_s))
                transition_matrix_plot(probs, save=True, path=fig_path + subject + '_' + cond)
                plt.close('all')
                stability = np.diag(probs[2:, 2:]).mean().round(4)
                hypno_pred = yasa.hypno_upsample_to_data(hypno = hypno_pred_s, data = epochs, 
                                                         sf_hypno = 1/30, sf_data = 128)
                print('U-Sleep Classfier Hypnogram Utilized! ')

            # artifact detection
            art_idx, _ = yasa.art_detect(epochs, hypno = hypno_pred, 
                                         window = 2)
            # The resolution of art is 2 seconds, so its sampling frequency is 1/2 (= 0.5 Hz)
            sf_art = 1 / 2
            art_up = yasa.hypno_upsample_to_data(art_idx, sf_art, epochs)
            # Add -1 to hypnogram where artifacts were detected
            hypno_with_art = hypno_pred.copy()
            hypno_with_art[art_up] = -1            
            # Proportion of each stage in ``hypno_with_art``
            pd.Series(hypno_with_art).value_counts(normalize=True).round(4)*100
                    
            # sw decay analysis
            if i == 0:
                try:
                    sws_decay = swa_decay(data = epochs.copy().pick('eeg'),
                                          hypno = hypno_with_art,
                                          epoch_length="5min").reset_index()
                except:
                    sws_decay = swa_decay(data = epochs.copy().pick('eeg'),
                                          hypno = hypno_with_art,
                                          epoch_length="4min").reset_index()
                # plot here
                swa_dissipation_plot(sws_decay)
                plt.savefig(fig_path + subject + '_' + cond + '_swa_dissipation.png', dpi=100, bbox_inches='tight')
                plt.close('all')
                
                sws_decay.rename(columns = {'index':'Channel'}, inplace = True)
                sws_decay.insert(0, "Subject", subject)
                sws_decay.insert(1, "Condition", cond)
                sws_decay = sws_decay_.append(sws_decay, 1)
            else:
                try:
                    sws_decay2 = swa_decay(data = epochs.copy().pick('eeg'), 
                                           hypno = hypno_with_art,
                                           epoch_length="5min").reset_index()
                except:
                    sws_decay2 = swa_decay(data = epochs.copy().pick('eeg'), 
                                           hypno = hypno_with_art,
                                           epoch_length="4min").reset_index()
                
                # plot here
                swa_dissipation_plot(sws_decay2)
                plt.savefig(fig_path + subject + '_' + cond + '_swa_dissipation.png', dpi=100, bbox_inches='tight')
                plt.close('all')
                
                sws_decay2.rename(columns = {'index':'Channel'}, inplace = True)
                sws_decay2.insert(0, "Subject", subject)
                sws_decay2.insert(1, "Condition", cond)
                sws_decay = sws_decay.append(sws_decay2, 1)      
                       
            # plot spectrogram, hypnogram, and SWA dissipation 
            yasa.plot_spectrogram(
                data=epochs.get_data('eeg', units='uV').mean(0), 
                hypno=hypno_with_art, sf=128, cmap='Spectral_r')
            plt.tight_layout()
            plt.savefig(fig_path + subject + '_' + cond + '_hypnogram.png', dpi=100, bbox_inches='tight')
            plt.close('all')       
            
            # sleep stats --> use down sampled hypno
            stats_ = yasa.sleep_statistics(hypno_pred_s, sf_hyp=1/30)
            sleep_stats_ = {'Subject': subject,
                            'Condition': cond, 
                            'Stability': stability}
            sleep_stats_.update(stats_)
            # update dataframe 
            sleep_stats = sleep_stats.append(sleep_stats_, ignore_index=True)
            
            del Data, epochs

    if save:
        save_path = '/media/administrator/data/Study_1_data/Statistics/Sleep/'
        sws_decay.to_csv(save_path + 'swa_decay.csv')
        sleep_stats.to_csv(save_path + 'sleep_stage.csv')
        
    return sws_decay, sleep_stats
        
def auditory_stim_results():
    # configure paths
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    report_path = '/media/administrator/data/Study_1_data/Statistics/'
    fig_path = '/media/administrator/data/Study_1_data/Figures/Group_pinknoise_epochs/'
    # initialize dictionary for results
    results = defaultdict(lambda: [])
    results_pci = defaultdict(lambda: [])
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
            
        if reference == 'Linked Mastoids':
            # load data & create evoked object
            evk = mne.read_epochs(file, preload=True).apply_baseline((-4, -1.5)).average().resample(128)
            
            # compute subject wise PCI
            global_pci = pci_st.calc_PCIst(signal_evk = evk.get_data('eeg', units='uV'), 
                                           times = evk.times*1000, 
                                           full_return=False, **par)
            
            # frontal contralateral pci
            FCContra = pci_st.calc_PCIst(signal_evk = evk.get_data(['F3','Fz'], 
                                                                   units='uV'),
                                             times = evk.times*1000, 
                                             full_return=False, **par)
        
            # frontal ipsilateral pci
            FCIpsi = pci_st.calc_PCIst(signal_evk = evk.get_data(['F4','Fz'],
                                                                   units='uV'),
                                           times = evk.times*1000, 
                                           full_return=False, **par)
                        
            # contalateral motor cortex pci
            MCContra = pci_st.calc_PCIst(signal_evk = evk.get_data(['C3','Cz'], 
                                                                   units='uV'), 
                                           times = evk.times*1000, 
                                           full_return=False, **par)
        
            # ipsilateral motor cortex pci
            MCIpsi = pci_st.calc_PCIst(signal_evk = evk.get_data(['C4','Cz'],
                                                                   units='uV'),
                                         times = evk.times*1000, 
                                         full_return=False, **par)
            
            # contralateral parietal cortex pci 
            PCContra = pci_st.calc_PCIst(signal_evk = evk.get_data(['P3','Pz'],
                                                                       units='uV'),
                                             times = evk.times*1000, 
                                             full_return=False, **par)
            
            # ipsilateral parietal cortex pci
            PCIpsi = pci_st.calc_PCIst(signal_evk = evk.get_data(['P4','Pz'],
                                                                     units='uV'),
                                           times = evk.times*1000, 
                                           full_return=False, **par)
    
            # update results dictionary
            results['Subject'].append(subject)
            results['Condition'].append(cond)
            results['Reference'].append(reference)
            results['Global_PCI'].append(global_pci)
            results['Data'].append(evk)
        
            # do local 
            for area in (['FC','MC','PC']):
                for laterality in (['Contra','Ipsi']):
                    results_pci['Subject'].append(subject)
                    results_pci['Condition'].append(cond)
                    results_pci['Brain_area'].append(area)
                    results_pci['Hemisphere'].append(laterality)
                    results_pci['PCI'].append(eval(area + laterality))
                     
        else:
            pass
        
    # convert results dict to dataframe
    df = pd.DataFrame(results)
    df_pci = pd.DataFrame(results_pci)
    pickle.dump(df, open(report_path + 'Sleep_evk_pci_global.p',"wb"))
    pickle.dump(df_pci, open(report_path + 'Sleep_evk_pci_local.p',"wb"))
    
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
    
    ## Plotting fun
    ts_args = dict(spatial_colors = True, gfp=True, time_unit='s')
    topomap_args = dict(outlines = 'head', time_unit='s', cmap = 'Spectral_r', time_format = "%0.2f s")
    times = np.asarray([-0.5, 0, 0.25, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0])
    all_times = np.arange(-3.0, 3.25, 0.25)
    results_gav = defaultdict(lambda: [])
    gav_down_comp = mne.grand_average(list(comp_df.copy().drop(comp_df.loc[comp_df['Evoked Difference Down']==0].index)
                                           ['Evoked Difference Down']))
    gav_down_comp.plot_joint(times, ts_args = ts_args, 
                             topomap_args = topomap_args,
                             title = 'Down Evoked Difference')
    plt.savefig(fig_path + 'down_erp_diff.png')
    gav_up_comp = mne.grand_average(list(comp_df.copy().drop(comp_df.loc[comp_df['Evoked Difference Up']==0].index)
                                         ['Evoked Difference Up']))
    gav_up_comp.plot_joint(times, ts_args = ts_args, 
                           topomap_args = topomap_args,
                           title = 'Up Evoked Difference') 
    plt.savefig(fig_path + 'up_erp_diff.png')
    
    for (cond, ref), evok in df.groupby(['Condition', 'Reference'])['Data']:
        print(cond, ref)
        gav = mne.grand_average(list(evok))
        gav.plot_joint(times, ts_args = ts_args, topomap_args = topomap_args,
                       title = cond.capitalize() + ' ' + ref.capitalize())     
        plt.savefig(fig_path + f'{cond} {ref} ERP.png')
        gav.plot_topomap(all_times, ch_type='eeg', ncols='auto', nrows='auto',
                          title=f'Amplitude distributions - {cond.capitalize()} - {ref.capitalize()}',
                          **topomap_args) 
        plt.savefig(fig_path + f'{cond} {ref} Topography.png')
        
        results_gav['Condition'].append(cond)
        results_gav['Reference'].append(ref)
        results_gav['Data'].append(gav)
    
    plt.close('all')
        
    # convert results dict to dataframe
    df_gav = pd.DataFrame(results_gav)
    pickle.dump(df_gav, open(report_path + 'Sleep_gav_comp.p',"wb"))
 
def auditory_phase_analysis(plot=True, save=True):
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/'
    sns.set_theme(style="darkgrid")
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
                          itc_calculation=None, plot=True, length=[-3,3], 
                          cmap = 'Spectral_r', save_path=fig_path + subject,
                          inst_power=None) 

    return Sxx
   
def tfr_sleep_analysis(picks='eeg'):
    sns.set_theme(color_codes=True) 
    # configure paths
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/ERPs/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    report_path = '/media/administrator/data/Study_1_data/Statistics/'
    fig_path = '/media/administrator/data/Study_1_data/Figures/Pinknoise_epochs/'
    # initialize dictionary for results
    results_ip = defaultdict(lambda: [])
    results_tfr = defaultdict(lambda: [])
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
            
            # Instantaneous power
            for bands, freqs in zip(['Delta', 'Spindle'], [(0.5, 4), (9, 16)]):
                _, _, amp = analytical_transform(epoch, picks='eeg', 
                                                 freqs=freqs, method='imf_hilbert')
                time_window1 = epoch.time_as_index([0, 1.075])
                time_window2 = epoch.time_as_index([1.075, 2.150])
                time_window3 = epoch.time_as_index([-3, 0])
                time_window4 = epoch.time_as_index([0, 3])
                if picks=='eeg':
                    for idx, chan in enumerate(epoch.copy().pick('eeg').ch_names):
                        # log subject/trial info
                        results_ip['Subject'].extend([subject])
                        results_ip['Condition'].extend([cond])  
                        amp_mean = amp.mean(0)[idx].squeeze()
                        results_ip['Channel'].extend([chan])
                        results_ip['Band'].extend([bands])
                        results_ip['IPI'].append(amp_mean[time_window1[0]:time_window1[1]].mean(0))
                        results_ip['Post2'].append(amp_mean[time_window2[0]:time_window2[1]].mean(0))
                        results_ip['Pre'].append(amp_mean[time_window3[0]:time_window3[1]].mean(0))
                        results_ip['Post'].append(amp_mean[time_window4[0]:time_window4[1]].mean(0))
                        results_ip['Epoch_IP'].append([amp_mean[time_window3[0]:time_window4[-1]]])
                else:
                    # log subject/trial info
                    results_ip['Subject'].extend([subject])
                    results_ip['Condition'].extend([cond])
                    amp_mean = amp.mean(0).squeeze()
                    results_ip['Band'].extend([bands])
                    results_ip['IPI'].append(amp_mean[time_window1[0]:time_window1[1]].mean(0))
                    results_ip['Post2'].append(amp_mean[time_window2[0]:time_window2[1]].mean(0))
                    results_ip['Pre'].append(amp_mean[time_window3[0]:time_window3[1]].mean(0))
                    results_ip['Post'].append(amp_mean[time_window4[0]:time_window4[1]].mean(0))
                    results_ip['Epoch_IP'].append([amp_mean[time_window3[0]:time_window4[-1]]])
         
            # TF analysis 
            _, Sxx = tfr_analysis(Data=epoch, l_freq=5, h_freq=25, steps=0.25, 
                                  method = 'wavelet', baseline=[-3, 3], chan = 'all', 
                                  itc_calculation=None, plot=True, length=[-3,3], 
                                  cmap = 'Spectral_r', save_path=fig_path + subject + '_' + cond,
                                  inst_power=True)
            results_tfr['Subject'].extend([subject])
            results_tfr['Condition'].extend([cond])
            results_tfr['TFR_avg'].append(Sxx.average())
            results_tfr['Evoked'].append(epoch.average())
            
            plt.close('all')

    return pd.DataFrame(results_tfr), pd.DataFrame(results_ip)

def group_pac_plot(df, chan='C3', bins=12):
    #df = df.groupby(['Channel','Condition']).mean().reset_index()
    df = df.reset_index()
    kwargs_arrow={'fc': 'tab:red', 'ec': 'tab:red'}
    with plt.style.context('seaborn'):
        sns.set_palette("flare")
        sns.set_theme(color_codes=True)
        #df[df.Channel==chan]
        g = sns.FacetGrid(df, col='Condition', hue='Condition',
                          subplot_kws=dict(projection='polar'), height=4.5,
                          sharex=True, sharey=True, despine=False, ylim=[0, 0.75])
        g.map_dataframe(sns.histplot, f'PhaseAtSigmaPeak', 
                        stat='density', bins=bins)
        for ax, con in zip(range(len(g.axes[0])), df.Condition.unique()):
            rv = pg.circ_r(df[df.Condition==con][f'PhaseAtSigmaPeak'])
            phi = pg.circ_mean(df[df.Condition==con][f'PhaseAtSigmaPeak'])
            circ_std = scipy.stats.circstd(df[df.Condition==con][f'PhaseAtSigmaPeak'], nan_policy='omit')
            z, pval = pg.circ_rayleigh(df[df.Condition==con][f'PhaseAtSigmaPeak'])
            g.axes[0, ax].arrow(0, 0, phi, rv, **kwargs_arrow)       
            plt.tight_layout()
            print(f'The circular mean for {con.upper()} is {phi.round(4)} with a stand deviation of {circ_std.round(4)}, and a resultant vector length of : {rv.round(4)}')
            print(f'The Rayleigh Z-statistic for non-uniformity of circular data for {con.upper()} : {z.round(4)} with a p-value : {pval.round(4)}')
        plt.show()
            
def group_so_sp_analysis():
    log_path = '/media/administrator/data/Study_1_data/Statistics/SW_spindle_summary/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(log_path) for i in files])
    fig_path = '/media/administrator/data/Study_1_data/Figures/Group_half_night/'
    df_sync, sws, sp = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i, files in enumerate(tqdm(maindir)):
        if 'half_co_mask_' in files:
            df_sync = pd.concat([df_sync,pd.read_csv(files, index_col=0)])
        elif 'half_sw_' in files:
            sws = pd.concat([sws,pd.read_csv(files, index_col=0)])
        elif 'half_sp_' in files:
            sp = pd.concat([sp,pd.read_csv(files, index_col=0)])
            
    ## SW/Spindle density analysis
    sws_density = df_sw_sp_density(sws, spindle=False)
    sws_density['Density'] = sws_density['Density'].astype(float)
    sp_density = df_sw_sp_density(sp, spindle=True)  
    sp_density['Density'] = sp_density['Density'].astype(float)
    # save
    stat_path = '/media/administrator/data/Study_1_data/Statistics/SW_spindle_summary/'
    sws_density.to_csv(stat_path + 'sws_density.csv')
    sp_density.to_csv(stat_path + 'sp_density.csv')
    sws.to_csv(stat_path + 'sws_full.csv')
    sp.to_csv(stat_path + 'sp_full.csv')
    df_sync.to_csv(stat_path + 'df_sync_full.csv')
    
    ## Coincidence matrices for spindles and SOs- Group level
    cm_sp = coincidence_matrix(sp, sf=128, plot=True, 
                               window='group', standardize=False)
    plt.suptitle('Spindle Coincidence Matrix', fontsize=16)
    plt.savefig(fig_path + 'group_coincidence_matrix_sp.png')
    cm_sw = coincidence_matrix(sws, sf=128, plot=True, 
                               window='group', standardize=False)
    plt.suptitle('Slow Wave Coincidence Matrix', fontsize=16)
    plt.savefig(fig_path + 'group_coincidence_matrix_sw.png')
    
    ## Plot ndPAC- Group level
    plt.figure()
    sns.violinplot(data = sws, x='Channel', y='ndPAC', 
                   hue='Condition', cut=0)
    plt.savefig(fig_path + '_ndPAC.png')
    print(f'ndPAC means: {sws.groupby("Channel")["ndPAC"].mean()}')
        
    group_pac_plot(sws, bins=12)
    plt.savefig(fig_path + 'group_circular_phase.png')


    ## Plot group co-occurence 
    plot_co_occurence(sws, mask, df_sync, fig_path, 
                      cond, save=True)
    
    ## stats
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    
    df = sws.reset_index(drop=True)
    md = smf.mixedlm("ndPAC ~ Condition", data=df, groups=df["Subject"], 
                     missing='drop')
    mdf = md.fit()
    print(mdf.summary())
    
    #
    new_df = df.groupby(['Subject','Condition']).mean()['ndPAC'].reset_index()
    slalom_df = pd.read_csv(r'/media/administrator/data/Study_1_data/Statistics/Slalom/Slalom_results.csv')
    slalom_df = slalom_df.groupby(["Subject","Condition"]).mean().reset_index()
    df_comb = pd.merge(new_df, slalom_df, on=["Subject","Condition"]) 

    # drop the following subjects 
    for i in zip(['9PJZ8Z8F','475MQ9BL','5LNKD1MG','6QF3HOJC','IBYYXKMB','NW2JV7YA']):
        df_comb.drop(df_comb.loc[df_comb['Subject']==i[0]].index, inplace=True)
    
    # stats 
    md = smf.mixedlm("Absolute_RMSE_difference ~ ndPAC + Condition", data=df_comb, groups=df_comb["Subject"], 
                     missing='drop')
    mdf = md.fit()
    print(mdf.summary())          

#%%
# tfr_df, ip_df = tfr_sleep_analysis()
# save_path = '/media/administrator/data/Study_1_data/Statistics/Sleep/'
# pickle.dump(tfr_df, open(save_path + 'TFR.p', "wb"))
# pickle.dump(ip_df,  open(save_path + 'Inst_power.p', "wb"))

# load_path = '/media/administrator/data/Study_1_data/Statistics/Sleep/TFR.p'
# tfr_df = pickle.load(open(load_path, "rb"))

def plot_tfr_contrast(tfr_df, length=[-3.05, 3.05]):
    double_contrast_df = pd.DataFrame(columns=list(tfr_df.columns)[1:])
    for contrast in zip(['up', 'down'], ['sham', 'sham down']):
        print(contrast)
        df = tfr_df.query(f"Condition == '{contrast[0]}' or Condition == '{contrast[1]}'")
        good_subs = []
        for sub in df.Subject.unique():
            count = len(df[df.Subject==sub])
            stimuli = [df[df.Subject==sub].reset_index()['Evoked'][i].nave for i in range(len(df[df.Subject==sub]))]
            t_stimuli = [i for i in stimuli if i >= 30]
            temp_data = (df[df.Subject==sub].reset_index()\
                         ['Evoked'][0].copy().pick('eeg').data*1e6).squeeze()
            if count == 2 and len(t_stimuli) == 2: # and np.ptp(temp_data) > 30:
                good_subs.append(sub)
                common_df = df.loc[df['Subject'].isin(good_subs)]
            else:
                print(f'Excluded Subjects: {sub}')
                
        inst_stim = common_df.copy()[common_df['Condition']==contrast[0]].reset_index()
        tfr_stim = mne.grand_average([inst_stim['TFR_avg'][i].copy().drop_channels(['M1','M2']) for i in range(len(inst_stim))])
        gav_stim = mne.grand_average([inst_stim['Evoked'][i].copy().drop_channels(['M1','M2']) for i in range(len(inst_stim))])
        Sxx_stim = tfr_stim.copy().pick('eeg').crop(tmin=length[0],tmax=length[1]).data.mean(0) #mean over channels
       
        inst_sham = common_df.copy()[common_df['Condition']==contrast[1]].reset_index()
        tfr_sham = mne.grand_average([inst_sham['TFR_avg'][i].copy().drop_channels(['M1','M2']) for i in range(len(inst_sham))])
        gav_sham = mne.grand_average([inst_sham['Evoked'][i].copy().drop_channels(['M1','M2']) for i in range(len(inst_sham))])
        Sxx_sham = tfr_sham.copy().pick('eeg').crop(tmin=length[0],tmax=length[1]).data.mean(0) #mean over channels
       
        Sxx_ = mne.combine_evoked([tfr_stim.copy().pick('eeg').crop(tmin=length[0],tmax=length[1]),
                                   tfr_sham.copy().pick('eeg').crop(tmin=length[0],tmax=length[1])],
                                   weights=[1, -1])
        gav_ = mne.combine_evoked([gav_stim.copy().pick('eeg').crop(tmin=length[0],tmax=length[1]),
                                   gav_sham.copy().pick('eeg').crop(tmin=length[0],tmax=length[1])], 
                                  weights = [1, -1])
        times = tfr_sham.copy().crop(tmin=length[0],tmax=length[1]).times
           
        fig, axs = plt.subplots(2, figsize=(10, 8))
        axs[0].set_title("\u0394" + f' Average TFR Plot - {contrast[0].capitalize()}')
        vmin=-.1; vmax=.6
        # vmin, vmax = np.percentile(Sxx_, [0 + 0.1, 100 - 0.1])
        # norm = Normalize(vmin=vmin, vmax=vmax)
        CM = axs[0].pcolormesh(times, tfr_stim.freqs, Sxx_.data.mean(0).squeeze(), shading='gouraud', 
                               cmap='Spectral_r', rasterized = True, #norm = norm, 
                               vmin=vmin, vmax=vmax, antialiased=True)
        axs[0].set_ylabel('Frequency (Hz)')
        axs[1].set_title("\u0394" + f' Average Evoked Response (uV)')
        axs[1].plot(gav_.times, gav_.data.T*1e6)
        axs[1].set_xlim([length[0], length[1]])
        axs[1].set_ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        # if save_path is not None:
        #     plt.savefig(save_path + '_tf_plot.png')

        ## Combine tfr contrasts for double contrast
        new_row = {'Condition': contrast[0],
                   'TFR_avg' : Sxx_.data.squeeze(),
                   'Evoked' : gav_.data.squeeze()*1e6}
        double_contrast_df = double_contrast_df.append(new_row, 
                                                       ignore_index=True)
    
    gav = (double_contrast_df['Evoked'][0] - double_contrast_df['Evoked'][1])
    Sxx = (double_contrast_df['TFR_avg'][0] - double_contrast_df['TFR_avg'][1])
    
    fig, axs = plt.subplots(2, figsize=(10, 8))
    axs[0].set_title("\u0394" + ' Average TFR Plot - Double Contrast')
    vmin=-.1; vmax=.6
    # vmin, vmax = np.percentile(Sxx_, [0 + 0.1, 100 - 0.1])
    # norm = Normalize(vmin=vmin, vmax=vmax)
    CM = axs[0].pcolormesh(times, tfr_stim.freqs, Sxx.mean(0), shading='gouraud', 
                           cmap='Spectral_r', rasterized = True, #norm = norm, 
                           vmin=vmin, vmax=vmax, antialiased=True)
    axs[0].set_ylabel('Frequency (Hz)')
    axs[1].set_title("\u0394" + f' Average Evoked Response (uV)')
    axs[1].plot(gav_.times, gav.T)
    axs[1].set_xlim([length[0], length[1]])
    axs[1].set_ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.tight_layout()
               
def plot_tfr_erp(tfr_df):
    for cond in tfr_df.Condition.unique():
        print(cond)
        inst = tfr_df.copy()[tfr_df['Condition']==cond].reset_index()
        tfr = mne.grand_average([inst['TFR_avg'][i] for i in range(len(inst))])
        gav = mne.grand_average([inst['Evoked'][i] for i in range(len(inst))])
        
        length=[-3.05, 3.05]
        Sxx_ = tfr.copy().pick('eeg').drop_channels(['M1','M2']).crop(tmin=length[0],tmax=length[1]).data.mean(0) #mean over channels
        times = tfr.copy().crop(tmin=length[0],tmax=length[1]).times  
        
        fig, axs = plt.subplots(2, figsize=(10, 8))
        axs[0].set_title(f' Average TFR Plot - {cond.capitalize()}')
        vmin=-.1; vmax=.6
        # vmin, vmax = np.percentile(Sxx_, [0 + 0.1, 100 - 0.1])
        # norm = Normalize(vmin=vmin, vmax=vmax)
        CM = axs[0].pcolormesh(times, tfr.freqs, Sxx_, shading='gouraud', 
                               cmap='Spectral_r', rasterized = True, #norm = norm, 
                               antialiased=True, vmin=vmin, vmax=vmax)
        axs[0].set_ylabel('Frequency (Hz)')
        axs[1].set_title(f'Average Evoked Response (uV)')
        t_min = (np.abs(gav.times - length[0])).argmin()
        t_max = (np.abs(gav.times - length[1])).argmin()
        axs[1].plot(gav.times[t_min:t_max+1], 
                    gav.copy().pick('eeg').data[:,t_min:t_max+1].T*1e6)
        axs[1].set_xlim([length[0], length[1]])
        axs[1].set_ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        
#%%

def tfr_stats(tfr_df, length=[-3.05, 3.05], pick='eeg'):   
    common_subs = []
    for cond_df in tfr_df.groupby('Condition')['Subject']:
        common_subs.append(list(cond_df[1]))
    to_del = reduce(np.setxor1d, [common_subs[0], common_subs[1],
                                  common_subs[2], common_subs[3]])
    to_del[-1] = ('P289L2RH')
    [tfr_df.drop(tfr_df.loc[tfr_df['Subject']==to_del[i]].index,
                 inplace=True) for i in range(len(to_del))]
    contrast_ = [] 
    for contrast in zip(['up', 'down'], ['sham', 'sham down']):
        print(contrast)
        df = tfr_df.query(f"Condition == '{contrast[0]}' or Condition == '{contrast[1]}'")
        good_subs = []
        for sub in df.Subject.unique():
            count = len(df[df.Subject==sub])
            stimuli = [df[df.Subject==sub].reset_index()['Evoked'][i].nave for i in range(len(df[df.Subject==sub]))]
            t_stimuli = [i for i in stimuli if i >= 30]
            temp_data = (df[df.Subject==sub].reset_index()\
                         ['Evoked'][0].copy().pick('eeg').data*1e6).squeeze()
            if count == 2 and len(t_stimuli) == 2: # and np.ptp(temp_data) > 30:
                good_subs.append(sub)
                common_df = df.loc[df['Subject'].isin(good_subs)]
                # print(f'Included Subjects: {sub}')
            else:
                continue
                #print(f'Excluded Subjects: {sub}')
                
        inst_stim = common_df.copy()[common_df['Condition']==contrast[0]].reset_index()
        tfr_stim = list([inst_stim['TFR_avg'][i].copy().drop_channels(['M1','M2']).crop(tmin=length[0],tmax=length[1]).pick(pick) 
                         for i in range(len(inst_stim))])
        Sxx_stim_ = np.concatenate([np.expand_dims(tfr_stim[i].copy().data, 0)
                                   for i in range(len(tfr_stim))])
        # subj x time x freq x chan
        Sxx_stim = np.moveaxis(Sxx_stim_, 1, 3) 
       
        inst_sham = common_df.copy()[common_df['Condition']==contrast[1]].reset_index()
        tfr_sham = list([inst_sham['TFR_avg'][i].copy().drop_channels(['M1','M2']).crop(tmin=length[0],tmax=length[1]).pick(pick) 
                         for i in range(len(inst_stim))])
        Sxx_sham_ = np.concatenate([np.expand_dims(tfr_sham[i].copy().data, 0)
                                   for i in range(len(tfr_sham))])
        # subj x time x freq x chan
        Sxx_sham = np.moveaxis(Sxx_sham_, 1, 3) 
    
        times = tfr_sham[0].copy().crop(tmin=length[0],tmax=length[1]).times
        
        # contrast 
        contrast_.append(Sxx_stim - Sxx_sham)
        
    # adjust contrast  
    contrast_ = (contrast_[0] - contrast_[1])
    
    # prepare adjacency matrix
    adjacency, ch_names = mne.channels.find_ch_adjacency(df['Evoked'][0].copy().drop_channels(['M1','M2']).info, 
                                                         'eeg')
    adj4D = mne.stats.combine_adjacency(contrast_.shape[1],contrast_.shape[2],
                                        adjacency)
    
    # compute threshold
    pval = 0.05  # arbitrary
    df = contrast_.shape[0] - 1  # degrees of freedom for the test
    thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
         
    # do stats 
    t_obs, clusters, cluster_p_values, h0 = spatio_temporal_cluster_1samp_test(
        contrast_,                          # numpy array for contrast [n_subjects, n_voltage, n_channels]
        n_permutations=1024,                # 1000 is the minimum
        threshold=thresh,                   # TFCE, starting at 0, in 0.2 steps (in t-values)
        tail=0,                             # two-tailed test (1 or -1 for one-tailed)
        n_jobs=-1,                          # increase value to speed up computations
        adjacency=None, #adj4D              # sparse matrix for channel adjacency as computed above
        buffer_size=None,
        out_type='mask',                    # returns a mask map instead of indices of sig. points
        seed= 1503
        ) 
    
    # return mask
    mask = cluster_p_values <= 0.05
    
    # get vars
    freqs = inst_stim['TFR_avg'][0].freqs
    times = inst_stim['TFR_avg'][0].copy().crop(length[0], length[1]).times
    
    # plot stats
    if np.mean(mask) == 0:
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(1, figsize=(10, 8))
        CM = ax.pcolormesh(times, freqs, t_obs.mean(-1), 
                           shading='gouraud', cmap='Spectral_r', rasterized = True, 
                           antialiased=True, vmin=-4, vmax=4)
        cbar = plt.colorbar(CM, ax=ax)
        cbar.set_label('T-Values', rotation=270)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        plt.tight_layout()
    
    # 
    else:
        fig, ax = plt.subplots(1, figsize=(10, 8))
        CM = ax.pcolormesh(times, freqs, t_obs.mean(-1), 
                           shading='gouraud', cmap='Spectral_r', rasterized = True, 
                           antialiased=True, vmin=-4, vmax=4, alpha=1)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        cbar = plt.colorbar(CM, ax=ax)
        cbar.set_label('T-Values', rotation=270)
        CM2 = ax.pcolormesh(times, freqs, np.ma.masked_where(clusters[np.argmax(mask)].squeeze(),
                                                             clusters[np.argmax(mask)].squeeze()),
                             shading='gouraud', cmap='Spectral_r', rasterized = True,
                             antialiased=True, vmin=-4, vmax=4, alpha=0.25)
        plt.tight_layout()

    
    return t_obs, clusters, cluster_p_values, h0, mask, times, freqs 

def erp_stats(tfr_df, length=[-3, 3]):   
    common_subs = []
    for cond_df in tfr_df.groupby('Condition')['Subject']:
        common_subs.append(list(cond_df[1]))
    to_del = reduce(np.setxor1d, [common_subs[0], common_subs[1],
                                  common_subs[2], common_subs[3]])
    to_del[-1] = ('P289L2RH')
    [tfr_df.drop(tfr_df.loc[tfr_df['Subject']==to_del[i]].index,
                 inplace=True) for i in range(len(to_del))]
    contrast_ = [] 
    gavs = []
    for contrast in zip(['up', 'down'], ['sham', 'sham down']):
        print(contrast)
        df = tfr_df.query(f"Condition == '{contrast[0]}' or Condition == '{contrast[1]}'")
        good_subs = []
        for sub in df.Subject.unique():
            count = len(df[df.Subject==sub])
            stimuli = [df[df.Subject==sub].reset_index()['Evoked'][i].nave for i in range(len(df[df.Subject==sub]))]
            t_stimuli = [i for i in stimuli if i >= 30]
            temp_data = (df[df.Subject==sub].reset_index()\
                         ['Evoked'][0].copy().pick('eeg').data*1e6).squeeze()
            if count == 2 and len(t_stimuli) == 2: # and np.ptp(temp_data) > 30:
                good_subs.append(sub)
                common_df = df.loc[df['Subject'].isin(good_subs)]
                print(f'Included Subjects: {sub}')
            else:
                print(f'Excluded Subjects: {sub}')
                
        inst_stim = common_df.copy()[common_df['Condition']==contrast[0]].reset_index()
        inst_sham = common_df.copy()[common_df['Condition']==contrast[1]].reset_index()

        # calculate difference        
        diff_waves = []
        for i in range(len(inst_sham)):
            diff_waves.append(mne.combine_evoked([inst_stim['Evoked'][i].copy().drop_channels(['M1','M2']).crop(tmin=length[0],tmax=length[1]), 
                                                  inst_sham['Evoked'][i].copy().drop_channels(['M1','M2']).crop(tmin=length[0],tmax=length[1])],
                                                 weights=[1, -1]))
        
        # gav for plotting
        gav = mne.grand_average(diff_waves).copy().crop(tmin=length[0],
                                                        tmax=length[1])
        gavs.append(diff_waves)
        
        # calculate contrast for all subjects
        contrast_sub = np.concatenate([np.expand_dims(diff_waves[i].copy().data*1e6, 0) 
                                       for i in range(len(diff_waves))], 0)

        # subj x timepoints x chan
        contrast_sub = np.moveaxis(contrast_sub, 1, 2)
       
        # times
        times = gav.copy().crop(tmin=length[0],tmax=length[1]).times
        
        # contrast 
        contrast_.append(contrast_sub)
        
    # final gav
    gav_f = mne.grand_average(gavs[0], gavs[1])
     
    # adjust contrast   
    contrast_ = (contrast_[0] - contrast_[1])
    
    # prepare adjacency matrix
    adjacency, ch_names = mne.channels.find_ch_adjacency(gav.info, 'eeg')
    
    # compute threshold
    pval = 0.05  # arbitrary
    df = contrast_.shape[0] - 1  # degrees of freedom for the test
    thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
         
    # do stats 
    t_obs, clusters, cluster_p_values, h0 = spatio_temporal_cluster_1samp_test(
        contrast_,                          # numpy array for contrast [n_subjects, n_voltage, n_channels]
        n_permutations=1024,                # 1000 is the minimum
        threshold=thresh,                   # TFCE, starting at 0, in 0.2 steps (in t-values)
        tail=0,                             # two-tailed test (1 or -1 for one-tailed)
        n_jobs=-1,                          # increase value to speed up computations
        adjacency=adjacency,                # sparse matrix for channel adjacency as computed above
        buffer_size=None,
        out_type='mask',                    # returns a mask map instead of indices of sig. points
        seed= 1503
        ) 
    
    # return mask
    mask = cluster_p_values <= 0.05
    
    # plot
    gav_f.plot_image(mask=clusters[np.argmax(mask)].T, 
                   show_names="all", mask_cmap='Blues')
    
    return t_obs, clusters, cluster_p_values, h0, mask 

#%%
# from pymer4.models import Lmer
# import pandas as pd
# import numpy as np

# save_path = '/media/administrator/data/Study_1_data/Statistics/Sleep/'
# df = pd.read_csv(save_path + 'swa_decay.csv', index_col=0)

# good_subs = ['0RCB4IRJ', '3LFLTILW', '6QJ3ITMT', '7XVWEVOK', '886MCPKG', 'A4VCLD2I', 
#              'CWESJCNJ', 'D1BOI2AY', 'FUOPOVNF', 'HTXEYPW6', 'IYPJJ2KE', 'KEQB5AWM', 
#              'RVQL2MRD', 'UDLD86TO', 'W6AX3IMN', 'Y9VJUA9F', 'YIOYSRPX']

# # remove mne objects for stats
# df2 = df.loc[np.logical_and(df["Channel"] != 'M1', 
#                             df['Channel'] != 'M2')]
# df2 = df2.loc[df2['Subject'].isin(good_subs)]
# idx = list([0,1,2,6])
# df2 = df2[df2.columns[idx]]

# df2 = df2[df2['Decay'] < 20]

# model = Lmer(f"Decay ~ Condition*Channel + (Condition|Subject)",
#              data=df2, family='gaussian')
    
# # Using dummy-coding; suppress summary output
# model.fit(factors={"Condition": ["sham", "up", "down"],
#                    "Channel" : list(df2.Channel.unique())}, 
#           ordered=True, summarize=True)

# # Get ANOVA table, but this time force orthogonality for valid SS III inferences
# # In this case the data are balanced so nothing changes
# print(model.anova(force_orthogonal=True))

# ## Post-hoc tests 
# marginal_estimates, comparisons = model.post_hoc(p_adjust="fdr",
#                                                  marginal_vars='Condition',
#                                                  grouping_vars=None)

# print(marginal_estimates);
# print(comparisons);


# #%%
# load_path = '/media/administrator/data/Study_1_data/Statistics/Sleep/TFR.p'
# tfr_df = pickle.load(open(load_path, "rb"))
# info = tfr_df['Evoked'][0].copy().pick('eeg').drop_channels(['M1','M2']).info
# def plot_sw_decay(df2):
#     sns.set_theme(style="darkgrid")
#     fig, axes = plt.subplots(1, 3, figsize=(20, 20))
#     for idx, cond in  enumerate(np.unique(df2.reset_index().Condition)):
#         # Create a topomap for the current oscillation band
#         decay = df2[df2.Condition == cond].groupby(['Channel']).mean()['Decay']
#         im, cn = mne.viz.plot_topomap(decay, info, cmap='Spectral_r', contours=1, axes=axes[idx], 
#                                       show=False, names=df2.Channel.unique(), show_names=True,
#                                       vmin=np.percentile(df2['Decay'], 5), vmax=np.percentile(df2['Decay'], 95))
#         # add color bar
#         mne.viz.topomap._add_colorbar(axes[idx], im, cmap = 'Spectral_r', side='right', pad=0.05, 
#                                       title=None, format=None, size='5%')
#         # Set the plot title
#         axes[idx].set_title(f'SWA Dissipation - {cond.capitalize()} Condition')

#         # tighten layout
#         plt.tight_layout()

# plot_sw_decay(df2)

# #%%

# contrast = np.ones([len(df2.Subject.unique()), 
#                     len(df2.Condition.unique()), 
#                     len(df2.Channel.unique())])*(np.nan)
# for i, cond in enumerate(df2.Condition.unique()):
#     print(cond)
#     decay = df2[df2.Condition == cond].reset_index()
#     for j, sub in enumerate(decay.Subject.unique()):
#         print(sub)
#         decay_ = decay[decay.Subject == sub].reset_index()
#         for k, ch in enumerate(info.ch_names):
#             #print(ch)
#             decay__ = decay_[decay_.Channel==ch]
#             contrast[j, i, k] = decay__.Decay.to_numpy()[0]
                 

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