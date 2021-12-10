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
import time
import pickle
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
from autoreject import Ransac
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, _pre_process_sleep_data, preprocess_sleep_data, 
                                                     compare_hypnograms, check_match_data_hypno_elements,
                                                     label_artifacts, load_preprocessed_data, Data_SW, PLV)
from sleepstim.sleep_funs import (load_xdf, channel_parser, bfr_butter_filt, thresholdcrossings)
from sleepstim.Analysis.pac import unit_root_test, add_stimulus_onset, ERPAC
from sleepstim.Analysis.time_frequency import tfr_analysis
from sleepstim.Utils.encryption_decryption import encrypt_file, decrypt_file
from sleepstim.Analysis.Resting_State.rs_preproc import (plot_psd, norm_wavelet_power, subject_cond_parser)
import tensorpac.methods as tpm
from sleepstim.Analysis.foof import oscillatory_plot_psd_map

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def extract_pha_amp(data_narrow, data_broad, sf, n_samples, nfast):
    # Extract the spindles-related sigma signal for coupling
    data_sp = mne.filter.filter_data(data_broad, sf, 12, 16, method='fir',
                                     l_trans_bandwidth=1.5, h_trans_bandwidth=1.5,
                                     verbose=0)
    # Now extract the instantaneous phase/amplitude using Hilbert transform
    sw_pha = np.angle(signal.hilbert(data_narrow, N=nfast)[:n_samples])
    sp_amp = np.abs(signal.hilbert(data_sp, N=nfast)[:n_samples])
    
    return sw_pha, sp_amp  
 
def SO_spindle_coupling(sw, data_broad, idx, sf, target='stim_onset'):
    data_broad = data_broad[idx,:,:]
    time_before = 1.5; time_after = 2.5
    bef = int(sf * time_before)
    aft = int(sf * time_after)
    
    ## Iterate by channels
    summaries = []
    for chan in range(23):
        print(chan)
        if target == 'neg_peak':
            sw_neg_times = sw.summary()['NegPeak'][sw.summary()['IdxChannel']==chan].to_numpy()
            idx_neg_nearest = find_nearest(sw_neg_times, 3)
            sw_neg_time = sw_neg_times[idx_neg_nearest]
            sw_neg_idx = sw_neg_time * sf
        elif target == 'stim_onset':
            ## One can ignore all the values in the dataframe, except the ndPAC info 
            sw_neg_times = sw.summary()['NegPeak'][sw.summary()['IdxChannel']==chan].to_numpy()
            if list(sw_neg_times) != []:
                idx_neg_nearest = find_nearest(sw_neg_times, 3)
                sw_neg_time = sw_neg_times[idx_neg_nearest]
                sw_neg_idx = 3*sf
            else:
                summary = sw.summary()[sw.summary()['IdxChannel']==chan].reset_index()
                summary['SigmaPeak'] = np.ones(1) * np.nan
                summary['PhaseAtSigmaPeak'] = np.ones(1) * np.nan
                summary['ndPAC'] = np.ones(1) * np.nan
    
        ## continue only if channel contains sw
        if list(sw_neg_times) != []:
            if max(data_broad.shape) - sw_neg_idx < aft:
                aft = max(data_broad.shape) - sw_neg_idx
                # compensate for shorter after period by making longer before period
                # if sw_neg_idx < bef:
                #     bef = max(data_broad.shape) - sw_neg_idx
                # else:
                #     bef = aft + bef
                # print('Post - compensation')
        
            if sw_neg_idx < bef:
                bef = sw_neg_idx
                # bef = max(data_broad.shape) - sw_neg_idx
                # print('Pre - compensation')
        

            sw_idx, valid_idx = yasa.get_centered_indices(sw._data[chan,:].squeeze(), 
                                                          np.asarray([sw_neg_idx]), bef, aft) 
            
            # extract analytical phase for SOs and amplitude for spindles
            sw_pha, sp_amp = extract_pha_amp(sw._data[chan,:].squeeze(), data_broad[chan,:].squeeze(), sf, n_samples, nfast)                               
            sw_pha, sp_amp = sw_pha[sw_idx[0][0]:sw_idx[0][-1]], sp_amp[sw_idx[0][0]:sw_idx[0][-1]]
                  
            # only keep peaks near stim onset
            idx_not = np.where(np.arange(0, len(sw_neg_times)) != idx_neg_nearest)[0]
            summary = sw.summary()[sw.summary()['IdxChannel']==chan].reset_index().drop(idx_not, inplace=False)
            
            # 1) Find location of max sigma amplitude in epoch
            idx_max_amp = sp_amp.argmax(axis=0)
            
            # Now we need to append it back to the original unmasked shape
            # to avoid error when idx.shape[0] != idx_valid.shape, i.e.
            # some epochs were out of data bounds.
            summary['SigmaPeak'] = np.ones(1) * np.nan
            
            # Timestamp at sigma peak, expressed in seconds from negative peak
            # e.g. -0.39, 0.5, 1, 2 -- limits are [time_before, time_after]
            time_sigpk = (idx_max_amp - bef) / sf
            
            # convert to absolute time from beginning of the recording
            # time_sigpk only includes valid epoch
            time_sigpk_abs = sw_neg_idx + time_sigpk
            summary['SigmaPeak'] = time_sigpk_abs
            
            # 2) PhaseAtSigmaPeak
            # Find SW phase at max sigma amplitude in epoch
            pha_at_max = np.squeeze(np.take_along_axis(sw_pha,
                                                       idx_max_amp[..., None],
                                                       axis=0))
            summary['PhaseAtSigmaPeak'] = np.ones(1) * np.nan
            summary['PhaseAtSigmaPeak'] = pha_at_max
            
            # 3) Normalized Direct PAC, with thresholding
            ndp = np.squeeze(tpm.norm_direct_pac(sw_pha[None, ...],
                                                 sp_amp[None, ...], p=0.05))
            summary['ndPAC'] = np.ones(1) * np.nan
            summary['ndPAC'] = ndp
            
        else:
            pass
        
        summaries.append(summary)
        
    return summaries

def SO_spindle_coupling2(sw, data_broad, idx, sf, target='neg_peak'):
    data_broad = np.expand_dims(data_broad[idx,:],0)
    time_before = 1.5; time_after = 2.5
    bef = int(sf * time_before)
    aft = int(sf * time_after)
    
    if target == 'neg_peak':
        sw_neg_times = sw.summary()['NegPeak'].to_numpy()
        idx_neg_nearest = find_nearest(sw_neg_times, 3)
        sw_neg_time = sw_neg_times[idx_neg_nearest]
        sw_neg_idx = sw_neg_time * sf
    elif target == 'stim_onset':
        ## One can ignore all the values in the dataframe, except the ndPAC info 
        sw_neg_times = sw.summary()['NegPeak'].to_numpy()
        idx_neg_nearest = find_nearest(sw_neg_times, 3)
        sw_neg_time = sw_neg_times[idx_neg_nearest]
        sw_neg_idx = 3*sf
    
    if max(data_broad.shape) - sw_neg_idx < aft:
        aft = max(data_broad.shape) - sw_neg_idx
        # compensate for shorter after period by making longer before period
        # if sw_neg_idx < bef:
        #     bef = max(data_broad.shape) - sw_neg_idx
        # else:
        #     bef = aft + bef
        # print('Post - compensation')
        
    if sw_neg_idx < bef:
        bef = sw_neg_idx
        # bef = max(data_broad.shape) - sw_neg_idx
        # print('Pre - compensation')
        
    sw_idx, valid_idx = yasa.get_centered_indices(sw._data.squeeze(), np.asarray([sw_neg_idx]), 
                                                  bef, aft) 
    
    # extract analytical phase for SOs and amplitude for spindles
    sw_pha, sp_amp = extract_pha_amp(sw._data.squeeze(), data_broad.squeeze(), sf, n_samples, nfast)                               
    sw_pha, sp_amp = sw_pha[sw_idx[0][0]:sw_idx[0][-1]], sp_amp[sw_idx[0][0]:sw_idx[0][-1]]
          
    # only keep peaks near stim onset
    idx_not = np.where(np.arange(0, len(sw_neg_times)) != idx_neg_nearest)[0]
    summary = sw.summary().drop(idx_not, inplace=False)
    
    # 1) Find location of max sigma amplitude in epoch
    idx_max_amp = sp_amp.argmax(axis=0)
    
    # Now we need to append it back to the original unmasked shape
    # to avoid error when idx.shape[0] != idx_valid.shape, i.e.
    # some epochs were out of data bounds.
    summary['SigmaPeak'] = np.ones(1) * np.nan
    
    # Timestamp at sigma peak, expressed in seconds from negative peak
    # e.g. -0.39, 0.5, 1, 2 -- limits are [time_before, time_after]
    time_sigpk = (idx_max_amp - bef) / sf
    
    # convert to absolute time from beginning of the recording
    # time_sigpk only includes valid epoch
    time_sigpk_abs = sw_neg_idx + time_sigpk
    summary['SigmaPeak'] = time_sigpk_abs
    
    # 2) PhaseAtSigmaPeak
    # Find SW phase at max sigma amplitude in epoch
    pha_at_max = np.squeeze(np.take_along_axis(sw_pha,
                                               idx_max_amp[..., None],
                                               axis=0))
    summary['PhaseAtSigmaPeak'] = np.ones(1) * np.nan
    summary['PhaseAtSigmaPeak'] = pha_at_max
    
    # 3) Normalized Direct PAC, with thresholding
    ndp = np.squeeze(tpm.norm_direct_pac(sw_pha[None, ...],
                                         sp_amp[None, ...], p=0.05))
    summary['ndPAC'] = np.ones(1) * np.nan
    summary['ndPAC'] = ndp
              
    return summary
            
#%%

path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/'
maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files if 'av.p' in i])

#%%

def auditory_stim_results(maindir):
    # decoding condition sheet 
    # input_file = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
    # output_file = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes_encrypted.csv'
    # encrypt_key = encrypt_file(input_file, output_file)
    # create log file 
    log_path = '/media/administrator/data/Study_1_data/Data_tracking/'
    out = open(log_path + 'pinknoise_report.txt', "w")
    for i, files in tqdm(enumerate(maindir)):
        print(i, files.split('/')[-1])
        # load data instead of pre-processing data 
        Data = load_preprocessed_data(files)[0]
        # print(i, files.split('/')[-2])
        # Data = _pre_process_sleep_data(files, low_density=True, reference=None, validation='auditory', stageing=False)
    
        # take first (adjusted) pinknoise bursts as center point
        first_bursts = Data.pinknoise_times_sync[::2]
        if first_bursts != []:
            # check to see if condition is sham, if so, create second instance of center points to generate down sham condition
            # decrypted = decrypt_file(encrypt_key, input_file=output_file, output_file=input_file)
            cond = subject_cond_parser(files, study_phase = 'sleep')
            del decrypted
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
                                                           npts_before = Data.sfreq*3, npts_after = Data.sfreq*3)
                
                # create mne epoch object with online reference + offline re-reference
                info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
                info['bads'] = Data.bad_chans
            
                # create epochs with online reference
                epochs_down = mne.EpochsArray(np.swapaxes(Data.data[center_down]/1e6, 1, 2), info, tmin = -3, 
                                              baseline=(-3, -1.25), proj=False, reject = reject_criteria)
                epochs_down.set_montage(mne.channels.make_standard_montage('standard_1005')) 
                epochs_down.interpolate_bads()
                
                # mastoid referenced
                epochs_down_mastoids = epochs_down.set_eeg_reference(['M1','M2'])
                epochs_down_mastoids.interpolate_bads()
            else:
                epochs_down = []
                epochs_down_mastoids = []
                
            # generate center of stimulation index points
            center, _ = yasa.get_centered_indices(Data.data[:,Data.chans.index('C3')], np.asarray(first_bursts), 
                                                  npts_before = Data.sfreq*3, npts_after = Data.sfreq*3)
            info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
            info['bads'] = Data.bad_chans
            
            # create epochs with online reference
            epochs = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = -3, 
                                     baseline=(-3, -1.25), proj=False, reject = reject_criteria)
            epochs.set_montage(mne.channels.make_standard_montage('standard_1005')) 
            epochs.interpolate_bads()
            
            # =============================================================================
            ## CREATES COMBNINED CHANNEL TO PERFORM ANALYSIS; ENHANCES SNR 
            #  EEG_index = np.r_[Data.chans.index('F3'), Data.chans.index('C3'), Data.chans.index('P3')]
            #  raw = mne.channels.combine_channels(epochs, groups=dict(Left_hemispher = EEG_index))       
            # =============================================================================
                    
            # create epochs with mastoids reference
            epochs_mastoids = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = -3,
                                              baseline=(-3, -1.25), proj=False) 
            epochs_mastoids.set_eeg_reference(['M1','M2'])
            epochs_mastoids.set_montage(mne.channels.make_standard_montage('standard_1005'))
            epochs_mastoids.interpolate_bads()
            
            # log pinknoise instances
            out.write(f'The dataset: {files.split("/")[-1]} has {len(first_bursts)*2} pinknoise bursts! ' + '\n')
            logging.warning(f'The dataset: {files.split("/")[-1]} has {len(first_bursts)*2} pinknoise bursts! ')
      
            # epochs.drop_channels(ransac.bad_chs_ + ['M1','M2','Oz'])
            # epochs_mastoids.drop_channels(ransac.bad_chs_ + ['M1','M2','Oz'])
                    
            out.write(f'The dataset: {files.split("/")[-1]} has the following detected bad channels: {Data.bad_chans} ' + '\n')
      
            ## compute CSD 
            # epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
            
            ## compute GED 
            # see GED script
            
            ## GFP
            #GFP = np.std(epochs.get_data(picks='eeg'),axis=0)
            
            # dump epochs
            if epochs_down != []:
                pickle.dump(epochs_down, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_pn_ERP_sham_down.p',"wb"))
                pickle.dump(epochs_down_mastoids, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_pn_ERP_sham_down_mastoid.p',"wb"))
            pickle.dump(epochs, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_pn_ERP.p',"wb"))
            pickle.dump(epochs_mastoids, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_pn_ERP_mastoids.p',"wb"))
            #pickle.dump(epochs_csd, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + files.split('/')[-1].split('.')[0] + '_pn_ERP_CSD.p',"wb"))
            
            ## Plotting
            fig_path = '/media/administrator/data/Study_1_data/Figures/Pinknoise_epochs/'
                   
            # C3 - avg
            epochs.average(method='mean', picks='C3').plot()
            plt.title(f'{files.split("/")[-1]} - C3')
            plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_C3.png')
            
            epochs_mastoids.average(method='mean', picks='C3').plot()
            plt.title(f'{files.split("/")[-1]} - C3 (LM)')
            plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_C3_lm.png')
            
            # epochs_csd.average(method='mean', picks='C3').plot()
            # plt.title(f'{files.split("/")[-1]} - C3 (CSD)')
            # plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_C3_csd.png')
            
            # C3 - all trials    
            epochs.plot_image(picks='C3', title = f'{files.split("/")[-1].split(".")[0]} - C3')
            plt.savefig(fig_path + files.split('/')[-1] + '_all_epochs_C3.png')
            
            epochs_mastoids.plot_image(picks='C3', title = f'{files.split("/")[-1].split(".")[0]} - C3 (LM)')
            plt.savefig(fig_path + files.split('/')[-1] + '_all_epochs_C3_lm.png')
            
            # epochs_csd.plot_image(picks='C3')
            # plt.savefig(fig_path + files.split('/')[-1] + '_all_epochs_C3_csd.png')
            
            plt.close('all')
            
            ## compartive plotting
            ts_args = dict(spatial_colors = True, gfp=True, time_unit='s')
            topomap_args = dict(outlines = 'head', time_unit='s', cmap = 'Spectral_r', time_format = "%0.2f s")
            times = np.asarray([-0.5, 0, 0.25, 0.5, .75, 1.0, 1.25, 1.5, 1.75, 2.0])
            epochs.average().plot_joint(times, title=f'{files.split("/")[-1]} - Cpz Reference', 
                                        ts_args=ts_args, topomap_args=topomap_args)
            plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_joint.png')
            epochs_mastoids.average().plot_joint(times, title=f'{files.split("/")[-1]} - Average Mastoids', 
                                        ts_args=ts_args, topomap_args=topomap_args)
            plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_joint_lm.png')
            # epochs_csd.average().plot_joint(title=f'{files.split("/")[-1]} - Current Source Density', 
            #                                  ts_args=ts_args, topomap_args=topomap_args)
            # plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_joint_csd.png')
            plt.close('all')                                                       
    
            ## Topoplots
            # Amplitude distributions
            all_times = np.arange(-3.0, 3.25, 0.25)
            epochs.average().plot_topomap(all_times, ch_type='eeg', ncols='auto', nrows='auto',
                                           title=f'{files.split("/")[-1]} - Amplitude distributions (Cpz)',
                                           **topomap_args) 
            plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_amplitude_topo.png')
            epochs_mastoids.average().plot_topomap(all_times, ch_type='eeg', ncols= 'auto', nrows='auto',
                                                   title=f'{files.split("/")[-1]} - Amplitude distributions (LM)',
                                                   **topomap_args)
            plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_amplitude_topo_lm.png')
            # epochs_csd.average().plot_topomap(all_times, ncols='auto', nrows='auto',
            #                                   title=f'{files.split("/")[-1]} - Amplitude distributions (CSD)',
            #                                   **topomap_args) 
            # plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_amplitude_topo_csd.png')
            plt.close('all')
    
            # Epochs PSDs
            #epochs_mastoids.plot_psd(dB=True, fmin=0.5, fmax=30, average=True)
            
            #%%
            ## Power distributions (Pre/Post stim onset)
            # for x,y in zip(np.arange(-3.0,3.0,.5).round(2), np.arange(-2.5,3.5,.5).round(2)): # Sequential 500 ms windows 
            for x,y in zip([-3.0, 0.],[0., 3.0]):
                epochs_mastoids.plot_psd_topomap(tmin=x, tmax=y, bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                                                                          (12, 16, 'Sigma'), (16, 30, 'Beta')], ch_type='eeg',
                                                 dB=False, normalize = True, cmap='Spectral_r')
                plt.savefig(fig_path + files.split('/')[-1] + '_power_' + str(x) + 's_' + str(y) + 's_C3_lm.png')
                # epochs_csd.plot_psd_topomap(tmin=x, tmax=y, bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                #                                                      (12, 16, 'Sigma'), (16, 30, 'Beta')],
                #                                  dB=False, normalize = True, cmap='Spectral_r')
                #plt.savefig(fig_path + files.split('/')[-1] + '_power_' + str(x) + 's_' + str(y) + 's_C3_csd.png')
                plt.close('all')
                
            ## Power with fooof 1/f component 
            for x,y in zip([-3., 0.],[0., 3.]):
                band_powers = oscillatory_plot_psd_map(epochs_mastoids, tmin=x, tmax=y)
                band_powers.append(epochs_mastoids.info.ch_names[0:23])
                plt.savefig(fig_path + files.split('/')[-1] + '_osc_power_' + str(x) + 's_' + str(y) + 's_C3_lm.png')
                plt.close('all')  
            
            #%%
                
            # plot difference for C3
            mne.viz.plot_compare_evokeds(dict(online_ref=epochs.average(method='mean', picks='C3'), 
                                              offline_ref=epochs_mastoids.average(method='mean', picks='C3')),
                                              legend='upper left', show_sensors='upper right',
                                              title=f'{files.split("/")[-1]} - C3')
            plt.savefig(fig_path + files.split('/')[-1] + '_average_epoch_ref_comparison.png')
            
            ## TF/ITC plots 
            itc_data, Sxx = tfr_analysis(Data = epochs_mastoids, l_freq = 5, h_freq = 20, steps = 0.25, method = 'wavelet', 
                                         baseline=[-3, -1.25], mode='zscore', chan = 'C3', itc_calculation = None,
                                         plot=True, output='avg', zscore=False, cmap = 'Spectral_r', 
                                         save_path=fig_path + files.split('/')[-1])
            pickle.dump(Sxx, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_TF.p',"wb"))
            #pickle.dump(itc_data, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_ITC.p',"wb"))
            del itc_data, Sxx
            
            if epochs_down != []:
                itc_data, Sxx = tfr_analysis(Data = epochs_down_mastoids, l_freq = 5, h_freq = 20, steps = 0.25, method = 'wavelet', 
                                             baseline=[-3, -1.25], mode='zscore', chan = 'C3', itc_calculation = None,
                                             plot=True, output='avg', zscore=False, cmap = 'Spectral_r', 
                                             save_path=fig_path + files.split('/')[-1])
                pickle.dump(Sxx, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_sham_down_TF.p',"wb"))
                del itc_data, Sxx
            
            ## ERPAC plots
            erpac = ERPAC(data = epochs_mastoids, f_pha=[0.3, 4], f_amp=(4, 20, .5, .5), n_perm=None, 
                          smooth=200, method = 'gc', edges=0.5, stationarity_t=False, plot=True, 
                          save_path=fig_path + files.split('/')[-1])
            pickle.dump(erpac, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_erpac.p',"wb"))
            plt.close('all')
            del erpac
            
            
            if epochs_down != []:
                erpac = ERPAC(data = epochs_down_mastoids, f_pha=[0.3, 4], f_amp=(4, 20, .5, .5), n_perm=None, 
                              smooth=200, method = 'gc', edges=0.5, stationarity_t=False, plot=True, 
                              save_path=fig_path + files.split('/')[-1])
                pickle.dump(erpac, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_sham_down_erpac.p',"wb"))
                plt.close('all')
                del erpac
            
            #%% PLV analysis 
            #for epoch_name, epoch in zip(['cpz_ref','lm_ref','cpz_ref_down','lm_ref_down'], [epochs, epochs_mastoids, epochs_down, epochs_down_mastoids]):
            for epoch_name, epoch in zip(['cpz_ref','lm_ref'], [epochs, epochs_mastoids]):
                print(epoch_name)
                _, plv = PLV(epoch.get_data(picks ='eeg'), epoch.ch_names[0:23], epoch.info['sfreq'], foi=(.834,2))
    
                ## Plot PLV 
                fig, ax = plt.subplots()
                im, cm = mne.viz.plot_topomap(plv, pos = epoch.info, vmin = np.percentile(plv, 5),
                                              vmax = np.percentile(plv, 95), cmap='Spectral_r', 
                                              axes=ax, show=True)
                fig.colorbar(im, ax=ax)   
                plt.title(f'PLV seeded to C3 - {files.split("/")[-1][0:10]} - {epoch_name}')
                
                pickle.dump(plv, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + epoch_name + '_plv.p',"wb"))
                #plt.savefig(fig_path + files.split('/')[-1] + epoch_name + '_plv.png')
                # plt.close('all')
                del plv 
                        
            #%% Phase analysis (filter with 2.0 Hz lp filter)
            for epoch_name, epoch in zip(['cpz_ref','lm_ref'], [epochs, epochs_mastoids]):
                # epoch = epoch.copy().crop(tmin=-1.0, tmax=1.0, include_tmax=True)
                C3 = epoch.get_data(picks='C3').squeeze()*1e6
                C3_lp = epoch.copy().filter(l_freq=None, h_freq=1.5).get_data()[:,Data.chans.index('C3'),:]*1e6
                data_lp = epoch.copy().filter(l_freq=None, h_freq=1.5).get_data(picks='eeg')*1e6
                # Compute (analytical) instantaneous phase from a signal
                n_samples = max(C3_lp.shape)
                nfast = next_fast_len(n_samples)
                # to obtain sine relative angles add 0.5 pi to angles
                sw_pha = [np.angle(hilbert(C3_lp[i,:], N=nfast)[:n_samples]) + 0.5*np.pi for i in range(min(C3_lp.shape))]
                pn_phase = [sw_pha[i][int(Data.sfreq*3)] for i in range(len(sw_pha))]
                
                ax = plt.subplot(111, projection='polar')
                ax.hist(pn_phase)
                ax.set_title('Phase targeting accuracy (Analytical): ' + files.split("/")[-1] + ' - C3', va='bottom')
                plt.savefig(fig_path + files.split('/')[-1] + '_phase_targeting_all_' + epoch_name + '.png')
                pickle.dump(pn_phase, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_pn_ERP_phase_hilbert_' + epoch_name + '.p',"wb"))
                plt.close('all')
                
                # Identify spindles in data
                sp = [yasa.spindles_detect(data = epoch.get_data(picks='eeg')[i,:,:]*1e6, sf=512, 
                                           ch_names=epoch.info.ch_names[0:23], freq_sp=(11, 16), freq_broad=(1, 30),
                                           duration=(0.5, 3), min_distance=500, 
                                           thresh={'rel_pow': None, 'corr': None, 'rms': 1.5}, multi_only=False)
                      for i in range(len(epoch))]
     
                # leave only epochs with spindles
                sp_summaries = []
                for idx, spindle in enumerate(sp):
                    if spindle != None:
                        sp_summaries.append(spindle.summary()) 
                        
                # Identify slow waves in data
                sws = [yasa.sw_detect(data = data_lp[i,:,:], sf=512, 
                                      ch_names=epoch.info.ch_names[0:23], 
                                      freq_sw=(None, None), dur_neg=(0.1, 2), dur_pos=(0.1, 2),
                                      amp_neg=(None, None), coupling = False, amp_pos=(None, None),
                                      amp_ptp=(20, 500)) 
                       for i in range(np.shape(data_lp)[0])]

                ## add ndPAC info and remove all empty sequences without slow waves 
                sw_summaries = []
                for i in range(len(sws)):
                    if sws[i] != None:
                        print('epoch : ', i)
                        sw_summaries.append(SO_spindle_coupling(sws[i], data_broad = epoch.get_data(picks='eeg')*1e6, 
                                                                idx = i, sf = 512, target='stim_onset'))
                 
                df_sw = pd.concat([pd.concat(sw_summaries[i]) for i in range(23)])
                df_sw = df_sw.reset_index()
                del df_sw['index'], df_sw['level_0']
                
                # Distribution of ndPAC value [ADD TO GROUP LEVEL ANALYSIS] 
                plt.figure()
                df_sw['ndPAC'].hist();
                plt.savefig(fig_path + files.split('/')[-1] + '_ndPAC_' + epoch_name + '.png')
                print(f'ndPAC means: {df_sw.groupby("Channel")["ndPAC"].mean()}')
                pg.plot_circmean(df_sw['PhaseAtSigmaPeak'])
                plt.savefig(fig_path + files.split('/')[-1] + '_circular_phase_' + epoch_name + '.png')
                print('Circular mean: %.3f rad' % pg.circ_mean(df_sw['PhaseAtSigmaPeak']))
                print('Vector length: %.3f' % pg.circ_r(df_sw['PhaseAtSigmaPeak']))      
                
                # save and append sw, sp summaries 
                sw_summaries = df_sw
                sw_summaries['Subject'] = files.split('/')[-1].split('.')[0][0:10]
                sw_summaries['Condition'] = cond
                sw_summaries.to_csv('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_sw_summaries_' + epoch_name + '.csv')
                
                sp_summaries = pd.concat(sp_summaries)
                sp_summaries['Subject'] = files.split('/')[-1].split('.')[0][0:10]
                sp_summaries['Condition'] = cond
                sp_summaries = sp_summaries.reset_index()
                del sp_summaries['index']
                sp_summaries.to_csv('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_sp_summaries_' + epoch_name + '.csv')
                                          
                #%%
                sws_idx = []
                for i in range(len(sws)):
                    if len(sws[i].summary()[sws[i].summary()['Channel']=='C3'].to_numpy()) > 0:
                        sws_idx.append(1)
                    else:
                        sws_idx.append(0)
                        
                sws_idx = np.asarray(sws_idx)
                
                # space 90 degree bins for slow waves representing negative/postive half wave durations
                half1, half2, half3, half4 = [],[],[],[]
                for i in np.where(sws_idx == 1)[0]:
                    C3_df = sws[i].summary()[sws[i].summary().Channel=='C3'].reset_index()
                    half1.append(np.linspace(C3_df['Start'][0]*Data.sfreq, 
                                             C3_df['NegPeak'][0]*Data.sfreq, num=90, endpoint=True).astype(int))
                    half2.append(np.linspace(C3_df['NegPeak'][0]*Data.sfreq + 1, 
                                             C3_df['MidCrossing'][0]*Data.sfreq, num=90).astype(int))
                    half3.append(np.linspace(C3_df['MidCrossing'][0]*Data.sfreq + 1, 
                                             C3_df['PosPeak'][0]*Data.sfreq, num=90).astype(int))
                    half4.append(np.linspace(C3_df['PosPeak'][0]*Data.sfreq + 1, 
                                             C3_df['End'][0]*Data.sfreq, num=90).astype(int))
                
                # determine pinknoise targeted slow wave phase 
                crossings = []
                for i in range(len(half1)):
                    crossing = thresholdcrossings(np.concatenate([half1[i], half2[i], half3[i], half4[i]]), Data.sfreq*3)
                    if len(crossing) != 0:
                        crossings.append(crossing[0])
                    else:
                        crossings.append(np.nan)
                
                # Plot difference betwen identified slow waves versus none 
                plt.figure() 
                plt.plot(C3[np.where(sws_idx == 0)[0],:].mean(0), label = 'PN modulation - No SW')
                plt.plot(C3[np.where(sws_idx == 1)[0],:].mean(0), label = 'PN modulation - SW')
                plt.plot(C3.mean(0), label = 'PN modulation - All')
                plt.legend()
                plt.savefig(fig_path + files.split('/')[-1] + '_sw_difference_' + epoch_name + '.png')
                plt.close('all')
                
                # takes radians 
                if ~np.all(np.isnan(crossings)):
                    plt.figure()
                    ax2 = plt.subplot(111, projection='polar')
                    ax2.hist(np.deg2rad(crossings))
                    ax2.set_title('Phase targeting accuracy (Slow waves): ' + files.split("/")[-1] + ' - C3', va='bottom')
                    plt.savefig(fig_path + files.split('/')[-1] + '_phase_targeting_sw_' + epoch_name + '.png')
                    pickle.dump(np.deg2rad(crossings), open(f'/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/' + " ".join(files.split('/')[-1].split('.')[0].split('_')[0:2]) + '_pn_ERP_phase_' + epoch_name + '.p',"wb"))
                    plt.close('all')
        else: 
            logging.warning(f'The dataset: {files.split("/")[-1]} has no pinknoise epochs! ') 
            out.write(f'The dataset: {files.split("/")[-1]} has {len(first_bursts)*2} pinknoise bursts! ' + '\n')
            out.write(f'The dataset: {files.split("/")[-1]} has the following detected bad channels: {Data.bad_chans} ' + '\n')   
    # close text file with pn info 
    out.close()


#%%
# ## Group level phase analysis
# ## ADD ndPAC, ERPAC, PLV, GFP group analysis 
# sheet = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
# subj_cond = np.loadtxt(sheet, delimiter=',', dtype='str', skiprows=1) 
# cond_dict = {0:'sham', 1:'up', 2:'down'}
# path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/erp_analysis/'
# target_path = ''
# exp_files = sorted(os.listdir(path))

# sham, up, down = [], [], []
# sham_mastoids, up_mastoids, down_mastoids = [], [], []
# sham_down, sham_down_mastoids = [], []
# # sham_csd, up_csd, down_csd = [], [], []

# sham_phase, up_phase, down_phase = [], [], []
# sham_phase_ana, up_phase_ana, down_phase_ana = [], [], []
# sham_phase_lm, up_phase_lm, down_phase_lm = [], [], []
# sham_phase_lm_ana, up_phase_lm_ana, down_phase_lm_ana = [], [], []

# sham_pac, up_pac, down_pac = [], [], []
# sham_tf, up_tf, down_tf = [], [], []
# sham_erpac, up_erpac, down_erpac = [], [], []
# down_plv, up_plv, sham_plv = [],[],[]

# reject_criteria = dict(eeg=500e-6)
# for j, d_files in enumerate(exp_files):
#     print(j, d_files)
#     subj_cond = d_files.split(' ')[0] + '_' + d_files.split(' ')[1][0]
#     cond = subject_cond_parser(subj_cond, study_phase = 'sleep')
#     # subj = d_files.split('_')[0]
    
    
#     if d_files.endswith('_ERP.p'):
#         if cond == 'sham':
#             epochs = pickle.load(open(path + d_files,"rb"))
#             epochs.drop_bad(reject_criteria)
#             sham.append(epochs.get_data(picks='eeg').mean(0)*1e6)
#         elif cond == 'up':
#             epochs = pickle.load(open(path + d_files,"rb"))
#             epochs.drop_bad(reject_criteria)
#             up.append(epochs.get_data(picks='eeg').mean(0)*1e6)
#         elif cond == 'down':
#             epochs = pickle.load(open(path + d_files,"rb"))
#             epochs.drop_bad(reject_criteria)
#             down.append(epochs.get_data(picks='eeg').mean(0)*1e6)
#     elif d_files.endswith('_ERP_mastoids.p'):
#         if cond == 'sham':
#             epochs = pickle.load(open(path + d_files,"rb"))
#             epochs.drop_bad(reject_criteria)
#             sham_mastoids.append(epochs.get_data(picks='eeg').mean(0)*1e6)
#         elif cond == 'up':
#             epochs = pickle.load(open(path + d_files,"rb"))
#             epochs.drop_bad(reject_criteria)
#             up_mastoids.append(epochs.get_data(picks='eeg').mean(0)*1e6)
#         elif cond == 'down':
#             epochs = pickle.load(open(path + d_files,"rb"))
#             epochs.drop_bad(reject_criteria)
#             down_mastoids.append(epochs.get_data(picks='eeg').mean(0)*1e6)
            
#     elif d_files.endswith('_ERP_sham_down.p'):
#         if cond == 'sham':
#             epochs = pickle.load(open(path + d_files,"rb"))
#             epochs.drop_bad(reject_criteria)
#             sham_down.append(epochs.get_data(picks='eeg').mean(0)*1e6)

#     elif d_files.endswith('_ERP_sham_down_mastoid.p'):
#         if cond == 'sham':
#             epochs = pickle.load(open(path + d_files,"rb"))
#             epochs.drop_bad(reject_criteria)
#             sham_down_mastoids.append(epochs.get_data(picks='eeg').mean(0)*1e6)

#     elif d_files.endswith('_phase_cpz_ref.p'): 
#         if cond == 'sham':
#             pha = pickle.load(open(path + d_files,"rb"))
#             sham_phase.append(pha)
#         elif cond == 'up':
#             pha = pickle.load(open(path + d_files,"rb"))
#             up_phase.append(pha)
#         elif cond == 'down':
#             pha = pickle.load(open(path + d_files,"rb"))
#             down_phase.append(pha) 
#     elif d_files.endswith('_phase_lm_ref.p'): 
#         if cond == 'sham':
#             pha = pickle.load(open(path + d_files,"rb"))
#             sham_phase_lm.append(pha)
#         elif cond == 'up':
#             pha = pickle.load(open(path + d_files,"rb"))
#             up_phase_lm.append(pha)
#         elif cond == 'down':
#             pha = pickle.load(open(path + d_files,"rb"))
#             down_phase_lm.append(pha) 
#     elif d_files.endswith('_phase_hilbert_cpz_ref.p'): 
#         if cond == 'sham':
#             pha = pickle.load(open(path + d_files,"rb"))
#             sham_phase_ana.append(pha)
#         elif cond == 'up':
#             pha = pickle.load(open(path + d_files,"rb"))
#             up_phase_ana.append(pha)
#         elif cond == 'down':
#             pha = pickle.load(open(path + d_files,"rb"))
#             down_phase_ana.append(pha) 
#     elif d_files.endswith('_phase_hilbert_lm_ref.p'): 
#         if cond == 'sham':
#             pha = pickle.load(open(path + d_files,"rb"))
#             sham_phase_lm_ana.append(pha)
#         elif cond == 'up':
#             pha = pickle.load(open(path + d_files,"rb"))
#             up_phase_lm_ana.append(pha)
#         elif cond == 'down':
#             pha = pickle.load(open(path + d_files,"rb"))
#             down_phase_lm_ana.append(pha) 
#     elif d_files.endswith('_erpac.p'): 
#         if cond == 'sham':
#             erpac = pickle.load(open(path + d_files,"rb"))
#             sham_erpac.append(erpac)
#         elif cond == 'up':
#             erpac = pickle.load(open(path + d_files,"rb"))
#             up_erpac.append(erpac)
#         elif cond == 'down':
#             erpac = pickle.load(open(path + d_files,"rb"))
#             down_erpac.append(erpac) 
#     elif d_files.endswith('_TF.p'): 
#         if cond == 'sham':
#             tf = pickle.load(open(path + d_files,"rb"))
#             sham_tf.append(tf)
#         elif cond == 'up':
#             tf = pickle.load(open(path + d_files,"rb"))
#             up_tf.append(tf)
#         elif cond == 'down':
#             tf = pickle.load(open(path + d_files,"rb"))
#             down_tf.append(tf) 
#     elif d_files.endswith('_plv.p'): 
#         if cond == 'sham':
#             plv = pickle.load(open(path + d_files,"rb"))
#             sham_plv.append(plv)
#         elif cond == 'up':
#             plv = pickle.load(open(path + d_files,"rb"))
#             up_plv.append(plv)
#         elif cond == 'down':
#             plv = pickle.load(open(path + d_files,"rb"))
#             down_plv.append(plv) 
  
# #%%


# ## Plot PLV 
# fig, ax = plt.subplots()
# im, cm = mne.viz.plot_topomap(np.vstack(up_plv).mean(0), pos = epochs.info, vmin = np.percentile(np.vstack(up_plv).mean(0), 5),
#                               vmax = np.percentile(np.vstack(up_plv).mean(0), 95), cmap='Spectral_r', axes=ax, show=True)
# fig.colorbar(im, ax=ax)   
# plt.title(f'PLV seeded to C3 - Up')

# fig2, ax2 = plt.subplots()
# im2, cm2 = mne.viz.plot_topomap(np.vstack(down_plv).mean(0), pos = epochs.info, vmin = np.percentile(np.vstack(down_plv).mean(0), 5),
#                               vmax = np.percentile(np.vstack(down_plv).mean(0), 95), cmap='Spectral_r', axes=ax2, show=True)
# fig2.colorbar(im2, ax=ax2)   
# plt.title(f'PLV seeded to C3 - Down')

# fig3, ax3 = plt.subplots()
# im3, cm3 = mne.viz.plot_topomap(np.vstack(sham_plv).mean(0), pos = epochs.info, vmin = np.percentile(np.vstack(sham_plv).mean(0), 5),
#                               vmax = np.percentile(np.vstack(sham_plv).mean(0), 95), cmap='Spectral_r', axes=ax3, show=True)
# fig3.colorbar(im3, ax=ax3)   
# plt.title(f'PLV seeded to C3 - Sham')


# #%%   
# # info = mne.create_info(ch_names=['C3'], sfreq=512, ch_types=['eeg'])
# info = epochs.pick('eeg').info
# # epochs_up = mne.EpochsArray(np.expand_dims(np.concatenate(up),1)/1e6, info, tmin = -3, baseline=None)
# rev_up = np.concatenate([np.expand_dims(up[i], 0) for i in range(len(up))], 0)
# epochs_up = mne.EpochsArray(rev_up/1e6, info, tmin = -3, baseline=None)
# epochs_up.set_montage(mne.channels.make_standard_montage('standard_1005'))

# rev_down = np.concatenate([np.expand_dims(down[i], 0) for i in range(len(down))], 0)
# epochs_down = mne.EpochsArray(rev_down/1e6, info, tmin = -3, baseline=None)
# epochs_down.set_montage(mne.channels.make_standard_montage('standard_1005'))

# rev_sham = np.concatenate([np.expand_dims(sham[i], 0) for i in range(len(sham))], 0)
# epochs_sham = mne.EpochsArray(rev_sham/1e6, info, tmin = -3, baseline=None)
# epochs_sham.set_montage(mne.channels.make_standard_montage('standard_1005'))

# # epochs_sham_down = mne.EpochsArray(np.expand_dims(np.concatenate(sham_down),1)/1e6, info, tmin = -3, baseline=None)
# # epochs_sham_down.set_montage(mne.channels.make_standard_montage('standard_1005'))

# rev_up_mastoids = np.concatenate([np.expand_dims(up_mastoids[i], 0) for i in range(len(up_mastoids))], 0)
# epochs_up_mastoids = mne.EpochsArray(rev_up_mastoids/1e6, info, tmin = -3, baseline=None)
# epochs_up_mastoids.set_montage(mne.channels.make_standard_montage('standard_1005'))

# rev_down_mastoids = np.concatenate([np.expand_dims(down_mastoids[i], 0) for i in range(len(down_mastoids))], 0)
# epochs_down_mastoids = mne.EpochsArray(rev_down_mastoids/1e6, info, tmin = -3, baseline=None)
# epochs_down_mastoids.set_montage(mne.channels.make_standard_montage('standard_1005'))

# rev_sham_mastoids = np.concatenate([np.expand_dims(sham_mastoids[i], 0) for i in range(len(sham_mastoids))], 0)
# epochs_sham_mastoids = mne.EpochsArray(rev_sham_mastoids/1e6, info, tmin = -3, baseline=None)
# epochs_sham_mastoids.set_montage(mne.channels.make_standard_montage('standard_1005'))

# rev_sham_mastoids_down = np.concatenate([np.expand_dims(sham_down_mastoids[i], 0) for i in range(len(sham_down_mastoids))], 0)
# epochs_sham_mastoids_down = mne.EpochsArray(rev_sham_mastoids_down/1e6, info, tmin = -3, baseline=None)
# epochs_sham_mastoids_down.set_montage(mne.channels.make_standard_montage('standard_1005'))

# # epochs_sham_csd = mne.EpochsArray(np.expand_dims(np.concatenate(sham_csd),1)/1e6, info, tmin = -3, baseline=None)
# # epochs_sham_csd.set_montage(mne.channels.make_standard_montage('standard_1005'))
# # epochs_down_csd = mne.EpochsArray(np.expand_dims(np.concatenate(down_csd),1)/1e6, info, tmin = -3, baseline=None)
# # epochs_down_csd.set_montage(mne.channels.make_standard_montage('standard_1005'))
# # epochs_up_csd = mne.EpochsArray(np.expand_dims(np.concatenate(up_csd),1)/1e6, info, tmin = -3, baseline=None)
# # epochs_up_csd.set_montage(mne.channels.make_standard_montage('standard_1005'))
# # sns.set_theme(color_codes=True)

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

# epochs_down_mastoids.average().plot_joint(times = times, title='GAV - Down', 
#                                         ts_args=ts_args, topomap_args=topomap_args)

# epochs_sham_mastoids.average().plot_joint(times = times, title='GAV - Sham Up', 
#                                         ts_args=ts_args, topomap_args=topomap_args)

# epochs_sham_mastoids_down.average().plot_joint(times = times, title='GAV - Sham Down', 
#                                         ts_args=ts_args, topomap_args=topomap_args)

# # plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_joint.png')



# # plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_joint_lm.png')
   
# # plt.savefig(fig_path + files.split('/')[-1] + '_avg_epoch_joint_csd.png')
# # plt.close('all')
    
# #%%
 
# #%%   
# for x,y in zip([-3.0, 0.],[0., 3.0]):
#     epochs_up_mastoids.plot_psd_topomap(tmin=x, tmax=y, bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
#                                                               (12, 16, 'Sigma'), (16, 30, 'Beta')], ch_type='eeg',
#                                       dB=False, normalize = True, cmap='Spectral_r')
#     # plt.savefig(fig_path + files.split('/')[-2] + '_power_' + str(x) + 's_' + str(y) + 's_C3_lm.png')
#     epochs_up_csd.plot_psd_topomap(tmin=x, tmax=y, bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
#                                                           (12, 16, 'Sigma'), (16, 30, 'Beta')],
#                                 dB=False, normalize = True, cmap='Spectral_r')
#     # plt.savefig(fig_path + files.split('/')[-2] + '_power_' + str(x) + 's_' + str(y) + 's_C3_lm.png')
#     plt.close('all')
    
# up_phase_1 = np.concatenate(up_phase_lm_ana)[~np.isnan(np.concatenate(up_phase_lm_ana))]
# down_phase_1 = np.concatenate(down_phase_lm_ana)[~np.isnan(np.concatenate(down_phase_lm_ana))]
# sham_phase_1 = np.concatenate(sham_phase_lm_ana)[~np.isnan(np.concatenate(sham_phase_lm_ana))]

# up_rv = pg.circ_r(up_phase_1)
# down_rv = pg.circ_r(down_phase_1)
# sham_rv = pg.circ_r(sham_phase_1)

# up_phi = pg.circ_mean(up_phase_1)
# down_phi = pg.circ_mean(down_phase_1)
# sham_phi = pg.circ_mean(sham_phase_1)
    
# df = pd.concat([pd.DataFrame({'Phase targets' : up_phase_1, 'Condition' : 'Up', 'Resultant vector' : up_rv, 'Circular mean' : up_phi}),
#                 pd.DataFrame({'Phase targets' : down_phase_1, 'Condition' : 'Down', 'Resultant vector' : down_rv, 'Circular mean' : down_phi}),
#                 pd.DataFrame({'Phase targets' : sham_phase_1, 'Condition' : 'Sham', 'Resultant vector' : sham_rv, 'Circular mean' : sham_phi})], 
#                 ignore_index=True)


# g = sns.FacetGrid(df, col= 'Condition', hue='Condition',
#                   subplot_kws=dict(projection='polar'), height=4.5,
#                   sharex=True, sharey=True, despine=False)

# g.map_dataframe(sns.histplot, 'Phase targets', stat = 'density')

# kwargs_arrow={'fc': 'tab:red', 'ec': 'tab:red'}
# for ax, phi_idx, rv_idx in zip(range(len(g.axes[0,:])), [up_phi, down_phi, sham_phi], 
#                                 [up_rv, down_rv, sham_rv]):
#     g.axes[0,ax].arrow(0, 0, phi_idx, rv_idx, **kwargs_arrow)
  
# # Individual plots
# for var,title in zip([up_phase, down_phase, sham_phase], 
#                       ['Up-stimulation', 'Down-stimulation', 'Sham-stimulation']):
#     plt.figure()
#     ax1 = plt.subplot(111, projection='polar')
#     ax1 = sns.histplot(var)
#     # ax1.hist(var)
#     # circular mean (add 2*pi to make between 0 and 2*pi)
#     phi = pg.circ_mean(var) #+ 2*np.pi
#     print('Circular mean: %.3f rad' %  phi)
#     # resultant vector length
#     rv = pg.circ_r(var)
#     print('Vector length: %.3f' % rv)
#     # circular mean plot
#     ax2 = pg.plot_circmean(var, kwargs_markers=dict(marker="None"))
    
#     # create arrow for circular mean and resulatant vector length
#     kwargs_arrow={'width': 0.01, 'head_width': 0.1, 'head_length': 0.1, 
#                   'fc': 'tab:red', 'ec': 'tab:red'}

    
#     ax1.set_title("Phase targeting accuracy : " + title, va='bottom')
#     ax2.set_title("Circular mean : " + title)
    


# #%%
# ## Phase analysis 
# C3 = epochs.get_data()[:,Data.chans.index('C3'),:]*1e6
 
# # Identify slow waves in data
# sws = [yasa.sw_detect(data = C3[i,:], sf=512, ch_names=['C3'], freq_sw=(0.5, 2), 
#                       amp_neg=(35, 300), amp_pos=(5, 200), amp_ptp=(75, 500)) for i in range(min(np.shape(C3)))]
# sws_idx = []
# for i in range(len(sws)):
#     if sws[i] != None:
#         sws_idx.append(1)
#     else:
#         sws_idx.append(0)
        
# sws_idx = np.asarray(sws_idx)

# # Compute instantaneous phase from a signal
# n_samples = max(C3.shape)
# nfast = next_fast_len(n_samples)
# #sw_pha = [np.angle(hilbert(C3[i,:], N=nfast)[:n_samples]) for i in range(C3.shape[0])]
# sw_pha = [np.angle(hilbert(C3[i,:], N=nfast)[:n_samples]) for i in np.where(sws_idx == 1)[0]]
# pn_phase = [sw_pha[i][int(Data.sfreq*1.5)] for i in range(len(sw_pha))]

# ax = plt.subplot(111, projection='polar')
# ax.hist(pn_phase)
# ax.set_title("Phase targeting accuracy (Analytical): {files.split("/")[-2]} - C3}", va='bottom')

# half1, half2, half3, half4 = [],[],[],[]
# for i in np.where(sws_idx == 1)[0]:
#     half1.append(np.linspace(sws[i].summary()['Start'][0]*Data.sfreq, 
#                               sws[i].summary()['NegPeak'][0]*Data.sfreq, num=90, endpoint=True).astype(int))
#     half2.append(np.linspace(sws[i].summary()['NegPeak'][0]*Data.sfreq + 1, 
#                               sws[i].summary()['MidCrossing'][0]*Data.sfreq, num=90).astype(int))
#     half3.append(np.linspace(sws[i].summary()['MidCrossing'][0]*Data.sfreq + 1, 
#                               sws[i].summary()['PosPeak'][0]*Data.sfreq, num=90).astype(int))
#     half4.append(np.linspace(sws[i].summary()['PosPeak'][0]*Data.sfreq + 1, 
#                               sws[i].summary()['End'][0]*Data.sfreq, num=90).astype(int))
    
# crossings = []
# for i in range(len(half1)):
#     crossing = thresholdcrossings(np.concatenate([half1[i], half2[i], half3[i], half4[i]]), Data.sfreq*1.5)
#     if len(crossing) != 0:
#         crossings.append(crossing[0])
#     else:
#         crossings.append(np.nan)
# # idx = np.nonzero(crossings)[0]

# # Plot some figures
# plt.figure() 
# plt.plot(C3[np.where(sws_idx == 0)[0],:].mean(0), label = 'PN modulation - No SW')
# plt.plot(C3[np.where(sws_idx == 1)[0],:].mean(0), label = 'PN modulation - SW')
# plt.plot(C3.mean(0), label = 'PN modulation - All')
# plt.legend()
   
# # takes radians 
# plt.figure()
# ax2 = plt.subplot(111, projection='polar')
# ax2.hist(np.deg2rad(crossings))
# ax2.set_title("Phase targeting accuracy : {files.split("/")[-2]} - C3}", va='bottom')


# # ptps = [np.ptp(C3[i,0:Data.sfreq*2]) for i in range(np.shape(C3)[0])]
# # loc = np.where(np.asarray(ptps)>75)[0]

# # for i in loc:
# #     plt.figure()
# #     plt.plot(C3_pn_narrow[i,0:Data.sfreq*3])
# #     plt.vlines(Data.sfreq*1.5, ymin=-75, ymax=40)

