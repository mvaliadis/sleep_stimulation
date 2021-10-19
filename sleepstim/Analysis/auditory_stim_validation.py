#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:41:47 2021

@author: administrator
"""

import numpy as np
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
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
from matplotlib import cm
from autoreject import Ransac
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, _pre_process_sleep_data, preprocess_sleep_data, 
                                                     compare_hypnograms, check_match_data_hypno_elements,
                                                     label_artifacts, load_preprocessed_data, Data_SW)
from sleepstim.sleep_funs import (load_xdf, channel_parser, bfr_butter_filt, thresholdcrossings)
from sleepstim.Analysis.pac import unit_root_test, add_stimulus_onset, ERPAC
from sleepstim.Analysis.time_frequency import tfr_analysis
from sleepstim.Utils.encryption_decryption import encrypt_file, decrypt_file
from sleepstim.Analysis.Resting_State.rs_preproc import (plot_psd, norm_wavelet_power, subject_cond_parser)

#%%

path = '/media/administrator/data/Study_1_data/Raw_data/Experimental/'
files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])

#%%
# decoding condition sheet 
input_file = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
output_file = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes_encrypted.csv'
encrypt_key = encrypt_file(input_file, output_file)
# create log file 
log_path = '/media/administrator/data/Study_1_data/Data_tracking/'
out = open(log_path + 'pinknoise_report.txt', "w")
for i, files in tqdm(enumerate(files_list)):
    print(i, files.split('/')[-2])
    Data = _pre_process_sleep_data(files, reference=None, validation='auditory', stageing=False)
    # take first (adjusted) pinknoise bursts as center point
    first_bursts = Data.pinknoise_times_sync[::2]
    if first_bursts != []:
        # check to see if condition is sham, if so, create second instance of center points to generate down sham condition
        decrypted = decrypt_file(encrypt_key, input_file=output_file, output_file=input_file)
        cond = subject_cond_parser(files, study_phase = 'sleep')
        del decrypted
        subject = files.split("/")[-2].split('_')[0]
        # create down sham instance subtracting first bursts by respective p2p duration difference
        if cond == 'sham':
            subject_table = pd.DataFrame(pd.read_csv('/media/administrator/data/Study_1_data/Data_tracking/Subject_table.csv', 
                                                     delimiter=',', dtype='str', header=0))
            p2p_delay = int(subject_table[subject_table['Subject ID'] == subject]['Peak to Peak duration'].to_list()[0][0:3])/1000
            sham_down_idx = np.asarray(first_bursts) - round(p2p_delay*Data.sfreq)

            # generate center of stimulation index points
            center_down, _ = yasa.get_centered_indices(Data.data[:,Data.chans.index('C3')], sham_down_idx, 
                                                       npts_before = Data.sfreq*3, npts_after = Data.sfreq*3)
        
        # generate center of stimulation index points
        center, _ = yasa.get_centered_indices(Data.data[:,Data.chans.index('C3')], np.asarray(first_bursts), 
                                              npts_before = Data.sfreq*3, npts_after = Data.sfreq*3)
        info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
        
        # create epochs with online reference
        epochs = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = -3, 
                                 baseline=(None), proj=False) 
        epochs.set_montage(mne.channels.make_standard_montage('standard_1005'))   
        
        # =============================================================================
        ## CREATES COMBNINED CHANNEL TO PERFORM ANALYSIS; ENHANCES SNR 
        #  EEG_index = np.r_[Data.chans.index('F3'), Data.chans.index('C3'), Data.chans.index('P3')]
        #  raw = mne.channels.combine_channels(epochs, groups=dict(Left_hemispher = EEG_index))       
        # =============================================================================
                
        # create epochs with mastoids reference
        epochs_mastoids = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = -3,
                                          baseline=(None), proj=False) 
        epochs_mastoids.set_eeg_reference(['M1','M2'])
        epochs_mastoids.set_montage(mne.channels.make_standard_montage('standard_1005'))
        
        # log pinknoise instances
        out.write(f'The dataset: {files.split("/")[-2]} has {len(first_bursts)*2} pinknoise bursts! ' + '\n')
        logging.warning(f'The dataset: {files.split("/")[-2]} has {len(first_bursts)*2} pinknoise bursts! ')
  
        # EQI method determines bad channels before CSD computation
        Data.detect_bad_chans(method='EQI')
        
        # epochs.drop_channels(ransac.bad_chs_ + ['M1','M2','Oz'])
        # epochs_mastoids.drop_channels(ransac.bad_chs_ + ['M1','M2','Oz'])
                
        out.write(f'The dataset: {files.split("/")[-2]} has the following detected bad channels: {Data.bad_chans} ' + '\n')
  
        ## compute CSD 
        epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
        
        ## TO-DO: compute GED 
        # see GED script
        
        # dump epochs
        pickle.dump(epochs, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/' + files.split('/')[-2].split('.')[0] + '_pn_ERP.p',"wb"))
        pickle.dump(epochs_mastoids, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/' + files.split('/')[-2].split('.')[0] + '_pn_ERP_mastoids.p',"wb"))
        pickle.dump(epochs_csd, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/' + files.split('/')[-2].split('.')[0] + '_pn_ERP_CSD.p',"wb"))
        
        ## Plotting
        fig_path = '/media/administrator/data/Study_1_data/Figures/Pinknoise_epochs/'
               
        # C3 - avg
        epochs.average(method='mean', picks='C3').plot()
        plt.title(f'{files.split("/")[-2]} - C3')
        plt.savefig(fig_path + files.split('/')[-2] + '_avg_epoch_C3.png')
        
        epochs_mastoids.average(method='mean', picks='C3').plot()
        plt.title(f'{files.split("/")[-2]} - C3 (LM)')
        plt.savefig(fig_path + files.split('/')[-2] + '_avg_epoch_C3_lm.png')
        
        epochs_csd.average(method='mean', picks='C3').plot()
        plt.title(f'{files.split("/")[-2]} - C3 (CSD)')
        plt.savefig(fig_path + files.split('/')[-2] + '_avg_epoch_C3_csd.png')
        
        # C3 - all trials    
        epochs.plot_image(picks='C3', title = f'{files.split("/")[-2].split(".")[0]} - C3')
        plt.savefig(fig_path + files.split('/')[-2] + '_all_epochs_C3.png')
        
        epochs_mastoids.plot_image(picks='C3', title = f'{files.split("/")[-2].split(".")[0]} - C3 (LM)')
        plt.savefig(fig_path + files.split('/')[-2] + '_all_epochs_C3_lm.png')
        
        epochs_csd.plot_image(picks='C3')
        plt.savefig(fig_path + files.split('/')[-2] + '_all_epochs_C3_csd.png')
        
        plt.close('all')
        
        ## compartive plotting
        ts_args = dict(spatial_colors = True, gfp=True, time_unit='s')
        topomap_args = dict(outlines = 'head', time_unit='s', cmap = 'Spectral_r', time_format = "%0.2f s")
        #times = [-1.0, -0.5, 0, 0.5, 1.0, 1.5]
        epochs.average().plot_joint(title=f'{files.split("/")[-2]} - Cpz Reference', 
                                    ts_args=ts_args, topomap_args=topomap_args)
        plt.savefig(fig_path + files.split('/')[-2] + '_avg_epoch_joint.png')
        epochs_mastoids.average().plot_joint(title=f'{files.split("/")[-2]} - Average Mastoids', 
                                    ts_args=ts_args, topomap_args=topomap_args)
        plt.savefig(fig_path + files.split('/')[-2] + '_avg_epoch_joint_lm.png')
        epochs_csd.average().plot_joint(title=f'{files.split("/")[-2]} - Current Source Density', 
                                         ts_args=ts_args, topomap_args=topomap_args)
        plt.savefig(fig_path + files.split('/')[-2] + '_avg_epoch_joint_csd.png')
        plt.close('all')

        ## Topoplots
        # Amplitude distributions
        all_times = np.arange(-3.0, 3.25, 0.25)
        epochs.average().plot_topomap(all_times, ch_type='eeg', ncols='auto', nrows='auto',
                                       title=f'{files.split("/")[-2]} - Amplitude distributions (Cpz)',
                                       **topomap_args) 
        plt.savefig(fig_path + files.split('/')[-2] + '_avg_epoch_amplitude_topo.png')
        epochs_mastoids.average().plot_topomap(all_times, ch_type='eeg', ncols= 'auto', nrows='auto',
                                               title=f'{files.split("/")[-2]} - Amplitude distributions (LM)',
                                               **topomap_args)
        plt.savefig(fig_path + files.split('/')[-2] + '_avg_epoch_amplitude_topo_lm.png')
        epochs_csd.average().plot_topomap(all_times, ncols='auto', nrows='auto',
                                          title=f'{files.split("/")[-2]} - Amplitude distributions (CSD)',
                                          **topomap_args) 
        plt.savefig(fig_path + files.split('/')[-2] + '_avg_epoch_amplitude_topo_csd.png')
        plt.close('all')

        # Epochs PSDs
        #epochs_mastoids.plot_psd(dB=True, fmin=0.5, fmax=30, average=True)
        
        ## Power distributions (Pre/Post stim onset)
        # for x,y in zip(np.arange(-3.0,3.0,.5).round(2), np.arange(-2.5,3.5,.5).round(2)): # Sequential 500 ms windows 
        for x,y in zip([-3.0, 0.],[0., 3.0]):
            epochs_mastoids.plot_psd_topomap(tmin=x, tmax=y, bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                                                                      (12, 16, 'Sigma'), (16, 30, 'Beta')], ch_type='eeg',
                                             dB=False, normalize = True, cmap='Spectral_r')
            plt.savefig(fig_path + files.split('/')[-2] + '_power_' + str(x) + 's_' + str(y) + 's_C3_lm.png')
            epochs_csd.plot_psd_topomap(tmin=x, tmax=y, bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                                                                 (12, 16, 'Sigma'), (16, 30, 'Beta')],
                                             dB=False, normalize = True, cmap='Spectral_r')
            plt.savefig(fig_path + files.split('/')[-2] + '_power_' + str(x) + 's_' + str(y) + 's_C3_csd.png')
            plt.close('all')
            
        # plot difference for C3
        mne.viz.plot_compare_evokeds(dict(online_ref=epochs.average(method='mean', picks='C3'), 
                                          offline_ref=epochs_mastoids.average(method='mean', picks='C3')),
                                          legend='upper left', show_sensors='upper right',
                                          title=f'{files.split("/")[-2]} - C3')
        plt.savefig(fig_path + files.split('/')[-2] + '_average_epoch_ref_comparison.png')
        
        ## TF/ITC plots 
        itc_data, Sxx = tfr_analysis(Data = epochs_mastoids, l_freq = 1, h_freq = 30, steps = 0.25, method = 'wavelet', 
                                     baseline=[-2, 2], mode='zscore', chan = 'C3', itc_calculation = 'tensorpac',
                                     plot=True, output='avg', zscore=False, cmap = cm.Spectral_r, 
                                     save_path=fig_path + files.split('/')[-2])
        pickle.dump(Sxx, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/' + files.split('/')[-2].split('.')[0] + '_TF.p',"wb"))
        pickle.dump(itc_data, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/' + files.split('/')[-2].split('.')[0] + '_ITC.p',"wb"))
        del itc_data, Sxx
        
        ## ERPAC plots
        erpac = ERPAC(data = epochs_mastoids, f_pha=[0.3, 4], f_amp=(4, 30, .3, .3), n_perm=None, 
                      smooth=200, method = 'gc', edges=2.0, stationarity_t=False, plot=True, 
                      save_path=fig_path + files.split('/')[-2])
        pickle.dump(erpac, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/' + files.split('/')[-2].split('.')[0] + '_erpac.p',"wb"))
        plt.close('all')
        del erpac
        
        ## Phase analysis (filter with 2.0 Hz lp filter)
        for epoch_name, epoch in zip(['cpz_ref','lm_ref'], [epochs, epochs_mastoids]):
            C3 = epoch.filter(l_freq=None, h_freq=2.).get_data()[:,Data.chans.index('C3'),:]*1e6
            # Compute (analytical) instantaneous phase from a signal
            n_samples = max(C3.shape)
            nfast = next_fast_len(n_samples)
            # to obtain sine relative angles add 0.5 pi to angles
            sw_pha = [np.angle(hilbert(C3[i,:], N=nfast)[:n_samples]) + 0.5*np.pi for i in range(min(C3.shape))]
            pn_phase = [sw_pha[i][int(Data.sfreq*3)] for i in range(len(sw_pha))]
            
            ax = plt.subplot(111, projection='polar')
            ax.hist(pn_phase)
            ax.set_title('Phase targeting accuracy (Analytical): ' + files.split("/")[-2] + ' - C3', va='bottom')
            plt.savefig(fig_path + files.split('/')[-2] + '_phase_targeting_all_' + epoch_name + '.png')
            pickle.dump(pn_phase, open('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/' + files.split('/')[-2].split('.')[0] + '_pn_ERP_phase_hilbert_' + epoch_name + '.p',"wb"))
            plt.close('all')
            
            # Identify slow waves in data
            sws = [yasa.sw_detect(data = C3[i,:], sf=512, ch_names=['C3'], freq_sw=(None, None), 
                                  dur_neg=(0.1, 2), dur_pos=(0.1, 2),
                                  amp_neg=(None, None), amp_pos=(None, None), 
                                  amp_ptp=(40, 500)) for i in range(min(np.shape(C3)))]
            sws_idx = []
            for i in range(len(sws)):
                if sws[i] != None:
                    sws_idx.append(1)
                else:
                    sws_idx.append(0)
                    
            sws_idx = np.asarray(sws_idx)
            
            # space 90 degree bins for slow waves representing negative/postive half wave durations
            half1, half2, half3, half4 = [],[],[],[]
            for i in np.where(sws_idx == 1)[0]:
                half1.append(np.linspace(sws[i].summary()['Start'][0]*Data.sfreq, 
                                         sws[i].summary()['NegPeak'][0]*Data.sfreq, num=90, endpoint=True).astype(int))
                half2.append(np.linspace(sws[i].summary()['NegPeak'][0]*Data.sfreq + 1, 
                                         sws[i].summary()['MidCrossing'][0]*Data.sfreq, num=90).astype(int))
                half3.append(np.linspace(sws[i].summary()['MidCrossing'][0]*Data.sfreq + 1, 
                                         sws[i].summary()['PosPeak'][0]*Data.sfreq, num=90).astype(int))
                half4.append(np.linspace(sws[i].summary()['PosPeak'][0]*Data.sfreq + 1, 
                                         sws[i].summary()['End'][0]*Data.sfreq, num=90).astype(int))
            
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
            plt.savefig(fig_path + files.split('/')[-2] + '_sw_difference_' + epoch_name + '.png')
            plt.close('all')
            
            # takes radians 
            if ~np.all(np.isnan(crossings)):
                plt.figure()
                ax2 = plt.subplot(111, projection='polar')
                ax2.hist(np.deg2rad(crossings))
                ax2.set_title('Phase targeting accuracy (Slow waves): ' + files.split("/")[-2] + ' - C3', va='bottom')
                plt.savefig(fig_path + files.split('/')[-2] + '_phase_targeting_sw_' + epoch_name + '.png')
                pickle.dump(np.deg2rad(crossings), open(f'/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/' + files.split('/')[-2].split('.')[0] + '_pn_ERP_phase_' + epoch_name + '.p',"wb"))
                plt.close('all')
    else: 
        logging.warning(f'The dataset: {files.split("/")[-2]} has no pinknoise epochs! ') 
        out.write(f'The dataset: {files.split("/")[-2]} has {len(first_bursts)*2} pinknoise bursts! ' + '\n')
        out.write(f'The dataset: {files.split("/")[-2]} has the following detected bad channels: {Data.bad_chans} ' + '\n')   
# close text file with pn info 
out.close()


#%%
## Group level phase analysis
sheet = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
subj_cond = np.loadtxt(sheet, delimiter=',', dtype='str', skiprows=1) 
cond_dict = {0:'sham', 1:'up', 2:'down'}
path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/'
target_path = ''
exp_files = sorted(os.listdir(path))

sham, up, down = [], [], []
sham_mastoids, up_mastoids, down_mastoids = [], [], []
sham_csd, up_csd, down_csd = [], [], []

sham_phase, up_phase, down_phase = [], [], []
sham_phase_ana, up_phase_ana, down_phase_ana = [], [], []
sham_phase_lm, up_phase_lm, down_phase_lm = [], [], []
sham_phase_lm_ana, up_phase_lm_ana, down_phase_lm_ana = [], [], []

sham_itc, up_itc, down_itc = [], [], []
sham_tf, up_tf, down_tf = [], [], []

for j, d_files in enumerate(exp_files):
    print(j, d_files)
    subj = d_files.split('_')[0]
    sub_pos = np.where(subj == subj_cond[:,0])[0][0]
    
    if '1' in d_files.split('_')[1]:
        cond = int(subj_cond[sub_pos,:][1])
    elif '2' in d_files.split('_')[1]:
        cond = int(subj_cond[sub_pos,:][2])
    elif '3' in d_files.split('_')[1]:
        cond = int(subj_cond[sub_pos,:][3])
    
    if d_files.endswith('_ERP.p'):
        if cond_dict[cond] == 'sham':
            epochs = pickle.load(open(path + d_files,"rb"))
            sham.append(epochs.get_data(picks='C3').mean(0)*1e6)
        elif cond_dict[cond] == 'up':
            epochs = pickle.load(open(path + d_files,"rb"))
            up.append(epochs.get_data(picks='C3').mean(0)*1e6)
        elif cond_dict[cond] == 'down':
            epochs = pickle.load(open(path + d_files,"rb"))
            down.append(epochs.get_data(picks='C3').mean(0)*1e6)
    elif d_files.endswith('_ERP_mastoids.p'):
        if cond_dict[cond] == 'sham':
            epochs = pickle.load(open(path + d_files,"rb"))
            sham_mastoids.append(epochs.get_data(picks='C3').mean(0)*1e6)
        elif cond_dict[cond] == 'up':
            epochs = pickle.load(open(path + d_files,"rb"))
            up_mastoids.append(epochs.get_data(picks='C3').mean(0)*1e6)
        elif cond_dict[cond] == 'down':
            epochs = pickle.load(open(path + d_files,"rb"))
            down_mastoids.append(epochs.get_data(picks='C3').mean(0)*1e6)
    elif d_files.endswith('_ERP_CSD.p'):
        if cond_dict[cond] == 'sham':
            epochs = pickle.load(open(path + d_files,"rb"))
            sham_csd.append(epochs.get_data(picks='C3').mean(0)*1e6)
        elif cond_dict[cond] == 'up':
            epochs = pickle.load(open(path + d_files,"rb"))
            up_csd.append(epochs.get_data(picks='C3').mean(0)*1e6)
        elif cond_dict[cond] == 'down':
            epochs = pickle.load(open(path + d_files,"rb"))
            down_csd.append(epochs.get_data(picks='C3').mean(0)*1e6)
    elif d_files.endswith('_phase_cpz_ref.p'): 
        if cond_dict[cond] == 'sham':
            pha = pickle.load(open(path + d_files,"rb"))
            sham_phase.append(pha)
        elif cond_dict[cond] == 'up':
            pha = pickle.load(open(path + d_files,"rb"))
            up_phase.append(pha)
        elif cond_dict[cond] == 'down':
            pha = pickle.load(open(path + d_files,"rb"))
            down_phase.append(pha) 
    elif d_files.endswith('_phase_lm_ref.p'): 
        if cond_dict[cond] == 'sham':
            pha = pickle.load(open(path + d_files,"rb"))
            sham_phase_lm.append(pha)
        elif cond_dict[cond] == 'up':
            pha = pickle.load(open(path + d_files,"rb"))
            up_phase_lm.append(pha)
        elif cond_dict[cond] == 'down':
            pha = pickle.load(open(path + d_files,"rb"))
            down_phase_lm.append(pha) 
    elif d_files.endswith('_phase_hilbert_cpz_ref.p'): 
        if cond_dict[cond] == 'sham':
            pha = pickle.load(open(path + d_files,"rb"))
            sham_phase_ana.append(pha)
        elif cond_dict[cond] == 'up':
            pha = pickle.load(open(path + d_files,"rb"))
            up_phase_ana.append(pha)
        elif cond_dict[cond] == 'down':
            pha = pickle.load(open(path + d_files,"rb"))
            down_phase_ana.append(pha) 
    elif d_files.endswith('_phase_hilbert_lm_ref.p'): 
        if cond_dict[cond] == 'sham':
            pha = pickle.load(open(path + d_files,"rb"))
            sham_phase_lm_ana.append(pha)
        elif cond_dict[cond] == 'up':
            pha = pickle.load(open(path + d_files,"rb"))
            up_phase_lm_ana.append(pha)
        elif cond_dict[cond] == 'down':
            pha = pickle.load(open(path + d_files,"rb"))
            down_phase_lm_ana.append(pha) 
    elif d_files.endswith('_ITC.p'): 
        if cond_dict[cond] == 'sham':
            itc = pickle.load(open(path + d_files,"rb"))
            sham_itc.append(itc)
        elif cond_dict[cond] == 'up':
            itc = pickle.load(open(path + d_files,"rb"))
            up_itc.append(itc)
        elif cond_dict[cond] == 'down':
            itc = pickle.load(open(path + d_files,"rb"))
            down_itc.append(itc) 
    elif d_files.endswith('_TF.p'): 
        if cond_dict[cond] == 'sham':
            tf = pickle.load(open(path + d_files,"rb"))
            sham_tf.append(itc)
        elif cond_dict[cond] == 'up':
            tf = pickle.load(open(path + d_files,"rb"))
            up_tf.append(itc)
        elif cond_dict[cond] == 'down':
            tf = pickle.load(open(path + d_files,"rb"))
            down_tf.append(itc) 
  
info = mne.create_info(ch_names=['C3'], sfreq=512, ch_types=['eeg'])
epochs_up = mne.EpochsArray(np.expand_dims(np.concatenate(up),1)/1e6, info, tmin = -3, baseline=None)
epochs_up.set_montage(mne.channels.make_standard_montage('standard_1005'))
epochs_down = mne.EpochsArray(np.expand_dims(np.concatenate(down),1)/1e6, info, tmin = -3, baseline=None)
epochs_down.set_montage(mne.channels.make_standard_montage('standard_1005'))
epochs_sham = mne.EpochsArray(np.expand_dims(np.concatenate(sham),1)/1e6, info, tmin = -3, baseline=None)
epochs_sham.set_montage(mne.channels.make_standard_montage('standard_1005'))
epochs_up_mastoids = mne.EpochsArray(np.expand_dims(np.concatenate(up_mastoids),1)/1e6, info, tmin = -3, baseline=None)
epochs_up_mastoids.set_montage(mne.channels.make_standard_montage('standard_1005'))
epochs_down_mastoids = mne.EpochsArray(np.expand_dims(np.concatenate(down_mastoids),1)/1e6, info, tmin = -3, baseline=None)
epochs_down_mastoids.set_montage(mne.channels.make_standard_montage('standard_1005'))
epochs_sham_mastoids = mne.EpochsArray(np.expand_dims(np.concatenate(sham_mastoids),1)/1e6, info, tmin = -3, baseline=None)
epochs_sham_mastoids.set_montage(mne.channels.make_standard_montage('standard_1005'))
epochs_sham_csd = mne.EpochsArray(np.expand_dims(np.concatenate(sham_csd),1)/1e6, info, tmin = -3, baseline=None)
epochs_sham_csd.set_montage(mne.channels.make_standard_montage('standard_1005'))
epochs_down_csd = mne.EpochsArray(np.expand_dims(np.concatenate(down_csd),1)/1e6, info, tmin = -3, baseline=None)
epochs_down_csd.set_montage(mne.channels.make_standard_montage('standard_1005'))
epochs_up_csd = mne.EpochsArray(np.expand_dims(np.concatenate(up_csd),1)/1e6, info, tmin = -3, baseline=None)
epochs_up_csd.set_montage(mne.channels.make_standard_montage('standard_1005'))

mne.viz.plot_compare_evokeds(dict(up_state=epochs_up_mastoids.average(method='mean', picks='C3'),
                                  down_state=epochs_down_mastoids.average(method='mean', picks='C3'),
                                  sham_state=epochs_sham_mastoids.average(method='mean', picks='C3')), 
                             legend='upper left', show_sensors='upper right',
                             title='Group level average pinknoise epochs - C3 (mastoids)')
# plt.savefig(fig_path + files.split('/')[-2] + '_average_epoch.png')
mne.viz.plot_compare_evokeds(dict(up_state=epochs_up.average(method='mean', picks='C3'),
                                  down_state=epochs_down.average(method='mean', picks='C3'),
                                  sham_state=epochs_sham.average(method='mean', picks='C3')), 
                             legend='upper left', show_sensors='upper right',
                             title='Group level average pinknoise epochs - C3')
# plt.savefig(fig_path + files.split('/')[-2] + '_average_epoch_mastoids.png')
mne.viz.plot_compare_evokeds(dict(up_state=epochs_up_csd.average(method='mean', picks='C3'),
                                  down_state=epochs_down_csd.average(method='mean', picks='C3'),
                                  sham_state=epochs_sham_csd.average(method='mean', picks='C3')), 
                             legend='upper left', show_sensors='upper right',
                             title='Group level average pinknoise epochs - C3 (CSD)')
# plt.savefig(fig_path + files.split('/')[-2] + '_average_epoch_csd.png')

for x,y in zip([-3.0, 0.],[0., 3.0]):
    epochs_up_mastoids.plot_psd_topomap(tmin=x, tmax=y, bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                                                              (12, 16, 'Sigma'), (16, 30, 'Beta')], ch_type='eeg',
                                     dB=False, normalize = True, cmap='Spectral_r')
    # plt.savefig(fig_path + files.split('/')[-2] + '_power_' + str(x) + 's_' + str(y) + 's_C3_lm.png')
    epochs_up_csd.plot_psd_topomap(tmin=x, tmax=y, bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                                                         (12, 16, 'Sigma'), (16, 30, 'Beta')],
                                dB=False, normalize = True, cmap='Spectral_r')
    # plt.savefig(fig_path + files.split('/')[-2] + '_power_' + str(x) + 's_' + str(y) + 's_C3_lm.png')
    plt.close('all')
    
up_phase_1 = np.concatenate(up_phase_lm_ana)[~np.isnan(np.concatenate(up_phase_lm_ana))]
down_phase_1 = np.concatenate(down_phase_lm_ana)[~np.isnan(np.concatenate(down_phase_lm_ana))]
sham_phase_1 = np.concatenate(sham_phase_lm_ana)[~np.isnan(np.concatenate(sham_phase_lm_ana))]

up_rv = pg.circ_r(up_phase_1)
down_rv = pg.circ_r(down_phase_1)
sham_rv = pg.circ_r(sham_phase_1)

up_phi = pg.circ_mean(up_phase_1)
down_phi = pg.circ_mean(down_phase_1)
sham_phi = pg.circ_mean(sham_phase_1)
    
df = pd.concat([pd.DataFrame({'Phase targets' : up_phase_1, 'Condition' : 'Up', 'Resultant vector' : up_rv, 'Circular mean' : up_phi}),
                pd.DataFrame({'Phase targets' : down_phase_1, 'Condition' : 'Down', 'Resultant vector' : down_rv, 'Circular mean' : down_phi}),
                pd.DataFrame({'Phase targets' : sham_phase_1, 'Condition' : 'Sham', 'Resultant vector' : sham_rv, 'Circular mean' : sham_phi})], 
                ignore_index=True)


g = sns.FacetGrid(df, col= 'Condition', hue='Condition',
                  subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=True, sharey=True, despine=False)

g.map_dataframe(sns.histplot, 'Phase targets', stat = 'density')

kwargs_arrow={'fc': 'tab:red', 'ec': 'tab:red'}
for ax, phi_idx, rv_idx in zip(range(len(g.axes[0,:])), [up_phi, down_phi, sham_phi], 
                               [up_rv, down_rv, sham_rv]):
    g.axes[0,ax].arrow(0, 0, phi_idx, rv_idx, **kwargs_arrow)
  
# Individual plots
for var,title in zip([up_phase, down_phase, sham_phase], 
                     ['Up-stimulation', 'Down-stimulation', 'Sham-stimulation']):
    plt.figure()
    ax1 = plt.subplot(111, projection='polar')
    ax1 = sns.histplot(var)
    # ax1.hist(var)
    # circular mean (add 2*pi to make between 0 and 2*pi)
    phi = pg.circ_mean(var) #+ 2*np.pi
    print('Circular mean: %.3f rad' %  phi)
    # resultant vector length
    rv = pg.circ_r(var)
    print('Vector length: %.3f' % rv)
    # circular mean plot
    ax2 = pg.plot_circmean(var, kwargs_markers=dict(marker="None"))
    
    # create arrow for circular mean and resulatant vector length
    kwargs_arrow={'width': 0.01, 'head_width': 0.1, 'head_length': 0.1, 
                  'fc': 'tab:red', 'ec': 'tab:red'}

    
    ax1.set_title("Phase targeting accuracy : " + title, va='bottom')
    ax2.set_title("Circular mean : " + title)
    


#%%
## Phase analysis 
C3 = epochs.get_data()[:,Data.chans.index('C3'),:]*1e6
 
# Identify slow waves in data
sws = [yasa.sw_detect(data = C3[i,:], sf=512, ch_names=['C3'], freq_sw=(0.5, 2), 
                      amp_neg=(35, 300), amp_pos=(5, 200), amp_ptp=(75, 500)) for i in range(min(np.shape(C3)))]
sws_idx = []
for i in range(len(sws)):
    if sws[i] != None:
        sws_idx.append(1)
    else:
        sws_idx.append(0)
        
sws_idx = np.asarray(sws_idx)

# Compute instantaneous phase from a signal
n_samples = max(C3.shape)
nfast = next_fast_len(n_samples)
#sw_pha = [np.angle(hilbert(C3[i,:], N=nfast)[:n_samples]) for i in range(C3.shape[0])]
sw_pha = [np.angle(hilbert(C3[i,:], N=nfast)[:n_samples]) for i in np.where(sws_idx == 1)[0]]
pn_phase = [sw_pha[i][int(Data.sfreq*1.5)] for i in range(len(sw_pha))]

ax = plt.subplot(111, projection='polar')
ax.hist(pn_phase)
ax.set_title("Phase targeting accuracy (Analytical): {files.split("/")[-2]} - C3}", va='bottom')

half1, half2, half3, half4 = [],[],[],[]
for i in np.where(sws_idx == 1)[0]:
    half1.append(np.linspace(sws[i].summary()['Start'][0]*Data.sfreq, 
                             sws[i].summary()['NegPeak'][0]*Data.sfreq, num=90, endpoint=True).astype(int))
    half2.append(np.linspace(sws[i].summary()['NegPeak'][0]*Data.sfreq + 1, 
                             sws[i].summary()['MidCrossing'][0]*Data.sfreq, num=90).astype(int))
    half3.append(np.linspace(sws[i].summary()['MidCrossing'][0]*Data.sfreq + 1, 
                             sws[i].summary()['PosPeak'][0]*Data.sfreq, num=90).astype(int))
    half4.append(np.linspace(sws[i].summary()['PosPeak'][0]*Data.sfreq + 1, 
                             sws[i].summary()['End'][0]*Data.sfreq, num=90).astype(int))
    
crossings = []
for i in range(len(half1)):
    crossing = thresholdcrossings(np.concatenate([half1[i], half2[i], half3[i], half4[i]]), Data.sfreq*1.5)
    if len(crossing) != 0:
        crossings.append(crossing[0])
    else:
        crossings.append(np.nan)
# idx = np.nonzero(crossings)[0]

# Plot some figures
plt.figure() 
plt.plot(C3[np.where(sws_idx == 0)[0],:].mean(0), label = 'PN modulation - No SW')
plt.plot(C3[np.where(sws_idx == 1)[0],:].mean(0), label = 'PN modulation - SW')
plt.plot(C3.mean(0), label = 'PN modulation - All')
plt.legend()
   
# takes radians 
plt.figure()
ax2 = plt.subplot(111, projection='polar')
ax2.hist(np.deg2rad(crossings))
ax2.set_title("Phase targeting accuracy : {files.split("/")[-2]} - C3}", va='bottom')


# ptps = [np.ptp(C3[i,0:Data.sfreq*2]) for i in range(np.shape(C3)[0])]
# loc = np.where(np.asarray(ptps)>75)[0]

# for i in loc:
#     plt.figure()
#     plt.plot(C3_pn_narrow[i,0:Data.sfreq*3])
#     plt.vlines(Data.sfreq*1.5, ymin=-75, ymax=40)

