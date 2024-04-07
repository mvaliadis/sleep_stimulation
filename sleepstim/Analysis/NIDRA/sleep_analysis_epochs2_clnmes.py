#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 16:42:03 2024

@author: administrator
"""
import os
import mne
from mne_connectivity import spectral_connectivity_epochs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import glob
from tqdm import tqdm
from neurodsp.timefrequency import amp_by_time, phase_by_time
import tensorpac.methods as tpm
from tensorpac import EventRelatedPac, Pac
from scipy.signal import hilbert
from scipy.fftpack import next_fast_len
from sleepstim.Analysis.NIDRA import pci
from sleepstim.Analysis.NIDRA import tct
import pingouin as pg
import pandas as pd
import yasa
import fooof
import neurokit2 as nk
from sleepecg import detect_heartbeats
mne.set_log_level('ERROR')

def analyze_sleep_nmes_epochs_tfr(epochs, subject, night,
                                  freq_range = (0.5,30), 
                                  steps = 0.25,  
                                  baseline=(-3,3), 
                                  mode='zscore', 
                                  length=(-3,3), 
                                  plot=True, 
                                  cmap = cm.Spectral_r):            
    
    Sxx_dict = []
    
    # give frequencies to determine # of cycles 
    freqs = np.arange(freq_range[0], freq_range[1]+steps, steps)
    # n_cyc = freqs
    
    # Define the starting cycles and the increment per Hz
    start_cycles = 4.5
    cycle_increment_per_hz = 0.5
    # Calculate the increment for each frequency starting from the lowest frequency
    cycle_increments = (freqs - freq_range[0]) * cycle_increment_per_hz
    # Calculate cycles for each frequency
    n_cyc = start_cycles + cycle_increments
   
    # compute tfr via wavelet method
    Sxx = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles=n_cyc, picks='csd',
                                        zero_mean=True, use_fft=True, decim=5, 
                                        output='power', n_jobs=-1, verbose=None,
                                        average=False, return_itc=False)
        
    # apply baseline correction
    Sxx = Sxx.apply_baseline(baseline, mode=mode)
        
    # Create a new dictionary for the tfr
    for cond in Sxx.event_id.keys():
        if 'c3' in cond:
            coi = 'C3'
        elif 'fz' in cond:
            coi = 'Fz'
            
        dict_Sxx = {
            'Subject': subject,
            'Night': night,
            'Stim': cond.split('_')[1],
            'Mode': cond.split('_')[0],
            'Target_chan': cond.split('_')[2], 
            'TFR': Sxx.average(),
        }
        
        # Append the dictionary to the list
        Sxx_dict.append(dict_Sxx)     
            
    # Plot TFR
    if plot: 
        # average TFR data, remove edge, and alter timepoints accordingly for different conds
        for cond in Sxx.event_id.keys():
            if 'c3' in cond:
                coi = 'C3'
            elif 'fz' in cond:
                coi = 'Fz'
            Sxx_ = Sxx[cond].copy().crop(tmin=length[0],tmax=length[1]).average().pick(coi).data
            times = Sxx[cond].copy().crop(tmin=length[0],tmax=length[1]).times
            
            fig, ax = plt.subplots()
            vmin, vmax = np.percentile(Sxx_, [0 + 0.1, 100 - 0.1])
            norm = Normalize(vmin=vmin, vmax=vmax)
            CM = ax.pcolormesh(times, freqs, Sxx_.squeeze(), shading='gouraud', cmap=cmap, 
                               rasterized = True, antialiased=True, norm=norm)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            cbar = plt.colorbar(CM, ax=ax)
            cbar.set_label(mode + ' (dB)')
            plt.tight_layout()
                       
            # Plotting with MNE       
            # Sxx[cond].average().plot(tmin=-2, tmax=2, 
            #                    # baseline=[-3, 3], mode='zscore', 
            #                    fmin=5, fmax=25, picks='C3')
            # Sxx[cond].average().plot_joint(tmin=-2, tmax=2, fmin=5, fmax=25,
            #                          # baseline=[-3, 3], mode='zscore', 
            #                          timefreqs=[(0, 12), (2, 12)])
            
            fig_path = '/media/administrator/Sleep_Data/Processed/Figures/Sleep/'
            # fig_path = '/media/administrator/data/Study_2_data/NIDRA/Figures/'
            plt.savefig(fig_path + f'{cond}_{subject}_{night}_tfr.png')
            plt.close('all')
          
    return pd.DataFrame(Sxx_dict)

def phase_extraction(epochs, method='neurodsp', coi='C3'):
    if method == 'neurodsp':
        # Extract instantaneous phase using Hilbert transform
        sig = epochs.get_data(coi).squeeze()*1e3
        inst_phase = [phase_by_time(sig[i, :], fs=epochs.info['sfreq'], f_range=(None, 2)) 
                      for i in range(min(sig.shape))]
        inst_phase = np.concatenate([inst_phase])
        # Extract phase at stimulation
        stim_phase = inst_phase[:, epochs.time_as_index(0)[0]]
    if method == 'hilbert':
        # Filter data
        epochs.copy().filter(None, 2)
        # Extract instantaneous phase using Hilbert transform
        n = epochs._data.shape[-1]
        nfast = next_fast_len(n)
        inst_phase = np.angle(hilbert(epochs, N=nfast)[:n])
        # Extract phase at stimulation
        stim_phase = inst_phase[:, epochs.ch_names.index(coi),
                                epochs.time_as_index(0)[0]]
    
    return stim_phase

def analyze_sleep_nmes_epochs_phase(epochs, subject, night):  
    # Extract SO phase
    phases = []
    for chan in ('C3','Fz'):
        for stim in ('sham', 'stim'):
            # print(chan, stim)
            try:
                stim_phase = phase_extraction(epochs[f'nmes_{stim}_{chan.lower()}'], 
                                              method='neurodsp', coi=chan)
                mode = 'nmes'
            except:
                stim_phase = phase_extraction(epochs[f'pn_{stim}_{chan.lower()}'], 
                                              method='neurodsp', coi=chan)
                mode = 'pn'
                
            # pg.circ_mean(stim_phase)
            # pg.plot_circmean(stim_phase)
            
            # Create a new dictionary for the current channel and stimulation
            phase_dict = {
                'Subject': subject,
                'Night': night,
                'Target_Chan': chan,
                'Stim': stim,
                'Mode': mode,
                'CircMean': pg.circ_mean(stim_phase) ,
                'RVL' : pg.circ_r(stim_phase),
            }
            
            # Append the dictionary to the list
            phases.append(phase_dict) 
            
    # Convert to df
    df = pd.DataFrame(phases) 
        
    return df

def get_inst_hr(epoch, sf, duration):
    pks = detect_heartbeats(epoch, fs=sf)
    # compute ecg signal quality
    hr_quality = nk.ecg_quality(epoch, sampling_rate=sf,
                                method="zhao2018", approach="fuzzy")
    # If not enough R-peaks were detected, skip epochs and return NaN
    # Here, we assume a minimal HR of 30 bpm
    constant_hr = 60 * (pks.size / (duration / sf))
    if constant_hr < 30:
        print("Too few detected heartbeats in epoch.")
        # continue

    # get instantaneous heart rate        
    hr = nk.signal_rate(pks, sampling_rate=sf, 
                        desired_length=len(epoch), 
                        interpolation_method='monotone_cubic', 
                        show=False)
    
    return hr, hr_quality
        
def analyze_sleep_nmes_hr(epochs, subject, night):
    sf = epochs.info['sfreq']
    duration = len(epochs.times)
    hrs = []
    for cond in epochs.event_id.keys():
        data = epochs[cond].get_data('ecg', units='uV').squeeze()
        for idx, epoch in enumerate(data):
            hr, hr_quality = get_inst_hr(epoch, sf, duration)
            if hr_quality != 'Unacceptable':
                hr_pre = np.nanmean(hr[0:len(hr)//2])
                hr_post = np.nanmean(hr[len(hr)//2:])
                hr_ratio_change = ((hr_post - hr_pre)/hr_pre)*100
            
                # Save rpeaks to dict
                hr_dict = {
                    'Subject': subject, 
                    'Night': night,
                    'Condition': cond,
                    'Epoch': idx,
                    'Quality': hr_quality, 
                    'HR_ratio_change': hr_ratio_change,
                    }
                
                hrs.append(hr_dict)
            else:
                continue
            
    df = pd.DataFrame(hrs)
    
    # Split the 'Condition' column into 'Stim/Sham' and 'Target Chan' columns
    df['Mode'], df['Stim'], df['Target_Chan'] = zip(*df['Condition'].apply(lambda x: x.split('_')))
    df.drop('Condition', axis=1, inplace=True)
    
    # Mean over epochs
    df = df.groupby(['Subject','Night','Mode','Stim','Target_Chan']).mean()['HR_ratio_change'].reset_index()
    
    return df
   
def analyze_sleep_nmes_epochs_power(epochs, subject, night, plot=False): 
    # Initialize list
    df = []
    
    # Compute power spectral and fit FOOOF models
    for cond in epochs.event_id.keys():
        
        # Compute PSD using the Welch method
        epo_spectrum = epochs[cond].compute_psd(method='welch', fmin=0.5, fmax=30, n_jobs=-1, 
                                                picks='csd', **dict(average='median', 
                                                                    n_fft=int(4*epochs.info['sfreq'])))
        psds, freqs = epo_spectrum.get_data(return_freqs=True)
        
        # Get relative power with YASA 
        power_df = yasa.bandpower_from_psd(psds.mean(0), freqs, ch_names=epo_spectrum.ch_names,
                                           bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'),
                                                  (8, 12, 'Alpha'), (12, 16, 'Sigma'),
                                                  (16, 30, 'Beta')])
        power_df.insert(0, 'Subject', subject)
        power_df.insert(1, 'Night', night)
        power_df.insert(2, 'Condition', cond)
        power_df.insert(3, 'Mode', cond.split('_')[0])
        
        # FOOOF data         
        fm = fooof.FOOOFGroup(max_n_peaks=5)
        fm.fit(freqs, psds.mean(0))
        
        # Extract aperiodic components for all channel
        power_df['Offset'] = fm.get_params(name='aperiodic_params', col='offset')
        power_df['Aperiodic'] = fm.get_params(name='aperiodic_params', col='exponent')           
    
        # Define and plot frequency bands of interest
        if plot:
            bands = {'Delta (0.5-4 Hz)': (0.5, 4), 'Theta (4-8 Hz)': (4, 8), 
                      'Alpha (8-12 Hz)': (8, 12), 'Sigma (12-16 Hz)': (12, 16), 
                      'Beta (16-30 Hz)': (16, 30)}
            
            epo_spectrum.plot_topomap(bands=bands, normalize=True)
            plt.suptitle(f'{cond.capitalize()}')
            plt.close('all')
            
        df.append(power_df)
    
    # Create df
    df = pd.concat(df).reset_index(drop=True)
        
    # Split the 'Condition' column into 'Stim/Sham' and 'Target Chan' columns
    df['Stim'], df['Target_Chan'] = zip(*df['Condition'].apply(lambda x: x.split('_')[1:]))
    df.drop('Condition', axis=1, inplace=True)
    
    return df

def analyze_sleep_nmes_epochs_tct(epochs, subject, night, method='spearman'):
    tcts = []
    for cond in epochs.event_id.keys():
        consistency_gfp = tct.calculate_gfp_correlation(epochs[cond], method=method)
        consistency_post = tct.calculate_topographic_consistency(epochs[cond])
        gfp_post_trial = tct.calculate_gfp_strength(epochs[cond], method='trial_avg')
        gfp_post_evoked = tct.calculate_gfp_strength(epochs[cond], method='evoked_avg')
        if 'c3' in cond:
            coi = 'C3'
        elif 'fz' in cond:
            coi = 'Fz'
        erp = epochs[cond].get_data(coi, tmin=0, tmax=.2).squeeze().mean(1).mean()*1e3
        # Create a new dictionary for the current channel and stimulation
        tct_dict = {
            'Subject': subject,
            'Night': night,
            'Condition': cond, 
            'Mode': cond.split('_')[0], 
            'Consistency_GFP': consistency_gfp,
            'Consistency': consistency_post,
            'GFP_Trial': gfp_post_trial, 
            'GFP_Evoked': gfp_post_evoked, 
            'ERP': erp,
        }
        
        # Append the dictionary to the list
        tcts.append(tct_dict)   
        
    # Convert to df
    df = pd.DataFrame(tcts)
    
    # Split the 'Condition' column into 'Stim/Sham' and 'Target Chan' columns
    df['Stim'], df['Target_Chan'] = zip(*df['Condition'].apply(lambda x: x.split('_')[1:]))
    df.drop('Condition', axis=1, inplace=True)
  
    return df
    
def analyze_sleep_nmes_epochs_pci(epochs, subject, night):  
    # Extract PCI
    pcis = []
    par = {'baseline_window':(-3,-1.5), 
           'response_window':(.01, 1.0), 
           'k':1.2, 
           'min_snr':1.1, 
           'max_var':99,
           'embed':False,
           'n_steps':100, 
           'avgref': False}
    for chan in ('C3','Fz'):
        for stim in ('sham', 'stim'):
            try:
                evk = epochs[f'nmes_{stim}_{chan.lower()}'].average()
            except:
                evk = epochs[f'pn_{stim}_{chan.lower()}'].average()
            pci_ = pci.calc_PCIst(evk.get_data()*1e3, evk.times, **par)

            # Create a new dictionary for the current channel and stimulation
            pci_dict = {
                'Subject': subject,
                'Night': night,
                'Target_chan': chan,
                'Stim': stim,
                'Mode': list(epochs.event_id)[0].split('_')[0], 
                'PCI': pci_,
            }
            
            # Append the dictionary to the list
            pcis.append(pci_dict)  
            
    # Convert to df
    df = pd.DataFrame(pcis) 
    
    return df

def analyze_sleep_nmes_epochs_ndPAC(epochs, subject, night, method='tensorpac'):
    # Initialize the list to hold all ndPAC data
    ndPACs = []
    for cond in epochs.event_id.keys():
        epoch = epochs[cond]
        if method=='tensorpac':
            p = EventRelatedPac(f_pha=[0.5, 2], f_amp=[12, 16])
            p = Pac(f_pha=[0.5, 2], f_amp=[5, 25, 0.25, 0.25])
        for chan in epoch.ch_names[0:64]:
            # Get channel data
            data = epoch.copy().get_data(chan).squeeze()*1e3
            if method=='tensorpac':
                # Get channel data
                sf = epoch.info['sfreq']
                sw_pha = p.filter(sf, data, ftype='phase', edges=int(sf), n_jobs=-1).squeeze()
                sp_amp = p.filter(sf, data, ftype='amplitude', edges=int(sf), n_jobs=-1).squeeze()
            else:
                # Extract instantaneous phase using Hilbert transform
                sw_pha = [phase_by_time(data[i, :], fs=epochs.info['sfreq'],
                                      f_range=(None, 2)) 
                        for i in range(min(data.shape))]
                sw_pha = np.concatenate([sw_pha])
                # Extract instantaneous amp using Hilbert transform
                sp_amp = [amp_by_time(data[i, :], fs=epochs.info['sfreq'],
                                      f_range=(12, 16)) 
                          for i in range(min(data.shape))]
                sp_amp = np.concatenate([sp_amp])
                
            # Extract phase at pre/post stimulation
            start_idx, end_idx = epoch.time_as_index((-1.2, 0))
            pre_sw_pha = sw_pha[:, start_idx:end_idx]
            start_idx2, end_idx2 = epoch.time_as_index((0, 1.2))
            post_sw_pha = sw_pha[:, start_idx2:end_idx2]
        
            # Extract phase at pre/post stimulation
            pre_sp_amp = sp_amp[:, start_idx:end_idx]
            post_sp_amp = sp_amp[:, start_idx2:end_idx2]
            
            # ndPAC calculation 
            # Find location of max sigma amplitude in pre/post epoch
            idx_max_amp_pre = pre_sp_amp.argmax(axis=1).reshape(-1, 1)
            idx_max_amp_post = post_sp_amp.argmax(axis=1).reshape(-1, 1)
            
            # Adjust indices to times relative to window start
            time_per_sample = 1 / epoch.info['sfreq']
            time_in_seconds_pre = idx_max_amp_pre * time_per_sample
            time_in_seconds_post = idx_max_amp_post * time_per_sample
            time_relative_to_start_pre = time_in_seconds_pre - 1.2
            time_relative_to_start_post = time_in_seconds_post + 0
            
            # PhaseAtSigmaPeak
            # Find SW phase at max sigma amplitude in epoch            
            pha_at_max_pre = np.squeeze(np.take_along_axis(pre_sw_pha,
                                                           idx_max_amp_pre,
                                                           axis=1))
            pha_at_max_post = np.squeeze(np.take_along_axis(pre_sw_pha,
                                                            idx_max_amp_post,
                                                            axis=1))

            # Normalized Direct PAC, with thresholding
            ndp_pre = np.squeeze(tpm.norm_direct_pac(pre_sw_pha[None, :],
                                                     pre_sp_amp[None, :], p=0.05))
            ndp_post = np.squeeze(tpm.norm_direct_pac(post_sw_pha[None, :],
                                                      post_sp_amp[None, :], p=0.05))
                               
            # Tuple pairs of conditions and their respective data for looping
            conditions_data = [
                ("Pre", time_relative_to_start_pre, pha_at_max_pre, ndp_pre),
                ("Post", time_relative_to_start_post, pha_at_max_post, ndp_post)
            ]
            
            # Loop through each condition to process and append the data
            for session, time_rel_start, pha_at_max, ndp in conditions_data:            
                # Calculate medians or other statistics as needed
                mean_time_rel_start = np.nanmean(time_rel_start)
                mean_pha_at_max = pg.circ_mean(pha_at_max) # circular mean
                mean_ndp = np.nanmean(ndp)
            
                # Create dictionary for the current session
                dict_ndpac = {
                    'Subject': subject,
                    'Night': night,
                    'Condition': cond,
                    'Session': session,
                    'Mode': cond.split('_')[0],
                    'Chan': chan, 
                    'SigmaPeakTime': mean_time_rel_start,
                    'PhaseAtSigmaPeak': mean_pha_at_max,
                    'ndPAC': mean_ndp,
                }
            
                # Append the dictionary to the ndPACs list
                ndPACs.append(dict_ndpac)
    
    # Convert to df
    df = pd.DataFrame(ndPACs) 
    
    # Split the 'Condition' column into 'Stim/Sham' and 'Target Chan' columns
    df['Stim'], df['Target_Chan'] = zip(*df['Condition'].apply(lambda x: x.split('_')[1:]))
    df.drop('Condition', axis=1, inplace=True)
    
    return df

def analyze_sleep_nmes_epochs_erpac(epochs, subject, night, plot=False):
    erpacs = []
    for cond in epochs.event_id.keys():
        epoch = epochs[cond]
        edges=int(1*epoch.info['sfreq'])
        if 'c3' in cond:
            coi = 'C3'
        elif 'fz' in cond:
            coi = 'Fz'
        p = EventRelatedPac(f_pha=[0.5, 2], f_amp=(5, 25, .25, .25))
        erpac = p.filterfit(int(epoch.info['sfreq']), epoch.get_data(picks=coi).squeeze()*1e3,
                            method='gc', smooth=100, edges=edges, n_perm=200, 
                            mcp='fdr', n_jobs=-1).squeeze()
        
        # Create a new dictionary for the erpac 
        dict_erpac = {
            'Subject': subject,
            'Night': night,
            'Condition': cond, 
            'Mode': cond.split('_')[0],
            'ERPAC': erpac,
        }
        
        erpacs.append(dict_erpac)
        
        if plot:        
            plt.figure(figsize=(8, 4))
            p.pacplot(erpac, epoch.times[edges:-edges], p.yvec, 
                      xlabel='Time (second)', cmap='Spectral_r',
                      # title='Event-Related PAC occurring for Delta phase',
                      # fz_labels=15, fz_title=18)
                      ylabel='Amplitude frequency (Hz)', title=p.method,
                      cblabel='ERPAC', vmin=0., rmaxis=True)
            plt.axvline(0., linestyle='--', color='w', linewidth=2)
            
            plt.tight_layout()
            p.show()
            
            fig_path = '/media/administrator/Sleep_Data/Processed/Figures/Sleep/'
            # fig_path = '/media/administrator/data/Study_2_data/NIDRA/Figures/'
            p.savefig(fig_path + f'{cond}_{subject}_{night}_erpac.png')
            plt.close('all')
        
    # Convert to df
    df = pd.DataFrame(erpacs) 
    
    # Split the 'Condition' column into 'Stim/Sham' and 'Target Chan' columns
    df['Stim'], df['Target_Chan'] = zip(*df['Condition'].apply(lambda x: x.split('_')[1:]))
    df.drop('Condition', axis=1, inplace=True)
    
    return df

def analyze_sleep_nmes_epochs_granger(epochs, subject, night, plot=False):
    # Include only eeg sensors post stimulus onset 
    epochs.crop(tmin=0, tmax=2).pick('csd')
    
    # lags 
    lags = 20 
    
    # frontal sensors
    signals_a = [epochs.ch_names.index(ch_idx) for ch_idx in (['F1','Fz','F2'])]
    
    # motor sensors
    signals_b = [epochs.ch_names.index(ch_idx) for ch_idx in (['C5','C3','C1'])]

    # indices
    indices_ab = (np.array([signals_a]), np.array([signals_b]))  # A => B
    indices_ba = (np.array([signals_b]), np.array([signals_a]))  # B => A
    
    trgc_df = []
    for cond in epochs.event_id.keys():
        # 1. compute Granger causality
        gc_ab = spectral_connectivity_epochs(
            epochs[cond],
            sfreq=epochs.info['sfreq'],
            method=["gc"],
            indices=indices_ab,
            fmin=0.5,
            fmax=30,
            #rank=(np.array([3]), np.array([3])),
            gc_n_lags=lags,
        )  # A => B
        gc_ba = spectral_connectivity_epochs(
            epochs[cond],
            sfreq=epochs.info['sfreq'],
            method=["gc"],
            indices=indices_ba,
            fmin=0.5,
            fmax=30,
            #rank=(np.array([3]), np.array([3])),
            gc_n_lags=lags,
        )  # B => A
        freqs = gc_ab.freqs
        
        # Drivers and receivers: analysing the net direction of information flow
        net_gc = gc_ab.get_data() - gc_ba.get_data()  # [A => B] - [B => A]
        
        # 2. compute GC on time-reversed signals
        gc_tr_ab = spectral_connectivity_epochs(
            epochs[cond],
            sfreq=epochs.info['sfreq'],
            method=["gc_tr"],
            indices=indices_ab,
            fmin=0.5,
            fmax=30,
            #rank=(np.array([3]), np.array([3])),
            gc_n_lags=lags,
        )  # TR[A => B]
        gc_tr_ba = spectral_connectivity_epochs(
            epochs[cond],
            sfreq=epochs.info['sfreq'],
            method=["gc_tr"],
            indices=indices_ba,
            fmin=0.5,
            fmax=30,
            #rank=(np.array([3]), np.array([3])),
            gc_n_lags=lags,
        )  # TR[B => A]
        
        # compute net GC on time-reversed signals (TR[A => B] - TR[B => A])
        net_gc_tr = gc_tr_ab.get_data() - gc_tr_ba.get_data()
    
        # compute TRGC
        trgc = net_gc - net_gc_tr
        
        # dict
        trgc_dict = {
            'Subject': [subject]*len(freqs),
            'Night': [night]*len(freqs),
            'Condition': [cond]*len(freqs),
            'Mode': [list(epochs.event_id)[0].split('_')[0]]*len(freqs), 
            'Freqs': freqs,
            'Net_GC': net_gc.squeeze(),
            'TRGC': trgc.squeeze(),
        }
        trgc_df.append(pd.DataFrame(trgc_dict))
        
        if plot:
            fig, axis = plt.subplots(1, 1)
            axis.plot((freqs[0], freqs[-1]), (0, 0), linewidth=2, linestyle="--", color="k")
            axis.plot(freqs, trgc[0], linewidth=2)
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel("Connectivity (A.U.)")
            fig.suptitle(f"{cond} TRGC: net[A => B] - net time-reversed[A => B]")
            plt.show()
            
    # Convert to df
    df = pd.concat(trgc_df).reset_index(drop=True)
    
    # Split the 'Condition' column into 'Stim/Sham' and 'Target Chan' columns
    df['Stim'], df['Target_Chan'] = zip(*df['Condition'].apply(lambda x: x.split('_')[1:]))
    df.drop('Condition', axis=1, inplace=True)
      
    return df
        
def create_evoked_contrasts(df):
    evoked_list = []
    
    for index, row in df.iterrows():
        epochs = row['Epochs']
        subject = row['Subject']
        night = row['Night']
        mode = row['Mode']
        
        # Compute evoked responses
        try:
            evoked_sham_c3 = epochs['nmes_sham_c3'].average()
            evoked_stim_c3 = epochs['nmes_stim_c3'].average()
            evoked_sham_fz = epochs['nmes_sham_fz'].average()
            evoked_stim_fz = epochs['nmes_stim_fz'].average()
        except:
            evoked_sham_c3 = epochs['pn_sham_c3'].average()
            evoked_stim_c3 = epochs['pn_stim_c3'].average()
            evoked_sham_fz = epochs['pn_sham_fz'].average()
            evoked_stim_fz = epochs['pn_stim_fz'].average()
        
        # Compute contrasts
        contrast_c3 = mne.combine_evoked([evoked_stim_c3, evoked_sham_c3], weights=[1, -1])
        contrast_fz = mne.combine_evoked([evoked_stim_fz, evoked_sham_fz], weights=[1, -1])
         
        # For C3 - Fz contrast
        contrast_c3_fz = mne.combine_evoked([contrast_c3, contrast_fz], weights=[1, -1])
        
        # Store in a list or dictionary
        evoked_list.append({
            'Subject': subject,
            'Night': night,
            'Mode': mode,
            'Evoked_sham_c3': evoked_sham_c3,
            'Evoked_stim_c3': evoked_stim_c3,
            'Evoked_sham_fz': evoked_sham_fz,
            'Evoked_stim_fz': evoked_stim_fz,
            'Contrast_c3': contrast_c3,
            'Contrast_fz': contrast_fz,
            'Contrast_c3_fz': contrast_c3_fz
        })
    
    return evoked_list

def create_gavs(df):
    # Create different GAV traces
    df = df.set_index(['Subject','Night','Mode'])
    gavs = []
    unique_modes = df.index.get_level_values('Mode').unique()
    for mode in unique_modes:
        # Filter the DataFrame for the current mode
        df_mode = df.xs(mode, level='Mode')
        
        for con in df_mode.columns:
            # Collect all Evoked objects for the current condition and mode
            evokeds = [evoked for evoked in df_mode[con] if isinstance(evoked, mne.Evoked)]
            
            if evokeds:
                # Compute the grand average for the current condition and mode
                gav = mne.grand_average(evokeds)
                
                # Store the result
                gavs.append({
                    'Contrast': con,
                    'GAV': gav,
                    'Mode': mode, 
                })
                
    return gavs
                

    # Create different GAV traces
    df = df.set_index(['Subject','Night','Mode'])
    gavs = []
    unique_modes = df.index.get_level_values('Mode').unique()
    for mode in unique_modes:
        # Filter the DataFrame for the current mode
        df_mode = df.xs(mode, level='Mode')
        
        for con in df_mode.columns:
            # Collect all Evoked objects for the current condition and mode
            evokeds = [evoked for evoked in df_mode[con] if isinstance(evoked, mne.Evoked)]
            
            if evokeds:
                # Compute the grand average for the current condition and mode
                gav = mne.grand_average(evokeds)
                
                # Store the result
                gavs.append({
                    'Contrast': con,
                    'GAV': gav,
                    'Mode': mode, 
                })
                
    return gavs
                
def load_or_initialize_df(filepath, columns=None):
    if os.path.exists(filepath):
        if filepath.endswith('.csv'):
            try:
                return pd.read_csv(filepath, index_col=0)
            except Exception as e:
                print(f"Error reading {filepath} as CSV: {e}")
                raise
        elif filepath.endswith('.p'):
            try:
                return pd.read_pickle(filepath)
            except Exception as e:
                print(f"Error reading {filepath} as pickle: {e}")
                raise
        else:
            raise ValueError(f"Unsupported file extension for {filepath}")
    else:
        if columns is None:
            return pd.DataFrame()
        else:
            return pd.DataFrame(columns=columns)
    
#%%
if __name__ == '__main__':
    path = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/*csd-epo.fif'
    stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'
    epochs_all = []
    df_phase_all = []
    df_power_all = []
    df_gc_all = []
    df_tct_all = []
    df_pci_all = []
    df_erpac_all = []
    df_ndpac_all = []
    df_tfr_all = []
    df_inst_hr_all = []
    
    # Paths to the dataframes
    paths_to_dfs = {
        'df_inst_hr': stats_path + 'df_inst_hr.csv',
        'df_phase': stats_path + 'df_phase.csv',
        'df_power': stats_path + 'df_power.csv',
        'df_gc': stats_path + 'df_gc.csv',
        'df_tct': stats_path + 'df_tct.csv',
        'df_pci': stats_path + 'df_pci.csv',
        'df_tfr': stats_path + 'df_tfr.p',
        'df_erpac': stats_path + 'df_erpac.p',
        'df_ndpac': stats_path + 'df_ndpac.csv',
        'df_epochs': stats_path + 'df_epochs.p',
        'df_evokeds': stats_path + 'df_evokeds.p',
        'df_gavs': stats_path + 'df_gavs.p',
    }

    # Load or initialize dataframes
    dataframes = {key: load_or_initialize_df(path) for key, path in paths_to_dfs.items()}
    
    for idx, file in tqdm(enumerate(glob.glob(path))):
        print(idx, file)
        # 0. Get subject, night info
        subject, night, ref = file.split('/')[-1].split('-')[0].split('_')
        
        # 1. Load & process epoched 
        epochs = mne.read_epochs(file)
        
        # 2. Phase analysis
        if not ((dataframes['df_phase']['Subject'] == subject) & (dataframes['df_phase']['Night'] == night)).any():
            # Data for this subject and night does not exist, proceed with analysis
            df_phase = analyze_sleep_nmes_epochs_phase(epochs, subject, night)
            # Append new data
            dataframes['df_phase'] = pd.concat([dataframes['df_phase'], df_phase]).reset_index(drop=True)

        # 3. Extract epoch-wise power & aperiodic params
        if not ((dataframes['df_power']['Subject'] == subject) & (dataframes['df_power']['Night'] == night)).any():
            df_power = analyze_sleep_nmes_epochs_power(epochs, subject, night, plot=False)
            # Append new data
            dataframes['df_power'] = pd.concat([dataframes['df_power'], df_power]).reset_index(drop=True)
    
        # 4. Time reversed multivariate Granger causality 
        if not ((dataframes['df_gc']['Subject'] == subject) & (dataframes['df_gc']['Night'] == night)).any():
            df_gc = analyze_sleep_nmes_epochs_granger(epochs, subject, night, plot=False)
            # Append new data
            dataframes['df_gc'] = pd.concat([dataframes['df_gc'], df_gc]).reset_index(drop=True) 
        
        # 5. TCT
        if not ((dataframes['df_tct']['Subject'] == subject) & (dataframes['df_tct']['Night'] == night)).any():
            df_tct = analyze_sleep_nmes_epochs_tct(epochs, subject, night, method='pearson')
            # Append new data
            dataframes['df_tct'] = pd.concat([dataframes['df_tct'], df_tct]).reset_index(drop=True)
        
        # 6. Extract PCI 
        if not ((dataframes['df_pci']['Subject'] == subject) & (dataframes['df_pci']['Night'] == night)).any():
            df_pci = analyze_sleep_nmes_epochs_pci(epochs, subject, night)
            # Append new data
            dataframes['df_pci'] = pd.concat([dataframes['df_pci'], df_pci]).reset_index(drop=True)
        
        # 7. Extract TFR
        if not ((dataframes['df_tfr']['Subject'] == subject) & (dataframes['df_tfr']['Night'] == night)).any():
            df_tfr = analyze_sleep_nmes_epochs_tfr(epochs, subject, night, 
                                                   freq_range=(5, 25), 
                                                   steps=0.5,  
                                                   baseline=(-3, -2),
                                                   mode='zscore',
                                                   length=(-2, 2), 
                                                   plot=False, 
                                                   cmap='Spectral_r')
            # Append new data
            dataframes['df_tfr'] = pd.concat([dataframes['df_tfr'], df_tfr]).reset_index(drop=True)
       
        # 8a. Coupling Analysis (ERPAC)
        if not ((dataframes['df_erpac']['Subject'] == subject) & (dataframes['df_erpac']['Night'] == night)).any():
            df_erpac = analyze_sleep_nmes_epochs_erpac(epochs, subject, night, plot=False)
            # Append new data
            dataframes['df_erpac'] = pd.concat([dataframes['df_erpac'], df_erpac]).reset_index(drop=True)
        
        # 8b. Coupling Analysis (ndPAC) 
        if not ((dataframes['df_ndpac']['Subject'] == subject) & (dataframes['df_ndpac']['Night'] == night)).any():
            df_ndpac = analyze_sleep_nmes_epochs_ndPAC(epochs, subject, night, method='neurodsp')
            # Append new data
            dataframes['df_ndpac'] = pd.concat([dataframes['df_ndpac'], df_ndpac]).reset_index(drop=True)
         
        # 9. Extract inst HR
        if not ((dataframes['df_inst_hr']['Subject'] == subject) & (dataframes['df_inst_hr']['Night'] == night)).any():
            df_inst_hr = analyze_sleep_nmes_hr(epochs, subject, night)
            # Append new data
            dataframes['df_inst_hr'] = pd.concat([dataframes['df_inst_hr'], df_inst_hr]).reset_index(drop=True)
        
        # 10a. Combine epochs with metainfo
        if not ((dataframes['df_epochs']['Subject'] == subject) & (dataframes['df_epochs']['Night'] == night)).any():
            dataframes['df_epochs'] = pd.concat([dataframes['df_epochs'], 
                                                pd.DataFrame({'Subject': [subject], 'Night': [night], 
                                                              'Mode' : [list(epochs.event_id)[0].split('_')[0]], 'Epochs': [epochs]})], 
                                                ignore_index=True)
