#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:44:33 2024

@author: administrator
"""

import mne
from mne_connectivity import spectral_connectivity_epochs#, phase_slope_index
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
import seaborn as sns
import yasa
import fooof
import neurokit2 as nk
from sleepecg import detect_heartbeats
mne.set_log_level('ERROR')

def analyze_sleep_nmes_epochs_tfr(epochs, subject, night,
                                  freq_range = (0.5, 45), 
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
    start_cycles = 4# 4.5 if starting with 5 Hz
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
            'TFR': Sxx[cond].average(),
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
        
    elif method == 'hilbert':
        # Filter data
        sig = epochs.copy().pick(coi).filter(None, 2)
        # Extract instantaneous phase using Hilbert transform
        n = sig._data.shape[-1]
        nfast = next_fast_len(n)
        inst_phase = np.angle(hilbert(sig, N=nfast)[:n]).squeeze()
        # Extract phase at stimulation
        stim_phase = inst_phase[:, epochs.time_as_index(0)[0]]
        
    elif method == 'tensorpac':
        sig = epochs.get_data(coi).squeeze()*1e3
        sf = epochs.info['sfreq']
        p = Pac(f_pha=[0.5, 2])
        inst_phase = p.filter(sf, sig, ftype='phase', edges=None, n_jobs=-1).squeeze(0)
        # Extract phase at stimulation
        stim_phase = inst_phase[:, epochs.time_as_index(0)[0]]
    
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
    
    return hr, hr_quality, pks
        
def analyze_sleep_nmes_hr(epochs, subject, night):
    sf = epochs.info['sfreq']
    duration = len(epochs.times)
    hrs = []
    for cond in epochs.event_id.keys():
        data = epochs[cond].get_data('ecg', units='uV').squeeze()
        for idx, epoch in enumerate(data):
            hr, hr_quality, pks = get_inst_hr(epoch, sf, duration)
            if hr_quality != 'Unacceptable':
                # hr_pre = np.nanmean(hr[0:len(hr)//2])
                # hr_post = np.nanmean(hr[len(hr)//2:])
                # hr_ratio_change = ((hr_post - hr_pre)/hr_pre)*100
                hr_mean = np.nanmean(hr)
                hrv = nk.hrv_time(pks, sampling_rate=sf, show=False)
                
                # Save rpeaks to dict
                hr_dict = {
                    'Subject': subject, 
                    'Night': night,
                    'Condition': cond,
                    'Epoch': idx,
                    'Quality': hr_quality, 
                    # 'HR_ratio_change': hr_ratio_change,
                    'HR' : hr_mean, 
                    'HRV_RMSSD' : hrv['HRV_RMSSD'].values[0], 
                    'HRV_SDNN' : hrv['HRV_SDNN'].values[0],
                    }
                
                hrs.append(hr_dict)
            else:
                continue
            
    df = pd.DataFrame(hrs)
    
    # Split the 'Condition' column into 'Stim/Sham' and 'Target Chan' columns
    df['Mode'], df['Stim'], df['Target_Chan'] = zip(*df['Condition'].apply(lambda x: x.split('_')))
    df.drop('Condition', axis=1, inplace=True)
    
    # Mean over epochs
    df = df.groupby(['Subject','Night','Mode','Stim','Target_Chan']).mean(numeric_only=True)[['HR', 'HRV_RMSSD', 'HRV_SDNN']].reset_index()
    
    # Pivot the dataframe to have 'Stim' as columns, so we can calculate the ratio
    pivot_df = df.pivot_table(index=['Subject', 'Night', 'Mode', 'Target_Chan'],
                              columns='Stim', values=['HR', 'HRV_RMSSD', 'HRV_SDNN']).reset_index()
    
    # Flatten the MultiIndex columns, making them easier to access
    pivot_df.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in pivot_df.columns]
    
    # Rename columns to remove trailing underscores from 'Subject_', 'Night_', 'Mode_', and 'Target_Chan_'
    pivot_df.rename(columns=lambda x: x.rstrip('_') if x.endswith('_') else x, inplace=True)
    
    # Calculate the ratio of Stim/Sham for HR, HRV_RMSSD, and HRV_SDNN
    pivot_df['Delta_HR'] = pivot_df['HR_stim'] - pivot_df['HR_sham']
    pivot_df['HR_Ratio_Change'] = ((pivot_df['HR_stim'] - pivot_df['HR_sham']) / pivot_df['HR_sham']) * 100
    
    pivot_df['Delta_HRV_RMSSD'] = pivot_df['HRV_RMSSD_stim'] - pivot_df['HRV_RMSSD_sham']
    pivot_df['HRV_RMSSD_Ratio_Change'] = ((pivot_df['HRV_RMSSD_stim'] - pivot_df['HRV_RMSSD_sham']) / pivot_df['HRV_RMSSD_sham']) * 100
    
    pivot_df['Delta_HRV_SDNN'] = pivot_df['HRV_SDNN_stim'] - pivot_df['HRV_SDNN_sham']
    pivot_df['HRV_SDNN_Ratio_Change'] = ((pivot_df['HRV_SDNN_stim'] - pivot_df['HRV_SDNN_sham']) / pivot_df['HRV_SDNN_sham']) * 100
    
    # Return the pivot_df without trailing underscores
    return pivot_df

def analyze_sleep_nmes_epochs_power(epochs, subject, night, plot=False): 
    # Initialize list
    df = []
    df_fooof = []
    
    # Compute power spectral and fit FOOOF models
    for cond in epochs.event_id.keys():
        
        # Compute PSD using the Welch method
        # epo_spectrum = epochs[cond].compute_psd(method='welch', fmin=0.5, fmax=30, n_jobs=-1, 
        #                                         picks='csd', **dict(average='median', 
        #                                                             n_fft=int(4*epochs.info['sfreq'])))
        epo_spectrum = epochs[cond].compute_psd(fmin=0.5, fmax=45,
                                                #tmin=0, 
                                                tmin=-2, tmax=2,
                                                n_jobs=-1, picks='csd')
        psds, freqs = epo_spectrum.get_data(return_freqs=True)
        
        # decimate freqs
        freqs = freqs[::3]
        psds = psds[:,:,::3]
        psds *= 1e6
        
        # FOOOF data         
        fm = fooof.FOOOFGroup(max_n_peaks=5)
        fm.fit(freqs, psds.mean(0), freq_range=(3, 45))
        
        
        df_fooof_dict ={
            'Subject': subject,
            'Night': night, 
            'Condition' : cond, 
            'Mode' : cond.split('_')[0], 
            'fooof' : [fm],
            #'Spectra_mne' : [epo_spectrum], 
            }
        
        
        # Get relative power with YASA 
        power_df = yasa.bandpower_from_psd(psds.mean(0), freqs, ch_names=epo_spectrum.ch_names,
                                           relative=False, 
                                           bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'),
                                                  (8, 12, 'Alpha'), (12, 16, 'Sigma'),
                                                  (16, 30, 'Beta'), (30, 45, 'Gamma')])
        power_df.insert(0, 'Subject', subject)
        power_df.insert(1, 'Night', night)
        power_df.insert(2, 'Condition', cond)
        power_df.insert(3, 'Mode', cond.split('_')[0])
                
        # Extract aperiodic components for all channel
        power_df['Offset'] = fm.get_params(name='aperiodic_params', col='offset')
        power_df['Aperiodic'] = fm.get_params(name='aperiodic_params', col='exponent')            
    
        # Define and plot frequency bands of interest
        if plot:
            # Plot FOOOF results
            fig = plt.figure() 
            plt.tight_layout()
            plt.yscale('log')
            plt.xscale('log')
            
            afm_fooofed_spectrum_ = np.median([fm.get_fooof(ind=idx, regenerate=True).fooofed_spectrum_ for idx in range(len(fm))], axis=0)
            afm_ap_fit = np.median([fm.get_fooof(ind=idx, regenerate=True)._ap_fit for idx in range(len(fm))], axis=0)
            aperiodic = np.median([fm.get_fooof(ind=idx, regenerate=True).aperiodic_params_[1] for idx in range(len(fm))])
  
            plt.plot(fm.freqs, 10**(np.median(fm.power_spectra, 0)), c='k', label="Power spectrum", lw=2)
            plt.plot(fm.freqs, 10**(afm_ap_fit), c='b',linestyle='--', label="Aperiodic fit", lw=2)
            plt.plot(fm.freqs, 10**(afm_fooofed_spectrum_), c='r', label="FOOOF model fit", lw=2)
            fig.suptitle(f'PSD - {cond}')
            fig.axes[0].set_xlabel("Frequency (Hz)")
            fig.axes[0].set_ylabel("PSD log($V^2$/Hz)")
            fig.axes[0].legend()
            # set text with fit parameters
            fig.axes[0].text(0.1, 0.5, f"Slope: {round(aperiodic, 2)}",
                            transform=fig.axes[0].transAxes)
            sns.despine()
            #plt.savefig(f'{fig_path}FOOOF_{obj_name}.png')
            
            ### old
            bands = {'Delta (0.5-4 Hz)': (0.5, 4), 'Theta (4-8 Hz)': (4, 8), 
                     'Alpha (8-12 Hz)': (8, 12), 'Sigma (12-16 Hz)': (12, 16), 
                     'Beta (16-30 Hz)': (16, 30), 'Gamma (30-45 Hz)': (30, 45)}
            
            epo_spectrum.plot_topomap(bands=bands, normalize=True)
            plt.suptitle(f'{cond.capitalize()}')
            plt.close('all')
            ####
            
        df.append(power_df)
        df_fooof.append(pd.DataFrame(df_fooof_dict))    
    
    # Create df
    df = pd.concat(df).reset_index(drop=True)
    df_fooof = pd.concat(df_fooof).reset_index(drop=True)
        
    # Split the 'Condition' column into 'Stim/Sham' and 'Target Chan' columns
    df['Stim'], df['Target_Chan'] = zip(*df['Condition'].apply(lambda x: x.split('_')[1:]))
    df.drop('Condition', axis=1, inplace=True)
    
    df_fooof['Stim'], df_fooof['Target_Chan'] = zip(*df_fooof['Condition'].apply(lambda x: x.split('_')[1:]))
    df_fooof.drop('Condition', axis=1, inplace=True)
    
    return df, df_fooof

def analyze_sleep_nmes_epochs_tct(epochs, subject, night, method='spearman'):
    tcts = []
    for cond in epochs.event_id.keys():
        #for times, latency in zip(((0.025, .225), (0.2, .4)), ('CLNMES', 'CLAS')):
        for times, latency in zip(((0.18, 0.28), (0,0), (0,0), (0.35, 0.6), (1, 1.5)), 
                                  ('P200', 'N550_1', 'N550_2', 'N550', 'P900')):
            for chans, region_names in zip((['F1', 'Fz', 'F2'],['C5', 'C3', 'CP1', 'C1']),
                                            ('Frontal','Motor')):

                tmin, tmax = times
                consistency_gfp = tct.calculate_gfp_correlation(epochs[cond], 
                                                                method=method,
                                                                tmin=tmin,
                                                                tmax=tmax)
                consistency_post = tct.calculate_topographic_consistency(epochs[cond],
                                                                         tmin=tmin, tmax=tmax)
                gfp_post_trial = tct.calculate_gfp_strength(epochs[cond], method='trial_avg',
                                                            tmin=tmin, tmax=tmax)
                gfp_post_evoked = tct.calculate_gfp_strength(epochs[cond], method='evoked_avg',
                                                             tmin=tmin, tmax=tmax)
           
                # Get ERP
                if 'c3' in cond:
                    coi = 'C3'
                elif 'fz' in cond:
                    coi = 'Fz'
                                    
                # Convert channel names to indices
                chans_idx = [epochs.ch_names.index(ch) for ch in chans if ch in epochs.ch_names]
                # Create combined channel
                region_epochs = mne.channels.combine_channels(
                    epochs[cond], 
                    groups={region_names : chans_idx},
                    method='mean'
                    )
                
                #erp = epochs[cond].get_data(coi, tmin=tmin, tmax=tmax).squeeze().mean(1).mean()*1e3
                erp = region_epochs.get_data(tmin=tmin, tmax=tmax).squeeze().mean(1).mean()*1e3
                
                # Create a new dictionary for the current channel and stimulation
                tct_dict = {
                    'Subject': subject,
                    'Night': night,
                    'Condition': cond, 
                    'Mode': cond.split('_')[0], 
                    'Latency' : latency, 
                    'Region' : region_names, 
                    'Consistency_RMS': consistency_gfp,
                    'Consistency': consistency_post,
                    'RMS_Trial': gfp_post_trial, 
                    'RMS_Evoked': gfp_post_evoked, 
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
    #for times in ((0.025, .225), (0.2, .4), (0.025, 0.8), (0.025, 1.5)):
    for times in ((0.025, .225), (0, .3), (0.2, .4), (0.025, 0.8), (0.18, .28),
                  (0.35, .60), (1, 1.5), (0.025, 1.5)):
        tmin, tmax = times
        par = {'baseline_window':(-3, -1.5), 
               'response_window':(tmin, tmax), 
               'k':1.2, 
               'min_snr':1.1, 
               'max_var':99,
               'embed':False,
               'n_steps':100, 
               'avgref': False}
        for chan in ('C3','Fz'):
            if 'nmes' in list(epochs.event_id.keys())[0]:
                mode = 'nmes'
            else:
                mode = 'pn'
            evk = mne.combine_evoked([epochs[f'{mode}_stim_{chan.lower()}'].average(),
                                      epochs[f'{mode}_sham_{chan.lower()}'].average()],
                                     weights=[1, -1])
            pci_ = pci.calc_PCIst(evk.get_data()*1e3, evk.times, **par)
            
            # Create a new dictionary for the current channel and stimulation
            pci_dict = {
                'Subject': subject,
                'Night': night,
                'Target_chan': chan,
                'Mode': mode, 
                'Latency': times, 
                'PCI': pci_,
            }
            
            # Append the dictionary to the list
            pcis.append(pci_dict)  
            
    # Convert to df
    df = pd.DataFrame(pcis) 
    
    return df

def analyze_sleep_nmes_epochs_lziv(epochs, subject, night):
    from numpy import apply_along_axis as apply
    def lziv(x):
        import antropy as ant
        """Binarize the EEG signal and calculate the Lempel-Ziv complexity.
        """
        return ant.lziv_complexity(x > np.median(x), normalize=True)

    # Placeholder for all Lziv data
    lzivs = []

    for chan in ('C3', 'Fz'):
        for stim in ('stim', 'sham'):
            if 'nmes' in list(epochs.event_id.keys())[0]:
                mode = 'nmes'
            else:
                mode = 'pn'

            eps = epochs[f'{mode}_{stim}_{chan.lower()}']
            eps.resample(256)
            eps_data = eps.get_data('csd') * 1e3

            amp = np.concatenate([np.expand_dims(amp_by_time(eps_data[i, :, :], fs=eps.info['sfreq']), 0) 
                                  for i in range(len(eps_data))], axis=0)

            tmin_idx, tmax_idx = eps.time_as_index([-2, 2])
            lziv_ = apply(lziv, axis=-1, arr=amp[:, :, tmin_idx:tmax_idx])

            # Flatten and create DataFrame for current channel and stimulation
            lziv_df = pd.DataFrame({
                'Subject': [subject] * len(eps.ch_names[0:64]),
                'Night': [night] * len(eps.ch_names[0:64]),
                'Target_chan': [chan] * len(eps.ch_names[0:64]),
                'Stim': [stim] * len(eps.ch_names[0:64]), 
                'Mode': [mode] * len(eps.ch_names[0:64]), 
                'Chan': eps.ch_names[0:64],
                'Lempel_Ziv': lziv_.mean(axis=0), 
            })
            
            # Append the DataFrame to the list
            lzivs.append(lziv_df)
    
    # Concatenate all DataFrames in the list
    df = pd.concat(lzivs, ignore_index=True)
    
    return df

def analyze_sleep_nmes_epochs_ndPAC_new(epochs, subject, night, method='tensorpac'):
    # Initialize the list to hold all ndPAC data
    ndPACs = []
    for cond in epochs.event_id.keys():
        epoch = epochs[cond]
        for chan in epoch.ch_names[0:64]:
            # Get channel data
            data = epoch.copy().get_data(chan).squeeze()*1e3
            # Get channel data
            sf = epoch.info['sfreq']
            
            if method == 'tensorpac':
                p = Pac(f_pha=[0.5, 2], f_amp=[12, 16.5, 0.5, 0.5])
                sw_pha = p.filter(sf, data, ftype='phase', edges=None, n_jobs=-1).squeeze()
                sp_amp = p.filter(sf, data, ftype='amplitude', edges=None, n_jobs=-1).squeeze()
                sp_amp = sp_amp.mean(0)
            elif method == 'hilbert':
                # filter sw data
                sw_data = mne.filter.filter_data(
                    data, 
                    sf, 
                    None, 
                    2, 
                    method="fir",
                    l_trans_bandwidth=0.2,
                    h_trans_bandwidth=0.2
                    )
                
                # filter sp data
                sp_data = mne.filter.filter_data(
                    data,
                    sf,
                    12,
                    16,
                    method="fir",
                    l_trans_bandwidth=1.5,
                    h_trans_bandwidth=1.5,
                )
                
                # Now extract the instantaneous phase/amplitude using Hilbert transform
                n_samples = data.shape[-1]
                nfast = next_fast_len(n_samples)
                sw_pha = np.angle(hilbert(sw_data, N=nfast)[:, :n_samples])
                sp_amp = np.abs(hilbert(sp_data, N=nfast)[:, :n_samples])
                                        
            # Extract phase at pre/post stimulation
            start_idx, end_idx = epoch.time_as_index((-1.5, 0.5))
            pre_sw_pha = sw_pha[:, start_idx:end_idx]
            start_idx2, end_idx2 = epoch.time_as_index((-1, 1))
            post_sw_pha = sw_pha[:, start_idx2:end_idx2]
        
            # Extract phase at pre/post stimulation
            pre_sp_amp = sp_amp[:, start_idx:end_idx]
            post_sp_amp = sp_amp[:, start_idx2:end_idx2]
            
            # ndPAC calculation 
            # Find location of max sigma amplitude in pre/post epoch
            idx_max_amp_pre = pre_sp_amp.argmax(axis=1).reshape(-1, 1)
            idx_max_amp_post = post_sp_amp.argmax(axis=1).reshape(-1, 1)
                      
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
                ("Pre", pha_at_max_pre, ndp_pre),
                ("Post", pha_at_max_post, ndp_post)
            ]
            
            # Loop through each condition to process and append the data
            for session, pha_at_max, ndp in conditions_data:            
                # Calculate medians or other statistics as needed
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

def analyze_sleep_nmes_epochs_ndPAC_old(epochs, subject, night, method='tensorpac'):
    # Initialize the list to hold all ndPAC data
    ndPACs = []
    for cond in epochs.event_id.keys():
        epoch = epochs[cond]
        if method=='tensorpac':
            p = Pac(f_pha=[0.5, 2], f_amp=[12, 16, 0.5, 0.5])
        for chan in epoch.ch_names[0:64]:
            # Get channel data
            data = epoch.copy().get_data(chan).squeeze()*1e3
            if method=='tensorpac':
                # Get channel data
                sf = epoch.info['sfreq']
                sw_pha = p.filter(sf, data, ftype='phase', edges=None, n_jobs=-1).squeeze()
                sp_amp = p.filter(sf, data, ftype='amplitude', edges=None, n_jobs=-1).squeeze()
                sp_amp = sp_amp.mean(0)
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
            # start_idx, end_idx = epoch.time_as_index((-1.2, 0))
            start_idx, end_idx = epoch.time_as_index((-1.2, 1.2))
            pre_sw_pha = sw_pha[:, start_idx:end_idx]
            # start_idx2, end_idx2 = epoch.time_as_index((0, 1.2))
            start_idx2, end_idx2 = epoch.time_as_index((-0.5, 1.5))
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
            time_relative_to_start_post = time_in_seconds_post + -0.5#0
            
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
        epoch = epochs[cond].copy()
        edges = int(1*epoch.info['sfreq'])
        if 'c3' in cond:
            coi = 'C3'
        elif 'fz' in cond:
            coi = 'Fz'
        # p = EventRelatedPac(f_pha=[0.5, 1.5], f_amp=(5, 25, .25, .25))
        # erpac = p.filterfit(int(epoch.info['sfreq']), epoch.get_data(picks=coi).squeeze()*1e3,
        #                     method='gc', smooth=50, edges=edges, n_perm=200, 
        #                     mcp='fdr', n_jobs=-1).squeeze()
                
        ###
        # ERPAC (+/- 2 sec to avoid filter edge)
        sf = epoch.info['sfreq']
        data_erpac = epoch.get_data(picks=coi).squeeze()*1e3
        erp = EventRelatedPac(f_pha=[0.5, 1.5], f_amp=np.arange(3.75, 24.75, 0.5), 
                              verbose=False)  # f_pha = 0.8 Hz
        #freqs = erp.f_amp.mean(1).astype(str)
        pha = erp.filter(sf, data_erpac, ftype='phase', edges=edges)
        amp = erp.filter(sf, data_erpac, ftype='amplitude', edges=edges)
        ergcpac = np.squeeze(erp.fit(pha, amp, method="gc", smooth=25, n_jobs=-1)) 
                       
        # Create a new dictionary for the erpac 
        dict_erpac = {
            'Subject': subject,
            'Night': night,
            'Condition': cond, 
            'Mode': cond.split('_')[0],
            'ERPAC': ergcpac,
            'Freqs' : erp.yvec, 
        }
        
        erpacs.append(dict_erpac)
        
        if plot:  
            fig_path = '/media/administrator/Sleep_Data/Processed/Figures/Sleep/'
    
            # ERGPAC Plot
            # tmin, tmax = epoch.time_as_index([-2, 2])
            # times = epoch[0].times[tmin:tmax]
            # fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
            # im = plt.imshow(ergcpac[:, tmin:tmax], aspect='auto', 
            #                 #cmap="Spectral_r", 
            #                 origin='upper',
            #                 #interpolation="gaussian", 
            #                 #vmin=-0.2, vmax=1,
            #                 extent=[times[0], times[-1], freqs[-1], freqs[0]])
            
            # plt.gca().invert_yaxis()
            
            # fig.suptitle(f"GC ERPAC - {cond}")
            # plt.xlabel("Time from stim onset (s)")
            # plt.ylabel("Frequency (Hz)")
            # plt.axvline(0, ls=":", lw=1.5, color="k")
            
            # cb = plt.colorbar(im, shrink=0.7, pad=0.05, aspect=20)
            # cb.set_label("Coupling")
            # cb.outline.set_visible(False)
            
            # ax_sw = ax.twinx()
            # data_erpac_ = epoch.copy().filter(None, 1.5).get_data(picks=coi, 
            #                                                       tmin=-2, 
            #                                                       tmax=3).squeeze()*1e3
            # ax_sw.plot(times, data_erpac_.mean(0), color="k", lw=2)
            # ax_sw.set_yticks([]);
            
            # fig.savefig(fig_path + f'{cond}_{subject}_{night}_erpac.png')
            
            ## Use this
            ax = erp.pacplot(ergcpac, epoch.times[edges:-edges], erp.yvec, 
                             xlabel='Time (second)', 
                             title=f'Event-Related PAC {cond}',
                             # fz_labels=15, fz_title=18
                             ylabel='Amplitude frequency (Hz)', #title=erp.method,
                             cblabel='ERPAC', vmin=0., rmaxis=True)
            ax.axvline(0., linestyle='--', color='w', linewidth=2)
            
            ax_sw = ax.twinx()
            data_erpac_ = epoch.copy().filter(None, 1.5).get_data(picks=coi, 
                                                                  tmin=-2, 
                                                                  tmax=2).squeeze()*1e3
            ax_sw.plot(epoch.times[edges:-edges], data_erpac_.mean(0), color="k", lw=2)
            ax_sw.set_yticks([]);
            
            plt.tight_layout()
            erp.show()         
            erp.savefig(fig_path + f'{cond}_{subject}_{night}_erpac.png')
            plt.close('all')
        
    # Convert to df
    df = pd.DataFrame(erpacs) 
    
    # Split the 'Condition' column into 'Stim/Sham' and 'Target Chan' columns
    df['Stim'], df['Target_Chan'] = zip(*df['Condition'].apply(lambda x: x.split('_')[1:]))
    df.drop('Condition', axis=1, inplace=True)
    
    return df

# def analyze_sleep_nmes_epochs_psi(epochs, subject, night):
#     # Number of channels (assuming 64 channels, indexed from 0 to 63)
#     n_channels = 64
    
#     # Seed channel index (e.g., channel 14)
#     seed_channel_idx = 14
    
#     # Generate indices for all pairs where the seed channel is connected to each other channel
#     # Note: We create a list of tuples (seed_channel_idx, other_channel_idx) for each other channel
#     indices = (np.array([seed_channel_idx] * n_channels),  # seed channel repeated
#                np.array([idx for idx in range(n_channels)]))  # all other channels
    
#     for eps in epochs:
#         # calculate PSI
#         psi = phase_slope_index(
#             eps[cond],
#             sfreq=eps.info['sfreq'],
#             indices=indices,
#             fmin=0.5,
#             fmax=2
#         )
#         yasa.topoplot(pd.Series(psi.get_data().squeeze(), psi.names))

def analyze_sleep_nmes_epochs_granger(epochs, subject, night, times=(-2, 2), plot=False):
    # Include only eeg sensors post stimulus onset 
    eps = epochs.copy().crop(tmin=times[0], tmax=times[1]).pick('csd')
    
    # lags 
    lags = 20 
    
    # frontal sensors
    signals_a = [eps.ch_names.index(ch_idx) for ch_idx in (['F1','Fz','F2'])]
    
    # motor sensors
    signals_b = [eps.ch_names.index(ch_idx) for ch_idx in (['C5','C3','C1'])]

    # indices
    indices_ab = (np.array([signals_a]), np.array([signals_b]))  # A => B
    indices_ba = (np.array([signals_b]), np.array([signals_a]))  # B => A
    
    trgc_df = []
    for cond in eps.event_id.keys():
        # 1. compute Granger causality
        gc_ab = spectral_connectivity_epochs(
            eps[cond],
            sfreq=eps.info['sfreq'],
            method=["gc"],
            indices=indices_ab,
            fmin=0.5,
            fmax=45,
            #rank=(np.array([3]), np.array([3])),
            gc_n_lags=lags,
        )  # A => B
        gc_ba = spectral_connectivity_epochs(
            eps[cond],
            sfreq=eps.info['sfreq'],
            method=["gc"],
            indices=indices_ba,
            fmin=0.5,
            fmax=45,
            #rank=(np.array([3]), np.array([3])),
            gc_n_lags=lags,
        )  # B => A
        freqs = gc_ab.freqs
        
        # Drivers and receivers: analysing the net direction of information flow
        net_gc = gc_ab.get_data() - gc_ba.get_data()  # [A => B] - [B => A]
        
        # 2. compute GC on time-reversed signals
        gc_tr_ab = spectral_connectivity_epochs(
            eps[cond],
            sfreq=eps.info['sfreq'],
            method=["gc_tr"],
            indices=indices_ab,
            fmin=0.5,
            fmax=45,
            #rank=(np.array([3]), np.array([3])),
            gc_n_lags=lags,
        )  # TR[A => B]
        gc_tr_ba = spectral_connectivity_epochs(
            eps[cond],
            sfreq=eps.info['sfreq'],
            method=["gc_tr"],
            indices=indices_ba,
            fmin=0.5,
            fmax=45,
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
            'Mode': [list(eps.event_id)[0].split('_')[0]]*len(freqs), 
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
    
    plt.close('all')
      
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
                
#%%
if __name__ == '__main__':
    path = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/*csd-epo.fif'
    stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'
    epochs_all = []
    df_phase_all = []
    df_power_all = []
    df_fooof_all = []
    df_gc_all = []
    df_tct_all = []
    df_pci_all = []
    df_lziv_all = []
    df_erpac_all = []
    df_ndpac_all = []
    df_tfr_all = []
    df_inst_hr_all = []
    for file in tqdm(glob.glob(path)):
        print(file)
        # 0. Get subject, night info
        subject, night, ref = file.split('/')[-1].split('-')[0].split('_')
        
        # 1a. Load & process epoched 
        epochs = mne.read_epochs(file)
        
        # 1b. If mode is 'pinknoise', then timeshift 100 ms -> not anymore
        mode = list(epochs.event_id)[0].split('_')[0]
        # if mode == 'pn':   
        #     evk = mne.combine_evoked([epochs['pn_stim_fz'].average(), 
        #                               epochs['pn_sham_fz'].average()],
        #                               weights=[1, -1])
        #     evk.plot('Fz', highlight=(0, .5), titles=f'{subject} {mode} - Fz')
        # else:
        #     evk = mne.combine_evoked([epochs['nmes_stim_fz'].average(), 
        #                               epochs['nmes_sham_fz'].average()],
        #                               weights=[1, -1])
        #     evk.plot('C3', xlim=(0, .5), titles=f'{subject} {mode} - C3')
            
        # 1c. Combine epochs with metainfo
        epochs_all.append([subject, night, mode, epochs])
        
        # 2a. Extract stimulation targeting phases 
        df_phase = analyze_sleep_nmes_epochs_phase(epochs, subject, night)
        df_phase_all.append(df_phase)
        
        # 3a. Extract epoch-wise power & aperiodic params
        df_power, df_fooof = analyze_sleep_nmes_epochs_power(epochs, subject, night, plot=False)
        df_power_all.append(df_power)
        df_fooof_all.append(df_fooof)
        
        # 4a. Spatio Spectral Decomposition (SSD) Analysis
        ssd = []  
        
        # 4b. Time reversed multivariate Granger causality 
        df_gc = analyze_sleep_nmes_epochs_granger(epochs, subject, night, plot=True)
        df_gc_all.append(df_gc)
        
        # 5a. TCT
        df_tct = analyze_sleep_nmes_epochs_tct(epochs, subject, night, method='pearson')
        df_tct_all.append(df_tct)
        
        # 6a. Extract PCI 
        df_pci = analyze_sleep_nmes_epochs_pci(epochs, subject, night)
        df_pci_all.append(df_pci)
        
        # 6b. Extract Lempel-Ziv
        df_lziv = analyze_sleep_nmes_epochs_lziv(epochs, subject, night)
        df_lziv_all.append(df_lziv)
        
        # 7a. Extract TFR
        df_tfr = analyze_sleep_nmes_epochs_tfr(epochs, subject, night, 
                                                # freq_range=(5, 25), 
                                                freq_range=(4, 25),
                                                steps=0.5,  
                                                baseline=(-3, -2),
                                                mode='zscore',
                                                length=(-2, 2), 
                                                plot=True, 
                                                cmap='Spectral_r')
        df_tfr_all.append(df_tfr) 
        
        # 8a. Coupling Analysis (ERPAC)
        df_erpac = analyze_sleep_nmes_epochs_erpac(epochs, subject, night, plot=True)
        df_erpac_all.append(df_erpac)   
        
        # 8b. Coupling Analysis (ndPAC) 
        df_ndpac = analyze_sleep_nmes_epochs_ndPAC_new(epochs, subject, night, method='hilbert')
        df_ndpac_all.append(df_ndpac) 
        
        # 9. Extract inst HR
        df_inst_hr = analyze_sleep_nmes_hr(epochs, subject, night)
        df_inst_hr_all.append(df_inst_hr)
        
        # delete to save memory
        del epochs, df_phase, df_power, df_tct, df_pci, df_lziv, df_tfr, df_ndpac, df_erpac
        
    # . Convert epochs into dataframe 
    epochs_all_df = pd.DataFrame(epochs_all, 
                                  columns=['Subject','Night','Mode','Epochs'])
    epochs_all_df.to_pickle(stats_path + 'df_epochs.p')
        
    # . Create evoked contrasts object 
    evoked_all_df = pd.DataFrame(create_evoked_contrasts(epochs_all_df))
    evoked_all_df.to_pickle(stats_path + 'df_evokeds.p')
    
    # . Create GAVs
    gavs = pd.DataFrame(create_gavs(evoked_all_df))                        
    gavs.to_pickle(stats_path + 'df_gavs.p') 
    
    # Convert inst hr to dataframe
    df_inst_hrs_all =  pd.concat(df_inst_hr_all).reset_index(drop=True)
    df_inst_hrs_all.to_csv(stats_path + 'df_inst_hr.csv')
    
    # 2b. Convert phases into dataframe 
    df_phases_all = pd.concat(df_phase_all).reset_index(drop=True)
    df_phases_all.to_csv(stats_path + 'df_phase.csv')
    
    # 3b. Convert power/aperiodic into dataframe
    df_powers_all = pd.concat(df_power_all).reset_index(drop=True)
    df_fooofs_all = pd.concat(df_fooof_all).reset_index(drop=True)
    df_powers_all.to_csv(stats_path + 'df_power.csv') 
    df_fooofs_all.to_pickle(stats_path + 'df_fooof.p')
    
    # 4b. SSD -> Not sure if its worth doing
    #ssd = ssd
    
    # 4c. Convert GC to df
    df_gcs_all = pd.concat(df_gc_all).reset_index(drop=True)
    df_gcs_all.to_csv(stats_path + 'df_gc.csv') 
    
    # 5b. Convert tct into dataframe
    df_tcts_all = pd.concat(df_tct_all).reset_index(drop=True)    
    df_tcts_all.to_csv(stats_path + 'df_tct.csv')

    # . Convert pci into dataframe 
    df_pci_all_df = pd.concat(df_pci_all).reset_index(drop=True)
    df_pci_all_df.to_pickle(stats_path + 'df_pci.p')
    
    # . Convert lziv into dataframe  
    df_lziv_all_df = pd.concat(df_lziv_all).reset_index(drop=True)
    df_lziv_all_df.to_pickle(stats_path + 'df_lziv.p')
    
    # . Combine TFRs
    df_tfrs_all = pd.concat(df_tfr_all).reset_index(drop=True)
    df_tfrs_all.to_pickle(stats_path + 'df_tfr.p') 
    
    # # . Combine ERPACs/ndPACS
    df_erpacs_all = pd.concat(df_erpac_all).reset_index(drop=True)
    df_erpacs_all.to_pickle(stats_path + 'df_erpac.p')
    
    df_ndpacs_all = pd.concat(df_ndpac_all).reset_index(drop=True)
    df_ndpacs_all.to_csv(stats_path + 'df_ndpac.csv')    
    
#%%

# df = evoked_all_df.set_index(['Subject', 'Night'])
# for con in df.columns:
#     print(con)
#     gav = mne.grand_average(list(df[con]))