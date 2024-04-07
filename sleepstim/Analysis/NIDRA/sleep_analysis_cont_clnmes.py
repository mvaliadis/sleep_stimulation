#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:44:33 2024

@author: administrator
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats
import yasa
# import sleepeegpy
from sleepecg import detect_heartbeats
import fooof
from sleepeegpy.pipeline import SpectralPipe
import glob
import os
import neurokit2 as nk
from tqdm import tqdm
import pandas as pd
mne.set_log_level('ERROR')
     
def graveyard(path_to_eeg='/media/administrator/Sleep_Data/Processed/LaKu_1_1-raw.fif'):
    spectral_pipe = SpectralPipe(
        path_to_eeg=path_to_eeg,
        output_dir='/media/administrator/data/Study_2_data/NIDRA/CLNMES/',
    )
    
    
    spectral_pipe.predict_hypno(
            eeg_name = "Cz",
            eog_name = "EOG",
            emg_name = "EMG_L",
            ref_name = 'M1',        
            save=False
    )
    
    
    spectral_pipe.plot_hypnospectrogram(picks=["Cz"])
    
    spectral_pipe.compute_psd(
        sleep_stages={"Wake": 0, "N1": 1, "N2/3": (2, 3), "REM": 4},
        reference=['M1','M2'],
        # Additional arguments passed to the Welch method:
        window="hamming",
        verbose=False
    )
        
    spectral_pipe.plot_psds(picks=["C3"])
    spectral_pipe.plot_topomap_collage()           

def analyze_sleep_nmes_so_spindles(file, stats_path, hypno_path):
    # 0a. Get subject, night info
    subject, night, ref = file.split('/')[-1].split('-')[0].split('_')
    
    # 0b. Check if this subject night has already been processed
    sw_output_path = os.path.join(stats_path, f'Events/{subject}_{night}_sw.csv')
    spindles_output_path = os.path.join(stats_path, f'Events/{subject}_{night}_spindles.csv')
    if os.path.exists(sw_output_path) and os.path.exists(spindles_output_path):
        print(f"Skipping {subject} {night} - already processed.")
        sw_summary = pd.read_csv(sw_output_path)
        sp_summary = pd.read_csv(spindles_output_path)
        return sw_summary, sp_summary

    # 1a. Load continuous data 
    raw = mne.io.read_raw_fif(file, preload=False) 
    
    # 1b. Get stimulation condition from epoch events
    events = mne.read_events(f'/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/{subject}_{night}_csd-epo.fif', 
                             return_event_id=True)
    if 'pn_sham_c3' in events[-1]:
        mode = 'pn'
    elif 'nmes_sham_c3' in events[-1]:
        mode = 'nmes'
    
    # 2. Load hypnogram
    hypno = np.load(hypno_path + f'{subject}_{night}_hypno.npy')
    
    # 3. SO/Spindle detection
    data = raw.get_data('csd')*1e3
    # Detect slow waves
    thresh = np.percentile(np.abs(data[:, hypno==3]), 75)
    print('75th percentile threshold in SWS: %.2f mm/V²' % thresh)
    sw = yasa.sw_detect(data=data, 
                        sf=raw.info['sfreq'], 
                        ch_names=raw.ch_names[0:64],
                        hypno=hypno,
                        include=(2, 3),
                        amp_neg=(None, None), # Disabled
                        amp_pos=(None, None), # Disabled
                        amp_ptp=(thresh, np.inf),  # No upper threshold: np.inf
                        remove_outliers=True,
                        coupling=True,
                        coupling_params={"freq_sp": (12, 16), "time": 1, "p": 0.05},
                        )
    # sw_summary = sw.summary(grp_chan=True, grp_stage=True)
    # yasa.topoplot(sw_summary.loc[3].Density)
    
    # Detect spindles
    spindle = yasa.spindles_detect(data=data,
                                   sf=raw.info['sfreq'],
                                   hypno=hypno,
                                   ch_names=raw.ch_names[0:64],
                                   include=(2, 3),
                                   freq_sp=(12, 16),
                                   freq_broad=(1, 30),
                                   duration=(0.5, 2), 
                                   thresh={'rel_pow':0.1,'corr':None,'rms':1.5},
                                   remove_outliers=True,
                                   )  
        
    sp_summary = spindle.summary(grp_chan=True, grp_stage=True)
    # yasa.topoplot(sp_summary.loc[2].Density)
    sp_summary.insert(0, 'Subject', subject)
    sp_summary.insert(1, 'Night', night)
    sp_summary.insert(2, 'Mode', mode)
                                    
    # Detect co-occurring SW/spindle events
    sw.find_cooccurring_spindles(spindle.summary(), lookaround=1.2)
    # yasa.topoplot(sw_summary.loc[3].CooccurringSpindle)
    sw_summary = sw.summary(grp_chan=True, grp_stage=True)
    sw_summary.insert(0, 'Subject', subject)
    sw_summary.insert(1, 'Night', night)
    sw_summary.insert(2, 'Mode', mode)
    sw_summary.insert(3, '75_Thresh', thresh)
    
    # Write SW and spindle summaries to CSV files
    sw_summary.reset_index().to_csv(sw_output_path, index=False)
    sp_summary.reset_index().to_csv(spindles_output_path, index=False)
    print(f"Processed and saved {subject} {night}")
    
    # delete heavy vars
    del raw, events, sw, spindle
    
    return sw_summary, sp_summary
     
def analyze_sleep_hrv_power(file, stats_path, figure_path, nrem_block=True, 
                            power_desired=True, plot=False):
    # Get subject, night info
    subject, night, _ = file.split('/')[-1].split('-')[0].split('_')
        
    # Check if this subject night has already been processed
    hrv_output_path = os.path.join(stats_path, f'Events/{subject}_{night}_hrv.csv')
    if os.path.exists(hrv_output_path):
        print(f"Skipping {subject} {night} - already processed.")
        hrv_df = pd.read_csv(hrv_output_path)
        return hrv_df
     
    # Read raw
    raw = mne.io.read_raw(file)
    ecg = raw.copy().pick('ecg')
    
    # Load hypnogram
    hypno = np.load(hypno_path + f'{subject}_{night}_hypno.npy')
        
    # Get stimulation condition from epoch events
    try:  
        events = mne.read_events(f'/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/{subject}_{night}_csd-epo.fif', 
                                 return_event_id=True)
        if 'pn_sham_c3' in events[-1]:
            mode = 'pn'
        elif 'nmes_sham_c3' in events[-1]:
            mode = 'nmes'
            
    except Exception as e:
        print(f"Error reading events: {e}")
        return None
    
    # If mode is not set, return None or skip further processing
    if mode is None:
        print("No relevant events found.")
        return None
       
    # Epoch data 
    data = ecg.get_data(units='uV').squeeze()
    sf = raw.info['sfreq']
    
    # Merge N2 and N3 into NREM sleep, if desired
    if nrem_block:
        hypno[hypno == 3] = 2
      
    # Find periods of equal duration
    epochs = yasa.hypno_find_periods(hypno=hypno, 
                                     sf_hypno=sf, 
                                     threshold="2.5min", 
                                     equal_length=True)
    epochs = epochs[epochs["values"].isin([2, 4])].reset_index(drop=True)
    
    # Sort by stage and add epoch number
    epochs = epochs.sort_values(by=["values", "start"])
    epochs["epoch"] = epochs.groupby("values")["start"].transform(lambda x: range(len(x)))
    epochs = epochs.set_index(["values", "epoch"])
    
    # Loop over epochs
    rpeaks = {}
    for idx in epochs.index:
        start = epochs.loc[idx, "start"]
        duration = epochs.loc[idx, "length"]
        end = int(epochs.loc[idx, "start"] + duration)
        # Detect R-peaks
        try:
            pks = detect_heartbeats(data[start:end], fs=sf)
            # peaks, info = nk.ecg_peaks(data[start:end], 
            #                            sampling_rate=sf, 
            #                            correct_artifacts=True)
            # pks = info['ECG_R_Peaks']
            # compute ecg signal quality
            hr_quality = nk.ecg_quality(data[start:end], sampling_rate=sf,
                                        method="zhao2018", approach="fuzzy")
                           
        except Exception as e:
            print(f"Heartbeat detection failed for epoch {idx[1]} of stage {idx[0]}: {e}")
            continue
 
        # Save rpeaks to dict
        rpeaks[idx] = pks
 
        # If not enough R-peaks were detected, skip epochs and return NaN
        # Here, we assume a minimal HR of 30 bpm
        constant_hr = 60 * (pks.size / (duration / sf))
        if constant_hr < 30:
            print(f"Too few detected heartbeats in epoch {idx[1]} of stage {idx[0]}.")
            continue
 
        # Find and correct RR intervals. Default is 400 ms (150 bpm) to 2000 ms (30 bpm)
        rr_limit=(400, 2000)
        rri = 1000 * np.diff(pks) / sf
        rri = np.ma.masked_outside(rri, rr_limit[0], rr_limit[1]).filled(np.nan)
        # Interpolate NaN values, but no more than 10 consecutive values
        if np.isnan(rri).any():
            rri = pd.Series(rri).interpolate(limit_direction="both", limit=10).to_numpy()
        if np.isnan(rri).any():
            # If there are still NaN present, skip current epoch
            print(f"Invalid RR intervals in epoch {idx[1]} of stage {idx[0]}.")
            continue
 
        # Heart rate
        hr = 60000 / rri
        epochs.loc[idx, "HR"] = np.mean(hr)
        if plot:
            hrv_indices = nk.hrv(pks, sampling_rate=sf, show=True)       
            # Get the current figure
            fig = plt.gcf()
            # Set the desired size (width, height) in inches
            fig.set_size_inches(12, 8)  # for example, 12 inches by 8 inches
            # Optionally adjust layout
            plt.tight_layout()
            # Save the figure
            plt.savefig(figure_path + f'{subject}_{night}_{idx}_{mode}_hrv.png')
            # Close the plot
            plt.close('all')
        else:
            hrv_indices = nk.hrv(pks, sampling_rate=sf, show=False)
        epochs.loc[idx, hrv_indices.columns] = hrv_indices.iloc[0]
        
        # Append info to HRV df
        epochs.loc[idx, 'Quality'] = hr_quality
 
    # Convert start and duration to seconds
    epochs["start"] /= sf
    epochs["length"] /= sf
    epochs = epochs.rename(columns={"length": "duration"}) 
    
    # drop bad ecg epochs
    hrv = epochs[epochs.Quality!='Unacceptable'].copy()

    if power_desired:
        hrv_df = add_power(hrv, raw.get_data('csd')*1e3, 
                           sf, raw.ch_names[0:64], 
                           subject, night, mode)
        
    # Write HRV summary to CSV files
    hrv_df.to_csv(hrv_output_path, index=False)
    print(f"Processed and saved {subject} {night}")

    return hrv_df # rpeaks

def add_power(hrv, eeg, sf, ch_names, subject, night, mode):
    # Initialize a list to hold DataFrames for each channel
    hrvs = []

    for chan_idx, chan in enumerate(ch_names):
        # Create a copy of the hrv DataFrame for current channel
        hrv_chan = hrv.copy()

        # Initialize EEG band columns with NaN values
        for band in ["Delta", "Theta", "Alpha", "Sigma"]:
            hrv_chan[band] = np.nan

        # Calculate bandpower for each epoch and update the DataFrame
        for idx, row in hrv_chan.iterrows():
            start = int(row["start"] * sf)
            end = int(sf * (row["start"] + row["duration"]))
            
            # Get PSD using the Welch method
            psds, freqs = mne.time_frequency.psd_array_welch(
                x=eeg[chan_idx, start:end].squeeze(),
                sfreq=sf, 
                fmin=0.5, 
                fmax=30, 
                n_jobs=-1, 
                **dict(average='median', 
                       n_fft=int(4*sf))
                )
            
            # FOOOF data         
            fm = fooof.FOOOF(max_n_peaks=5)
            fm.fit(freqs, psds)          
                       
            # Get relative power with YASA 
            bp = yasa.bandpower_from_psd(psds, freqs, ch_names=[chan],
                                         bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'),
                                                (8, 12, 'Alpha'), (12, 16, 'Sigma'),
                                                (16, 30, 'Beta')]).set_index('Chan')
           
            # Update bandpower values
            hrv_chan.at[idx, "Delta"] = bp.loc[chan, "Delta"]
            hrv_chan.at[idx, "Theta"] = bp.loc[chan, "Theta"]
            hrv_chan.at[idx, "Alpha"] = bp.loc[chan, "Alpha"]
            hrv_chan.at[idx, "Sigma"] = bp.loc[chan, "Sigma"]
            
            # Extract aperiodic components for all channel
            hrv_chan.at[idx, 'Offset'] = fm.get_params(name='aperiodic_params', col='offset')
            hrv_chan.at[idx, 'Aperiodic'] = fm.get_params(name='aperiodic_params', col='exponent') 

        # Group by 'values', calculate mean, and add additional info
        hrv_chan_grouped = hrv_chan.groupby('values').mean().reset_index()
        hrv_chan_grouped['Subject'] = subject
        hrv_chan_grouped['Night'] = night
        hrv_chan_grouped['Mode'] = mode
        hrv_chan_grouped['Channel'] = chan

        # Rename 'values' column to 'Stage'
        hrv_chan_grouped.rename(columns={'values': 'Stage'}, inplace=True)

        # Append the grouped DataFrame to the list
        hrvs.append(hrv_chan_grouped)

    # Concatenate all channel DataFrames
    hrv_final = pd.concat(hrvs).reset_index(drop=True)

    return hrv_final

def combine_hrv_coupling(sw_summary, df_hrv):
    ndPAC_grouped = sw_summary.groupby(['Stage','Channel']).mean()['ndPAC'].reset_index()
        
    # Adjust Stage values: convert both stages 2 and 3 to stage 2 for NREM grouping
    ndPAC_grouped['Stage'] = ndPAC_grouped['Stage'].replace({3: 2})
    
    # Now, group by the new 'Stage' and 'Channel' again and calculate the mean to combine original stages 2 and 3
    ndPAC_grouped = ndPAC_grouped.groupby(['Stage', 'Channel']).mean().reset_index()
    
    # Assuming df2 is your second DataFrame
    df2_reset = df_hrv.reset_index()
    
    # Merge the adjusted ndPAC_grouped DataFrame with your second DataFrame based on 'Stage' and 'Chan'
    merged_df = pd.merge(df2_reset, ndPAC_grouped, on=['Stage', 'Channel'], how='left')
    
    # Drop the 'index' column from the merged DataFrame
    merged_df.drop(columns='index', inplace=True)
    
    return merged_df

def analyze_sleep_macroarchitecture(file, stats_path, hypno_path):
    # Get subject, night info
    subject, night, _ = file.split('/')[-1].split('-')[0].split('_')
             
    # Read raw
    raw = mne.io.read_raw(file, preload=False)
    
    # Load hypnogram
    hypno = np.load(hypno_path + f'{subject}_{night}_hypno.npy')
    
    # Get stimulation condition from epoch events
    try:  
        events = mne.read_events(f'/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/{subject}_{night}_csd-epo.fif', 
                                 return_event_id=True)
        if 'pn_sham_c3' in events[-1]:
            mode = 'pn'
        elif 'nmes_sham_c3' in events[-1]:
            mode = 'nmes'
            
    except Exception as e:
        print(f"Error reading events: {e}")
        return None
    
    # Extract sleep stats based on hypnogram
    sleep_stats = yasa.sleep_statistics(hypno, sf_hyp=raw.info['sfreq'])
    
    # Make dataframe
    df = pd.DataFrame([sleep_stats])
    
    # Add info
    df.insert(0, 'Subject', subject)
    df.insert(1, 'Night', night)
    df.insert(2, 'Mode', mode)
    
    return df    
    
def analyze_sleep_transitions(file, stats_path, hypno_path):
    # Get subject, night info
    subject, night, _ = file.split('/')[-1].split('-')[0].split('_')
                
    # Load hypnogram
    hypno = np.load(hypno_path + f'{subject}_{night}_hypno30s.npy')
    
    # Get stimulation condition from epoch events
    try:  
        events = mne.read_events(f'/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/{subject}_{night}_csd-epo.fif', 
                                 return_event_id=True)
        if 'pn_sham_c3' in events[-1]:
            mode = 'pn'
        elif 'nmes_sham_c3' in events[-1]:
            mode = 'nmes'
            
    except Exception as e:
        print(f"Error reading events: {e}")
        return None
    
    # Extract sleep transitions based on hypnogram
    _, probs = yasa.transition_matrix(hypno)
                
    # Make dataframe
    df = pd.DataFrame(probs)
    
    # Add info
    df['Subject'] = subject
    df['Night'] = night
    df['Mode'] = mode
    
    return df  
    
#%%
if __name__ == '__main__':
    path = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Raw/*csd-raw.fif'
    stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'
    # stats_path = '/media/administrator/data/Study_2_data/NIDRA/CLNMES/Statistics/'
    hypno_path = '/media/administrator/Sleep_Data/Processed/Hypnograms/' 
    # hypno_path = '/media/administrator/data/Study_2_data/NIDRA/CLNMES/Hypnograms/' 
    figure_path = '/media/administrator/Sleep_Data/Processed/Figures/Sleep/'
    slowwaves = []
    spindles = []
    hrvs = []
    sleep_stats = [] 
    trans_p = []
    for file in tqdm(glob.glob(path)):
        print(file)
        
        # Macroarchitecture analysis
        df_macro = analyze_sleep_macroarchitecture(file, stats_path, hypno_path)
        sleep_stats.append(df_macro)
        
        # Transition analysis 
        df_trans = analyze_sleep_transitions(file, stats_path, hypno_path)
        trans_p.append(df_trans)
        
        # Extract SO and Spindle summaries
        sw_summary, sp_summary = analyze_sleep_nmes_so_spindles(file, stats_path, hypno_path)
        
        # HRV analysis
        df_hrv = analyze_sleep_hrv_power(file, stats_path, figure_path, nrem_block=True, 
                                         power_desired=True, plot=False)
        
        # Combine HRV with ndPAC
        df_hrv_power_couping = combine_hrv_coupling(sw_summary, df_hrv)
          
        # Append summaries
        slowwaves.append(sw_summary)
        spindles.append(sp_summary)
        hrvs.append(df_hrv_power_couping)
        
        # Delete heavy objects
        del sw_summary, sp_summary, df_hrv, df_hrv_power_couping

    # Combine sleep stats
    sleep_stats_df = pd.concat(sleep_stats).reset_index(drop=True)
    sleep_stats_df.to_csv(stats_path + 'df_sleep_stats.csv')
    
    # Combine transition probs
    trans_p_df = pd.concat(trans_p).reset_index()
    trans_p_df.to_csv(stats_path + 'df_trans.csv')
    
    # Combined SW/Spindle detection summaries
    df_spindles = pd.concat(spindles).reset_index(drop=True)
    df_spindles.to_csv(stats_path + 'df_spindles.csv')
    df_sw = pd.concat(slowwaves).reset_index(drop=True)
    df_sw.to_csv(stats_path + 'df_sw.csv')
    df_hrv = pd.concat(hrvs).reset_index(drop=True)
    df_hrv.to_csv(stats_path + 'df_sleep_hrv.csv')