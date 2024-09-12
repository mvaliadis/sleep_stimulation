#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 23:46:26 2024

@author: administrator
"""

from glob import glob
import mne
import numpy as np
import yasa
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt

mne.set_log_level('ERROR')

def drop_bads_df(df, subject_nights=[('ChrSt', 1), ('UyDe', 1), ('IsEb', 2)]):
    from functools import reduce
    import operator
    # Ensure the night numbers are of the same type as in the DataFrame
    # If Night is a string in the DataFrame, convert the night numbers to strings
    subject_nights = [(subj, str(night)) if isinstance(df['night'].iloc[0], str) else (subj, night) for subj, night in subject_nights]
    
    # Create masks for each condition to drop
    masks = [((df['subject'] == subj) & (df['night'] == night)) for subj, night in subject_nights]
    
    # Combine the individual masks with a logical OR
    if masks:
        combined_mask = reduce(operator.or_, masks)
    else:
        combined_mask = pd.Series([False] * len(df))
    
    # Apply the mask to filter out the rows
    df = df[~combined_mask]
    return df

data_dir = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/*csd-epo.fif'

# Define regions
channels_fa = ['F1', 'F2', 'Fz']
channels_ma = ['C1', 'C3', 'CP3']
regions = {'MC': channels_ma, 'FA': channels_fa}
bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
         (12, 16, 'Sigma'), (16, 30, 'Beta'), (30, 45, 'Gamma')]

# Collect rows in a list
rows = []

for file_path in glob(data_dir):
    print(f"Processing {file_path}")

    # Extract subject and night from the file name
    night = file_path.split('/')[-1].split('_')[1]
    subject = file_path.split('/')[-1].split('_')[0]

    # Check event IDs before loading the full epochs
    temp_epochs = mne.read_epochs(file_path, preload=False)
    if 'pn_sham_c3' in list(temp_epochs.event_id):
        mode = 'CLAS'
    else:
        mode = 'CLNMES'

    epochs = mne.read_epochs(file_path, preload=True)

    for event_id in list(temp_epochs.event_id):
        epochs_cond = epochs[event_id].copy()
        
        # Extract condition and reference from event_id
        condition = 'stim' if 'stim' in event_id else 'sham'
        reference = 'C3' if 'c3' in event_id else 'Fz'

        for region, channels in regions.items():
            epochs_region = epochs_cond.copy().pick(channels)
            # Extract power for each channel per area of interest per epoch
            data = epochs_region.get_data(picks=channels, tmin=0, tmax=2) * 1e3
            sf = epochs.info['sfreq']
            win = int(2 * sf)  # Window size is set to 4 seconds
            freqs, psd = welch(data, sf, nperseg=win, axis=-1)
            
            # Fooof
            import fooof
            fm = fooof.FOOOFGroup(max_n_peaks=5)
            fm.fit(freqs, psd.mean(0), freq_range=(3, 45))  
            spect_exponent = fm.get_params(name='aperiodic_params', col='exponent')
            
            fooofed_spectrum_ = np.mean([fm.get_fooof(ind=idx, regenerate=True).fooofed_spectrum_ for idx in range(len(fm))], axis=0)
            ap_fit = np.mean([fm.get_fooof(ind=idx, regenerate=True)._ap_fit for idx in range(len(fm))], axis=0)
            
            oscillatory_component = fooofed_spectrum_ - ap_fit

            # Calculate fooofed oscillatory spectrum bandpower
            bandpower_osc = yasa.bandpower_from_psd(oscillatory_component, 
                                                    fm.freqs, 
                                                    ch_names=region,
                                                    bands=[(4, 8, 'Theta'), (8, 12, 'Alpha'),
                                                           (12, 16, 'Sigma'), (16, 30, 'Beta'), 
                                                           (30, 45, 'Gamma')], relative=True)
            
            # Calculate the bandpower on 3-D PSD array
            bandpower = yasa.bandpower_from_psd_ndarray(psd, freqs, bands, relative=False)
            bandpower_roi = bandpower.mean(1).mean(-1)  # Average power for all epochs per region of interest

            # Prepare the row to be appended to the DataFrame
            row = {
                'subject': subject,
                'night': night,
                'mode': mode,
                'region': region,
                'condition': condition,
                'reference': reference,
                'Aperiodic': np.nanmean(spect_exponent),
                'Delta': np.log(bandpower_roi[0]),
                'Theta': np.log(bandpower_roi[1]),
                'Alpha': np.log(bandpower_roi[2]),
                'Sigma': np.log(bandpower_roi[3]),
                'Beta': np.log(bandpower_roi[4]),
                'Gamma': np.log(bandpower_roi[5]),
                'Theta_Osc': bandpower_osc['Theta'][0],
                'Alpha_Osc': bandpower_osc['Alpha'][0],
                'Sigma_Osc': bandpower_osc['Sigma'][0],
                'Beta_Osc': bandpower_osc['Beta'][0],
                'Gamma_Osc': bandpower_osc['Gamma'][0],
                # 'Oscillatory_Spectrum' : [oscillatory_component],
            }

            # Append the row to the list
            rows.append(row)

# Create DataFrame from the list of rows
all_bandpower_results = pd.DataFrame(rows)
all_bandpower_results = drop_bads_df(all_bandpower_results)

# Save the DataFrame to a CSV file
all_bandpower_results.to_csv('/media/administrator/Sleep_Data/Processed/Statistics/bp_log.csv', index=False)

#%%

# # Create stim - sham contrast
# stim_results = all_bandpower_results[all_bandpower_results['condition'] == 'stim']
# sham_results = all_bandpower_results[all_bandpower_results['condition'] == 'sham']

# # Ensure matching subjects, nights, modes, regions, and references
# common_keys = ['subject', 'night', 'mode', 'region', 'reference']
# stim_sham_contrast = pd.merge(stim_results, sham_results, on=common_keys, suffixes=('_stim', '_sham'))

# # Calculate contrast
# for band in ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma', 'Aperiodic']:
#     stim_sham_contrast[f'{band}_contrast'] = stim_sham_contrast[f'{band}_stim'] - stim_sham_contrast[f'{band}_sham']

# # Select only the relevant columns for contrast
# contrast_columns = common_keys + [f'{band}_contrast' for band in ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma', 'Aperiodic']]
# stim_sham_contrast = stim_sham_contrast[contrast_columns]

# # Save the contrast results to a CSV file
# stim_sham_contrast.to_csv('/media/administrator/Sleep_Data/Processed/Statistics/bp_contrast.csv', index=False)

