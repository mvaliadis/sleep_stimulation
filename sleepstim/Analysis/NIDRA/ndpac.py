#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:20:36 2024

@author: administrator
"""

import tensorpac.methods as tpm
from tensorpac import EventRelatedPac, Pac
import pandas as pd
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator
import pingouin as pg

def drop_bads_df(df, subject_nights=[('ChrSt', 1), ('UyDe', 1), ('IsEb', 2)]):
    # Ensure the night numbers are of the same type as in the DataFrame
    # If Night is a string in the DataFrame, convert the night numbers to strings
    subject_nights = [(subj, str(night)) if isinstance(df['Night'].iloc[0], str) else (subj, night) for subj, night in subject_nights]
    
    # Create masks for each condition to drop
    masks = [((df['Subject'] == subj) & (df['Night'] == night)) for subj, night in subject_nights]
    
    # Combine the individual masks with a logical OR
    if masks:
        combined_mask = reduce(operator.or_, masks)
    else:
        combined_mask = pd.Series([False] * len(df))
    
    # Apply the mask to filter out the rows
    df = df[~combined_mask]
    return df

stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'
df_evoked = pd.read_pickle(os.path.join(stats_path, 'df_evokeds.p'))
df_evoked = drop_bads_df(df_evoked)
df_evoked.set_index(['Mode','Subject'], inplace=True)

# Initialize the list to hold all ndPAC data
ndPACs = []
for cond in ['Evoked_sham_c3', 'Evoked_stim_c3',
             'Evoked_sham_fz', 'Evoked_stim_fz']:
    data = df_evoked[cond].loc['nmes']
    p_ndpac = Pac(f_pha=[0.5, 1.5], f_amp=[12, 16.5, 0.5, 0.5])
    p = Pac(idpac = (6, 0, 0), f_pha=[0.375, 5.25, 0.25, 0.25], f_amp=[4.75, 30.5, 0.5, 0.5])
    for chan in ['C3','Fz']: # data[0].ch_names:
        # Sf
        sf = data[0].info['sfreq']
        
        # Get channel data
        chan_data = np.concatenate([np.expand_dims(data[i].get_data(chan).squeeze()*1e3, 0) 
                                    for i in range(len(data))])
        
        # PAC
        # Filter the data and extract the PAC values
        xpac = p.filterfit(sf, chan_data, edges=sf, n_jobs=-1)
        # Plot the comodulogram
        plt.figure()
        p.comodulogram(xpac.mean(-1), vmin=0, title=str(p), plotas='imshow');
        
        # Extract PAC values into a DataFrame
        df_pac = pd.DataFrame(xpac.mean(-1), columns=p.xvec, index=p.yvec)
        df_pac.columns.name = 'FreqPhase'
        df_pac.index.name = 'FreqAmplitude'

        print(f"Maximum Phase Value: {df_pac.max(axis=0).idxmax()}")
        print(f"Maximum Amplitude Value: {df_pac.max(axis=1).idxmax()}")
                 
        # ERPAC (+/- 3 sec to avoid filter edge)
        data_erpac = chan_data
        erp = EventRelatedPac(f_pha=[0.5, 1.5], f_amp=np.arange(4.75, 25.75, 0.5), 
                              verbose=False)  # f_pha = 0.8 Hz
        freqs = erp.f_amp.mean(1).astype(str)
        pha = erp.filter(sf, data_erpac, ftype='phase')
        amp = erp.filter(sf, data_erpac, ftype='amplitude')
        ergcpac = np.squeeze(erp.fit(pha, amp, method="gc", smooth=50))  # Slow -- Gaussian Copula
                
        # ERGPAC Plot
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        im = plt.imshow(ergcpac, aspect='auto', cmap="Spectral_r", origin='upper',
                        interpolation="gaussian", 
                        #vmin=-0.2, vmax=1,
                        extent=[data[0].times[0], data[0].times[-1], 
                                freqs[-1], freqs[0]])
        
        plt.gca().invert_yaxis()
        
        fig.suptitle(f"{cond} ({chan})")
        plt.xlabel("Time from stim onset (s)")
        plt.ylabel("Frequency (Hz)")
        plt.axvline(0, ls=":", lw=1.5, color="k")
        
        cb = plt.colorbar(im, shrink=0.7, pad=0.05, aspect=20)
        cb.set_label("Coupling (z-score)")
        cb.outline.set_visible(False)
        
        ax_sw = ax.twinx()
        ax_sw.plot(data[0].times, chan_data.mean(0), color="k", lw=3)
        ax_sw.set_yticks([]);

        # Get channel data
        sw_pha = p_ndpac.filter(sf, chan_data, ftype='phase', edges=None, n_jobs=-1).squeeze(0)
        sp_amp = p_ndpac.filter(sf, chan_data, ftype='amplitude', edges=None, n_jobs=-1).squeeze()
        sp_amp = sp_amp.mean(0)
            
        # Extract phase at pre/post stimulation
        start_idx, end_idx = data[0].time_as_index((-1, 1))
        pre_sw_pha = sw_pha[:, start_idx:end_idx]
    
        # Extract phase at pre/post stimulation
        pre_sp_amp = sp_amp[:, start_idx:end_idx]
        
        # ndPAC calculation 
        # Find location of max sigma amplitude in pre/post epoch
        idx_max_amp_pre = pre_sp_amp.argmax(axis=1).reshape(-1, 1)
                  
        # PhaseAtSigmaPeak
        # Find SW phase at max sigma amplitude in epoch            
        pha_at_max_pre = np.squeeze(np.take_along_axis(pre_sw_pha,
                                                       idx_max_amp_pre,
                                                       axis=1))
      
        # Normalized Direct PAC, with thresholding
        ndp_pre = np.squeeze(tpm.norm_direct_pac(pre_sw_pha[None, :],
                                                 pre_sp_amp[None, :], p=0.05))
        
        # Plot
        f, ax = plt.subplots(3, 1, layout='constrained')
        times = np.arange(-1.5, 0.5, 1/sf)
        ax[0].plot(times, chan_data[:, start_idx:end_idx].mean(0))
        ax[1].plot(times, pre_sp_amp.mean(0))
        ax[1].vlines(np.median(times[idx_max_amp_pre]), ymin=0, ymax=1)
        ax[2].plot(times, pg.circ_mean(pre_sw_pha))
        f.suptitle(f'{cond} : {chan} : ndPAC : {np.round(np.mean(ndp_pre), 2)}')
        
        
                                 
        # Tuple pairs of conditions and their respective data for looping
        conditions_data = [
            ("Pre", pha_at_max_pre, ndp_pre),
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

