#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:08:28 2024

@author: administrator
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import fooof 
import numpy as np
import seaborn as sns 

stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'
df = pd.read_pickle(stats_path + 'df_epochs.p')

psds_all = []
for mode, df_ in df.groupby('Mode'):
    print(mode)
    eps = [df_.reset_index().loc[d].Epochs for d in range(len(df_))]
    for ep in eps:
        epo_spectrum = ep.compute_psd(fmin=0.5, fmax=30, n_jobs=-1, picks='csd')
        for cond in list(ep.event_id):
            psds, freqs = epo_spectrum[cond].get_data(return_freqs=True)
            psds *= 1e6 
            
            psds_all.append(psds)
            
            # FOOOF data         
            fm = fooof.FOOOFGroup(max_n_peaks=5)
            fm.fit(freqs, psds.mean(0), freq_range=(3, 45))

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

#%%                
# 0. Load fooofed data
fooof_df = pd.read_pickle(os.path.join(stats_path, 'df_fooof.p'))
fooof_df = drop_bads_df(fooof_df)

nmes_stim = fooof_df.set_index(['Mode','Stim','Target_Chan']).loc['nmes', 'stim','c3']
nmes_sham = fooof_df.set_index(['Mode','Stim','Target_Chan']).loc['nmes', 'sham','c3']

fg_necm = fooof.objs.combine_fooofs(list(nmes_stim.fooof))
fooof_plot(fg_necm, 'CLNMES (Stim)')
fg_nece = fooof.objs.combine_fooofs(list(nmes_sham.fooof))
fooof_plot(fg_nece, 'CLNMES (Sham)')

pn_stim = fooof_df.set_index(['Mode','Stim','Target_Chan']).loc['pn','stim','fz']
pn_sham = fooof_df.set_index(['Mode','Stim','Target_Chan']).loc['pn','sham','fz']

fg_pecm = fooof.objs.combine_fooofs(list(pn_stim.fooof))
fooof_plot(fg_pecm, 'CLAS (Stim)')
fg_pece = fooof.objs.combine_fooofs(list(pn_sham.fooof))
fooof_plot(fg_pece, 'CLAS (Sham)')


nmes_stim_spectra = np.concatenate([np.expand_dims(nmes_stim.Spectra_mne[i].average().data, 0) 
                                    for i in range(len(nmes_stim))])
nmes_sham_spectra = np.concatenate([np.expand_dims(nmes_sham.Spectra_mne[i].average().data, 0) 
                                    for i in range(len(nmes_sham))])
plt.semilogy(nmes_stim.Spectra_mne[0].freqs, nmes_stim_spectra.mean(0).mean(0).T*1e6,
             linewidth=.5, color='k')
plt.semilogy(nmes_stim.Spectra_mne[0].freqs, nmes_sham_spectra.mean(0).mean(0).T*1e6,
             linewidth=.5, color='y')


#%%

df_power = pd.read_csv(stats_path + '/df_power.csv', index_col=0)
df_power = drop_bads_df(df_power)

test_df = df_power.set_index(['Mode','Stim','Target_Chan','Chan'])
test_df = test_df[test_df.Subject=='LiFe']

fooof_test_df = fooof_df[fooof_df.Subject=='LiFe'].set_index(['Mode','Stim','Target_Chan'])
fooof_test_df.loc['nmes','stim','c3'].Spectra_mne.plot()

yasa.topoplot(np.log(test_df.loc['nmes','stim','c3'].Delta) -
              np.log(test_df.loc['nmes','sham','c3'].Delta))


test_df.loc['nmes','sham','c3'].Delta / test_df.loc['nmes','sham','c3'].TotalAbsPow.sum()