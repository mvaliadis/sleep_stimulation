#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 21:32:12 2021

@author: administrator
"""

import numpy as np
import mne
from tensorpac import Pac, EventRelatedPac
from tensorpac.stats import test_stationarity
import matplotlib.pyplot as plt

#%%
## Test stationarity - Augmented Dickey-Fuller test (unit root test)
def unit_root_test(data, p=0.05):
    df_stationarity = test_stationarity(data.get_data(picks='C3').squeeze()*1e6, p=0.05)
    # return epochs which don't violate stationarity assumption
    return np.where(df_stationarity['Stationary']==True)[0]

def add_stimulus_onset(color = 'white'):
    plt.axvline(0., lw=2, color=color)
    plt.axvline(1.075, lw=2, color=color)

## Create ERPAC plot
def ERPAC(data, f_pha=[0.5, 4], f_amp=(4, 30, .25, .25), n_perm=None, smooth=200, 
          method = 'gc', edges=0.5, stationarity_t=False, plot=True, save_path=None):
    rp_obj = EventRelatedPac(f_pha=f_pha, f_amp=f_amp)
    edges=int(edges*data.info['sfreq'])
    if stationarity_t:
        stationary_epochs = unit_root_test(data, p=0.05)
        if len(stationary_epochs) > 0:
            erpac = rp_obj.filterfit(int(data.info['sfreq']), data.get_data(picks='C3').squeeze()[stationary_epochs,:],
                                     method=method, smooth=smooth, edges=edges, n_perm=n_perm)
        else:
            return np.nan
    else:
        erpac = rp_obj.filterfit(int(data.info['sfreq']), data.get_data(picks='C3').squeeze(),
                                 method=method, smooth=smooth, edges=edges, n_perm=n_perm)
    if plot:
        plt.figure(figsize=(8, 6))
        rp_obj.pacplot(erpac.squeeze(), data.times[edges:-1-edges], rp_obj.yvec, xlabel='Time',
                       ylabel='Amplitude frequency (Hz)',
                       title='Event-Related PAC occurring for Delta phase',
                       fz_labels=15, fz_title=18)
        add_stimulus_onset(color = 'white')
        plt.show()
        if save_path is not None:
            plt.savefig(save_path + '_erpac_plot.png')
    
    return erpac

# erpac = ERPAC(epochs_mastoids)

# #%%
# p_obj = Pac(idpac=(6, 0, 0), f_pha=(0.3, 4.0, 1, .3), f_amp=(4, 30, 1, .3))
# # extract all of the phases and amplitudes
# amp_p = p_obj.filter(512, epochs_mastoids.get_data(picks='C3').squeeze(), ftype='amplitude')
# pha_p = p_obj.filter(512, epochs_mastoids.get_data(picks='C3').squeeze(), ftype='phase')
# # define time indices 
# pre_stim = slice(256, 1024)
# post_stim = slice(1024, 2561-256)
# # define phase / amplitude 
# pha_pre, amp_pre = pha_p[..., pre_stim], amp_p[..., pre_stim]
# pha_post, amp_post = pha_p[..., post_stim], amp_p[..., post_stim]
# # compute PAC 
# pac_pre = p_obj.fit(pha_pre, amp_pre).mean(-1)
# pac_post = p_obj.fit(pha_post, amp_post).mean(-1)

# # plot the comodulogram
# vmax = np.max([pac_pre.max(), pac_post.max()])
# vmin = np.min([pac_pre.min(), pac_post.min()])
# kw = dict(vmax=vmax, vmin=vmin, cmap='viridis')
# plt.figure(figsize=(10, 4))
# plt.subplot(121)
# p_obj.comodulogram(pac_pre, title="PAC [-1.5, 0]s", **kw)
# plt.subplot(122)
# p_obj.comodulogram(pac_post, title="PAC [0, 2.5]s", **kw)
# plt.ylabel('')
# plt.tight_layout()
# plt.show()

# #%%
# # for computing the permutations
# p_obj.idpac = (6, 2, 0)
# # compute pac and 200 surrogates
# pac_prep = p_obj.fit(pha_p[..., time], amp_p[..., time], n_perm=200,
#                      random_state=0)
# # get the p-values
# mcp = 'maxstat'
# pvalues = p_obj.infer_pvalues(p=0.05, mcp=mcp)
# # sphinx_gallery_thumbnail_number = 7
# plt.figure(figsize=(8, 6))
# title = (r"Significant delta$\Leftrightarrow$broadband (theta-beta) coupling occurring during "
#          f"pinknoise stimulation\n(p<0.05, {mcp}-corrected for multiple "
#           "comparisons)")
# # plot the non-significant pac in gray
# pac_prep_ns = pac_prep.mean(-1).copy()
# pac_prep_ns[pvalues < .05] = np.nan
# if np.nan in pac_prep_ns:
#     colorbar_s = False
# else:
#     colorbar_s = True
# p_obj.comodulogram(pac_prep_ns, cmap='gray', vmin=np.nanmin(pac_prep_ns),
#                    vmax=np.nanmax(pac_prep_ns), colorbar=colorbar_s)
# # plot the significant pac in color
# pac_prep_s = pac_prep.mean(-1).copy()
# pac_prep_s[pvalues >= .05] = np.nan
# p_obj.comodulogram(pac_prep_s, cmap='Spectral_r', vmin=np.nanmin(pac_prep_s),
#                    vmax=np.nanmax(pac_prep_s), title=title)
# plt.gca().invert_yaxis()
# plt.show()
