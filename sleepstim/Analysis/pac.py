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
import yasa
from scipy.fftpack import next_fast_len
import emd 
import tensorpac.methods as tpm

#%%
## Test stationarity - Augmented Dickey-Fuller test (unit root test)
def unit_root_test(data, p=0.05):
    df_stationarity = test_stationarity(data.get_data(picks='C3').squeeze()*1e6, p=0.05)
    # return epochs which don't violate stationarity assumption
    return np.where(df_stationarity['Stationary']==True)[0]

def add_stimulus_onset(color = 'white'):
    plt.axvline(0., lw=2, color=color)
    plt.axvline(1.075, lw=2, color=color)

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

## extarct phase and amplitude 
def imf_analytical_transform(epoch):
    ## Phase analysis (filter with 2.0 Hz lp filter)
    C3_lp = epoch.copy().filter(l_freq=None, h_freq=2.0).get_data(picks='C3').squeeze()*1e6
    ## phase with imf - use freq transform 
    imf_sw = [emd.sift.sift(C3_lp[i,:], imf_opts={'sd_thresh': 0.1}, max_imfs=1) for i in range(min(C3_lp.shape))]
    sw_pha, sw_freq, sw_amplitude = [], [], []
    for i in range(min(C3_lp.shape)):
        sw_pha.append(emd.spectra.frequency_transform(imf_sw[i], epoch.info['sfreq'], 'nht')[0][:,0])
        sw_freq.append(emd.spectra.frequency_transform(imf_sw[i], epoch.info['sfreq'], 'nht')[1][:,0])
        sw_amplitude.append(emd.spectra.frequency_transform(imf_sw[i], epoch.info['sfreq'], 'nht')[2][:,0])
        
    return sw_pha, sw_freq, sw_amplitude

def extract_pha_amp(data_narrow, data_broad, sf, method = 'hilbert'):
    from scipy.fftpack import next_fast_len
    # Extract the spindles-related sigma signal for coupling
    data_sp = mne.filter.filter_data(data_broad, sf, 12, 16, method='fir',
                                     l_trans_bandwidth=1.5, h_trans_bandwidth=1.5,
                                     verbose=0)
    n_samples = max(data_narrow.shape)
    nfast = next_fast_len(n_samples)
    if method == 'hilbert':
        # Now extract the instantaneous phase/amplitude using Hilbert transform
        sw_pha = np.angle(signal.hilbert(data_narrow, N=nfast)[:n_samples])
        sp_amp = np.abs(signal.hilbert(data_sp, N=nfast)[:n_samples])
    elif method == 'emd':
        import emd           
        ## get sw phase    
        imf_sw = emd.sift.sift(data_narrow, imf_opts={'sd_thresh': 0.1}, max_imfs = 1)[:,0]
        # emd.plotting.plot_imfs(imf_sw, cmap=True, scale_y=True)
        sw_pha, _, _ = emd.spectra.frequency_transform(imf_sw, sf, 'nht')
        
        ## get sp phase
        imf_sp = emd.sift.sift(data_sp, imf_opts={'sd_thresh': 0.1}, max_imfs = 1)[:,0]
        # emd.plotting.plot_imfs(imf_sp, cmap=True, scale_y=True)
        _, _, sp_amp = emd.spectra.frequency_transform(imf_sw, sf, 'nht')
                 
    return sw_pha, sp_amp  

## Create ERPAC plot
def ERPAC(data, f_pha=[0.5, 4], f_amp=(5, 25, .25, .25), n_perm=None, smooth=100, 
          method = 'gc', edges=1.0, stationarity_t=False, plot=True, save_path=None):
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

## Iterate over yasa results to add ndPAC values to specified target SOs
def SO_spindle_coupling(sw, data_broad, idx, sf, target='stim_onset'):
    data_broad = data_broad[idx,:,:]
    time_before = 2.0; time_after = 2.0
    bef = int(sf * time_before)
    aft = int(sf * time_after)
      
    ## Iterate by channels
    summaries = []
    for chan in range(23):
        if target == 'neg_peak':
            sw_neg_times = sw.summary()['NegPeak'][sw.summary()['IdxChannel']==chan].to_numpy()
            idx_neg_nearest = find_nearest(sw_neg_times, 4)
            sw_neg_time = sw_neg_times[idx_neg_nearest]
            sw_neg_idx = sw_neg_time * sf
        elif target == 'stim_onset':
            ## One can ignore all the values in the dataframe, except the ndPAC info 
            sw_neg_times = sw.summary()['MidCrossing'][sw.summary()['IdxChannel']==chan].to_numpy()
            if list(sw_neg_times) != []:
                idx_neg_nearest = find_nearest(sw_neg_times, 4)
                sw_neg_time = sw_neg_times[idx_neg_nearest]
                sw_neg_idx = int(sw_neg_time*sf) #4*sf
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
            sw_pha, sp_amp = extract_pha_amp(sw._data[chan,:].squeeze(), data_broad[chan,:].squeeze(), sf, method='emd')                               
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
            time_sigpk_abs = (sw_neg_idx / sf) + time_sigpk
            summary['SigmaPeak'] = time_sigpk_abs
            
            # 2) PhaseAtSigmaPeak
            # Find SW phase at max sigma amplitude in epoch
            pha_at_max = np.squeeze(np.take_along_axis(sw_pha,
                                                       idx_max_amp[..., None],
                                                       axis=0))
            summary['PhaseAtSigmaPeak'] = np.ones(1) * np.nan
            summary['PhaseAtSigmaPeak'] = pha_at_max
            
            # 3) Normalized Direct PAC, with thresholding
            ndp = np.squeeze(tpm.norm_direct_pac(sw_pha.T[None, ...],
                                                 sp_amp.T[None, ...], p=0.05))
            summary['ndPAC'] = np.ones(1) * np.nan
            summary['ndPAC'] = ndp
            
        else:
            pass
        
        summaries.append(summary)
        
    return summaries

#%%
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
