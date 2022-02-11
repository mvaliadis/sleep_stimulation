#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:37:19 2021

@author: administrator
"""


import scipy
import emd
import seaborn as sns 

#%%
def hilbert_huang_spectrum(IF, IA, sf, tmin, tmax, foi=(0,20), smooth=True):
    freq_edges, freq_centres = emd.spectra.define_hist_bins(foi[0], foi[1], foi[1]*3, 'linear')
    time_centres = np.arange(tmin, tmax + 1/sf, 1/sf)
    hht = [emd.spectra.hilberthuang(IF[i], IA[i], freq_edges, mode='amplitude') for i in range(len(IF))]
    hht = np.mean(hht, 0)
    if smooth:
        hht = scipy.ndimage.gaussian_filter(hht, 2)
    
    if plot:
        IF = np.nanmean(IF, 0)
        IA = np.nanmean(IA, 0)
        
        plt.figure(figsize=(8, 4))
        
        plt.subplot(121)
        # Plot a simple histogram using frequency bins from 0-20Hz
        plt.hist(IF, np.linspace(foi[0], foi[1]), density=True)
        plt.grid(True)
        plt.title('IF Histogram')
        plt.xticks(np.arange(foi[0], foi[1], 2))
        plt.xlabel('Frequency (Hz)')
        
        plt.subplot(122)
        # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
        plt.hist(IF, np.linspace(foi[0], foi[1]), weights=IA, 
                 density=True)
        plt.grid(True)
        plt.title('IF Histogram\nweighted by IA')
        plt.xticks(np.arange(foi[0], foi[1], 2))
        plt.xlabel('Frequency (Hz)')
        
        plt.tight_layout()
        sns.despine()

        emd.plotting.plot_hilberthuang(hht, time_centres, freq_centres,
                                       cmap='viridis', time_lims=(tmin + 0.25 ,tmax - 0.25),  
                                       log_y=True)

    return hht
