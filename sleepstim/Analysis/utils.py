#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:17:01 2022

@author: administrator
"""

import numpy as np 
import yasa
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 
from scipy import stats

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx
            
def _coincidence(x, y, scaled=True):
    """Calculate the (scaled) coincidence."""
    coincidence = (x * y).sum()
    if scaled:
        # Handle division by zero error
        denom = (x.sum() * y.sum())
        if denom == 0:
            coincidence = np.nan
        else:
            coincidence /= denom
    return coincidence
        
def coincidence_matrix(events, sf, plot=True, window='group', standardize=False):       
    # noqa
    if window == 'whole':
        coin_mat = [events[i].get_coincidence_matrix(scaled=True) for i in range(len(events)) if events[i] != None]
        cumulative = [coin_mat[i].to_numpy() for i in range(len(coin_mat))]
        
    elif window == 'group':       
        cumulative = []
        # remove mastoids
        mask_ch = np.logical_and(events.Channel!='M1', events.Channel!='M2')
        for (sub, cond), sess in events[mask_ch].groupby(['Subject','Condition']):
            mask = get_mask(sess, sf=sf) 
            mask_ = pd.DataFrame(mask.T, 
                                 columns=events[mask_ch].Channel.unique()) 
            mask_.columns.name = "Channel" 
            coincidences = mask_.corr(method=_coincidence)
            cumulative.append(coincidences)
        cumulative = np.nanmean(np.asarray([cumulative[i].to_numpy() 
                                 for i in range(len(cumulative))]), 0)
    
    # noqa
    elif window == 'post_stim':
        cumulative = []
        for epoch in range(len(events)):
            if events[epoch] != None:
                mask = events[epoch].get_mask()[:,sf*4:]
                mask = pd.DataFrame(mask.T, columns=events[epoch]._ch_names) 
                mask.columns.name = "Channel" 
                cumulative.append(mask.corr(method=_coincidence))
            
    if plot: 
        if window != 'group':
            cumulative = np.nanmean(cumulative, 0)
        if standardize:
            from sklearn.preprocessing import StandardScaler
            #####
            for i in range(21):
                cumulative.to_numpy()[i,i]=0
            scaler = StandardScaler()
            cumulative = scaler.fit_transform(cumulative)
            #####
            # cumulative = stats.zscore(cumulative)
        # Plot
        plot_cm(cumulative, events, mask_ch)
        
    
    return cumulative

def plot_cm(cumulative, events, mask_ch):
    mask = np.triu(np.ones_like(cumulative, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(cumulative, mask=mask, cmap=cmap, center=0,
                vmin = np.percentile(cumulative, 5), 
                vmax = np.percentile(cumulative, 95),
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, 
                yticklabels=events[mask_ch].Channel.unique(), 
                xticklabels=events[mask_ch].Channel.unique())
    plt.tight_layout()

def get_mask(events, sf):
    """get_mask"""
    from yasa.others import _index_to_events
    # chans = len(events['IdxChannel'].unique())
    chans = 21 #should be pre-configured as such to exclude mastoids
    data_len = 210*60*sf
    mask = np.zeros(shape = (chans, data_len), 
                    dtype = int)
    for i in range(chans):
        ev_chan = events[events['IdxChannel'] == i]
        if len(ev_chan) > 0:
            idx_ev = _index_to_events(
                ev_chan[['Start', 'End']].to_numpy() * sf)
            mask[i, idx_ev] = 1
        else:
            mask[i, :] = 0
            
    return np.squeeze(mask)     
   
def spectral_sorting_func_mne(times, epochs, trange=(-0.5, 0.5)):
    """
    For more details on spectral sortin, see:
        
         Gramfort, A., Keriven, R., & Clerc, M. (2010). 
         Graph-based variability estimation in single-trial 
         event-related neural responses. 
         IEEE Transactions on Biomedical Engineering, 57(5), 1051-1061.
         
    see also:
        https://mne.tools/stable/auto_examples/visualization/channel_epochs_image.html#sphx-glr-auto-examples-visualization-channel-epochs-image-py   

    Returns
    -------
    None.

    """
    from sklearn.manifold import spectral_embedding 
    from sklearn.metrics.pairwise import rbf_kernel

    # process epoched data
    this_data = epochs[:, (times > trange[0]) & (times < trange[1])]
    this_data /= np.sqrt(np.sum(this_data ** 2, axis=1))[:, np.newaxis]
    
    # return sorted epochs 
    return np.argsort(spectral_embedding(rbf_kernel(this_data, gamma=1.),
                      n_components=1, random_state=0).ravel())