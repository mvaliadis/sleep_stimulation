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
        cumulative_ = []
        sub_con = []
        # remove mastoids
        mask_ch = np.logical_and(events.Channel!='M1', events.Channel!='M2')
        for (sub, cond), sess in events[mask_ch].groupby(['Subject','Condition']):
            mask = get_mask(sess, sf=sf) 
            mask_ = pd.DataFrame(mask.T, 
                                 columns=events[mask_ch].Channel.unique()) 
            mask_.columns.name = "Channel" 
            coincidences = mask_.corr(method=_coincidence)
            cumulative_.append(coincidences.replace(1, np.nan))
            sub_con.append([sub, cond])
        cumulative = np.nanmean(np.asarray([cumulative_[i].to_numpy() 
                                 for i in range(len(cumulative_))]), 0)
        sub_con = np.concatenate([sub_con])
    
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
        
    
    return cumulative, cumulative_, sub_con

def plot_cm(cumulative, events, mask_ch):
    mask = np.triu(np.ones_like(cumulative, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(cumulative, mask=mask, cmap=cmap, center=0,
                vmin = np.nanpercentile(cumulative, 5), 
                vmax = np.nanpercentile(cumulative, 95),
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


def generate_random_hypno(tib=90, sf=1/30):
    """Simulate a hypnogram based on known transition probabilities.
    copied from: https://github.com/remrama/yasa/blob/agreement/notebooks/17_performance_evaluation.ipynb
    
    Transition probabilities were quickly snagged from Figure 8 in
    Metzner et al., 2021, Commun Biol, Sleep as a random walk: a super-statistical
    analysis of EEG data across sleep stages. https://doi.org/10.1038/s42003-021-02912-6
    
    Parameters
    ----------
    tib : int
        Time in bed, expressed in minutes.
    sf : float
        Sampling frequency.
        
    Returns
    -------
    hypno : np.ndarray
        Hypnogram containing simulated sleep stages.
    """

    def _markov_sequence(p_init, p_transition, sequence_length):
        """Generate a Markov sequence based on p_init and p_transition.
        Stolen from https://ericmjl.github.io/essays-on-data-science/machine-learning/markov-models
        """
        from scipy.stats import multinomial
        initial_state = list(multinomial.rvs(1, p_init)).index(1)
        states = [initial_state]
        for _ in range(sequence_length - 1):
            p_tr = p_transition[states[-1]]
            new_state = list(multinomial.rvs(1, p_tr)).index(1)
            states.append(new_state)
        return states

    # https://www.nature.com/articles/s42003-021-02912-6/figures/5
    trans_probas = np.array([
        [0.9468, 0.0002, 0.0461, 0.0068, 0.0002], # W R 1 2 3
        [0.0055, 0.9681, 0.0182, 0.0081, 0.0002], # R
        [0.0323, 0.0068, 0.7690, 0.1907, 0.0013], # 1
        [0.0089, 0.0096, 0.0377, 0.9259, 0.0178], # 2
        [0.0048, 0.0012, 0.0173, 0.0275, 0.9491], # 3
    ])

    init_probas = np.array([0.9468, 0.0002, 0.0461, 0.0068, 0.0002])

    n_epochs = np.floor(tib * 60 * sf).astype(int)
    hypno = _markov_sequence(init_probas, trans_probas, n_epochs)
    hypno = pd.Series(hypno).map({0:0, 1:4, 2:1, 3:2, 4:3}).to_numpy(int)
    
    return hypno