#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:36:53 2024

@author: administrator
"""

import numpy as np
import mne  
import matplotlib.pyplot as plt
import yasa
       
def surface_laplacian_pre(inst, sphere='auto', lambda2=1e-3,
                          stiffness=4, n_legendre_terms=50):    
    picks = mne.pick_types(inst.info, meg=False, eeg=True, exclude=[])
    radius, origin_head, origin_device = mne.bem.fit_sphere_to_headshape(inst.info)
    x, y, z = origin_head - origin_device
    sphere = (x, y, z, radius)
    sphere = np.array(sphere, float)
    x, y, z, radius = sphere
        
    pos = np.array([inst.info['chs'][pick]['loc'][:3] for pick in picks])
    pos -= (x, y, z)
    
    # Project onto a unit sphere to compute the cosine similarity:
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    cos_dist = np.clip(np.dot(pos, pos.T), -1, 1)
    # This is equivalent to doing one minus half the squared Euclidean:
    # from scipy.spatial.distance import squareform, pdist
    # cos_dist = 1 - squareform(pdist(pos, 'sqeuclidean')) / 2.
    del pos    
    
    G = mne.preprocessing._csd._calc_g(cos_dist, stiffness=stiffness,
                                       n_legendre_terms=n_legendre_terms)
    H = mne.preprocessing._csd._calc_h(cos_dist, stiffness=stiffness,
                                       n_legendre_terms=n_legendre_terms)
    
    G_precomputed = mne.preprocessing._csd._prepare_G(G, lambda2)
       
    trans_csd = mne.preprocessing._csd._compute_csd(G_precomputed=G_precomputed,
                                                    H=H, radius=radius)
    
    return trans_csd 

def surface_laplacian_rt(data, trans_csd):
    #epochs = inst._data
    epochs = np.expand_dims(data.T, 0)
    for epo in epochs:
        csd_data = np.dot(trans_csd, epo)
    
    return csd_data

def re_reference(data, ch_names, trans_csd=None,
                 sf=512, reference='common average'):
    """EEG data re_referencing function.
    
    This function takes a numpy array of the data & rereferences the data based on
    the new selected reference. 
    
    Parameters
    ----------
    data : numpy array of [n_samples x n_chans]
        The data. Include only EEG channels for this argument!
    
    reference : str {‘common average’, ‘mastoids’}
        The new reference; default is ‘common average’.
    
    Returns
    -------
    data : numpy array of shape [n_samples x n_chans]
        The re-referenced data
    """
    # %timeit = 97 microseconds for 30s of data (fs = 256 Hz)
    # Initial safety check to confirm data shape is [n_samples x n_chans]
    dpnts, chans = data.shape
    if chans > dpnts:
        data = np.transpose(data)
    if reference == 'common average':
        # Compute mean of each epoch per channel then subtract common average
        ref_data = data[..., :].mean(-1, keepdims=True)
        data -= ref_data
        #print('Data will be re-referenced to the common average of all selected EEG electrodes!')
    elif reference == 'mastoids':
        # May change to name indexed instead of numerical index...
        ref_data = data[..., [ch_names.index('M1'), ch_names.index('M2')]].mean(-1, keepdims=True)
        data -= ref_data
        #print('Data will be re-referenced to the average of the mastoids!')
    elif reference == 'csd':
        data = surface_laplacian_rt(data=data, trans_csd=trans_csd) #np.expand_dims(d,1))
    else:
        raise ValueError('Please select a valid reference!')
    
    return data


filename = '/media/administrator/data/Study_2_data/NIDRA/CLNMES/PaJa_1-test.fif'
data = mne.io.read_raw(filename, preload=True)
data.filter(l_freq=0.5, h_freq=4)

#%%
trans_csd_new = surface_laplacian_pre(data, sphere='auto', lambda2=1e-5,
                                      stiffness=4, n_legendre_terms=50)
       
trans_csd_old = np.load('/home/administrator/sleep_stimulation/sleepstim/Analysis/CSD_matrix.npy')
sl_data = re_reference(data=data.get_data(picks='eeg', units='uV').T, 
                       ch_names=data.copy().pick('eeg').ch_names,
                       trans_csd=trans_csd_new, sf=data.info['sfreq'], 
                       reference='csd')

sl_data_mne = mne.preprocessing.compute_current_source_density(
    data.copy().pick('eeg'), 
    sphere='auto',
    lambda2=1e-05,
    stiffness=4,
    n_legendre_terms=50
    ).get_data()*1e6

# # Compute the diagonal of the correlation matrix
# diagonal_correlation = np.array([np.corrcoef(sl_data[i, :], 
#                                              sl_data_mne[i, :])[0, 1] for i in range(64)])

# # Plot the diagonal correlations
# plt.figure(figsize=(10, 6))
# plt.plot(diagonal_correlation, marker='o')
# plt.xlabel("Channel")
# plt.ylabel("Correlation")
# plt.title("Diagonal Correlations Across Channels")
# plt.grid(True)
# plt.show()

#%%
sf = data.info['sfreq']
sw = yasa.sw_detect(data=data.copy().pick('eeg'), sf=data.info['sfreq'], 
                    remove_outliers=True, coupling=False, freq_sw=(0.3, 1.5),
                    dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_ptp=(50, 500))
peak_idx, _ = yasa.get_centered_indices(data.get_data('C3', units='uV').squeeze(),  
                                        np.asarray(sw.summary()[sw.summary().Channel=='C3']['PosPeak']*sf), 
                                        npts_before = int(sf*1.98), npts_after = int(sf*0.02))
epochs = data.get_data('eeg', units='uV')[:, peak_idx]
# n_epochs, n_channels, n_times
epochs_mne = mne.EpochsArray(np.swapaxes(epochs, 0, 1)/1e6, 
                             data.copy().pick('eeg').info, tmin = -1.98, baseline=None)
epochs_mne.set_montage(mne.channels.make_standard_montage('standard_1005'))

epochs_csd = mne.preprocessing.compute_current_source_density(epochs_mne, 
                                                              stiffness=4,
                                                              n_legendre_terms=50,
                                                              lambda2=1e-05)
epochs_csd.average().plot_joint()

#%%
# Example matrices (randomly generated for demonstration)
# Replace these with your actual matrices
matrix1 = trans_csd_old
matrix2 = trans_csd_new

# Method: Pearson Correlation Coefficient
# Flatten matrices to 1D arrays for correlation computation
flattened_matrix1 = matrix1.flatten()
flattened_matrix2 = matrix2.flatten()

# Compute Pearson correlation coefficient
correlation_coefficient = np.corrcoef(flattened_matrix1, flattened_matrix2)[0, 1]
correlation_coefficient







