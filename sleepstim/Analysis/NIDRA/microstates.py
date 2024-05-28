#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:07:44 2024

@author: administrator
"""

import mne
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import seaborn as sns
from pycrostates.preprocessing import extract_gfp_peaks
from pycrostates.preprocessing import resample
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
mne.set_log_level('ERROR')

file = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/TiKl_2_csd-epo.fif'

epochs = mne.read_epochs(file).pick('csd')

tmin, tmax = (-0.5, 1.5)

resamples = resample(epochs['nmes_stim_c3'], n_resamples=3, 
                     n_samples=np.diff(epochs.time_as_index((tmin, tmax)))[0], 
                     random_state=42, tmin=tmin, tmax=tmax)

resample_results = []
for k, resamp in enumerate(resamples):
    # fit Modified K-means
    ModK = ModKMeans(n_clusters=5, random_state=42)
    ModK.fit(resamp, n_jobs=-1, picks='csd', verbose="WARNING")
    resample_results.append(ModK.cluster_centers_)

all_resampling_results = np.vstack(resample_results).T
all_resampling_results = ChData(all_resampling_results, epochs.info)

ModK = ModKMeans(n_clusters=5, random_state=42)
ModK.fit(all_resampling_results, picks='csd', n_jobs=-1, verbose="WARNING")
fig = ModK.plot()
fig.suptitle(f"GEV: {ModK.GEV_:.2f}", fontsize=14)

#%%

segmentation = ModK.predict(
    epochs['nmes_stim_c3'],
    reject_by_annotation=True,
    factor=10,
    half_window_size=10,
    min_segment_length=5,
    reject_edges=True,
)

segmentation.plot()
plt.show()

#%%
parameters = segmentation.compute_parameters()

x = ModK.cluster_names
y = [parameters[elt + "_gev"] for elt in x]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel("Microstates")
ax.set_ylabel("Global explained Variance (ratio)")
plt.show()


x = ModK.cluster_names
y = [parameters[elt + "_mean_corr"] for elt in x]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel("Microstates")
ax.set_ylabel("Mean correlation")
plt.show()

x = ModK.cluster_names
y = [parameters[elt + "_timecov"] for elt in x]

ax = sns.barplot(x=x, y=y)
ax.set_xlabel("Microstates")
ax.set_ylabel("Time Coverage (ratio)")
plt.show()