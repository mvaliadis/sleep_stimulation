#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:51:42 2023

@author: administrator
"""

import mne
import mne_microstates
from sleepstim.Analysis.analysis_pre_process import load_preprocessed_data 
from sleepstim.Analysis.Resting_State.rs_preproc import Resting_State_Data_Struct
                                                         
# example file paths
pre_f = '/media/administrator/data/Study_1_data/Pre-processed_data_resting_state/YIOYSRPX_2_pre_rs_preprocessed.p'
post_f = '/media/administrator/data/Study_1_data/Pre-processed_data_resting_state/YIOYSRPX_2_post_rs_preprocessed.p'
# load pre/post
Data_pre = load_preprocessed_data(pre_f)
Data_post = load_preprocessed_data(post_f)
# eyes open epochs
open_pre = mne.EpochsArray(Data_pre.data_eyes_open/1e3, Data_pre.mne_info, 
                                       tmin = 0, baseline=(None), verbose=0)
open_post = mne.EpochsArray(Data_post.data_eyes_open/1e3, Data_post.mne_info, 
                                        tmin = 0, baseline=(None), verbose=0) 
# eyes closed epochs
close_pre = mne.EpochsArray(Data_pre.data_eyes_close/1e3, Data_pre.mne_info, 
                                        tmin = 0, baseline=(None), verbose=0) 
close_post = mne.EpochsArray(Data_post.data_eyes_close/1e3, Data_post.mne_info, 
                                        tmin = 0, baseline=(None), verbose=0) 

#%%
## MNE microstates
# Segment the data into 6 microstates
maps, segmentation = mne_microstates.segment(open_pre.get_data()[:,0:64,:].mean(0)*1e3, n_states=4)

# Plot the topographic maps of the found microstates
mne_microstates.plot_maps(maps, open_pre.info);

# Plot the segmentation of the first 500 samples
mne_microstates.plot_segmentation(segmentation[:500], open_pre.get_data()[:,0:64,:].mean(0)[:,:500]*1e3, 
                                  post.times[:500])


#%%
import neurokit2 as nk

# CAR reference
eeg = nk.eeg_rereference(post, 'average')

# Extract microstates
microstates = nk.microstates_segment(open_pre.average(), n_microstates=4, sampling_rate=1000)

# Visualize the extracted microstates
nk.microstates_plot(microstates)

nk.microstates_static(microstates, sampling_rate=1000, show=True)

nk.microstates_dynamic(microstates, show=True)

microstates_all = nk.microstates_segment(eeg.mean(0), n_microstates=4, 
                                         train="all", sampling_rate=1000)

nk.microstates_plot(microstates_all, info=post.info)

# Note: we cropped the data as this function takes some time to compute 
n_optimal, scores = nk.microstates_findnumber(open_pre.average(), n_max=8, 
                                              show=True)  
print("Optimal number of microstates: ", n_optimal)

# Extract microstates
microstates_kmeans = nk.microstates_segment(open_pre.average(), n_microstates=4, method="kmeans")
microstates_kmod = nk.microstates_segment(open_pre.average(), n_microstates=4, method="kmod")  

# Global Explained Variance
gev_kmeans = microstates_kmeans['GEV']
gev_kmod = microstates_kmod['GEV']
print( f' Using conventional Kmeans,  GEV = {gev_kmeans*100:.2f}%')
print( f' Using modified Kmeans,  GEV = {gev_kmod*100:.2f}%')

# Visualize the extracted microstates
nk.microstates_plot(microstates_kmeans, epoch = (150, 450))
nk.microstates_plot(microstates_kmod, epoch = (150, 450))

