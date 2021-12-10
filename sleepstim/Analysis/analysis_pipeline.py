#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:26:05 2021

@author: administrator
"""

# =============================================================================
# Analysis Pre-processing script
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle 
import pandas as pd
import pyxdf
import yasa 
import mne
import pingouin as pg
from mne.stats import permutation_cluster_test
import logging
import time
import wonambi
import seaborn as sns
from os import chdir as cd
from os import listdir
import os, shutil
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, preprocess_sleep_data, compare_hypnograms, check_match_data_hypno_elements, 
                                                     label_artifacts, load_preprocessed_data, Data_SW, SW_ERPs)

#%%
## Step 1a - Pre-process data ##
## The following function allows for preprocessing of data yet to be processed from requisite given path 
## by checking whether files have been preprocessed. The end result is a data object which contains the data, 
## channels, channel types, time stamps, pinknoise timestamps, classifier predict, classifier timestamps, 
## sampling rate info
# =============================================================================
# Preprocessing steps include the following:
# a. extraction of markers 
# 1. Filtering data 
#   a. EEG, EOG: both 0.5 - 35 Hz forward-backward butterworth filter + denoising source seperation for powerline removal (50 & 100 Hz)
#   b. ECG: 0.5 - 70 Hz forward-backward butterworth filter + denoising source seperation for powerline removal (50 & 100 Hz)
#   c. EMG: 10 - 100 Hz forward-backward butterworth filter + spectrum fit interpolation for powerline removal (50, 100, & 150 Hz)
# 2. Rereference data to selection of average of mastoids (preferred), common average, surface Laplacian
# 3. Save data into Data object
# =============================================================================

# Below path is the general case, if a specific dataset wants to be processed, add to path entry; 
# also can choose to save file or not, use smaller subset of data with channels of stageing interest; 
# also can select different rereferencing techniques 
path = '/media/administrator/data/Study_1_data/Raw_data/'
preprocess_sleep_data(path, save=True, reference='mastoids', stageing=False, validation=None)

## Step 1b - Pre-process data (classifier valiadation)
path = '/media/administrator/data/Study_1_data/Raw_data/Experimental/'
preprocess_sleep_data(path, save=True, stageing = True, validation='classifier')

## Step 1c - Pre-process data (auditory stimulation validation)
path = '/media/administrator/data/Study_1_data/Raw_data/Experimental/'
preprocess_sleep_data(path, low_density=True, reference=None, validation='auditory', stageing=False, save=True)

#%%
## Step 2a - Sleep stageing ## 
## Sleep stageing should typically be performed on a windows PC or Mac using the visbrain Sleep package

"""
Code available in additional script

"""

## Step 2b - Hypnogram comparison and analysis ##
## Subject should be selected along with base rater and comp rater. A report for each recording is then given which includes
## agreement accuracies, sleep stage statistics, cohen's kappa inter-rater agreement (a more robust measure than standard % error),
## as well as confusion matrices, and I may do cohen's kappa for each stage, not sure if this is necessary yet... 
subject = '6QJ3ITMT'
save = [compare_hypnograms(subj_night = subject + '_' + str(i), group=3, base_rater='Nadya', comp_rater='Sydney', save_hypno=True) for i in range(1,4)]

#%%
## Step 3 - Clean data ##
## Once data has matching hypnogram, we can label some additional artifacts. Typically during sleep stageing, periods 
## with more than 10s of noise per epoch were labelled as wakefulness to avoid having to do manually label many such epochs.
## Here we update the data structure object to include automatically labelled artifact windows (2s, typically) in the form of 
## a new property of the object in the form of hypno_with_art, which is upsampled to the comport with the data time points. 
## A covarianced based clustering algorithm is the default method and takes into consideration covariance matrices of given 
## channels and accounts for sleep stage variability. See documentation for more detail.
path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental/'
hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/'
label_artifacts(path, hypno_path)

#%%
## Step 4 - Unblinding data ##
## The following step unblinds the data and hypnograms with respect to the true conditions stored on a master excel sheet used
## during the experiment to determine conditions, targeting i. up state, ii. down state, and, iii. sham w.r.t anticipated slow 
## wave occurence. Adding specific recording file allows you to individually assess this by file.  
sheet = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
unblind_files(sheet, recording_file=None)

#%%
## Step 5a - Event detection - Slow waves + Phase Amplitude Coupling ##
## 
##
## 
path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded/'
hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental_unblinded/'
Data_SW(path, hypno_path)

## Step 5b - Event detection - Spindles (fast/slow) ##
Data_spindles(path, hypno_path)
















