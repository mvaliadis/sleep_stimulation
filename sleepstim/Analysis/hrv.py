#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:13:29 2021

@author: administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import ECGtoolbox as ecg_nti
import EntropyHub as EH
import pyhrv
import biosignalsnotebooks as bsnb
import heartpy as hpy
import neurokit2 as nkit
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, load_preprocessed_data)

Data = load_preprocessed_data('/media/administrator/data/Study_1_data/Pre-processed_data/'
                              'Adaption/6QJ3ITMT_adaption_preproc_data.p')[0]

#%%
# load data
ecg_data = Data.data[:, Data.chans.index('bipECG')]

# preprocess and compute time/freq/non-linear domain analyses with lab toolbox
ECG_nti = ecg_nti.ecg_analyzer(ecg_data, fs=Data.sfreq)

