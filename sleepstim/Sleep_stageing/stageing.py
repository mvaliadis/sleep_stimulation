#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:29:43 2021

@author: administrator
"""

from visbrain.gui import Sleep
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, load_preprocessed_data)
import yasa
import numpy as np
import matplotlib.pyplot as plt
import os

Data = load_preprocessed_data('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental/D1BOI2AY_1_preproc_data.p')[0]

#%% Initialize gui here
Sleep(data = Data.data.T, channels = Data.chans, sf = Data.sfreq).show()
