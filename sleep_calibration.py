# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:33:08 2020

@author: neuro
"""

# =============================================================================
# Run whole script at once - Press F5
# =============================================================================

from os import chdir
chdir('C:/Users/neuro/Documents/projects/sleep_stimulation-development')
import liesl
import reiz.marker
import reiz
from sleep_funs import SO_detection, sleep_staging, subject_param_pull
import matplotlib.pyplot as plt
from reiz import clock
import numpy as np
import threading
from scipy import signal
import yasa
import pickle


#%% set parameters - only do for testing as excel file should contain proper info.
## load subject code and condition script
file = 'subject_codes'
subjID = input('Please enter the correct subject ID: ')
#evening: adaption for adaption, 1 for first experimental evening, 2 for second, 3 for third
evening = input(f'Please enter the correct recording session for the subject {subjID}: ')
  
#%%
# Run calibration whole block 
from liesl.files.session import Session
#reiz.marker.start()
rec_id = subjID + '_' + evening + '_calibration'
streamargs = [{"name":"eego"}, {"name":"reiz-marker"}]
mainfolder = 'C:/Users/neuro/Documents/sleep_stimulation/recordings'
session = Session(prefix = rec_id, streamargs = streamargs, mainfolder = mainfolder) 
## Quick 'hack'
session.start_recording()
# 3 minutes to record
reiz.clock.sleep(60*3)
session.stop_recording()