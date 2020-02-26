# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:36:25 2020

@author: neuro
"""
from os import chdir
chdir('C:/Users/neuro/Documents/sleep_stimulation')
import liesl
import reiz.marker
from sleep_funs import PinkNoise
import matplotlib.pyplot as plt
from reiz import clock
import numpy as np
#from reiz import clock
#%% set parameters

thresh = 0.9 #threshold to detect SO, should be set to -35 later
winshift_in_ms = 20

#%% set up EEG and marker stream
if not reiz.marker.available():
    reiz.marker.start()
sinfo = liesl.get_streaminfo_matching(type = 'EEG')
bfr = liesl.RingBuffer(sinfo, duration_in_ms = 2000)
bfr.start()
bfr.await_running()
n = PinkNoise()

winshift_in_ms = 20
winshift_in_samples = int((winshift_in_ms/1000)*sinfo.nominal_srate())
maxamp = 1
block_auditory_stim = False
tblock = 0
for interval in range(10000):
    if clock.now() -tblock > 1.9:
        block_auditory_stim = False
    d = bfr.get_data()[:,1] #test channel is 1Hz sinusoid
    if max(d[-2:]) > maxamp * .9 and block_auditory_stim == False:
        n.play()
        block_auditory_stim = True
        tblock = clock.now()

    clock.sleep(winshift_in_ms/1000)
#TODO process EEG (filter? common avg. ref?)
# b,a = signal.butter(4, (.5, 1.5), btype = 'bandpass', fs = bfr.fs)
# t = signal.filtfilt(b,a,bfr.get_data(), axis = 0)

#TODO set up sleep stage classifier, gate SO detection
#TODO send marker
#TODO set up SO detection

#TODO block audio presentation for 2 s 
#set up a pink noise burst generator
#play sound burst
#tm = np.zeros(100)
#for tr in range(100):
#    n.play()
#    clock.sleep(.5)
