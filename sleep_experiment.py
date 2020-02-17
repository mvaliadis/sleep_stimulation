# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:36:25 2020

@author: neuro
"""
from os import chdir
chdir('C:/Users/neuro/Documents')
import liesl
from sleep_funs import PinkNoise
#from reiz import clock
#%% set parameters
thresh = 0.9 #threshold to detect SO, should be set to -35 later
winshift_in_ms = 20

#%% set up EEG stream
sinfo = liesl.get_streaminfo_matching(type = 'EEG')
bfr = liesl.RingBuffer(sinfo, duration_in_ms = 2000, fs = 100)
bfr.start()



#TODO process EEG (filter? common avg. ref?)
# b,a = signal.butter(4, (.5, 1.5), btype = 'bandpass', fs = bfr.fs)
# t = signal.filtfilt(b,a,bfr.get_data(), axis = 0)

#TODO set up sleep stage classifier, gate SO detection
#TODO send marker
#TODO set up SO detection
t = bfr.get_data()[:,0] #pick one of the mock channels

#TODO send marker
#set up a pink noise burst generator
n = PinkNoise()
#play sound burst
n.play()


