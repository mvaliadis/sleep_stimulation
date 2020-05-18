# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:36:25 2020

@author: neuro
"""
from os import chdir
chdir('C:/Users/neuro/Documents/sleep_stimulation')
import liesl
import reiz.marker
from sleep_funs import SO_detection, sleep_staging
import matplotlib.pyplot as plt
from reiz import clock
import numpy as np
import threading
from scipy import signal
import yasa
import pickle


#%% set parameters - only do for testing as excel file should contain proper info.
time_delay = .5
volume = 1

#%% set parameters = load subject code and condition script
#subjID = input('Please enter the correct subject ID: ')
#evening = int(input(f'Please enter the correct recording session for the subject {subjID}: '))
#time_delay, volume = subject_param_pull(file='subject_codes', subjID, evening)

#%% set up EEG and marker stream for testing mock data
#-----------------------------
#----Tip: for development and testing, you can just set up a fake EEG stream by
#----entering the following into your command line:
#----liesl mock --type EEG
#----Alternatively, a replay stream can be set-up to test already recorded data,
#----to determine accuracy of detection, by entering the following:
#----python -m replay --file <path_to_xdf_file>

    
#%% put sleep stager into separate thread that can run in the background
def main():

     
     # get sample info and pull data ringbuffer
     sinfo = liesl.get_streaminfos_matching(type = 'EEG')
     bfr = liesl.RingBuffer(sinfo[0], duration_in_ms = 30000) #30 s buffer to allow sleep staging
     bfr.start()

     bfr.await_running()
     sleep_stager = threading.Thread(target = sleep_staging, args = (bfr,))

     # start thread
     sleep_stager.start()


     ## initialize second thread for SW detection
     SO_detection(time_delay, volume)
    
     # sleep for the remainder of the evening or until the participant awakens
     print('Sleep time! The recording will continue, although the stimulation paradigm has officially ended. ')
     reiz.clock.sleep(30000) 
     

if __name__ == '__main__':
    from liesl.files.session import Session
    subj_id = input('Enter Subject ID... ')
    streamargs = [{"name":"eeg_replay"}, {"name":"reiz-marker"}]
    mainfolder = 'C:/Users/neuro/Documents/sleep_stimulation/recordings'
    session = Session(prefix = subj_id, streamargs = streamargs, mainfolder = mainfolder)
     
    with session('sleepstim'):
        main()

       
## to-do: test timing of integrated functions in simulation 
## to-do: functionalize subject specific parameters
## to-do: increase sensitivity of classifier
## to-do: pilot testing
## to-do: finalize docstrings 
