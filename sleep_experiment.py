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


#%% load classifier models
rf = pickle.load(open("rf_model_2_cfs.p", "rb"))            #1 EEG, 1 EOG, 1 EMG
rf_2EEG = pickle.load(open("rf_model_3_cfs.p", "rb"))       #2 EEG
rf_1EEG = pickle.load(open("rf_model_4_cfs.p", "rb"))       #1 EEG
rf_EEG_EOG = pickle.load(open("rf_model_5_cfs.p", "rb"))    #1 EEG, 1 EOG

#%% set parameters - only do for testing as excel file should contain proper info.
time_delay = float(input('Please select the participants average peak to peak SO amplitude from the
                   'adaption evening. '))
volume = int(input('Please select the volume for the experiment, either 0 or 1. '))

#%% load subject code and condition script
### Create function to make info directly inaccessible to the experimenter for blinding, 
### instead of more cumbersome decryption, encryption method.
# subj_cond = np.loadtxt('subject_codes.csv',delimiter=',', dtype='str', skiprows=1)
### determine stimulation condition based on recording evening and subject code 
# cond = int(input('please enter experimental recording evening: ')) 
# index_pos = []
# # obtain index position of subject from excel file
# while index_pos == []:
#     # to-do later: might be a more elegant way to obtain this
#     subj = input('Please enter the subject token: ')
#     index_pos = [i for i,item in enumerate(subj_cond) if subj in item]
#     if index_pos == []:
#         print('Subject code is invalid, please enter a valid subject code!')
#     else:    
#         ## time delay + volume based on condition
#         # 0 corresponds to sham (same trigger as up, without volume)
#         if int(subj_cond[index_pos[0],:][cond]) == 0:
#             time_delay = .500
#             volume = 0
#         # 1 corresponds to up-state targeting 
#         elif int(subj_cond[index_pos[0],:][cond]) == 1:
#             time_delay = .500
#             volume = 1
#         # 2 corresponds to down-state targeting     
#         elif int(subj_cond[index_pos[0],:][cond]) == 2:
#             time_delay = 0
#             volume = 1 

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


     #%% initialize second thread for SW detection

     SO_detection()
     

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
