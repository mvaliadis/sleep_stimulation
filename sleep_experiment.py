# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:36:25 2020

@author: neuro
"""

from os import chdir
chdir('C:/Users/neuro/Documents/projects/sleep_stimulation-development')
import liesl
import reiz.marker
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
     
     infostruct = bfr.info['desc']['channels']['channel']
     chanlabels = [infostruct[i]['label'] for i in range(len(infostruct))]
     chanselection = ['Cz', 'C3', 'C4', 'M1', 'M2', 'EOG_L', 'EOG_R', 'EMG_L', 'EMG_R']
     indices_to_pull = np.array([ix for ix,val in enumerate(chanlabels) if val in chanselection])
     
     # initialize second thread for SW detection
     if evening!='adaption':
        #adaptation info from sheet 
        mean_pk2pk = float(input('Please select the participants average peak to peak SO amplitude from the adaption evening (Please enter in seconds). '))
        # enter runtime for experiment - should place 210 in the beginning
        totalruntime = int(input('Please enter the remaining amount of time in the sleep stimulation in minutes (enter 210 at the beginning): '))*60 
        # sleep for 10s
        reiz.clock.sleep(10)
	# create time delay and volume based on subject condition list, also will not allow thread to run if wrong!
        time_delay, volume = subject_param_pull(file, subjID, int(evening), mean_pk2pk)
        # set up sleep stageing thread 
        sleep_stager = threading.Thread(target = sleep_staging, args = (bfr, indices_to_pull))
        # start thread
        sleep_stager.start()
        # initiate SO detection
        SO_detection(time_delay, volume, totalruntime = totalruntime)
        # sleep for the remainder of the evening or until participant awakens
        print('Sleep time! The recording will continue, although the stimulation paradigm has officially ended!')
        reiz.clock.sleep(18000) 
     else:
        # sleep for 10s
        reiz.clock.sleep(10)
        # set up sleep stageing thread 
        sleep_stager = threading.Thread(target = sleep_staging, args = (bfr, indices_to_pull))
        # start thread
        sleep_stager.start()
        # initiate SO detection - change to delay of 0 alternating adaption nights 
        SO_detection(time_delay=.500, volume=0, totalruntime = 28800)
        # sleep until participant awakens
        print('The recording will continue until the participant is awoken.')
        reiz.clock.sleep(30000)  


    


if __name__ == '__main__':
    from liesl.files.session import Session
    reiz.marker.start()
    rec_id = subjID + '_' + evening
    streamargs = [{"name":"eego"}, {"name":"reiz-marker"}]
    mainfolder = 'C:/Users/neuro/Documents/sleep_stimulation/recordings'
    session = Session(prefix = rec_id, streamargs = streamargs, mainfolder = mainfolder)
     
    with session('sleepstim'):
        main()
      
## to-do: test timing of integrated functions in simulation 
## to-do: increase sensitivity of classifier
## to-do: finalize docstrings
