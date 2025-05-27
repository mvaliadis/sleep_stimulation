# -*- coding: utf-8 -*-
"""Core functions for real-time sleep stage dependent auditory stimulation and parameter adjustments for the sleepstim project."""

import numpy as np
import pandas as pd
import threading
import pyaudio # Required for PinkNoise class
from scipy import signal

# LSL and timing related imports
import liesl
import reiz # reiz.marker and reiz.clock

# Imports from other modules in this project
from sleepstim.core.classification import stage_predictArrays, channel_failure
from sleepstim.core.utils import get_coords # Updated import for get_coords

# Global variable coords is no longer imported, it will be fetched by get_coords()
# coords = None # This can be removed if get_coords() is called within SO_detection when needed

def subject_param_pull(file, subjID, evening, mean_pk2pk):
    """ This function pulls the participant parameter information from excel file to
        prevent experimenter from directly observing condition relevant information 
        for blinding purposes.

    Parameters
    ----------
    file : str
        Input file name with subject information minus .csv ending. 
        This will be the base for '../../subject_codes.csv'.
    subjID : str
        Subject ID #.
    evening : int, either: {0, 1, 2}
        Select the experimental recording evening for the participant. 
    mean_pk2pk : float
        Mean peak-to-peak SO amplitude from adaptation night, used for time_delay.

    Returns
    -------
    time_delay : float
        Returns requisite time delay based on recording evening.
    volume : int
        Returns volume based on recording evening condition.
    """
    # Path adjusted to be relative to the project root, assuming stimulation.py is in sleepstim/core
    # The 'file' parameter is a bit confusing here if it's meant to be part of the path.
    # Assuming 'file' argument was meant to be the base name and we construct the path.
    # For robustness, using a fixed relative path from this file to the project root.
    # If subject_codes.csv is at project root: ../../subject_codes.csv
    # If it's in 'sleepstim' directory: ../subject_codes.csv
    # Assuming it's at the project root for now.
    csv_path = "../../subject_codes.csv" 
    try:
        subj_cond = np.loadtxt(csv_path, delimiter=',', dtype='str', skiprows=1) 
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Ensure the path is correct.")
        return None, None # Or raise an error

    index_pos = []
    try:
        index_pos = [i for i,item in enumerate(subj_cond) if subjID in item]
        if not index_pos: # Check if list is empty
            raise NameError(f'Subject code {subjID} is invalid or not found in {csv_path}.')
        
        condition_val = int(subj_cond[index_pos[0],:][evening]) # Convert to int after indexing
        
        if condition_val == 0: # Sham
            time_delay = mean_pk2pk
            volume = 0
        elif condition_val == 1: # Up-state targeting 
            time_delay = mean_pk2pk
            volume = 1
        elif condition_val == 2: # Down-state targeting     
            time_delay = 0
            volume = 1 
        else:
            raise ValueError(f"Invalid condition value {condition_val} found in {csv_path}.")
                
    except NameError as e:
        print(e)
        return None, None
    except IndexError:
        print(f"Error: Evening index {evening} might be out of bounds for subject {subjID} in {csv_path}.")
        return None, None
    except ValueError as e:
        print(f"Error processing data for subject {subjID}, evening {evening}: {e}")
        return None, None
            
    return time_delay, volume

class PinkNoise():
    def generate_noise(self,duration_in_s = 0.05, fs = 48000, ncols=16):
        """Generates pink noise using the Voss-McCartney algorithm.
        
        nrows: number of values to generate
        rcols: number of random sources to add
        
        returns: NumPy array
        """
        nrows = int(fs * duration_in_s)
        array = np.empty((nrows, ncols))
        array.fill(np.nan)
        array[0, :] = np.random.random(ncols)
        array[:, 0] = np.random.random(nrows)
        
        n = nrows
        cols = np.random.geometric(0.5, n)
        cols[cols >= ncols] = 0
        rows = np.random.randint(nrows, size=n)
        array[rows, cols] = np.random.random(n)
        
        flank_samples = int(0.005*fs)
        rising = np.linspace(0,1,flank_samples)
        falling = np.linspace(1,0,flank_samples)
        plateau = np.ones(nrows-2*flank_samples)
        window = np.concatenate((rising,plateau,falling))
        df = pd.DataFrame(array)
        df.fillna(method='ffill', axis=0, inplace=True)
        total = df.sum(axis=1)
        noise = total.values
        noise /= max(noise) # Normalize
        return noise*window  
    
    def open_stream(self):  
        fs = 48000 # Standard audio sampling rate
        self.stream = self.p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=fs,
                        output=True)
      
    def play(self):
        if self.reizmarker:
            if reiz.marker.available(): # Check if marker server is actually available
                 reiz.marker.push('pinknoise')
            else:
                 print("Reiz marker server not available, cannot push 'pinknoise' marker.")
        self.stream.write(self.samples.astype(np.float32).tobytes()) # Ensure data is float32 and bytes
        self.stream.stop_stream() # Should be careful with frequent stop/start
        self.stream.start_stream() # This can introduce latency/gaps
        
    def __init__(self, volume):
        #threading.Thread.__init__(self) # PinkNoise itself is not a thread here.
        self.samples = self.generate_noise()*volume
        self.p = pyaudio.PyAudio()
        self.open_stream()
        self.reizmarker = True # By default, try to use reiz markers
        if not reiz.marker.available():
            print('Reiz Marker Server not available!')
            self.reizmarker = False

def SO_detection(time_delay, volume, nepochsthresh = 4, minamp = -35, 
                 winshift_in_ms = 20, totalruntime = 12600):
    """Detects slow oscillations (SO) in real-time EEG data and triggers auditory stimulation.

    This function continuously monitors an LSL EEG stream, performs sleep staging based
    on global variables, and if SWS is detected for a specified number of epochs,
    it then attempts to detect slow oscillations on a target channel (typically C3).
    If an SO meeting amplitude criteria is found, and no stimulation is currently blocked
    (refractory period), it triggers auditory stimulation (e.g., pink noise)
    timed according to the `time_delay` parameter relative to the SO detection.

    Parameters
    ----------
    time_delay : float
        The delay (in seconds) after SO detection to trigger the auditory stimulus.
        Different values are used for up-state or down-state targeting.
    volume : float
        The volume level for the auditory stimulus (0 for sham, 1 for real stimulation).
    nepochsthresh : int, optional
        Number of consecutive SWS epochs required before SO detection is active. Default is 4.
    minamp : float, optional
        Minimum negative amplitude (microvolts) for SO detection relative to a baseline. Default is -35.
    winshift_in_ms : int, optional
        The interval (in milliseconds) at which the function attempts to pull new data
        and perform detection. Default is 20.
    totalruntime : int, optional
        Total duration (in seconds) for which the SO detection and stimulation loop will run.
        Default is 12600 (3.5 hours).

    Global Variables
    ----------------
    stage_predictArrays : list
        Expected to be populated by `sleep_staging` function, containing binary indicators
        of SWS (1) or other stages (0).
    channel_failure : list
        Expected to be populated by `channel_failure_test`, indicating the status of EEG channels.
        `channel_failure[0]` is used to check the status of the primary SO detection channel.
    coords : array_like
        Expected to be set by `get_coords`, containing electrode coordinates. (Currently not directly used within SO_detection itself but often part of the broader context).
    """
    # These globals are now imported
    # global stage_predictArrays 
    # global channel_failure
    # coords will be fetched by get_coords() inside this function if needed by uncommented code
    # coords will be fetched by get_coords() inside this function if needed by uncommented code

    # Initialize coords by calling get_coords()
    # This is placed here in case future uncommented code needs it.
    # The second return value (valid_channel_list) is ignored for now.
    coords, _ = get_coords()

    sinfo = liesl.get_streaminfos_matching(type = 'EEG')
    if not sinfo:
        print("Error: No EEG stream found. Ensure LSL stream is running.")
        return
    sinfo = sinfo[0] # Take the first found stream

    bfr2 = liesl.RingBuffer(sinfo, duration_in_ms = 30000) 
    bfr2.start()
    bfr2.await_running()
    
    infostruct = bfr2.info['desc']['channels']['channel']
    chanlabels = [infostruct[i]['label'] for i in range(len(infostruct))]
    
    # Ensure 'C3' and mastoids are in chanlabels before trying to get their index
    required_chans = ['C3', 'M1', 'M2']
    if not all(ch in chanlabels for ch in required_chans):
        print(f"Error: Not all required channels ({required_chans}) found in stream.")
        bfr2.stop()
        return
        
    chanselection = ['C3', 'M1', 'M2'] # Example, ensure these are valid
    indices_to_pull = np.array([ix for ix,val in enumerate(chanlabels) if val in chanselection])
    
    # Filter design (assuming bfr2.fs is available and correct)
    filtparams_lp = signal.butter(4, 4, fs = bfr2.fs, output='sos') # Use SOS for stability
    
    noise_player = PinkNoise(volume) # Corrected variable name

    block_auditory_stim = False
    tblock = 0
    tz = reiz.clock.now()
    
    while reiz.clock.now() - tz < totalruntime:
        reiz.clock.tick() # Recommended for reiz loops                   
        if len(stage_predictArrays) >= nepochsthresh and \
           sum(stage_predictArrays[-nepochsthresh:]) >= nepochsthresh:
            if reiz.clock.now() - tblock > 2.99: # Refractory period
                block_auditory_stim = False
            
            if channel_failure[0] == 1: # Assuming channel_failure[0] corresponds to C3 or relevant channel
                current_data = bfr2.get_data()[:,indices_to_pull]*1e6 # Ensure indices_to_pull is correct
                                                               
                filtered_data = signal.sosfiltfilt(filtparams_lp, current_data, axis=0)
                if np.isnan(filtered_data).any(): # Check after filtering
                    print('Warning: Filtering resulted in NaNs!')
                    reiz.clock.sleep_debiased(winshift_in_ms/1000) # Safely skip this iteration
                    continue
                
                # Re-reference to linked mastoids (assuming indices for M1, M2 are correct for `d`)
                # d is (samples, 3) where 3 are C3, M1, M2 based on chanselection & indices_to_pull
                ref_data = filtered_data[:, [1,2]].mean(axis=1, keepdims=True) # M1, M2 are index 1 and 2
                C3_data = filtered_data[:,0] - ref_data.squeeze() # C3 is index 0
                
                current_median_C3 = np.median(C3_data) # Median of current C3 data for centering
                minamp_calc = min(np.percentile((C3_data[-2* int(bfr2.fs):]) - current_median_C3, 10), minamp)
                
                if minamp_calc < -300 and np.ptp(C3_data[-2*int(bfr2.fs):]) > 500: # Increased PTP threshold
                    minamp_calc = minamp # Reset to default minamp
                    reiz.clock.sleep(10) # Wait longer if drift detected
                    block_auditory_stim = True 
                 
                crit = min(C3_data[int(-0.02*bfr2.fs):]) - current_median_C3
                
                if crit < minamp_calc and not block_auditory_stim: 
                    reiz.clock.sleep(time_delay - 0.025) # time_delay should be > 0.025
                    noise_player.play() # Corrected variable name
                    reiz.clock.sleep(1.075 - 0.025)
                    noise_player.play()
                    block_auditory_stim = True 
                    tblock = reiz.clock.now()
            else: # Channel failure case
                pass # Do nothing or log
                
        reiz.clock.sleep_debiased(winshift_in_ms/1000)

    bfr2.stop() # Stop the buffer after the loop
    if hasattr(noise_player, 'p'): # Clean up pyaudio
        noise_player.stream.close()
        noise_player.p.terminate()
