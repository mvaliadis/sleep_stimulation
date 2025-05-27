# -*- coding: utf-8 -*-
"""
Core I/O functions for sleepstim project.
"""

import numpy as np
import pandas as pd # unravel_hypnogram_visbrain uses it, though not explicitly imported in sleep_funs.py
import mne
import yasa
import pyxdf
import xml.etree.ElementTree as ET
import neurokit2 as nk # for nk.ecg_clean in process_raw_EDF_cfs

# Utility functions for DSP are now in core.dsp
from sleepstim.core.dsp import bfr_butter_filt, downsample_scaled

def load_xdf(path: str):
    """Loads XDF file streams into a dictionary.

    Parameters
    ----------
    path : str
        Path to the XDF file.

    Returns
    -------
    stream_dict : dict
        Dictionary where keys are stream names and values are stream data.
    """
    streams, fileheader = pyxdf.load_xdf(path)
    stream_dict = {}
    for ix, stream in enumerate(streams):
        streamname = stream['info']['name'][0]
        stream_dict[streamname] = stream
    return stream_dict

def channel_parser(info, data):
    ch_names = []
    ch_types = [] 
    bipolar_names = {'chan_1': 'EDC_L', 'chan_2': 'ECR_L', 'chan_3': 'FCR_L', 
                     'chan_4': 'FDS_L', 'chan_5': 'ECG', 'chan_6': 'EmptyChan', 
                     'chan_7': 'EDC_R', 'chan_8': 'ECR_R', 'chan_9': 'FCR_R', 
                     'chan_10': 'FDS_R'}
    for i in range(np.size(data,1)):
        chan = info['desc'][0]['channels'][0]['channel'][i]['label'][0]  
        chtype = info['desc'][0]['channels'][0]['channel'][i]['type'][0]
        if chan.startswith('chan') and int(chan.split('_')[-1]) <= 10:
            ch_names.append(bipolar_names[chan])
            if bipolar_names[chan] == 'ECG':
                ch_types.append('ecg')
            elif bipolar_names[chan] == 'EmptyChan':
                ch_types.append('misc')
            else:
                ch_types.append('emg')
        elif chan.startswith('chan') and int(chan.split('_')[-1]) >= 11:
            ch_names.append('EmptyChan')
            ch_types.append('misc')
        else:
            if chan == 'EMG_L':
                ch_names.append(chan)
                ch_types.append('emg')
            elif chan == 'EMG_R':
                ch_names.append(chan)
                ch_types.append('emg')
            elif chan == 'bipECG':
                ch_names.append(chan)
                ch_types.append('ecg')
            elif chan == 'EOG_L':
                ch_names.append(chan)
                ch_types.append('eog')
            elif chan == 'EOG_R':
                ch_names.append(chan)
                ch_types.append('eog')
            elif chan == 'EmptyChan':
                ch_names.append(chan)
                ch_types.append('misc')
            elif chan == 'EmptyChan1':
                ch_names.append(chan)
                ch_types.append('misc')
            elif chan == 'EmptyChan2':
                ch_names.append(chan)
                ch_types.append('misc')
            elif chtype == 'EEG':
                ch_names.append(chan)
                ch_types.append('eeg')

    return ch_names, ch_types

def unravel_hypnogram_visbrain(hypnogram_file, data=None):  
    ## TO-DO LATER: integate with other unravel function for NSRR dataset
    # load hypnogram file
    hypno = np.genfromtxt(hypnogram_file, delimiter='\t', dtype=str)
    # define stages 
    stages = hypno[1::,0]
    # define stage duration
    stage_len = np.round(hypno[1::,1].astype(dtype=float))
    # define length of 30s epoched stage as difference between durations
    diff = np.diff(stage_len/30)
    
    # add first epoch as difference between first epoch and 0 and to iterated list
    total_diff = [stage_len[0]/30] + list(diff)
    # parse stageing information to fit total of epochs by stages 
    stagelens = []
    for index, length in enumerate(total_diff):
        if stages[index] == 'Wake':
            stagelens.append(np.zeros(int(length)))
        elif stages[index] == 'Art':
            stagelens.append(np.zeros(int(length))) # Assuming Art should be treated as Wake (0)
        elif stages[index] == 'N1':
            stagelens.append(np.ones(int(length)))
        elif stages[index] == 'N2':
            stagelens.append(2*np.ones(int(length)))
        elif stages[index] == 'N3':
            stagelens.append(3*np.ones(int(length)))
        elif stages[index] == 'REM':
            stagelens.append(4*np.ones(int(length)))
            
    hypnogram = np.concatenate(stagelens)
   
    # sanity check - does length of hypnogram match data epoch length
    if data is not None: # data here refers to epoched data
        if abs(data.shape[0] - len(hypnogram)) < 5: # Allow a small tolerance
            # Pad if hypnogram is shorter
            if data.shape[0] > len(hypnogram):
                 padding = np.zeros(abs(data.shape[0] - len(hypnogram)))
                 hypnogram = np.concatenate([hypnogram, padding])
            # Truncate if hypnogram is longer
            elif data.shape[0] < len(hypnogram):
                hypnogram = hypnogram[:data.shape[0]]

        if data.shape[0] != len(hypnogram):
            print('WARNING - The length of the scaled hypnogram does not match the amount of total epochs in the data') #raise ValueError
     
    return hypnogram 

def process_raw_EDF_cfs(file):
    # import raw edf file
    raw_train = mne.io.read_raw_edf(file + '.edf', eog = ['LOC','ROC'], preload = 'True')
    
    # create dictionary of channels we are interested in  
    mapping = {'C3': 'eeg',
               'C4': 'eeg',
               'M1': 'eeg',
               'M2': 'eeg',
               'LOC': 'eog',
               'ROC': 'eog',
               'EMG2': 'emg',
               'EMG3': 'emg',
               'ECG1': 'ecg'}
    
    # select channels in object and give labels for channel type
    raw_train.pick_channels(ch_names=list(mapping.keys())) # Ensure only known channels are picked
    raw_train.set_channel_types(mapping) 
    
    # rereference eeg data to average of mastoids
    raw_train.set_eeg_reference(ref_channels=['M1','M2'])
     
    # bipolarize eog and emg data 
    try:
        raw_train = mne.set_bipolar_reference(raw_train, 'LOC', 'ROC', ch_name='EOG') # mne expects new channel name
        raw_train = mne.set_bipolar_reference(raw_train, 'EMG2', 'EMG3', ch_name='EMG') # mne expects new channel name
    except Exception as e: # More specific exception if possible
        # This part might be problematic if raw_train is not an MNE object after the first set_bipolar_reference
        # It's safer to work with a copy or ensure the object type is consistent
        print(f"Bipolar reference failed: {e}")
        # if not isinstance(raw_train, mne.io.Raw): # This check might be too late or incorrect
        #     ref_inst = mne.io.RawArray(raw_train.get_data(), raw_train.info) 
        #     raw_train = mne.set_bipolar_reference(ref_inst, 'LOC', 'ROC', ch_name='EOG')
        #     raw_train = mne.set_bipolar_reference(raw_train, 'EMG2', 'EMG3', ch_name='EMG')
    
    # extract data, time, sampling rate information
    EEG = raw_train.get_data(picks='eeg', return_times=False)*1e6
    EOG = raw_train.get_data(picks='eog', return_times=False)*1e6
    EMG = raw_train.get_data(picks='emg', return_times=False)*1e6
    ECG = raw_train.get_data(picks='ecg', return_times=False)*1e6
    # times = raw_train.times # Not used later
    fs = raw_train.info['sfreq']
    
    # delete mne object as it is no longer necessary
    del raw_train
    
    # filter data 
    EEG = bfr_butter_filt(EEG, fs, lfreq=0.5, hfreq=35)
    EOG = bfr_butter_filt(EOG, fs, lfreq=0.5, hfreq=35)
    EMG = bfr_butter_filt(EMG, fs, lfreq=10, hfreq=100)
    ECG = np.expand_dims(nk.ecg_clean(ECG.squeeze(), sampling_rate=fs, 
                                      method='neurokit', **dict(powerline=60)), 0)

    # combine data 
    data = np.concatenate([EEG, EOG, EMG, ECG], axis=0) # Axis should be 0 if concatenating channel-wise
    
    # epoch data into 30s segments 
    _, epochs = yasa.sliding_window(data, fs, window=30) # Expects (n_chan, n_samples)
     
    # downsample data to 128 Hz (up-sampled due to ECG being sampled at 512 Hz) 
    epoched_data = []
    # The loop below iterates over channels, but downsample_scaled expects (epochs, samples) or (samples)
    # This needs careful reshaping or applying downsample per channel per epoch
    for index in range(np.size(epochs, 1)): # This iterates over channels
        # Assuming epochs is (n_epochs, n_chans, n_samples_epoch)
        # We need to downsample each channel for each epoch
        # This part is tricky and depends on the exact shape of `epochs` from yasa.sliding_window
        # If epochs is (n_windows, n_channels, n_samples_per_window)
        # current `epochs[:,index,:]` would be (n_windows, n_samples_per_window) for a single channel
        
        # Correct approach: Downsample each epoch and each channel within that epoch
        # This requires a nested loop or careful application of downsample_scaled.
        # For simplicity, if `downsample_scaled` can handle 2D (n_epochs, n_samples), apply per channel:
        channel_data_to_downsample = epochs[:, index, :] # (n_epochs, n_samples_per_window)
        downsampled_channel_data = downsample_scaled(channel_data_to_downsample, fs, new_sf=128)
        epoched_data.append(np.expand_dims(downsampled_channel_data, axis=1))
            
    epoched_data = np.concatenate(epoched_data, axis=1) # Result should be (n_epochs, n_chans, n_new_samples)
    # The original code had `np.swapaxes(..., 2, 0)` which seems incorrect for typical epoch structures.
    # Final shape should be (n_epochs, n_channels, n_samples_per_epoch_downsampled)
  
    # import hypnogram
    stages, stagelens = read_xml(file + '-nsrr.xml')
            
    # unravel hypnogram
    hypnogram = unravel_hypnogram(stages, stagelens)

    # Ensure hypnogram length matches number of epochs
    if len(hypnogram) > epoched_data.shape[0]:
        hypnogram = hypnogram[:epoched_data.shape[0]]
    elif len(hypnogram) < epoched_data.shape[0]:
        padding = np.full(epoched_data.shape[0] - len(hypnogram), -1) # Pad with unscored
        hypnogram = np.concatenate([hypnogram, padding])
  
    return epoched_data, hypnogram

def read_xml(file):
    # import xml annotation file to extract hypnogram
    tree = ET.parse(file)
    
    # obtain roots from xml annotation tree
    root = tree.getroot()
    
    # extract sleep stageing related information
    stages = [] 
    stagelens = []
    for child in root.iter('ScoredEvent'):
        var = child[0].text
        if var == None:
            pass
        elif 'Stages' in var: # Ensures it's a sleep stage event
            stage_text = child[1].text
            # Extract the numeric stage value, assuming format "Stage X" or similar
            try:
                stage_val = int(stage_text.split(' ')[-1]) # Takes the last part after space
                stages.append(stage_val)
                stagelen = int(float(child[3].text))
                # append epoch lengths in increments of 30s
                stagelens.append(int(stagelen/30))
            except ValueError:
                print(f"Warning: Could not parse stage from '{stage_text}'")
                pass # Or handle error appropriately
        else:
            pass
        
    # return numpy arrays with stages and corresponding length info
    return np.array(stages).astype(int), np.array(stagelens).astype(int)

def unravel_hypnogram(stages, stagelens):  
    stage_len = []
    # parse stageing information to fit total of epochs by stages 
    for index, length in enumerate(stagelens):
        current_stage_val = stages[index]
        # Mapping: 0->Wake, 1->N1, 2->N2, 3->N3, 4->N3 (collapse S4 to S3), 5->REM
        # YASA/MNE mapping: 0->Wake, 1->N1, 2->N2, 3->N3, 4->REM
        if current_stage_val == 0: # Wake
            stage_code = 0
        elif current_stage_val == 1: # N1
            stage_code = 1
        elif current_stage_val == 2: # N2
            stage_code = 2
        elif current_stage_val == 3 or current_stage_val == 4: # N3/S4
            stage_code = 3
        elif current_stage_val == 5: # REM
            stage_code = 4
        else: # Handle unscored or artifact if necessary, or raise error
            # For now, let's assume unscored/artifact is mapped to Wake or a specific code like -1
            # print(f"Warning: Unrecognized stage value {current_stage_val}. Mapping to Wake.")
            stage_code = 0 # Or -1 if you have a code for unscored/artifact
        
        stage_len.append(np.full(length, stage_code))
            
    hypnogram = np.concatenate(stage_len)
    
    # sanity check - does hypnogram 
    if len(hypnogram) != stagelens.sum():
        raise ValueError('The length of the scaled hypnogram does not match the amount of total epochs')
    
    return hypnogram 

def process_raw_EDF(fname):
    """
    The following function exlusively preprocesses physionet datasets and 
    extracts the epoched data along with the corresponding sleep stages for 
    and eventual classification
    
        Parameters
    ----------
    fname : EDF data file(s) 
        fname is root and will append both PSG.edf & Hypnogram.edf

    Returns
    -------
    datArray : np.array
        Epoched data array
    stageArray : np.array
        Epoched stage array
    
    """
    raw_train = mne.io.read_raw_edf(fname + '-PSG.edf', preload=True)
    
    # load in hypnogram
    annot_train = mne.read_annotations(fname + '-Hypnogram.edf')
        
    # filter data and select channels to use
    # Ensure channel names are correct and exist in the raw object
    ch_to_pick = []
    if 'EEG Fpz-Cz' in raw_train.ch_names: ch_to_pick.append('EEG Fpz-Cz')
    if 'EEG Pz-Oz' in raw_train.ch_names: ch_to_pick.append('EEG Pz-Oz')
    if 'EOG horizontal' in raw_train.ch_names: ch_to_pick.append('EOG horizontal')
    if 'EMG submental' in raw_train.ch_names: ch_to_pick.append('EMG submental')
    
    raw_train.pick_channels(ch_to_pick)
    
    if 'EEG Fpz-Cz' in raw_train.ch_names and 'EEG Pz-Oz' in raw_train.ch_names and 'EOG horizontal' in raw_train.ch_names:
         raw_train.filter(0.5, 35, picks = ['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal'])
    elif 'EEG Fpz-Cz' in raw_train.ch_names and 'EOG horizontal' in raw_train.ch_names: # Pz-Oz missing
         raw_train.filter(0.5, 35, picks = ['EEG Fpz-Cz','EOG horizontal'])
    elif 'EEG Pz-Oz' in raw_train.ch_names and 'EOG horizontal' in raw_train.ch_names: # Fpz-Cz missing
         raw_train.filter(0.5, 35, picks = ['EEG Pz-Oz','EOG horizontal'])
    # Add more conditions if necessary for other missing channels

    raw_train.notch_filter(49) # Assuming 50Hz line noise, notch filter around it.
    if 'EMG submental' in raw_train.ch_names:
        raw_train.filter(20, 45, picks = (['EMG submental']))
    
    # select channels for annotations
    mapping = {}
    if 'EEG Fpz-Cz' in raw_train.ch_names: mapping['EEG Fpz-Cz'] = 'eeg'
    if 'EEG Pz-Oz' in raw_train.ch_names: mapping['EEG Pz-Oz'] = 'eeg'
    if 'EOG horizontal' in raw_train.ch_names: mapping['EOG horizontal'] = 'eog'
    if 'EMG submental' in raw_train.ch_names: mapping['EMG submental'] = 'emg'
     
    raw_train.set_annotations(annot_train, emit_warning=False)
    if mapping: # Only set channel types if mapping is not empty
        raw_train.set_channel_types(mapping)           
       
    # extract 30s events from annotations
    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4, # Collapsing S4 into S3
                                  'Sleep stage R': 5}
        
    events_train, _ = mne.events_from_annotations(
    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)
        
    # create a new event_id that unifies stages 3 and 4
    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3/4': 4,
                'Sleep stage R': 5}

    # create Epochs from the data based on the events found in the annotations
    tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included
        
    epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                              event_id=event_id, tmin=0., tmax=tmax, baseline=None)
            
    datArray = epochs_train.get_data()*1e6
    stageArray = epochs_train.events[:,2] # event IDs are in the 3rd column
    # Adjust stageArray to map original event IDs to 0-4 range if necessary
    # This depends on how event_id was defined for mne.Epochs
    # If event_id used 1-5, then map to 0-4:
    stageArray = stageArray - 1 
    
    return datArray, stageArray
