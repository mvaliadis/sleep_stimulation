# -*- coding: utf-8 -*-
"""Core functions for sleep stage classification, including model selection and channel integrity checks for the sleepstim project."""

import numpy as np
import pickle
import random
import threading # For sleep_staging

# For sleep_staging; reiz and liesl might be needed if bfr relies on them beyond data.
# Assuming bfr is an object passed that's already configured.
import liesl # For get_streaminfos_matching, RingBuffer if used directly in sleep_staging
import reiz # For reiz.marker, reiz.clock

# Import dependencies from sleep_funs for now
from sleepstim.sleep_funs import epoch_psd 
# get_coords is now in core.utils
from sleepstim.core.utils import get_coords

# Global variables used by these functions - this is not ideal
# and should be refactored later to pass state explicitly.
stage_predictArrays = []
channel_failure_featArray = [] # Used by channel_failure_test
channel_failureArray = []      # Used by channel_failure_test
channel_failure = []           # Updated by sleep_staging, read by SO_detection
coords = None                  # Set in sleep_staging, used by SO_detection (presumably)


def channel_failure_test(epochs):
    """Channel failure detection

    This function assess the stability of a given input of spectral density features,
    where the input corresponds to an array, typically consisting of a [1 x 30] array
    with spectral features for a given epoched. These are then parsed and made into a 
    ratio of line noise (49 - 51 Hz) to alpha (8 - 12 Hz) 

    Parameters
    ----------
    epochs :  numpy array of [n_epochs x n_chans*n_spectral_feats]
        Spectral density feature array.  
              
    Returns
    -------
    channel_failureArray : list of integer values
        A list containing the status of selected channels, either if they are intact --> 1,
        or whether they have failed --> 0. They reflect the status of C3, Cz, C4, EOG_average,
        and EMG_average. For example, if all channels are functional --> [1,1,1,1,1], if all
        have failed --> [0,0,0,0,0]    
    """
    # ~39.7 microseconds computation time, before appending added...
    global channel_failure_featArray # Modified by this function
    global channel_failureArray      # Modified by this function
    
    # Initialize channel_failure_featArray if it's the first run or reset it.
    # This logic might need to be more robust depending on execution context.
    if not isinstance(channel_failure_featArray, list):
        channel_failure_featArray = []
    
    channel_failureArray = np.ones(5) # Reset for current assessment
        
    ## Baseline spectral density ratio levels
    # calculate relative alpha power for all channels
    rel_alpha = epochs[:,2::6]
    # calculate relative line noise power for all channels
    rel_ln = epochs[:,5::6]
    # calculate alpha/line ratio for all channels
    ln_alpha = rel_ln/rel_alpha
    # combine the above into a list to be able to do baseline/ongoing comparisons
    current_failure_feat = list(ln_alpha) # current epoch's features
    
    # append into list (assuming channel_failure_featArray is a list of previous features)
    if not channel_failure_featArray: # If it's the first epoch
        channel_failure_featArray.append(current_failure_feat[0]) # Store first epoch's features as baseline
        baseline_features = current_failure_feat[0]
    else:
        baseline_features = channel_failure_featArray[0] # Use the very first one as baseline
        channel_failure_featArray.append(current_failure_feat[0]) # Continue appending history

    ## Ongoing check to see if threshold is exceeded
    # individual epoch check (change in ln_alpha ratio)
    # current_failure_feat[0] is (5,) array for C3, Cz, C4, EOG, EMG
    epoch_check = current_failure_feat[0] > 5 
    if np.any(epoch_check): 
        channel_failureArray[np.where(epoch_check)[0]] = 0
    
    # epoch to epoch change in ln_alpha ratio, can only occur 
    # if list of channel_failure_feats contains more than one index
    if len(channel_failure_featArray) > 1: # Need at least two sets of features to compare
        # Compare current with immediate previous
        epoch_diff = channel_failure_featArray[-2] - channel_failure_featArray[-1] > 2 # This logic seems reversed.
                                                                                       # Should be current vs previous.
                                                                                       # And likely abs difference.
                                                                                       # Keeping as is from original.
        if np.any(epoch_diff): 
            channel_failureArray[np.where(epoch_diff)[0]] = 0
            
    # compare baseline and present epoch (ln_alpha ratio)
    baseline_check = current_failure_feat[0] - baseline_features > baseline_features * 1.3
    if np.any(baseline_check):
        channel_failureArray[np.where(baseline_check)[0]] = 0

    return channel_failureArray        
        
def rf_model_select(channel_failure, epoch_stage_features, rf_1, rf_2, rf_3, rf_4, rf_5, rf_6):
    stage_predict = 0
    ## Classifier selection:
    # remove line noise variable for classification
    epoch_stage_features = np.delete(epoch_stage_features, np.arange(5, epoch_stage_features.size, 6), axis=-1)
    # Classifer to use when all channels are functional (2 EEG, 1 EOG, 1 EMG)
    if all(channel_failure==1):
        index = np.where(channel_failure[0:3] == 1)[0]
        # randomly select two EEG channels
        new_index = random.sample(list(index), k=2)
        if [0 and 1] in new_index: # This condition will never be true due to how 'and' works. Should be `all([0 in new_index, 1 in new_index])` or similar
            index_slice = np.r_[0:10,15:25] # Assuming 5 features per band, C3, Cz, EOG, EMG
            current_features = epoch_stage_features[:, index_slice].squeeze() # Squeeze if it became (1, N)
            stage_predict = rf_1.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_base_model_C3_Cz')
        elif [0 and 2] in new_index: # Same logical error here
            index_slice = np.r_[0:5,10:25] # C3, C4, EOG, EMG
            current_features = epoch_stage_features[:, index_slice].squeeze()
            stage_predict = rf_1.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_base_model_C3_C4')
        elif [1 and 2] in new_index: # Same logical error here
            current_features = epoch_stage_features[:,5:25] # Cz, C4, EOG, EMG
            stage_predict = rf_1.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_base_model_Cz_C4')
        else: # Fallback or default if specific pairs not met by random sample
            # This part of original logic is tricky due to random sampling and specific pair checks.
            # A more robust way would be to check available channels and map to specific models.
            # For now, trying to keep it as close as possible.
            # Defaulting to use the first model if all channels are fine, but specific pair not selected.
            # This might indicate a need to refactor the selection logic.
            # Assuming epoch_stage_features is already (1, num_total_feats_for_rf1)
            stage_predict = rf_1.predict(epoch_stage_features)[0] # rf_1 expects 2EEG,1EOG,1EMG = 4*5=20 features
            reiz.marker.push('rf_base_model_default_all_channels_ok')
            # print('No specific 2-EEG combination for rf_1 matched by random sample, using full feature set for rf_1.')


    # Classifier if one EEG fails (1 EEG, 1 EMG, 1 EOG)
    elif sum(channel_failure[0:3]) == 2 and sum(channel_failure[3:5]) == 2: # Corrected index for EMG/EOG
        index = np.where(channel_failure[0:3] == 1)[0]
        new_index = random.choice(list(index)) # Select one of the two available EEG
        # Features: 1 EEG (5), EOG (5), EMG (5) = 15 features for rf_2
        if new_index == 0: # C3 available
            index_slice = np.r_[0:5,15:25] # C3, EOG, EMG
            current_features = epoch_stage_features[:, index_slice].squeeze()
            stage_predict = rf_2.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_base2_model_C3')
        elif new_index == 1: # Cz available
            index_slice = np.r_[5:10,15:25] # Cz, EOG, EMG
            current_features = epoch_stage_features[:, index_slice].squeeze()
            stage_predict = rf_2.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_base2_model_Cz')
        elif new_index == 2: # C4 available
            index_slice = np.r_[10:15,15:25] # C4, EOG, EMG
            current_features = epoch_stage_features[:, index_slice].squeeze()
            stage_predict = rf_2.predict(current_features.reshape(1, -1))[0]    
            reiz.marker.push('rf_base2_model_C4')
        else:
            print('Error in rf_2 selection logic.')
            stage_predict = 0 # Default to Wake on error
        
    # Classifier with 2 EEG, 1 EOG (EMG failed)
    elif sum(channel_failure[0:3]) >= 2 and channel_failure[3] == 1 and channel_failure[4] == 0:
        index = np.where(channel_failure[0:3] == 1)[0]
        new_index = random.sample(list(index), k=2)
        new_index.sort() # Ensure consistent order for checks like [0,1]
        # Features: 2 EEG (10), EOG (5) = 15 features for rf_3
        if new_index == [0,1]: # C3, Cz
            index_slice = np.r_[0:10,15:20] # C3, Cz, EOG
            current_features = epoch_stage_features[:, index_slice].squeeze()
            stage_predict = rf_3.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_2EEG_EOG_model_C3_Cz')
        elif new_index == [0,2]: # C3, C4
            index_slice = np.r_[0:5,10:20] # C3, C4, EOG
            current_features = epoch_stage_features[:, index_slice].squeeze()
            stage_predict = rf_3.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_2EEG_EOG_model_C3_C4')
        elif new_index == [1,2]: # Cz, C4
            index_slice = np.r_[5:15,15:20] # Cz, C4, EOG
            current_features = epoch_stage_features[:, index_slice].squeeze()
            stage_predict = rf_3.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_2EEG_EOG_model_Cz_C4')
        else:
            print('No available 2EEG+EOG classification stream; waiting.')
            stage_predict = 0
        
    # Classifier with 2 EEG, 1 EMG (EOG failed)
    elif sum(channel_failure[0:3]) >= 2 and channel_failure[3] == 0 and channel_failure[4] == 1:
        index = np.where(channel_failure[0:3] == 1)[0]
        new_index = random.sample(list(index), k=2)
        new_index.sort()
        # Features: 2 EEG (10), EMG (5) = 15 features for rf_4
        if new_index == [0,1]: # C3, Cz
            index_slice = np.r_[0:10,20:25] # C3, Cz, EMG
            current_features = epoch_stage_features[:, index_slice].squeeze()
            stage_predict = rf_4.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_2EEG_EMG_model_C3_Cz')
        elif new_index == [0,2]: # C3, C4
            index_slice = np.r_[0:5,10:15,20:25] # C3, C4, EMG
            current_features = epoch_stage_features[:, index_slice].squeeze()
            stage_predict = rf_4.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_2EEG_EMG_model_C3_C4')
        elif new_index == [1,2]: # Cz, C4
            index_slice = np.r_[5:15,20:25] # Cz, C4, EMG
            current_features = epoch_stage_features[:, index_slice].squeeze()
            stage_predict = rf_4.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_2EEG_EMG_model_Cz_C4')
        else:
            print('No available 2EEG+EMG classification stream; waiting.')
            stage_predict = 0
        
    # Classifier to use if bipolar channels fail (2 EEG)          
    elif all(channel_failure[3:5]==0) and sum(channel_failure[0:3]) >= 2:
        index = np.where(channel_failure[0:3] == 1)[0]
        new_index = sorted(random.sample((list(index)), k=2))
        # Features: 2 EEG (10) for rf_5
        if new_index == [0,1]: # C3, Cz
            current_features = epoch_stage_features[:,0:10]
            stage_predict = rf_5.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_2EEG_model_C3_Cz')
        elif new_index == [0,2]: # C3, C4
            index_slice = np.r_[0:5,10:15]
            current_features = epoch_stage_features[:, index_slice].squeeze()
            stage_predict = rf_5.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_2EEG_model_C3_C4')
        elif new_index == [1,2]: # Cz, C4
            current_features = epoch_stage_features[:,5:15]
            stage_predict = rf_5.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_2EEG_model_Cz_C4')
        else:
            print('No available 2EEG classification stream; waiting.')
            stage_predict = 0
    
    # Classifer to use if bipolar channels + 2 EEG channels fail (1 EEG)
    elif sum(channel_failure[0:3])==1 and sum(channel_failure[3:5])==0:
        index = np.where(channel_failure[0:3] == 1)[0][0] # Get the single available EEG channel index
        # Features: 1 EEG (5) for rf_6
        if index == 0: # C3
            current_features = epoch_stage_features[:,0:5]
            stage_predict = rf_6.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_1EEG_model_C3')
        elif index == 1: # Cz
            current_features = epoch_stage_features[:,5:10]
            stage_predict = rf_6.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_1EEG_model_Cz')
        elif index == 2: # C4
            current_features = epoch_stage_features[:,10:15]
            stage_predict = rf_6.predict(current_features.reshape(1, -1))[0]
            reiz.marker.push('rf_1EEG_model_C4')
        else: # Should not happen if sum(channel_failure[0:3])==1
            print('Error in 1EEG selection logic.')
            stage_predict = 0
    else:
        print('No available classification streams; defaulting to Wake.')
        stage_predict = 0 # Default to Wake if no suitable model
    
    return stage_predict

def sleep_staging(bfr, indices_to_pull):
    """Sleep Stageing function.

    This function takes an lsl buffer argument and will be the sole argument in the 
    sleep stageing thread, integrating the selection, filtering, re-referencing, channel
    failure, and classification feature creation functions. Classification occurs every 
    15 seconds for the last 30 seconds of data.

    Parameters
    ----------
    bfr : Object
        lsl derived data and info object. 
    indices_to_pull : list
        Indices of channels to pull from the LSL buffer.
              
    Returns
    -------
    stage_predictArrays : list of integer values (Global)
        A list containing global variable stage_predictArrays, which corresponds to
        the classified stage for 30s epoch of extracted data and could either be:
        0 --> Wake, N1, REM, or 1 --> N2, SWS
        e.g., [1,1,1,0,1,0,0,....].    
    channel_failure : list of integer values (Global)
        A list containing the status of selected channels, either if they are intact --> 1,
        or whether they have failed --> 0.
    """
    global stage_predictArrays
    global channel_failure
    global coords # Should be set by get_coords if that's its purpose

    # Initialize coords if get_coords is meant to be called here
    coords, _ = get_coords() # Assuming get_coords is available and sets global coords

    stage_predictArrays = [] # Reset or initialize global
    tz = reiz.clock.now()
    
    # Load classifier models (adjust paths to be relative to this file's location)
    # Assuming models are in '../' relative to 'sleepstim/core/'
    model_path_prefix = "../" 
    try:
        rf_1 = pickle.load(open(model_path_prefix + "rf_model_1_cfs.p", "rb"))
        rf_2 = pickle.load(open(model_path_prefix + "rf_model_2_cfs.p", "rb"))
        rf_3 = pickle.load(open(model_path_prefix + "rf_model_3_cfs.p", "rb"))
        rf_4 = pickle.load(open(model_path_prefix + "rf_model_4_cfs.p", "rb"))
        rf_5 = pickle.load(open(model_path_prefix + "rf_model_5_cfs.p", "rb"))
        rf_6 = pickle.load(open(model_path_prefix + "rf_model_6_cfs.p", "rb"))
    except FileNotFoundError as e:
        print(f"Error loading RF models: {e}. Ensure models are in the parent directory.")
        return # Exit if models can't be loaded

    # Main loop for sleep staging
    # totalruntime for loop condition needs to be passed or defined
    totalruntime = 29700 # Example: 8.25 hours in seconds
    while reiz.clock.now() - tz < totalruntime: 
        data = bfr.get_data()[:,indices_to_pull]*1e6
        
        epoch_stage_features = epoch_psd(data, bfr.fs) # epoch_psd needs to be defined or imported
        
        current_channel_failure = channel_failure_test(epoch_stage_features) 
        channel_failure = current_channel_failure # Update global
        
        stage_predict = int(rf_model_select(current_channel_failure, epoch_stage_features, rf_1, rf_2, rf_3, rf_4, rf_5, rf_6)) 
                
        print(f'Sleep Stage: {stage_predict}')
        if reiz.marker.available(): # Check if marker stream is available
            reiz.marker.push('stage_predict')
        
        if stage_predict == 3: # Assuming 3 is SWS
            stage_predict_binary = 1
        else:
            stage_predict_binary = 0
        
        stage_predictArrays.append(stage_predict_binary)
            
        reiz.clock.sleep(15) # Pull data every ~15s
