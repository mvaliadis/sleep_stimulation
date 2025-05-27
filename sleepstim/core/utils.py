# -*- coding: utf-8 -*-
"""
Core utility functions for sleepstim project.
"""

import numpy as np
import mne # For mne.channels.make_standard_montage
import logging # For transition_matrix
from random import SystemRandom # Added import

sr = SystemRandom() # create an instance of the SystemRandom class - module level instance

def get_coords(): # Removed channels argument
    """Load channel coordinates from the channels.npy file."""
    # Path adjusted to be relative to sleepstim/core/
    # Assumes channels.npy is in the parent sleepstim/ directory
    try:
        loaded_channels_data = np.load('../channels.npy')
    except FileNotFoundError:
        print("Error: channels.npy not found at ../channels.npy. Please check the path.")
        return None, None

    if loaded_channels_data.ndim > 1 and loaded_channels_data.shape[1] > 0:
        # Assuming channel names are in the first column if it's 2D
        channel_names = loaded_channels_data[:,0]
    elif loaded_channels_data.ndim == 1:
        channel_names = loaded_channels_data
    else:
        print("Error: channels.npy has unexpected shape.")
        return None, None
        
    montage = mne.channels.make_standard_montage('standard_1005')
    # Get all channel positions from the montage
    montage_ch_coords = montage.get_positions()['ch_pos']
    
    coords_list = []
    valid_channels_list = []
    
    for ch_name in channel_names:
        if ch_name in montage_ch_coords:
            coords_list.append(montage_ch_coords[ch_name])
            valid_channels_list.append(ch_name)
        else:
            print(f"Warning: Channel '{ch_name}' not found in standard_1005 montage.")
            
    if not coords_list: # If no valid channels were found
        return None, None
        
    coords = np.array(coords_list)
    
    return coords, valid_channels_list

def transition_matrix(transitions):
    # the function takes a list with states labeled as successive integers and
    # returns a transition matrix, trans_max of all transitions between given states
    
    # derive number of states
    unique_states = np.unique(transitions)
    if not unique_states.size: # handle empty transitions array
        return np.array([])
        
    max_state = np.max(unique_states)
    # Ensure states are 0-indexed for matrix creation, or map them.
    # For simplicity, assuming states are 0 to N-1.
    # If states can be non-contiguous (e.g., 0, 2, 4), this needs adjustment.
    # The original code assumed states = len(np.unique(transitions)), which works if states are 0,1,2,...N-1
    # A more robust way might be to use max_state + 1 if states are 0-indexed.
    
    # Original logic:
    # states_count = len(np.unique(transitions)) 
    # if states_count == 5:
    #     trans_matrix = [[0]*states_count for _ in range(states_count)] 
    # else:
    #     trans_matrix = [[0]*5 for _ in range(5)] # Default to 5x5 if not 5 unique states? This seems odd.
    #     logging.warning(f'One of the datasets contains only {states_count} sleep stages, but a 5x5 matrix was created.')

    # Revised logic for dynamic sizing based on max_state, assuming 0-indexed states:
    num_states_dynamic = int(max_state + 1) # Ensure it's an int for matrix dimensions
    trans_matrix = np.zeros((num_states_dynamic, num_states_dynamic), dtype=int)
    
    # extract instances of each state
    # Ensure transitions is integer type for indexing
    transitions_int = transitions.astype(int)
    for (i,j) in zip(transitions_int[:-1], transitions_int[1:]): # Iterate up to second to last
        if i < num_states_dynamic and j < num_states_dynamic: # Bounds check
            trans_matrix[i][j] += 1
        
    return trans_matrix

def transition_matrix_prob(trans_matrix):
    # convert occurences to a transitional probability matrix to indicate the 
    # probability of transitioning from one state to the next
    if trans_matrix.size == 0:
        return np.array([])
        
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for states with no transitions from them
    # Replace 0s in row_sums with 1s to avoid division by zero, result will be 0 for these rows.
    probs = np.divide(trans_matrix, row_sums, out=np.zeros_like(trans_matrix, dtype=float), where=row_sums!=0)

    return np.round(probs.astype(float), 4)

def thresholdcrossings(x, threshold):
    """Find indices of threshold-crossings in a 1D array. 
    
    This function can be utilized to determine zero crossing as well as any 
    negative or postive crossing. 

    Parameters
    ----------
    x : np.array
        One dimensional data vector.
    threshold : float or int
        The threshold value to check crossings against.

    Returns
    -------
    idx_zc : np.array
        Indices of threshold-crossings (indices are for the point *before* the crossing)
    """
    pos = x > threshold
    npos = ~pos # This is equivalent to x <= threshold
    # Find where the sign of (x - threshold) changes
    # (pos[:-1] & npos[1:]) captures positive to non-positive crossings
    # (npos[:-1] & pos[1:]) captures non-positive to positive crossings
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]

def generate_subject_code(length, 
                      valid_chars=None):
    """ generate_subject_code(length, check_char) -> subject code
        length: the length of the created subject code
        check_char: a Boolean function used to check the validity of a char
    """
    # sr instance is now at module level
    if valid_chars==None:
        valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        valid_chars += "0123456789"
    
    code = ""
    counter = 0
    while counter < length:
        rnum = sr.randint(0, 128) # SystemRandom.randint is [a,b] inclusive
        # Ensure rnum is within valid ASCII range for chr() if that's intended,
        # or handle potential errors if rnum can be outside typical printable ASCII.
        # For now, assuming 0-128 is fine for chr().
        char = chr(rnum)
        if char in valid_chars:
            code += char # Use chr(rnum) as per original
            counter += 1
    return code
