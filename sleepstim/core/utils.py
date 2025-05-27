# -*- coding: utf-8 -*-
"""Core utility functions providing miscellaneous helper tools for the sleepstim project."""

import numpy as np
import mne # For mne.channels.make_standard_montage
import logging # For transition_matrix
from random import SystemRandom # Added import
from functools import reduce # Added import
import operator # Added import
import pandas as pd # Added import for drop_bads_df

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

    num_states_dynamic = int(max_state + 1) 
    trans_matrix = np.zeros((num_states_dynamic, num_states_dynamic), dtype=int)
    
    transitions_int = transitions.astype(int)
    for (i,j) in zip(transitions_int[:-1], transitions_int[1:]): 
        if i < num_states_dynamic and j < num_states_dynamic: 
            trans_matrix[i][j] += 1
        
    return trans_matrix

def transition_matrix_prob(trans_matrix):
    if trans_matrix.size == 0:
        return np.array([])
        
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    probs = np.divide(trans_matrix, row_sums, out=np.zeros_like(trans_matrix, dtype=float), where=row_sums!=0)

    return np.round(probs.astype(float), 4)

def thresholdcrossings(x, threshold):
    pos = x > threshold
    npos = ~pos 
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]

def generate_subject_code(length, 
                      valid_chars=None):
    if valid_chars==None:
        valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        valid_chars += "0123456789"
    
    code = ""
    counter = 0
    while counter < length:
        rnum = sr.randint(0, 128) 
        char = chr(rnum)
        if char in valid_chars:
            code += char 
            counter += 1
    return code

def drop_bads_df(df, subject_col='subject', night_col='night', subject_nights_to_drop=[('ChrSt', 1), ('UyDe', 1), ('IsEb', 2)]):
    """
    Removes specified subject-night combinations from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame. Must contain columns for subject identifiers and night numbers.
    subject_col : str, optional
        The name of the column in `df` that contains subject identifiers. Defaults to 'subject'.
    night_col : str, optional
        The name of the column in `df` that contains night numbers. Defaults to 'night'.
    subject_nights_to_drop : list of tuples, optional
        A list of (subject, night) tuples to be removed. 
        Example: [('ChrSt', 1), ('UyDe', 1)]

    Returns
    -------
    pd.DataFrame
        DataFrame with the specified rows removed.
    """
    if df.empty:
        return df

    # Ensure the night numbers in subject_nights_to_drop match the dtype of the night_col in df
    # If night_col is string, convert night numbers in subject_nights_to_drop to string.
    # If night_col is int/float, convert night numbers in subject_nights_to_drop to that type.
    example_night_val = df[night_col].iloc[0]
    if isinstance(example_night_val, str):
        processed_subject_nights = [(subj, str(night)) for subj, night in subject_nights_to_drop]
    elif isinstance(example_night_val, (int, np.integer)):
        processed_subject_nights = [(subj, int(night)) for subj, night in subject_nights_to_drop]
    elif isinstance(example_night_val, (float, np.floating)):
        processed_subject_nights = [(subj, float(night)) for subj, night in subject_nights_to_drop]
    else:
        processed_subject_nights = subject_nights_to_drop # Keep as is if type is unknown or mixed

    # Create masks for each condition to drop
    masks = []
    for subj, night_val in processed_subject_nights:
        masks.append(((df[subject_col] == subj) & (df[night_col] == night_val)))
    
    # Combine the individual masks with a logical OR
    if masks:
        combined_mask = reduce(operator.or_, masks)
        # Apply the mask to filter out the rows
        df_filtered = df[~combined_mask]
    else: # No conditions to drop
        df_filtered = df.copy()
    
    return df_filtered
