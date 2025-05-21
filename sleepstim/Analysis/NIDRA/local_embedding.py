#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:11:16 2025

@author: administrator
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import mne
from mne import io
from mne.datasets import sample
from sklearn.model_selection import train_test_split

from pyriemann.estimation import XdawnCovariances
from pyriemann.utils.viz import plot_embedding

stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'

def drop_bads_df(df, subject_nights=[('ChrSt', 1), ('UyDe', 1), ('IsEb', 2)]):
    from functools import reduce
    import operator
    # Ensure the night numbers are of the same type as in the DataFrame
    # If Night is a string in the DataFrame, convert the night numbers to strings
    subject_nights = [(subj, str(night)) if isinstance(df['Night'].iloc[0], str) else (subj, night) for subj, night in subject_nights]
    
    # Create masks for each condition to drop
    masks = [((df['Subject'] == subj) & (df['Night'] == night)) for subj, night in subject_nights]
    
    # Combine the individual masks with a logical OR
    if masks:
        combined_mask = reduce(operator.or_, masks)
    else:
        combined_mask = pd.Series([False] * len(df))
    
    # Apply the mask to filter out the rows
    df = df[~combined_mask]
    return df

#%%
df_epochs = pd.read_pickle(os.path.join(stats_path, 'df_epochs.p')) 
df_epochs = drop_bads_df(df_epochs)
df_epochs.set_index(['Mode','Subject','Night'], inplace=True)
 
epochs_nmes = df_epochs.loc['nmes'].Epochs[4]
epochs_pn = df_epochs.loc['pn'].Epochs[4]

# Find the maximum event code in the nmes epochs:
max_event_nmes = max(df_epochs.loc['nmes'].Epochs[4].event_id.values())

# Define an offset (for example, one more than the maximum)
offset = max_event_nmes + 1

# Update the events and the event_id for epochs_pn
epochs_pn.events[:, -1] += offset
epochs_pn.event_id = {key: val + offset for key, val in epochs_pn.event_id.items()}

epochs = mne.concatenate_epochs([epochs_nmes, epochs_pn])

X = epochs.get_data()
y = epochs.events[:, -1]

## Embedding of Xdawn covariance matrices¶
nfilter = 4
xdwn = XdawnCovariances(estimator="scm", nfilter=nfilter)
split = train_test_split(X, y, train_size=0.25, random_state=42)
Xtrain, Xtest, ytrain, ytest = split
covs = xdwn.fit(Xtrain, ytrain).transform(Xtest)

## Laplacian Eigenmaps (LE), also called Spectral Embedding (SE)¶
plot_embedding(covs, ytest, metric="riemann", embd_type="Spectral",
               normalize=True)
plt.show()