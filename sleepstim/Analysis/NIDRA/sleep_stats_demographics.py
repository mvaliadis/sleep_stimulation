#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:35:57 2024

@author: administrator
"""


import pandas as pd
import numpy as np
import seaborn as sns
import pingouin as pg

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
## load subject csv
df = pd.read_csv('/mnt/server/data03/2023_NIDRA/Study documents/subject_demo_final.csv')
df['SSS_difference'] = df['SSS post'] - df['SSS pre']

## All Subjects
# Mean Age ± STD
print(df.Age.agg(['mean', 'std']))

# PQSI
print(df.PSQI.agg(['mean', 'std']))

# MEQ counts
print(df.MEQ.value_counts())

# NMES intensity
print(df.groupby('Stim technique').NMES.agg(['mean', 'std']))

# SSS : 1=feeling active, wide awake to 7=no longer fighting sleep, above > 3 is sleepy
# SSS (pre) 
print(df.groupby('Stim technique')['SSS pre'].agg(['mean', 'std']))

# SSS (post)
print(df.groupby('Stim technique')['SSS post'].agg(['mean', 'std']))

# SSS difference
print(df.groupby('Stim technique')['SSS_difference'].agg(['mean', 'std']))


#%%
## Included subjects
df = drop_bads_df(df)

# Mean Age ± STD
print(df.Age.agg(['mean', 'std']))