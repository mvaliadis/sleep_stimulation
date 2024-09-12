#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:30:09 2024

@author: administrator
"""

import numpy as np
import pandas as pd
import os
import pingouin as pg
import matplotlib.pyplot as plt

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

def pvt_plot(df_pvt, dv='Lapse_Probability'):
    # Create a figure with four subplots
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    
    pg.plot_paired(data=df_pvt,
                   dv=dv,
                   within='Mode',
                   subject='Subject',
                   ax=axs,
                   #boxplot=False,
                   boxplot_in_front=False, 
                   )
    
    # Set titles and labels
    if dv=='Lapse_Probability' or dv=='Lapses_Transformed':
        axs.set_title('Lapses', fontsize=18)
        axs.set_ylabel(f'{" ".join(dv.split("_"))}', fontsize=15)
    elif dv=='RT':
        axs.set_title('RT', fontsize=18)
        axs.set_ylabel(f'{dv} (ms)', fontsize=15)
    axs.set_xlabel('')
    axs.tick_params(axis='x', labelsize=15)
    plt.tight_layout()
  
def nan_imputation_missing_data(df):
    # Prepare a list to collect new DataFrame rows
    new_rows = []
    
    # Get unique combinations of Subject and Night
    subject_night_combinations = df[['Subject', 'Night']].drop_duplicates()
    
    for _, row in subject_night_combinations.iterrows():
        subject = row['Subject']
        night = row['Night']
        
        # Check for existing modes for each subject-night combination
        existing_modes = df[(df['Subject'] == subject) & (df['Night'] == night)]['Mode'].unique()
        
        # Determine missing modes
        all_modes = {'pn', 'nmes'}
        missing_modes = all_modes - set(existing_modes)
        
        # For each missing mode, create a new row with NaNs for certain columns
        for mode in missing_modes:
            new_row = {
                'Subject': subject,
                'Night': night,
                'Mode': mode,
                'RT': np.nan,
                'Speed':  np.nan,
                'Lapses':  np.nan,
                'Lapse_Probability':  np.nan,
                'Trial':  np.nan  # or adjust based on available data logic if needed
            }
            new_rows.append(new_row)
    
    # Create DataFrame from new rows
    new_df = pd.DataFrame(new_rows)
    
    # Combine with the original DataFrame and sort
    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=['Subject', 'Night', 'Mode'])
    
    return combined_df

stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'

#%%
## 0. PVT analysis
df_pvt = pd.read_csv(os.path.join(stats_path, 'df_pvt.csv'), index_col=0)
df_pvt = drop_bads_df(df_pvt)
df_pvt_mean = df_pvt.groupby(['Subject','Night','Mode']).mean()
df_pvt_mean = nan_imputation_missing_data(df_pvt_mean.reset_index()).reset_index(drop=True)

# Plot
pvt_plot(df_pvt_mean, dv='RT')
pvt_plot(df_pvt_mean, dv='Lapse_Probability')
pvt_plot(df_pvt_mean, dv='Lapses_Transformed')

# Stats
res_RT = pg.wilcoxon(x=df_pvt_mean.groupby(['Mode','Subject']).mean().loc['nmes'].RT,
                     y=df_pvt_mean.groupby(['Mode','Subject']).mean().loc['pn'].RT)
print(f'RT : \n {res_RT}')

res_Lapses = pg.wilcoxon(x=df_pvt_mean.groupby(['Mode','Subject']).mean().loc['nmes'].Lapse_Probability,
                         y=df_pvt_mean.groupby(['Mode','Subject']).mean().loc['pn'].Lapse_Probability)
print(f'Lapses : \n {res_Lapses}')

res_Lapses = pg.wilcoxon(x=df_pvt_mean.groupby(['Mode','Subject']).mean().loc['nmes'].Lapses_Transformed,
                         y=df_pvt_mean.groupby(['Mode','Subject']).mean().loc['pn'].Lapses_Transformed)
print(f'Lapses (Transformed) : \n {res_Lapses}')
