#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:34:55 2024

@author: administrator
"""

import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pingouin as pg
from functools import reduce
import operator
import scipy
import fooof

# Define the path to the summaries
stats_path = '/media/administrator/Sleep_Data/Processed/Statistics'

# Functions
def drop_bads_df(df, subject_nights = [('ChrSt', 1), ('UyDe', 1), ('IsEb', 2)]):
    # Create tuples of (Subject, Night) pairs to drop
    bads_and_nights = subject_nights
    # Check each row if it belongs to the tuples to be dropped
    mask = [(df['Subject'] == subj) & (df['Night'] == night) for subj, night in bads_and_nights]
    
    # Combine the individual masks with a logical OR
    combined_mask = reduce(operator.or_, mask)
    
    # Apply the mask to filter out the rows
    df = df[~combined_mask]
    
    return df

def nan_imputation_missing_data(df):
    # Prepare a list to collect new DataFrame rows
    new_rows = []
    
    # Get unique combinations of Subject and Night
    subject_night_combinations = df[['Subject', 'Night','Session']].drop_duplicates()
    
    for _, row in subject_night_combinations.iterrows():
        subject = row['Subject']
        night = row['Night']
        session = row['Session']
        
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
                'Session': session, 
                'Delta': np.nan,
                'Theta':  np.nan,
                'Alpha':  np.nan,
                'Beta':  np.nan,
                'TotalAbsPow':  np.nan,
                'Offset': np.nan,
                'Aperiodic': np.nan,
            }
            
            new_rows.append(new_row)
    
    # Create DataFrame from new rows
    new_df = pd.DataFrame(new_rows)
    
    # Combine with the original DataFrame and sort
    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=['Subject', 'Night', 'Mode'])
    
    return combined_df   

def prepare_df_to_numpy(df, features=['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Aperiodic']):
    # Assuming `df` is your DataFrame
    subjects = df['Subject'].unique()
    channels = df['Chan'].unique()

    # Number of subjects, channels, and features
    n_subjects = len(subjects)
    n_channels = len(channels)
    n_features = len(features)

    # Pre-allocate numpy array with NaNs to handle missing data gracefully
    data_array = np.nan * np.zeros((n_subjects, n_channels, n_features))

    # Populate the numpy array
    for i, subject in enumerate(subjects):
        for j, channel in enumerate(channels):
            for k, feature in enumerate(features):
                # Check if Aperiodic feature is requested and handle it specifically
                if feature == 'Aperiodic':
                    # Extracting 'Aperiodic' feature data, assuming it corresponds to a column in your DataFrame
                    feature_data = df[(df['Subject'] == subject) & (df['Chan'] == channel)]['Aperiodic']
                else:
                    # Extracting other features (Delta, Theta, Alpha, Sigma, Beta)
                    feature_data = df[(df['Subject'] == subject) & (df['Chan'] == channel)][feature]
                
                if not feature_data.empty:
                    data_array[i, j, k] = feature_data.values[0]  # Assuming one value per subject-channel-feature combination

    return data_array

def plot_psd_diff(rs_df):
    Info = mne.create_info(rs_df.Chan.unique().tolist(), 
                           sfreq=1000, ch_types="eeg")
    Info.set_montage(mne.channels.make_standard_montage('standard_1005'), 
                     match_case=False, on_missing="ignore")
    fig, axes = plt.subplots(2, 5, figsize=(2 * 5, 2.5 * 1.5))
    fig.suptitle('Resting-State EEG PSD Overnight Changes')
    grk = ['\u03B4', '\u03B8', '\u03B1', '\u03B2', 'Spec. ' '\u03B5']
    for idx, cond in enumerate(np.unique(rs_df.Mode)):
        for ind, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Aperiodic']):
                        
            # Get the PSD diff across channels for the current band
            bp_contrast_post = rs_df.groupby(['Session','Mode','Chan'])[f'{band}'].mean().loc['post'][cond] 
            bp_contrast_post = bp_contrast_post.reindex(list(rs_df.Chan.unique()))
            bp_contrast_pre = rs_df.groupby(['Session','Mode','Chan'])[f'{band}'].mean().loc['pre'][cond]
            bp_contrast_pre = bp_contrast_pre.reindex(list(rs_df.Chan.unique()))
            
            if band == 'Aperiodic':
                bp_contrast = bp_contrast_post - bp_contrast_pre
            else:
                bp_contrast = np.log(bp_contrast_post) - np.log(bp_contrast_pre)

            # Plot
            im, cn = mne.viz.plot_topomap(bp_contrast, 
                                          pos=Info, 
                                          cmap='Spectral_r', 
                                          #vlim=(-0.1, 0.1),
                                          #contours=0,
                                          sensors=False, 
                                          axes=axes[idx, ind], 
                                          show=False, names=None)
            
            # add color bar
            mne.viz.topomap._add_colorbar(axes[idx,ind], im, cmap = 'Spectral_r', 
                                          #side='right', pad=0.05, size='5%'
                                          title=None)
                
            # Set the plot title
            axes[idx, ind].set_title('\u0394 ' + grk[ind] + f' {cond}')
            
        # tighten layout
        plt.tight_layout()

def psd_stats(rs_df):
    contrasts = []
    for mode in rs_df.Mode.unique():
        df_post = rs_df[rs_df.Mode==mode][rs_df.Session=='post']#[rs_df.Condition=='eyes_closed']
        df_pre = rs_df[rs_df.Mode==mode][rs_df.Session=='pre']#[rs_df.Condition=='eyes_closed']
        
        grouped_post = df_post.groupby(['Subject', 'Chan']).mean(numeric_only=True)['Aperiodic'].unstack()[list(rs_df.Chan.unique())]
        grouped_pre = df_pre.groupby(['Subject', 'Chan']).mean(numeric_only=True)['Aperiodic'].unstack()[list(rs_df.Chan.unique())]
        
        def combine_common_subjects(grouped_post, grouped_pre):
            # Align both DataFrames to ensure only overlapping subjects are considered
            common_subjects = grouped_post.index.intersection(grouped_pre.index)
            
            # Subset both DataFrames to include only common subjects
            post_common = grouped_post.loc[common_subjects]
            pre_common = grouped_pre.loc[common_subjects]
        
            #contrast = np.log(post_common) - np.log(pre_common)
            contrast = post_common - pre_common
            
            return contrast
            
        contrasts.append(combine_common_subjects(grouped_post, grouped_pre))
    
    contrast = combine_common_subjects(contrasts[1], contrasts[0]).to_numpy()
        
    t_thresh = scipy.stats.t.ppf(1 - 0.05 / 2, df=contrast.shape[0])
    
    clus_kwargs = {'n_permutations' : 1024,  # 1000 is the minimum
                   'threshold' : t_thresh, 
                   'tail' : 1,               # one-tailed test (0 for two-tailed)
                   'n_jobs' : -1,            # increase value to speed up computations
                   'buffer_size' : None,
                   'out_type' : 'mask',      # returns a mask map instead of indices of sig. points
                   'seed' : 1503}
            
    Info = mne.create_info(rs_df.Chan.unique().tolist(), 
                           sfreq=1000, ch_types="eeg")
    Info.set_montage(mne.channels.make_standard_montage('standard_1005'), 
                     match_case=False, on_missing="ignore")
    adjacency, ch_names = mne.channels.find_ch_adjacency(Info, 'eeg')
    
    T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
       contrast,
       adjacency=adjacency,
       **clus_kwargs)
    
    fig, axes = plt.subplots(1,1)
    
    im, cn = mne.viz.plot_topomap(T_obs, 
                                  pos=Info, 
                                  cmap='Spectral_r', 
                                  #vlim=(-0.1, 0.1),
                                  #contours=0,
                                  mask=clusters[0],
                                  sensors=False, 
                                  axes=axes, 
                                  show=False, names=None)
    
    mne.viz.topomap._add_colorbar(axes, im, cmap = 'Spectral_r', 
                                  #side='right', pad=0.05, size='5%'
                                  title=None)
    
def psd_stats_rm_anova(rs_df, target='Aperiodic', test='Cluster'):
    # Initialize a structure to hold contrasts for each mode and session
    contrasts = {mode: {} for mode in rs_df.Mode.unique()}
    
    # Gather subjects present in all modes and sessions
    common_subjects = None
    
    for mode in rs_df.Mode.unique():
        mode_common = None
        for session in ['post', 'pre']:
            df_session = rs_df[(rs_df['Mode'] == mode) & (rs_df['Session'] == session)]
            grouped_session = df_session.groupby(['Subject', 'Chan']).mean(numeric_only=True)[target].unstack()[list(rs_df.Chan.unique())]
            if target != 'Aperiodic':
                if target.endswith('_Osc'):
                    pass
                else:
                    grouped_session = np.log(grouped_session)
            contrasts[mode][session] = grouped_session
            if mode_common is None:
                mode_common = set(grouped_session.index)
            else:
                mode_common &= set(grouped_session.index)
        if common_subjects is None:
            common_subjects = mode_common
        else:
            common_subjects &= mode_common
    
    common_subjects = list(common_subjects)  # Convert to list to ensure consistent indexing
    num_subjects = len(common_subjects)
    num_channels = len(rs_df.Chan.unique())
    
    # Initialize the array with correct dimensions
    data_array = np.zeros((2, 2, num_subjects, num_channels))
    
    # Populate the array
    for i, mode in enumerate(contrasts.keys()):
        for j, session in enumerate(['post', 'pre']):
            # Make sure to index only common subjects
            session_data = contrasts[mode][session].loc[common_subjects].values
            data_array[i, j] = session_data
    
    # Perform the factorial analysis
    reshaped_data = data_array.transpose(2, 0, 1, 3).reshape(num_subjects, 4, num_channels)
    
    if test == 'Cluster':
        full_list = [reshaped_data[:, [i], :].copy() for i in range(4)]

        def stat_fun(effect):
            def inner_stat_fun(*args):
                result = mne.stats.f_mway_rm(
                    np.swapaxes(args, 1, 0),
                    factor_levels=[2, 2],
                    effects=effect,
                    return_pvals=False,
                )[0]
                return result.flatten()
            return inner_stat_fun

        # Create adjacency
        Info = mne.create_info(rs_df.Chan.unique().tolist(), sfreq=1000, ch_types="eeg")
        Info.set_montage(mne.channels.make_standard_montage('standard_1005'), match_case=False, on_missing="ignore")
        adjacency, ch_names = mne.channels.find_ch_adjacency(Info, 'eeg')
        
        effects = ["A", "B", "A:B"]
        effect_labels = ["Mode", "Session", "Session x Mode"]
        fvals, pvals, clusters_all = [], [], []
        
        for effect in effects:
            pthresh = 0.05
            f_thresh = mne.stats.f_threshold_mway_rm(reshaped_data.shape[0], [2, 2], 
                                                     effect, pthresh)
            
            # Perform the permutation cluster test
            F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
                full_list,
                stat_fun=stat_fun(effect),
                adjacency=adjacency,
                threshold=f_thresh,
                tail=1,  # f-test, so tail > 0
                n_jobs=-1,
                n_permutations=1024,
                buffer_size=None,
                out_type="mask",
            )
            
            # Append the observed F-values and cluster p-values
            fvals.append(F_obs)
            pvals.append(cluster_p_values)
            clusters_all.append(clusters)
        
        # Plot the results
        plot_cluster_results(fvals, pvals, clusters_all, effect_labels, Info, target)

    else:
        fvals, pvals = mne.stats.f_mway_rm(reshaped_data, factor_levels=[2, 2], effects="A*B")
        effect_labels = ["Mode", "Session", "Session x Mode"]
        plot_effects(fvals, pvals, effect_labels, rs_df)

def plot_cluster_results(fvals, pvals, clusters_all, effect_labels, Info, target):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True, layout="constrained")
    for fval, pval, clusters, effect_label, ax in zip(fvals, pvals, clusters_all, effect_labels, axes):
        if len(pval) == 0 or all(pval > 0.05):
            im, cn = mne.viz.plot_topomap(fval.squeeze(), 
                                          pos=Info, 
                                          contours=0,
                                          mask=None,
                                          mask_params=dict(markersize=6, markerfacecolor='y'),
                                          sensors=False, 
                                          axes=ax, 
                                          show=False, names=None)
        else:
            im, cn = mne.viz.plot_topomap(fval.squeeze(), 
                                          pos=Info, 
                                          contours=0,
                                          mask=clusters[pval.argmin()].squeeze(),
                                          mask_params=dict(markersize=6, markerfacecolor='y'),
                                          sensors=False, 
                                          axes=ax, 
                                          show=False, names=None)
        ax.set_title(f"{effect_label} (p < 0.05)")
        # Add a colorbar
        cbar = plt.colorbar(ax.images[-1], ax=ax, orientation="horizontal")
        cbar.set_label("F-values")
    fig.suptitle(f'{target} rmANOVA Cluster Test', fontsize=18)

def plot_effects(fvals, pvals, effect_labels, rs_df):
    Info = mne.create_info(rs_df.Chan.unique().tolist(), sfreq=1000, ch_types="eeg")
    Info.set_montage(mne.channels.make_standard_montage('standard_1005'), 
                     match_case=False, on_missing="ignore")
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True, layout="constrained")
    for effect, pval, effect_label, ax in zip(fvals, pvals, effect_labels, axes):
        masks, _ = mne.stats.fdr_correction(pval)
        mne.viz.plot_topomap(effect, 
                             pos=Info, 
                             vlim=(0, 20),
                             contours=False,
                             mask=masks, 
                             mask_params=dict(markersize=6, markerfacecolor='y'), 
                             axes=ax, sensors=False, show=False)
        ax.set_title(f"{effect_label} (p < 0.05)")
        # Add a colorbar
        cbar = plt.colorbar(ax.images[-1], ax=ax, orientation="horizontal")
        cbar.set_label("F-values")
  
def fooof_plot(fm, cond):
    # Plot FOOOF results
    fig = plt.figure() 
    plt.tight_layout()
    plt.yscale('log')
    plt.xscale('log')
    
    afm_fooofed_spectrum_ = np.median([fm.get_fooof(ind=idx, regenerate=True).fooofed_spectrum_ for idx in range(len(fm))], axis=0)
    afm_ap_fit = np.median([fm.get_fooof(ind=idx, regenerate=True)._ap_fit for idx in range(len(fm))], axis=0)
    aperiodic = np.median([fm.get_fooof(ind=idx, regenerate=True).aperiodic_params_[1] for idx in range(len(fm))])
  
    plt.plot(fm.freqs, 10**(np.median(fm.power_spectra, 0)), c='k', label="Power spectrum", lw=2)
    plt.plot(fm.freqs, 10**(afm_ap_fit), c='b',linestyle='--', label="Aperiodic fit", lw=2)
    plt.plot(fm.freqs, 10**(afm_fooofed_spectrum_), c='r', label="FOOOF model fit", lw=2)
    fig.suptitle(f'PSD - {cond}')
    fig.axes[0].set_xlabel("Frequency (Hz)")
    fig.axes[0].set_ylabel("PSD log($V^2$/Hz)")
    fig.axes[0].legend()
    
    # set text with fit parameters
    fig.axes[0].text(0.1, 0.5, f"Slope: {round(aperiodic, 2)}",
                    transform=fig.axes[0].transAxes)
    sns.despine()
    
#%%
# 0. Load fooofed data
fooof_df = pd.read_pickle(os.path.join(stats_path, 'df_rs_fooof.p'))
fooof_df = drop_bads_df(fooof_df)
fooof_df['Session'] = fooof_df['Session'].replace({'pre': 'Evening', 'post': 'Morning'})

nmes_eyes_closed_morning = fooof_df.set_index(['Mode','Condition','Session']).loc['nmes', 'eyes_closed','Morning']
nmes_eyes_closed_evening = fooof_df.set_index(['Mode','Condition','Session']).loc['nmes', 'eyes_closed','Evening']

fg_necm = fooof.objs.combine_fooofs(list(nmes_eyes_closed_morning.fooof))
fooof_plot(fg_necm, 'CLNMES (Morning)')
fg_nece = fooof.objs.combine_fooofs(list(nmes_eyes_closed_evening.fooof))
fooof_plot(fg_nece, 'CLNMES (Evening)')

pn_eyes_closed_morning = fooof_df.set_index(['Mode','Condition','Session']).loc['pn', 'eyes_closed','Morning']
pn_eyes_closed_evening = fooof_df.set_index(['Mode','Condition','Session']).loc['pn', 'eyes_closed','Evening']

fg_pecm = fooof.objs.combine_fooofs(list(pn_eyes_closed_morning.fooof))
fooof_plot(fg_pecm, 'CLAS (Morning)')
fg_pece = fooof.objs.combine_fooofs(list(pn_eyes_closed_evening.fooof))
fooof_plot(fg_pece, 'CLAS (Evening)')

#%%
# 1. Load resting state data
rs_df = pd.read_csv(os.path.join(stats_path, 'df_rs.csv'), index_col=0)
rs_df = drop_bads_df(rs_df)

# Interaction stats:
psd_stats_rm_anova(rs_df, target='Aperiodic', test='Cluster')

# # Plot
# plot_psd_diff(rs_df[rs_df.Condition=='eyes_open'])
# plot_psd_diff(rs_df[rs_df.Condition=='eyes_closed'])
# plot_psd_diff(rs_df)
             
# 2. Select only frontocentral channels
frontocentral = ['Fz', 'FC1', 'FCz', 'FC2', 'Cz'] #, 'C1', 'C2']
fc_df = rs_df[rs_df.Chan.isin(frontocentral)]
# mean over fc channels
fc_df = fc_df.groupby(['Condition','Mode','Subject','Session']).mean()
# only eyes open condition
fc_df = fc_df.reset_index() #loc['eyes_closed'].reset_index()
fc_df['Session'] = fc_df['Session'].replace({'pre': 'Evening', 'post': 'Morning'})
 
# 3. FC theta plots
# Convert scale to log here
fc_df['Theta'] = np.log(fc_df['Theta'])
fc_df = nan_imputation_missing_data(fc_df).reset_index(drop=True)

# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)  # Adjust the size as needed

# First plot for mode 'pn'
pg.plot_paired(data=fc_df[fc_df.Mode == 'pn'],
               dv='Theta',
               within='Session',
               subject='Subject',
               boxplot_in_front=True,
               ax=axs[0])

# Second plot for mode 'nmes'
pg.plot_paired(data=fc_df[fc_df.Mode == 'nmes'],
               dv='Theta',
               within='Session',
               subject='Subject',
               boxplot_in_front=True,
               ax=axs[1])

# Set titles for each subplot for clarity
axs[0].set_title('CLAS', fontsize=18)
axs[0].set_ylabel('\u03B8 Spectral Density (log \u03BCV²/Hz)', fontsize=15)
axs[1].set_title('CLNMES', fontsize=18)

# Increase x-axis labels size
axs[0].set_xlabel('')  # Removes the x-axis label for the first subplot
axs[1].set_xlabel('')  # Removes the x-axis label for the second subplot
axs[0].tick_params(axis='x', labelsize=15) 
axs[1].tick_params(axis='x', labelsize=15)  

plt.tight_layout() 
plt.show()

fc_df.rm_anova(dv='Theta', subject='Subject', within=['Mode','Session'])


#%%
# 4. Convert dataframe to numpy for post and pre
rs_df_array_post = prepare_df_to_numpy(rs_df[np.logical_and(rs_df.Condition=='eyes_open',
                                                            rs_df.Session=='post')], 
                                 features=['Delta', 'Theta', 'Alpha', 'Beta', 'Aperiodic'])

rs_df_array_pre = prepare_df_to_numpy(rs_df[np.logical_and(rs_df.Condition=='eyes_open',
                                                           rs_df.Session=='pre')], 
                                 features=['Delta', 'Theta', 'Alpha', 'Beta', 'Aperiodic'])

#%%
# Load HRV data from RS 
hrv_rs_df = pd.read_csv(os.path.join(stats_path, 'df_rs_hrv.csv'), index_col=0)
hrv_rs_df = drop_bads_df(hrv_rs_df)
hrv_rs_df['Session'] = hrv_rs_df['Session'].replace({'pre': 'Evening', 'post': 'Morning'})

# Remove bad HR recordings
hrv_rs_df = hrv_rs_df[hrv_rs_df.Quality!='Unnacceptable']

# Log-transform HRV metrics, except heart rate
hrv_rs_df['HRV_RMSSD'] = np.log(hrv_rs_df['HRV_RMSSD'])
hrv_rs_df['HRV_SDNN'] = np.log(hrv_rs_df['HRV_SDNN'])
hrv_rs_df['HRV_LF'] = np.log(hrv_rs_df['HRV_LF'])
hrv_rs_df['HRV_HF'] = np.log(hrv_rs_df['HRV_HF'])
# hrv_rs_df['HRV_HFn'] = np.log(hrv_rs_df['HRV_HFn'])
# hrv_rs_df['HRV_LFn'] = np.log(hrv_rs_df['HRV_LFn'])
hrv_rs_df['HRV_LFHF'] = np.log(hrv_rs_df['HRV_LFHF'])
hrv_rs_df['HRV_MeanNN'] = np.log(hrv_rs_df['HRV_MeanNN'])

# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)  # Adjust the size as needed

# First plot for mode 'pn'
pg.plot_paired(data=hrv_rs_df[hrv_rs_df.Mode == 'pn'],
               dv='HRV_LFHF',
               within='Session',
               subject='Subject',
               ax=axs[0])

# Second plot for mode 'nmes'
pg.plot_paired(data=hrv_rs_df[hrv_rs_df.Mode == 'nmes'],
               dv='HRV_LFHF',
               within='Session',
               subject='Subject',
               ax=axs[1])

# Set titles for each subplot for clarity
axs[0].set_title('CLAS', fontsize=18)
axs[0].set_ylabel('Log HRV LF/HF', fontsize=15)
axs[1].set_title('CLNMES', fontsize=18)

# Increase x-axis labels size
axs[0].set_xlabel('')  # Removes the x-axis label for the first subplot
axs[1].set_xlabel('')  # Removes the x-axis label for the second subplot
axs[0].tick_params(axis='x', labelsize=15) 
axs[1].tick_params(axis='x', labelsize=15)  

plt.tight_layout() 
plt.show()

hrv_rs_df.rm_anova(dv='HR', subject='Subject', within=['Mode','Session'])
hrv_rs_df.rm_anova(dv='HRV_LF', subject='Subject', within=['Mode','Session'])
hrv_rs_df.rm_anova(dv='HRV_LFn', subject='Subject', within=['Mode','Session'])
hrv_rs_df.rm_anova(dv='HRV_HF', subject='Subject', within=['Mode','Session'])
hrv_rs_df.rm_anova(dv='HRV_LnHF', subject='Subject', within=['Mode','Session'])
hrv_rs_df.rm_anova(dv='HRV_LFHF', subject='Subject', within=['Mode','Session'])
hrv_rs_df.rm_anova(dv='HRV_RMSSD', subject='Subject', within=['Mode','Session'])
hrv_rs_df.rm_anova(dv='HRV_SDNN', subject='Subject', within=['Mode','Session'])
hrv_rs_df.rm_anova(dv='HRV_MeanNN', subject='Subject', within=['Mode','Session'])
