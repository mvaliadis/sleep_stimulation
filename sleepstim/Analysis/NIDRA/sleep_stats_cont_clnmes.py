#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:28:21 2024

@author: administrator
"""

import yasa
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from sleepstim.Analysis.NIDRA.swa_decay import plot_group_swa_decay
import scipy
import numpy as np
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import mne
import os

## Helper functions
def plot_mode_condition(data, dv, ax, title):
    # Plot paired data 
    pg.plot_paired(data=data,
                   dv=dv,
                   within='Mode',
                   subject='Subject',
                   ax=ax,
                   #boxplot=False,
                   boxplot_in_front=False, 
                   )
    
    # Set titles and labels
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('')
    ax.set_ylabel(f'{dv}', fontsize=15)
    ax.tick_params(axis='x', labelsize=15)
    plt.tight_layout()
    
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
                'Lapses_Transformed':  np.nan,
                'Trial':  np.nan  # or adjust based on available data logic if needed
            }
            new_rows.append(new_row)
    
    # Create DataFrame from new rows
    new_df = pd.DataFrame(new_rows)
    
    # Combine with the original DataFrame and sort
    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=['Subject', 'Night', 'Mode'])
    
    return combined_df   

def topoplot(
    data,
    montage="standard_1020",
    vmin=None,
    vmax=None,
    mask=None,
    title=None,
    cmap=None,
    n_colors=100,
    cbar_title=None,
    cbar_ticks=None,
    show_cbar=True,
    figsize=(6, 6),
    dpi=80,
    fontsize=14,
    axes=None,
    **kwargs,
):
    """
    Topoplot.

    This is a wrapper around :py:func:`mne.viz.plot_topomap`.

    For more details, please refer to this `example notebook
    <https://github.com/raphaelvallat/yasa/blob/master/notebooks/15_topoplot.ipynb>`_.

    .. versionadded:: 0.4.1

    Parameters
    ----------
    data : :py:class:`pandas.Series`
        A pandas Series with the values to plot. The index MUST be the channel
        names (e.g. ['C4', 'F4'] or ['C4-M1', 'C3-M2']).
    montage : str
        The name of the montage to use. Valid montages can be found at
        :py:func:`mne.channels.make_standard_montage`.
    vmin, vmax : float
        The minimum and maximum values of the colormap. If None, these will be
        defined based on the min / max values of ``data``.
    mask : :py:class:`pandas.Series`
        A pandas Series indicating the significant electrodes. The index MUST
        be the channel names (e.g. ['C4', 'F4'] or ['C4-M1', 'C3-M2']).
    title : str
        The plot title.
    cmap : str
        A matplotlib color palette. A list of color palette can be found at:
        https://seaborn.pydata.org/tutorial/color_palettes.html
    n_colors : int
        The number of colors to discretize the color palette.
    cbar_title : str
        The title of the colorbar.
    cbar_ticks : list
        The ticks of the colorbar.
    figsize : tuple
       Width, height in inches.
    dpi : int
        The resolution of the plot.
    fontsize : int
        Global font size of all the elements of the plot.
    **kwargs : dict
        Other arguments that are passed to :py:func:`mne.viz.plot_topomap`.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        Matplotlib Figure

    Examples
    --------

    1. Plot all-positive values

    .. plot::

        >>> import yasa
        >>> import pandas as pd
        >>> data = pd.Series([4, 8, 7, 1, 2, 3, 5],
        ...                  index=['F4', 'F3', 'C4', 'C3', 'P3', 'P4', 'Oz'],
        ...                  name='Values')
        >>> fig = yasa.topoplot(data, title='My first topoplot')

    2. Plot correlation coefficients (values ranging from -1 to 1)

    .. plot::

        >>> import yasa
        >>> import pandas as pd
        >>> data = pd.Series([-0.5, -0.7, -0.3, 0.1, 0.15, 0.3, 0.55],
        ...                  index=['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'Pz'])
        >>> fig = yasa.topoplot(data, vmin=-1, vmax=1, n_colors=8,
        ...                     cbar_title="Pearson correlation")
    """
    from matplotlib.colors import ListedColormap
    
    # Increase font size while preserving original
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams.update({"savefig.bbox": "tight"})
    plt.rcParams.update({"savefig.transparent": "True"})

    # Make sure we don't do any in-place modification
    assert isinstance(data, pd.Series), "Data must be a Pandas Series"
    data = data.copy()

    # Add mask, if present
    if mask is not None:
        assert isinstance(mask, pd.Series), "mask must be a Pandas Series"
        assert mask.dtype.kind in "bi", "mask must be True/False or 0/1."
    else:
        mask = pd.Series(1, index=data.index, name="mask")

    # Convert to a dataframe (col1 = values, col2 = mask)
    data = data.to_frame().join(mask, how="left")

    # Preprocess channel names: C4-M1 --> C4
    data.index = data.index.str.split("-").str.get(0)

    # Define electrodes coordinates
    Info = mne.create_info(data.index.tolist(), sfreq=100, ch_types="eeg")
    Info.set_montage(montage, match_case=False, on_missing="ignore")
    chan = Info.ch_names

    # Define vmin and vmax
    if vmin is None:
        vmin = data.iloc[:, 0].min()
    if vmax is None:
        vmax = data.iloc[:, 0].max()

    # Choose and discretize colormap
    if cmap is None:
        if vmin < 0 and vmax <= 0:
            cmap = "mako"
        elif vmin < 0 and vmax > 0:
            cmap = "Spectral_r"
        elif vmin >= 0 and vmax > 0:
            cmap = "rocket_r"

    cmap = ListedColormap(sns.color_palette(cmap, n_colors).as_hex())

    if "sensors" not in kwargs:
        kwargs["sensors"] = True
    if "res" not in kwargs:
        kwargs["res"] = 256
    if "names" not in kwargs:
        kwargs["names"] = chan   
    if "mask_params" not in kwargs:
        kwargs["mask_params"] = dict(marker=None)

    # Hidden feature: if names='values', show the actual values.
    if kwargs["names"] == "values":
        kwargs["names"] = data.iloc[:, 0][chan].round(2).to_numpy()

    # Start the plot
    with sns.axes_style("white"):
        if axes is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = plt.gcf()
            ax = axes
        # Plot topomap
        im, _ = mne.viz.plot_topomap(
            data=data.iloc[:, 0][chan],
            pos=Info,
            vlim=(vmin, vmax), 
            mask=data.iloc[:, 1][chan],
            cmap=cmap,
            show=False,
            axes=ax,
            **kwargs,
            
        )

        if title is not None:
            ax.set_title(title)

        # Add colorbar
        if cbar_title is None:
            cbar_title = data.iloc[:, 0].name

        if show_cbar == True:
            cbar = fig.colorbar(ax.images[-1], ax=ax, orientation="vertical",
                            ticks=None, fraction=0.05)
            cbar.set_label(cbar_title)

            #cax = fig.add_axes([0.95, 0.3, 0.02, 0.5])
            #cbar = fig.colorbar(im, cax=cax, ticks=cbar_ticks, fraction=0.5)
            #cbar.set_label(cbar_title)

        # Revert font-size
        plt.rcParams.update({"font.size": old_fontsize})
        
    return fig

def filter_pvt_outliers(df, plot=False):
    # Group data by 'Subject' and 'Session'
    groups = df.groupby(['Subject', 'Night'])
    
    # Define a function to calculate IQR and remove outliers for each group
    def filter_group(group):
        Q1 = group['RT'].quantile(0.25)
        Q3 = group['RT'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return group[(group['RT'] >= lower_bound) & (group['RT'] <= upper_bound)]
   
    # Apply the function to each group and concatenate the results
    data_filtered = groups.apply(filter_group).reset_index(drop=True)
           
    if plot:   
        # Before removing outliers
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Subject', y='RT', data=df)
        plt.title('Before Removing Outliers')
        plt.show()
        
        # After removing outliers
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Subject', y='RT', data=data_filtered)
        plt.title('After Removing Outliers')
        plt.show()
    
    return data_filtered
# Load data 
stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'

#%%
df_sleep_stats = pd.read_csv(stats_path + 'df_sleep_stats.csv', index_col=0)
df_sleep_stats = drop_bads_df(df_sleep_stats)

df_sleep_trans = pd.read_csv(stats_path + 'df_trans.csv', index_col=0)
df_sleep_trans = drop_bads_df(df_sleep_trans)

df_hrv = pd.read_csv(stats_path + 'df_sleep_hrv.csv', index_col=0)
df_hrv = drop_bads_df(df_hrv)

df_sws = pd.read_csv(stats_path + 'df_sw.csv', index_col=0)
df_sws = drop_bads_df(df_sws)

df_spindle = pd.read_csv(stats_path + 'df_spindles.csv', index_col=0)
df_spindle = drop_bads_df(df_spindle)

df_decay = pd.read_pickle('/media/administrator/Sleep_Data/Processed/Statistics/df_swa_decay.p')
df_decay = drop_bads_df(df_decay)
# plot_group_swa_decay(df_decay, channel=None)
    
#%%
# 0. PVT/Sleep Macroarchitecture df
df_pvt = pd.read_csv(stats_path + 'df_pvt.csv', index_col=0)
df_pvt = drop_bads_df(df_pvt)
df_pvt = filter_pvt_outliers(df_pvt, plot=False)
df_pvt_mean = df_pvt.groupby(['Subject','Night','Mode']).mean()
df_pvt_mean = nan_imputation_missing_data(df_pvt_mean.reset_index()).reset_index(drop=True)

# Merge the dataframes
merged_df = pd.merge(df_pvt_mean, df_sleep_stats, on=['Subject', 'Night', 'Mode'])
sns.lmplot(merged_df, y='Lapses_Transformed', x='SE')
sns.lmplot(merged_df, y='RT', x='SE')
merged_df.to_csv('/media/administrator/Sleep_Data/Processed/Statistics/df_pvt_sleep_stats.csv')

def plot_pvt_sleep_heatmap(df_merged):
    # Select relevant columns for the heatmap
    columns_of_interest = ['RT', 'Lapses_Transformed', 'TST', 'SE', 'SME', '%REM', '%NREM', 'WASO']
    df_corr = merged_df[columns_of_interest]
        
    # Compute the correlation matrix
    corr_matrix = df_corr.corr()
    
    # Filter the correlation matrix to only show desired correlations
    corr_matrix_filtered = corr_matrix.loc[['RT', 'Lapses_Transformed'], 
                                           ['TST', 'SE', 'SME', '%REM', '%NREM', 'WASO']]
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))
    
    # Generate a heatmap
    sns.heatmap(corr_matrix_filtered, annot=True, linewidths=0.5)
    
    # Title and labels
    plt.title('Correlation Heatmap: Sleep Quality & PVT', fontsize=18)
    plt.show()

# Plot correlation heatmap
plot_pvt_sleep_heatmap(merged_df)
 
#%%
## 1. Sleep Macroarchitecture 
def sleep_marcroarchitecture_stats_old(df_sleep_stats):
    # Partition data into 2 stim nights    
    x = df_sleep_stats.set_index(['Mode','Subject']).loc['pn']
    y = df_sleep_stats.set_index(['Mode','Subject']).loc['nmes'] 
    
    # Get the intersection of the indices
    common_subjects = x.index.intersection(y.index)

    # Filter the original Series to keep only common subjects
    x_common = x.loc[common_subjects]
    y_common = y.loc[common_subjects]
    common_df = df_sleep_stats.set_index('Subject').loc[common_subjects]
    
    for column in df_sleep_stats.columns[3:]:       
        print(f"{column}: \n {pg.wilcoxon(x_common[column], y_common[column], alternative='two-sided')}")

        # Create a figure with four subplots
        fig, axs = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
    
        # Plot 
        plot_mode_condition(data = common_df, 
                            dv = column, 
                            ax = axs,
                            title = column)

   
        # Adjust layout
        plt.show()
        
        
    common_df = df_sleep_trans.set_index('Subject').loc[common_subjects]
    # Create a figure with four subplots
    fig, axs = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
    plot_mode_condition(data = common_df, 
                        dv = 'Stability', 
                        ax = axs,
                        title = 'Sleep Stability')
    
    nmes_stability = df_sleep_trans.groupby(['Mode','Subject'])['Stability'].mean().loc['nmes']
    pn_stability = df_sleep_trans.groupby(['Mode','Subject'])['Stability'].mean().loc['pn']

    x_common = nmes_stability.loc[common_subjects]
    y_common = pn_stability.loc[common_subjects]
    
    print(f"Sleep Stability: \n {pg.wilcoxon(x_common, y_common, alternative='two-sided')}")

def sleep_macroarchitecture_stats(df_sleep_stats):
    # Partition data into 2 stim nights
    x = df_sleep_stats.set_index(['Mode', 'Subject']).loc['pn']
    y = df_sleep_stats.set_index(['Mode', 'Subject']).loc['nmes']

    # Get the intersection of the indices
    common_subjects = x.index.intersection(y.index)

    # Filter the original DataFrame to keep only common subjects
    x_common = x.loc[common_subjects]
    y_common = y.loc[common_subjects]
    common_df = df_sleep_stats.set_index('Subject').loc[common_subjects]

    results = []
    
    columns = ['TST','SE','SME','SOL','Lat_REM','WASO','%N1','%N2','%N3','%REM']
    # for column in df_sleep_stats.columns[3:]:
    for column in columns:
        wilcoxon_result = pg.wilcoxon(x_common[column], y_common[column], alternative='two-sided')
        wilcoxon_result['Metric'] = column
        results.append(wilcoxon_result)

        # Create a figure and plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
        plot_mode_condition(data=common_df.reset_index(), dv=column, ax=ax, title=column)
        plt.show()

    results_df = pd.concat(results).reset_index(drop=True)

    # Correct p-values for multiple comparisons
    corrected_pvalues = multipletests(results_df['p-val'], method='fdr_bh')[1]
    results_df['p-val_corrected'] = corrected_pvalues

    # Save the results table
    # results_df.to_csv('/mnt/data/sleep_macroarchitecture_results.csv', index=False)

    return results_df

results_df = sleep_macroarchitecture_stats(df_sleep_stats)
print(results_df)
plt.close('all')

#%%
# 2. Spindle Analysis

# Load spindle data
df_sp = pd.read_csv(os.path.join(stats_path, 'df_spindles.csv'), index_col=0)
df_sp = drop_bads_df(df_sp)
df_sp.rename(columns={'Channel':'Chan'}, inplace=True)
df_sp = df_sp.groupby('Subject').filter(lambda x: set(x['Mode']) >= {'pn', 'nmes'})

# yasa.topoplot(df_sp[df_sp.Stage==2].groupby(['Mode','Chan']).mean(numeric_only=True).Density.loc['nmes'] -
#               df_sp[df_sp.Stage==2].groupby(['Mode','Chan']).mean(numeric_only=True).Density.loc['pn'])

def SO_spindle_stats(df_sp, target='Density', oscillation='spindle', stage=2):
    
    # Initialize a structure to hold contrasts for each level of the factor
    contrasts = {}
    
    # Create copy of df
    if stage != None:
        df = df_sp[df_sp.Stage==stage].copy().reset_index(drop=True)
    else:
        df = df_sp.copy()      
    
    # Drop subject DoGi
    df = df[df.Subject!='DoGi']
    
    for level in df.Mode.unique():
        grouped = df.groupby(['Mode','Subject','Chan']).mean(numeric_only=True).loc[level][target].unstack()[list(df.Chan.unique())]
        contrasts[level] = grouped
              
    # Create contrast for nmes - pn spindle density
    contrasts_diff = (np.log1p(contrasts['nmes']) - np.log1p(contrasts['pn']) 
                      / np.log1p(contrasts['pn'])) * 100
    # contrasts_diff = contrasts['nmes'] - contrasts['pn']
    # contrast = scipy.stats.zscore(contrasts['nmes'].values, axis=-1) - scipy.stats.zscore(contrasts['pn'].values, axis=-1)
    contrast = contrasts_diff.values
    # s_na, ch_na = np.where(np.isnan(contrast))
    # contrast[s_na[0], ch_na[0]] = 0
    
    # threshold for cluster test
    pval = 0.05
    dof = contrast.shape[0] - 1  # degrees of freedom for the test
    thresh = scipy.stats.t.ppf(1 - pval / 2, dof)  # two-tailed, t distribution
    
    # adjacency matrix
    Info = mne.create_info(df.Chan.unique().tolist(), sfreq=128, ch_types="eeg")
    Info.set_montage(mne.channels.make_standard_montage('standard_1005'), match_case=False, on_missing="ignore")
    adjacency, ch_names = mne.channels.find_ch_adjacency(Info, 'eeg')
    
    # Perform the permutation cluster test
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        contrast,
        threshold=thresh,
        adjacency=adjacency,
        tail=0,  # two-tailed test
        n_jobs=-1,
        n_permutations=1024,
        buffer_size=None,
        out_type="mask",
        seed=4,
      )
    
    # Get significant clusters
    if all(p > 0.05 for p in cluster_p_values):
        df_stats = pd.DataFrame({'Chan': ch_names, 
                                 'Power': contrast.mean(0).squeeze(), 
                                 'T-Stat': t_obs.squeeze(),
                                 'Null' : [False] * len(ch_names),
                                 'Sig': [False] * len(ch_names)})
    else:
        df_stats = pd.DataFrame({'Chan': ch_names, 
                                 'Power': contrast.mean(0).squeeze(), 
                                 'T-Stat': t_obs.squeeze(), 
                                 'Null' : [False] * len(ch_names),
                                 'Sig': clusters[np.argmin(cluster_p_values)]
                                 # 'Sig': clusters[0]
                                 })
    
    # Set channel as index
    df_stats = df_stats.set_index("Chan")
    
    # Plot feature and stats
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    unit = 'Log Ratio' + f' ({target})'
    
    # Plot topomaps
    topoplot(df_stats['Power'], 
             mask=df_stats['Null'], 
             cmap='Reds',
             names=None,
             axes=ax[0],
             cbar_title=unit,
            )
    
    topoplot(df_stats['T-Stat'], 
            mask=df_stats['Sig'], 
            cmap='Reds',
            names=None,
            axes=ax[1],
            mask_params=dict(markersize=6, markerfacecolor='y'), 
            )
    
    # Plot cluster p-values at the bottom of the figure
    p_value_str = ', '.join([f'Cluster p={cluster_p_values.min():.3f}'])
    fig.text(0.5, 0.01, p_value_str, ha='center', fontsize=12, color='black')

    # Add figure title
    if oscillation == 'SO':
        fig.suptitle(f'Slow Wave {target.capitalize()} (CLNMES - CLAS)', fontsize=18)
    elif oscillation == 'spindle':
        fig.suptitle(f'Spindle {target.capitalize()} (CLNMES - CLAS)', fontsize=18)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# spindle stats
SO_spindle_stats(df_sp, target='Density', oscillation='spindle', stage=2)
SO_spindle_stats(df_sp, target='Density', oscillation='spindle', stage=None)

# spindle stats with channels averaged
# df_avg_sp = df_sp.set_index('Stage').loc[2].groupby(['Mode','Subject']).mean(numeric_only=True).reset_index()
# pg.plot_paired(df_avg_sp, dv='Density', within='Mode', subject='Subject', boxplot=False)
# df_avg_sp.rm_anova(dv='Density', subject='Subject', within='Mode')

#%%
# 3. Slow Wave Analysis
variables = ['Count', 'Density', 'Duration','ValNegPeak', 
             'ValPosPeak', 'PTP', 'Slope', 'Frequency',
             'PhaseAtSigmaPeak', 'ndPAC', 'CooccurringSpindle','DistanceSpindleToSW']
    
# Load slow wave and spindle data
df_sw = pd.read_csv(stats_path + 'df_sw.csv', index_col=0)
df_sw = drop_bads_df(df_sw)
df_sw.rename(columns={'Channel':'Chan'}, inplace=True) 
df_sw = df_sw.groupby('Subject').filter(lambda x: set(x['Mode']) >= {'pn', 'nmes'})

# SO stats
SO_spindle_stats(df_sw, target='Density', oscillation='SO', stage=2)
SO_spindle_stats(df_sw, target='Density', oscillation='SO', stage=3)
SO_spindle_stats(df_sw, target='Density', oscillation='SO', stage=None)

# df_avg_sw = df_sw.set_index('Stage').loc[3].groupby(['Mode','Subject']).mean(numeric_only=True).reset_index()

def group_microarchiteture_stats(df_sw, stage=2, feature='ndPAC'):

    # df_sw = df_sw.groupby('Subject').filter(lambda x: set(x['Mode']) >= {'pn', 'nmes'})
    df_sw = df_sw[df_sw.Stage==stage]
    
    df_nmes = df_sw[(df_sw.Mode == 'nmes')]
    df_nmes = df_nmes.fillna(1)
    df_pn = df_sw[(df_sw.Mode == 'pn')]
    df_pn = df_pn.fillna(1)
    
    unique_channels = df_nmes['Chan'].unique()
    
    # if feature == 'Aperiodic' or feature == 'Offset':
    contrast = (
        df_nmes.pivot_table(index='Subject', 
                            columns='Chan', 
                            values=feature, 
                            aggfunc='mean').reindex(columns=unique_channels, level=1).values  
        -                    
        df_pn.pivot_table(index='Subject', 
                          columns='Chan',
                          values=feature, 
                          aggfunc='mean').reindex(columns=unique_channels, level=1).values
    ) 
        # /  df_pn.pivot_table(index='Subject', 
        #                      columns='Chan',
        #                      values=feature, 
        #                      aggfunc='mean').reindex(columns=unique_channels, level=1).values) * 100
     
        
    
    contrast[1, 17] = 0
    
    # contrast = scipy.stats.zscore(contrast, axis=-1)
                                                                              
    # Adjacency matrix
    adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/LuPf_1_1-epo.fif')
    adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
    
    # Permutation Test
    pval = 0.05
    dof = contrast.shape[0] - 1  # degrees of freedom for the test
    if all(np.sign(contrast.mean(0)) == -1.0):
        cmap = 'Blues'
        #tail = -1
        #thresh = scipy.stats.t.ppf(pval, dof)  # One-tailed test (left tail)
    elif any(np.sign(contrast.mean(0)) == -1.0):
        cmap = 'RdBu_r'
        #tail = 0
        #thresh = scipy.stats.t.ppf(1 - pval / 2, dof)  # two-tailed, t distribution
    else:
        cmap = 'Reds'
        #tail = 1
        #thresh = scipy.stats.t.ppf(1 - pval, dof)

    tail = 0
    thresh = scipy.stats.t.ppf(1 - pval / 2, dof)  
        
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        contrast,                            # numpy array for contrast [n_subjects, n_channels]
        n_permutations=1024,                 # 1000 is the minimum
        threshold=thresh,                    # threshold for cluster formation
        tail=tail,                           # 0 for two-tailed test (1 or -1 for one-tailed)
        n_jobs=-1,                           # increase value to speed up computations
        adjacency=adjacency,                 # sparse matrix for channel adjacency as computed above
        buffer_size=None,
        out_type='mask',                     # returns a mask map instead of indices of sig. points
        seed=1503
    )
    
    # Get significant clusters
    if all(p > 0.05 for p in cluster_p_values):
        print(f'No significant clusters found for {feature} -> {tail} tail Permutation Test')
        df_stats = pd.DataFrame({'Chan': ch_names, 
                                 'Power': contrast.mean(0).squeeze(), 
                                 'T-Stat': t_obs.squeeze(),
                                 'Null' : [False] * len(ch_names),
                                 'Sig': [False] * len(ch_names)})
    else:
        min_p_value = min(cluster_p_values)
        print(f"P-Value: {min_p_value} for {feature} -> {tail} tail Permutation Test")
        df_stats = pd.DataFrame({'Chan': ch_names, 
                                 'Power': contrast.mean(0).squeeze(), 
                                 'T-Stat': t_obs.squeeze(), 
                                 'Null' : [False] * len(ch_names),
                                 'Sig': clusters[np.argmin(cluster_p_values)]})
    
    # Set channel as index
    df_stats = df_stats.set_index("Chan")
    
    # Plot feature and stats
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    if feature == 'Aperiodic':
        unit = '\u0394' + ' Spectral Exponent' 
    else:
        unit = '\u0394' + f' {feature}'
    
    # Plot topomaps
    topoplot(df_stats['Power'], 
             mask=df_stats['Null'], 
             cmap=cmap,
             names=None,
             axes=ax[0],
             cbar_title=unit,
            )
    
    topoplot(df_stats['T-Stat'], 
             mask=df_stats['Sig'], 
             cmap=cmap,
             names=None,
             axes=ax[1],
             mask_params=dict(markersize=6, markerfacecolor='y'), 
             )
    
    if stage == 2:
        fig.suptitle(f'NREM2 {unit} (CLNMES - CLAS)', fontsize=18)
    else:
        fig.suptitle(f'SWS {unit} (CLNMES - CLAS)', fontsize=18)
        
    plt.tight_layout()
    plt.show()
    
#%%
# Plot data
# Loop over each combination of stimulation type and channel
for stim in df_ndpac.Stim.unique():
    for chan in df_ndpac.Target_Chan.unique():
        # Calculate the mean ndPAC for the specified conditions
        data = df_ndpac.groupby(['Mode', 'Session', 'Target_Chan', 'Chan']).mean().loc['nmes', stim, 'Post', chan].ndPAC
        
        # Generate the topoplot for each condition
        yasa.topoplot(data, cmap='Spectral_r', vmin=0.2, vmax=0.3)


# Pivot table to restructure data for difference calculation
df_pivot = df_ndpac.pivot_table(index=['Mode','Subject','Night','Chan'], 
                                columns='Session',
                                values=['SigmaPeakTime', 'PhaseAtSigmaPeak', 'ndPAC'])

# Calculate the difference (post - pre)
df_contrasts = df_pivot.copy()
df_contrasts['SigmaPeakTime_diff'] =  df_pivot['SigmaPeakTime']['Post'] - df_pivot['SigmaPeakTime']['Pre']
df_contrasts['PhaseAtSigmaPeak_diff'] = df_pivot['PhaseAtSigmaPeak']['Post'] - df_pivot['PhaseAtSigmaPeak']['Pre']
df_contrasts['ndPAC_diff'] = df_pivot['ndPAC']['Post'] - df_pivot['ndPAC']['Pre']
df_contrasts.columns = df_contrasts.columns.droplevel('Session')
df_contrasts = df_contrasts[['SigmaPeakTime_diff','PhaseAtSigmaPeak_diff', 'ndPAC_diff']].reset_index()

# Plot ndPAC difference plot
yasa.topoplot(df_contrasts.groupby(['Condition','Chan']).mean().loc['nmes_sham_fz'].ndPAC_diff, 
              cmap = 'Spectral_r', vmin=-0.01, vmax=0.1)
yasa.topoplot(df_contrasts.groupby(['Condition','Chan']).mean().loc['nmes_stim_fz'].ndPAC_diff,
              cmap = 'Spectral_r', vmin=-0.01, vmax=0.1)
yasa.topoplot(df_contrasts.groupby(['Condition', 'Chan']).mean().loc['nmes_sham_c3'].ndPAC_diff,
              cmap = 'Spectral_r', vmin=-0.01, vmax=0.1)
yasa.topoplot(df_contrasts.groupby(['Condition','Chan']).mean().loc['nmes_stim_c3'].ndPAC_diff,
              cmap = 'Spectral_r', vmin=-0.01, vmax=0.1)

#%%
def hrv_sleep_correlations(hrv):
    # Correlation in NREM sleep
    cols = hrv.columns #["hr_mean", "hrv_rmssd", "delta", "theta", "alpha", "sigma"]
    nrem_corr = hrv.xs(2)[cols].corr(method="spearman").round(2).iloc[:2]
    
    # Correlation in REM sleep
    rem_corr = hrv.xs(4)[cols].corr(method="spearman").round(2).iloc[:2]
    
    return nrem_corr, rem_corr