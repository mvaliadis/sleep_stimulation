#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:37:37 2024

@author: administrator
"""

import mne
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import os 
from matplotlib.colors import Normalize
import scipy
import yasa
import statsmodels.api as sm
import neurokit2 as nk 
mne.set_log_level('ERROR')

stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'

## Helper functions
def plot_mode_condition(data, dv, ax, title):
    # Plot paired data 
    pg.plot_paired(data=data,
                   dv=dv,
                   #within='Stim',
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
                'Trial':  np.nan  # or adjust based on available data logic if needed
            }
            new_rows.append(new_row)
    
    # Create DataFrame from new rows
    new_df = pd.DataFrame(new_rows)
    
    # Combine with the original DataFrame and sort
    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=['Subject', 'Night', 'Mode'])
    
    return combined_df   
    
## 1. Phase funcs
def phase_targeting_plot(df):        
    # Create a figure with 2 rows and 4 columns transposed from previous arrangement
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), subplot_kw={'aspect': 'equal'})  # Ensure equal aspect ratio
    axes = axes.flatten()  # Flatten the array of axes for easier iteration
    
    # Iterate over each unique combination and plot
    for i, ((chan, mode, stim), group) in enumerate(df.groupby(['Target_Chan', 'Mode', 'Stim'])):
        if i < 8:  # Ensure we do not try to plot more than 8 plots
            pg.plot_circmean(group['CircMean'], square=True, ax=axes[i], 
                             kwargs_markers={'color': 'tab:blue', 'marker': 'o', 'mfc': 'none', 'ms': 10}, 
                             kwargs_arrow={'width': 0.01, 'head_width': 0.1, 'head_length': 0.1, 'fc': 'tab:red', 'ec': 'tab:red'})
            # Update title formatting here
            title = f"{chan} {mode} {stim}"
            axes[i].set_title(title, pad=30)  # Increase padding to ensure title is not too close to the plot
                       
    # Adjust layout
    #plt.tight_layout()
    plt.show()
    
def phase_targeting_stats(df):
    # Initialize an empty list to store the results
    results = []
    # Iterate over each unique combination and plot
    for i, ((chan, mode, stim), group) in enumerate(df.groupby(['Target_Chan', 'Mode', 'Stim'])):
        print(chan, mode, stim)
        circmean = pg.circ_mean(group['CircMean'])
        print(f"Circular mean: {circmean.round(3)}")
        circstd = scipy.stats.circstd(group['CircMean'])
        print(f"Circular STD: {circstd.round(3)}")
        mean_rvl = pg.circ_r(group['CircMean'])
        print(f"Mean RVL: {mean_rvl.round(3)}")
        print(f"Mean RVL: {group['RVL'].mean().round(3)}")
        z, pval = pg.circ_rayleigh(group['CircMean'])
        print(f'The Rayleigh Z-statistic for non-uniformity of circular data: {z.round(4)} with a p-value : {pval.round(4)}')
        # V-test for non-uniformity of circular data with a specified mean direction
        V, pvalv = pg.circ_vtest(group['CircMean'], dir=0)
        print(f'The V-statistic for non-uniformity of circular data: {V.round(4)} with a p-value : {pvalv.round(4)}')
           
        # Append results to the list
        results.append({
            'Channel': chan,
            'Mode': mode,
            'Stimulation': stim,
            'Circular Mean': circmean,
            'Circular STD': circstd,
            'Mean RVL': mean_rvl,
            'Rayleigh Z': z,
            'Rayleigh p-value': pval,
            'V Statistic': V,
            'V p-value': pvalv
        })

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df
    
## 2. ERP analysis funcs
def plot_erp_consistency(df_tct):
    for col in ['Consistency_RMS', 'Consistency', 'RMS_Trial', 
                'RMS_Evoked', 'ERP']:
        # Plotting
        g = sns.catplot(data=df_tct, kind="box", x="Stim", row="Mode",
                        y=col, col="Target_Chan", palette="pastel")
        
        # Adjusting the plot
        #g.set_titles("{col_name} Channel")
        g.set_axis_labels("", f"{col}", fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.show()

def group_sleep_epoch_stats(gavs, obj, times=(-0.01, 0.01), title=None):
    # prepare adjacency matrix (only takes eeg)
    adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/LuPf_1_1-epo.fif')
    adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
    # extract evoked data 
    eps = [list(obj[1].Epochs)[i].get_data(tmin=times[0], tmax=times[1]).mean(1)*1e3 for i in range(len(obj[1]))]
    contrast = np.concatenate([eps])
        
    # spatial permuation cluster test
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        contrast,                           # numpy array for contrast [n_subjects, n_voltage, n_channels]
        n_permutations=1024,                # 1000 is the minimum
        threshold=dict(start=0, step=0.2),  # TFCE, starting at 0, in 0.2 steps (in t-values)
        tail=0,                             # two-tailed test (1 or -1 for one-tailed)
        n_jobs=-1,                          # increase value to speed up computations
        adjacency=adjacency,                # sparse matrix for channel adjacency as computed above
        buffer_size=None,
        out_type='mask',                    # returns a mask map instead of indices of sig. points
        seed=1503
    )
    
    df_stats = pd.DataFrame({'Chan': ch_names, 'T-Stat': t_obs, 
                             'Pval': cluster_p_values}) 
    df_stats = df_stats.set_index("Chan")
    df_stats['Sig'] = (df_stats['Pval'] < 0.05)

    

    # Plot
    unit='mV/m2'
    fig, ax = plt.subplots(1, 2, figsize=(8,6))
    im1, _ = mne.viz.plot_topomap(contrast.mean(0), 
                                  pos=gavs.info,
                                  axes=ax[0], show=0, cmap='RdBu_r',
                                  names=None, show_names=False)
    cbar1 = fig.colorbar(im1, fraction=0.05, ax=ax[0])   
    cbar1.ax.set_ylabel(unit, rotation=270)
    plt.tight_layout()
    im2, _ = mne.viz.plot_topomap(t_obs, 
                                  pos=gavs.info, mask=df_stats['Sig'],
                                  axes=ax[1], show=0, cmap='RdBu_r',
                                  names=None, show_names=False, 
                                  mask_params=dict(markersize=8, markerfacecolor='y'))
    cbar2 = fig.colorbar(im2, fraction=0.05, ax=ax[1])   
    cbar2.ax.set_ylabel('t-stat', rotation=270)
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

def double_contrast_erp(df_evoked):
    # Step 1: Reset the 'Night' from the index and aggregate if needed
    nmes_data = df_evoked.xs('nmes', level='Mode').reset_index('Night', drop=True)
    pn_data = df_evoked.xs('pn', level='Mode').reset_index('Night', drop=True)

    # Step 2: Form double contrast dataframe
    double_contrast_df = nmes_data[['Contrast_c3','Contrast_fz']].join(pn_data[['Contrast_c3', 'Contrast_fz']], 
                                                                       lsuffix='_nmes', rsuffix='_pn', 
                                                                       how='outer')

    # Step 3: Drop nans
    dc_df = double_contrast_df.dropna()

    # Step 4: Convert dataframes to numpy arrays
    C3_evokeds = [mne.combine_evoked([list(dc_df['Contrast_c3_nmes'])[i], 
                                      list(dc_df['Contrast_c3_pn'])[i]],
                                     weights=[1, -1]) for i in range(len(dc_df))]

    Fz_evokeds = [mne.combine_evoked([list(dc_df['Contrast_fz_nmes'])[i], 
                                      list(dc_df['Contrast_fz_pn'])[i]],
                                     weights=[1, -1]) for i in range(len(dc_df))]

    return C3_evokeds, Fz_evokeds

def erp_stats(evoked, length=[0, 2], thresh_method='ttest'): 
    sns.set_theme(style="white") 
    # C3_evokeds, Fz_evokeds = double_contrast_erp(df_evoked)
    # TODO: Add double contrasts to stats below...   
    for con in ['Contrast_c3', 'Contrast_fz']:
        for mode, evoked in df_evoked[con].groupby('Mode'):
            target_chan = con.split('_')[-1].upper()
            evk = evoked.reset_index()[con].copy()
            gav = mne.grand_average(list(evk)).crop(tmin=length[0], tmax=length[1])
            evk_np = np.concatenate([np.expand_dims(evk[idx].copy().crop(length[0], tmax=length[1]).get_data(), 0)*1e3 
                                     for idx in range(len(evk))], 0)
            contrast = np.swapaxes(evk_np, 2, 1)
        
            # prepare adjacency matrix (only takes eeg)
            adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/LuPf_1_1-epo.fif')
            adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
            
            # compute threshold
            if thresh_method == 'ttest':
                pval = 0.05  # arbitrary
                df = contrast.shape[0] - 1  # degrees of freedom for the test
                thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
            elif thresh_method == 'tfce': # interpretation very complex....
                thresh = dict(start=0, step=0.2)
                
            # do stats 
            t_obs, clusters, cluster_p_values, h0 = mne.stats.spatio_temporal_cluster_1samp_test(
                contrast,                           # numpy array for contrast [n_subjects, n_voltage, n_channels]
                n_permutations=1024,                # 1000 is the minimum
                threshold=thresh,                   # TFCE, starting at 0, in 0.2 steps (in t-values)
                tail=0,                             # two-tailed test (1 or -1 for one-tailed)
                n_jobs=-1,                          # increase value to speed up computations
                adjacency=adjacency,                # sparse matrix for channel adjacency as computed above
                buffer_size=None,
                out_type='mask',                    # returns a mask map instead of indices of sig. points
                seed= 1503
                ) 
            
            # return mask
            mask = cluster_p_values < 0.05
            
            if mask.mean() == 0:
                pass
            else:
                # plot
                selections = mne.channels.make_1020_channel_selections(gav.info, midline="12z")
                fig, axes = plt.subplots(nrows=3, figsize=(20, 10), dpi=100)
                axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
                gav.plot_image(mask=clusters[np.argsort(cluster_p_values)[0]].T, axes=axes,
                               show_names="all", #mask_cmap='Spectral_r', cmap='Spectral_r',
                               titles=None, colorbar=True, mask_alpha=0.80, 
                               group_by=selections, show=False) 
                plt.tight_layout()
            
                # pararameters
                times = np.asarray([0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
                ts_args = dict(spatial_colors = True, gfp=True, time_unit='s')
                topomap_args = dict(outlines = 'head', time_unit='s', time_format = "%0.2f s")
                if mode == 'pn':
                    title = f'CLAS Evoked Difference ({target_chan})'
                elif mode == 'nmes':
                    title = f'CLNMES Evoked Difference ({target_chan})'
                    
                gav.plot_joint(times=times, title=title, 
                               ts_args=ts_args, topomap_args=topomap_args)
            
def stats_targeting(gav_double, double_contrast, title=' '):
    gav = gav_double.loc[6].GAV
    # prepare adjacency matrix, must load non-csd mne epochs info
    adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/LuPf_1_1-epo.fif')
    adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
         
    # Cluster test 
    # spatial permuation cluster test
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        double_contrast,                    # numpy array for contrast [n_subjects, n_voltage, n_channels]
        n_permutations=1024,                # 1000 is the minimum
        threshold=dict(start=0, step=0.2),  # TFCE, starting at 0, in 0.2 steps (in t-values)
        tail=0,                             # two-tailed test (1 or -1 for one-tailed)
        n_jobs=-1,                          # increase value to speed up computations
        adjacency=adjacency,                # sparse matrix for channel adjacency as computed above
        buffer_size=None,
        out_type='mask',                    # returns a mask map instead of indices of sig. points
        seed=1503
    )
    
    df_stats = pd.DataFrame({'Chan': ch_names, 'T-Stat': t_obs, 
                             'Pval': cluster_p_values}) 
    df_stats = df_stats.set_index("Chan")
    df_stats['Sig'] = (df_stats['Pval'] < 0.05)
    unit='uV'
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(8,6))
    im1, _ = mne.viz.plot_topomap(double_contrast.mean(0), 
                                  pos=gav.info,
                                  axes=ax[0], show=0, cmap='RdBu_r',
                                  names=None)
    cbar1 = fig.colorbar(im1, fraction=0.05, ax=ax[0])   
    cbar1.ax.set_ylabel(unit, rotation=270)
    #plt.tight_layout()
    im2, _ = mne.viz.plot_topomap(t_obs, 
                                  pos=gav.info, mask=df_stats['Sig'],
                                  axes=ax[1], show=0, cmap='RdBu_r',
                                  names=None,
                                  mask_params=dict(markersize=8, markerfacecolor='y'))
    cbar2 = fig.colorbar(im2, fraction=0.05, ax=ax[1])   
    cbar2.ax.set_ylabel('T-stat', rotation=270)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

def roi_ttest(df_evoked):
    # Relevant columns and modes
    columns = ['Contrast_c3', 'Contrast_fz']
    modes = ['pn', 'nmes']  
    
    # Mapping of column labels to plot labels
    label_map = {
        'pn': {
            'Contrast_c3': 'CLAS (C3)',
            'Contrast_fz': 'CLAS (Fz)'
        },
        'nmes': {
            'Contrast_c3': 'CLNMES (C3)',
            'Contrast_fz': 'CLNMES (Fz)'
        }
    }
    
    # Initialize the plot
    
    
    # Loop over modes and relevant columns
    for mode in modes:
        plt.figure(figsize=(10, 5))
        for column in columns:
            data = list(df_evoked.set_index('Mode').loc[mode][column])
            data_gav = mne.grand_average(data)
            subject_data = np.concatenate([np.expand_dims(data[i].get_data(tmin=-0.01, tmax=2, 
                                                                           picks=['F1', 'Fz', 'F2']).mean(0) * 1e3, 0) 
                                           for i in range(len(data))], axis=0)
            threshold = 4
            T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
                [subject_data, subject_data2],
                n_permutations=1000,
                threshold=threshold,
                tail=1,
                n_jobs=None,
                out_type="mask",
            )
            
            fc = data_gav.get_data(tmin=-0.1, tmax=2, picks=['F1', 'Fz', 'F2']).mean(0) * 1e3
            mc = data_gav.get_data(tmin=-0.1, tmax=2, picks=['C1', 'C3', 'C5']).mean(0) * 1e3
            
            times = data_gav.copy().crop(tmin=-0.1, tmax=2).times
            
            plt.plot(times, fc, label=f'frontal response - {label_map[mode][column]}')
            plt.plot(times, mc, label=f'motor response - {label_map[mode][column]}')
    
        # Add legend and show plot
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('CSD Voltage')
        plt.title('Evoked Responses')
        plt.show()

## 3. TFR funcs
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

def plot_max_sig(t_obs, spindle_frange, tmin, tmax, ch_names, aggregated_cluster_map, spindle=True):
    # Create mask for significant clusters
    cluster_df = pd.DataFrame({'Chan' : ch_names,
                               'T-Values': t_obs[spindle_frange, tmin:tmax, :].mean(axis=(0, 1)),
                               'pval': aggregated_cluster_map.mean(axis=(0, 1)) > 0})
    cluster_df = cluster_df.set_index('Chan')
    
    # Identify significant clusters
    cluster_df['sig'] = (cluster_df['pval'] == True)
    
    # Identify maximum effect size within significant clusters
    cluster_df['max_effect_size'] = cluster_df.apply(
        #lambda row: row['T-Values'] == cluster_df['T-Values'][cluster_df['sig']].max() if row['sig'] else False, axis=1
        lambda row: row['T-Values'] == cluster_df['T-Values'].max(), axis=1
    )
    
    if spindle==True:
        title = 'Spindles at 1s ± .25s'
    else:
        title = 'Theta at 0.5s ± .25s'
        
    # Create a topoplot using yasa
    fig, ax = plt.subplots()
    topoplot(cluster_df['T-Values'], 
             names=None, 
             #mask=cluster_df['sig'], 
             #mask_params=dict(markersize=5, markerfacecolor='y'),
             axes=ax,
             title=title)
    
    # Add cross to indicate maximal effect size at correct sensor location
    topoplot(cluster_df['T-Values'], 
            names=None,
            mask=cluster_df['max_effect_size'], 
            show_cbar=False,
            mask_params=dict(markersize=10, marker='o', 
                             markerfacecolor='w'),
            axes=ax,
            title=title)
    plt.tight_layout()
    
def tfr_stats_new(df_tfr, plot=True):
    df_tfr_ = df_tfr.copy()
    
    df_tfr_contrast, df_tfr_contrast_double, df_best_contrasts = create_tfr_contrasts(df_tfr_)
    
    df_tfr_contrast = df_tfr_contrast.groupby(['Subject']).filter(lambda x: len(x) == 4)
    
    #plot_tfr_gav(df_tfr_contrast, trange=(-0.5, 2))
    
    # prepare adjacency matrix (only takes eeg)
    adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/LuPf_1_1-epo.fif')
    adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
    
    for (mode, target), df_ in df_tfr_contrast.groupby(['Mode','Target']):
        print(mode, target)
        data = list(df_['TFR Contrast'])
        
        # Double contrast
        # data = list(df_tfr_contrast_double[df_tfr_contrast_double.Target=='c3']['Double Contrast'])
        # data = list(df_tfr_contrast_double[df_tfr_contrast_double.Target=='fz']['Double Contrast'])
        
        # Best contrast
        # data = list(df_best_contrasts['Best Contrasts'])
        
        X = np.swapaxes(np.concatenate([np.expand_dims(x.crop(-0.5, 2).data, -1) 
                                        for x in data], axis=-1), -1, 0)

        # Create variables
        times = data[0].times
        freqs = data[0].freqs
        
        # Theta variables
        tmin_theta, tmax_theta = data[0].time_as_index([.25, .75])
        theta_frange = np.where(np.logical_and(freqs >= 4, freqs <= 8))[0]
        # theta_power = X[:, theta_frange, tmin_theta:tmax_theta, :].mean(axis=(0, 1, 2))
        # # Plot topomap
        # yasa.topoplot(pd.Series(theta_power, index=data[0].ch_names))
        
        # Spindle variables
        tmin, tmax = data[0].time_as_index([.75, 1.25])
        spindle_frange = np.where(np.logical_and(freqs >= 12, freqs <= 17))[0]
        # spindle_power = X[:, spindle_frange, tmin:tmax, :].mean(axis=(0, 1, 2))
        # # Plot topomap
        # yasa.topoplot(pd.Series(spindle_power, index=data[0].ch_names))
        
        # our data at each observation is of shape frequencies × times × channels
        tfr_adjacency = mne.stats.combine_adjacency(X.shape[1], X.shape[2], adjacency)
    
        # Arbitrary threshold
        tfr_threshold = 5.0
    
        # run cluster based permutation analysis
        t_obs, clusters, cluster_p_values, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
            X,
            n_permutations=1024,
            threshold=tfr_threshold,
            tail=0,
            n_jobs=-1,
            buffer_size=None,
            adjacency=tfr_adjacency,
            out_type='mask',      
            seed=42,
        )
        
        # Boolean mask for significant p-values
        significant_clusters_mask = cluster_p_values < 0.05
        
        # Get the indices of clusters with significant p-values
        significant_cluster_indices = np.where(significant_clusters_mask)[0]
        
        # Use these indices to select the corresponding clusters
        significant_clusters = np.take(clusters, significant_cluster_indices, axis=0)
        
        # Combine all the significant clusters into a single binary mask using logical OR
        aggregated_cluster_map = np.any(significant_clusters, axis=0)
        
        # Plot
        if plot: 
            if mode == 'pn':
                name = 'CLAS'
            elif mode == 'nmes':
                name = 'CLNMES'
                
            fig, ax = plt.subplots(1, figsize=(10, 8))
            fig.suptitle(f'{name} ERSP ({target.capitalize()} Target Contrast)', fontsize=18)
            vmax = np.percentile(t_obs, 97.5)
            vmin = np.percentile(t_obs, 2.5)
            
            # Define the channels for different ROIs
            if target == 'fz':
                cs = ['F1', 'Fz', 'F2']
            elif target == 'c3':
                cs = ['C5', 'C3', 'CP1', 'C1']
                
            roi = [ch_names.index(item) for item in cs]
            
            # Plot the non-significant background data (average across ROI channels)
            CM = ax.pcolormesh(
                times,
                freqs,
                t_obs[:, :, roi].mean(-1),  # Averaging the t-values across the selected channels
                shading='gouraud',
                cmap='RdBu_r',
                rasterized=True,
                antialiased=True,
                alpha=0.9,
                vmin=vmin,
                vmax=vmax
            )
            
            # Overlay significant clusters using logical OR across ROI channels
            roi_cluster_map = np.any(aggregated_cluster_map[:, :, roi], axis=-1)
            
            ax.pcolormesh(
                times,
                freqs,
                roi_cluster_map,
                shading='gouraud',
                cmap='Greys',
                rasterized=True,
                antialiased=True,
                alpha=0.25,  # Increase transparency to highlight the underlying data
                vmin=0,
                vmax=1
            )
            
            # Add contour lines to highlight cluster boundaries
            ax.contour(
                times,
                freqs,
                roi_cluster_map,
                levels=[0.5],  # Adjust based on cluster significance threshold
                colors='black',
                linewidths=1.5
            )
            
            # Customize the plot labels
            ax.set_ylabel('Frequency (Hz)', fontsize=16)
            ax.set_xlabel('Time (s)', fontsize=16)
            
            # Single unified colorbar
            cbar = plt.colorbar(CM, ax=ax)
            cbar.set_label('T-Values', rotation=270, fontsize=16, labelpad=20)
            
            # Set tighter limits for the x and y axes if needed
            ax.set_xlim([times[0], times[-1]])
            ax.set_ylim([freqs[0], freqs[-1]])
            
            # Enhance the layout and display the plot
            plt.tight_layout()
            plt.show()
            
            # Plot Spindle and Theta power topomaps in t-units          
            plot_max_sig(t_obs, spindle_frange, tmin, tmax, ch_names, aggregated_cluster_map, spindle=True)
            plot_max_sig(t_obs, theta_frange, tmin_theta, tmax_theta, ch_names, aggregated_cluster_map, spindle=False)

def max_stat_tfr_plot(T_obs_plot, adj_epochs):
    # Create a mask for the significant sensors
    significant_sensors_mask = np.any(np.any(~np.isnan(T_obs_plot), axis=2), axis=1)
    
    # Average the significant t-values across time for each sensor
    mean_t_per_sensor = np.nanmean(T_obs_plot, axis=(1, 2))
            
    # Create the topomap
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")
    
    t_evoked = mne.EvokedArray(np.expand_dims(mean_t_per_sensor,1), 
                               adj_epochs.copy().pick('eeg').info, tmin=0)
    t_evoked.plot_topomap(times=0, 
                          axes=ax_topo, 
                          show=False,
                          colorbar=False,
                          mask=np.expand_dims(significant_sensors_mask,1),
                          cmap='Reds',
                          sensors=True)
    image = ax_topo.images[0]
    
    # remove the title that would otherwise say "0.000 s"
    ax_topo.set_title("")
    
    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)
    
    # add axes for colorbar
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        "Averaged F-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
    )
    
    
    # Plot the TFR
    # Max t-value across channels for each time-frequency point
    max_t_per_tf = np.nanmax(T_obs_plot, axis=0)
    
    # Generate the TFR plot
    fig, ax = plt.subplots(1, figsize=(10, 4))
    
    # We will only plot the positive t-values since the plot you provided does not show negative values
    pos = max_t_per_tf > 0
    im = ax.imshow(max_t_per_tf * pos, aspect='auto', origin='lower', extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap='Reds')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Cluster #1, T-value Spectrogram (max over channels)')
    fig.colorbar(im, ax=ax)
        
def create_tfr_contrasts(df_tfr):
    gavs_dict = []
    for (mode, target, subject), tfr in df_tfr.groupby(['Mode','Target_chan','Subject']):
        stim = tfr[tfr.Stim=='stim'].reset_index(drop=True).TFR[0]
        sham = tfr[tfr.Stim=='sham'].reset_index(drop=True).TFR[0]
        contrast = stim.__sub__(sham)
        
        dict_gav = {
            'Subject' : subject,
            'Mode': mode, 
            'Target': target,
            'TFR Contrast' : contrast,
        }
        
        # Append the dictionary to the list
        gavs_dict.append(dict_gav) 
    
    df = pd.DataFrame(gavs_dict)

    gavs_dict_con = []
    for (subj, target), tfr_double in df.groupby(['Subject', 'Target']):
        if len(tfr_double) > 1:
            nmes = tfr_double.set_index(['Mode']).loc['nmes']['TFR Contrast']
            pn = tfr_double.set_index(['Mode']).loc['pn']['TFR Contrast']
            dc = nmes.__sub__(pn)
            
            dict_gav2 = {
                'Subject' : subj,
                'Target': target,
                'Double Contrast' : dc
                }
                
            # Append the dictionary to the list
            gavs_dict_con.append(dict_gav2) 
        else:
            pass
    
    df2 = pd.DataFrame(gavs_dict_con)
    
    # Third, create a contrast between nmes C3 and pn Fz
    gavs_dict_final = []
    for subj, tfr_triplet in df.groupby('Subject'):
        if len(tfr_triplet)==4:
            nmes_c3 = tfr_triplet[np.logical_and(tfr_triplet.Target == 'c3',
                                  tfr_triplet.Mode == 'nmes')]['TFR Contrast'].values[0]
            pn_fz = tfr_triplet[np.logical_and(tfr_triplet.Target == 'fz',
                                  tfr_triplet.Mode == 'pn')]['TFR Contrast'].values[0]
            combined_contrast = nmes_c3.__sub__(pn_fz)
            
            dict_gav3 = {
                'Subject': subj,
                'Best Contrasts': combined_contrast
            }
            gavs_dict_final.append(dict_gav3)
    
    df3 = pd.DataFrame(gavs_dict_final)
    
    # gavs_dict_con2 = []    
    # for subj, tfr_ in df2.groupby('Subject'):
    #     if len(tfr_) == 4:
    #         print(subj)
    #         nmes = tfr_double.set_index(['Mode']).loc['nmes']['TFR Contrast']
    #         pn = tfr_double.set_index(['Mode']).loc['pn']['TFR Contrast']
    #         dc = nmes.__sub__(pn)
            
    #         dict_gav3 = {
    #             'Subject' : subj,
    #             'Target': target,
    #             'Double Contrast' : dc
    #             }
                
    #         # Append the dictionary to the list
    #         gavs_dict_con2.append(dict_gav3) 
    #     else:
    #         pass
    
    # df4 = pd.DataFrame(gavs_dict_con2)
        
    return df, df2, df3   
  
def plot_tfr_gav(df, trange=(-0.5, 2)):
    for (mode, target), tfr_gav in df.groupby(['Mode','Target']):
        
        gavs = list(tfr_gav['TFR Contrast'])
        
        if target == 'c3':
            coi = 'C3'
        else:
            coi = 'Fz'
            
        Sxx_ = np.concatenate([gavs[idx].copy().crop(tmin=trange[0], tmax=trange[1]).pick(coi).data 
                              #[gavs[idx].copy().crop(tmin=trange[0], tmax=trange[1]).data.mean(0, keepdims=True)#.pick('Fz').data 
                               for idx in range(len(gavs))], 0)
        con = np.concatenate([np.expand_dims(gavs[idx].copy().crop(tmin=trange[0], tmax=trange[1]).data, 0) 
                              for idx in range(len(gavs))], 0)
        times = gavs[0].copy().crop(tmin=trange[0], tmax=trange[1]).times
        freqs = gavs[0].freqs
        
        fig, ax = plt.subplots()
        vmin, vmax = np.percentile(Sxx_, [0 + 2.5, 100 - 2.5])
        norm = Normalize(vmin=vmin, vmax=vmax)
        CM = ax.pcolormesh(times, freqs, Sxx_.mean(0).squeeze(), 
                           shading='gouraud', cmap='Spectral_r', 
                           rasterized = True, antialiased=True, 
                           norm=norm
                           )
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        cbar = plt.colorbar(CM, ax=ax)
        cbar.set_label(f'{mode} {target} (dB)')
        plt.tight_layout()
                   
        # Plotting with MNE      
        # mne.grand_average(gavs).plot_joint(baseline=None, picks=coi, 
        #                                    cmap='Spectral_r', vlim=(vmin,vmax),
        #                                    tmin=-0.5, tmax=2, mode='mean',
        #                                    timefreqs={(1, 13): (1, 4)})
        mne.grand_average(gavs).plot_joint(baseline=None, picks=coi, 
                                           cmap='Spectral_r', vlim=(vmin,vmax),
                                           tmin=trange[0], tmax=trange[1],
                                           mode='mean',
                                           timefreqs={(.75, 14): (1, 4)})

def tfr_stats(contrast, coi, target):
    # prepare adjacency matrix (only takes eeg)
    adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/LuPf_1_1-epo.fif')
    adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
    adj4D = mne.stats.combine_adjacency(adjacency,
                                        contrast.shape[2], contrast.shape[3])
    
    # compute threshold
    pval = 0.05  # arbitrary
    df = contrast.shape[0] - 1  # degrees of freedom for the test
    thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
         
    # do stats 
    t_obs, clusters, cluster_p_values, h0 = mne.stats.spatio_temporal_cluster_1samp_test(
        contrast,                 # numpy array for contrast [n_subjects, n_chans, n_freqs, n_times]
        n_permutations=1024,      # 1000 is the minimum
        threshold=thresh,         # t-threshold from above
        tail=0,                   # two-tailed test 
        n_jobs=-1,
        adjacency=adj4D,          # sparse matrix for channel adjacency as computed above
        buffer_size=None,
        out_type='mask',          # returns a mask map instead of indices of sig. points
        seed= 1503
        ) 
    
    print(t_obs.shape)

    # return mask
    #mask = cluster_p_values < 0.05
    print(f'Lowest cluster p-value {cluster_p_values.min()}')
            
    # Initialize Mask
    T_obs_plot = np.nan * np.ones_like(t_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = t_obs[c]
        
    if coi == 'C3' or coi == 'Fz':
        #####
        fig, ax = plt.subplots(1, figsize=(10, 10), layout="constrained")
        vmax = np.percentile(t_obs, 97.5)
        vmin = np.percentile(t_obs, 2.5)
        CM = ax.pcolormesh(times, freqs, t_obs[ch_names.index(coi), :, :],
                           shading='gouraud', cmap=plt.cm.gray,
                           rasterized = True, antialiased=True, alpha=.25,
                           vmin=vmin, vmax=vmax)
       
        fig.suptitle(f'ERSP ({target.upper()} Target Contrast)', fontsize=18)
        ax.set_ylabel('Frequency (Hz)', fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=16)
        cbar = plt.colorbar(CM, ax=ax)
        cbar.set_label('T-Values', rotation=270, fontsize=16)
        
        CM2 = ax.pcolormesh(times, freqs, 
                            T_obs_plot[ch_names.index(coi)],
                            shading='gouraud', cmap=plt.cm.Spectral_r,
                            rasterized=True, antialiased=True, alpha=1,
                            vmin=vmin, vmax=vmax)
        plt.colorbar(CM2, ax=ax)
    else:
        #####
        fig, ax = plt.subplots(1, figsize=(10, 10), layout="constrained")
        vmax = np.percentile(t_obs, 97.5)
        vmin = np.percentile(t_obs, 2.5)
        CM = ax.pcolormesh(times, freqs, t_obs.mean(0),
                           shading='gouraud', cmap=plt.cm.gray,
                           rasterized = True, antialiased=True, alpha=.25,
                           vmin=vmin, vmax=vmax)
       
        fig.suptitle(f'ERSP ({target.upper()} Target Contrast)', fontsize=18)
        ax.set_ylabel('Frequency (Hz)', fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=16)
        cbar = plt.colorbar(CM, ax=ax)
        cbar.set_label('T-Values', rotation=270, fontsize=16)
        
        CM2 = ax.pcolormesh(times, freqs, 
                            T_obs_plot.mean(0),
                            shading='gouraud', cmap=plt.cm.Spectral_r,
                            rasterized=True, antialiased=True, alpha=1,
                            vmin=vmin, vmax=vmax)
        plt.colorbar(CM2, ax=ax)

## 4. ndPAC funcs
def ndPAC_stats(df, session='Post'):
    # Filter the data to include only post-session measurements
    df_post = df[df['Session'] == session]
    
    for mode in df_post.Mode.unique():
        # Separate data by Target_Chan and Stim type
        c3_stim = df_post[(df_post['Target_Chan'] == 'c3') & (df_post['Stim'] == 'stim') & (df_post['Mode'] == mode)]
        c3_sham = df_post[(df_post['Target_Chan'] == 'c3') & (df_post['Stim'] == 'sham') & (df_post['Mode'] == mode)]
        fz_stim = df_post[(df_post['Target_Chan'] == 'fz') & (df_post['Stim'] == 'stim') & (df_post['Mode'] == mode)]
        fz_sham = df_post[(df_post['Target_Chan'] == 'fz') & (df_post['Stim'] == 'sham') & (df_post['Mode'] == mode)]
    
        # prepare adjacency matrix, must load non-csd mne epochs info
        adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/LuPf_1_1-epo.fif')
        adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
        
        # Calculate the contrasts (stim - sham)
        # c3_stim_ = c3_stim.groupby(['Subject','Chan']).mean().ndPAC.unstack().reindex(columns=ch_names).to_numpy() 
        # c3_sham_ = c3_sham.groupby(['Subject','Chan']).mean().ndPAC.unstack().reindex(columns=ch_names).to_numpy()
        # contrast_c3 = ((c3_stim_ - c3_sham_)/(c3_sham_))*100
        contrast_c3 = (c3_stim.groupby(['Subject','Chan']).mean().ndPAC.unstack().reindex(columns=ch_names).to_numpy() -
                       c3_sham.groupby(['Subject','Chan']).mean().ndPAC.unstack().reindex(columns=ch_names).to_numpy())
        contrast_fz = (fz_stim.groupby(['Subject','Chan']).mean().ndPAC.unstack().reindex(columns=ch_names).to_numpy() -
                       fz_sham.groupby(['Subject','Chan']).mean().ndPAC.unstack().reindex(columns=ch_names).to_numpy())

        # Cluster test 
        # spatial permuation cluster test
        pval = 0.05  # arbitrary
        dof = contrast_c3.shape[0] - 1  # degrees of freedom for the test
        thresh = scipy.stats.t.ppf(1 - pval / 2, dof)  # two-tailed, t distribution
        
        t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
            contrast_c3,                        # numpy array for contrast [n_subjects, n_voltage, n_channels]
            n_permutations=1024,                # 1000 is the minimum
            threshold=thresh, 
            tail=0,                             # two-tailed test (1 or -1 for one-tailed)
            n_jobs=-1,                          # increase value to speed up computations
            adjacency=adjacency,                # sparse matrix for channel adjacency as computed above
            buffer_size=None,
            out_type='mask',                    # returns a mask map instead of indices of sig. points
            seed=1503
        )
        
        df_stats = pd.DataFrame({'Chan': ch_names,
                                 'T-Stat': t_obs, 
                                 'Pval': clusters[cluster_p_values.argmin()],
                                #'Pval': cluster_p_values,
                                })
        df_stats = df_stats.set_index("Chan")
        df_stats['Sig'] = (df_stats['Pval'] > 0.05)
    
        yasa.topoplot(df_stats['T-Stat'], mask=df_stats['Sig'])
        
        print(cluster_p_values.min())
        
        plt.show()
        
## 5. ERPAC funcs
def plot_ergpac_diff(df_erpac, times=(-2, 2), do_stats=False):
    erpacs = []
    df = df_erpac.copy().groupby(['Subject']).filter(lambda x: len(x) == 8)
    for (mode, target), df_ in df.groupby(['Mode','Target_Chan']):
        print(mode, target)
        
        df_stim = df_.set_index('Stim').loc['stim'].reset_index(drop=True)
        df_sham = df_.set_index('Stim').loc['sham'].reset_index(drop=True)
        
        erpac_stim = np.concatenate([np.expand_dims(df_stim.iloc[i].ERPAC, 0) 
                                     for i in range(len(df_stim))], 0)
        erpac_sham = np.concatenate([np.expand_dims(df_sham.iloc[i].ERPAC, 0) 
                                     for i in range(len(df_sham))], 0)
        
        erpac = erpac_stim - erpac_sham
        erpacs.append([mode, target, erpac])
        
        df_evoked = pd.read_pickle(os.path.join(stats_path, 'df_evokeds.p'))
        df_evoked = drop_bads_df(df_evoked)
        df_evoked.set_index(['Mode','Subject'], inplace=True)
        df_evoked = df_evoked.groupby(['Subject']).filter(lambda x: len(x) == 2)
        
        if target == 'fz':
            tar = 'Fz'
        else:
            tar = 'C3'
            
        data = list(df_evoked.loc[mode][f'Contrast_{target}'])
        data_sw = np.concatenate([data[idx].copy().filter(None, 1.5).get_data(tar, tmin=times[0], tmax=times[1])*1e3 for idx in range(len(data))])
        data_sp = np.concatenate([data[idx].copy().filter(11.75, 16.25).get_data(tar, tmin=times[0], tmax=times[1])*1e3 for idx in range(len(data))])
        
        tmin, tmax = data[0].time_as_index(times)
        
        cond = mode, target
        plot_ergpac(erpac, data_sw.mean(0), data_sp.mean(0), cond, times)
        
        if do_stats:
            print('')
            erpac_stats(erpac, tmin, tmax, tar, mode, times)
            # erpac_stats(erpacs[0][2] - erpacs[2][2], tmin, tmax, 'C3', 'DC', times)
            # erpac_stats(erpacs[1][2] - erpacs[3][2], tmin, tmax, 'Fz', 'DC', times)
    
def plot_ergpac(erpac, data_sw, data_sp, cond, times=(-2, 2)):
    freqs = np.arange(4, 24, 0.5)
    times = np.arange(times[0], times[1], 1/128)
    
    # ERGPAC Plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    im = plt.imshow(erpac.mean(0),
                    aspect='auto', 
                    cmap="Spectral_r", 
                    origin='upper',
                    interpolation="gaussian", 
                    vmin=-0.01, 
                    vmax=0.02, 
                    extent=[times[0], times[-1], freqs[-1], freqs[0]])
    
    plt.gca().invert_yaxis()
    
    fig.suptitle(f"{cond}")
    plt.xlabel("Time from stim onset (s)")
    plt.ylabel("Frequency (Hz)")
    plt.axvline(0, ls=":", lw=1.5, color="k")
    
    cb = plt.colorbar(im, shrink=0.7, pad=0.05, aspect=20)
    cb.set_label("Coupling (z-score)")
    cb.outline.set_visible(False)
    
    ax_sw = ax.twinx()
    ax_sw.plot(times, data_sw, color="k", lw=3)
    ax_sw.plot(times, data_sp*5 + 2, color="k", lw=1.5)
    ax_sw.set_yticks([]);
   
def erpac_stats(erpac, tmin, tmax, target, mode, times=(-2, 2)):
    X = erpac[:,:,tmin:tmax]

    # Create variables
    times = np.linspace(times[0], times[1], X.shape[-1])
    freqs = np.arange(4.25, 24.75, 0.5)
        
    # our data at each observation is of shape frequencies × times
    tfr_adjacency = mne.stats.combine_adjacency(X.shape[1], X.shape[2])

    # Arbitrary threshold
    tfr_threshold = 2.0

    # run cluster based permutation analysis
    t_obs, clusters, cluster_p_values, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
        X,
        n_permutations=1024,
        threshold=tfr_threshold,
        tail=0,
        n_jobs=-1,
        buffer_size=None,
        adjacency=tfr_adjacency,
        out_type='mask',      
        seed=42,
    )
    
    # Boolean mask for significant p-values
    significant_clusters_mask = cluster_p_values < 0.05
    
    # Get the indices of clusters with significant p-values
    significant_cluster_indices = np.where(significant_clusters_mask)[0]
    
    # Use these indices to select the corresponding clusters
    significant_clusters = np.take(clusters, significant_cluster_indices, axis=0)
    
    # Combine all the significant clusters into a single binary mask using logical OR
    aggregated_cluster_map = np.any(significant_clusters, axis=0)
    
    # Plot
    if mode == 'pn':
        name = 'CLAS'
    elif mode == 'nmes':
        name = 'CLNMES'
    else:
        name = 'DC'
        
    fig, ax = plt.subplots(1, figsize=(10, 8))
    fig.suptitle(f'{name} ERGPAC ({target.capitalize()} Target Contrast)', fontsize=18)
    vmax = np.percentile(t_obs, 97.5)
    vmin = np.percentile(t_obs, 2.5)
    
    # Plot the non-significant background data (average across ROI channels)
    CM = ax.pcolormesh(
        times,
        freqs,
        t_obs,  # Averaging the t-values across the selected channels
        shading='gouraud',
        cmap='RdBu_r',
        rasterized=True,
        antialiased=True,
        alpha=0.9,
        vmin=vmin,
        vmax=vmax
    )
    
    # Overlay significant clusters  
    ax.pcolormesh(
        times,
        freqs,
        aggregated_cluster_map,
        shading='gouraud',
        cmap='Greys',
        rasterized=True,
        antialiased=True,
        alpha=0.25,  # Increase transparency to highlight the underlying data
        vmin=0,
        vmax=1
    )
    
    # Add contour lines to highlight cluster boundaries
    ax.contour(
        times,
        freqs,
        aggregated_cluster_map,
        levels=[0.5],  # Adjust based on cluster significance threshold
        colors='black',
        linewidths=1.5
    )
    
    # Customize the plot labels
    ax.set_ylabel('Frequency (Hz)', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=16)
    
    # Single unified colorbar
    cbar = plt.colorbar(CM, ax=ax)
    cbar.set_label('T-Values', rotation=270, fontsize=16, labelpad=20)
    
    # Set tighter limits for the x and y axes if needed
    ax.set_xlim([times[0], times[-1]])
    ax.set_ylim([freqs[0], freqs[-1]])
    
    # Enhance the layout and display the plot
    plt.tight_layout()
    plt.show()
    
#%%
## Stats Pipeline

# 1. Compare stimulation targeting phases 
# 2. Compare epoch-wise power & aperiodic params
# 3. Compare time reversed multivariate Granger causality 
# 4. Compare topographic consistency of stimulation epoch
# 5. Compare PCIst for stimulation contrasts
# 6. Compare epoch TFRs
# 7. Compare epoch SO-Spindle coupling (ndPAC)
# 8. Compare epoch event related SO-Spindle coupling (GC-ERPAC)
# 9. Compare instantaneous HR changes due to stimulation

#%%
## 0. Instantaneous HR changes
df_inst_hr =  pd.read_csv(stats_path + 'df_inst_hr.csv', index_col=0)
df_inst_hr = drop_bads_df(df_inst_hr)
df_inst_hr = df_inst_hr.groupby('Subject').filter(lambda x: set(x['Mode']) >= {'pn', 'nmes'})

def plot_hr_changes(df, metric='HR_Ratio_Change'):
    df['Target_Chan'] = df['Target_Chan'].str.replace('_', ' ').str.capitalize()

    g = sns.catplot(
        data=df,
        kind="box",
        x="Mode",
        y=metric,
        hue="Target_Chan",
        palette="pastel",
        legend=True,
        showfliers=False,
    )
    
    # include points for data 
    g.map_dataframe(sns.stripplot, x="Mode", y=metric, 
                    hue="Target_Chan",
                    dodge=True, palette="pastel", 
                    size=10, alpha=0.5, linewidth=1)
    
    g.set_axis_labels("", "Ratio Change (%)", fontsize=14)
    g.set_titles("{col_name} Target ({row_name})", size=16)
    g.fig.suptitle(f"{metric}", fontsize=16, weight='bold') 
    plt.show()
    
    # stats
    df.rm_anova(within=['Mode','Target_Chan'], 
                subject='Subject', dv=metric)
  
df_inst_hr.rm_anova(within=['Mode','Target_Chan'], subject='Subject', dv='HR_Ratio_Change')
plot_hr_changes(df_inst_hr, metric='HR_Ratio_Change')

df_inst_hr.rm_anova(within=['Mode','Target_Chan'], subject='Subject', dv='HRV_RMSSD_Ratio_Change')
plot_hr_changes(df_inst_hr, metric='HRV_RMSSD_Ratio_Change')

df_inst_hr.rm_anova(within=['Mode','Target_Chan'], subject='Subject', dv='HRV_SDNN_Ratio_Change')
plot_hr_changes(df_inst_hr, metric='HRV_SDNN_Ratio_Change')

#%%
## 1. Phase targeting analysis
df_phase = pd.read_csv(os.path.join(stats_path, 'df_phase.csv'), index_col=0)
df_phase = drop_bads_df(df_phase)

# Plot
phase_plot = phase_targeting_plot(df_phase)

# Stats
phase_stats = phase_targeting_stats(df_phase)

#%%
## 2. Event related analyses
# ERP - evoked data (with and without contrasts)
df_evoked = pd.read_pickle(os.path.join(stats_path, 'df_evokeds.p')) #'df_evokeds_lm.p'
df_evoked = drop_bads_df(df_evoked)
df_evoked.set_index(['Mode','Subject','Night'], inplace=True)
 
#%%
## 3. Time frequency analyses
df_tfr = pd.read_pickle(os.path.join(stats_path, 'df_tfr.p'))
df_tfr = drop_bads_df(df_tfr)

#%%
## 4. ndPAC analysis
df_ndpac = pd.read_csv(os.path.join(stats_path, 'df_ndpac.csv'), index_col=0)
df_ndpac = drop_bads_df(df_ndpac)

df_ndpac = df_ndpac[df_ndpac.Session=='Post'].reset_index(drop=True)

# df_ndpac.set_index(['Mode','Stim','Target_Chan','Chan']).loc['nmes','stim','c3']
d =  (df_ndpac.groupby(['Mode','Stim','Target_Chan','Chan']).mean().loc['nmes','stim','c3'].ndPAC  
    - df_ndpac.groupby(['Mode','Stim','Target_Chan','Chan']).mean().loc['nmes','sham','c3'].ndPAC)
     
yasa.topoplot(d)

d = (df_ndpac.groupby(['Mode','Stim','Target_Chan','Chan']).mean().loc['pn','stim','c3'].ndPAC  
     - df_ndpac.groupby(['Mode','Stim','Target_Chan','Chan']).mean().loc['pn','sham','c3'].ndPAC)
     
yasa.topoplot(d)
plt.show()

#%%
## 5. ERPAC analysis
df_erpac = pd.read_pickle(os.path.join(stats_path, 'df_erpac.p'))
df_erpac = drop_bads_df(df_erpac)

plot_ergpac_diff(df_erpac, times=(-0.5, 2))
        
#%%
# Extract grand averages for pinknoise stimulation
gav_pn_sham_c3 = mne.grand_average(list(df_evoked.loc['pn'].Evoked_sham_c3)).crop(tmin=-2, tmax=2)
gav_pn_stim_c3  = mne.grand_average(list(df_evoked.loc['pn'].Evoked_stim_c3)).crop(tmin=-2, tmax=2)
gav_pn_sham_fz = mne.grand_average(list(df_evoked.loc['pn'].Evoked_sham_fz)).crop(tmin=-2, tmax=2)
gav_pn_stim_fz  = mne.grand_average(list(df_evoked.loc['pn'].Evoked_stim_fz)).crop(tmin=-2, tmax=2)

# Extract grand averages for nmes stimulation
gav_nmes_sham_c3 = mne.grand_average(list(df_evoked.loc['nmes'].Evoked_sham_c3)).crop(tmin=-2, tmax=2)
gav_nmes_stim_c3  = mne.grand_average(list(df_evoked.loc['nmes'].Evoked_stim_c3)).crop(tmin=-2, tmax=2)
gav_nmes_sham_fz = mne.grand_average(list(df_evoked.loc['nmes'].Evoked_sham_fz)).crop(tmin=-2, tmax=2)
gav_nmes_stim_fz  = mne.grand_average(list(df_evoked.loc['nmes'].Evoked_stim_fz)).crop(tmin=-2, tmax=2)

# Plotting pararameters
times = np.asarray([-0.5, 0.0, 0.25, 0.5, 0.75, 1])
ts_args = dict(spatial_colors = True, gfp=True, time_unit='s')
topomap_args = dict(outlines = 'head', time_unit='s', time_format = "%0.2f s")

# Plot grand averages for pinknoise stimulation
p1 = gav_pn_sham_c3.plot_joint(title='Pinknoise sham C3', times=times, ts_args=ts_args, topomap_args=topomap_args)
p2 = gav_pn_stim_c3.plot_joint(title='Pinknoise stim C3', times=times, ts_args=ts_args, topomap_args=topomap_args)
p3 = gav_pn_sham_fz.plot_joint(title='Pinknoise sham Fz', times=times, ts_args=ts_args, topomap_args=topomap_args)
p4 = gav_pn_stim_fz.plot_joint(title='Pinknoise stim Fz', times=times, ts_args=ts_args, topomap_args=topomap_args)

# Plot grand averages for nmes stimulation
p5 = gav_nmes_sham_c3.plot_joint(title='NMES sham C3', times=times, ts_args=ts_args, topomap_args=topomap_args)
p6 = gav_nmes_stim_c3.plot_joint(title='NMES stim C3', times=times, ts_args=ts_args, topomap_args=topomap_args)
p7 = gav_nmes_sham_fz.plot_joint(title='NMES sham Fz', times=times, ts_args=ts_args, topomap_args=topomap_args)
p8 = gav_nmes_stim_fz.plot_joint(title='NMES stim Fz', times=times, ts_args=ts_args, topomap_args=topomap_args)

# 
#%%

# def plot_diff()
c3_pn = list(df_evoked.loc['pn'].Contrast_c3)
c3_pn_ma = np.concatenate([np.expand_dims(c3_pn[i].get_data(picks=['C5','C3','C1','CP3']).mean(0), 0)
                           for i in range(len(c3_pn))])

c3_nmes = list(df_evoked.loc['nmes'].Contrast_c3)
c3_nmes_ma = np.concatenate([np.expand_dims(c3_nmes[i].get_data(picks=['C5','C3','C1','CP3']).mean(0), 0)
                             for i in range(len(c3_nmes))])

# spatial permuation cluster test
t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
    c3_nmes_ma,  #c3_pn_ma,             # numpy array for contrast [n_subjects, n_voltage, n_channels]
    n_permutations=1024,                # 1000 is the minimum
    threshold=3, #dict(start=0, step=0.2),  # TFCE, starting at 0, in 0.2 steps (in t-values)
    tail=0,                             # two-tailed test (1 or -1 for one-tailed)
    n_jobs=-1,                          # increase value to speed up computations
    buffer_size=None,
    out_type='mask',                    # returns a mask map instead of indices of sig. points
    seed=1503
)

# Plot the first subplot with time series data
plt.plot(times, c3_pn_ma.mean(0) * 1e3, label='Motor Area Response - CLAS (C3)')
# plt.plot(times, c3_pn_ma.T*1e3, linewidth=0.2)

plt.plot(times, c3_nmes_ma.mean(0) * 1e3, label='Motor Area Response - CLNMES (C3)')
# plt.plot(times, c3_nmes_ma.T*1e3, linewidth=0.2)

# # Plot the second subplot with T-values
# for cluster, p_value in zip(clusters, cluster_p_values):
#     if p_value < 0.05:
#         ax1.fill_between(times[cluster[0]], t_obs[cluster[0]], color='red', alpha=0.3)

# Highlighting significant clusters (p < 0.05) using fill_between
for cluster, p_value in zip(clusters, cluster_p_values):
    if p_value < 0.05:
        plt.fill_between(times[cluster[0]], c3_nmes_ma.mean(0)[cluster[0]] * 1e3,
                         color='blue', alpha=0.3)
       
# spatial permuation cluster test
t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
    c3_pn_ma,  #c3_pn_ma,             # numpy array for contrast [n_subjects, n_voltage, n_channels]
    n_permutations=1024,                # 1000 is the minimum
    threshold=3, #dict(start=0, step=0.2),  # TFCE, starting at 0, in 0.2 steps (in t-values)
    tail=0,                             # two-tailed test (1 or -1 for one-tailed)
    n_jobs=-1,                          # increase value to speed up computations
    buffer_size=None,
    out_type='mask',                    # returns a mask map instead of indices of sig. points
    seed=1503
)

# Highlighting significant clusters (p < 0.05) using fill_between
for cluster, p_value in zip(clusters, cluster_p_values):
    if p_value < 0.05:
        plt.fill_between(times[cluster[0]], c3_pn_ma.mean(0)[cluster[0]] * 1e3,
                         color='red', alpha=0.3)

plt.legend()

#%%
# Plot RMS responses by condition
# evokeds = dict(
#     CLAS_Sham_C3=list(df_evoked.loc['pn'].Evoked_sham_c3), 
#     CLAS_Stim_C3=list(df_evoked.loc['pn'].Evoked_stim_c3), 
#     CLAS_Sham_Fz=list(df_evoked.loc['pn'].Evoked_sham_fz), 
#     CLAS_Stim_Fz=list(df_evoked.loc['pn'].Evoked_stim_fz),
#     CLNMES_Sham_C3=list(df_evoked.loc['nmes'].Evoked_sham_c3),
#     CLNMES_Stim_C3=list(df_evoked.loc['nmes'].Evoked_stim_c3),
#     CLNMES_Sham_Fz=list(df_evoked.loc['nmes'].Evoked_sham_fz),
#     CLNMES_Stim_Fz=list(df_evoked.loc['nmes'].Evoked_stim_fz)
#     )

evokeds = dict(
    CLAS_C3=list(df_evoked.loc['pn'].Contrast_c3), 
    CLAS_Fz=list(df_evoked.loc['pn'].Contrast_fz),
    CLNMES_C3=list(df_evoked.loc['nmes'].Contrast_c3),
    CLNMES_Fz=list(df_evoked.loc['nmes'].Contrast_fz)
    )

# evokeds = dict(
#                CLAS=list(df_evoked.loc['pn'].Contrast_c3_fz), 
#                CLNMES=list(df_evoked.loc['nmes'].Contrast_c3_fz),
#                )


# Function to calculate gfp from evoked data dictionary
def process_evoked_data_gfp(evokeds):
    results = {}
    for condition, evoked_list in evokeds.items():
        # Stack all data arrays vertically (assuming data shape is consistent)
        all_data = np.concatenate([np.expand_dims(evk.data, 1)*1e3 for evk in evoked_list], 1)
        
        # Calculate the norm over columns (i.e., across trials), then normalize by sqrt(N)
        #norm_data = np.linalg.norm(all_data, axis=0) / np.sqrt(len(evoked_list)) 
        norm_data = np.sqrt((all_data**2).mean(axis=0))       
        
        # Initialize the dictionary for the current condition
        results[condition] = {}
                   
        # Store the processed data back in a dictionary
        results[condition] = norm_data
    
    return results

results_gfp = process_evoked_data_gfp(evokeds)

rms_all = []
for condition, rms in results_gfp.items():
    rms_all.append(np.expand_dims(rms, 1))
    print(condition)

pval = 0.05  # arbitrary
df = rms_all[0].shape[0] - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
          
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
    rms_all,
    n_permutations=1024,
    threshold=thresh,
    tail=1,
    n_jobs=-1,
    out_type="mask",
)

times = np.arange(-3*128, 3*128)/128

fig, ax = plt.subplots(1, figsize=(10, 8))
# Combine applies lambda data: np.sqrt((data**2).mean(axis=0)) for GFP
mne.viz.plot_compare_evokeds(evokeds, combine="gfp", ci=True, axes=ax)
# Let's assume 'sig' is your boolean array indicating significant clusters
sig = clusters[np.argmin(cluster_p_values)]
# Find indices where True segments start and end
starts = np.where(np.diff(sig.astype(int)) == 1)[1][0]
ends = np.where(np.diff(sig.astype(int)) == -1)[1][0]
# Fill the areas of significance
ax.axvspan(times[starts], times[ends], color='red', alpha=0.1)
# Create a custom legend entry for the shaded area
red_patch = mpatches.Patch(color='red', alpha=0.1, label='Cluster p-value < 0.05')
# Collect existing legend handles and labels
handles, labels = ax.get_legend_handles_labels()
# Include the custom legend entry
handles.append(red_patch)
labels.append('Cluster p-value < 0.05')
# Create the combined legend
ax.legend(handles, labels, loc='upper right')
plt.show()


#%%
# ERP Stats
erp_stats(df_evoked, length=[0, 2], thresh_method='ttest')

#%%
## . Granger Analysis
df_gc = pd.read_csv(stats_path + 'df_gc.csv', index_col=0) 
df_gc = drop_bads_df(df_gc)

np.subtract(df_gc.groupby(['Mode','Stim','Target_Chan','Freqs']).mean().loc['nmes','stim','c3'].TRGC, 
            df_gc.groupby(['Mode','Stim','Target_Chan','Freqs']).mean().loc['nmes','sham','c3'].TRGC).plot()

sns.lineplot(df_gc, x='Freqs', y='TRGC', hue='Mode')

diff = df_gc[np.logical_and(df_gc.Stim=='stim', df_gc.Mode=='nmes')]

df = df_gc
# Filter the data for each target and mode combination
groups = df.groupby(["Target_Chan", "Mode", "Stim"])

# Calculate averages
averaged_data = groups[["Net_GC", "TRGC"]].mean().reset_index()

# Compute contrasts
# Assuming you have full data with both 'stim' and 'sham' for each group
contrasts = []
for (target_chan, mode), group in averaged_data.groupby(["Target_Chan", "Mode"]):
    stim_data = group[group["Stim"] == "stim"].set_index(["Target_Chan", "Mode", "Stim"])
    sham_data = group[group["Stim"] == "sham"].set_index(["Target_Chan", "Mode", "Stim"])
    contrast = stim_data - sham_data
    contrasts.append(contrast.reset_index())

# Combine all contrasts into a single DataFrame
final_contrasts = pd.concat(contrasts, ignore_index=True)

#%%
## Power Analysis (Epoched)
df_power = pd.read_csv(stats_path + 'df_power.csv', index_col=0) 
df_power = drop_bads_df(df_power)

def group_sleep_power_stats(df_power):
    # Load adjacency matrix
    adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/LuPf_1_1-epo.fif')
    adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
    
    # Filter subjects with both 'pn' and 'nmes' conditions
    df_power = df_power.groupby('Subject').filter(lambda x: set(x['Mode']) >= {'pn', 'nmes'})

    # Convert specified bands to logarithmic scale (excluding Aperiodic and Offset)
    #bands_to_log = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma', 'TotalAbsPow']
    bands_to_log = ['Delta']
    df_power[bands_to_log] = np.log(df_power[bands_to_log])

    # Features to be analyzed, including log-transformed bands and untransformed features like 'Aperiodic' and 'Offset'
    features = bands_to_log + ['Aperiodic', 'Offset']

    # Loop over modes, target channels, and features
    for mode in df_power.Mode.unique():
        for target in df_power.Target_Chan.unique():
            for feature in features:
                # Separate stim and sham conditions
                df_stim = df_power[(df_power.Mode == mode) & (df_power.Stim == 'stim') & (df_power.Target_Chan == target)]
                df_sham = df_power[(df_power.Mode == mode) & (df_power.Stim == 'sham') & (df_power.Target_Chan == target)]

                # Ensure consistent channel ordering
                unique_channels = df_stim['Chan'].unique()

                # Extract contrast (stim - sham) for the current feature
                contrast = (df_stim.pivot_table(index='Subject', columns='Chan').reindex(columns=unique_channels, level=1)[feature].values - 
                            df_sham.pivot_table(index='Subject', columns='Chan').reindex(columns=unique_channels, level=1)[feature].values)

                # Calculate t-threshold for the cluster test
                pval = 0.05
                dof = contrast.shape[0] - 1  # degrees of freedom for the test
                thresh = scipy.stats.t.ppf(1 - pval / 2, dof)  # two-tailed, t distribution

                # Spatial permutation cluster test
                t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
                    contrast,                            # numpy array for contrast [n_subjects, n_channels]
                    n_permutations=1024,                 # 1000 is the minimum
                    threshold=thresh,                    # threshold for cluster formation
                    tail=0,                              # two-tailed test (1 or -1 for one-tailed)
                    n_jobs=-1,                           # increase value to speed up computations
                    adjacency=adjacency,                 # sparse matrix for channel adjacency as computed above
                    buffer_size=None,
                    out_type='mask',                     # returns a mask map instead of indices of sig. points
                    seed=1503
                )

                # Handle significant clusters and visualization
                if all(p > 0.05 for p in cluster_p_values):
                    print(f'No significant clusters found for {feature} in {mode} {target}')
                    df_stats = pd.DataFrame({'Chan': ch_names, 
                                             'Power': contrast.mean(0).squeeze(), 
                                             'T-Stat': t_obs.squeeze(), 
                                             'Sig': [False] * len(ch_names)})
                    #mask = [False] * len(ch_names)
                else:
                    min_p_value = min(cluster_p_values)
                    print(f"P-Value: {min_p_value} for {feature} in {mode} {target}")
                    df_stats = pd.DataFrame({'Chan': ch_names, 
                                             'Power': contrast.mean(0).squeeze(), 
                                             'T-Stat': t_obs.squeeze(), 
                                             'Sig': clusters[np.argmin(cluster_p_values)]})
                    #mask = clusters[np.argmin(cluster_p_values)]

                df_stats = df_stats.set_index("Chan")

                # Visualization
                if feature == 'Aperiodic':
                    unit = 'Spectral Exponent' 
                elif feature == 'Offset':
                    unit = 'Spectral Offset'
                else:
                    unit = f'{feature}' #'log (mV/m2 / Hz)'
                
                # Plot
                fig, ax = plt.subplots(1, 2, figsize=(8, 6))
                
                if any(np.sign(df_stats['T-Stat']) == -1.0):
                    cmap = 'RdBu_r'
                else:
                    cmap = 'Reds'
                    
                topoplot(df_stats['Power'], 
                         mask=None, 
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
                             
                plt.tight_layout()
                plt.suptitle(f'{mode.upper()} ({target.capitalize()} Target) -> {feature}', fontsize=18)
                plt.show()

def group_sleep_power_stats_double_contrast(df_power):
    # Load adjacency matrix for EEG channels
    adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/LuPf_1_1-epo.fif')
    adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
    
    # Filter subjects with both 'pn' and 'nmes' conditions
    df_power = df_power.groupby('Subject').filter(lambda x: set(x['Mode']) >= {'pn', 'nmes'})

    # Convert specified bands to logarithmic scale (excluding Aperiodic and Offset)
    bands_to_log = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma', 'TotalAbsPow']
    df_power[bands_to_log] = np.log(df_power[bands_to_log])

    # Features to be analyzed, including log-transformed bands and untransformed features like 'Aperiodic' and 'Offset'
    features = bands_to_log + ['Aperiodic', 'Offset']

    # Loop over target channels and features
    for target in df_power.Target_Chan.unique():
        for feature in features:
            # Calculate contrasts for 'nmes' and 'pn' separately
            df_nmes_stim = df_power[(df_power.Mode == 'nmes') & (df_power.Stim == 'stim') & (df_power.Target_Chan == target)]
            df_nmes_sham = df_power[(df_power.Mode == 'nmes') & (df_power.Stim == 'sham') & (df_power.Target_Chan == target)]
            df_pn_stim = df_power[(df_power.Mode == 'pn') & (df_power.Stim == 'stim') & (df_power.Target_Chan == target)]
            df_pn_sham = df_power[(df_power.Mode == 'pn') & (df_power.Stim == 'sham') & (df_power.Target_Chan == target)]

            # Ensure consistent channel ordering
            unique_channels = df_nmes_stim['Chan'].unique()

            # Calculate individual contrasts (stim - sham)
            contrast_nmes = (df_nmes_stim.pivot_table(index='Subject', columns='Chan').reindex(columns=unique_channels, level=1)[feature].values - 
                             df_nmes_sham.pivot_table(index='Subject', columns='Chan').reindex(columns=unique_channels, level=1)[feature].values)

            contrast_pn = (df_pn_stim.pivot_table(index='Subject', columns='Chan').reindex(columns=unique_channels, level=1)[feature].values - 
                           df_pn_sham.pivot_table(index='Subject', columns='Chan').reindex(columns=unique_channels, level=1)[feature].values)

            # Calculate the double contrast: (nmes contrast) - (pn contrast)
            double_contrast = scipy.stats.zscore(contrast_nmes, axis=1) - scipy.stats.zscore(contrast_pn, axis=1)

            # Calculate t-threshold for the cluster test
            pval = 0.05
            dof = double_contrast.shape[0] - 1  # degrees of freedom for the test
            thresh = scipy.stats.t.ppf(1 - pval / 2, dof)  # two-tailed, t distribution

            # Spatial permutation cluster test on the double contrast
            t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
                double_contrast,                     # numpy array for double contrast [n_subjects, n_channels]
                n_permutations=1024,                 # 1000 is the minimum
                threshold=thresh,                    # threshold for cluster formation
                tail=0,                              # two-tailed test (1 or -1 for one-tailed)
                n_jobs=-1,                           # increase value to speed up computations
                adjacency=adjacency,                 # sparse matrix for channel adjacency as computed above
                buffer_size=None,
                out_type='mask',                     # returns a mask map instead of indices of sig. points
                seed=1503
            )

            # Handle significant clusters and visualization
            if all(p > 0.05 for p in cluster_p_values):
                print(f'No significant clusters found for {feature} in double contrast {target}')
                df_stats = pd.DataFrame({'Chan': ch_names, 
                                         'Power': double_contrast.mean(0).squeeze(), 
                                         'T-Stat': t_obs.squeeze(), 
                                         'Sig': [False] * len(ch_names)})
                #mask = [False] * len(ch_names)
            else:
                min_p_value = min(cluster_p_values)
                print(f"P-Value: {min_p_value} for {feature} in double contrast {target}")
                df_stats = pd.DataFrame({'Chan': ch_names, 
                                         'Power': double_contrast.mean(0).squeeze(), 
                                         'T-Stat': t_obs.squeeze(), 
                                         'Sig': clusters[np.argmin(cluster_p_values)]})
                #mask = clusters[np.argmin(cluster_p_values)]

            df_stats = df_stats.set_index("Chan")

            # Visualization
            if feature == 'Aperiodic':
                unit = 'Spectral Exponent' 
            elif feature == 'Offset':
                unit = 'Spectral Offset'
            else:
                unit = f'{feature}' #'log (mV/m2 / Hz)'
            
            # Plot
            fig, ax = plt.subplots(1, 2, figsize=(8, 6))
            
            if any(np.sign(df_stats['T-Stat']) == -1.0):
                cmap = 'RdBu_r'
            else:
                cmap = 'Reds'
                
            topoplot(df_stats['Power'], 
                     mask=None, 
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
                         
            # im1, _ = mne.viz.plot_topomap(double_contrast.mean(0).squeeze(), 
            #                               pos=adj_epochs.info,
            #                               axes=ax[0], 
            #                               cmap='RdBu_r',
            #                               show=False, 
            #                               names=None)
            # cbar1 = fig.colorbar(im1, fraction=0.05, ax=ax[0])   
            # cbar1.ax.set_ylabel(unit, rotation=270)
            # plt.tight_layout()

            # if all(p > 0.05 for p in cluster_p_values): 
            #     im2, _ = mne.viz.plot_topomap(t_obs.squeeze(), 
            #                                   pos=adj_epochs.info,
            #                                   axes=ax[1], 
            #                                   cmap='RdBu_r',
            #                                   names=None,
            #                                   show=False)
            # else:
            #     im2, _ = mne.viz.plot_topomap(t_obs.squeeze(), 
            #                                   pos=adj_epochs.info,
            #                                   axes=ax[1], 
            #                                   mask=mask,
            #                                   cmap='RdBu_r',
            #                                   names=None,
            #                                   show=False,  
            #                                   mask_params=dict(markersize=8, markerfacecolor='y'))
            
            # cbar2 = fig.colorbar(im2, fraction=0.05, ax=ax[1])   
            # cbar2.ax.set_ylabel('t-stat', rotation=270)
            
            plt.tight_layout()
            plt.suptitle(f'CLNMES - CLAS ({target.capitalize()} Target) -> {feature}', fontsize=18)
            plt.show()

#%%
## Power Analysis (Continuous)
df_power = pd.read_csv(stats_path + 'df_sleep_hrv.csv', index_col=0)
df_power = drop_bads_df(df_power)
df_power = df_power[df_power.columns[:-8]]

def group_sleep_power_stats_cont(df_power, stage=2, feature='Theta'):

    df_power = df_power.groupby('Subject').filter(lambda x: set(x['Mode']) >= {'pn', 'nmes'})
    df_power = df_power[df_power.Stage==stage]
    
    df_nmes = df_power[(df_power.Mode == 'nmes')]
    df_pn = df_power[(df_power.Mode == 'pn')]
    
    unique_channels = df_nmes['Channel'].unique()
    
    #if feature == 'Aperiodic' or feature == 'Offset':
    contrast = (
        df_nmes.pivot_table(index='Subject', 
                            columns='Channel', 
                            values=feature, 
                            aggfunc='mean').reindex(columns=unique_channels, level=1).values  
        -                    
        df_pn.pivot_table(index='Subject', 
                          columns='Channel',
                          values=feature, 
                          aggfunc='mean').reindex(columns=unique_channels, level=1).values
    )
    # else: 
    #     contrast = (
    #         scipy.stats.zscore(df_nmes.pivot_table(index='Subject', 
    #                             columns='Channel', 
    #                             values=feature, 
    #                             aggfunc='mean').reindex(columns=unique_channels, level=1).values, -1)  
    #         -                    
    #         scipy.stats.zscore(df_pn.pivot_table(index='Subject', 
    #                             columns='Channel',
    #                             values=feature, 
    #                             aggfunc='mean').reindex(columns=unique_channels, level=1).values, -1)
    #     )
        
    # contrast[1, 15] = 0
    
    # Adjacency matrix
    adj_epochs = mne.read_epochs('/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/LuPf_1_1-epo.fif')
    adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, 'eeg')
    
    # Permutation Test
    pval = 0.05
    dof = contrast.shape[0] - 1  # degrees of freedom for the test
    if all(np.sign(contrast.mean(0)) == -1.0):
        cmap = 'Blues'
        # tail = -1
        # thresh = scipy.stats.t.ppf(pval, dof)  # One-tailed test (left tail)
    elif any(np.sign(contrast.mean(0)) == -1.0):
        cmap = 'RdBu_r'
        # tail = 0
        # thresh = scipy.stats.t.ppf(1 - pval / 2, dof)  # two-tailed, t distribution
    else:
        cmap = 'Reds'
        # tail = 1
        # thresh = scipy.stats.t.ppf(1 - pval, dof)
        
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
        fig.suptitle(f'NREM {unit} (CLNMES - CLAS)', fontsize=18)
    else:
        fig.suptitle(f'REM {unit} (CLNMES - CLAS)', fontsize=18)
        
    plt.tight_layout()
    plt.show()

#%%
## Bayesian analysis 
import bambi as bmb
import arviz as az
az.style.use("arviz-white")

# Initialize model
model = bmb.Model('HRV_LFHF ~ Mode + (1|Subject)', df_power)
print(model)

# Fit the model using 1000 on each chain
results = model.fit(draws=1000)

# Use ArviZ to plot the results
az.plot_trace(results)
az.summary(results)

# Posterior predictive check
ppc = model.predict(results)
az.plot_ppc(ppc)

#%%
# TCT analysis
df_tct = pd.read_csv(stats_path + 'df_tct.csv', index_col=0)
df_tct = drop_bads_df(df_tct)

# Filtering data for each mode
data_pn = df_tct[df_tct['Mode'] == 'pn']
data_nmes = df_tct[df_tct['Mode'] == 'nmes']

for dv in ['Consistency_RMS', 'Consistency',
           'RMS_Trial', 'RMS_Evoked', 'ERP']:
    # Create a figure with four subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharey=True)

    # Plot for mode 'pn'
    plot_mode_condition(data_pn[data_pn['Target_Chan'] == 'fz'], dv, axs[0, 0], 'CLAS (Fz)')
    plot_mode_condition(data_pn[data_pn['Target_Chan'] == 'c3'], dv, axs[0, 1], 'CLAS (C3)')
    
    # Plot for mode 'nmes'
    plot_mode_condition(data_nmes[data_nmes['Target_Chan'] == 'fz'], dv, axs[1, 0], 'CLNMES (Fz)')
    plot_mode_condition(data_nmes[data_nmes['Target_Chan'] == 'c3'], dv, axs[1, 1], 'CLNMES (C3)')
    
    # Adjust layout
    plt.show()

    ## Stats
    # Fit the repeated measures ANOVA without the three-way interaction
    # Define the model formula with two-way interactions
    formula = f"{dv} ~ (Target_Chan) * (Stim) * (Mode)"
    
    # Fit the model using MixedLM from statsmodels
    model = sm.MixedLM.from_formula(formula, groups=df_tct['Subject'], data=df_tct)
    result = model.fit()
    
    # Display the results
    result.summary()

#%%
## . PCI analysis
df_pci = pd.read_csv(stats_path + 'df_pci.csv', index_col=0)
df_pci = drop_bads_df(df_pci)

# Filtering data for each mode
data_pn = df_pci[df_pci['Mode'] == 'pn']
data_nmes = df_pci[df_pci['Mode'] == 'nmes']

# Create a figure with four subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharey=True)

# Plot for mode 'pn'
plot_mode_condition(data_pn[data_pn['Target_chan'] == 'Fz'], 'PCI', axs[0, 0], 'CLAS (Fz)')
plot_mode_condition(data_pn[data_pn['Target_chan'] == 'C3'], 'PCI', axs[0, 1], 'CLAS (C3)')

# Plot for mode 'nmes'
plot_mode_condition(data_nmes[data_nmes['Target_chan'] == 'Fz'], 'PCI', axs[1, 0], 'CLNMES (Fz)')
plot_mode_condition(data_nmes[data_nmes['Target_chan'] == 'C3'], 'PCI', axs[1, 1], 'CLNMES (C3)')

# Adjust layout
plt.show()

## Stats
# Fit the repeated measures ANOVA without the three-way interaction
# Define the model formula with two-way interactions
formula = "PCI ~ C(Target_chan) * C(Stim) + C(Target_chan) * C(Mode) + C(Stim) * C(Mode)"

# Fit the model using MixedLM from statsmodels
model = sm.MixedLM.from_formula(formula, groups=df_pci['Subject'], data=df_pci)
result = model.fit()

# Display the results
result.summary()

# Group the data by 'Subject', 'Target_chan', 'Mode' for 'stim' and 'sham' conditions
grouped_stim = df_pci[df_pci['Stim'] == 'stim'].groupby(['Subject', 'Target_chan', 'Mode']).mean()['PCI']
grouped_sham = df_pci[df_pci['Stim'] == 'sham'].groupby(['Subject', 'Target_chan', 'Mode']).mean()['PCI']

# Calculate the contrast for each subject within each group
within_subject_contrast = grouped_stim - grouped_sham
within_subject_contrast = within_subject_contrast.reset_index()
within_subject_contrast

# Plotting
g = sns.catplot(data=within_subject_contrast, kind="box", x="Target_chan",
                col="Mode", y="PCI", palette="pastel")

# Adjusting the plot
#g.set_titles("{col_name} Channel")
g.set_axis_labels("", "PCI (Stim - Sham)", fontsize=16)
plt.subplots_adjust(top=0.9)
plt.tight_layout()
plt.show()

# stats
within_subject_contrast.rm_anova(dv='PCI', subject='Subject', 
                                 within=['Mode', 'Target_chan'], detailed=True)