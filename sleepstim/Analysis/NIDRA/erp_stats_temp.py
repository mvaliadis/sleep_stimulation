#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:02:09 2025

@author: administrator
"""

def erp_stats(df_evoked, length=[0, 2], thresh_method='ttest'):
    import scipy.stats
    import matplotlib.pyplot as plt
    import mne
    import seaborn as sns
    import numpy as np

    sns.set_theme(style="white")
    
    # Define the contrasts you wish to compute stats for.
    # This list includes your single contrasts. If double contrasts exist, add them.
    contrast_keys = ['Contrast_c3', 'Contrast_fz']
    for key in ['DoubleContrast_c3', 'DoubleContrast_fz']:
        if key in df_evoked.keys():
            contrast_keys.append(key)
    
    for con in contrast_keys:
        # Group by mode (e.g. 'pn', 'nmes') if applicable
        for mode, evoked_grp in df_evoked[con].groupby('Mode'):
            target_chan = con.split('_')[-1].upper()
            evk = evoked_grp.reset_index()[con].copy()
            
            # Compute a grand average and crop to the time range of interest
            gav = mne.grand_average(list(evk)).crop(tmin=length[0], tmax=length[1])
            
            # Build a 3D numpy array from individual evokeds [n_subjects, n_times, n_channels]
            evk_np = np.concatenate([
                np.expand_dims(evk[idx].copy().crop(tmin=length[0], tmax=length[1]).get_data(), 0) * 1e3
                for idx in range(len(evk))
            ], axis=0)
            # Rearrange dimensions so that data is of shape [n_subjects, n_channels, n_times]
            contrast = np.swapaxes(evk_np, 2, 1)
        
            # Prepare an adjacency matrix (here using an example epochs file; update path if needed)
            adj_epochs = mne.read_epochs(
                '/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/LuPf_1_1-epo.fif'
            )
            adjacency, ch_names = mne.channels.find_ch_adjacency(adj_epochs.info, ch_type='eeg')
            
            # Compute a threshold for the cluster test
            if thresh_method == 'ttest':
                pval = 0.05  # arbitrary alpha level
                df_val = contrast.shape[0] - 1  # degrees of freedom
                thresh = scipy.stats.t.ppf(1 - pval / 2, df_val)  # two-tailed threshold
            elif thresh_method == 'tfce':
                thresh = dict(start=0, step=0.2)
            else:
                raise ValueError("Unknown threshold method")
                
            # Run the spatio-temporal cluster 1-sample test
            t_obs, clusters, cluster_p_values, h0 = mne.stats.spatio_temporal_cluster_1samp_test(
                contrast,                           # data array [n_subjects, n_channels, n_times]
                n_permutations=1024,                # minimum recommended number of permutations
                threshold=thresh,
                tail=0,                             # two-tailed test
                n_jobs=-1,                          # use all available cores
                adjacency=adjacency,
                out_type='mask',                    # returns a mask for significant clusters
                seed=1503
            )
            
            # Find significant clusters (here p < 0.05)
            mask = cluster_p_values < 0.05
            
            if mask.mean() == 0:
                # No significant cluster found; optionally you could print a message here.
                continue
            else:
                # Plot significant clusters over a 1020 channel selection layout
                selections = mne.channels.make_1020_channel_selections(gav.info, midline="12z")
                fig, axes = plt.subplots(nrows=3, figsize=(20, 10), dpi=100)
                axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
                
                # Plot an image with the mask for the cluster with the smallest p-value
                best_cluster = clusters[np.argsort(cluster_p_values)[0]]
                gav.plot_image(mask=best_cluster.T, axes=axes,
                               show_names="all",
                               colorbar=True, mask_alpha=0.80, 
                               group_by=selections, show=False)
                plt.tight_layout()
            
                # Plot a joint view with topomaps and time-series snapshots
                times = np.array([0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
                ts_args = dict(spatial_colors=True, gfp=True, time_unit='s')
                topomap_args = dict(outlines='head', time_unit='s', time_format="%0.2f s")
                if mode == 'pn':
                    title = f'CLAS Evoked Difference ({target_chan})'
                elif mode == 'nmes':
                    title = f'CLNMES Evoked Difference ({target_chan})'
                else:
                    title = f'Evoked Difference ({target_chan})'
                    
                gav.plot_joint(times=times, title=title, ts_args=ts_args, topomap_args=topomap_args)
