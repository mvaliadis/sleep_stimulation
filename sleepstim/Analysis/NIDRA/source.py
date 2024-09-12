#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:00:26 2024

@author: administrator
"""

import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.minimum_norm import apply_inverse, make_inverse_operator
import numpy as np
from glob import glob
from tqdm import tqdm
import seaborn as sns
import pandas as pd 

mne.set_log_level('ERROR')

def common_pipeline_source_preprocess(save=False):   
    save_path = '/media/administrator/Sleep_Data/HM/study/'
    if save:
        # Create the source space
        src = mne.setup_source_space(
            subject='fsaverage', spacing='oct6', add_dist=False)
        
        # Create the BEM surfaces
        model = mne.make_bem_model(subject='fsaverage', ico=4, 
                                   conductivity=(0.3, 0.006, 0.3))  # Three-layer model
        bem = mne.make_bem_solution(model)
        
        # Compute the forward solution
        info =  mne.read_epochs('/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/LuPf_1_1-epo.fif').info
        trans = 'fsaverage'  # Use the template MRI transformation
        fwd = mne.make_forward_solution(
            info, trans=trans, src=src,
            bem=bem, eeg=True, meg=False)
        
        # Save froward solution and source space
        fwd.save(save_path + 'fsaverage-fwd.fif.gz', overwrite=True)
        src.save(save_path + 'fsaverage-src.fif.gz', overwrite=True)
    else:
        fwd = mne.read_forward_solution(save_path + 'fsaverage-fwd.fif.gz')
        src = mne.read_source_spaces(save_path + 'fsaverage-fwd.fif.gz')
        
    return fwd, src

data_dir_lm = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/*lm-epo.fif'
data_dir_csd = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/*csd-epo.fif'

#%%
## Load source space estimation
fwd, src = common_pipeline_source_preprocess(save=False)

# Set the signal-to-noise ratio (SNR)
snr = 3.0
lambda2 = 1.0 / snr ** 2
        
## Masking 
# Load the parcellation
labels = mne.read_labels_from_annot(subject='fsaverage', parc='HCPMMP1_combined')

# Select specific labels (ROIs)
rois = [
    # 'DorsoLateral Prefrontal Cortex-lh', 'DorsoLateral Prefrontal Cortex-rh',
    # 'Anterior Cingulate and Medial Prefrontal Cortex-lh', 'Anterior Cingulate and Medial Prefrontal Cortex-rh',
    # 'Premotor Cortex-lh', 'Premotor Cortex-rh',
    'Somatosensory and Motor Cortex-lh', 'Somatosensory and Motor Cortex-rh',
    # 'Early Auditory Cortex-lh', 'Early Auditory Cortex-rh',
    # 'Auditory Association Cortex-lh', 'Auditory Association Cortex-rh', 
    ]
    
selected_labels = [label for label in labels if label.name in rois]

## Loop it
sources = []
for file_lm, file_csd in tqdm(zip(sorted(glob(data_dir_lm)), sorted(glob(data_dir_csd)))):
    print(f"Processing: {file_lm.split('/')[-1]} {file_csd.split('/')[-1]}")
    
    # Get subject, night info
    subject, night, ref = file_csd.split('/')[-1].split('-')[0].split('_')

    # if subject == 'RoKi':
    #     break
    
    # Load epochs
    epochs = mne.read_epochs(file_lm, preload=True)
    epochs_csd = mne.read_epochs(file_csd, preload=True)
    
    # Get stimulation modality
    mode = list(epochs.event_id)[0].split('_')[0]
    
    # Re-reference to the average reference
    epochs.set_eeg_reference('average', projection=True)
    
    # Compute noise covariance from the baseline period
    noise_cov = mne.compute_covariance(epochs, tmin=-3, tmax=-1.5)
    
    # Compute the inverse operator
    inverse_operator = make_inverse_operator(
        epochs.info, fwd, noise_cov, loose=0.2, depth=0.8)
             
    ## Iterate over conditions
    for cond in epochs.event_id.keys():
        
        # Compute the evoked response
        evoked = epochs[cond].average()
        #evoked = epochs_csd[cond].average()
               
        # Compute the source estimate
        stc = apply_inverse(evoked, inverse_operator, lambda2, method='dSPM')
            
        # # Extract the indices of the vertices in the selected labels
        # lh_vertices = selected_labels[0].vertices if 'lh' in selected_labels[0].name else selected_labels[1].vertices
        # rh_vertices = selected_labels[1].vertices if 'rh' in selected_labels[1].name else selected_labels[0].vertices
        
        # # Create masks for both hemispheres
        # lh_mask = np.in1d(stc.vertices[0], lh_vertices)
        # rh_mask = np.in1d(stc.vertices[1], rh_vertices)
        
        # # Combine the masks
        # combined_mask = np.concatenate([lh_mask, rh_mask])
        
        # # Apply the combined mask to the source estimate
        # stc_masked = stc.copy()
        # stc_masked.data[~combined_mask, :] = 0
        
        # Extract time series for the selected labels
        label_ts = stc.extract_label_time_course(selected_labels, src, mode='mean')
        
        # Create dataframe with results
        from scipy import stats
        time_series_df = pd.DataFrame(stats.zscore(label_ts.T), #/1e3, 
                                      columns=[label.name for label in selected_labels], 
                                      index=stc.times)
        time_series_df['Subject'] = subject
        time_series_df['Night'] = night
        time_series_df['Condition'] = cond
        time_series_df['Mode'] = mode
        sources.append(time_series_df)
                
        ## Plotting
        # # Plot unmasked source estimate
        # brain = stc.plot(subject='fsaverage', hemi='split',
        #                   initial_time=-0.5, time_unit='s', views=['lat'])#, 'med'])
        
        # # Add parcellation to plot (red=auditory, green=motor, blue=visual)
        # brain.add_annotation("HCPMMP1_combined", borders=2)
        # plt.show()
        
        # # Plot the masked source estimate
        # brain_mask = stc_masked.plot(subject='fsaverage', hemi='split',
        #                               initial_time=-0.5, time_unit='s', views=['lat'])#, 'med'])
        
        # # Add parcellation to plot (red=auditory, green=motor, blue=visual)
        # brain_mask.add_annotation("HCPMMP1_combined", borders=2)
        # plt.show()
        
        # # Plot standalone activations per ROI
        # plt.figure()
        # for i, label in enumerate(selected_labels):
        #     plt.plot(stc.times, label_ts[i]/1e3, label=label.name)
        
        # plt.xlabel('Time (s)')
        # plt.ylabel('Activation (AU) ')
        # plt.title(f'Average Activation - {cond}')
        # plt.legend()
        # plt.show()

# Concatenate all dataframes         
df = pd.concat(sources)

# Split the 'Condition' column into 'Stim/Sham' and 'Target Chan' columns
df['Stim'], df['Target_Chan'] = zip(*df['Condition'].apply(lambda x: x.split('_')[1:]))
df.drop('Condition', axis=1, inplace=True)
   
# Save df
stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'
df.to_csv(stats_path + 'df_source_analysis.csv')
    
#%%
## Stats
snr = 3.0
lambda2 = 1.0 / snr**2
method = "dSPM"  # use dSPM method (could also be MNE, sLORETA, or eLORETA)
inverse_operator = stc
sample_vertices = [s["vertno"] for s in inverse_operator["src"]]

# Let's average and compute inverse, resampling to speed things up
evoked1 = epochs_csd['nmes_stim_c3'].average()
condition1 = apply_inverse(evoked1, inverse_operator, lambda2, method)
evoked2 = epochs_csd['nmes_sham_c3'].average()
condition2 = apply_inverse(evoked2, inverse_operator, lambda2, method)

# Let's only deal with t > 0, cropping to reduce multiple comparisons
tmin = -1
tstep = condition1.tstep * 1000  # convert to milliseconds

# 
n_vertices_sample, n_times = condition1.data.shape
n_subjects = 6
print(f"Simulating data for {n_subjects} subjects.")

# Let's make sure our results replicate, so set the seed.
np.random.seed(0)
X = np.random.randn(n_vertices_sample, n_times, n_subjects, 2) * 10
X[:, :, :, 0] += condition1.data[:, :, np.newaxis]
X[:, :, :, 1] += condition2.data[:, :, np.newaxis]

# Source space morphing
fsave_vertices = [s["vertno"] for s in src]
morph_mat = mne.compute_source_morph(
    src=inverse_operator["src"],
    subject_to="fsaverage",
    spacing=fsave_vertices,
).morph_mat

n_vertices_fsave = morph_mat.shape[0]

# We have to change the shape for the dot() to work properly
X = X.reshape(n_vertices_sample, n_times * n_subjects * 2)
print("Morphing data.")
X = morph_mat.dot(X)  # morph_mat is a sparse matrix
X = X.reshape(n_vertices_fsave, n_times, n_subjects, 2)

X = np.abs(X)  # only magnitude
X = X[:, :, :, 0] - X[:, :, :, 1]  # make paired contrast

print("Computing adjacency.")
adjacency = mne.spatial_src_adjacency(src)

# Note that X needs to be a multi-dimensional array of shape
# observations (subjects) × time × space, so we permute dimensions
X = np.transpose(X, [2, 1, 0])

# Here we set a cluster forming threshold based on a p-value for
# the cluster based permutation test.
# We use a two-tailed threshold, the "1 - p_threshold" is needed
# because for two-tailed tests we must specify a positive threshold.
p_threshold = 0.001
df = n_subjects - 1  # degrees of freedom for the test
from scipy import stats
t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

# Now let's actually do the clustering. This can take a long time...
print("Clustering.")
T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(
    X,
    adjacency=adjacency,
    n_jobs=None,
    threshold=t_threshold,
    buffer_size=None,
    verbose=True,
)

# Select the clusters that are statistically significant at p < 0.05
good_clusters_idx = np.where(cluster_p_values < 0.05)[0]
good_clusters = [clusters[idx] for idx in good_clusters_idx]

print("Visualizing clusters.")

# Now let's build a convenient representation of our results, where consecutive
# cluster spatial maps are stacked in the time dimension of a SourceEstimate
# object. This way by moving through the time dimension we will be able to see
# subsequent cluster maps.
stc_all_cluster_vis = mne.stats.summarize_clusters_stc(
    clu, tstep=tstep, vertices=fsave_vertices, subject="fsaverage"
)

# Let's actually plot the first "time point" in the SourceEstimate, which
# shows all the clusters, weighted by duration.

# blue blobs are for condition A < condition B, red for A > B
brain = stc_all_cluster_vis.plot(
    hemi="both",
    views="lateral",

    time_label="temporal extent (ms)",
    size=(800, 800),
    smoothing_steps=5,
    clim=dict(kind="value", pos_lims=[0, 1, 40]),
)

# We could save this via the following:
# brain.save_image('clusters.png')

#%%
def plot_source_differences(df, target='fz', mode='nmes'):
    # Reset index and group by relevant columns to calculate the mean
    grouped_df = df.reset_index().groupby(['Mode', 'Stim', 'Target_Chan', 'Subject', 'index']).mean()
    
    # Extract data for 'stim' condition
    stim_data = grouped_df.loc[mode, 'stim', target].reset_index()
    
    # Extract data for 'sham' condition
    sham_data = grouped_df.loc[mode, 'sham', target].reset_index()
    
    # Merge dataframes on 'Subject' and 'index'
    data = pd.merge(stim_data, sham_data, on=['Subject', 'index'], suffixes=('_stim', '_sham'))
    
    # Perform the subtraction for both hemispheres
    data['Somatosensory and Motor Cortex-lh'] = data['Somatosensory and Motor Cortex-lh_stim'] - data['Somatosensory and Motor Cortex-lh_sham']
    data['Somatosensory and Motor Cortex-rh'] = data['Somatosensory and Motor Cortex-rh_stim'] - data['Somatosensory and Motor Cortex-rh_sham']
    
    # Plot each subject's runs with thinner lines
    plt.figure(figsize=(14, 7))
    
    for subject in data['Subject'].unique():
        subject_data = data[data['Subject'] == subject]
        sns.lineplot(x='index', y='Somatosensory and Motor Cortex-lh', data=subject_data, alpha=0.2, color='k', linewidth=0.5, label='_nolegend_')
        sns.lineplot(x='index', y='Somatosensory and Motor Cortex-rh', data=subject_data, alpha=0.2, color='r', linewidth=0.5, label='_nolegend_')
    
    # Calculate and plot the mean activation with a thicker line
    mean_data = data.groupby('index').mean().reset_index()
    sns.lineplot(x='index', y='Somatosensory and Motor Cortex-lh', data=mean_data, linewidth=2.5, color='k', label='Mean Somatosensory and Motor Cortex-lh')
    sns.lineplot(x='index', y='Somatosensory and Motor Cortex-rh', data=mean_data, linewidth=2.5, color='r', label='Mean Somatosensory and Motor Cortex-rh')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Activation (AU)')
    plt.title(f'Mean Activation - {mode} Contrast {target}')
    plt.legend()
    plt.show()
