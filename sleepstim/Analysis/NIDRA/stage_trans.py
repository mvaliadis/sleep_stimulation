#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:57:17 2024

@author: administrator
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
import pandas as pd
import mne
import seaborn as sns

hypno_path = '/media/administrator/Sleep_Data/Processed/Hypnograms/' 
stats_path = '/media/administrator/Sleep_Data/Processed/Statistics/'

def plot_stage_transition_nodes1(transition_matrix):
   
    # Sleep stage names
    stage_names = ["Wake", "N1", "N2", "N3", "REM"]
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with sleep stage labels
    for i, stage in enumerate(stage_names):
        G.add_node(i, label=stage)
    
    # Add edges with weights
    for i, row in enumerate(transition_matrix):
        for j, prob in enumerate(row):
            if prob > 0:  # Add an edge only if there's a transition
                G.add_edge(i, j, weight=prob)
    
    # Position nodes using the spring layout
    pos = nx.spring_layout(G)
    
    # Draw nodes with labels
    nx.draw_networkx_nodes(G, pos, node_size=700)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20,
                           edge_cmap=plt.cm.Blues, width=2)
    
    # Draw edge labels (transition probabilities)
    edge_labels = {(i, j): f'{G[i][j]["weight"]:.2f}' for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
    
    # Show plot
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_stage_transition_nodes(transition_matrix, percentage_changes):
    # Sleep stage names
    stage_names = ["Wake", "N1", "N2", "N3", "REM"]
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with sleep stage labels and change values
    for i, stage in enumerate(stage_names):
        G.add_node(i, label=stage, change=percentage_changes[i])
    
    # Add edges with weights
    for i, row in enumerate(transition_matrix):
        for j, change in enumerate(row):
            if abs(change) > 0:  # Add an edge only if there's a transition change
                G.add_edge(i, j, weight=change)
    
    # Position nodes using a predefined circular layout
    pos = {
        0: (0, 1),
        1: (1, 0.5),
        2: (1, -0.5),
        3: (0, -1),
        4: (-1, 0)
    }
    
    # Define node colors and sizes based on percentage changes
    node_colors = ['green' if change > 0 else 'red' if change < 0 else 'grey' for change in percentage_changes]
    node_sizes = [abs(change) * 300 for change in percentage_changes]
    
    # Draw nodes with labels
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    
    # Draw edges with colors and widths based on weights
    edges = G.edges(data=True)
    edge_colors = ['green' if data['weight'] > 0 else 'red' for _, _, data in edges]
    edge_widths = [abs(data['weight']) * 5 for _, _, data in edges]
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_widths, arrowstyle='-|>', arrowsize=20)
    
    # Draw edge labels (percentage changes)
    edge_labels = {(i, j): f'{G[i][j]["weight"]:.1f}%' for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
    
    # Display percentage changes next to nodes
    node_labels = {i: f'{percentage_changes[i]:+.1f}%' for i in range(len(percentage_changes))}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black', verticalalignment='bottom')
    
    # Show plot
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def compute_transition_matrix(df, subjects):
    transition_matrices = {}
    for subject in subjects:
        subject_data = df[df['Subject'] == subject]
        transition_matrix = subject_data.groupby('From Stage').mean(numeric_only=True).drop(columns=['Night'])
        transition_matrices[subject] = transition_matrix.to_numpy()
    return transition_matrices

def transition_matrix_stats(stacked_diffs):
    # Get subject count
    n_subj = stacked_diffs.shape[0]
      
    # Reshape the data to combine axis 1 and axis 2
    reshaped_diffs = stacked_diffs.reshape(n_subj, -1)
    
    # Identify zero-variance columns
    variance = np.var(reshaped_diffs, axis=0)
    non_zero_variance_columns = variance > 0
    
    # Filter out zero-variance columns
    filtered_diffs = reshaped_diffs[:, non_zero_variance_columns]
    # Perform the permutation test
    tvals, pvals, H0 = mne.stats.permutation_t_test(
        filtered_diffs,  # (n_samples, n_tests)
        n_permutations=5000, 
        n_jobs=-1,
        seed=42
    )
           
    # Map results back to the original 5x5 structure
    final_tvals = np.full((5, 5), np.nan)
    final_pvals = np.full((5, 5), np.nan)
    
    # Insert values back into the original shape
    final_tvals.reshape(-1)[non_zero_variance_columns] = tvals
    final_pvals.reshape(-1)[non_zero_variance_columns] = pvals
    
    # print("Mapped t-values (5x5):")
    # print(final_tvals)
    
    # print("Mapped p-values (5x5):")
    # print(final_pvals)
    
    return final_tvals, final_pvals

## Stage transition plots 
transitions = pd.read_csv(stats_path + 'df_trans.csv', index_col=0).drop(columns=['MixingTime','Stability'])

#%%

# Filter subjects who have observations in both modes
subjects_pn = set(transitions[transitions['Mode'] == 'pn']['Subject'])
subjects_nmes = set(transitions[transitions['Mode'] == 'nmes']['Subject'])
common_subjects = subjects_pn.intersection(subjects_nmes)

df_pn = transitions[(transitions['Mode'] == 'pn') & (transitions['Subject'].isin(common_subjects))]
df_nmes = transitions[(transitions['Mode'] == 'nmes') & (transitions['Subject'].isin(common_subjects))]

transition_matrices_pn = compute_transition_matrix(df_pn, common_subjects)
transition_matrices_nmes = compute_transition_matrix(df_nmes, common_subjects)

percentage_differences = {}
percentage_differences_nodes = {}
for subject in common_subjects:
    matrix_pn = transition_matrices_pn[subject]
    matrix_nmes = transition_matrices_nmes[subject]
    # Add a small constant to avoid division by zero and compute log ratio
    epsilon = 1e-10
    # Calculate changes for each node (sleep stage) using log ratios of diagonal elements
    percentage_differences[subject] = np.log((matrix_nmes + epsilon) / (matrix_pn + epsilon)) #* 100
    # percentage_differences[subject] = 100 * (matrix_nmes - matrix_pn) / (matrix_pn + 1e-10) 
    percentage_differences_nodes[subject] = np.log((matrix_nmes.diagonal() + epsilon) / (matrix_pn.diagonal() + epsilon)) #* 100
    # percentage_differences_nodes[subject] = ((matrix_nmes.diagonal() + epsilon) / (matrix_pn.diagonal() + epsilon)) * 100

# Convert the percentage differences to a list of arrays
percentage_diff_list = [percentage_differences[subject] for subject in common_subjects]
percentage_diff_list_nodes = [percentage_differences_nodes[subject] for subject in common_subjects]

# Stack the percentage differences for permutation testing
stacked_diffs = np.stack(percentage_diff_list)
stacked_diffs_nodes = np.stack(percentage_diff_list_nodes)

# Stats
final_tvals, final_pvals = transition_matrix_stats(stacked_diffs)

# Plot the final results for visualization
labels = ['Wake','N1','N2','N3','REM']
sns.heatmap(final_tvals, cmap='Blues',
            xticklabels=labels, yticklabels=labels, 
            mask=None, square=True)
plt.title('Sleep Transition Matrix Heatmap (T-Vals)')
plt.show()

# Plot network 
# plot_stage_transition_nodes1(stacked_diffs.mean(0))
# plot_stage_transition_nodes(stacked_diffs.mean(0), stacked_diffs_nodes.mean(0))
