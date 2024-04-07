#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:57:17 2024

@author: administrator
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
import yasa

hypno_path = '/media/administrator/data/Study_2_data/NIDRA/CLNMES/Hypnograms/' 
hypno = np.load(hypno_path + 'IsEb_1_hypno30s.npy')

counts, probs = yasa.transition_matrix(hypno)

# Sleep stage transition matrix
transition_matrix = list(probs.to_numpy().round(2))

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
edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20,
                               edge_cmap=plt.cm.Blues, width=2)

# Draw edge labels (transition probabilities)
edge_labels = {(i, j): f'{G[i][j]["weight"]:.2f}' for i, j in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)

# Show plot
plt.axis('off')
plt.tight_layout()
plt.show()
