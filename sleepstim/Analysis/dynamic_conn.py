#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:20:41 2022

@author: administrator
"""

from dyconnmap import tvfcg
from dyconnmap.fc import IPLV

fb = [9,16]
fs = 128 
cc = 4.0
step = 80

estimator = IPLV(fb, fs)

eeg_eyes_open = epoch.get_data(picks='eeg',units='uV')
X = np.squeeze(eeg_eyes_open[0])
fcgs = tvfcg(X, estimator, fb, fs, cc, step)
fcgs_eyes_open = np.array(np.real(fcgs))

for i in tqdm(range(1, eeg_eyes_open.shape[0])):
    X = np.squeeze(eeg_eyes_open[i])
    fcgs = tvfcg(X, estimator, fb, fs, cc, step)
        
    fcgs_eyes_open = np.vstack([fcgs_eyes_open, np.real(fcgs)])
    
    
eeg_eyes_closed = epoch2.get_data(picks='eeg',units='uV')
X = np.squeeze(eeg_eyes_closed[0])
fcgs = tvfcg(X, estimator, fb, fs, cc, step)

fcgs_eyes_closed = np.array(np.real(fcgs))

for i in tqdm(range(1, eeg_eyes_closed.shape[0])):
    X = np.squeeze(eeg_eyes_closed[i])
    fcgs = tvfcg(X, estimator, fb, fs, cc, step)
        
    fcgs_eyes_closed = np.vstack([fcgs_eyes_closed, np.real(fcgs)])
    
from dyconnmap.cluster import NeuralGas

num_fcgs_eo, _, _ = np.shape(fcgs_eyes_open)
num_fcgs_ec, _, _ = np.shape(fcgs_eyes_closed)


fcgs = np.vstack([fcgs_eyes_open, fcgs_eyes_closed])
num_fcgs, num_channels, num_channels = np.shape(fcgs)

triu_ind = np.triu_indices_from(np.squeeze(fcgs[0, ...]), k=1)

fcgs = fcgs[:, triu_ind[0], triu_ind[1]]


rng = np.random.RandomState(0)

mdl = NeuralGas(n_protos=5, rng=rng).fit(fcgs)
encoding, symbols = mdl.encode(fcgs)



grp_dist_eo = symbols[0:num_fcgs_eo]
grp_dist_ec = symbols[num_fcgs_eo:]

h_grp_dist_eo = np.histogram(grp_dist_eo, bins=mdl.n_protos, normed=True)
h_grp_dist_ec = np.histogram(grp_dist_ec, bins=mdl.n_protos, normed=True)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

ind = np.arange(mdl.n_protos)
p1 = ax.bar(ind - 0.125, h_grp_dist_ec[0], 0.25, label='Sham-targeted')
p2 = ax.bar(ind + 0.125, h_grp_dist_eo[0], 0.25, label='Up-targeted')

ax.legend()
ax.set_xlabel('Symbol Index')
ax.set_ylabel('Hits %')
ax.set_xticks(np.arange(mdl.n_protos))
plt.show()



protos_mtx = np.zeros((mdl.n_protos, 23, 23))

for i in range(mdl.n_protos):
    symbol_state = np.zeros((23, 23))
    symbol_state[triu_ind] = mdl.protos[i, :]
    symbol_state = symbol_state + symbol_state.T
    np.fill_diagonal(symbol_state, 1.0)
    
    protos_mtx[i, :, :] = symbol_state

mtx_min = np.min(protos_mtx)
mtx_max = np.max(protos_mtx)

f, ax = plt.subplots(ncols=mdl.n_protos, figsize=(12, 12))
for i in range(mdl.n_protos):
    cax = ax[i].imshow(np.squeeze(protos_mtx[i,...]), vmin=mtx_min, vmax=mtx_max, cmap=plt.cm.Spectral)
    ax[i].set_title('#{0}'.format(i))

# move the colorbar to the side ;)
f.subplots_adjust(right=0.8)
cbar_ax = f.add_axes([0.82, 0.445, 0.0125, 0.115])
cb = f.colorbar(cax, cax=cbar_ax)
cb.set_label('Imaginary PLV')

#%%

grp_sym_eo = np.array_split(grp_dist_eo, 10, axis=0)
grp_sym_ec = np.array_split(grp_dist_ec, 10, axis=0)


subj1_eyes_open = grp_sym_eo[0]
subj1_eyes_closed = grp_sym_ec[0]

from dyconnmap.ts import markov_matrix

markov_matrix_eo = markov_matrix(subj1_eyes_open)
markov_matrix_ec = markov_matrix(subj1_eyes_closed)

from mpl_toolkits.axes_grid1 import ImageGrid
f = plt.figure(figsize=(8, 6))
grid = ImageGrid(f, 111,
                 nrows_ncols=(1,2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
im = grid[0].imshow(markov_matrix_eo, vmin=0.0, vmax=.15, cmap=plt.cm.Spectral)
grid[0].set_xlabel('Prototype')
grid[0].set_ylabel('Prototype')
grid[0].set_title('Up-targeted')

im = grid[1].imshow(markov_matrix_ec, vmin=0.0, vmax=.15, cmap=plt.cm.Spectral)
grid[1].set_xlabel('Prototype')
grid[1].set_ylabel('Prototype')
grid[1].set_title('Sham-targeted')

cb = grid[1].cax.colorbar(im)
cax = grid.cbar_axes[0]
axis = cax.axis[cax.orientation]
axis.label.set_text("Transition Probability")

plt.show()

from dyconnmap.ts import transition_rate, occupancy_time

tr_eo = transition_rate(subj1_eyes_open)
tr_ec = transition_rate(subj1_eyes_closed)

print(f"""
Transition rate
===============
  Up-targeted: {tr_eo:.3f}
  Sham-targeted: {tr_ec:.3f}
""")

occ_eo = occupancy_time(subj1_eyes_open)[0]
occ_ec = occupancy_time(subj1_eyes_closed)[0]

print("""
Occupancy time
==============
      State \t 0 \t 1 \t 2 \t 3 \t 4
      -----
  Up-targeted \t {0:.3f} \t {1:.3f} \t {2:.3f} \t {3:.3f} \t {4:.3f}
  Sham-targeted \t {5:.3f} \t {6:.3f} \t {7:.3f} \t {8:.3f} \t {9:.3f}
""".format(*occ_eo, *occ_ec))

