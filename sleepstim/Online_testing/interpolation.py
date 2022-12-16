#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:13:57 2022

@author: administrator
"""

import mne
from mne.channels.interpolation import *
import numpy as np
inst = mne.read_epochs('/media/administrator/data/Study_1_data/ex_epo.fif')

def interpolate_bads_eeg(inst, origin, exclude=None, verbose=None):
    if exclude is None:
        exclude = list()
    bads_idx = np.zeros(len(inst.ch_names), dtype=bool)
    goods_idx = np.zeros(len(inst.ch_names), dtype=bool)

    picks = pick_types(inst.info, meg=False, eeg=True, exclude=exclude)
    picks = inst.picks(eeg=True, exclude=exclude)
    inst.info._check_consistency()
    bads_idx[picks] = [inst.ch_names[ch] in inst.info['bads'] for ch in picks]

    if len(picks) == 0 or bads_idx.sum() == 0:
        return

    goods_idx[picks] = True
    goods_idx[bads_idx] = False

    pos = inst._get_channel_positions(picks)

    # Make sure only EEG are used
    bads_idx_pos = bads_idx[picks]
    goods_idx_pos = goods_idx[picks]

    # test spherical fit
    distance = np.linalg.norm(pos - origin, axis=-1)
    distance = np.mean(distance / np.mean(distance))
    if np.abs(1. - distance) > 0.1:
        warn('Your spherical fit is poor, interpolation results are '
             'likely to be inaccurate.')

    pos_good = pos[goods_idx_pos] - origin
    pos_bad = pos[bads_idx_pos] - origin
    logger.info('Computing interpolation matrix from {} sensor '
                'positions'.format(len(pos_good)))
    interpolation = _make_interpolation_matrix(pos_good, pos_bad)

    logger.info('Interpolating {} sensors'.format(len(pos_bad)))
    _do_interp_dots(inst, interpolation, goods_idx, bads_idx)
    
def make_interpolation_matrix(pos_from, pos_to, alpha=1e-5):
    """Compute interpolation matrix based on spherical splines.
    Implementation based on [1]
    Parameters
    ----------
    pos_from : np.ndarray of float, shape(n_good_sensors, 3)
        The positions to interpoloate from.
    pos_to : np.ndarray of float, shape(n_bad_sensors, 3)
        The positions to interpoloate.
    alpha : float
        Regularization parameter. Defaults to 1e-5.
    Returns
    -------
    interpolation : np.ndarray of float, shape(len(pos_from), len(pos_to))
        The interpolation matrix that maps good signals to the location
        of bad signals.
    References
    ----------
    [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
        Spherical splines for scalp potential and current density mapping.
        Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
    """
    from scipy import linalg
    pos_from = pos_from.copy()
    pos_to = pos_to.copy()
    n_from = pos_from.shape[0]
    n_to = pos_to.shape[0]

    # normalize sensor positions to sphere
    _normalize_vectors(pos_from)
    _normalize_vectors(pos_to)

    # cosine angles between source positions
    cosang_from = pos_from.dot(pos_from.T)
    cosang_to_from = pos_to.dot(pos_from.T)
    G_from = _calc_g(cosang_from)
    G_to_from = _calc_g(cosang_to_from)
    assert G_from.shape == (n_from, n_from)
    assert G_to_from.shape == (n_to, n_from)

    if alpha is not None:
        G_from.flat[::len(G_from) + 1] += alpha

    C = np.vstack([np.hstack([G_from, np.ones((n_from, 1))]),
                   np.hstack([np.ones((1, n_from)), [[0]]])])
    C_inv = linalg.pinv(C)

    interpolation = np.hstack([G_to_from, np.ones((n_to, 1))]) @ C_inv[:, :-1]
    assert interpolation.shape == (n_to, n_from)
    return interpolation

def do_interp_dots(data, interpolation, goods_idx, bads_idx):
    """Dot product of channel mapping matrix to channel data."""
    interpolated_data[..., bads_idx, :] = np.matmul(
        interpolation, data[..., goods_idx, :])