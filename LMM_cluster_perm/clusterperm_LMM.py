#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:45:10 2020

@author: marius
"""
import mne.stats
import numpy as np

mne.stats.permutation_cluster_test
mne.stats.parametric.ttest_1samp_no_p

from statsmodels.formula.api import mixedlm

from mne.stats.cluster_level import _dep_con, _check_fun, warn, check_n_jobs, _check_option, \
    _setup_adjacency, _validate_type, logger, _get_partitions_from_adjacency, _find_clusters, \
    _cluster_indices_to_mask, _cluster_mask_to_indices, check_random_state, \
    _do_1samp_permutations, _get_1samp_orders, _do_permutations, parallel_func, \
    ProgressBar, split_list, _pl, _pval_from_histogram, _reshape_clusters
 

import analyze_ephys.statistical.generalized_linear_models as glm


def mixedlm_cond_grp(data, cond, grouping):
    '''
    Testing rendom effects model contrast between a and b
    '''
    # n_times = data.shape[1]
    # # a=pd.DataFrame.from_dict(data)
    # t_obs2 = np.nan*np.zeros(n_times)
    # for t in range(n_times):
    #     dat_dict = {'dat': data[:,t], 'cond': cond, 'subj': grouping}
    #     a=mixedlm("dat ~ cond", data = dat_dict, groups = dat_dict['subj']).fit()
    #     A = np.identity(len(a.params))
    #     A = A[1:2,:] #test condition factor against zero
    #     b=a.f_test(A)
    #     t_obs2[t] = b.fvalue
    #     print(b.summary)
    
    n_times = data.shape[1]
    data_label = ["data", "cond", "subj"]
    glm_model_type = "gaussian"
    glm_contrasts = " "
    glm_factor_types = ["continuous", "categorical",  "categorical"]
    glm_formula = 'data ~ cond + (1|subj)'
    # aa = glm.anova(stat_results[6])
    t_obs = np.nan*np.zeros(n_times)
    for t in range(n_times):
        d = np.vstack((data[:,t], cond, grouping)).transpose()
        stat_results = glm.run(data = d, label_name= data_label, factor_type = glm_factor_types, 
                               formula = glm_formula, contrasts = glm_contrasts, data_type = glm_model_type)
        (final_scores, final_df, final_p_values, final_coefficients, final_std_error, final_anova_factors, lm0, lsm_res) = stat_results
        aa = glm.anova(lm0, silent = True)
        f_val_id = [aa.colnames[i] == 'F value' for i in range(len(aa.colnames))]
        t_obs[t] = np.array(aa)[f_val_id]
    return t_obs
        
def permutation_cluster_test_multilevel(
        X, threshold=None, n_permutations=1024, tail=0, stat_fun_ml=None,
        mm_groups=None,
        adjacency=None, n_jobs=1, seed=None, max_step=1, exclude=None,
        step_down_p=0, t_power=1, out_type=None, check_disjoint=False,
        buffer_size=1000, connectivity=None, verbose=None):
    """Cluster-level statistical permutation test.

    For a list of :class:`NumPy arrays <numpy.ndarray>` of data,
    calculate some statistics corrected for multiple comparisons using
    permutations and cluster-level correction. Each element of the list ``X``
    should contain the data for one group of observations (e.g., 2D arrays for
    time series, 3D arrays for time-frequency power values). Permutations are
    generated with random partitions of the data. See
    :footcite:`MarisOostenveld2007` for details.

    Parameters
    ----------
    X : list of array, shape (n_observations, p[, q])
        The data to be clustered. Each array in ``X`` should contain the
        observations for one group. The first dimension of each array is the
        number of observations from that group; remaining dimensions comprise
        the size of a single observation. For example if ``X = [X1, X2]``
        with ``X1.shape = (20, 50, 4)`` and ``X2.shape = (17, 50, 4)``, then
        ``X`` has 2 groups with respectively 20 and 17 observations in each,
        and each data point is of shape ``(50, 4)``. Note: that the
        *last dimension* of each element of ``X`` should correspond to the
        dimension represented in the ``adjacency`` parameter
        (e.g., spectral data should be provided as
        ``(observations, frequencies, channels/vertices)``).
    mm_groups : list of array, shape (n_observations,)
                contains for each observation the index of observation it
                was coming from. In the above example, assuming that there
                are 10 subjects in group 1 and 5 subjects in gorup 2, then 
                "mm_group = [mm_group1, mm_group2]" with "mm_group1.shape = (20,)"
                and "mm_group1.shape = (17,)"
    %(clust_thresh_f)s
    %(clust_nperm_int)s
    %(clust_tail)s
    %(clust_stat_f)s
    %(clust_adj_n)s
    %(n_jobs)s
    %(seed)s
    %(clust_maxstep)s
    exclude : bool array or None
        Mask to apply to the data to exclude certain points from clustering
        (e.g., medial wall vertices). Should be the same shape as X. If None,
        no points are excluded.
    %(clust_stepdown)s
    %(clust_power_f)s
    %(clust_out_none)s
    %(clust_disjoint)s
    %(clust_buffer)s
    %(clust_con_dep)s
    %(verbose)s

    Returns
    -------
    F_obs : array, shape (n_tests,)
        Statistic (F by default) observed for all variables.
    clusters : list
        List type defined by out_type above.
    cluster_pv : array
        P-value for each cluster.
    H0 : array, shape (n_permutations,)
        Max cluster level stats observed under permutation.

    References
    ----------
    .. footbibliography::
    """
    adjacency = _dep_con(adjacency, connectivity)
    stat_fun_ml, threshold = _check_fun(X, stat_fun_ml, threshold, tail, 'between')
    if out_type is None:
        warn('The default for "out_type" will change from "mask" to "indices" '
             'in version 0.22. To avoid this warning, explicitly set '
             '"out_type" to one of its string values.', DeprecationWarning)
        out_type = 'mask'
    return _permutation_cluster_test_multilevel(
        X=X, threshold=threshold, n_permutations=n_permutations, tail=tail,
        stat_fun_ml=stat_fun_ml, mm_groups=mm_groups,
        adjacency=adjacency, n_jobs=n_jobs, seed=seed,
        max_step=max_step, exclude=exclude, step_down_p=step_down_p,
        t_power=t_power, out_type=out_type, check_disjoint=check_disjoint,
        buffer_size=buffer_size)


def _permutation_cluster_test_multilevel(X, threshold, n_permutations, tail, stat_fun_ml, mm_groups,
                              adjacency, n_jobs, seed, max_step,
                              exclude, step_down_p, t_power, out_type,
                              check_disjoint, buffer_size):
    n_jobs = check_n_jobs(n_jobs)
    """Aux Function.

    Note. X is required to be a list. Depending on the length of X
    either a 1 sample t-test or an F test / more sample permutation scheme
    is elicited.
    """
    _check_option('out_type', out_type, ['mask', 'indices'])
    _check_option('tail', tail, [-1, 0, 1])
    if not isinstance(threshold, dict):
        threshold = float(threshold)
        if (tail < 0 and threshold > 0 or tail > 0 and threshold < 0 or
                tail == 0 and threshold < 0):
            raise ValueError('incompatible tail and threshold signs, got '
                             '%s and %s' % (tail, threshold))

    # check dimensions for each group in X (a list at this stage).
    X = [x[:, np.newaxis] if x.ndim == 1 else x for x in X]
    n_samples = X[0].shape[0]
    n_times = X[0].shape[1]

    sample_shape = X[0].shape[1:]
    for x in X:
        if x.shape[1:] != sample_shape:
            raise ValueError('All samples mush have the same size')

    # flatten the last dimensions in case the data is high dimensional
    X = [np.reshape(x, (x.shape[0], -1)) for x in X]
    n_tests = X[0].shape[1]
    
    
    if adjacency is not None and adjacency is not False:
        adjacency = _setup_adjacency(adjacency, n_tests, n_times)

    if (exclude is not None) and not exclude.size == n_tests:
        raise ValueError('exclude must be the same shape as X[0]')

    # Step 1: Calculate t-stat for original data
    # -------------------------------------------------------------
    cond = []
    for c, x in enumerate(X):
        cond.append(c*np.ones(x.shape[0]))
    t_obs = stat_fun_ml(np.vstack(X), np.hstack(cond), np.hstack(mm_groups))
    # t_obs = stat_fun(*X)
    _validate_type(t_obs, np.ndarray, 'return value of stat_fun')
    logger.info('stat_fun(H1): min=%f max=%f' % (np.min(t_obs), np.max(t_obs)))

    # # test if stat_fun treats variables independently
    # if buffer_size is not None:
    #     t_obs_buffer = np.zeros_like(t_obs)
    #     for pos in range(0, n_tests, buffer_size):
    #         t_obs_buffer[pos: pos + buffer_size] =\
    #             stat_fun(*[x[:, pos: pos + buffer_size] for x in X])

    #     if not np.alltrue(t_obs == t_obs_buffer):
    #         warn('Provided stat_fun does not treat variables independently. '
    #              'Setting buffer_size to None.')
    #         buffer_size = None

    # The stat should have the same shape as the samples for no adj.
    if t_obs.size != np.prod(sample_shape):
        raise ValueError('t_obs.shape %s provided by stat_fun_ml %s is not '
                         'compatible with the sample shape %s'
                         % (t_obs.shape, stat_fun_ml, sample_shape))
    if adjacency is None or adjacency is False:
        t_obs.shape = sample_shape

    if exclude is not None:
        include = np.logical_not(exclude)
    else:
        include = None

    # determine if adjacency itself can be separated into disjoint sets
    if check_disjoint is True and (adjacency is not None and
                                   adjacency is not False):
        partitions = _get_partitions_from_adjacency(adjacency, n_times)
    else:
        partitions = None
    logger.info('Running initial clustering')
    out = _find_clusters(t_obs, threshold, tail, adjacency,
                         max_step=max_step, include=include,
                         partitions=partitions, t_power=t_power,
                         show_info=True)
    clusters, cluster_stats = out

    # The stat should have the same shape as the samples
    t_obs.shape = sample_shape

    # For TFCE, return the "adjusted" statistic instead of raw scores
    if isinstance(threshold, dict):
        t_obs = cluster_stats.reshape(t_obs.shape) * np.sign(t_obs)

    logger.info('Found %d clusters' % len(clusters))

    # convert clusters to old format
    if adjacency is not None and adjacency is not False:
        # our algorithms output lists of indices by default
        if out_type == 'mask':
            clusters = _cluster_indices_to_mask(clusters, n_tests)
    else:
        # ndimage outputs slices or boolean masks by default
        if out_type == 'indices':
            clusters = _cluster_mask_to_indices(clusters)

    # convert our seed to orders
    # check to see if we can do an exact test
    # (for a two-tailed test, we can exploit symmetry to just do half)
    extra = ''
    rng = check_random_state(seed)
    del seed
    # if len(X) == 1:  # 1-sample test
    #     do_perm_func = _do_1samp_permutations_multilevel
    #     X_full = X[0]
    #     slices = None
    #     orders, n_permutations, extra = _get_1samp_orders(
    #         n_samples, n_permutations, tail, rng)
    # else:
    #     n_permutations = int(n_permutations)
    #     do_perm_func = _do_permutations_multilevel
    #     X_full = np.concatenate(X, axis=0)
    #     n_samples_per_condition = [x.shape[0] for x in X]
    #     splits_idx = np.append([0], np.cumsum(n_samples_per_condition))
    #     slices = [slice(splits_idx[k], splits_idx[k + 1])
    #               for k in range(len(X))]
    #     orders = [rng.permutation(len(X_full))
    #               for _ in range(n_permutations - 1)]
    
    n_samples_per_condition = [x.shape[0] for x in X]
    splits_idx = np.append([0], np.cumsum(n_samples_per_condition))
    slices = [slice(splits_idx[k], splits_idx[k + 1])
              for k in range(len(X))]
    
    
    # permuting samples within groups only
    n_permutations = int(n_permutations)
    do_perm_func = _do_permutations_multilevel
    X_full = np.concatenate(X, axis=0)
    orders = [np.nan*np.ones(len(X_full)) for _ in range(n_permutations)]
    mm_groups_full = np.hstack(mm_groups)
    for i in range(n_permutations):
        all_ids = np.arange(len(X_full))
        for iG in np.unique(mm_groups_full): # go through all groups
            grp_ids = all_ids[mm_groups_full==iG]
            orders[i][mm_groups_full==iG] = grp_ids[rng.permutation(int(sum(mm_groups_full==iG)))]
        
        orders[i] = orders[i].astype(np.int64)
    
    del rng
    parallel, my_do_perm_func, _ = parallel_func(
        do_perm_func, n_jobs, verbose=False)

    if len(clusters) == 0:
        warn('No clusters found, returning empty H0, clusters, and cluster_pv')
        return t_obs, np.array([]), np.array([]), np.array([])

    # Step 2: If we have some clusters, repeat process on permuted data
    # -------------------------------------------------------------------
    # Step 3: repeat permutations for step-down-in-jumps procedure
    n_removed = 1  # number of new clusters added
    total_removed = 0
    step_down_include = None  # start out including all points
    n_step_downs = 0

    while n_removed > 0:
        # actually do the clustering for each partition
        if include is not None:
            if step_down_include is not None:
                this_include = np.logical_and(include, step_down_include)
            else:
                this_include = include
        else:
            this_include = step_down_include
        logger.info('Permuting %d times%s...' % (len(orders), extra))
        with ProgressBar(len(orders)) as progress_bar:
            H0 = parallel(
                my_do_perm_func(X_full, slices, threshold, tail, adjacency,
                                stat_fun_ml, mm_groups, max_step, this_include, partitions,
                                t_power, order, sample_shape, buffer_size,
                                progress_bar.subset(idx))
                for idx, order in split_list(orders, n_jobs, idx=True))
        # include original (true) ordering
        if tail == -1:  # up tail
            orig = cluster_stats.min()
        elif tail == 1:
            orig = cluster_stats.max()
        else:
            orig = abs(cluster_stats).max()
        H0.insert(0, [orig])
        H0 = np.concatenate(H0)
        logger.info('Computing cluster p-values')
        cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)

        # figure out how many new ones will be removed for step-down
        to_remove = np.where(cluster_pv < step_down_p)[0]
        n_removed = to_remove.size - total_removed
        total_removed = to_remove.size
        step_down_include = np.ones(n_tests, dtype=bool)
        for ti in to_remove:
            step_down_include[clusters[ti]] = False
        if adjacency is None and adjacency is not False:
            step_down_include.shape = sample_shape
        n_step_downs += 1
        if step_down_p > 0:
            a_text = 'additional ' if n_step_downs > 1 else ''
            logger.info('Step-down-in-jumps iteration #%i found %i %s'
                        'cluster%s to exclude from subsequent iterations'
                        % (n_step_downs, n_removed, a_text,
                           _pl(n_removed)))
    logger.info('Done.')
    # The clusters should have the same shape as the samples
    clusters = _reshape_clusters(clusters, sample_shape)
    return t_obs, clusters, cluster_pv, H0

def _do_permutations_multilevel(X_full, slices, threshold, tail, adjacency, stat_fun_ml, mm_groups,
                     max_step, include, partitions, t_power, orders,
                     sample_shape, buffer_size, progress_bar):
    n_samp, n_vars = X_full.shape

    if buffer_size is not None and n_vars <= buffer_size:
        buffer_size = None  # don't use buffer for few variables

    # allocate space for output
    max_cluster_sums = np.empty(len(orders), dtype=np.double)

    if buffer_size is not None:
        # allocate buffer, so we don't need to allocate memory during loop
        X_buffer = [np.empty((len(X_full[s]), buffer_size), dtype=X_full.dtype)
                    for s in slices]
        mm_groups_buffer = [np.empty((len(np.hstack(mm_groups)[s]), buffer_size), dtype=X_full.dtype)
                    for s in slices]

    for seed_idx, order in enumerate(orders):
        # shuffle sample indices
        assert order is not None
        idx_shuffle_list = [order[s] for s in slices]

        if buffer_size is None:
            # shuffle all data at once
            X_shuffle_list = [X_full[idx, :] for idx in idx_shuffle_list]
            mm_groups_shuffle_list = [np.hstack(mm_groups)[idx] for idx in idx_shuffle_list]
            
            assert np.all([np.all(mm_groups[i] == mm_groups_shuffle_list[i]) for i in range(len(mm_groups))]), "something went wrong when shuffling groups"
            
            cond = []
            for c, x in enumerate(X_shuffle_list):
                cond.append(c*np.ones(x.shape[0]))
            t_obs_surr = stat_fun_ml(np.vstack(X_shuffle_list), np.hstack(cond), np.hstack(mm_groups_shuffle_list))
            
            # t_obs_surr = stat_fun(*X_shuffle_list)
        else:
            # only shuffle a small data buffer, so we need less memory
            t_obs_surr = np.empty(n_vars, dtype=X_full.dtype)

            for pos in range(0, n_vars, buffer_size):
                # number of variables for this loop
                n_var_loop = min(pos + buffer_size, n_vars) - pos

                # fill buffer
                for i, idx in enumerate(idx_shuffle_list):
                    X_buffer[i][:, :n_var_loop] =\
                        X_full[idx, pos: pos + n_var_loop]
                    mm_groups_buffer[i][:, :n_var_loop] =\
                        np.hstack(mm_groups)[idx, pos: pos + n_var_loop]

                # apply stat_fun and store result
                cond = []
                for c, x in enumerate(X_buffer):
                    cond.append(c*np.ones(x.shape[0]))
                tmp = stat_fun_ml(np.vstack(X_buffer), np.hstack(cond), np.hstack(mm_groups_buffer))
                # tmp = stat_fun(*X_buffer)
                t_obs_surr[pos: pos + n_var_loop] = tmp[:n_var_loop]

        # The stat should have the same shape as the samples for no adj.
        if adjacency is None:
            t_obs_surr.shape = sample_shape

        # Find cluster on randomized stats
        out = _find_clusters(t_obs_surr, threshold=threshold, tail=tail,
                             max_step=max_step, adjacency=adjacency,
                             partitions=partitions, include=include,
                             t_power=t_power)
        perm_clusters_sums = out[1]

        if len(perm_clusters_sums) > 0:
            max_cluster_sums[seed_idx] = np.max(perm_clusters_sums)
        else:
            max_cluster_sums[seed_idx] = 0

        progress_bar.update(seed_idx + 1)

    return max_cluster_sums


def _do_1samp_permutations_multilevel(X, slices, threshold, tail, adjacency, stat_fun_ml, mm_groups,
                           max_step, include, partitions, t_power, orders,
                           sample_shape, buffer_size, progress_bar):
    n_samp, n_vars = X.shape
    assert slices is None  # should be None for the 1 sample case

    if buffer_size is not None and n_vars <= buffer_size:
        buffer_size = None  # don't use buffer for few variables

    # allocate space for output
    max_cluster_sums = np.empty(len(orders), dtype=np.double)

    if buffer_size is not None:
        # allocate a buffer so we don't need to allocate memory in loop
        X_flip_buffer = np.empty((n_samp, buffer_size), dtype=X.dtype)

    for seed_idx, order in enumerate(orders):
        assert isinstance(order, np.ndarray)
        # new surrogate data with specified sign flip
        assert order.size == n_samp  # should be guaranteed by parent
        signs = 2 * order[:, None].astype(int) - 1
        if not np.all(np.equal(np.abs(signs), 1)):
            raise ValueError('signs from rng must be +/- 1')

        if buffer_size is None:
            # be careful about non-writable memmap (GH#1507)
            if X.flags.writeable:
                X *= signs
                # Recompute statistic on randomized data
                cond = []
                for c, x in enumerate(X):
                    cond.append(c*np.ones(x.shape[0]))
                t_obs_surr = stat_fun_ml(np.vstack(X), np.hstack(cond), np.hstack(mm_groups)) # no need to shuffle groups, because I randomly shuffled the X signs, group assignment remains the same
                # t_obs_surr = stat_fun(X)
                # Set X back to previous state (trade memory eff. for CPU use)
                X *= signs
            else:
                cond = []
                for c, x in enumerate(X):
                    cond.append(c*np.ones(x.shape[0]))
                t_obs_surr = stat_fun_ml(np.vstack(X * signs), np.hstack(cond), np.hstack(mm_groups)) # no need to shuffle groups, because I randomly shuffled the X signs, group assignment remains the same
                # t_obs_surr = stat_fun(X * signs)
        else:
            # only sign-flip a small data buffer, so we need less memory
            t_obs_surr = np.empty(n_vars, dtype=X.dtype)

            for pos in range(0, n_vars, buffer_size):
                # number of variables for this loop
                n_var_loop = min(pos + buffer_size, n_vars) - pos

                X_flip_buffer[:, :n_var_loop] =\
                    signs * X[:, pos: pos + n_var_loop]
                    
                # apply stat_fun and store result
                cond = []
                for c, x in enumerate(X_flip_buffer):
                    cond.append(c*np.ones(x.shape[0]))
                tmp = stat_fun_ml(np.vstack(X_flip_buffer), np.hstack(cond), np.hstack(mm_groups))
                
                # tmp = stat_fun(X_flip_buffer)
                t_obs_surr[pos: pos + n_var_loop] = tmp[:n_var_loop]

        # The stat should have the same shape as the samples for no adj.
        if adjacency is None:
            t_obs_surr.shape = sample_shape

        # Find cluster on randomized stats
        out = _find_clusters(t_obs_surr, threshold=threshold, tail=tail,
                             max_step=max_step, adjacency=adjacency,
                             partitions=partitions, include=include,
                             t_power=t_power)
        perm_clusters_sums = out[1]
        if len(perm_clusters_sums) > 0:
            # get max with sign info
            idx_max = np.argmax(np.abs(perm_clusters_sums))
            max_cluster_sums[seed_idx] = perm_clusters_sums[idx_max]
        else:
            max_cluster_sums[seed_idx] = 0

        progress_bar.update(seed_idx + 1)

    return max_cluster_sums

# def permutation_cluster_test_multilevel(X, design = None, threshold=5, n_permutations=1024, tail=0, stat_fun=None,
#         adjacency=None, n_jobs=1, seed=None, max_step=1, exclude=None,
#         step_down_p=0, t_power=1, out_type='mask', check_disjoint=False,
#         buffer_size=1000, connectivity=None, verbose=None):
    
#     """Cluster-level statistical permutation test.
    
#     copied and modified from mne.stats.cluster_level
    
#      For a list of :class:`NumPy arrays <numpy.ndarray>` of data,
#      calculate some statistics corrected for multiple comparisons using
#      permutations and cluster-level correction. Each element of the list ``X``
#      should contain the data for one group of observations (e.g., 2D arrays for
#      time series, 3D arrays for time-frequency power values). Permutations are
#      generated with random partitions of the data. See
#      :footcite:`MarisOostenveld2007` for details.
    
#      Parameters
#      ----------
#      X : list of array, shape (n_observations, p[, q])
#          The data to be clustered. Each array in ``X`` should contain the
#          observations for one group. The first dimension of each array is the
#          number of observations from that group; remaining dimensions comprise
#          the size of a single observation. For example if ``X = [X1, X2]``
#          with ``X1.shape = (20, 50, 4)`` and ``X2.shape = (17, 50, 4)``, then
#          ``X`` has 2 groups with respectively 20 and 17 observations in each,
#          and each data point is of shape ``(50, 4)``. Note: that the
#          *last dimension* of each element of ``X`` should correspond to the
#          dimension represented in the ``adjacency`` parameter
#          (e.g., spectral data should be provided as
#          ``(observations, frequencies, channels/vertices)``).
#      %(clust_thresh_f)s
#      %(clust_nperm_int)s
#      %(clust_tail)s
#      %(clust_stat_f)s
#      %(clust_adj_n)s
#      %(n_jobs)s
#      %(seed)s
#      %(clust_maxstep)s
#      exclude : bool array or None
#          Mask to apply to the data to exclude certain points from clustering
#          (e.g., medial wall vertices). Should be the same shape as X. If None,
#          no points are excluded.
#      %(clust_stepdown)s
#      %(clust_power_f)s
#      %(clust_out_none)s
#      %(clust_disjoint)s
#      %(clust_buffer)s
#      %(clust_con_dep)s
#      %(verbose)s
    
#      Returns
#      -------
#      F_obs : array, shape (n_tests,)
#          Statistic (F by default) observed for all variables.
#      clusters : list
#          List type defined by out_type above.
#      cluster_pv : array
#          P-value for each cluster.
#      H0 : array, shape (n_permutations,)
#          Max cluster level stats observed under permutation.
    
#      References
#      ----------
#      .. footbibliography::
#      """
    

#     from mne.stats.cluster_level import _pl, _pval_from_histogram, split_list, ProgressBar, parallel_func, _do_permutations, _get_1samp_orders, _do_1samp_permutations, check_random_state, _cluster_mask_to_indices, _cluster_indices_to_mask, _reshape_clusters, check_n_jobs, _check_option, _setup_adjacency, _validate_type,_get_partitions_from_adjacency,_find_clusters
#     n_jobs = check_n_jobs(n_jobs)
#     _check_option('out_type', out_type, ['mask', 'indices'])
#     _check_option('tail', tail, [-1, 0, 1])
#     if not isinstance(threshold, dict):
#         threshold = float(threshold)
#         if (tail < 0 and threshold > 0 or tail > 0 and threshold < 0 or
#                 tail == 0 and threshold < 0):
#             raise ValueError('incompatible tail and threshold signs, got '
#                              '%s and %s' % (tail, threshold))

#     # check dimensions for each group in X (a list at this stage).
#     X = [x[:, np.newaxis] if x.ndim == 1 else x for x in X]
#     n_subjects = X[0].shape[0]
#     n_times = X[0].shape[-1]
#     sample_shape = X[0].shape[1:]
#     for x in X:
#         if x.shape[1:] != sample_shape:
#             raise ValueError('All samples mush have the same size')

#     # flatten the last dimensions in case the data is high dimensional
#     X = [np.reshape(x, (-1, x.shape[-1])) for x in X]
#     design = [d.reshape(-1,1) for d in design]

#     n_tests = X[0].shape[-1]

#     if adjacency is not None and adjacency is not False:
#         adjacency = _setup_adjacency(adjacency, n_tests, n_times)

#     if (exclude is not None) and not exclude.size == n_tests:
#         raise ValueError('exclude must be the same shape as X[0]')

#     # Step 1: Calculate t-stat for original data
#     # -------------------------------------------------------------
#     from statsmodels.formula.api import mixedlm
#     # a=pd.DataFrame.from_dict(data)
#     subj = np.concatenate(design)
#     t_obs = np.nan*np.zeros(n_times)
#     for t in range(n_times):
#         dat = []
#         cond = []
#         for i, x in enumerate(X):
#             dat.extend(x[:,t])
#             cond.extend([str(i)]*x.shape[0])
#         data = {'dat': dat, 'cond': cond, 'subj':np.squeeze(subj)}
#         a=mixedlm("dat ~ cond", data = data, groups = data['subj']).fit()
#         A = np.identity(len(a.params))
#         A = A[1:3,:] #test both fixed coefficients jointly against zero
#         b=a.f_test(A)
#         t_obs[t] = b.fvalue
#     _validate_type(t_obs, np.ndarray, 'return value of stat_fun')
#     print('stat_fun(H1): min=%f max=%f' % (np.min(t_obs), np.max(t_obs)))

#     # # test if stat_fun treats variables independently
#     # if buffer_size is not None:
#     #     t_obs_buffer = np.zeros_like(t_obs)
#     #     for pos in range(0, n_tests, buffer_size):
#     #         t_obs_buffer[pos: pos + buffer_size] =\
#     #             stat_fun(*[x[:, pos: pos + buffer_size] for x in X])

#     #     if not np.alltrue(t_obs == t_obs_buffer):
#     #         print('Provided stat_fun does not treat variables independently. '
#     #              'Setting buffer_size to None.')
#     #         buffer_size = None

#     # The stat should have the same shape as the samples for no adj.
#     # if t_obs.size != np.prod(sample_shape):
#     #     raise ValueError('t_obs.shape %s provided by stat_fun %s is not '
#     #                       'compatible with the sample shape %s'
#     #                       % (t_obs.shape, stat_fun, sample_shape))
#     # if adjacency is None or adjacency is False:
#     #     t_obs.shape = sample_shape

#     # if exclude is not None:
#     #     include = np.logical_not(exclude)
#     # else:
#     #     include = None

#     # # determine if adjacency itself can be separated into disjoint sets
#     # if check_disjoint is True and (adjacency is not None and
#     #                                 adjacency is not False):
#     #     partitions = _get_partitions_from_adjacency(adjacency, n_times)
#     # else:
#     #     partitions = None
#     # print('Running initial clustering')
#     out = _find_clusters(t_obs, threshold, tail, adjacency,
#                          max_step=max_step, include=None,
#                          partitions=None, t_power=t_power,
#                          show_info=True)
#     clusters, cluster_stats = out

#     # The stat should have the same shape as the samples
#     # t_obs.shape = sample_shape

#     # # For TFCE, return the "adjusted" statistic instead of raw scores
#     # if isinstance(threshold, dict):
#     #     t_obs = cluster_stats.reshape(t_obs.shape) * np.sign(t_obs)

#     print('Found %d clusters' % len(clusters))

#     # convert clusters to old format
#     if adjacency is not None and adjacency is not False:
#         # our algorithms output lists of indices by default
#         if out_type == 'mask':
#             clusters = _cluster_indices_to_mask(clusters, n_tests)
#     else:
#         # ndimage outputs slices or boolean masks by default
#         if out_type == 'indices':
#             clusters = _cluster_mask_to_indices(clusters)

#     # convert our seed to orders
#     # check to see if we can do an exact test
#     # (for a two-tailed test, we can exploit symmetry to just do half)
#     extra = ''
#     rng = check_random_state(seed)
#     del seed

#     n_permutations = int(n_permutations)
#     do_perm_func = _do_permutations
#     X_full = np.concatenate(X, axis=0)
#     n_samples_per_condition = [x.shape[0] for x in X]
#     splits_idx = np.append([0], np.cumsum(n_samples_per_condition))
#     slices = [slice(splits_idx[k], splits_idx[k + 1])
#               for k in range(len(X))]
#     orders = [rng.permutation(len(X_full))
#               for _ in range(n_permutations - 1)]
#     del rng
#     parallel, my_do_perm_func, _ = parallel_func(
#         do_perm_func, n_jobs, verbose=False)

#     if len(clusters) == 0:
#         print('No clusters found, returning empty H0, clusters, and cluster_pv')
#         return t_obs, np.array([]), np.array([]), np.array([])

#     # Step 2: If we have some clusters, repeat process on permuted data
#     # -------------------------------------------------------------------
#     # Step 3: repeat permutations for step-down-in-jumps procedure
    
    
    
    
#     n_removed = 1  # number of new clusters added
#     total_removed = 0
#     step_down_include = None  # start out including all points
#     n_step_downs = 0

#     while n_removed > 0:
#         # actually do the clustering for each partition
#         if include is not None:
#             if step_down_include is not None:
#                 this_include = np.logical_and(include, step_down_include)
#             else:
#                 this_include = include
#         else:
#             this_include = step_down_include
#         print('Permuting %d times%s...' % (len(orders), extra))
#         with ProgressBar(len(orders)) as progress_bar:
#             H0 = parallel(
#                 my_do_perm_func(X_full, slices, threshold, tail, adjacency,
#                                 stat_fun, max_step, this_include, partitions,
#                                 t_power, order, sample_shape, buffer_size,
#                                 progress_bar.subset(idx))
#                 for idx, order in split_list(orders, n_jobs, idx=True))
#         # include original (true) ordering
#         if tail == -1:  # up tail
#             orig = cluster_stats.min()
#         elif tail == 1:
#             orig = cluster_stats.max()
#         else:
#             orig = abs(cluster_stats).max()
#         H0.insert(0, [orig])
#         H0 = np.concatenate(H0)
#         print('Computing cluster p-values')
#         cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)

#         # figure out how many new ones will be removed for step-down
#         to_remove = np.where(cluster_pv < step_down_p)[0]
#         n_removed = to_remove.size - total_removed
#         total_removed = to_remove.size
#         step_down_include = np.ones(n_tests, dtype=bool)
#         for ti in to_remove:
#             step_down_include[clusters[ti]] = False
#         if adjacency is None and adjacency is not False:
#             step_down_include.shape = sample_shape
#         n_step_downs += 1
#         if step_down_p > 0:
#             a_text = 'additional ' if n_step_downs > 1 else ''
#             print('Step-down-in-jumps iteration #%i found %i %s'
#                         'cluster%s to exclude from subsequent iterations'
#                         % (n_step_downs, n_removed, a_text,
#                            _pl(n_removed)))
#     print('Done.')
#     # The clusters should have the same shape as the samples
#     clusters = _reshape_clusters(clusters, sample_shape)
#     return t_obs, clusters, cluster_pv, H0




# from __future__ import print_function

# import mne
# import os.path as op
# import numpy as np
# from matplotlib import pyplot as plt
# import mne
# import numpy as np

# # raw = mne.io.RawFIF(
# #     op.join(mne.datasets.sample.data_path(), 'MEG', 'sample',
# #             'sample_audvis_raw.fif'))

# # # If your raw object has a stim channel, you can construct an event array
# # # easily
# # events = mne.find_events(raw, stim_channel='STI 014')

# # # Show the number of events (number of rows)
# # print('Number of events:', len(events))

# # # Show all unique event codes (3rd column)
# # print('Unique event codes:', np.unique(events[:, 2]))

# # # Specify event codes of interest with descriptive labels
# # event_id = dict(left=1, right=2)

# # epochs = mne.Epochs(raw, events, event_id, tmin=-0.1, tmax=1,
# #                     baseline=(None, 0), preload=True)



# import numpy as np

# X= np.random.randn(60,3,5)
# X2 = [X[:20,:,:], X[20:40,:,:], X[40:60,:,:]] # X contains the data for each condition
# design = [np.tile(np.arange(20).reshape(-1, 1),[1, 376])]*3
# groups = [np.arange(20),np.arange(20),np.arange(20)] # the subject assignments for each condition

# # permutation_cluster_test_multilevel(X2, design)

# max_cluster_sums = permutation_cluster_test_multilevel(X2, stat_fun_ml=mixedlm_cond_grp, mm_groups=groups)
