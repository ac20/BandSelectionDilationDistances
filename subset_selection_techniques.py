"""
Code for various subset selection techniques

Notes
-----
There are several options for selecting the subsets given the distances. 
The minimax and maximim are good heuristics. However, in real datasets
due to the noise spectral clustering provided the best selection. Even
in spectral clustering approach, there are several other options for 
selecting the center for each cluster:

a) band which maximizes the minimum similarity among the cluster
b) band which maximizes the mean similarity among the cluster
c) band which maximizes the median similarity among the cluster
d) Identifying the mean is the one which minimizes SSE, we consider
   the band which minimizes the squares of distances.
e) The average distances could be distorted with the presence of
   outliers. So, we can the median of the distances instead.
"""

import numpy as np
import pdb

from sklearn.cluster import spectral_clustering


def brute_force_maximin_subset(DD_original, num_bands_select=3):
    """
    """
    DD = np.array(DD_original, copy=True)
    nbands = np.shape(DD)[0]
    diag = np.arange(nbands)
    DD[diag, diag] = 1e56

    maxval = -1*np.inf
    for i1 in range(nbands):
        for j1 in range(nbands):
            for k1 in range(nbands):
                if not(i1 == j1 or i1 == k1 or j1 == k1):
                    set_tmp = (i1, j1, k1)
                    val = np.min(DD[(i1, j1, k1), :][:, (i1, j1, k1)])
                    if val > maxval:
                        maxval = val
                        bands_select = set([i1, j1, k1])
    print(maxval)
    return bands_select


def brute_force_minimax_subset(DD_original, num_bands_select=3):
    """
    """
    return brute_force_maximin_subset(-1*DD_original, num_bands_select)


def select_maximin_subset(DD_original, num_bands_select=3):
    """Here we select the subset which maximizes the cost $C[K]$
     C[K] = min (DD[K, K])
     where $K$ is a subset of features

     The procedure is akin to beam-search as done for NLP!!
    """
    DD = np.array(DD_original, copy=True)
    nbands = np.shape(DD)[0]
    diag = np.arange(nbands)
    DD[diag, diag] = 1e56

    # bandwidth indicaates the width used in beam-search
    bandwidth = 10

    # We first pick all the 2-subsets (sized 2 subsets) with
    # the largest values. This initializes the list of subsets.

    # ** list_subsets is the list of subsets currently in the band
    # ** val_subsets is the value of the subsets corresponding to
    #    list_subsets
    list_subsets = [set()]*bandwidth
    val_subsets = [-1*np.inf]*bandwidth
    for i in range(nbands):
        for j in range(nbands):
            if i != j:
                subset = set([i, j])
                val = DD[i, j]
                if val > np.min(val_subsets):
                    indmin = np.argmin(val_subsets)
                    list_subsets[indmin] = subset
                    val_subsets[indmin] = val

    # Now for higher order subsets
    list_subsets_new = list_subsets
    val_subsets_new = val_subsets
    for _ in range(num_bands_select-2):
        list_subsets, val_subsets = list_subsets_new, val_subsets_new
        list_subsets_new = [set()]*bandwidth
        val_subsets_new = [-1*np.inf]*bandwidth
        for i in range(nbands):
            for subset in list_subsets:
                if i in subset:
                    continue
                set_tmp = tuple(subset.union(set([i])))
                val = np.min(DD[set_tmp, :][:, set_tmp])
                if val > np.min(val_subsets_new):
                    indmin = np.argmin(val_subsets_new)
                    list_subsets_new[indmin] = set_tmp
                    val_subsets_new[indmin] = val

    maxval = np.max(val_subsets_new)
    print("Value obtained: ", maxval)
    indselect = np.where(val_subsets_new == maxval)[0]
    return [list_subsets_new[i] for i in indselect]


def select_minimax_subset(DD, num_bands_select=3):
    """Select the minimax subset
    """
    return select_maximin_subset(-1*DD, num_bands_select)


def select_spectral_subset(DD, num_bands_select=3, dist_type='dissimilarity', beta=3.0):
    """Use spectral clustering to obtain the subset of bands
    """
    size_data = np.shape(DD)[0]
    diag = np.arange(size_data, dtype=np.int32)
    if dist_type == "dissimilarity":
        simDD = np.array(DD, copy=True)
        simDD[diag, diag] = 0.0
        simDD = np.exp(-1*beta*(simDD/simDD.std()) + 1e-6)
        simDD[diag, diag] = 0.0
    else:
        raise Exception("Not implemented for anyother than 'dissimilarity'.")

    labels = spectral_clustering(simDD, n_clusters=num_bands_select, assign_labels='discretize')

    band_selected = []
    DD_tmp = np.array(DD, copy=True)
    DD_tmp[diag, diag] = 0.0
    for l in np.unique(labels):
        indchoose = np.where(labels == l)[0]
        arr_tmp = np.mean(simDD[indchoose, :], axis=1)
        band = indchoose[np.argmax(arr_tmp)]
        band_selected.append(band)

    return set(band_selected)
