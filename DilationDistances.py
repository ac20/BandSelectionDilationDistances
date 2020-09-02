"""
Code to compute the dilation distances.
"""

import numpy as np
from matplotlib import pyplot as plt
import pdb

import itertools
import skimage.morphology as mm


def dilation_distance_lambda(fi, fj, lam=0.0):
    """Return the dilation distance at Lambda.
    fi, fj are assumed to have 2 dimensions, i.e shape (n1, n2)

    d_{\lambda}(fi, fj) = min{ n | (fi + lam*1) \oplus nB >= fj }
    """
    assert np.all(fi.shape == fj.shape), "Dimensions of inputs do not match"

    selem = mm.disk(1)
    nmin, nmax = 0, fi.shape[0]+fi.shape[1]+1

    dilated_fi = fi + lam*np.ones(fi.shape)
    for n_dilate in range(nmin, nmax):
        if np.all(dilated_fi >= fj):
            return n_dilate
        else:
            dilated_fi = mm.dilation(dilated_fi, selem)
    raise Exception("Dilations of fi never crossed fj!!!!")


def dilation_distance(fi, fj):
    """Return the dilation distance between fi and fj computed using
    the average of all dilation-distances at lambda
    """

    d_lam = 0
    count = 0
    for lam in np.arange(0, 1, 0.01):
        d_lam += dilation_distance_lambda(fi, fj, lam=lam)
        count += 1

    return d_lam/count


def get_dilation_distances_all_pairs(img):
    """
    """
    sx, sy, sz = img.shape
    DD = np.zeros((sz, sz), dtype=np.float64)
    for i in range(sz):
        for j in range(sz):
            if i != j:
                DD[i, j] = dilation_distance(img[:, :, i], img[:, :, j])
            elif i == j:
                DD[i, j] = 0
    return DD


def select_bands_dilation_distance_dynamic(DD, num_bands_select=3):
    """Select the bands obtained by dilation distance

    DD : 2d array containing the dilation distances

    ** Diagonals are assumed to have the value 1e56
    """

    nbands = DD.shape[0]
    bandwidth = 10

    # Initialize the set of 2-subsets to check
    list_subsets = []
    val_subsets = []
    values_list = np.sort(np.unique(DD.flatten()))[(-bandwidth-1):-1]
    for val in values_list[::-1]:
        indx, indy = np.where(DD == val)
        for i in range(len(indx)):
            list_subsets.append(set([indx[i], indy[i]]))
            val_subsets.append(val)
            # Break if the bandwidth is reached
            if len(list_subsets) > bandwidth:
                break
        # Break if the bandwidth is reached
        if len(list_subsets) > bandwidth:
            break

    # For each subset select the best bands to add and
    # update the list_subsets
    list_subsets_new = list_subsets
    val_subsets_new = val_subsets
    for _ in range(num_bands_select-2):
        list_subsets = list_subsets_new
        val_subsets = val_subsets_new

        list_subsets_new = [None]*bandwidth
        val_subsets_new = [-1*np.inf]*bandwidth
        for i in range(nbands):
            for subset in list_subsets:
                if i in subset:
                    continue
                set_tmp = tuple(subset.union(set([i])))
                val = np.min(DD[set_tmp, :][:, set_tmp])
                if val >= np.min(val_subsets_new):
                    ind_replace = np.argmin(val_subsets_new)
                    list_subsets_new[ind_replace] = set(set_tmp)
                    val_subsets_new[ind_replace] = val
        print("\r {}".format(_), end="")
    print("\nBest value is {}".format(np.max(val_subsets_new)))
    return list_subsets_new
