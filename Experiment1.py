"""
Code to generate the images for Figure 1
"""

import numpy as np
import pdb
from matplotlib import pyplot as plt

from Data import get_image_experiment1, get_image_experiment2
from DilationDistances import get_dilation_distances_all_pairs
from DilationDistances import select_bands_dilation_distance_dynamic
from subset_selection_techniques import select_maximin_subset, select_minimax_subset
from subset_selection_techniques import brute_force_maximin_subset, brute_force_minimax_subset


def get_corr_dist_all_pairs(img):
    """
    """
    sx, sy, sz = img.shape
    DD = np.zeros((sz, sz), dtype=np.float64)
    for i in range(sz):
        for j in range(sz):
            if i != j:
                DD[i, j] = np.corrcoef(img[:, :, i].flatten(), img[:, :, j].flatten())[1, 0]
            elif i == j:
                DD[i, j] = 0
    return DD


def _increase_res(img, res=10):
    try:
        sx, sy, sz = img.shape
        img_tmp = img
        new_shape = (res*sx, res*sy, sz)
    except:
        sx, sy = img.shape
        sz = 1
        img_tmp = img[:, :, np.newaxis]
        new_shape = (res*sx, res*sy)
    img_highres = np.zeros((res*sx, res*sy, sz))
    for i in range(sz):
        for j in range(sx):
            for k in range(sy):
                img_highres[j*res:(j+1)*res, k*res:(k+1)*res, i] = img_tmp[j, k, i]
    return img_highres.reshape(new_shape)


if __name__ == "__main__":

    # Dataset
    img = get_image_experiment1(size=(6, 6), num_bands=10)

    # # Generate dilation distance subset of bands !
    DD_dil = get_dilation_distances_all_pairs(img)
    list_subsets_dil = brute_force_maximin_subset(DD_dil, num_bands_select=3)
    print(list_subsets_dil)

    # Generate correlation distances subset of bands
    DD_corr = get_corr_dist_all_pairs(img)
    print(-1*DD_corr)
    list_subsets_corr = brute_force_minimax_subset(DD_corr, num_bands_select=3)
    print(list_subsets_corr)

    # Plot the images

    # Base Image - The data was constructed this way!!!
    # (Check the Data.py file)
    img_tmp = 1 - np.array(img[:, :, -3:], dtype=np.float64)
    sx, sy, sz = np.shape(img)
    rx, ry = int(1*sx/6), int(1*sy/6)
    gx, gy = int(3*sx/6), int(4*sy/6)
    bx, by = int(4*sx/6), int(1*sy/6)
    img_tmp[ry, rx, :] = img[ry, rx, -3:]
    img_tmp[gy, gx, :] = img[gy, gx, -3:]
    img_tmp[by, bx, :] = img[by, bx, -3:]
    base_image = np.array(img_tmp[:, :, -3:], dtype=np.float64)
    base_image_highres = _increase_res(base_image, res=15)
    plt.imsave("./img/img_base.png", base_image_highres)

    # Correlation-Distance Images
    subsets_select = list_subsets_corr
    count = 1
    for band in list(subsets_select):
        band_img = 1 - img[:, :, band]
        band_img_highres = _increase_res(band_img, res=15)
        plt.imsave("./img/img_count"+str(count)+".png", band_img_highres, cmap=plt.cm.gray)
        count += 1

    # # Dilation-Distance Images
    subsets_select = list_subsets_dil
    for band in list(subsets_select):
        band_img = 1 - img[:, :, band]
        band_img_highres = _increase_res(band_img, res=15)
        plt.imsave("./img/img_count"+str(count)+".png", band_img_highres, cmap=plt.cm.gray)
        count += 1
