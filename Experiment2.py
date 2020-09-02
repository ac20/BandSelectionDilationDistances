"""
Code to show that the bands selected are the ones which seperate the objects!
"""

import numpy as np
import pdb
from matplotlib import pyplot as plt

from Data import get_indianpines_dataset, get_paviaU_dataset
from subset_selection_techniques import select_spectral_subset

if __name__ == "__main__":

    """
    ------------------------------------------------------------------------
    ------------------------- INDIAN-PINES DATASET -------------------------
    ------------------------------------------------------------------------
    """
    img, gt = get_indianpines_dataset()
    sx, sy, sz = np.shape(img)

    # Get Spectral Bands
    DD = np.loadtxt("./indianpines_distances.csv", delimiter=",")
    DD = DD + DD.transpose()
    bands_select = select_spectral_subset(DD, num_bands_select=4)

    # Visualize the bands
    plt.figure()
    plt.axis([0, 200, 0, 1])
    plt.xlabel("Bands")
    plt.ylabel("Intensity")
    for l in np.unique(gt):
        if l != 0:
            ax, ay = np.where(gt == l)
            arr = np.mean(img[ax, ay, :], axis=0)
            plt.plot(arr, "+-")

    for b in np.array(list(bands_select)):
        tmp = np.arange(0, 1, 0.01)
        plt.plot([b]*len(tmp), tmp, 'k--')
    plt.savefig("./img/Experiment2_indianpines.eps")
    plt.savefig("./img/Experiment2_indianpines.png")
    plt.close()

    """
    ------------------------------------------------------------------------
    ----------------------- PAVIA UNIVERSITY DATASET -----------------------
    ------------------------------------------------------------------------
    """
    img, gt = get_paviaU_dataset()
    sx, sy, sz = np.shape(img)

    # Get Spectral Bands
    DD = np.loadtxt("./paviau_distances.csv", delimiter=",")
    DD = DD + DD.transpose()
    bands_select = select_spectral_subset(DD, num_bands_select=3)

    # Visualize the bands
    plt.figure()
    plt.axis([0, 103, 0, 1])
    plt.xlabel("Bands")
    plt.ylabel("Intensity")
    for l in np.unique(gt):
        if l != 0:
            ax, ay = np.where(gt == l)
            arr = np.mean(img[ax, ay, :], axis=0)
            plt.plot(arr, "+-")

    for b in np.array(list(bands_select)):
        tmp = np.arange(0, 1, 0.01)
        plt.plot([b]*len(tmp), tmp, 'k--')
    plt.savefig("./img/Experiment2_paviau.eps")
    plt.savefig("./img/Experiment2_paviau.png")
    plt.close()
