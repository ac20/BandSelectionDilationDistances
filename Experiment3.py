"""
Code to obtain the accuracy vs number_bands
"""

import numpy as np
import pdb
from matplotlib import pyplot as plt

from Data import get_indianpines_dataset, get_paviaU_dataset
from Data import get_indianpines_dataset_classification
from Data import get_paviaU_dataset_classification

from Evaluate import SVC_accuracy_subset
from subset_selection_techniques import select_spectral_subset

if __name__ == "__main__":

    # Create a file to write the results to
    fname = "./results_Experiment3.csv"
    with open(fname, "w") as f:
        f.write("Dataset,number_bands,beta,rep,accuracy\n")

    """
    --------------------------------------------------------------------------
    -------------------------- INDIAN PINES DATASET --------------------------
    --------------------------------------------------------------------------
    """

    X, y = get_indianpines_dataset_classification()
    DD = np.loadtxt("./indianpines_distances.csv", delimiter=",")
    DD = DD + DD.transpose()

    for number_bands in np.arange(2, 60, 2):
        for rep in range(10):
            for beta in [1., 3., 5., 10.]:
                band_select = select_spectral_subset(DD, num_bands_select=number_bands, beta=beta)
                acc = SVC_accuracy_subset(X, y, band_select)
                with open(fname, "a") as f:
                    f.write("{},{},{},{},{}\n".format("indianpines", number_bands, beta, rep, acc))

    sx, sy = np.shape(X)
    for rep in range(10):
        all_bands = set(list(np.arange(sy)))
        acc = SVC_accuracy_subset(X, y, all_bands)
        with open(fname, "a") as f:
            f.write("{},{},{},{},{}\n".format("indianpines", sy, 999, rep, acc))

    """
    --------------------------------------------------------------------------
    ------------------------ PAVIA UNIVERSITY DATASET ------------------------
    --------------------------------------------------------------------------
    """

    X, y = get_paviaU_dataset_classification()
    DD = np.loadtxt("./paviau_distances.csv", delimiter=",")
    DD = DD + DD.transpose()

    for number_bands in np.arange(2, 60, 2):
        for rep in range(10):
            for beta in [1., 3., 5., 10.]:
                band_select = select_spectral_subset(DD, num_bands_select=number_bands, beta=beta)
                acc = SVC_accuracy_subset(X, y, band_select)
                with open(fname, "a") as f:
                    f.write("{},{},{},{},{}\n".format("paviaU", number_bands, beta, rep, acc))

    sx, sy = np.shape(X)
    for rep in range(10):
        all_bands = set(list(np.arange(sy)))
        acc = SVC_accuracy_subset(X, y, all_bands)
        with open(fname, "a") as f:
            f.write("{},{},{},{},{}\n".format("paviaU", sy, 999, rep, acc))
