"""
Code to obtain the ACA, OCA and KC for all bands and subset of bands.
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import cohen_kappa_score

from subset_selection_techniques import select_spectral_subset
from Data import get_indianpines_dataset_classification
from Data import get_paviaU_dataset_classification
from Evaluate import SVC_accuracy_subset

import pandas as pd


def average_classification_accuracy(ypred, ytrue):
    acc = []
    for l in np.unique(ytrue):
        indselect = np.where(ytrue == l)[0]
        acc.append(np.average(ytrue[indselect] == ypred[indselect]))

    return np.mean(acc)


if __name__ == "__main__":

    fname = "./results_Experiment4.csv"
    with open(fname, "w") as f:
        f.write("Dataset,number_bands,rep,oca,aca,kc\n")

    """
    --------------------------------------------------------------------------
    -------------------------- INDIAN PINES DATASET --------------------------
    --------------------------------------------------------------------------
    """

    DD = np.loadtxt("./indianpines_distances.csv", delimiter=",")
    DD = DD + DD.transpose()

    X, y = get_indianpines_dataset_classification()
    sx, sy = np.shape(X)

    for rep in range(10):
        subset_selected = select_spectral_subset(DD, num_bands_select=30, beta=5.0)
        OCA, ypred, ytest = SVC_accuracy_subset(X, y, subset_selected, return_preds=True)
        KC = cohen_kappa_score(ypred, ytest)
        ACA = average_classification_accuracy(ypred, ytest)
        with open(fname, "a") as f:
            f.write("{},{},{},{},{},{}\n".format("indianpines", 30, rep, OCA, ACA, KC))

        all_bands = set(list(np.arange(sy)))
        OCA, ypred, ytest = SVC_accuracy_subset(X, y, all_bands, return_preds=True)
        KC = cohen_kappa_score(ypred, ytest)
        ACA = average_classification_accuracy(ypred, ytest)
        with open(fname, "a") as f:
            f.write("{},{},{},{},{},{}\n".format("indianpines", 200, rep, OCA, ACA, KC))

    """
     -------------------------------------------------------------------------
     ---------------------- UNIVERSITY OF PAVIA DATASET ----------------------
     -------------------------------------------------------------------------
     """

    DD = np.loadtxt("./paviau_distances.csv", delimiter=",")
    DD = DD + DD.transpose()

    X, y = get_paviaU_dataset_classification()
    sx, sy = np.shape(X)

    for rep in range(10):
        subset_selected = select_spectral_subset(DD, num_bands_select=30, beta=3.0)
        OCA, ypred, ytest = SVC_accuracy_subset(X, y, subset_selected, return_preds=True)
        KC = cohen_kappa_score(ypred, ytest)
        ACA = average_classification_accuracy(ypred, ytest)
        with open(fname, "a") as f:
            f.write("{},{},{},{},{},{}\n".format("paviau", 30, rep, OCA, ACA, KC))

        all_bands = set(list(np.arange(sy)))
        OCA, ypred, ytest = SVC_accuracy_subset(X, y, all_bands, return_preds=True)
        KC = cohen_kappa_score(ypred, ytest)
        ACA = average_classification_accuracy(ypred, ytest)
        with open(fname, "a") as f:
            f.write("{},{},{},{},{},{}\n".format("paviau", 103, rep, OCA, ACA, KC))

    """
    SUMMARY OF THE RESULTS
    """
    data = pd.read_csv(fname)

    i1 = np.array(data['Dataset'] == "indianpines")
    i2 = np.array(data['number_bands'] == 30)
    indselect = np.logical_and(i1, i2)
    arr = data[indselect]
    print("-----------------------------")
    print("Indian Pines (Subset Bands)")
    print("-----------------------------")
    print("OCA : {:0.4f} \pm {:0.4f}".format(np.mean(arr['oca']), np.std(arr['oca'])))
    print("ACA : {:0.4f} \pm {:0.4f}".format(np.mean(arr['aca']), np.std(arr['aca'])))
    print(" KC : {:0.4f} \pm {:0.4f}".format(np.mean(arr['kc']), np.std(arr['kc'])))

    i1 = np.array(data['Dataset'] == "indianpines")
    i2 = np.array(data['number_bands'] == 200)
    indselect = np.logical_and(i1, i2)
    arr = data[indselect]
    print("-----------------------------")
    print("Indian Pines (All Bands)")
    print("-----------------------------")
    print("OCA : {:0.4f} \pm {:0.4f}".format(np.mean(arr['oca']), np.std(arr['oca'])))
    print("ACA : {:0.4f} \pm {:0.4f}".format(np.mean(arr['aca']), np.std(arr['aca'])))
    print(" KC : {:0.4f} \pm {:0.4f}".format(np.mean(arr['kc']), np.std(arr['kc'])))

    i1 = np.array(data['Dataset'] == "paviau")
    i2 = np.array(data['number_bands'] == 30)
    indselect = np.logical_and(i1, i2)
    arr = data[indselect]
    print("-------------------------------")
    print("Pavia University (Subset Bands)")
    print("-------------------------------")
    print("OCA : {:0.4f} \pm {:0.4f}".format(np.mean(arr['oca']), np.std(arr['oca'])))
    print("ACA : {:0.4f} \pm {:0.4f}".format(np.mean(arr['aca']), np.std(arr['aca'])))
    print(" KC : {:0.4f} \pm {:0.4f}".format(np.mean(arr['kc']), np.std(arr['kc'])))

    i1 = np.array(data['Dataset'] == "paviau")
    i2 = np.array(data['number_bands'] == 103)
    indselect = np.logical_and(i1, i2)
    arr = data[indselect]
    print("-----------------------------")
    print("Pavia University (All Bands)")
    print("-----------------------------")
    print("OCA : {:0.4f} \pm {:0.4f}".format(np.mean(arr['oca']), np.std(arr['oca'])))
    print("ACA : {:0.4f} \pm {:0.4f}".format(np.mean(arr['aca']), np.std(arr['aca'])))
    print(" KC : {:0.4f} \pm {:0.4f}".format(np.mean(arr['kc']), np.std(arr['kc'])))
