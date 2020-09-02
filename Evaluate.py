"""
SVM Evaluation of subsets
"""

import numpy as np
import pdb

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                                       'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                   {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]


def SVC_accuracy_subset(X, y, subset, return_preds=False):
    """Return the accuracy for a subset of features of X
    """
    subset_arr = np.array(list(subset), dtype=np.int32)
    Xtmp = X[:, subset_arr]

    # Train Test Split
    n_samples, n_features = np.shape(Xtmp)
    arr = np.arange(n_samples)
    np.random.shuffle(arr)
    cutpoint = int(0.1*n_samples)
    indtrain, indtest = arr[:cutpoint], arr[cutpoint:]
    XTrain, XTest = Xtmp[indtrain, :], Xtmp[indtest, :]
    yTrain, yTest = y[indtrain], y[indtest]

    base_clf = SVC(class_weight="balanced")
    clf = GridSearchCV(base_clf, SVM_GRID_PARAMS, n_jobs=4, verbose=0)
    clf.fit(XTrain, yTrain)
    ypred = clf.predict(XTest)
    accuracy = np.mean(ypred == yTest)

    if return_preds:
        return accuracy, ypred, yTest
    else:
        return accuracy


def SVC_accuracy_list_subsets(X, y, list_subsets):
    """Return the maximum accuracy of list_subsets
    """
    maxacc = 0
    set_select = None
    for subset in list_subsets:
        acc_subset = SVC_accuracy_subset(X, y, subset)
        if acc_subset > maxacc:
            maxacc = acc_subset
            set_select = subset

    return maxacc, set_select
