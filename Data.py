"""
Code to get the datasets.
"""

import os
import pdb
import numpy as np
from matplotlib import pyplot as plt
import wget
from scipy.io import loadmat, savemat


def get_image_experiment1(size=(6, 6), num_bands=100):
    """Generate the multi-band image with three 
    """
    sx, sy = size

    rx, ry = int(1*sx/6), int(1*sy/6)
    gx, gy = int(3*sx/6), int(4*sy/6)
    bx, by = int(4*sx/6), int(1*sy/6)

    imgr = np.zeros(size, dtype=np.float64)
    imgr[rx, ry] = 1
    imgg = np.zeros(size, dtype=np.float64)
    imgg[gx, gy] = 1
    imgb = np.zeros(size, dtype=np.float64)
    imgb[bx, by] = 1
    img = np.stack((imgr, imgg, imgb))
    img = np.transpose(img)

    # Convert into multi-band
    img_band = np.zeros((sx, sy, num_bands-3), dtype=np.float64)
    np.random.seed(42)
    list_values = np.arange(0, 1, 0.01)
    for band in range(num_bands-3):
        rval, gval, bval = np.random.choice(list_values, 3, replace=False)
        img_band[rx, ry, band] = rval
        img_band[gx, gy, band] = gval
        img_band[bx, by, band] = bval
    img = np.concatenate((img_band, img), axis=-1)

    min_img = np.min(img, axis=(0, 1), keepdims=True)
    max_img = np.max(img, axis=(0, 1), keepdims=True)
    img = (img - min_img)/(max_img - min_img)

    return img


def get_image_experiment2(size=(6, 6), num_bands=None):
    """Generate the multi-band image with three 
    """
    sx, sy = size

    rx, ry = int(1*sx/6), int(1*sy/6)
    gx, gy = int(3*sx/6), int(4*sy/6)
    bx, by = int(4*sx/6), int(1*sy/6)

    imgr = np.zeros(size, dtype=np.float64)
    imgr[rx, ry] = 1
    imgg = np.zeros(size, dtype=np.float64)
    imgg[gx, gy] = 1
    imgb = np.zeros(size, dtype=np.float64)
    imgb[bx, by] = 1

    img1 = np.zeros(size, dtype=np.float64)
    img1[rx, ry], img1[gx, gy], img1[gx, gy] = 0.5, 0.5, 0.5
    img2 = np.zeros(size, dtype=np.float64)
    img2[rx, ry], img2[gx, gy], img2[gx, gy] = 1, 0.5, 0.5
    img3 = np.zeros(size, dtype=np.float64)
    img3[rx, ry], img3[gx, gy], img3[gx, gy] = 0.5, 1, 0.5

    img = np.stack((imgr, imgg, imgb, img1, img2, img3))
    img = np.transpose(img)

    min_img = np.min(img, axis=(0, 1), keepdims=True)
    max_img = np.max(img, axis=(0, 1), keepdims=True)
    img = (img - min_img)/(max_img - min_img)

    return img


def get_indianpines_dataset():
    """Indian-Pines Dataset
    """

    link = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
    if not os.path.exists("./data"):
        os.system("mkdir data")
    if not os.path.exists("./data/Indian_pines_corrected.mat"):
        wget.download(link, "./data/")
    data = np.array(loadmat("./data/Indian_pines_corrected.mat")['indian_pines_corrected'], dtype=np.float32)

    # Scale all the features to be between (0,1) across the pixels!!!
    mindata = np.min(data, axis=(0, 1), keepdims=True)
    maxdata = np.max(data, axis=(0, 1), keepdims=True)
    data = (data - mindata)/(maxdata-mindata)

    link = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
    if not os.path.exists("./data/Indian_pines_gt.mat"):
        wget.download(link, "./data/")
    gt = np.array(loadmat("./data/Indian_pines_gt.mat")['indian_pines_gt'], dtype=np.float32)

    return data, gt


def get_indianpines_dataset_classification():
    """Indian-Pines Dataset
    """

    link = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
    if not os.path.exists("./data"):
        os.system("mkdir data")
    if not os.path.exists("./data/Indian_pines_corrected.mat"):
        wget.download(link, "./data/")
    data = np.array(loadmat("./data/Indian_pines_corrected.mat")['indian_pines_corrected'], dtype=np.float32)

    mu_data = np.mean(data, axis=(0, 1), keepdims=True)
    std_data = np.std(data, axis=(0, 1), keepdims=True)
    data = (data - mu_data)/std_data

    link = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
    if not os.path.exists("./data/Indian_pines_gt.mat"):
        wget.download(link, "./data/")
    gt = np.array(loadmat("./data/Indian_pines_gt.mat")['indian_pines_gt'], dtype=np.float32)

    sx, sy, sz = data.shape
    X, y = data.reshape((-1, sz)), gt.flatten()
    indfilter = (y > 0)
    X, y = X[indfilter, :], y[indfilter]

    return X, y


def get_paviaU_dataset():
    """Indian-Pines Dataset
    """

    link = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
    if not os.path.exists("./data"):
        os.system("mkdir data")
    if not os.path.exists("./data/PaviaU.mat"):
        wget.download(link, "./data/")
    data = np.array(loadmat("./data/PaviaU.mat")['paviaU'], dtype=np.float32)

    # Scale all the features to be between (0,1) across the pixels!!!
    mindata = np.min(data, axis=(0, 1), keepdims=True)
    maxdata = np.max(data, axis=(0, 1), keepdims=True)
    data = (data - mindata)/(maxdata-mindata)

    link = "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat"
    if not os.path.exists("./data/PaviaU_gt.mat"):
        wget.download(link, "./data/")
    gt = np.array(loadmat("./data/PaviaU_gt.mat")['paviaU_gt'], dtype=np.float32)

    return data, gt


def get_paviaU_dataset_classification():
    """Indian-Pines Dataset
    """

    link = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
    if not os.path.exists("./data"):
        os.system("mkdir data")
    if not os.path.exists("./data/PaviaU.mat"):
        wget.download(link, "./data/")
    data = np.array(loadmat("./data/PaviaU.mat")['paviaU'], dtype=np.float32)

    mu_data = np.mean(data, axis=(0, 1), keepdims=True)
    std_data = np.std(data, axis=(0, 1), keepdims=True)
    data = (data - mu_data)/std_data

    link = "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat"
    if not os.path.exists("./data/PaviaU_gt.mat"):
        wget.download(link, "./data/")
    gt = np.array(loadmat("./data/PaviaU_gt.mat")['paviaU_gt'], dtype=np.float32)

    sx, sy, sz = data.shape
    X, y = data.reshape((-1, sz)), gt.flatten()
    indfilter = (y > 0)
    X, y = X[indfilter, :], y[indfilter]

    return X, y
