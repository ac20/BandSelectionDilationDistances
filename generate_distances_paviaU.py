"""
Here we generate the dilation distances between all pairs of features and 
save it for future use.
"""

import numpy as np

from DilationDistances import dilation_distance
from Data import get_paviaU_dataset


def generate_distance_matrix_paviaU():
    """
    """
    img, gt = get_paviaU_dataset()
    nbands = img.shape[2]
    DD = np.zeros((nbands, nbands), dtype=np.float64)
    for ax in range(nbands):
        for ay in range(nbands):
            if ax != ay:
                DD[ax, ay] = dilation_distance(img[:, :, ax], img[:, :, ay])
            else:
                DD[ax, ay] = 1e56
            print("\r {} out of {} done...".format(ax*nbands + ay + 1, nbands**2), end="")

    np.savetxt("./paviau_distances.csv", DD, delimiter=',')


if __name__ == "__main__":
    generate_distance_matrix_paviaU()
