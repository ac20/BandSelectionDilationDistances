"""
Here we generate the dilation distances between all pairs of features and
save it for future use.
"""

import numpy as np
import multiprocessing as mp

from DilationDistances import dilation_distance
from Data import get_paviaU_dataset


def parallel_dilation_distance(ax, ay):
    """
    """
    DD_np = np.frombuffer(var_dict['DD']).reshape(var_dict['nbands'], var_dict['nbands'])
    img_np = np.frombuffer(var_dict['img']).reshape(var_dict['shape'])
    DD_np[ax, ay] = dilation_distance(img_np[:, :, ax], img_np[:, :, ay])
    print(ax, ay)


var_dict = {}


def init_worker(DD, img, shape, nbands):
    """Initializer for each child process"""
    var_dict['DD'] = DD
    var_dict['img'] = img
    var_dict['shape'] = shape
    var_dict['nbands'] = nbands


def generate_distance_matrix_paviaU():
    """
    """
    img, gt = get_paviaU_dataset()
    sx, sy, sz = img.shape
    img_buffer = mp.RawArray('d', sx*sy*sz)
    img_np = np.frombuffer(img_buffer)
    np.copyto(img_np, img.flatten())

    nbands = sz

    DD = np.zeros((nbands, nbands), dtype=np.float64)
    DD_buffer = mp.RawArray('d', DD.shape[0]*DD.shape[1])
    DD_np = np.frombuffer(DD_buffer)

    # Initialize pooling
    pool = mp.Pool(8, initializer=init_worker, initargs=(DD_buffer, img_buffer, img.shape, nbands))

    for ax in range(nbands):
        for ay in range(nbands):
            if ax != ay:
                pool.apply_async(parallel_dilation_distance, args=(ax, ay))
            else:
                (DD_np.reshape(nbands, nbands))[ax, ay] = 1e56
            print("\r {} out of {} done...".format(ax*nbands + ay + 1, nbands**2), end="")

    pool.close()
    pool.join()

    DD = np.array(DD_np).reshape((nbands, nbands))
    np.savetxt("./paviau_distances_parallel.csv", DD, delimiter=',')


if __name__ == "__main__":
    generate_distance_matrix_paviaU()
