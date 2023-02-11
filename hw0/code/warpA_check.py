import scipy.ndimage
import numpy as np


def warp(img, A, output_shape):
    return scipy.ndimage.affine_transform(img, np.linalg.inv(A), output_shape=output_shape, order=0)
