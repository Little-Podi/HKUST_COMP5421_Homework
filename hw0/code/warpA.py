import numpy as np


def warp(img, A, output_shape):
    """
    Warps (h, w) image im using affine (3, 3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation.
    """

    warped_img = np.empty(output_shape)
    A = np.linalg.inv(A)
    H, W = img.shape

    for h, row in enumerate(img):
        for w, col in enumerate(row):
            output_coord = np.array([h, w, 1])
            h_input, w_input, _ = A @ output_coord
            h_input = round(h_input)
            w_input = round(w_input)
            if 0 <= h_input < H and 0 <= w_input < W:
                warped_img[h, w] = img[h_input, w_input]

    return warped_img
