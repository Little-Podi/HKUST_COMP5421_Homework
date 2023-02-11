"""
Problem 2: Image Warping.
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import warpA_check
import warpA

# Read the image
img = imageio.v2.imread('../data/mug.jpg')
# Convert to float
img = img / 255.0

# Convert to grayscale
img_gray = np.dot(img, [0.299, 0.587, 0.114])

# Create figure
f, axes = plt.subplots(2, 2)
f.set_size_inches(8, 8)
axes[0, 0].imshow(img)
axes[0, 0].set_title('original')
axes[0, 1].imshow(img_gray, cmap=plt.get_cmap('gray'))
axes[0, 1].set_title('grayscale')
# axes[1, 1].remove()


# Define some helper functions to create affine transformations
def scalef(s):
    return np.diag([s, s, 1])


def transf(tx, ty):
    A = np.eye(3)
    A[0, 2] = ty
    A[1, 2] = tx
    return A


def rotf(t):
    return np.array([
        [np.cos(t), np.sin(t), 0],
        [-np.sin(t), np.cos(t), 0],
        [0, 0, 1]
    ])


output_shape = img_gray.shape
cx = img_gray.shape[1] // 2
cy = img_gray.shape[0] // 2

A = (transf(output_shape[1] // 2, output_shape[0] // 2, )
     .dot(scalef(0.8))
     .dot(rotf(- 30 * np.pi / 180))
     .dot(transf(-cx, -cy)))

# Plot a dot at the rotation center
axes[0, 1].plot(cx, cy, 'r+')

warped_img = warpA_check.warp(img_gray, A, output_shape)
axes[1, 0].imshow(warped_img, cmap=plt.get_cmap('gray'))
axes[1, 0].set_title('warped_check')

warped_img = warpA.warp(img_gray, A, output_shape)
axes[1, 1].imshow(warped_img, cmap=plt.get_cmap('gray'))
axes[1, 1].set_title('warped')

# Write the plot to an image
plt.savefig('../results/transformed.jpg')
# plt.show()
