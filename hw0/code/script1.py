"""
Problem 1: Image Alignment.
"""

from align_channels import align_channels
import numpy as np
from PIL import Image

# 1. Load images (all 3 channels)
red = np.load('../data/red.npy')
green = np.load('../data/green.npy')
blue = np.load('../data/blue.npy')

# 2. Find best alignment
rgb_aligned = align_channels(red, green, blue)

# 3. Save result to rgb_output.jpg (in the 'results' folder)
rgb_output = Image.fromarray(rgb_aligned)
rgb_output.save('../results/rgb_output.jpg')
