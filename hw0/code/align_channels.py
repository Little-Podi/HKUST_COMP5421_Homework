import numpy as np

MAX_DIS = 30


def update_current(red, crop, current, red_norm, metric='SSD'):
    if metric == 'SSD':
        distance = np.sum(np.power((red - crop), 2))
        if current is None:
            return distance, True
        else:
            if distance < current:
                return distance, True
            else:
                return current, False
    elif metric == 'NCC':
        correlation = np.sum(np.multiply(red / red_norm, crop / np.linalg.norm(crop)))
        if current is None:
            return correlation, True
        else:
            if correlation > current:
                return correlation, True
            else:
                return current, False
    else:
        raise NotImplementedError


def align_channels(red, green, blue):
    """
    Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations.

    Args:
      red, green, blue: Each is a HxW matrix corresponding to an HxW image.

    Returns:
      rgb_output: HxWx3 color image output, aligned as desired.
    """

    H, W = red.shape
    green_pad = np.pad(green, (MAX_DIS, MAX_DIS))
    blue_pad = np.pad(blue, (MAX_DIS, MAX_DIS))
    green_current = blue_current = None
    green_h_idx = green_w_idx = blue_h_idx = blue_w_idx = None
    red_norm = np.linalg.norm(red)

    for h in range(2 * MAX_DIS):
        for w in range(2 * MAX_DIS):
            green_crop = green_pad[h:h + H, w:w + W]
            blue_crop = blue_pad[h:h + H, w:w + W]

            green_current, updated = update_current(red, green_crop, green_current, red_norm)
            if updated:
                green_h_idx = h
                green_w_idx = w

            blue_current, updated = update_current(red, blue_crop, blue_current, red_norm)
            if updated:
                blue_h_idx = h
                blue_w_idx = w

    rgb = np.array([red, green_pad[green_h_idx:green_h_idx + H, green_w_idx:green_w_idx + W],
                    blue_pad[blue_h_idx:blue_h_idx + H, blue_w_idx:blue_w_idx + W]])
    rgb = np.transpose(rgb, (1, 2, 0))
    return rgb
