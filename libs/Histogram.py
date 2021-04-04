#
# Histograms Implementations
#

import numpy as np


def histogram(data: np.array, bins_num: int = 255):
    """

    :param data:
    :param bins_num:
    :return:
    """
    if bins_num == 2:
        new_data = data
    else:
        new_data = np.round(np.interp(data, (data.min(), data.max()), (0, bins_num))).astype('uint8')
    bins = np.arange(0, bins_num)
    hist = np.bincount(new_data.ravel(), minlength=bins_num)
    return hist, bins


def equalize_histogram(data: np.ndarray, bins_num: int = 256):
    """
        Histogram Equalization Implementation
    :param source: Input Source Image
    :return: Equalized Image
    """
    # Calculate the Occurrences of each pixel in the input
    hist_array = np.bincount(data.flatten(), minlength=bins_num)

    # Normalize Resulted array
    px_count = np.sum(hist_array)
    hist_array = hist_array/px_count

    # Calculate the Cumulative Sum
    hist_array = np.cumsum(hist_array)

    # Pixel Mapping
    trans_map = np.floor(255 * hist_array)

    # Transform Mapping to Image
    img1d = list(data.flatten())
    map_img1d = [trans_map[px] for px in img1d]

    # Reshape Image
    map_img2d = np.reshape(np.asarray(map_img1d), data.shape)

    return map_img2d


def normalize_histogram(data: np.array, bins_num: int = 255):
    mn = np.min(data)
    mx = np.max(data)
    norm = (data - mn) * (1.0 / (mx - mn))
    histo, bins = histogram(norm, bins_num=bins_num)
    return norm, histo, bins


def threshold_image(data: np.ndarray, threshold: int, type: str = "global", size: int = 32):
    if type == "global":
        gray_img = rgb_to_gray(data)
        return (gray_img > threshold).astype(int)

    elif type == "local":
        pass


def rgb_to_gray(data: np.ndarray):
    return np.dot(data[..., :3], [0.299, 0.587, 0.114]).astype('uint8')
