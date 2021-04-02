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


def equalize_histogram(data, bins_num: int = 255):
    pass


def normalize_histogram(data: np.array, bins_num: int = 255):
    mn = np.min(data)
    mx = np.max(data)
    norm = (data - mn) * (1.0 / (mx - mn))
    histo, bins = histogram(norm, bins_num=bins_num)
    return norm, histo, bins


def threshold_image(data: np.ndarray, threshold: int, type: str = "global"):
    if type == "global":
        gray_img = rgb_to_gray(data)
        return (gray_img > threshold).astype(int)

    elif type == "local":
        pass

def rgb_to_gray(data: np.ndarray):
    return np.dot(data[..., :3], [0.299, 0.587, 0.114])