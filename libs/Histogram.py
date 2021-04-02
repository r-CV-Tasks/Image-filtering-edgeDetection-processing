
#
# Histograms Implementations
#

import numpy as np

def histogram(data, bins):
    bins_arr = np.arange(0, bins)
    hist = np.bincount(data.ravel(), minlength=256)
    return hist, bins_arr

def equalize_histogram(data, bins):

    pass

def normalize_histogram(data, bins):
    pass