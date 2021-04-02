
#
# Histograms Implementations
#

import numpy as np

def histogram(data, bins):
    bins_arr = np.arange(0, 256)
    hist = np.bincount(data.ravel(), minlength=256)
    return hist, bins_arr
