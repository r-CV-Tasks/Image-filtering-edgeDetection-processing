
#
# Low Pass Filters Implementations
#

import numpy as np
from scipy.signal import convolve2d
np.random.seed(0)


def CreateSquareKernel(size: int, mode: str, sigma: [int, float]) -> np.ndarray:
    """

    :param size:
    :param mode:
    :param sigma:
    :return:
    """
    if mode == 'ones':
        return np.ones((size, size))
    elif mode == 'gaussian':
        return np.random.normal(0, sigma, (size, size))


def ApplyKernel(source: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    """
        Main Convolution Function for a given Kernel

    :param source: First Array
    :param kernel: Calculated Kernel
    :param mode: Convolution mode ['valid', 'same']
    :return: ndarray
    """
    src = np.copy(source)

    # Check for Grayscale Image
    if len(src.shape) == 2 or src.shape[-1] == 1:
        conv = convolve2d(src, kernel, mode)
        return conv.astype('uint8')

    out = []
    # Apply Kernel using Convolution
    for channel in range(src.shape[-1]):
        conv = convolve2d(src[:, :, channel], kernel, mode)
        out.append(conv)
    return np.stack(out, -1)


def AverageFilter(source: np.ndarray, shape: int) -> np.ndarray:
    """
        Implementation of Average Low-pass Filter
    :param source: Image to apply Filter to
    :param shape: An Integer that denotes the Kernel size if 3
                   then the kernel is (3, 3)
    :return: Filtered Image
    """
    src = np.copy(source)

    # Create the Average Kernel
    kernel = np.ones((shape, shape)) * (1/shape**2)

    # Check for Grayscale Image
    out = ApplyKernel(src, kernel, 'valid')
    return out.astype('uint8')


def GaussianFilter(source: np.ndarray, shape: int, sigma: [int, float]) -> np.ndarray:
    """
        Gaussian Low Pass Filter Implementation
    :param source: Image to Apply Filter to
    :param shape: An Integer that denotes th Kernel size if 3
                  then the kernel is (3, 3)
    :param sigma: Standard Deviation
    :return: Filtered Image
    """
    src = np.copy(source)

    # Create a Gaussian Kernel
    kernel = CreateSquareKernel(shape, 'gaussian', sigma)

    # Apply the Kernel
    out = ApplyKernel(src, kernel, 'valid')
    return out.astype('uint8')
