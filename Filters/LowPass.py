
#
# Low Pass Filters Implementations
#

import numpy as np
from scipy.signal import convolve2d


def ZeroPadImage(source: np.ndarray, f: int) -> np.ndarray:
    """
        Pad Image with p size calculated from filter size f
        to obtain a 'same' Convolution.
    :param source: Input Image
    :param f: Filter Size
    :return: Padded Image
    """
    src = np.copy(source)

    # Calculate Padding size
    p = (f - 1)/2

    # Apply Zero Padding
    out = np.pad(src, (p, p), 'constant', constant_values=0)
    return out


def CreateSquareKernel(size: int, mode: str, sigma: [int, float] = None) -> np.ndarray:
    """
        Create/Calculate a square kernel for different low pass filter modes

    :param size: Kernel Size
    :param mode: Low Pass Filter Mode ['ones' -> Average Filter Mode, 'gaussian', 'median' ]
    :param sigma: Variance amount in case of 'Gaussian' mode
    :return: Square Array Kernel
    """
    if mode == 'ones':
        return np.ones((size, size))
    elif mode == 'gaussian':
        return np.random.normal(0, sigma, (size, size))


def ApplyKernel(source: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    """
        Calculate/Apply Convolution of two arrays, one being the kernel
        and the other is the image

    :param source: First Array
    :param kernel: Calculated Kernel
    :param mode: Convolution mode ['valid', 'same']
    :return: Convoluted Result
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
    kernel = CreateSquareKernel(shape, 'ones') * (1/shape**2)

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


def MedianFilter(source: np.ndarray, shape: int) -> np.ndarray:
    """
        Median Low Pass Filter Implementation
    :param source: Image to Apply Filter to
    :param shape: An Integer that denotes th Kernel size if 3
                  then the kernel is (3, 3)
    :return: Filtered Image
    """
    src = np.copy(source)
    i = 0
    j = 0
    r, g, b = [], [], []
    ch = [r, g, b]
    img_shape = src.shape

    # Pad the Image to obtain a Same Convolution
    src = ZeroPadImage(src, shape)

    # Loop each channel in the image with the kernel size
    # Calculate the Median value
    # Append to the Channels List
    for _ in range(0, source.shape[0]):
        for _ in range(0, source.shape[1]):
            for c in range(0, source.shape[2]):
                kernel = src[i: i+3, j: j+3, c]
                ch[c].append(np.median(kernel))
            j += 1
        i += 1

    for c in ch:
        pass
    pass
