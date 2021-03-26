#
# Noise Filters Implementation
#

# Imports
import numpy as np


def UniformNoise(source: np.ndarray) -> np.ndarray:
    """
        Implementation of Uniform/Quantization Noise Filter

    :param source: Image to add Noise to
    :return: Noisy Image
    """
    # Create Noise
    noise = np.random.uniform(-255, 255, size=source.shape)
    src = np.copy(source)

    # Apply Uniform Noise Mask
    out = src + noise

    # CLipping Image in range 0, 255
    out[out > 255] = 255
    out[out < 0] = 0
    return out.astype(int)


def GaussianNoise(source: np.ndarray) -> np.ndarray:
    """
        Implementation of Gaussian Noise Filter

    :param source: Image to add Noise to
    :return: Noisy Image
    """
    noise = np.random.normal(0, 1, size=source.shape)
    src = np.copy(source)

    return src + noise
