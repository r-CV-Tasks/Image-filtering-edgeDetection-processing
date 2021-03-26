#
# Noise Filters Implementation
#

# Imports
import numpy as np


def clip(source: np.ndarray, high: [int, float], low: [int, float]) -> np.ndarray:
    """
        Clip Image in given Range
    :param source: Input Image
    :param high: Higher cut
    :param low: Lower cut
    :return: Clipped Image
    """
    out = np.copy(source)
    out[out > high] = high
    out[out < low] = low
    return out


def UniformNoise(source: np.ndarray, snr: float) -> np.ndarray:
    """
        Implementation of Uniform/Quantization Noise Filter

    :param source: Image to add Noise to
    :param snr : Signal to Noise Ratio
    :return: Noisy Image
    """
    # Create Noise
    noise = np.random.uniform(-255, 255, size=source.shape)
    src = np.copy(source)

    # Apply Uniform Noise Mask
    out = src*snr + noise*(1-snr)

    # CLipping Image in range 0, 255
    out = clip(out, 255, 0)
    return out.astype(int)


def GaussianNoise(source: np.ndarray, sigma: [int, float], snr: float) -> np.ndarray:
    """
        Implementation of Gaussian Noise Filter

    :param source: Image to add Noise to
    :param snr: Signal to Noise Ratio
    :param sigma: Noise Variance
    :return: Noisy Image
    """
    noise = np.random.normal(0, sigma, size=source.shape)
    src = np.copy(source)
    out = src*snr + noise * (1-snr)

    # CLipping Image in range 0, 255
    out = clip(out, 255, 0)
    return out.astype(int)


def SaltPepperNoise(source: np.ndarray, snr: float) ->np.ndarray:
    pass
