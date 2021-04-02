import cv2
import numpy as np
from libs import Noise, LowPass
from libs import Histogram


class ImageModel():

    """
    A class that represents the ImageModel
    """

    def __init__(self, imgPath: str):
        """

        :param imgPath: absolute path of the image
        """

        self.imgPath = imgPath
        self.imgByte = cv2.imread(self.imgPath, flags=cv2.IMREAD_GRAYSCALE).T
        self.imgShape = self.imgByte.shape

    def add_noise(self, type: str, snr: float = 0.5, sigma: int = 64) -> np.ndarray:
        """
        This function adds different types of noises to the given image
        :param type: Specify the type of noise to be added
        :return: numpy array of the new noisy image
        """

        noisy_image = None

        if type == "uniform":
            noisy_image = Noise.UniformNoise(source=self.imgByte, snr=snr)

        elif type == "gaussian":
            noisy_image = Noise.GaussianNoise(source=self.imgByte, sigma=sigma, snr=snr)

        elif type == "salt & pepper":
            noisy_image = Noise.SaltPepperNoise(source=self.imgByte, snr=snr)

        return noisy_image


    def apply_filter(self, data: np.ndarray, type: str, shape: int, sigma: [int, float] = 0) -> np.ndarray:
        """
        This function adds different types of filters to the given image
        :param data: The given image numpy array
        :param type: The type of filter to be applied on the given image
        :return: numpy array of the new filtered image
        """

        filtered_image = None

        if type == "average":
            filtered_image = LowPass.AverageFilter(source=data, shape=shape)

        elif type == "gaussian":
            filtered_image = LowPass.GaussianFilter(source=data, shape=shape, sigma=sigma)

        elif type == "median":
            filtered_image = LowPass.MedianFilter(source=data, shape=shape)

        return filtered_image


    def apply_edge_mask(self, type: str):
        """

        :param type:
        :return:
        """
        edged_image = None

        if type == "sobel":
            # TODO: Add Sobel Mask Algorithm on self.imgByte
            pass

        elif type == "roberts":
            # TODO: Add Roberts Mask Algorithm on self.imgByte
            pass

        elif type == "prewitt":
            # TODO: Add Prewitt Mask Algorithm on self.imgByte
            pass

        elif type == "canny":
            # TODO: Add Canny Mask Algorithm on self.imgByte
            pass

        return edged_image

    def get_histogram(self, type: str, bins_number: int = 256):
        """

        :param type:
        :return:
        """

        histo_plot = None

        if type == "original":
            hist, bins = Histogram.histogram(self.imgByte, bins=bins_number)
            return hist, bins

        if type == "equalized":
            # TODO: Get Equalized Histogram of self.imgByte
            pass

        elif type == "normalized":
            # TODO: Get Normalized Histogram of self.imgByte
            pass

    def thresholding(self, type: str):
        """

        :param type:
        :return:
        """

        threshold_image = None
        if type == "local":
            # TODO: Apply Local Thresholding on self.imgByte
            pass

        elif type == "global":
            # TODO: Apply Global Thresholding on self.imgByte
            pass

        return threshold_image


    def to_gray(self):
        """

        :return:
        """

        gray_image = None

        # TODO: Apply RGB to Gray Scale Conversion

        return gray_image
