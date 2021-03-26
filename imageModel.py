import numpy as np
import cv2
from matplotlib import pyplot as plt


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
        self.dft = np.fft.fft2(self.imgByte)
        self.real = np.real(self.dft)
        self.imaginary = np.imag(self.dft)
        self.magnitude = np.abs(self.dft)
        self.phase = np.angle(self.dft)
        self.uniformMagnitude = np.ones(self.imgByte.shape)
        self.uniformPhase = np.zeros(self.imgByte.shape)

        # Noisy images
        self.uniform_noise = self.add_noise("uniform")
        self.gaussian_noise = self.add_noise("gaussian")
        self.saltpepper_noise = self.add_noise("salt & pepper")


    def mix(self, imageToBeMixed: 'ImageModel', magnitudeOrRealRatio: float, phaesOrImaginaryRatio: float):
        """
        a function that takes ImageModel object mag ratio, phase ration and
        return the magnitude of ifft of the mix
        return type ---> 2D numpy array
        """
        w1 = magnitudeOrRealRatio
        w2 = phaesOrImaginaryRatio

        # mix2 = (w1 * R1 + (1 - w1) * R2) + j * ((1 - w2) * I1 + w2 * I2)
        print("Mixing Real and Imaginary")
        R1 = self.real
        R2 = imageToBeMixed.real

        I1 = self.imaginary
        I2 = imageToBeMixed.imaginary

        realMix = w1*R1 + (1-w1)*R2
        imaginaryMix = (1-w2)*I1 + w2*I2

        combined = realMix + imaginaryMix * 1j
        mixInverse = np.real(np.fft.ifft2(combined))

        return abs(mixInverse)


    def add_noise(self, type: str):
        """

        :param type:
        :return:
        """
        noisy_image = None

        # This will be deleted
        noisy_image = self.imgByte.T

        if type == "uniform":
            # TODO: Add Uniform Noise Algorithm on self.imgByte
            pass

        elif type == "gaussian":
            # TODO: Add Gaussian Noise Algorithm on self.imgByte
            pass

        elif type == "salt & pepper":
            # TODO: Add Salt & Pepper Noise Algorithm on self.imgByte
            pass

        return noisy_image


    def apply_filter(self, data, type: str):
        """

        :param data:
        :param type:
        :return:
        """
        filtered_image = None

        # This will be deleted
        filtered_image = self.imgByte

        if type == "average":
            # TODO: Add Uniform Noise Algorithm on self.imgByte
            pass

        elif type == "gaussian":
            # TODO: Add Gaussian Noise Algorithm on self.imgByte
            pass

        elif type == "median":
            # TODO: Add Salt & Pepper Noise Algorithm on self.imgByte
            pass

        return filtered_image


    def apply_edge_mask(self, type: str):
        """

        :param type:
        :return:
        """
        edged_image = None

        # This will be deleted
        edged_image = self.imgByte

        if type == "sobel":
            # TODO: Add Sobel Mask Algorithm on self.imgByte
            pass

        elif type == "roberts":
            # TODO: Add Roberts Mask Algorithm on self.imgByte
            pass

        elif type == "perwitt":
            # TODO: Add Perwitt Mask Algorithm on self.imgByte
            pass

        elif type == "canny":
            # TODO: Add Canny Mask Algorithm on self.imgByte
            pass

        return edged_image
