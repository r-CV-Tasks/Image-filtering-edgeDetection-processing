import cv2
from Filters import Noise

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

        # Noisy images
        self.uniform_noise = self.add_noise("uniform")
        self.gaussian_noise = self.add_noise("gaussian")
        self.saltpepper_noise = self.add_noise("salt & pepper")


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
            noisy_image = Noise.UniformNoise(source=self.imgByte, snr=0.1)

        elif type == "gaussian":
            # TODO: Add Gaussian Noise Algorithm on self.imgByte
            noisy_image = Noise.GaussianNoise(source=self.imgByte, sigma=10, snr=0.7)

        elif type == "salt & pepper":
            # TODO: Add Salt & Pepper Noise Algorithm on self.imgByte
            noisy_image = Noise.SaltPepperNoise(source=self.imgByte, snr=0.7)

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

        elif type == "prewitt":
            # TODO: Add Prewitt Mask Algorithm on self.imgByte
            pass

        elif type == "canny":
            # TODO: Add Canny Mask Algorithm on self.imgByte
            pass

        return edged_image

    def get_histogram(self, type: str):
        """

        :param type:
        :return:
        """

        histo_plot = None

        # This will be deleted
        histo_plot = self.imgByte

        if type == "original":
            # TODO: Get Original Histogram of self.imgBye
            pass

        if type == "equalized":
            # TODO: Get Equalized Histogram of self.imgByte
            pass

        elif type == "normalized":
            # TODO: Get Normalized Histogram of self.imgByte
            pass

        return histo_plot

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
