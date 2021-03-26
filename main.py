# Importing Packages
import os, sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from UI import mainGUI as m
import cv2
import numpy as np
from imageModel import ImageModel

# importing module
import logging

# Create and configure logger
logging.basicConfig(level=logging.DEBUG,
                    filename="app.log",
                    format='%(lineno)s - %(levelname)s - %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ImageProcessor(m.Ui_MainWindow):
    """
    Main Class of the program GUI
    """
    def __init__(self, starterWindow):
        """
        Main loop of the UI
        :param mainWindow: QMainWindow Object
        """
        super(ImageProcessor, self).setupUi(starterWindow)

        # Images Lists
        self.inputImages = [self.img1_input, self.img2_input, self.imgA_input, self.imgB_input]
        self.filtersImages = [self.img1_noisy, self.img1_filtered, self.img1_edged]
        self.histoImages = [self.img2_input_histo, self.img2_output_histo, self.img2_output]


        self.imageWidgets = [self.img1_input, self.img1_noisy, self.img1_filtered, self.img1_edged,
                             self.img2_input, self.img2_output,
                             self.imgA_input, self.imgB_input, self.imgX_output]

        self.imagesModels = [..., ..., ..., ...]
        self.heights = [..., ..., ..., ...]
        self.weights = [..., ..., ..., ...]

        # Images Labels and Sizes
        self.imagesLabels = [self.label_imgName_1, self.label_imgName_2, self.label_imgName_3, self.label_imgName_4]
        self.imagesSizes = [self.label_imgSize_1, self.label_imgSize_2, self.label_imgSize_3, self.label_imgSize_4]


        # No Noisy Image Array yet
        self.currentNoiseImage = None

        # Combo Lists
        self.updateCombos = [self.combo_noise, self.combo_filter, self.combo_edges]

        # Setup Load Buttons Connections
        self.btn_load_1.clicked.connect(lambda: self.load_file(0))
        self.btn_load_2.clicked.connect(lambda: self.load_file(1))
        self.btn_load_3.clicked.connect(lambda: self.load_file(2))
        self.btn_load_4.clicked.connect(lambda: self.load_file(3))

        # Setup Combo Connections
        self.combo_noise.activated.connect(lambda: self.updateCombosChanged(0))
        self.combo_filter.activated.connect(lambda: self.updateCombosChanged(1))
        self.combo_edges.activated.connect(lambda: self.updateCombosChanged(2))
        self.combo_histogram.activated.connect(lambda: self.updateCombosChanged(3))

        self.setupImagesView()

    def setupImagesView(self):
        """
        Adjust the shape and scales of the widgets
        Remove unnecessary options
        :return:
        """
        for widget in self.imageWidgets:
            widget.ui.histogram.hide()
            widget.ui.roiBtn.hide()
            widget.ui.menuBtn.hide()
            widget.ui.roiPlot.hide()
            widget.getView().setAspectLocked(False)
            widget.view.setAspectLocked(False)

    def load_file(self, imgID):
        """
        Load the File from User
        :param imgID: 0, 1, 2 or 3
        :return:
        """

        # Open File & Check if it was loaded correctly
        logger.info("Browsing the files...")
        repo_path = "./src/Images"
        self.filename, self.format = QtWidgets.QFileDialog.getOpenFileName(None, "Load Image", repo_path,
                                                                           "*.jpg;;" "*.jpeg;;" "*.png;;")
        imgName = self.filename.split('/')[-1]
        if self.filename == "":
            pass
        else:
            image = cv2.imread(self.filename, flags=cv2.IMREAD_GRAYSCALE).T
            self.heights[imgID], self.weights[imgID] = image.shape
            self.imagesModels[imgID] = ImageModel(self.filename)

            # When Images in Tab1, Tab2 or Image A in Tab 3
            if imgID != 3:
                # Create and Display Original Image
                self.displayImage(self.imagesModels[imgID].imgByte, self.inputImages[imgID])

                # Enable Combo Boxes
                self.updateCombos[imgID].setEnabled(True)
                self.updateCombos[imgID + 1].setEnabled(True)
                self.updateCombos[imgID + 2].setEnabled(True)

                # Set Image Name and Size
                self.imagesLabels[imgID].setText(imgName)
                self.imagesSizes[imgID].setText(f"{image.shape[0]}x{image.shape[1]}")

                logger.info(f"Added Image{imgID + 1}: {imgName} successfully")

            # When Loading Image B in Tab 3
            else:
                if self.heights[3] != self.heights[2] or self.weights[3] != self.weights[2]:
                    self.showMessage("Warning!!", "Images sizes must be the same, please upload another image",
                                     QMessageBox.Ok, QMessageBox.Warning)
                    logger.warning("Warning!!. Images sizes must be the same, please upload another image")
                else:
                    self.displayImage(self.imagesModels[imgID].imgByte, self.inputImages[imgID])
                    self.btn_hybrid.setEnabled(True)
                    logger.info(f"Added Image{imgID + 1}: {imgName} successfully")

    def updateCombosChanged(self, combo_id):
        selectedComponent = self.updateCombos[combo_id].currentText().lower()

        # Noise Options
        if combo_id == 0:
            if selectedComponent == "uniform noise":
                self.displayImage(self.imagesModels[0].uniform_noise, self.filtersImages[combo_id])
                self.currentNoiseImage = self.imagesModels[0].uniform_noise
            elif selectedComponent == "gaussian noise":
                self.displayImage(self.imagesModels[0].gaussian_noise, self.filtersImages[combo_id])
                self.currentNoiseImage = self.imagesModels[0].gaussian_noise
            elif selectedComponent == "salt & pepper noise":
                self.displayImage(self.imagesModels[0].saltpepper_noise, self.filtersImages[combo_id])
                self.currentNoiseImage = self.imagesModels[0].saltpepper_noise

        # Filters Options
        if combo_id == 1:
            if selectedComponent == "average filter":
                filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="average")
                self.displayImage(filtered_image, self.filtersImages[combo_id])
            elif selectedComponent == "gaussian filter":
                filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="gaussian")
                self.displayImage(filtered_image, self.filtersImages[combo_id])
            elif selectedComponent == "median filter":
                filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="median")
                self.displayImage(filtered_image, self.filtersImages[combo_id])

        # Edge Detection Options
        if combo_id == 2:
            if selectedComponent == "sobel mask":
                edged_image = self.imagesModels[0].apply_edge_mask(type="sobel")
                self.displayImage(edged_image, self.filtersImages[combo_id])
            elif selectedComponent == "roberts mask":
                edged_image = self.imagesModels[0].apply_edge_mask(type="roberts")
                self.displayImage(edged_image, self.filtersImages[combo_id])
            elif selectedComponent == "perwitt mask":
                edged_image = self.imagesModels[0].apply_edge_mask(type="perwitt")
                self.displayImage(edged_image, self.filtersImages[combo_id])
            elif selectedComponent == "canny mask":
                edged_image = self.imagesModels[0].apply_edge_mask(type="canny")
                self.displayImage(edged_image, self.filtersImages[combo_id])

        # Histograms Options
        if combo_id == 3:
            if selectedComponent == "original histogram":
                histo = self.imagesModels[1].get_histogram(type="original")
                # self.displayImage(histo, self.img2_input_histo)
                # TODO plot histogram and distribution curve
            if selectedComponent == "equalized histogram":
                histo = self.imagesModels[1].get_histogram(type="equalized")
                # self.displayImage(histo, self.img2_output_histo)
                # TODO plot histogram and distribution curve
            elif selectedComponent == "normalized histogram":
                histo = self.imagesModels[1].get_histogram(type="normalized")
                # self.displayImage(histo, self.img2_output_histo)

            elif selectedComponent == "local thresholding":
                local_threshold = self.imagesModels[1].thresholding(type="local")
                self.displayImage(local_threshold, self.img2_output)
            elif selectedComponent == "global thresholding":
                global_threshold = self.imagesModels[1].thresholding(type="global")
                self.displayImage(global_threshold, self.img2_output)
            elif selectedComponent == "transform to gray":
                gray_image = self.imagesModels[1].to_gray()
                self.displayImage(gray_image, self.img2_output)

                # TODO: Plot R, G and B Histograms separately

        logger.info(f"Viewing {selectedComponent} Component Of Image{combo_id + 1}")

    def displayImage(self, data, widget):
        """
        Display the given data
        :param data: 2d numpy array
        :param widget: ImageView object
        :return:
        """
        widget.setImage(data)
        widget.view.setRange(xRange=[0, self.imagesModels[0].imgShape[0]], yRange=[0, self.imagesModels[0].imgShape[1]],
                             padding=0)
        widget.ui.roiPlot.hide()

    @staticmethod
    def showMessage(header, message, button, icon):
        msg = QMessageBox()
        msg.setWindowTitle(header)
        msg.setText(message)
        msg.setIcon(icon)
        msg.setStandardButtons(button)
        x = msg.exec_()


def main():
    """
    the application startup functions
    :return:
    """
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = ImageProcessor(MainWindow)
    MainWindow.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
