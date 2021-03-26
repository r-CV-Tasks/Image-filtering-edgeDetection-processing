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

        # Load Buttons
        # self.loadButtons = [self.actionImage_1, self.actionImage_2]

        # Images Lists
        self.inputImages = [self.img1_input, self.img2_input, self.imgA_input, self.imgB_input]
        self.filtersImages = [self.img1_noisy, self.img1_filtered, self.img1_edged]
        self.imageWidgets = [self.img1_input, self.img1_noisy, self.img1_filtered, self.img1_edged,
                             self.img2_input, self.img2_input_histo, self.img2_output_histo, self.img2_output,
                             self.imgA_input, self.imgB_input, self.imgX_output]

        self.imagesModels = [..., ..., ..., ...]
        self.heights = [..., ...]
        self.weights = [..., ...]

        # No Noisy Image Array yet
        self.currentNoiseImage = None

        # Combo Listss
        self.updateCombos = [self.combo_noise, self.combo_filter, self.combo_edges]

        # Setup Combo Connections
        self.combo_noise.activated.connect(lambda: self.updateCombosChanged(0))
        self.combo_filter.activated.connect(lambda: self.updateCombosChanged(1))
        self.combo_edges.activated.connect(lambda: self.updateCombosChanged(2))

        # Setup Load Button Connection
        self.btn_load_1.clicked.connect(lambda: self.load_file(0))

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
        :param imgID: 0 or 1
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

            if type(self.imagesModels[~imgID]) == type(...):
                # Create and Display Original Image
                self.displayImage(self.imagesModels[imgID].imgByte, self.inputImages[imgID])
                self.updateCombos[imgID].setEnabled(True)
                self.updateCombos[imgID+1].setEnabled(True)
                self.updateCombos[imgID+2].setEnabled(True)

                self.label_imgName_1.setText(imgName)
                self.label_imgSize_1.setText(f"{image.shape[0]}x{image.shape[1]}")

                logger.info(f"Added Image{imgID + 1}: {imgName} successfully")
            else:
                if self.heights[1] != self.heights[0] or self.weights[1] != self.weights[0]:
                    self.showMessage("Warning!!", "Image sizes must be the same, please upload another image",
                                     QMessageBox.Ok, QMessageBox.Warning)
                    logger.warning("Warning!!. Image sizes must be the same, please upload another image")
                else:
                    self.displayImage(self.imagesModels[imgID].imgByte, self.inputImages[imgID])
                    self.updateCombos[imgID].setEnabled(True)
                    logger.info(f"Added Image{imgID + 1}: {imgName} successfully")

    def updateCombosChanged(self, id):
        selectedComponent = self.updateCombos[id].currentText().lower()

        # Noise Options
        if selectedComponent == "uniform noise":
            self.displayImage(self.imagesModels[0].uniform_noise, self.outputImages[id])
            self.currentNoiseImage = self.imagesModels[0].uniform_noise
        elif selectedComponent == "gaussian noise":
            self.displayImage(self.imagesModels[0].gaussian_noise, self.outputImages[id])
            self.currentNoiseImage = self.imagesModels[0].gaussian_noise
        elif selectedComponent == "salt & pepper noise":
            self.displayImage(self.imagesModels[0].saltpepper_noise, self.outputImages[id])
            self.currentNoiseImage = self.imagesModels[0].saltpepper_noise

        # Filters Options
        if selectedComponent == "average filter":
            filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="average")
            self.displayImage(filtered_image, self.outputImages[id])
        elif selectedComponent == "gaussian filter":
            filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="gaussian")
            self.displayImage(filtered_image, self.outputImages[id])
        elif selectedComponent == "median filter":
            filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="median")
            self.displayImage(filtered_image, self.outputImages[id])

        # Edge Detection Options
        if selectedComponent == "sobel mask":
            edged_image = self.imagesModels[0].apply_edge_mask(type="sobel")
            self.displayImage(edged_image, self.outputImages[id])
        elif selectedComponent == "roberts mask":
            edged_image = self.imagesModels[0].apply_edge_mask(type="roberts")
            self.displayImage(edged_image, self.outputImages[id])
        elif selectedComponent == "perwitt mask":
            edged_image = self.imagesModels[0].apply_edge_mask(type="perwitt")
            self.displayImage(edged_image, self.outputImages[id])
        elif selectedComponent == "canny mask":
            edged_image = self.imagesModels[0].apply_edge_mask(type="canny")
            self.displayImage(edged_image, self.outputImages[id])

        logger.info(f"Viewing {selectedComponent} Component Of Image{id + 1}")

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
