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

        # Setup Tab Widget Connections
        self.tabWidget_process.setCurrentIndex(0)
        self.tab_index = self.tabWidget_process.currentIndex()
        self.tabWidget_process.currentChanged.connect(self.tabChanged)

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
        self.imagesLabels = {1: [self.label_imgName_1], 2: [self.label_imgName_2],
                             3: [self.label_imgName_3], 4: [self.label_imgName_4]}

        self.imagesSizes = {1: [self.label_imgSize_1], 2: [self.label_imgSize_2],
                             3: [self.label_imgSize_3], 4: [self.label_imgSize_4]}

        # list contains the last pressed values
        self.sliderValuesClicked = {0: ..., 1: ..., 2: ..., 3: ...}
        self.sliders = [self.snr_slider_1, self.sigma_slider_1, self.sigma_slider_2, self.mask_size_1]


        # Sliders Connections
        for slider in self.sliders:
            slider.id = self.sliders.index(slider)
            slider.signal.connect(self.sliderChanged)

        # No Noisy Image Array yet
        self.currentNoiseImage = None

        # Combo Lists
        self.updateCombos = [self.combo_noise, self.combo_filter, self.combo_edges]

        # Setup Load Buttons Connections
        self.btn_load_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_2.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_3.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_4.clicked.connect(lambda: self.load_file(self.tab_index+1))

        # Setup Combo Connections
        self.combo_noise.activated.connect(lambda: self.updateCombosChanged(self.tab_index, 0))
        self.combo_filter.activated.connect(lambda: self.updateCombosChanged(self.tab_index, 1))
        self.combo_edges.activated.connect(lambda: self.updateCombosChanged(self.tab_index, 2))
        self.combo_histogram.activated.connect(lambda: self.updateCombosChanged(self.tab_index+1, 3))


        self.setupImagesView()

        self.img2_input_histo.plotItem.setTitle("Histogram")
        self.img2_input_histo.plotItem.showGrid(True, True, alpha=0.8)
        self.img2_input_histo.plotItem.setLabel("bottom", text="Pixels")

        x = np.arange(1000)
        y = np.random.normal(size=(3, 1000))

        for i in range(3):
            # setting pen=(i,3) automatically creates three different-colored pens
            self.img2_input_histo.plot(x, y[i], pen=(i, 3))


    def tabChanged(self):
        self.tab_index = self.tabWidget_process.currentIndex()
        print(self.tab_index)

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
                # Reset Results
                self.clearResults(tab_id=imgID)

                # Create and Display Original Image
                self.displayImage(self.imagesModels[imgID].imgByte, self.inputImages[imgID])

                # Enable the combo box and parameters input
                self.enableGUI(tab_id=imgID)

                # Set Image Name and Size
                self.imagesLabels[imgID+1][0].setText(imgName)
                self.imagesSizes[imgID+1][0].setText(f"{image.shape[0]}x{image.shape[1]}")

                logger.info(f"Added Image{imgID + 1}: {imgName} successfully")

            # When Loading Image B in Tab 3
            else:
                if self.heights[3] != self.heights[2] or self.weights[3] != self.weights[2]:
                    self.showMessage("Warning!!", "Images sizes must be the same, please upload another image",
                                     QMessageBox.Ok, QMessageBox.Warning)
                    logger.warning("Warning!!. Images sizes must be the same, please upload another image")
                else:
                    self.displayImage(self.imagesModels[imgID].imgByte, self.inputImages[imgID])
                    # Set Image Name and Size
                    self.imagesLabels[imgID + 1][0].setText(imgName)
                    self.imagesSizes[imgID + 1][0].setText(f"{image.shape[0]}x{image.shape[1]}")
                    self.btn_hybrid.setEnabled(True)
                    logger.info(f"Added Image{imgID + 1}: {imgName} successfully")


    def enableGUI(self, tab_id):
        """

        :param tab_id:
        :return:
        """
        if tab_id == 0:
            for i in range(len(self.updateCombos)):
                # Enable Combo Boxes
                self.updateCombos[i].setEnabled(True)

                # Enable Filters Parameters
                self.sliders[i].setEnabled(True)

        elif tab_id == 1:
            self.combo_histogram.setEnabled(True)
        elif tab_id == 2:
            if type(self.imagesModels[3]) != type(...):
                self.btn_hybrid.setEnabled(True)

    def clearResults(self, tab_id):
        # Reset previous outputs
        if tab_id == 0:
            # Clear Images Widgets
            for i in range(len(self.filtersImages)):
                self.filtersImages[i].clear()

            # Clear Text Labels



    def updateCombosChanged(self, tab_id, combo_id):
        """

        :param tab_id:
        :param combo_id:
        :return:
        """
        # If 1st tab is selected
        if tab_id == 0:
            selectedComponent = self.updateCombos[combo_id].currentText().lower()
            noise_snr = self.snr_slider_1.value() / 10
            noise_sigma = self.sigma_slider_1.value()

            filter_sigma = self.sigma_slider_2.value() / 10
            mask_size = self.mask_size_1.value()

            # Noise Options
            if combo_id == 0:
                if selectedComponent == "uniform noise":
                    self.currentNoiseImage = self.imagesModels[0].add_noise(type="uniform", snr=noise_snr)
                    self.displayImage(data=self.currentNoiseImage, widget=self.filtersImages[combo_id])
                elif selectedComponent == "gaussian noise":
                    self.currentNoiseImage = self.imagesModels[0].add_noise(type="gaussian", snr=noise_snr, sigma=noise_sigma)
                    self.displayImage(data=self.currentNoiseImage, widget=self.filtersImages[combo_id])
                elif selectedComponent == "salt & pepper noise":
                    self.currentNoiseImage = self.imagesModels[0].add_noise(type="salt & pepper", snr=noise_snr)
                    self.displayImage(data=self.currentNoiseImage, widget=self.filtersImages[combo_id])

            # Filters Options
            if combo_id == 1:
                if selectedComponent == "average filter":
                    filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="average", shape=mask_size)
                    self.displayImage(data=filtered_image, widget=self.filtersImages[combo_id])
                elif selectedComponent == "gaussian filter":
                    filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="gaussian", shape=mask_size, sigma=filter_sigma)
                    self.displayImage(data=filtered_image, widget=self.filtersImages[combo_id])
                elif selectedComponent == "median filter":
                    filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="median", shape=mask_size)
                    self.displayImage(data=filtered_image, widget=self.filtersImages[combo_id])

            # Edge Detection Options
            if combo_id == 2:
                if selectedComponent == "sobel mask":
                    edged_image = self.imagesModels[0].apply_edge_mask(type="sobel")
                    self.displayImage(edged_image, self.filtersImages[combo_id])
                elif selectedComponent == "roberts mask":
                    edged_image = self.imagesModels[0].apply_edge_mask(type="roberts")
                    self.displayImage(edged_image, self.filtersImages[combo_id])
                elif selectedComponent == "prewitt mask":
                    edged_image = self.imagesModels[0].apply_edge_mask(type="perwitt")
                    self.displayImage(edged_image, self.filtersImages[combo_id])
                elif selectedComponent == "canny mask":
                    edged_image = self.imagesModels[0].apply_edge_mask(type="canny")
                    self.displayImage(edged_image, self.filtersImages[combo_id])

            logger.info(f"Viewing {selectedComponent} Component Of Image{combo_id + 1}")

        # If 2nd tab is selected
        elif tab_id == 1:
            selectedComponent = self.combo_histogram.currentText().lower()
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

    def sliderChanged(self, indx, val):
        """
        detects the changes in the sliders using the indx given by ith slider
        and the slider value
        :param indx: int
        :param val: int
        :return: none
        """
        self.sliderValuesClicked[indx] = val/10
        self.updateCombosChanged(tab_id=0, combo_id=0)


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

    def mapRanges(self, inputValue: float, inMin: float, inMax: float, outMin: float, outMax: float):
        """
        Map a given value from range 1 -> range 2
        :param inputValue: The value you want to map
        :param inMin: Minimum Value of Range 1
        :param inMax: Maximum Value of Range 1
        :param outMin: Minimum Value of Range 2
        :param outMax: Maximum Value of Range 2
        :return: The new Value in Range 2
        """
        slope = (outMax-outMin) / (inMax-inMin)
        return outMin + slope*(inputValue-inMin)

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
