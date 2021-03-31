# Importing Packages
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from UI import mainGUI as m
import cv2
import numpy as np
from imageModel import ImageModel
from threading import Thread

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


class LoaderThread(QThread):
    signal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        super(LoaderThread, self).__init__()
        self.filepath = ...
        self.file = ...

    def run(self):
        # self.file = loadAudioFile(self.filepath)
        self.signal.emit(self.file)

    # TODO Apply QThread for testing Median Filter
    # self.loadThread.filepath = self.filename
    # self.loadThread.start()
    # self.loadThread.signal.connect(self.loadFileConfiguration)

class ImageProcessor(m.Ui_MainWindow):
    """
    Main Class of the program GUI
    """

    def __init__(self, starter_window):
        """
        Main loop of the UI
        :param starter_window: QMainWindow Object
        """
        super(ImageProcessor, self).setupUi(starter_window)

        # Setup Tab Widget Connections
        self.tabWidget_process.setCurrentIndex(0)
        self.tab_index = self.tabWidget_process.currentIndex()
        self.tabWidget_process.currentChanged.connect(self.tab_changed)

        # Images Lists
        self.inputImages = [self.img1_input, self.img2_input, self.imgA_input, self.imgB_input]
        self.filtersImages = [self.img1_noisy, self.img1_filtered, self.img1_edged]
        self.histoImages = [self.img2_input_histo, self.img2_output_histo, self.img2_output]

        self.imageWidgets = [self.img1_input, self.img1_noisy, self.img1_filtered, self.img1_edged,
                             self.img2_input, self.img2_output,
                             self.imgA_input, self.imgB_input, self.imgX_output]

        # No Noisy Image Array yet
        self.currentNoiseImage = None

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
        self.sliders = [self.snr_slider_1, self.sigma_slider_1, self.mask_size_1, self.sigma_slider_2]

        # Sliders Connections
        for slider in self.sliders:
            slider.id = self.sliders.index(slider)
            slider.signal.connect(self.slider_changed)

        # Combo Lists
        self.updateCombos = [self.combo_noise, self.combo_filter, self.combo_edges]

        # Setup Load Buttons Connections
        self.btn_load_1.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_2.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_3.clicked.connect(lambda: self.load_file(self.tab_index))
        self.btn_load_4.clicked.connect(lambda: self.load_file(self.tab_index + 1))

        # # Setup Combo Connections
        self.combo_noise.activated.connect(lambda: self.combo_box_changed(self.tab_index, 0))
        self.combo_filter.activated.connect(lambda: self.combo_box_changed(self.tab_index, 1))
        self.combo_edges.activated.connect(lambda: self.combo_box_changed(self.tab_index, 2))
        self.combo_histogram.activated.connect(lambda: self.combo_box_changed(self.tab_index + 1, 3))

        # # Test Threads
        # # Setup Combo Connections
        # self.combo_noise.activated.connect(lambda: self.handle_combo(self.tab_index, 0))
        # self.combo_filter.activated.connect(lambda: self.handle_combo(self.tab_index, 1))
        # self.combo_edges.activated.connect(lambda: self.handle_combo(self.tab_index, 2))
        # self.combo_histogram.activated.connect(lambda: self.handle_combo(self.tab_index + 1, 3))

        self.setup_images_view()

        self.img2_input_histo.plotItem.setTitle("Histogram")
        self.img2_input_histo.plotItem.showGrid(True, True, alpha=0.8)
        self.img2_input_histo.plotItem.setLabel("bottom", text="Pixels")

        x = np.arange(1000)
        y = np.random.normal(size=(3, 1000))

        for i in range(3):
            # setting pen=(i,3) automatically creates three different-colored pens
            self.img2_input_histo.plot(x, y[i], pen=(i, 3))

    def tab_changed(self):
        self.tab_index = self.tabWidget_process.currentIndex()
        print(self.tab_index)

    def setup_images_view(self):
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

    def load_file(self, img_id):
        """
        Load the File from User
        :param img_id: 0, 1, 2 or 3
        :return:
        """

        # Open File & Check if it was loaded correctly
        logger.info("Browsing the files...")
        repo_path = "./src/Images"
        self.filename, self.format = QtWidgets.QFileDialog.getOpenFileName(None, "Load Image", repo_path,
                                                                           "*.jpg;;" "*.jpeg;;" "*.png;;")
        img_name = self.filename.split('/')[-1]
        if self.filename == "":
            pass
        else:
            image = cv2.imread(self.filename, flags=cv2.IMREAD_GRAYSCALE).T
            self.heights[img_id], self.weights[img_id] = image.shape
            self.imagesModels[img_id] = ImageModel(self.filename)

            # When Images in Tab1, Tab2 or Image A in Tab 3
            if img_id != 3:
                # Reset Results
                self.clear_results(tab_id=img_id)

                # Create and Display Original Image
                self.display_image(self.imagesModels[img_id].imgByte, self.inputImages[img_id])

                # Enable the combo box and parameters input
                self.enable_gui(tab_id=img_id)

                # Set Image Name and Size
                self.imagesLabels[img_id + 1][0].setText(img_name)
                self.imagesSizes[img_id + 1][0].setText(f"{image.shape[0]}x{image.shape[1]}")

                logger.info(f"Added Image{img_id + 1}: {img_name} successfully")

            # When Loading Image B in Tab 3
            else:
                if self.heights[3] != self.heights[2] or self.weights[3] != self.weights[2]:
                    self.show_message("Warning!!", "Images sizes must be the same, please upload another image",
                                      QMessageBox.Ok, QMessageBox.Warning)
                    logger.warning("Warning!!. Images sizes must be the same, please upload another image")
                else:
                    self.display_image(self.imagesModels[img_id].imgByte, self.inputImages[img_id])
                    # Set Image Name and Size
                    self.imagesLabels[img_id + 1][0].setText(img_name)
                    self.imagesSizes[img_id + 1][0].setText(f"{image.shape[0]}x{image.shape[1]}")
                    self.btn_hybrid.setEnabled(True)
                    logger.info(f"Added Image{img_id + 1}: {img_name} successfully")

    def enable_gui(self, tab_id):
        """
        This function enables the required elements in the gui
        :param tab_id: if of the current tab
        :return:
        """
        if tab_id == 0:
            for i in range(len(self.updateCombos)):
                # Enable Combo Boxes
                self.updateCombos[i].setEnabled(True)

        elif tab_id == 1:
            self.combo_histogram.setEnabled(True)
        elif tab_id == 2:
            if type(self.imagesModels[3]) != type(...):
                self.btn_hybrid.setEnabled(True)

    def clear_results(self, tab_id):
        # Reset previous outputs
        if tab_id == 0:
            # Clear Images Widgets
            for i in range(len(self.filtersImages)):
                self.filtersImages[i].clear()

    def handle_combo(self, tab_id, combo_id):
        # Start a new thread for this Client
        # Using Multiple threads to allow Multi-client connections.
        # Each thread handles one client in a separate function
        thread = Thread(target=self.combo_box_changed, args=(tab_id, combo_id))
        thread.start()

    def combo_box_changed(self, tab_id, combo_id):
        """

        :param tab_id:
        :param combo_id:
        :return:
        """

        # If 1st tab is selected
        if tab_id == 0:
            # Get Values from combo box and sliders
            selected_component = self.updateCombos[combo_id].currentText().lower()

            noise_snr = self.snr_slider_1.value() / 10
            noise_sigma = self.sigma_slider_1.value()  # This value from 0 -> 4
            noise_sigma = np.round(self.map_ranges(noise_sigma, 0, 4, 0, 255))  # This value from 0 -> 255

            filter_sigma = self.sigma_slider_2.value()
            filter_sigma = np.round(self.map_ranges(filter_sigma, 0, 4, 0, 255))

            mask_size = self.mask_size_1.value()
            mask_size = int(np.round(self.map_ranges(mask_size, 1, 4, 3, 9)))

            # Noise Options
            if combo_id == 0:
                self.snr_slider_1.setEnabled(True)
                if selected_component == "uniform noise":
                    self.currentNoiseImage = self.imagesModels[0].add_noise(type="uniform", snr=noise_snr)
                    self.display_image(data=self.currentNoiseImage, widget=self.filtersImages[combo_id])
                elif selected_component == "gaussian noise":
                    self.sigma_slider_1.setEnabled(True)
                    self.currentNoiseImage = self.imagesModels[0].add_noise(type="gaussian", snr=noise_snr, sigma=noise_sigma)
                    self.display_image(data=self.currentNoiseImage, widget=self.filtersImages[combo_id])
                elif selected_component == "salt & pepper noise":
                    self.currentNoiseImage = self.imagesModels[0].add_noise(type="salt & pepper", snr=noise_snr)
                    self.display_image(data=self.currentNoiseImage, widget=self.filtersImages[combo_id])

                # TODO Apply filter when changing the SNR of the noise if it is selected
                # Check if there is selected filter
                if self.currentNoiseImage is not None:
                    selected_component = self.updateCombos[1].currentText().lower()

            # Filters Options
            if combo_id == 1:
                self.mask_size_1.setEnabled(True)

                # Check if there's a noisy image already
                if self.currentNoiseImage is None:
                    self.show_message(header="Warning!!", message="Apply noise to the image first",
                                     button=QMessageBox.Ok, icon=QMessageBox.Warning)
                elif selected_component == "average filter":
                    filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="average",
                                                                       shape=mask_size)
                    self.display_image(data=filtered_image, widget=self.filtersImages[combo_id])
                elif selected_component == "gaussian filter":
                    self.sigma_slider_2.setEnabled(True)
                    filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="gaussian",
                                                                       shape=mask_size, sigma=filter_sigma)
                    self.display_image(data=filtered_image, widget=self.filtersImages[combo_id])
                elif selected_component == "median filter":
                    filtered_image = self.imagesModels[0].apply_filter(data=self.currentNoiseImage, type="median",
                                                                       shape=mask_size)
                    self.display_image(data=filtered_image, widget=self.filtersImages[combo_id])

            # Edge Detection Options
            if combo_id == 2:
                if selected_component == "sobel mask":
                    edged_image = self.imagesModels[0].apply_edge_mask(type="sobel")
                    self.display_image(edged_image, self.filtersImages[combo_id])
                elif selected_component == "roberts mask":
                    edged_image = self.imagesModels[0].apply_edge_mask(type="roberts")
                    self.display_image(edged_image, self.filtersImages[combo_id])
                elif selected_component == "prewitt mask":
                    edged_image = self.imagesModels[0].apply_edge_mask(type="perwitt")
                    self.display_image(edged_image, self.filtersImages[combo_id])
                elif selected_component == "canny mask":
                    edged_image = self.imagesModels[0].apply_edge_mask(type="canny")
                    self.display_image(edged_image, self.filtersImages[combo_id])

            logger.info(f"Viewing {selected_component} Component Of Image{combo_id + 1}")

        # If 2nd tab is selected
        elif tab_id == 1:
            selected_component = self.combo_histogram.currentText().lower()
            # Histograms Options
            if combo_id == 3:
                if selected_component == "original histogram":
                    histo = self.imagesModels[1].get_histogram(type="original")
                    # self.displayImage(histo, self.img2_input_histo)
                    # TODO plot histogram and distribution curve
                if selected_component == "equalized histogram":
                    histo = self.imagesModels[1].get_histogram(type="equalized")
                    # self.displayImage(histo, self.img2_output_histo)
                    # TODO plot histogram and distribution curve
                elif selected_component == "normalized histogram":
                    histo = self.imagesModels[1].get_histogram(type="normalized")
                    # self.displayImage(histo, self.img2_output_histo)

                elif selected_component == "local thresholding":
                    local_threshold = self.imagesModels[1].thresholding(type="local")
                    self.display_image(local_threshold, self.img2_output)
                elif selected_component == "global thresholding":
                    global_threshold = self.imagesModels[1].thresholding(type="global")
                    self.display_image(global_threshold, self.img2_output)
                elif selected_component == "transform to gray":
                    gray_image = self.imagesModels[1].to_gray()
                    self.display_image(gray_image, self.img2_output)

                    # TODO: Plot R, G and B Histograms separately

            logger.info(f"Viewing {selected_component} Component Of Image{combo_id + 1}")

    def slider_changed(self, indx, val):
        """
        detects the changes in the sliders using the indx given by ith slider
        and the slider value
        :param indx: int
        :param val: int
        :return: none
        """
        print(f"Slider {indx} With Value {val}")
        # self.sliderValuesClicked[indx] = val / 10
        if indx == 0 or indx == 1:
            self.combo_box_changed(tab_id=self.tab_index, combo_id=0)
        elif indx == 2 or indx == 3:
            self.combo_box_changed(tab_id=self.tab_index, combo_id=1)

    def display_image(self, data, widget):
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
    def map_ranges(inputValue: float, inMin: float, inMax: float, outMin: float, outMax: float):
        """
        Map a given value from range 1 -> range 2
        :param inputValue: The value you want to map
        :param inMin: Minimum Value of Range 1
        :param inMax: Maximum Value of Range 1
        :param outMin: Minimum Value of Range 2
        :param outMax: Maximum Value of Range 2
        :return: The new Value in Range 2
        """
        slope = (outMax - outMin) / (inMax - inMin)
        return outMin + slope * (inputValue - inMin)

    @staticmethod
    def show_message(header, message, button, icon):
        msg = QMessageBox()
        msg.setWindowTitle(header)
        msg.setText(message)
        msg.setIcon(icon)
        msg.setStandardButtons(button)
        msg.exec_()


def main():
    """
    the application startup functions
    :return:
    """
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ImageProcessor(main_window)
    main_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
