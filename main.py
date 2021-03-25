# Importing Packages
from PyQt5 import QtWidgets, QtCore
import mainGUI as m
import sys


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

        # Setup Button Connections
        # self.connect_btn.clicked.connect(self.connect_server)
        # self.disconnect_btn.clicked.connect(self.disconnect_server)
        # self.client_message_text.returnPressed.connect(self.message_changed)
        self.disableGUI()

    def connect_server(self):
        """
        This function responsible for:
        1.
        2.
        3.
        :return:
        """

        # self.client_name = self.client_name_text.text()

        # self.status_label.setText("Connected To Successfully!")
        self.enableGUI()


    def disconnect_server(self):
        """
        This function responsible for:
        1.
        2.
        :return:
        """

        # self.chat_area.clear()
        self.disableGUI()


    def disableGUI(self):
        """
        This function disables the non-necessary elements in the GUI
        if the user is not connected
        :return:
        """
        # self.client_message_text.setDisabled(True)
        # self.disconnect_btn.setDisabled(True)
        # self.chat_area.setDisabled(True)
        # self.chat_area.setReadOnly(True)

        pass

    def enableGUI(self):
        """
        This function enables the necessary elements in the GUI
        if the user is connected
        :return:
        """
        # self.client_message_text.setDisabled(False)
        # self.disconnect_btn.setDisabled(False)
        # self.chat_area.setDisabled(False)
        # self.chat_area.setReadOnly(True)

        pass


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
