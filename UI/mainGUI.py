# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainGUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(754, 556)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(754, 510))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 700))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.tabWidget_process = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget_process.sizePolicy().hasHeightForWidth())
        self.tabWidget_process.setSizePolicy(sizePolicy)
        self.tabWidget_process.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget_process.setTabsClosable(False)
        self.tabWidget_process.setTabBarAutoHide(False)
        self.tabWidget_process.setObjectName("tabWidget_process")
        self.tab_1 = QtWidgets.QWidget()
        self.tab_1.setObjectName("tab_1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.btn_load_2 = QtWidgets.QPushButton(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_load_2.sizePolicy().hasHeightForWidth())
        self.btn_load_2.setSizePolicy(sizePolicy)
        self.btn_load_2.setMaximumSize(QtCore.QSize(200, 25))
        self.btn_load_2.setObjectName("btn_load_2")
        self.gridLayout_4.addWidget(self.btn_load_2, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_imgName_2 = QtWidgets.QLabel(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_imgName_2.sizePolicy().hasHeightForWidth())
        self.label_imgName_2.setSizePolicy(sizePolicy)
        self.label_imgName_2.setObjectName("label_imgName_2")
        self.horizontalLayout.addWidget(self.label_imgName_2)
        self.gridLayout_4.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.label_imgSize_2 = QtWidgets.QLabel(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_imgSize_2.sizePolicy().hasHeightForWidth())
        self.label_imgSize_2.setSizePolicy(sizePolicy)
        self.label_imgSize_2.setObjectName("label_imgSize_2")
        self.horizontalLayout_2.addWidget(self.label_imgSize_2)
        self.gridLayout_4.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(16777215, 200, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_4.addItem(spacerItem, 3, 0, 1, 1)
        self.combo_noise_2 = QtWidgets.QComboBox(self.tab_1)
        self.combo_noise_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.combo_noise_2.sizePolicy().hasHeightForWidth())
        self.combo_noise_2.setSizePolicy(sizePolicy)
        self.combo_noise_2.setMaximumSize(QtCore.QSize(200, 25))
        self.combo_noise_2.setObjectName("combo_noise_2")
        self.combo_noise_2.addItem("")
        self.combo_noise_2.addItem("")
        self.combo_noise_2.addItem("")
        self.combo_noise_2.addItem("")
        self.gridLayout_4.addWidget(self.combo_noise_2, 4, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(16777215, 200, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_4.addItem(spacerItem1, 5, 0, 1, 1)
        self.combo_filter_2 = QtWidgets.QComboBox(self.tab_1)
        self.combo_filter_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.combo_filter_2.sizePolicy().hasHeightForWidth())
        self.combo_filter_2.setSizePolicy(sizePolicy)
        self.combo_filter_2.setMaximumSize(QtCore.QSize(200, 25))
        self.combo_filter_2.setObjectName("combo_filter_2")
        self.combo_filter_2.addItem("")
        self.combo_filter_2.addItem("")
        self.combo_filter_2.addItem("")
        self.combo_filter_2.addItem("")
        self.gridLayout_4.addWidget(self.combo_filter_2, 6, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(16777215, 200, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_4.addItem(spacerItem2, 7, 0, 1, 1)
        self.combo_edges_2 = QtWidgets.QComboBox(self.tab_1)
        self.combo_edges_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.combo_edges_2.sizePolicy().hasHeightForWidth())
        self.combo_edges_2.setSizePolicy(sizePolicy)
        self.combo_edges_2.setMaximumSize(QtCore.QSize(200, 25))
        self.combo_edges_2.setObjectName("combo_edges_2")
        self.combo_edges_2.addItem("")
        self.combo_edges_2.addItem("")
        self.combo_edges_2.addItem("")
        self.combo_edges_2.addItem("")
        self.combo_edges_2.addItem("")
        self.gridLayout_4.addWidget(self.combo_edges_2, 8, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(16777215, 200, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_4.addItem(spacerItem3, 9, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout_4, 0, 0, 1, 1)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.groupBox = QtWidgets.QGroupBox(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.img1_original = ImageView(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img1_original.sizePolicy().hasHeightForWidth())
        self.img1_original.setSizePolicy(sizePolicy)
        self.img1_original.setMinimumSize(QtCore.QSize(200, 200))
        self.img1_original.setObjectName("img1_original")
        self.gridLayout.addWidget(self.img1_original, 0, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox, 0, 0, 1, 1)
        self.groupBox_8 = QtWidgets.QGroupBox(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_8.sizePolicy().hasHeightForWidth())
        self.groupBox_8.setSizePolicy(sizePolicy)
        self.groupBox_8.setObjectName("groupBox_8")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_8)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.img1_edged_2 = ImageView(self.groupBox_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img1_edged_2.sizePolicy().hasHeightForWidth())
        self.img1_edged_2.setSizePolicy(sizePolicy)
        self.img1_edged_2.setMinimumSize(QtCore.QSize(200, 200))
        self.img1_edged_2.setObjectName("img1_edged_2")
        self.gridLayout_10.addWidget(self.img1_edged_2, 0, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_8, 1, 1, 1, 1)
        self.groupBox_6 = QtWidgets.QGroupBox(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_6.sizePolicy().hasHeightForWidth())
        self.groupBox_6.setSizePolicy(sizePolicy)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.img1_filtered_2 = ImageView(self.groupBox_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img1_filtered_2.sizePolicy().hasHeightForWidth())
        self.img1_filtered_2.setSizePolicy(sizePolicy)
        self.img1_filtered_2.setMinimumSize(QtCore.QSize(200, 200))
        self.img1_filtered_2.setObjectName("img1_filtered_2")
        self.gridLayout_8.addWidget(self.img1_filtered_2, 0, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_6, 0, 1, 1, 1)
        self.groupBox_7 = QtWidgets.QGroupBox(self.tab_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_7.sizePolicy().hasHeightForWidth())
        self.groupBox_7.setSizePolicy(sizePolicy)
        self.groupBox_7.setObjectName("groupBox_7")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.groupBox_7)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.img1_noisy_2 = ImageView(self.groupBox_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img1_noisy_2.sizePolicy().hasHeightForWidth())
        self.img1_noisy_2.setSizePolicy(sizePolicy)
        self.img1_noisy_2.setMinimumSize(QtCore.QSize(200, 200))
        self.img1_noisy_2.setObjectName("img1_noisy_2")
        self.gridLayout_9.addWidget(self.img1_noisy_2, 0, 1, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_7, 1, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout_7, 0, 1, 1, 1)
        self.tabWidget_process.addTab(self.tab_1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget_process.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tabWidget_process.addTab(self.tab_3, "")
        self.horizontalLayout_4.addWidget(self.tabWidget_process)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 754, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionImage_1 = QtWidgets.QAction(MainWindow)
        self.actionImage_1.setObjectName("actionImage_1")
        self.actionImage_2 = QtWidgets.QAction(MainWindow)
        self.actionImage_2.setObjectName("actionImage_2")

        self.retranslateUi(MainWindow)
        self.tabWidget_process.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_load_2.setText(_translate("MainWindow", "Load Image"))
        self.label.setText(_translate("MainWindow", "Name:"))
        self.label_imgName_2.setText(_translate("MainWindow", "No Image"))
        self.label_2.setText(_translate("MainWindow", "Size:"))
        self.label_imgSize_2.setText(_translate("MainWindow", "No Image."))
        self.combo_noise_2.setItemText(0, _translate("MainWindow", "Add Noise"))
        self.combo_noise_2.setItemText(1, _translate("MainWindow", "Uniform Noise"))
        self.combo_noise_2.setItemText(2, _translate("MainWindow", "Gaussian Noise"))
        self.combo_noise_2.setItemText(3, _translate("MainWindow", "Salt & Pepper Noise"))
        self.combo_filter_2.setItemText(0, _translate("MainWindow", "Apply Filter"))
        self.combo_filter_2.setItemText(1, _translate("MainWindow", "Average Filter"))
        self.combo_filter_2.setItemText(2, _translate("MainWindow", "Gaussian Filter"))
        self.combo_filter_2.setItemText(3, _translate("MainWindow", "Median Filter"))
        self.combo_edges_2.setCurrentText(_translate("MainWindow", "Detect Edges"))
        self.combo_edges_2.setItemText(0, _translate("MainWindow", "Detect Edges"))
        self.combo_edges_2.setItemText(1, _translate("MainWindow", "Sobel Mask"))
        self.combo_edges_2.setItemText(2, _translate("MainWindow", "Roberts Mask"))
        self.combo_edges_2.setItemText(3, _translate("MainWindow", "Prewitt Mask"))
        self.combo_edges_2.setItemText(4, _translate("MainWindow", "Canny Mask"))
        self.groupBox.setTitle(_translate("MainWindow", "Original Image"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Edge Detection Image"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Filtered Image"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Noisy Image"))
        self.tabWidget_process.setTabText(self.tabWidget_process.indexOf(self.tab_1), _translate("MainWindow", "Filters"))
        self.tabWidget_process.setTabText(self.tabWidget_process.indexOf(self.tab_2), _translate("MainWindow", "Histograms"))
        self.tabWidget_process.setTabText(self.tabWidget_process.indexOf(self.tab_3), _translate("MainWindow", "Hybrid"))
        self.actionImage_1.setText(_translate("MainWindow", "Image 1"))
        self.actionImage_2.setText(_translate("MainWindow", "Image 2"))
from pyqtgraph import ImageView
