# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'search.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1101, 867)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_search = QtWidgets.QPushButton(self.centralwidget)
        self.btn_search.setGeometry(QtCore.QRect(840, 400, 93, 41))
        self.btn_search.setStyleSheet("background-color: rgb(255, 170, 0);\n"
"color: rgb(255, 255, 255);\n"
"font: 12pt \"黑体\";")
        self.btn_search.setObjectName("btn_search")
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        self.label_title.setGeometry(QtCore.QRect(410, 290, 281, 71))
        self.label_title.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 40pt \"黑体\";")
        self.label_title.setObjectName("label_title")
        self.text_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.text_edit.setGeometry(QtCore.QRect(180, 400, 641, 41))
        self.text_edit.setObjectName("text_edit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_search.setText(_translate("MainWindow", "搜索"))
        self.label_title.setText(_translate("MainWindow", "摆渡一下"))


