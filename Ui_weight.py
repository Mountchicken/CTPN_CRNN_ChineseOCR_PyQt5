# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\LearningStuff\DLcode\Pytorch\OCR\CTPN+CRNN my first ocr APP\weight.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1441, 879)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(20)
        Form.setFont(font)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(20, 30, 751, 711))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setEnabled(True)
        self.pushButton.setGeometry(QtCore.QRect(280, 760, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ink Free")
        font.setPointSize(16)
        self.pushButton.setFont(font)
        self.pushButton.setTabletTracking(False)
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(1050, 10, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Ink Free")
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(520, 760, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ink Free")
        font.setPointSize(16)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setTabletTracking(False)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(60, 760, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ink Free")
        font.setPointSize(16)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setTabletTracking(False)
        self.pushButton_3.setObjectName("pushButton_3")
        self.textBrowser = QtWidgets.QTextBrowser(Form)
        self.textBrowser.setGeometry(QtCore.QRect(880, 50, 491, 701))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(20)
        self.textBrowser.setFont(font)
        self.textBrowser.setObjectName("textBrowser")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "Load Image"))
        self.label_2.setText(_translate("Form", "Detected"))
        self.pushButton_2.setText(_translate("Form", "Detect"))
        self.pushButton_3.setText(_translate("Form", "Initializing"))
        self.textBrowser.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'黑体\',\'黑体\'; font-size:20pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'黑体\',\'Ink Free\'; font-size:24pt;\">Please touch the Initializing button first</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'黑体\',\'Ink Free\'; font-size:24pt;\"><br /></p></body></html>"))

