# -*-coding:Latin-1 -*
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QDir, QObject, QRect, Qt, QSize, QDate, pyqtSignal, QCoreApplication, QThread
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QPixmap
import PyQt5.QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QProgressBar, QDialog


import mmpi
import os
from scanner import *
from raw_plt import *
from onlineTest import *
from credits import *

class GUIMain(QtWidgets.QDialog):
    def plotButton_clicked(self):
        self.raw_plt = raw_plt()
    def scannerButton_clicked(self):
        self.scanner = scanner()
    def onlineTestButton_clicked(self):
        self.onlineTest = onlineTest()
    def creditsButton_clicked(self):
        self.credits = credits()
    
    def __init__(self):
        self.directory = ''
		# load design
        super(GUIMain, self).__init__()
        uic.loadUi('gui/settings.ui', self)
        self.show()
        ui1 = scanner()
        ui2 = raw_plt()
        ui3 = onlineTest()
        ui4 = credits()
		# add slots
        self.scannerButton.clicked.connect(self.scannerButton_clicked)
        self.plotButton.clicked.connect(self.plotButton_clicked)
        self.onlineTestButton.clicked.connect(self.onlineTestButton_clicked)
        self.creditsButton.clicked.connect(self.creditsButton_clicked)

		
        


	


