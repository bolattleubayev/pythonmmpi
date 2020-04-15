# -*-coding:Latin-1 -*
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QDir, QObject, QRect, Qt, QSize, QDate, pyqtSignal, QCoreApplication, QThread
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QPixmap
import PyQt5.QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QProgressBar, QDialog


class credits(QtWidgets.QDialog):
    
    def __init__(self):
        super(credits, self).__init__()
        uic.loadUi('gui/credits.ui', self)
        self.show()
