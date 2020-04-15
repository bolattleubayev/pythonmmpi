# -*-coding:Latin-1 -*
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QDir, QObject, QRect, Qt, QSize, QDate, pyqtSignal, QCoreApplication, QThread
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QPixmap
import PyQt5.QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QProgressBar, QDialog


import mmpi
import os
from raw_plt import *

class scanner(QtWidgets.QDialog):

    def __init__(self):
        self.directory = ''
		# load design
        super(scanner, self).__init__()
        uic.loadUi('gui/mmpi.ui', self)
        self.show()
		# add slots
        self.buttonGo.clicked.connect(self.buttonGo_clicked)
        self.buttonGo.setEnabled(False)

        self.buttonFile.clicked.connect(self.buttonFile_clicked)
        self.processed_image.setPixmap(QPixmap('/Users/macbook/Desktop/MMPI_last_win/misc/clear.jpg').scaled(650, 500))
        
    def buttonGo_clicked(self):
        #gender
        if str(self.gender.currentText()) == 'Мужчина'.encode('latin1').decode('utf8'):
            sex='male'
        else:
            sex='female'
        firstName= str(self.firstName.text())
        lastName=str(self.lastName.text())
        age = str(self.age.text())
        work = str(self.work.text())
        birthDate = str(self.birthDay.currentText())+"_"+str(self.birthMonth.currentText())+"_"+str(self.birthYear.text())
        notes = str(self.notes.text())
        #read image
        imgLocation = self.directory
        modelLocation= 'mmpi_reader_binary_classidication.h5py'
        p1 = mmpi.Person(firstName, lastName, sex, age, work, birthDate, notes, imgLocation,modelLocation)
 
        plotLcn = p1.plot()
        p1.writeToTextFile()
        self.processed_image.setPixmap(QPixmap("profiles/"+str(firstName+lastName) + "/" + plotLcn).scaled(650, 500))
        QApplication.processEvents()
        self.update()
        self.show()

            
	
    def buttonFile_clicked(self):
        self.directory,_ = QtWidgets.QFileDialog.getOpenFileName(self,'Single file',
			'/Users/macbook/Desktop/MMPI_last_win/raw','*')
		#self.raw_image.setPixmap(QPixmap(self.directory).scaled(650, 650))
        self.buttonGo.setEnabled(True)
	


