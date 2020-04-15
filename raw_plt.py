# -*-coding:Latin-1 -*
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QDir, QObject, QRect, Qt, QSize, QDate, pyqtSignal, QCoreApplication, QThread
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QPixmap
import PyQt5.QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QProgressBar, QDialog

import mmpi
from datetime import datetime
import os

class raw_plt(QtWidgets.QDialog):

    def __init__(self):

        self.directory = ''

        # load design
        super(raw_plt, self).__init__()
        uic.loadUi('gui/raw_plt.ui', self)

        self.pushButton.clicked.connect(self.pushButton_clicked)
        self.show()
    def pushButton_clicked(self):
        if str(self.gender.currentText()) == 'Мужчина'.encode('latin1').decode('utf8'):
            sex='male'
        else:
            sex='female'
        #loading the model


        name = str(self.name.text())
        age = str(self.age.text())
        time = datetime.now().strftime('%d%m%Y_%H%M%S')
        lfk = [int(self.L.text()), int(self.F.text()), int(self.K.text())]
        rest = [int(self.s1.text()), int(self.s2.text()), int(self.s3.text()),int(self.s4.text()), int(self.s5.text()), int(self.s6.text()),int(self.s7.text()), int(self.s8.text()), int(self.s9.text()), int(self.s0.text())]
        try:
            os.mkdir("profiles/"+str(name)+"/")
        except:
            pass
        plotLcn = mmpi.graphPlotter(sex, name, age, time, lfk, rest)
        self.plot.setPixmap(QPixmap("profiles/"+str(name)+"/"+plotLcn).scaled(650, 500))
        
        with open("profiles/"+name+"/"+name+"_"+ time + ".txt", 'w') as file:
            file.write("Name: " + name + "\n" +
                       "Sex: " + sex + "\n" +
                       "Age: " + age + "\n" +
                       "Test Date: " + time + "\n" +
                       "LFK: " + str(lfk) + "\n" +
                       "Scales: " + str(rest) + "\n")
        
        QApplication.processEvents()
        self.update()
        self.show()

