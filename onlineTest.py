# -*-coding:Latin-1 -*
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QDir, QObject, QRect, Qt, QSize, QDate, pyqtSignal, QCoreApplication, QThread
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QPixmap
import PyQt5.QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QProgressBar, QDialog
import numpy as np
from datetime import datetime
import mmpi

class onlineTest(QtWidgets.QDialog):

    def buttonTrue_clicked(self):
        
        if self.questionNumber - 1 >= 377:
            self.buttonTrue.setDisabled(True)
            self.buttonFalse.setDisabled(True)
            self.buttonNext.setDisabled(True)
        else:
            #print(self.questionNumber)
            self.trueAnswers[self.questionNumber - 1] = 0
            self.falseAnswers[self.questionNumber - 1] = 0
            self.trueAnswers[self.questionNumber-1] = 1
            #print(self.trueAnswers)

            if self.questionNumber - 1 >= 0 and self.questionNumber - 1 < 377:
                self.buttonTrue.setDisabled(False)
                self.buttonFalse.setDisabled(False)
                self.buttonPrevious.setDisabled(False)
                self.questionNumber = self.questionNumber + 1

            with open('questions.txt') as f:
                for i, line in enumerate(f, 1):
                    if i == self.questionNumber:
                        break

            self.questionLabel.setText(line)
            self.questionLabel.repaint()
        
    def buttonFalse_clicked(self):
        if self.questionNumber - 1 >= 377:
            self.buttonTrue.setDisabled(True)
            self.buttonFalse.setDisabled(True)
            self.buttonNext.setDisabled(True)
        else:
            #print(self.questionNumber)
            self.trueAnswers[self.questionNumber - 1] = 0
            self.falseAnswers[self.questionNumber - 1] = 0
            self.falseAnswers[self.questionNumber-1] = 1
            #print(self.falseAnswers)

            if self.questionNumber - 1 >= 0 and self.questionNumber - 1 < 377:
                self.buttonTrue.setDisabled(False)
                self.buttonFalse.setDisabled(False)
                self.buttonPrevious.setDisabled(False)
                self.questionNumber = self.questionNumber + 1

            with open('questions.txt') as f:
                for i, line in enumerate(f, 1):
                    if i == self.questionNumber:
                        break

            self.questionLabel.setText(line)
            self.questionLabel.repaint()
        
    def buttonNext_clicked(self):
        if self.questionNumber - 1 >= 377:
            self.buttonTrue.setDisabled(True)
            self.buttonFalse.setDisabled(True)
            self.buttonNext.setDisabled(True)
        else:
            self.buttonTrue.setDisabled(False)
            self.buttonFalse.setDisabled(False)
            self.buttonPrevious.setDisabled(False)
            self.questionNumber = self.questionNumber + 1

            with open('questions.txt') as f:
                for i, line in enumerate(f, 1):
                    if i == self.questionNumber:
                        break
            self.questionLabel.setText(line)
            self.questionLabel.repaint()
        
    def buttonPrevious_clicked(self):
        if self.questionNumber - 1 <= 0:
            self.buttonPrevious.setDisabled(True)
        else:
            self.buttonTrue.setDisabled(False)
            self.buttonFalse.setDisabled(False)
            self.buttonPrevious.setDisabled(False)
            
            self.questionNumber = self.questionNumber - 1
            with open('questions.txt') as f:
                for i, line in enumerate(f, 1):
                    if i == self.questionNumber:
                        break
            self.questionLabel.setText(line)
            self.questionLabel.repaint()
    
    def buttonFinish_clicked(self):
        
        self.firstName = str(self.firstName.text())
        self.lastName = str(self.lastName.text())
        if str(self.gender.currentText()) == 'Мужчина'.encode('latin1').decode('utf8'):
            self.sex='male'
        else:
            self.sex='female'
            
        self.age = str(self.age.text())
        self.work = str(self.work.text())
        self.testDate = datetime.now().strftime('%d_%m_%Y')
        self.birthDate = str(self.birthDay.currentText())+"_"+str(self.birthMonth.currentText())+"_"+str(self.birthYear.text())
        
        self.plotLoc = ''
        
        score_lfk, score_09 = mmpi.rawScoreCounter(self.trueAnswers, self.falseAnswers)
        
        lfk, rest = mmpi.tScore(self.sex, score_lfk, score_09)
        
        plotLcn = mmpi.graphPlotter(self.sex, str(self.firstName + self.lastName), self.age, datetime.now().strftime('%d%m%Y_%H%M%S'), lfk, rest)
        
        with open("profiles/"+self.firstName+self.lastName+"/"+self.firstName+"_"+self.lastName+"_"+datetime.now().strftime('%d%m%Y_%H%M%S') + ".txt", 'w') as file:
            file.write("First Name: " + self.firstName + "\n" +
                       "Last Name: " + self.lastName + "\n" +
                       "Sex: " + self.sex + "\n" + 
                       "Age: " + self.age + "\n" + 
                       "Work: " + self.work + "\n" + 
                       "Test Date: " + self.testDate + "\n" + 
                       "Birth Date: " + self.birthDate + "\n" + 
                       "LFK: " + str(lfk) + "\n" + 
                       "Scales: " + str(rest) + "\n")
            
        with open("profiles/"+self.firstName+self.lastName+"/"+self.firstName+"_"+self.lastName+"_"+datetime.now().strftime('%d%m%Y_%H%M%S') + "true.txt", 'w') as file:
            file.write(str(self.trueAnswers))
        with open("profiles/"+self.firstName+self.lastName+"/"+self.firstName+"_"+self.lastName+"_"+datetime.now().strftime('%d%m%Y_%H%M%S') + "false.txt", 'w') as file:
            file.write(str(self.falseAnswers))
            
        self.questionLabel.setPixmap(QPixmap("profiles/"+str(self.firstName+self.lastName) + "/" + plotLcn).scaled(791, 281))
        self.update()
        
        
    def __init__(self):
        self.directory = ''
		# load design
        super(onlineTest, self).__init__()
        uic.loadUi('gui/onlineTest.ui', self)
        self.buttonTrue.clicked.connect(self.buttonTrue_clicked)
        self.buttonFalse.clicked.connect(self.buttonFalse_clicked)
        self.buttonNext.clicked.connect(self.buttonNext_clicked)
        self.buttonPrevious.clicked.connect(self.buttonPrevious_clicked)
        self.buttonFinish.clicked.connect(self.buttonFinish_clicked)
        
        self.buttonTrue.setDisabled(True)
        self.buttonFalse.setDisabled(True)
        self.buttonPrevious.setDisabled(True)
        
        with open('instructions.txt', 'r') as f:
            self.questionLabel.setText(f.read())
            
        
        self.questionNumber = 0
        
        self.trueAnswers = np.zeros((377))
        self.falseAnswers = np.zeros((377))
                
        self.show()
