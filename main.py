#!/usr/bin/env python
from PyQt5 import uic, QtWidgets
from GUIMain import GUIMain

import sys
import os


if __name__ == "__main__":

	app = QtWidgets.QApplication(sys.argv)
	GUImain = GUIMain()
	sys.exit(app.exec_())


	

	
