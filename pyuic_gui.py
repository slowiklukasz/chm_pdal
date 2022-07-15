import sys, os
from PyQt5.QtWidgets import *


class DlgMain(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pyuic5 GUI")
        self.resize(800, 500)

        ##### Create widgets
        self.ledPyuic = QLineEdit(r"C:\Users\lukas\PycharmProjects\INWENTARZ\venv\Scripts/pyuic5")
        self.ledPyuic.setPlaceholderText("Location of pyuic5 file")
        self.ledPyuic.textChanged.connect(self.evt_change)

        self.ledUIFile = QLineEdit()
        self.ledUIFile.setPlaceholderText("Input .ui file")
        self.ledUIFile.textChanged.connect(self.evt_change)

        self.ledPyFile = QLineEdit()
        self.ledPyFile.setPlaceholderText("Output .py file")
        self.ledPyFile.textChanged.connect(self.evt_change)

        self.chkExecutable = QCheckBox("Make executable")
        self.chkExecutable.toggled.connect(self.evt_change)

        self.txtCommand = QPlainTextEdit()

        self.btnPyuic = QPushButton("...")
        self.btnPyuic.clicked.connect(self.evt_btnPyuic_clicked)

        self.btnUIFile = QPushButton("...")
        self.btnUIFile.clicked.connect(self.evt_btnUIFile_clicked)

        self.btnPyFile = QPushButton("...")
        self.btnPyFile.clicked.connect(self.evt_btnPyFile_clicked)

        self.btnExecute = QPushButton("Execute")
        self.btnExecute.clicked.connect(self.evt_btnExecute_clicked)

        self.lytMain = QVBoxLayout()
        self.lytPyuic = QHBoxLayout()
        self.lytUIFile = QHBoxLayout()
        self.lytPyFile = QHBoxLayout()
        self.lytCommand = QHBoxLayout()

        self.lytPyuic.addWidget(self.ledPyuic, 9)
        self.lytPyuic.addWidget(self.btnPyuic, 1)
        self.lytMain.addLayout(self.lytPyuic)
        self.lytUIFile.addWidget(self.ledUIFile, 9)
        self.lytUIFile.addWidget(self.btnUIFile, 1)
        self.lytMain.addLayout(self.lytUIFile)
        self.lytPyFile.addWidget(self.ledPyFile, 9)
        self.lytPyFile.addWidget(self.btnPyFile, 1)
        self.lytMain.addLayout(self.lytPyFile)
        self.lytMain.addWidget(self.chkExecutable)
        self.lytCommand.addWidget(self.txtCommand, 9)
        self.lytCommand.addWidget(self.btnExecute, 1)
        self.lytMain.addLayout(self.lytCommand)

        self.setLayout(self.lytMain)

    #####  Event handlers
    def evt_btnPyuic_clicked(self):
        sFile, sExt = QFileDialog.getOpenFileName(self, "Pyuic5", os.getcwd()+"/venv/bin", "Executable Files (*)")
        if sFile:
            self.ledPyuic.setText(sFile)

    def evt_btnUIFile_clicked(self):
        sFile, sExt = QFileDialog.getOpenFileName(self, "User Interface File", os.getcwd()+"/ui", "User Interface files (*.ui)")
        if sFile:
            self.ledUIFile.setText(sFile)
            bn = os.path.basename(sFile)
            self.ledPyFile.setText(os.getcwd()+"/ui_modules/"+bn.split(".")[0]+"_ui.py")

    def evt_btnPyFile_clicked(self):
        sFile, sExt = QFileDialog.getSaveFileName(self, "Python Module", os.getcwd()+"/ui_modules", "Python Files (*.py)")
        if sFile:
            self.ledPyFile.setText(sFile)

    def evt_btnExecute_clicked(self):
        os.system(self.txtCommand.toPlainText())
        QMessageBox.information(self, "Execute", "You have successfully compiled {} to {}".format(self.ledUIFile.text(), self.ledPyFile.text()))

    def evt_change(self):
        if self.chkExecutable.isChecked():
            chkEx = " -x"
        else:
            chkEx = ""
        self.txtCommand.setPlainText("{}{} {} -o {}".format(self.ledPyuic.text(), chkEx, self.ledUIFile.text(), self.ledPyFile.text()))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec_())