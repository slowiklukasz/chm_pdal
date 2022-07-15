# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:/Users/lukas/PycharmProjects/GIS_PROJECTS/CHM/CHM_PDAL/CHM_CALCULATOR/ui/chm_dlg.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dlg_chm(object):
    def setupUi(self, dlg_chm):
        dlg_chm.setObjectName("dlg_chm")
        dlg_chm.setWindowModality(QtCore.Qt.NonModal)
        dlg_chm.resize(597, 706)
        self.stc_wdg = QtWidgets.QStackedWidget(dlg_chm)
        self.stc_wdg.setGeometry(QtCore.QRect(0, 10, 591, 691))
        self.stc_wdg.setObjectName("stc_wdg")
        self.page_menu = QtWidgets.QWidget()
        self.page_menu.setObjectName("page_menu")
        self.verticalFrame_2 = QtWidgets.QFrame(self.page_menu)
        self.verticalFrame_2.setGeometry(QtCore.QRect(110, 180, 361, 341))
        self.verticalFrame_2.setObjectName("verticalFrame_2")
        self.lyt_select = QtWidgets.QVBoxLayout(self.verticalFrame_2)
        self.lyt_select.setObjectName("lyt_select")
        self.btn_sel_com = QtWidgets.QPushButton(self.verticalFrame_2)
        self.btn_sel_com.setMinimumSize(QtCore.QSize(150, 75))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_sel_com.setFont(font)
        self.btn_sel_com.setObjectName("btn_sel_com")
        self.lyt_select.addWidget(self.btn_sel_com)
        self.btn_sel_lid = QtWidgets.QPushButton(self.verticalFrame_2)
        self.btn_sel_lid.setEnabled(True)
        self.btn_sel_lid.setMinimumSize(QtCore.QSize(120, 75))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_sel_lid.setFont(font)
        self.btn_sel_lid.setObjectName("btn_sel_lid")
        self.lyt_select.addWidget(self.btn_sel_lid)
        self.btn_sel_seg = QtWidgets.QPushButton(self.verticalFrame_2)
        self.btn_sel_seg.setMinimumSize(QtCore.QSize(150, 75))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_sel_seg.setFont(font)
        self.btn_sel_seg.setObjectName("btn_sel_seg")
        self.lyt_select.addWidget(self.btn_sel_seg)
        self.stc_wdg.addWidget(self.page_menu)
        self.page_cpl = QtWidgets.QWidget()
        self.page_cpl.setObjectName("page_cpl")
        self.horizontalFrame_3 = QtWidgets.QFrame(self.page_cpl)
        self.horizontalFrame_3.setGeometry(QtCore.QRect(10, 630, 581, 61))
        self.horizontalFrame_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.horizontalFrame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.horizontalFrame_3.setObjectName("horizontalFrame_3")
        self.lyt_com_btn = QtWidgets.QHBoxLayout(self.horizontalFrame_3)
        self.lyt_com_btn.setObjectName("lyt_com_btn")
        self.btn_com_rtn = QtWidgets.QPushButton(self.horizontalFrame_3)
        self.btn_com_rtn.setMinimumSize(QtCore.QSize(0, 0))
        self.btn_com_rtn.setMaximumSize(QtCore.QSize(130, 45))
        self.btn_com_rtn.setBaseSize(QtCore.QSize(0, 0))
        self.btn_com_rtn.setObjectName("btn_com_rtn")
        self.lyt_com_btn.addWidget(self.btn_com_rtn)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.lyt_com_btn.addItem(spacerItem)
        self.btn_com_exit = QtWidgets.QPushButton(self.horizontalFrame_3)
        self.btn_com_exit.setMaximumSize(QtCore.QSize(130, 45))
        self.btn_com_exit.setMouseTracking(False)
        self.btn_com_exit.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.btn_com_exit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_com_exit.setDefault(True)
        self.btn_com_exit.setObjectName("btn_com_exit")
        self.lyt_com_btn.addWidget(self.btn_com_exit)
        self.btn_com_cal = QtWidgets.QPushButton(self.horizontalFrame_3)
        self.btn_com_cal.setMaximumSize(QtCore.QSize(130, 45))
        self.btn_com_cal.setMouseTracking(False)
        self.btn_com_cal.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.btn_com_cal.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_com_cal.setDefault(True)
        self.btn_com_cal.setObjectName("btn_com_cal")
        self.lyt_com_btn.addWidget(self.btn_com_cal)
        self.verticalFrame = QtWidgets.QFrame(self.page_cpl)
        self.verticalFrame.setGeometry(QtCore.QRect(10, 369, 581, 251))
        self.verticalFrame.setFrameShape(QtWidgets.QFrame.Box)
        self.verticalFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalFrame.setLineWidth(-2)
        self.verticalFrame.setObjectName("verticalFrame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalFrame)
        self.verticalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_3.setSpacing(5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.txt_com_cmd = QtWidgets.QPlainTextEdit(self.verticalFrame)
        self.txt_com_cmd.setEnabled(True)
        self.txt_com_cmd.setPlainText("")
        self.txt_com_cmd.setObjectName("txt_com_cmd")
        self.verticalLayout_3.addWidget(self.txt_com_cmd)
        self.verticalFrame_5 = QtWidgets.QFrame(self.page_cpl)
        self.verticalFrame_5.setGeometry(QtCore.QRect(10, 0, 581, 61))
        self.verticalFrame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.verticalFrame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalFrame_5.setObjectName("verticalFrame_5")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.verticalFrame_5)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.lbl_com_menu = QtWidgets.QLabel(self.verticalFrame_5)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_com_menu.setFont(font)
        self.lbl_com_menu.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_com_menu.setObjectName("lbl_com_menu")
        self.verticalLayout_6.addWidget(self.lbl_com_menu)
        self.frame_7 = QtWidgets.QFrame(self.page_cpl)
        self.frame_7.setGeometry(QtCore.QRect(10, 60, 581, 121))
        self.frame_7.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.lyt_com_6 = QtWidgets.QFormLayout(self.frame_7)
        self.lyt_com_6.setContentsMargins(10, 10, 10, 10)
        self.lyt_com_6.setObjectName("lyt_com_6")
        self.lbl_com_in = QtWidgets.QLabel(self.frame_7)
        self.lbl_com_in.setObjectName("lbl_com_in")
        self.lyt_com_6.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lbl_com_in)
        self.lyt_input_4 = QtWidgets.QHBoxLayout()
        self.lyt_input_4.setObjectName("lyt_input_4")
        self.led_com_in = QtWidgets.QLineEdit(self.frame_7)
        self.led_com_in.setEnabled(True)
        self.led_com_in.setReadOnly(False)
        self.led_com_in.setObjectName("led_com_in")
        self.lyt_input_4.addWidget(self.led_com_in)
        self.btn_com_in = QtWidgets.QToolButton(self.frame_7)
        self.btn_com_in.setObjectName("btn_com_in")
        self.lyt_input_4.addWidget(self.btn_com_in)
        self.lyt_com_6.setLayout(1, QtWidgets.QFormLayout.FieldRole, self.lyt_input_4)
        self.lbl_com_out = QtWidgets.QLabel(self.frame_7)
        self.lbl_com_out.setObjectName("lbl_com_out")
        self.lyt_com_6.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.lbl_com_out)
        self.lyt_output_4 = QtWidgets.QHBoxLayout()
        self.lyt_output_4.setObjectName("lyt_output_4")
        self.led_com_out = QtWidgets.QLineEdit(self.frame_7)
        self.led_com_out.setEnabled(True)
        self.led_com_out.setDragEnabled(False)
        self.led_com_out.setReadOnly(False)
        self.led_com_out.setObjectName("led_com_out")
        self.lyt_output_4.addWidget(self.led_com_out)
        self.btn_com_out = QtWidgets.QToolButton(self.frame_7)
        self.btn_com_out.setObjectName("btn_com_out")
        self.lyt_output_4.addWidget(self.btn_com_out)
        self.lyt_com_6.setLayout(2, QtWidgets.QFormLayout.FieldRole, self.lyt_output_4)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.lyt_com_6.setItem(0, QtWidgets.QFormLayout.FieldRole, spacerItem1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.lyt_com_6.setItem(3, QtWidgets.QFormLayout.FieldRole, spacerItem2)
        self.frame_8 = QtWidgets.QFrame(self.page_cpl)
        self.frame_8.setGeometry(QtCore.QRect(10, 180, 581, 191))
        self.frame_8.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.lyt_com_7 = QtWidgets.QFormLayout(self.frame_8)
        self.lyt_com_7.setObjectName("lyt_com_7")
        self.lbl_seg_out_3 = QtWidgets.QLabel(self.frame_8)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.lbl_seg_out_3.setFont(font)
        self.lbl_seg_out_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_seg_out_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lbl_seg_out_3.setObjectName("lbl_seg_out_3")
        self.lyt_com_7.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lbl_seg_out_3)
        self.frame_9 = QtWidgets.QFrame(self.frame_8)
        self.frame_9.setMinimumSize(QtCore.QSize(80, 130))
        self.frame_9.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_9.setObjectName("frame_9")
        self.lyt_grid_6 = QtWidgets.QGridLayout(self.frame_9)
        self.lyt_grid_6.setObjectName("lyt_grid_6")
        self.cmb_com_ftp = QtWidgets.QComboBox(self.frame_9)
        self.cmb_com_ftp.setObjectName("cmb_com_ftp")
        self.cmb_com_ftp.addItem("")
        self.cmb_com_ftp.addItem("")
        self.cmb_com_ftp.addItem("")
        self.lyt_grid_6.addWidget(self.cmb_com_ftp, 0, 1, 1, 1)
        self.dbx_com_comp = QtWidgets.QDoubleSpinBox(self.frame_9)
        self.dbx_com_comp.setDecimals(1)
        self.dbx_com_comp.setSingleStep(0.1)
        self.dbx_com_comp.setProperty("value", 1.3)
        self.dbx_com_comp.setObjectName("dbx_com_comp")
        self.lyt_grid_6.addWidget(self.dbx_com_comp, 3, 1, 1, 1)
        self.lbl_com_ftp = QtWidgets.QLabel(self.frame_9)
        self.lbl_com_ftp.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_com_ftp.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lbl_com_ftp.setObjectName("lbl_com_ftp")
        self.lyt_grid_6.addWidget(self.lbl_com_ftp, 0, 0, 1, 1)
        self.lbl_com_gaus = QtWidgets.QLabel(self.frame_9)
        self.lbl_com_gaus.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_com_gaus.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lbl_com_gaus.setObjectName("lbl_com_gaus")
        self.lyt_grid_6.addWidget(self.lbl_com_gaus, 2, 0, 1, 1)
        self.cmb_com_line = QtWidgets.QComboBox(self.frame_9)
        self.cmb_com_line.setObjectName("cmb_com_line")
        self.cmb_com_line.addItem("")
        self.cmb_com_line.addItem("")
        self.lyt_grid_6.addWidget(self.cmb_com_line, 4, 1, 1, 1)
        self.cmb_com_gaus = QtWidgets.QComboBox(self.frame_9)
        self.cmb_com_gaus.setObjectName("cmb_com_gaus")
        self.cmb_com_gaus.addItem("")
        self.cmb_com_gaus.addItem("")
        self.cmb_com_gaus.addItem("")
        self.cmb_com_gaus.addItem("")
        self.cmb_com_gaus.addItem("")
        self.cmb_com_gaus.addItem("")
        self.lyt_grid_6.addWidget(self.cmb_com_gaus, 2, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.lyt_grid_6.addItem(spacerItem3, 5, 0, 1, 1)
        self.lbl_com_comp = QtWidgets.QLabel(self.frame_9)
        self.lbl_com_comp.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_com_comp.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lbl_com_comp.setLineWidth(1)
        self.lbl_com_comp.setObjectName("lbl_com_comp")
        self.lyt_grid_6.addWidget(self.lbl_com_comp, 3, 0, 1, 1)
        self.lbl_com_line = QtWidgets.QLabel(self.frame_9)
        self.lbl_com_line.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_com_line.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lbl_com_line.setObjectName("lbl_com_line")
        self.lyt_grid_6.addWidget(self.lbl_com_line, 4, 0, 1, 1)
        self.lbl_com_trun = QtWidgets.QLabel(self.frame_9)
        self.lbl_com_trun.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_com_trun.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lbl_com_trun.setObjectName("lbl_com_trun")
        self.lyt_grid_6.addWidget(self.lbl_com_trun, 1, 0, 1, 1)
        self.cmb_com_trun = QtWidgets.QComboBox(self.frame_9)
        self.cmb_com_trun.setObjectName("cmb_com_trun")
        self.cmb_com_trun.addItem("")
        self.cmb_com_trun.addItem("")
        self.cmb_com_trun.addItem("")
        self.cmb_com_trun.addItem("")
        self.cmb_com_trun.addItem("")
        self.cmb_com_trun.addItem("")
        self.lyt_grid_6.addWidget(self.cmb_com_trun, 1, 1, 1, 1)
        self.lyt_com_7.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.frame_9)
        self.stc_wdg.addWidget(self.page_cpl)
        self.page_lid = QtWidgets.QWidget()
        self.page_lid.setObjectName("page_lid")
        self.verticalFrame_3 = QtWidgets.QFrame(self.page_lid)
        self.verticalFrame_3.setGeometry(QtCore.QRect(10, 0, 581, 61))
        self.verticalFrame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.verticalFrame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalFrame_3.setObjectName("verticalFrame_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalFrame_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lbl_lid_menu = QtWidgets.QLabel(self.verticalFrame_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_lid_menu.setFont(font)
        self.lbl_lid_menu.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_lid_menu.setObjectName("lbl_lid_menu")
        self.verticalLayout_2.addWidget(self.lbl_lid_menu)
        self.horizontalFrame_5 = QtWidgets.QFrame(self.page_lid)
        self.horizontalFrame_5.setGeometry(QtCore.QRect(10, 630, 581, 61))
        self.horizontalFrame_5.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.horizontalFrame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.horizontalFrame_5.setObjectName("horizontalFrame_5")
        self.lyt_com_btn_3 = QtWidgets.QHBoxLayout(self.horizontalFrame_5)
        self.lyt_com_btn_3.setObjectName("lyt_com_btn_3")
        self.btn_lid_rtn = QtWidgets.QPushButton(self.horizontalFrame_5)
        self.btn_lid_rtn.setMinimumSize(QtCore.QSize(0, 0))
        self.btn_lid_rtn.setMaximumSize(QtCore.QSize(130, 45))
        self.btn_lid_rtn.setBaseSize(QtCore.QSize(0, 0))
        self.btn_lid_rtn.setObjectName("btn_lid_rtn")
        self.lyt_com_btn_3.addWidget(self.btn_lid_rtn)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.lyt_com_btn_3.addItem(spacerItem4)
        self.btn_lid_exit = QtWidgets.QPushButton(self.horizontalFrame_5)
        self.btn_lid_exit.setMaximumSize(QtCore.QSize(130, 45))
        self.btn_lid_exit.setMouseTracking(False)
        self.btn_lid_exit.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.btn_lid_exit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_lid_exit.setDefault(True)
        self.btn_lid_exit.setObjectName("btn_lid_exit")
        self.lyt_com_btn_3.addWidget(self.btn_lid_exit)
        self.btn_lid_cal = QtWidgets.QPushButton(self.horizontalFrame_5)
        self.btn_lid_cal.setMaximumSize(QtCore.QSize(130, 45))
        self.btn_lid_cal.setMouseTracking(False)
        self.btn_lid_cal.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.btn_lid_cal.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_lid_cal.setDefault(True)
        self.btn_lid_cal.setObjectName("btn_lid_cal")
        self.lyt_com_btn_3.addWidget(self.btn_lid_cal)
        self.verticalFrame1 = QtWidgets.QFrame(self.page_lid)
        self.verticalFrame1.setGeometry(QtCore.QRect(10, 188, 581, 431))
        self.verticalFrame1.setFrameShape(QtWidgets.QFrame.Box)
        self.verticalFrame1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalFrame1.setLineWidth(-2)
        self.verticalFrame1.setObjectName("verticalFrame1")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalFrame1)
        self.verticalLayout_5.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_5.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.txt_lid_cmd = QtWidgets.QPlainTextEdit(self.verticalFrame1)
        self.txt_lid_cmd.setObjectName("txt_lid_cmd")
        self.verticalLayout_5.addWidget(self.txt_lid_cmd)
        self.frame_10 = QtWidgets.QFrame(self.page_lid)
        self.frame_10.setGeometry(QtCore.QRect(10, 60, 581, 121))
        self.frame_10.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.lyt_com_8 = QtWidgets.QFormLayout(self.frame_10)
        self.lyt_com_8.setContentsMargins(10, 10, 10, 10)
        self.lyt_com_8.setObjectName("lyt_com_8")
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.lyt_com_8.setItem(0, QtWidgets.QFormLayout.FieldRole, spacerItem5)
        self.lbl_lid_in = QtWidgets.QLabel(self.frame_10)
        self.lbl_lid_in.setObjectName("lbl_lid_in")
        self.lyt_com_8.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lbl_lid_in)
        self.lyt_input_5 = QtWidgets.QHBoxLayout()
        self.lyt_input_5.setObjectName("lyt_input_5")
        self.led_lid_in = QtWidgets.QLineEdit(self.frame_10)
        self.led_lid_in.setEnabled(True)
        self.led_lid_in.setReadOnly(False)
        self.led_lid_in.setObjectName("led_lid_in")
        self.lyt_input_5.addWidget(self.led_lid_in)
        self.btn_lid_in = QtWidgets.QToolButton(self.frame_10)
        self.btn_lid_in.setObjectName("btn_lid_in")
        self.lyt_input_5.addWidget(self.btn_lid_in)
        self.lyt_com_8.setLayout(1, QtWidgets.QFormLayout.FieldRole, self.lyt_input_5)
        self.lbl_lid_out = QtWidgets.QLabel(self.frame_10)
        self.lbl_lid_out.setObjectName("lbl_lid_out")
        self.lyt_com_8.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.lbl_lid_out)
        self.lyt_output_5 = QtWidgets.QHBoxLayout()
        self.lyt_output_5.setObjectName("lyt_output_5")
        self.led_lid_out = QtWidgets.QLineEdit(self.frame_10)
        self.led_lid_out.setEnabled(True)
        self.led_lid_out.setDragEnabled(False)
        self.led_lid_out.setReadOnly(False)
        self.led_lid_out.setObjectName("led_lid_out")
        self.lyt_output_5.addWidget(self.led_lid_out)
        self.btn_lid_out = QtWidgets.QToolButton(self.frame_10)
        self.btn_lid_out.setObjectName("btn_lid_out")
        self.lyt_output_5.addWidget(self.btn_lid_out)
        self.lyt_com_8.setLayout(2, QtWidgets.QFormLayout.FieldRole, self.lyt_output_5)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.lyt_com_8.setItem(3, QtWidgets.QFormLayout.FieldRole, spacerItem6)
        self.stc_wdg.addWidget(self.page_lid)
        self.page_seg = QtWidgets.QWidget()
        self.page_seg.setObjectName("page_seg")
        self.verticalFrame_4 = QtWidgets.QFrame(self.page_seg)
        self.verticalFrame_4.setGeometry(QtCore.QRect(10, 0, 581, 61))
        self.verticalFrame_4.setFrameShape(QtWidgets.QFrame.Box)
        self.verticalFrame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalFrame_4.setObjectName("verticalFrame_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalFrame_4)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.lbl_seg_menu = QtWidgets.QLabel(self.verticalFrame_4)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_seg_menu.setFont(font)
        self.lbl_seg_menu.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_seg_menu.setObjectName("lbl_seg_menu")
        self.verticalLayout_4.addWidget(self.lbl_seg_menu)
        self.frame_6 = QtWidgets.QFrame(self.page_seg)
        self.frame_6.setGeometry(QtCore.QRect(10, 60, 581, 121))
        self.frame_6.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.lyt_com_5 = QtWidgets.QFormLayout(self.frame_6)
        self.lyt_com_5.setContentsMargins(10, 10, 10, 10)
        self.lyt_com_5.setObjectName("lyt_com_5")
        self.lbl_seg_in = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.lbl_seg_in.setFont(font)
        self.lbl_seg_in.setObjectName("lbl_seg_in")
        self.lyt_com_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lbl_seg_in)
        self.lyt_input_3 = QtWidgets.QHBoxLayout()
        self.lyt_input_3.setObjectName("lyt_input_3")
        self.led_seg_in = QtWidgets.QLineEdit(self.frame_6)
        self.led_seg_in.setEnabled(True)
        self.led_seg_in.setReadOnly(False)
        self.led_seg_in.setObjectName("led_seg_in")
        self.lyt_input_3.addWidget(self.led_seg_in)
        self.btn_seg_in = QtWidgets.QToolButton(self.frame_6)
        self.btn_seg_in.setObjectName("btn_seg_in")
        self.lyt_input_3.addWidget(self.btn_seg_in)
        self.lyt_com_5.setLayout(1, QtWidgets.QFormLayout.FieldRole, self.lyt_input_3)
        self.lbl_seg_out = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.lbl_seg_out.setFont(font)
        self.lbl_seg_out.setObjectName("lbl_seg_out")
        self.lyt_com_5.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.lbl_seg_out)
        self.lyt_output_2 = QtWidgets.QHBoxLayout()
        self.lyt_output_2.setObjectName("lyt_output_2")
        self.led_seg_out = QtWidgets.QLineEdit(self.frame_6)
        self.led_seg_out.setEnabled(True)
        self.led_seg_out.setDragEnabled(False)
        self.led_seg_out.setReadOnly(False)
        self.led_seg_out.setObjectName("led_seg_out")
        self.lyt_output_2.addWidget(self.led_seg_out)
        self.btn_seg_out = QtWidgets.QToolButton(self.frame_6)
        self.btn_seg_out.setObjectName("btn_seg_out")
        self.lyt_output_2.addWidget(self.btn_seg_out)
        self.lyt_com_5.setLayout(2, QtWidgets.QFormLayout.FieldRole, self.lyt_output_2)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.lyt_com_5.setItem(3, QtWidgets.QFormLayout.FieldRole, spacerItem7)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.lyt_com_5.setItem(0, QtWidgets.QFormLayout.FieldRole, spacerItem8)
        self.frame_5 = QtWidgets.QFrame(self.page_seg)
        self.frame_5.setGeometry(QtCore.QRect(10, 180, 581, 191))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.lyt_com_4 = QtWidgets.QFormLayout(self.frame_5)
        self.lyt_com_4.setObjectName("lyt_com_4")
        self.lbl_seg_out_2 = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.lbl_seg_out_2.setFont(font)
        self.lbl_seg_out_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_seg_out_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lbl_seg_out_2.setObjectName("lbl_seg_out_2")
        self.lyt_com_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lbl_seg_out_2)
        self.frame_2 = QtWidgets.QFrame(self.frame_5)
        self.frame_2.setMinimumSize(QtCore.QSize(80, 130))
        self.frame_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_2.setObjectName("frame_2")
        self.lyt_grid_5 = QtWidgets.QGridLayout(self.frame_2)
        self.lyt_grid_5.setObjectName("lyt_grid_5")
        self.cmb_seg_line = QtWidgets.QComboBox(self.frame_2)
        self.cmb_seg_line.setObjectName("cmb_seg_line")
        self.cmb_seg_line.addItem("")
        self.cmb_seg_line.addItem("")
        self.lyt_grid_5.addWidget(self.cmb_seg_line, 4, 1, 1, 1)
        self.lbl_seg_trun = QtWidgets.QLabel(self.frame_2)
        self.lbl_seg_trun.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_seg_trun.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lbl_seg_trun.setObjectName("lbl_seg_trun")
        self.lyt_grid_5.addWidget(self.lbl_seg_trun, 1, 0, 1, 1)
        self.lbl_seg_gaus = QtWidgets.QLabel(self.frame_2)
        self.lbl_seg_gaus.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_seg_gaus.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lbl_seg_gaus.setObjectName("lbl_seg_gaus")
        self.lyt_grid_5.addWidget(self.lbl_seg_gaus, 2, 0, 1, 1)
        self.cmb_seg_trun = QtWidgets.QComboBox(self.frame_2)
        self.cmb_seg_trun.setObjectName("cmb_seg_trun")
        self.cmb_seg_trun.addItem("")
        self.cmb_seg_trun.addItem("")
        self.cmb_seg_trun.addItem("")
        self.cmb_seg_trun.addItem("")
        self.cmb_seg_trun.addItem("")
        self.cmb_seg_trun.addItem("")
        self.lyt_grid_5.addWidget(self.cmb_seg_trun, 1, 1, 1, 1)
        self.lbl_seg_comp = QtWidgets.QLabel(self.frame_2)
        self.lbl_seg_comp.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_seg_comp.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lbl_seg_comp.setLineWidth(1)
        self.lbl_seg_comp.setObjectName("lbl_seg_comp")
        self.lyt_grid_5.addWidget(self.lbl_seg_comp, 3, 0, 1, 1)
        self.lbl_seg_ftp = QtWidgets.QLabel(self.frame_2)
        self.lbl_seg_ftp.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_seg_ftp.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lbl_seg_ftp.setObjectName("lbl_seg_ftp")
        self.lyt_grid_5.addWidget(self.lbl_seg_ftp, 0, 0, 1, 1)
        self.cmb_seg_ftp = QtWidgets.QComboBox(self.frame_2)
        self.cmb_seg_ftp.setObjectName("cmb_seg_ftp")
        self.cmb_seg_ftp.addItem("")
        self.cmb_seg_ftp.addItem("")
        self.cmb_seg_ftp.addItem("")
        self.lyt_grid_5.addWidget(self.cmb_seg_ftp, 0, 1, 1, 1)
        self.cmb_seg_gaus = QtWidgets.QComboBox(self.frame_2)
        self.cmb_seg_gaus.setObjectName("cmb_seg_gaus")
        self.cmb_seg_gaus.addItem("")
        self.cmb_seg_gaus.addItem("")
        self.cmb_seg_gaus.addItem("")
        self.cmb_seg_gaus.addItem("")
        self.cmb_seg_gaus.addItem("")
        self.cmb_seg_gaus.addItem("")
        self.lyt_grid_5.addWidget(self.cmb_seg_gaus, 2, 1, 1, 1)
        self.dbx_seg_comp = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.dbx_seg_comp.setDecimals(1)
        self.dbx_seg_comp.setSingleStep(0.1)
        self.dbx_seg_comp.setProperty("value", 1.3)
        self.dbx_seg_comp.setObjectName("dbx_seg_comp")
        self.lyt_grid_5.addWidget(self.dbx_seg_comp, 3, 1, 1, 1)
        self.lbl_seg_line = QtWidgets.QLabel(self.frame_2)
        self.lbl_seg_line.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_seg_line.setFrameShadow(QtWidgets.QFrame.Raised)
        self.lbl_seg_line.setObjectName("lbl_seg_line")
        self.lyt_grid_5.addWidget(self.lbl_seg_line, 4, 0, 1, 1)
        self.lyt_com_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.frame_2)
        self.horizontalFrame_6 = QtWidgets.QFrame(self.page_seg)
        self.horizontalFrame_6.setGeometry(QtCore.QRect(10, 630, 581, 61))
        self.horizontalFrame_6.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.horizontalFrame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.horizontalFrame_6.setObjectName("horizontalFrame_6")
        self.lyt_com_btn_4 = QtWidgets.QHBoxLayout(self.horizontalFrame_6)
        self.lyt_com_btn_4.setObjectName("lyt_com_btn_4")
        self.btn_seg_rtn = QtWidgets.QPushButton(self.horizontalFrame_6)
        self.btn_seg_rtn.setMinimumSize(QtCore.QSize(0, 0))
        self.btn_seg_rtn.setMaximumSize(QtCore.QSize(130, 45))
        self.btn_seg_rtn.setBaseSize(QtCore.QSize(0, 0))
        self.btn_seg_rtn.setObjectName("btn_seg_rtn")
        self.lyt_com_btn_4.addWidget(self.btn_seg_rtn)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.lyt_com_btn_4.addItem(spacerItem9)
        self.btn_seg_exit = QtWidgets.QPushButton(self.horizontalFrame_6)
        self.btn_seg_exit.setMaximumSize(QtCore.QSize(130, 45))
        self.btn_seg_exit.setMouseTracking(False)
        self.btn_seg_exit.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.btn_seg_exit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_seg_exit.setDefault(True)
        self.btn_seg_exit.setObjectName("btn_seg_exit")
        self.lyt_com_btn_4.addWidget(self.btn_seg_exit)
        self.btn_seg_cal = QtWidgets.QPushButton(self.horizontalFrame_6)
        self.btn_seg_cal.setMaximumSize(QtCore.QSize(130, 45))
        self.btn_seg_cal.setMouseTracking(False)
        self.btn_seg_cal.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.btn_seg_cal.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_seg_cal.setDefault(True)
        self.btn_seg_cal.setObjectName("btn_seg_cal")
        self.lyt_com_btn_4.addWidget(self.btn_seg_cal)
        self.verticalFrame_6 = QtWidgets.QFrame(self.page_seg)
        self.verticalFrame_6.setGeometry(QtCore.QRect(10, 370, 581, 251))
        self.verticalFrame_6.setFrameShape(QtWidgets.QFrame.Box)
        self.verticalFrame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalFrame_6.setLineWidth(-2)
        self.verticalFrame_6.setObjectName("verticalFrame_6")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalFrame_6)
        self.verticalLayout_7.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.txt_seg_cmd = QtWidgets.QPlainTextEdit(self.verticalFrame_6)
        self.txt_seg_cmd.setObjectName("txt_seg_cmd")
        self.verticalLayout_7.addWidget(self.txt_seg_cmd)
        self.stc_wdg.addWidget(self.page_seg)

        self.retranslateUi(dlg_chm)
        self.stc_wdg.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(dlg_chm)

    def retranslateUi(self, dlg_chm):
        _translate = QtCore.QCoreApplication.translate
        dlg_chm.setWindowTitle(_translate("dlg_chm", "Dialog"))
        self.btn_sel_com.setText(_translate("dlg_chm", "Pełna kalkulacja"))
        self.btn_sel_lid.setText(_translate("dlg_chm", "Obliczenie CHM"))
        self.btn_sel_seg.setText(_translate("dlg_chm", "Obliczenie koron (segmentacja)"))
        self.btn_com_rtn.setText(_translate("dlg_chm", "Wróć"))
        self.btn_com_exit.setText(_translate("dlg_chm", "Zamknij"))
        self.btn_com_cal.setText(_translate("dlg_chm", "Uruchom"))
        self.lbl_com_menu.setText(_translate("dlg_chm", "PEŁNA KALKULACJA"))
        self.lbl_com_in.setText(_translate("dlg_chm", "Dane wejściowe"))
        self.btn_com_in.setText(_translate("dlg_chm", "..."))
        self.lbl_com_out.setText(_translate("dlg_chm", "Katalog wyjściowy"))
        self.btn_com_out.setText(_translate("dlg_chm", "..."))
        self.lbl_seg_out_3.setText(_translate("dlg_chm", "USTAWIENIA\n"
"SEGMENTACJI   "))
        self.cmb_com_ftp.setCurrentText(_translate("dlg_chm", "3:3"))
        self.cmb_com_ftp.setItemText(0, _translate("dlg_chm", "3:3"))
        self.cmb_com_ftp.setItemText(1, _translate("dlg_chm", "5:5"))
        self.cmb_com_ftp.setItemText(2, _translate("dlg_chm", "7:7"))
        self.lbl_com_ftp.setText(_translate("dlg_chm", "max picks footprint"))
        self.lbl_com_gaus.setText(_translate("dlg_chm", "gaussian filter offset"))
        self.cmb_com_line.setItemText(0, _translate("dlg_chm", "False"))
        self.cmb_com_line.setItemText(1, _translate("dlg_chm", "True"))
        self.cmb_com_gaus.setCurrentText(_translate("dlg_chm", "0"))
        self.cmb_com_gaus.setItemText(0, _translate("dlg_chm", "0"))
        self.cmb_com_gaus.setItemText(1, _translate("dlg_chm", "1"))
        self.cmb_com_gaus.setItemText(2, _translate("dlg_chm", "2"))
        self.cmb_com_gaus.setItemText(3, _translate("dlg_chm", "3"))
        self.cmb_com_gaus.setItemText(4, _translate("dlg_chm", "4"))
        self.cmb_com_gaus.setItemText(5, _translate("dlg_chm", "5"))
        self.lbl_com_comp.setText(_translate("dlg_chm", "watershed compactness"))
        self.lbl_com_line.setText(_translate("dlg_chm", "watershed line"))
        self.lbl_com_trun.setText(_translate("dlg_chm", "truncate"))
        self.cmb_com_trun.setItemText(0, _translate("dlg_chm", "0"))
        self.cmb_com_trun.setItemText(1, _translate("dlg_chm", "1"))
        self.cmb_com_trun.setItemText(2, _translate("dlg_chm", "2"))
        self.cmb_com_trun.setItemText(3, _translate("dlg_chm", "3"))
        self.cmb_com_trun.setItemText(4, _translate("dlg_chm", "4"))
        self.cmb_com_trun.setItemText(5, _translate("dlg_chm", "5"))
        self.lbl_lid_menu.setText(_translate("dlg_chm", "OBLICZENIE CHM"))
        self.btn_lid_rtn.setText(_translate("dlg_chm", "Wróć"))
        self.btn_lid_exit.setText(_translate("dlg_chm", "Zamknij"))
        self.btn_lid_cal.setText(_translate("dlg_chm", "Uruchom"))
        self.lbl_lid_in.setText(_translate("dlg_chm", "Dane wejściowe"))
        self.btn_lid_in.setText(_translate("dlg_chm", "..."))
        self.lbl_lid_out.setText(_translate("dlg_chm", "Katalog wyjściowy"))
        self.btn_lid_out.setText(_translate("dlg_chm", "..."))
        self.lbl_seg_menu.setText(_translate("dlg_chm", "SEGMENTACJA"))
        self.lbl_seg_in.setText(_translate("dlg_chm", "Dane wejściowe"))
        self.btn_seg_in.setText(_translate("dlg_chm", "..."))
        self.lbl_seg_out.setText(_translate("dlg_chm", "Katalog wyjściowy"))
        self.btn_seg_out.setText(_translate("dlg_chm", "..."))
        self.lbl_seg_out_2.setText(_translate("dlg_chm", "USTAWIENIA\n"
"SEGMENTACJI   "))
        self.cmb_seg_line.setItemText(0, _translate("dlg_chm", "False"))
        self.cmb_seg_line.setItemText(1, _translate("dlg_chm", "True"))
        self.lbl_seg_trun.setText(_translate("dlg_chm", "truncate"))
        self.lbl_seg_gaus.setText(_translate("dlg_chm", "gaussian filter offset"))
        self.cmb_seg_trun.setItemText(0, _translate("dlg_chm", "0"))
        self.cmb_seg_trun.setItemText(1, _translate("dlg_chm", "1"))
        self.cmb_seg_trun.setItemText(2, _translate("dlg_chm", "2"))
        self.cmb_seg_trun.setItemText(3, _translate("dlg_chm", "3"))
        self.cmb_seg_trun.setItemText(4, _translate("dlg_chm", "4"))
        self.cmb_seg_trun.setItemText(5, _translate("dlg_chm", "5"))
        self.lbl_seg_comp.setText(_translate("dlg_chm", "watershed compactness"))
        self.lbl_seg_ftp.setText(_translate("dlg_chm", "max picks footprint"))
        self.cmb_seg_ftp.setCurrentText(_translate("dlg_chm", "3:3"))
        self.cmb_seg_ftp.setItemText(0, _translate("dlg_chm", "3:3"))
        self.cmb_seg_ftp.setItemText(1, _translate("dlg_chm", "5:5"))
        self.cmb_seg_ftp.setItemText(2, _translate("dlg_chm", "7:7"))
        self.cmb_seg_gaus.setCurrentText(_translate("dlg_chm", "0"))
        self.cmb_seg_gaus.setItemText(0, _translate("dlg_chm", "0"))
        self.cmb_seg_gaus.setItemText(1, _translate("dlg_chm", "1"))
        self.cmb_seg_gaus.setItemText(2, _translate("dlg_chm", "2"))
        self.cmb_seg_gaus.setItemText(3, _translate("dlg_chm", "3"))
        self.cmb_seg_gaus.setItemText(4, _translate("dlg_chm", "4"))
        self.cmb_seg_gaus.setItemText(5, _translate("dlg_chm", "5"))
        self.lbl_seg_line.setText(_translate("dlg_chm", "watershed line"))
        self.btn_seg_rtn.setText(_translate("dlg_chm", "Wróć"))
        self.btn_seg_exit.setText(_translate("dlg_chm", "Zamknij"))
        self.btn_seg_cal.setText(_translate("dlg_chm", "Uruchom"))
