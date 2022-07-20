import sys
import os
import time
import chm_calc
from PyQt5.QtWidgets import *
from ui_modules.chm_dlg_ui import *


class DlgMain(QDialog, Ui_dlg_chm):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setWindowTitle("Kalkulator CHM")

        self.btn_sel_com.clicked.connect(self.evt_btn_sel_com_clicked)
        self.btn_sel_lid.clicked.connect(self.evt_btn_sel_lid_clicked)
        self.btn_sel_seg.clicked.connect(self.evt_btn_sel_seg_clicked)

        self.btn_com_rtn.clicked.connect(self.evt_btn_rtn_clicked)
        self.btn_lid_rtn.clicked.connect(self.evt_btn_rtn_clicked)
        self.btn_seg_rtn.clicked.connect(self.evt_btn_rtn_clicked)

        self.btn_com_exit.clicked.connect(self.close)
        self.btn_lid_exit.clicked.connect(self.close)
        self.btn_seg_exit.clicked.connect(self.close)

        self.txt_com_cmd.setReadOnly(True)
        self.txt_lid_cmd.setReadOnly(True)
        self.txt_seg_cmd.setReadOnly(True)

        self.cmb_com_ftp.setCurrentIndex(1)
        self.cmb_seg_ftp.setCurrentIndex(1)

        self.btn_com_in.clicked.connect(self.evt_btn_com_in_clicked)
        self.btn_com_out.clicked.connect(self.evt_btn_com_out_clicked)
        self.btn_com_cal.clicked.connect(self.evt_btn_com_cal_clicked)

        self.btn_lid_in.clicked.connect(self.evt_btn_lid_in_clicked)
        self.btn_lid_out.clicked.connect(self.evt_btn_lid_out_clicked)
        self.btn_lid_cal.clicked.connect(self.evt_btn_lid_cal_clicked)

        self.btn_seg_in.clicked.connect(self.evt_btn_seg_in_clicked)
        self.btn_seg_out.clicked.connect(self.evt_btn_seg_out_clicked)
        self.btn_seg_cal.clicked.connect(self.evt_btn_seg_cal_clicked)

        self.timer_com = QtCore.QTimer()
        self.timer_com.setSingleShot(True)
        self.timer_com.setInterval(100)
        self.timer_com.timeout.connect(self.evt_timer_com_timeout)

        self.timer_lid = QtCore.QTimer()
        self.timer_lid.setSingleShot(True)
        self.timer_lid.setInterval(100)
        self.timer_lid.timeout.connect(self.evt_timer_lid_timeout)

        self.timer_seg = QtCore.QTimer()
        self.timer_seg.setSingleShot(True)
        self.timer_seg.setInterval(100)
        self.timer_seg.timeout.connect(self.evt_timer_seg_timeout)

    def evt_led_input_return(self):
        fn = self.led_input.text()
        if not os.path.exists(fn):
            QMessageBox.warning(self, "", "Brak pliku", "Wskazany plik nie istnieje")
            self.led_input.clear()

    def evt_led_out_return(self):
        fc = self.led_output.text()
        if not os.path.isdir(fc):
            QMessageBox.warning(self, "", "Brak katalogu", "Wskazany katalog nie istnieje")
            self.led_output.clear()

    def evt_btn_sel_com_clicked(self):
        self.stc_wdg.setCurrentIndex(1)

    def evt_btn_sel_lid_clicked(self):
        self.stc_wdg.setCurrentIndex(2)

    def evt_btn_sel_seg_clicked(self):
        self.stc_wdg.setCurrentIndex(3)

    def evt_btn_rtn_clicked(self):
        self.stc_wdg.setCurrentIndex(0)

    # COMPLETE CALCULATION
    def evt_btn_com_in_clicked(self):
        # fn, b_ok = QFileDialog.getOpenFileName(self, "Select file", "C:\\", "Lidar Files (*.las)")
        fn, b_ok = QFileDialog.getOpenFileName(self, "Select file", "C:\\Users\\lukas\\Desktop\\"
                                                                    "PDAL  CHM", "Lidar files (*.las, *.laz)")
        if b_ok:
            self.led_com_in.setText(fn)

    def evt_btn_com_out_clicked(self):
        # fc = QFileDialog.getExistingDirectory(self, "Output directory", "C:\\")
        fc = QFileDialog.getExistingDirectory(self, "Output directory", "C:\\Users\\lukas\\Desktop\\PDAL  CHM")
        if fc:
            self.led_com_out.setText(fc)

    def evt_btn_com_cal_clicked(self):
        self.txt_com_cmd.clear()
        self.txt_com_cmd.setPlainText("Obliczenia w trakcie, proces może zająć kilka minut...")
        self.timer_com.start()

    def evt_timer_com_timeout(self):
        script_start = time.time()

        fn = self.led_com_in.text()
        fc = self.led_com_out.text()

        if not os.path.exists(fn) or not os.path.isdir(fc):
            QMessageBox.critical(dlgMain, "Błąd ścieżki", "Brak pliku lub niepoprawna ścieżka do wyjściowego katalogu")
            dlgMain.txt_com_cmd.clear()
        else:
            dlgMain.txt_com_cmd.setPlainText("Obliczenia w trakcie, proces może zająć kilka minut...")

            # CREATE OUTPUT FILENAMES
            dtm_fn = chm_calc.create_files_names(fc, fn, "DTM", "tif")
            tmp_dtm_fn = chm_calc.create_files_names(fc, fn, "TMP_DTM", "tif")
            dsm_fn = chm_calc.create_files_names(fc, fn, "DSM", "tif")
            chm_fn = chm_calc.create_files_names(fc, fn, "CHM", "tif")
            seg_tif_fn = chm_calc.create_files_names(fc, fn, "SEGMENTS", "tif")
            segments_fn = chm_calc.create_files_names(fc, fn, "SEGMENTS", "shp")
            csv_fn = chm_calc.create_files_names(fc, fn, "ZSTATS", "csv")

            # EXTRACT DTM AND DSM FROM LIDAR DATA
            chm_calc.extract_dsm(fn, fc, "max")
            chm_calc.extract_dtm(fn, fc)

            # CALCULATE DTM WITH FILLED NO DATA
            chm_calc.match_size_dtm(dsm_fn, tmp_dtm_fn, dtm_fn)

            # CALCULATE CHM
            chm_calc.chm_calculate(tmp_dtm_fn, dtm_fn, dsm_fn, chm_fn)

            # CREATE SEGMENTS OF CHM DATA
            footprint = self.cmb_com_ftp.currentText()
            footprint = int(footprint[0])
            f_offset = int(self.cmb_com_gaus.currentText())
            truncate = int(self.cmb_com_trun.currentText())
            compactness = self.dbx_com_comp.value()
            line = self.cmb_seg_line.currentText()

            segments = chm_calc.chm_segmentation(chm_fn, footprint, f_offset, truncate, compactness, line) # dsm_fn-> chm_fn

            # SAVE SEGMENTS TO NEW RASTER
            chm_calc.save_segments_to_raster(seg_tif_fn, segments, dsm_fn)

            # CREATE SHAPEFILE SEGMENTS, CREATE STATISTIC CSV
            chm_calc.create_segments_shp(seg_tif_fn, segments_fn, chm_fn, csv_fn)

            # CREATE MESSAGE TO PRINT
            files_list = [dtm_fn, tmp_dtm_fn, dsm_fn, chm_fn, seg_tif_fn, segments_fn, csv_fn]
            msg = ""
            for file in files_list:
                msg += "{}\n".format(file)

        script_end = time.time()
        self.txt_com_cmd.appendPlainText("\nWygenerowane pliki:\n{}\nZakończono powodzeniem "
                                         "w {:.2f} sec".format(msg, script_end - script_start))

    # CHM CALCULATION
    def evt_btn_lid_in_clicked(self):
        # fn, b_ok = QFileDialog.getOpenFileName(self, "Select file", "C:\\", "Lidar Files (*.las)")
        fn, b_ok = QFileDialog.getOpenFileName(self, "Select file", "C:\\Users\\lukas\\Desktop\\"
                                                                    "PDAL  CHM", "Lidar files (*.las, *.laz)")
        if b_ok:
            self.led_lid_in.setText(fn)

    def evt_btn_lid_out_clicked(self):
        # fc = QFileDialog.getExistingDirectory(self, "Output directory", "C:\\")
        fc = QFileDialog.getExistingDirectory(self, "Output directory", "C:\\Users\\lukas\\Desktop\\PDAL  CHM")
        if fc:
            self.led_lid_out.setText(fc)

    def evt_btn_lid_cal_clicked(self):
        self.txt_lid_cmd.clear()
        self.txt_lid_cmd.setPlainText("Obliczenia w trakcie, proces może zająć kilka minut...")
        self.timer_lid.start()

    def evt_timer_lid_timeout(self):
        script_start = time.time()

        fn = self.led_lid_in.text()
        fc = self.led_lid_out.text()

        if not os.path.exists(fn) or not os.path.isdir(fc):
            QMessageBox.critical(dlgMain, "Błąd ścieżki", "Brak pliku lub niepoprawna ścieżka do wyjściowego katalogu")
            self.txt_lid_cmd.clear()
        else:
            self.txt_lid_cmd.setPlainText("Obliczenia w trakcie, proces może zająć kilka minut...")

            dtm_fn = chm_calc.create_files_names(fc, fn, "DTM", "tif")
            tmp_dtm_fn = chm_calc.create_files_names(fc, fn, "TMP_DTM", "tif")
            dsm_fn = chm_calc.create_files_names(fc, fn, "DSM", "tif")
            chm_fn = chm_calc.create_files_names(fc, fn, "CHM", "tif")

            # EXTRACT DTM AND DSM FROM LIDAR DATA
            chm_calc.extract_dsm(fn, fc, "max")
            chm_calc.extract_dtm(fn, fc)

            # CALCULATE DTM WITH FILLED NO DATA
            chm_calc.match_size_dtm(dsm_fn, tmp_dtm_fn, dtm_fn)

            # CALCULATE CHM
            chm_calc.chm_calculate(tmp_dtm_fn, dtm_fn, dsm_fn, chm_fn)

            # CREATE MESSAGE TO PRINT
            files_list = [dtm_fn, tmp_dtm_fn, dsm_fn, chm_fn]
            msg = ""
            for file in files_list:
                # msg += "\t{}\n".format(os.path.basename(file))
                msg += "{}\n".format(file)

        script_end = time.time()
        self.txt_lid_cmd.appendPlainText("\nWygenerowane pliki:\n{}\nZakończono powodzeniem "
                                         "w {:.2f} sec".format(msg, script_end - script_start))

    # SEGMENTATION CALCULATION
    def evt_btn_seg_in_clicked(self):
        # fn, b_ok = QFileDialog.getOpenFileName(self, "Select file", "C:\\", "Lidar Files (*.las)")
        fn, b_ok = QFileDialog.getOpenFileName(self, "Select file", "C:\\Users\\lukas\\Desktop\\"
                                                                    "PDAL  CHM", "Image files (*.tif)")
        if b_ok:
            self.led_seg_in.setText(fn)

    def evt_btn_seg_out_clicked(self):
        # fc = QFileDialog.getExistingDirectory(self, "Output directory", "C:\\")
        fc = QFileDialog.getSaveFileName(self, "Output file", "C:\\Users\\lukas\\Desktop\\PDAL  CHM",
                                         "Shape Files (*.shp)")
        if fc:
            self.led_seg_out.setText(fc[0])

    def evt_btn_seg_cal_clicked(self):
        self.txt_seg_cmd.clear()
        self.txt_seg_cmd.setPlainText("Obliczenia w trakcie, proces może zająć kilka minut...")
        self.timer_seg.start()

    def evt_timer_seg_timeout(self):
        script_start = time.time()
        in_fn = self.led_seg_in.text()
        out_fn = self.led_seg_out.text()
        fc = os.path.dirname(self.led_seg_out.text())

        if not os.path.isdir(fc):
            QMessageBox.critical(dlgMain, "Błąd ścieżki", "Brak pliku lub niepoprawna ścieżka do wyjściowego katalogu")
            self.txt_seg_cmd.clear()
        else:
            self.txt_seg_cmd.setPlainText("Obliczenia w trakcie, proces może zająć kilka minut...")

            # READ EXISTING FILENAMES
            chm_fn = in_fn
            dsm_fn = fc + r"\DSM{}.tif".format(os.path.basename(in_fn)[3:-4])

            # CREATE OUTPUT FILENAMES
            seg_tif_fn = fc + r"\{}.{}".format(os.path.basename(out_fn)[:-4], "tif")
            segments_fn = fc + r"\{}.{}".format(os.path.basename(out_fn)[:-4], "shp")
            csv_fn = fc + r"\{}.{}".format(os.path.basename(out_fn)[:-4], "csv")

            # CREATE SEGMENTS OF CHM DATA
            footprint = self.cmb_seg_ftp.currentText()
            footprint = int(footprint[0])
            f_offset = int(self.cmb_seg_gaus.currentText())
            truncate = int(self.cmb_seg_trun.currentText())
            compactness = self.dbx_seg_comp.value()
            line = self.cmb_seg_line.currentText()

            segments = chm_calc.chm_segmentation(chm_fn, footprint, f_offset, truncate, compactness, line) # dsm

            # SAVE SEGMENTS TO NEW RASTER
            chm_calc.save_segments_to_raster(seg_tif_fn, segments, chm_fn) # dms

            # CREATE SHAPEFILE SEGMENTS, CREATE STATISTIC CSV
            chm_calc.create_segments_shp(seg_tif_fn, segments_fn, chm_fn, csv_fn)

            # CREATE MESSAGE TO PRINT
            files_list = [seg_tif_fn, segments_fn, csv_fn]
            msg = ""
            for file in files_list:
                msg += "{}\n".format(file)

        script_end = time.time()
        self.txt_seg_cmd.appendPlainText("\nWygenerowane pliki:\n{}\nZakończono powodzeniem "
                                         "w {:.2f} sec".format(msg, script_end - script_start))


if "__main__" == __name__:
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec_())
