from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import numpy as np
import gdal
import ogr
import osr
import pdal
import json
import sys
import os
import time
import csv
from PyQt5.QtWidgets import *
from ui_modules.chm_dlg_ui import *


class DlgMain(QDialog, Ui_dlg_chm):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setWindowTitle("Kalkulator CHM")

        self.btn_calculate.setDefault(True)
        self.txt_command.setReadOnly(True)
        self.led_input.returnPressed.connect(self.evt_led_input_return)
        self.led_output.returnPressed.connect(self.evt_led_out_return)

        self.btn_input.clicked.connect(self.evt_btn_input_clicked)
        self.btn_output.clicked.connect(self.evt_btn_output_clicked)
        self.btn_calculate.clicked.connect(self.evt_btn_calculate_clicked)

    def evt_led_input_return(self):
        fn = self.led_input.text()
        if not os.path.exists(fn):
            QMessageBox.warning(self, "Brak pliku", "Wskazany plik nie istnieje")
            self.led_input.clear()

    def evt_led_out_return(self):
        fc = self.led_output.text()
        if not os.path.isdir(fc):
            QMessageBox.warning(self, "Brak katalogu", "Wskazany katalog nie istnieje")
            self.led_output.clear()

    def evt_btn_input_clicked(self):
        fn, b_ok = QFileDialog.getOpenFileName(self, "Select file", "C:\\", "Lidar Files (*.las)")
        if b_ok:
            self.led_input.setText(fn)

    def evt_btn_output_clicked(self):

        fc = QFileDialog.getExistingDirectory(self, "Output directory", "C:\\")
        if fc:
            self.led_output.setText(fc)

    def evt_btn_calculate_clicked(self):
        self.txt_command.clear()
        self.txt_command.setPlainText("Obliczenia w trakcie, proces może zająć kilka minut...")
        fn = self.led_input.text()
        fc = self.led_output.text()

        if not os.path.exists(fn) or not os.path.isdir(fc):
            QMessageBox.critical(self, "Błąd ścieżki", "Brak pliku lub niepoprawna ścieżka do wyjściowego katalogu")
            self.txt_command.clear()
        else:
            self.txt_command.setPlainText("Obliczenia w trakcie, proces może zająć kilka minut...")
            script_start = time.time()

            # CREATE OUTPUT FILENAMES
            dtm_fn = create_files_names(fc, fn, "DTM","tif")
            dsm_fn = create_files_names(fc, fn, "DSM", "tif")
            chm_fn = create_files_names(fc, fn, "CHM", "tif")
            seg_tif_fn = create_files_names(fc, fn, "SEGMENTS", "tif")
            segments_fn = create_files_names(fc, fn, "SEGMENTS", "shp")
            csv_fn = create_files_names(fc, fn, "ZSTATS", "csv")
            print(dtm_fn, os.path.exists(dtm_fn))
            print(dsm_fn, os.path.exists(dsm_fn))
            print(chm_fn, os.path.exists(chm_fn))
            print(seg_tif_fn, os.path.exists(seg_tif_fn))
            print(segments_fn, os.path.exists(segments_fn))
            print(csv_fn, os.path.exists(csv_fn))



            # EXTRACT DTM AND DSM FROM LIDAR DATA
            dem_extract(fn, fc, "min")
            dem_extract(fn, fc, "max")

            # CALCULATE CHM
            chm_calculate(dtm_fn, dsm_fn, chm_fn)

            # CREATE SEGMENTS OF CHM DATA
            segments = chm_segmentation(dsm_fn)

            # SAVE SEGMENTS TO NEW RASTER
            save_segments_to_raster(seg_tif_fn, segments, dsm_fn)

            # CREATE SHAPEFILE SEGMENTS, CREATE STATISTIC CSV
            create_segments_shp(seg_tif_fn, segments_fn, chm_fn, csv_fn)

            # CREATE MESSAGE TO PRINT
            files_list = [dtm_fn, dsm_fn, chm_fn, seg_tif_fn, segments_fn, csv_fn]

            msg = ""
            for file in files_list:
                msg += "{}\n".format(os.path.basename(file))

            script_end = time.time()
            # self.txt_command.appendPlainText("Zakończono powodzeniem w {:.2f} sec.\n "
            #                                  "W {} Wygenerowano pliki:\n{}".format(fc, script_end - script_start, msg))



def create_files_names(fc, fn, name, format_type):
    out_fn = fc + r"\{}_{}.{}".format(name, os.path.basename(fn)[:-4], format_type)
    return out_fn


def save_segments_to_raster(seg_tif_fn, segments_array, dsm_fn):
    driver = gdal.GetDriverByName("GTiff")
    dsm_ds = gdal.Open(dsm_fn)
    tif_ds = driver.Create(seg_tif_fn,
                           dsm_ds.RasterXSize,
                           dsm_ds.RasterYSize,
                           1,
                           gdal.GDT_Float32)
    tif_ds.SetGeoTransform(dsm_ds.GetGeoTransform())
    tif_ds.SetProjection(dsm_ds.GetProjection())
    tif_ds.GetRasterBand(1).WriteArray(segments_array)

            # SAVE SEGMENTS TO NEW RASTER


def create_segments_shp(seg_tif_fn, segments_fn, chm_fn, csv_fn):
    # CREATE TEMPORARY.SHP
    # (due to to problems with deleting features with ogr via DeleteFeature)
    tif_ds = gdal.Open(seg_tif_fn)

    driver_tmp = ogr.GetDriverByName("Memory")
    tmp_ds = driver_tmp.CreateDataSource("temp")
    tmp_lyr = create_layer(tmp_ds, epsg=2178)

    # POLYGONIZE SEGMENTS TO TEMPORARY.SHP
    gdal.Polygonize(tif_ds.GetRasterBand(1),
                    None,
                    tmp_lyr,
                    0,
                    ['SEG_NB'],
                    callback=None)

    # CREATE SEGMENTS.SHP
    driver_shp = ogr.GetDriverByName("ESRI Shapefile")
    segments_ds = driver_shp.CreateDataSource(segments_fn)
    segments_lyr = create_layer(segments_ds, epsg=2178)

    # COPY SELECTED FEATURES TO SEGMENTS.SHP
    # (due to to problems with deleting features with ogr via DeleteFeature)
    copy_attributes(tmp_lyr, segments_lyr)
    segments_ds = None

    # CREATE MEMORY FILE TO STORE DATA
    driver_mem_gdal = gdal.GetDriverByName("MEM")
    driver_tmp = ogr.GetDriverByName("Memory")
    shp_name = "temp"

    # OPEN DS, READ GEOTRANSFORM
    segments_ds = ogr.Open(segments_fn, 1)
    lyr = segments_ds.GetLayer()
    chm_ds = gdal.Open(chm_fn)
    transform = chm_ds.GetGeoTransform()
    nodata = chm_ds.GetRasterBand(1).GetNoDataValue()
    zstats = []

    feat = lyr.GetNextFeature()

    while feat:
        if feat.GetGeometryRef() is not None:
            if os.path.exists(shp_name):
                driver_tmp.DeleteDataSource(shp_name)
            tmp_seg_ds = driver_tmp.CreateDataSource(shp_name)
            tmp_seg_lyr = tmp_seg_ds.CreateLayer("polygon", None, ogr.wkbPolygon)
            tmp_seg_lyr.CreateFeature(feat.Clone())

            offsets = bounding_box_to_offsets(feat.GetGeometryRef().GetEnvelope(), transform)
            new_transform = offsets_transform(offsets[0], offsets[2], transform)

            # CREATE NEW TEMPORARY RASTER WITH RASTERIZED LAYER
            mem_ds = driver_mem_gdal.Create("",
                                            offsets[3] - offsets[2],
                                            offsets[1] - offsets[0],
                                            1,
                                            gdal.GDT_Byte)
            mem_ds.SetGeoTransform(new_transform)

            gdal.RasterizeLayer(mem_ds, [1], tmp_seg_lyr, burn_values=[1])
            mem_array = mem_ds.ReadAsArray()
            chm_array = chm_ds.GetRasterBand(1).ReadAsArray(offsets[2],
                                                            offsets[0],
                                                            offsets[3] - offsets[2],
                                                            offsets[1] - offsets[0])

            obj_id = feat.GetField("SEG_NB")
            if chm_array is not None:
                fix_array = np.ma.fix_invalid(chm_array)
                mask_array = np.ma.MaskedArray(fix_array, mask=np.logical_not(mem_array))

                if mask_array is not None:
                    max_h = mask_array.max().item()
                    feat.SetField("MAX_H", round(max_h, 2))
                    lyr.SetFeature(feat)
                    zstats.append(calculate_zstats(obj_id,
                                                   mask_array.min(),
                                                   mask_array.max(),
                                                   mask_array.mean(),
                                                   np.ma.median(mask_array),
                                                   mask_array.std(),
                                                   mask_array.sum(),
                                                   mask_array.count()))
                else:
                    feat.SetField("MAX_H", nodata)
                    lyr.SetFeature(feat)
                    zstats.append(calculate_zstats(obj_id,
                                                   nodata,
                                                   nodata,
                                                   nodata,
                                                   nodata,
                                                   nodata,
                                                   nodata,
                                                   nodata))
            else:
                feat.SetField("MAX_H", nodata)
                lyr.SetFeature(feat)
                zstats.append(calculate_zstats(obj_id,
                                               nodata,
                                               nodata,
                                               nodata,
                                               nodata,
                                               nodata,
                                               nodata,
                                               nodata))

            mem_ds = None
            feat = lyr.GetNextFeature()

    col_names = zstats[0].keys()
    with open(csv_fn, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, col_names)
        writer.writeheader()
        writer.writerows(zstats)


    chm_ds = None
    tif_ds = None
    tmp_ds = None
    segments_ds = None


def dem_extract(lidar_fn, out_fn, stat, in_srs="EPSG:2180", out_srs="EPSG:2178"):
    """Creating DSM and DTM (both trees only) from lidar data"""
    start = time.time()
    elevation = "DTM" if stat == "min" else "DSM"
    print("{} extracting...".format(elevation))

    pdal_json = {
        "pipeline": [
            "{}".format(lidar_fn),
            {
                "type": "filters.reprojection",
                "in_srs": "{}".format(in_srs),
                "out_srs": "{}".format(out_srs)
            },
            {
                "type": "filters.hag_nn",
            },
            {
                "type": "filters.range",
                "limits": "Classification[5:5]",
            },
            {
                "filename": "{}\{}_{}.tif".format(out_fn, elevation, os.path.basename(lidar_fn)[:-4]),
                "gdaldriver": "GTiff",
                "output_type": "{}".format(stat),
                "resolution": "1",
                "type": "writers.gdal"
            }
        ]
    }

    pdal_json_str = json.dumps(pdal_json)
    pipeline = pdal.Pipeline(pdal_json_str)
    pipeline.execute()
    end = time.time()

    print("{} extracted successfully in {:.2f} sec\n".format(elevation, end - start))


def chm_calculate(dtm_fn, dsm_fn, chm_fn):
    """Calculating CHM from DSM and DTM"""
    start = time.time()
    print("CHM calculating...")



    # LOADING DRIVER
    driver_tiff = gdal.GetDriverByName("GTiff")

    # OPEN DATASET & READ DATA
    dtm_ds = gdal.Open(dtm_fn)
    dtm_data = dtm_ds.GetRasterBand(1).ReadAsArray()
    dsm_ds = gdal.Open(dsm_fn)
    dsm_data = dsm_ds.GetRasterBand(1).ReadAsArray()

    # CALCULATE CHM
    chm_data = dsm_data - dtm_data
    chm_data[chm_data == 0] = np.NaN

    # CREATE FILTERED RASTER AND SAVE DATA
    chm_ds = driver_tiff.Create(chm_fn,
                                dtm_ds.RasterXSize,
                                dtm_ds.RasterYSize,
                                1,
                                gdal.GDT_Float32)

    chm_ds.SetProjection(dtm_ds.GetProjection())
    chm_ds.SetGeoTransform(dtm_ds.GetGeoTransform())
    chm_ds.GetRasterBand(1).WriteArray(chm_data)

    # CLOSING DATASETS
    dtm_ds = None
    dsm_ds = None
    chm_ds = None
    end = time.time()
    print("CHM calculated successfully in {:.4f} sec\n".format(end - start))


def chm_segmentation(dsm_fn):
    """CHM ata filtering, masking and watershed segmentation.
    Idea from https://www.neonscience.org/resources/learning-hub/tutorials/calc-biomass-py"""

    driver = gdal.GetDriverByName("GTiff")
    dsm_ds = gdal.Open(dsm_fn)
    dsm_ds.GetRasterBand(1).SetNoDataValue(0)
    chm_array = dsm_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    start = time.time()
    print("Watershed segmentation...")
    # APPLYING GAUSSIAN FILTER TO REMOVE WRONG POINTS
    chm_array_smooth = ndi.gaussian_filter(chm_array, 1,
                                           mode='constant',
                                           cval=0,
                                           truncate=1)
    chm_array_smooth[chm_array == 0] = 0

    # CALCULATE LOCAL MAXIMUM POINTS
    max_coords = peak_local_max(chm_array_smooth,
                                footprint=np.ones((5, 5)))

    # CREATE MASK TO MATCH INPUT ARRAY SIZE
    markers = np.zeros_like(chm_array_smooth, dtype=bool)
    markers[tuple(max_coords.T)] = True

    # IDENTIFY ALL THE MAXIMUM POINTS
    markers = ndi.label(markers)[0]

    # CREATE MASK TO REDUCE SEGMENTATION ONLY TO TREES
    chm_mask = chm_array_smooth.copy()
    chm_mask[chm_array_smooth > 0] = 1
    chm_mask[chm_array_smooth <= 0] = 0

    # WATERSHED SEGMENTATION
    labels = watershed(chm_array_smooth,
                       markers,
                       compactness=1.3,
                       mask=chm_mask,
                       watershed_line=False)

    # CHANGING 0 VALUES TO -9999
    segments = np.where(labels == 0, -9999, labels)

    dsm_ds = None
    end = time.time()
    print("CHM segmented in {:.4f} sec\n".format(end - start))

    return segments


def create_layer(ds, epsg=2178):
    """Create shapefile layer. Used for temporary and segments shp"""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    layer = ds.CreateLayer('segments.shp', srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn('SEG_NB', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('MAX_H', ogr.OFTReal))
    return layer


def copy_attributes(tmp_lyr, segments_lyr):
    """Copying attributes form temporary shp layer to segments"""
    for feat in tmp_lyr:
        if feat.GetField(0) != -9999:
            geom = feat.GetGeometryRef()
            fld_name = feat.GetFieldDefnRef(0).GetName()

            out_feat = ogr.Feature(segments_lyr.GetLayerDefn())
            out_feat.SetGeometry(geom)
            out_feat.SetField(fld_name, feat.GetField(0))
            segments_lyr.CreateFeature(out_feat)


def bounding_box_to_offsets(bbox, transform):
    """Calculating boxboundary offsets for each segment"""
    col1 = int((bbox[0] - transform[0]) / transform[1])
    col2 = int((bbox[1] - transform[0]) / transform[1]) + 1
    row1 = int((bbox[3] - transform[3]) / transform[5])
    row2 = int((bbox[2] - transform[3]) / transform[5]) + 1
    return [row1, row2, col1, col2]


def offsets_transform(row_offset, col_offset, transform):
    """Calculating new geotransform for each segment boxboundary"""
    new_geotransform = [
        transform[0] + (col_offset * transform[1]),
        transform[1],
        0.0,
        transform[3] + (row_offset * transform[5]),
        0.0,
        transform[5]]
    return new_geotransform


def calculate_zstats(fid, min, max, mean, median, sd, sum, count):
    """Calculating basic statistic, determining maximum height in segment"""
    names = ["id", "min", "max", "mean", "median", "sd", "sum", "count"]
    feat_stats = {names[0]: fid,
                  names[1]: min,
                  names[2]: max,
                  names[3]: mean,
                  names[4]: median,
                  names[5]: sd,
                  names[6]: sum,
                  names[7]: count
                  }
    return feat_stats


if '__main__' == __name__:
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec_())
