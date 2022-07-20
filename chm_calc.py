"""IDEA TAKEN FROM:
ZONAL STATISTICS:   https://www.youtube.com/watch?v=bPHHblgjm40&list=PLzHdTn7Pdxs4oGHlbrACNGrqxzTT04VYx
WATERSHED:          https://www.neonscience.org/resources/learning-hub/tutorials/calc-biomass-py
"""

# GDAL: https://stackoverflow.com/questions/56764046/gdal-ogr2ogr-cannot-find-proj-db-error
import os
os.environ['PROJ_LIB'] = r'C:\Users\lukas\miniconda3\envs\chm_calc\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\lukas\miniconda3\envs\chm_calc\Library\share'
from osgeo import gdal, ogr, osr

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import numpy as np

import pdal
import json
import os
import time
import csv


def summary_calculation(fn, fc):
    # CREATE OUTPUT FILENAMES
    dtm_fn = create_files_names(fc, fn, "DTM", "tif")
    tmp_dtm_fn = create_files_names(fc, fn, "TMP_DTM", "tif")
    dsm_fn = create_files_names(fc, fn, "DSM", "tif")
    chm_fn = create_files_names(fc, fn, "CHM", "tif")
    seg_tif_fn = create_files_names(fc, fn, "SEGMENTS", "tif")
    segments_fn = create_files_names(fc, fn, "SEGMENTS", "shp")
    csv_fn = create_files_names(fc, fn, "ZSTATS", "csv")

    # EXTRACT DTM AND DSM FROM LIDAR DATA
    # extract_dsm(fn, fc, "max")
    # extract_dtm(fn, fc)

    # CALCULATE DTM WITH FILLED NO DATA
    match_size_dtm(dsm_fn, tmp_dtm_fn, dtm_fn)

    # CALCULATE CHM
    chm_calculate(tmp_dtm_fn, dtm_fn, dsm_fn, chm_fn)

    # CREATE SEGMENTS OF CHM DATA
    segments = chm_segmentation(dsm_fn)

    # SAVE SEGMENTS TO NEW RASTER
    save_segments_to_raster(seg_tif_fn, segments, dsm_fn)

    # CREATE SHAPEFILE SEGMENTS, CREATE STATISTIC CSV
    create_segments_shp(seg_tif_fn, segments_fn, chm_fn, csv_fn)

    # CREATE MESSAGE TO PRINT
    files_list = [dtm_fn, tmp_dtm_fn, dsm_fn, chm_fn, seg_tif_fn, segments_fn, csv_fn]

    msg = ""
    for file in files_list:
        # msg += "\t{}\n".format(os.path.basename(file))
        msg += "{}\n".format(file)
    return msg


def extract_dtm(lidar_fn, out_fn, in_srs="EPSG:2180", out_srs="EPSG:2178"):
    start = time.time()
    print("DTM extracting...")
    pdal_json = {
        "pipeline": [
            "{}".format(format(lidar_fn)),
            {
                "type": "filters.reprojection",
                "in_srs": "{}".format(in_srs),
                "out_srs": "{}".format(out_srs)
            },
            {
                "type": "filters.assign",
                "assignment": "Classification[:]=0"
            },
            {
                "type": "filters.elm"
            },
            {
                "type": "filters.outlier"
            },
            {
                "type": "filters.smrf",
                "ignore": "Classification[7:7]",
                "slope": 0.2,
                "window": 16,
                "threshold": 0.45,
                "scalar": 1.2
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:2]"
            },
            {
                "filename": "{}\TMP_DTM_{}.tif".format(out_fn, os.path.basename(lidar_fn)[:-4]),
                "gdaldriver": "GTiff",
                "output_type": "min",
                "resolution": "1",
                "type": "writers.gdal"
            }
        ]
    }

    pdal_json_str = json.dumps(pdal_json)
    pipeline = pdal.Pipeline(pdal_json_str)
    pipeline.execute()
    end = time.time()

    print("DTM extracted successfully in {:.2f} sec\n".format(end - start))


def extract_dsm(lidar_fn, out_fn, stat, in_srs="EPSG:2180", out_srs="EPSG:2178"):
        """the first assumption of the function was to calculate the results for both
        DSM and DTM (only in trees areas) from lidar data. Due to complication with
        DTM (the height was overstated by about 5m) right now it's used only for
        calculating DSM. The "stat" parameter was originally needed to differentiate
        between both layers"""

        start = time.time()
        elevation = "DTM" if stat == "min" else "DSM"
        print("{} extracting...".format(elevation))

        # pdal_json = {
        #     "pipeline": [
        #         "{}".format(lidar_fn),
        #         {
        #             "type": "filters.reprojection",
        #             "in_srs": "{}".format(in_srs),
        #             "out_srs": "{}".format(out_srs)
        #         },
        #         {
        #             "type": "filters.hag_nn",
        #         },
        #         {
        #             "type": "filters.range",
        #             "limits": "Classification[5:5]",
        #         },
        #         {
        #             "filename": "{}\{}_{}.tif".format(out_fn, elevation, os.path.basename(lidar_fn)[:-4]),
        #             "gdaldriver": "GTiff",
        #             "output_type": "{}".format(stat),
        #             "resolution": "1",
        #             "type": "writers.gdal"
        #         }
        #     ]
        # }

        pdal_json = {
            "pipeline": [
                "{}".format(lidar_fn),
                {
                    "type": "filters.reprojection",
                    "in_srs": "{}".format(in_srs),
                    "out_srs": "{}".format(out_srs)
                },
                {
                    "type": "filters.range",
                    "limits": "returnnumber[1:1]"
                },
                {
                    "type": "filters.range",
                    "limits": "Classification[5:5]",
                },
                {
                    "type": "writers.gdal",
                    "filename": "{}\{}_{}.tif".format(out_fn, elevation, os.path.basename(lidar_fn)[:-4]),
                    "output_type": "{}".format(stat),
                    "gdaldriver": "GTiff",
                    "resolution": 1,
                    "radius": 1

                }
            ]
        }

        pdal_json_str = json.dumps(pdal_json)
        pipeline = pdal.Pipeline(pdal_json_str)
        pipeline.execute()
        end = time.time()

        print("{} extracted successfully in {:.2f} sec\n".format(elevation, end - start))


def match_size_dtm(dsm_fn, tmp_dtm_fn, dtm_fn):
    """Filling no data value in calculated DTM raster, also changing raster sizes matching it to
    DSM raster size (PDAL creates DSM a little bit smaller)"""

    driver_tiff = gdal.GetDriverByName("GTiff")
    dsm_ds = gdal.Open(dsm_fn)
    tmp_dtm_ds = gdal.Open(tmp_dtm_fn)

    cols = dsm_ds.RasterXSize
    rows = dsm_ds.RasterYSize

    # dtm_data = temp_dtm_ds.GetRasterBand(1).ReadAsArray()
    temp_dtm_data = tmp_dtm_ds.GetRasterBand(1).ReadAsArray()

    dtm_data = np.zeros((rows, cols))
    dtm_data[0:rows, 0:cols] = temp_dtm_data[0:rows, 0:cols]
    dtm_data[dtm_data == -9999] = np.NaN

    dtm_ds = driver_tiff.Create(dtm_fn,
                                dsm_ds.RasterXSize,
                                dsm_ds.RasterYSize,
                                1,
                                gdal.GDT_Float32)

    dtm_ds.SetProjection(dsm_ds.GetProjection())
    dtm_ds.SetGeoTransform(dsm_ds.GetGeoTransform())
    dtm_ds.GetRasterBand(1).WriteArray(dtm_data)
    dtm_ds.GetRasterBand(1).SetNoDataValue(np.NaN)

    # FILLING NO DATA VALUE IN GROUND RASTER
    # inplace, filling gaps under buildings/trees with interpolated values
    gdal.FillNodata(targetBand=dtm_ds.GetRasterBand(1),
                    maskBand=None,
                    maxSearchDist=10,
                    smoothingIterations=1)
    dtm_ds = None
    return dtm_data


def chm_calculate(tmp_dtm_fn, dtm_fn, dsm_fn, chm_fn):
    """Calculating CHM from DSM and DTM"""
    start = time.time()
    print("CHM calculating...")

    # LOADING DRIVER
    driver_tiff = gdal.GetDriverByName("GTiff")

    # OPEN DATASET & READ DATA
    temp_dtm_ds = gdal.Open(tmp_dtm_fn, 1)  # GA_Update
    dsm_ds = gdal.Open(dsm_fn, 1)
    dsm_data = dsm_ds.GetRasterBand(1).ReadAsArray()

    dtm_ds = gdal.Open(dtm_fn, 1)
    dtm_data = dtm_ds.GetRasterBand(1).ReadAsArray()

    # PDAL CREATES DIFFERENT XSIZE, YSIZE RASTERS FOR DEM AND DSM
    # saving data from ground raster to new one with changed x,y sizes

    # cols = dsm_ds.RasterXSize if dsm_ds.RasterXSize < dtm_ds.RasterXSize else dtm_ds.RasterXSize
    # rows = dsm_ds.RasterYSize if dsm_ds.RasterYSize < dtm_ds.RasterYSize else dtm_ds.RasterYSize

    # CALCULATE CHM
    chm_data = dsm_data - dtm_data
    chm_data[chm_data <= 0] = np.NaN
    chm_data = np.where(np.isfinite(chm_data), chm_data, np.NaN) # replacing nan with -9999 -> np.NaN is enough


    # CREATE FILTERED RASTER AND SAVE DATA
    chm_ds = driver_tiff.Create(chm_fn,
                                dsm_ds.RasterXSize,
                                dsm_ds.RasterYSize,
                                1,
                                gdal.GDT_Float32)

    chm_ds.SetProjection(dsm_ds.GetProjection())
    chm_ds.SetGeoTransform(dsm_ds.GetGeoTransform())
    chm_ds.GetRasterBand(1).WriteArray(chm_data)
    chm_ds.GetRasterBand(1).SetNoDataValue(-9999)

    # CLOSING DATASETS
    dtm_ds = None
    dsm_ds = None
    chm_ds = None
    temp_dtm_ds = None
    end = time.time()
    print("CHM calculated successfully in {:.4f} sec\n".format(end - start))


def chm_segmentation(chm_fn, footprint, f_offset, truncate, compactness, line):
    """CHM ata filtering, masking and watershed segmentation.
    Idea from https://www.neonscience.org/resources/learning-hub/tutorials/calc-biomass-py"""

    driver = gdal.GetDriverByName("GTiff")
    chm_ds = gdal.Open(chm_fn)
    chm_ds.GetRasterBand(1).SetNoDataValue(0)
    chm_array = chm_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    start = time.time()
    print("Watershed segmentation...")
    # APPLYING GAUSSIAN FILTER TO REMOVE WRONG POINTS
    chm_array_smooth = ndi.gaussian_filter(chm_array, f_offset,
                                           mode='constant',
                                           cval=0,
                                           truncate=truncate)
    chm_array_smooth[chm_array == 0] = 0

    # CALCULATE LOCAL MAXIMUM POINTS
    local_maxi = peak_local_max(chm_array_smooth,
                                footprint=np.ones((footprint, footprint)))

    # CREATE MASK TO MATCH INPUT ARRAY SIZE
    markers = np.zeros_like(chm_array_smooth, dtype=bool)
    markers[tuple(local_maxi.T)] = True

    # IDENTIFY ALL THE MAXIMUM POINTS
    markers = ndi.label(markers)[0]

    # CREATE MASK TO REDUCE SEGMENTATION ONLY TO TREES
    chm_mask = chm_array_smooth.copy()
    chm_mask[chm_array_smooth > 0] = 1
    chm_mask[chm_array_smooth <= 0] = 0

    # WATERSHED SEGMENTATION
    labels = watershed(chm_array_smooth,
                       markers,
                       compactness=compactness,
                       mask=chm_mask,
                       watershed_line=False if line == "False" else True)

    # CHANGING 0 VALUES TO -9999
    segments = np.where(labels == 0, -9999, labels)

    chm_ds = None
    end = time.time()
    print("CHM segmented in {:.4f} sec\n".format(end - start))

    return segments


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
    # (due to to problems with deleting features with ogr using DeleteFeature)
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


def create_files_names(fc, fn, name, format_type):
    out_fn = fc + r"\{}_{}.{}".format(name, os.path.basename(fn)[:-4], format_type)
    return out_fn


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




