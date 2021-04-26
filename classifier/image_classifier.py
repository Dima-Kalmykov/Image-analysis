import time
import numpy as np
from PyQt5.QtCore import QFileInfo
from PyQt5.QtGui import QColor
from osgeo import gdal
import pickle
from skimage.segmentation import slic
from qgis.core import (QgsRasterLayer,
                       QgsColorRampShader,
                       QgsPalettedRasterRenderer,
                       QgsRasterPipe,
                       QgsRasterFileWriter)

from file_paths import FilePaths
from utils import Utils


start_program_time = time.time()

file_path_to_source_dataset = 'C:/temp/clipped2.tif'
driver_tiff = gdal.GetDriverByName("GTiff")

source_dataset = gdal.Open(file_path_to_source_dataset)
image = Utils.get_image(source_dataset)

start_segmentation_time = time.time()
segments = slic(image, n_segments=50000, compactness=0.1, start_label=0)
Utils.print_duration("Segmentation done:", start_segmentation_time)

start_pixel_bypass_time = time.time()
segment_ids = np.unique(segments)
objects, object_ids = Utils.get_objects_and_ids(segment_ids, segments, image)
Utils.print_duration("Pixel bypass done:", start_pixel_bypass_time)

classifier = pickle.load(open(FilePaths.classifier_path, 'rb'))

start_prediction_time = time.time()
predicted = classifier.predict(objects)

segments_copy = np.copy(segments)

for segment_id, point_class in zip(segment_ids, predicted):
    segments_copy[segments_copy == segment_id] = point_class

mask = np.sum(image, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0

segments_copy = np.multiply(segments_copy, mask)
segments_copy[segments_copy < 0] = -9999.0
Utils.print_duration("Prediction done:", start_prediction_time)

file_path_to_result = FilePaths.result_path_without_color

result_dataset = driver_tiff.Create(file_path_to_result, source_dataset.RasterXSize,
                                    source_dataset.RasterYSize,
                                    1, gdal.GDT_Float32)

result_dataset.SetGeoTransform(source_dataset.GetGeoTransform())
result_dataset.SetProjection(source_dataset.GetProjection())
result_dataset.GetRasterBand(1).SetNoDataValue(-9999.0)
result_dataset.GetRasterBand(1).WriteArray(segments_copy)
result_dataset = None


fileInfo = QFileInfo(file_path_to_result)
baseName = fileInfo.baseName()
layer = QgsRasterLayer(file_path_to_result, baseName)

colors = [QgsColorRampShader.ColorRampItem(1, QColor("#d2ca97")),
          QgsColorRampShader.ColorRampItem(2, QColor("#f7f7f7")),
          QgsColorRampShader.ColorRampItem(3, QColor("#a1d99b")),
          QgsColorRampShader.ColorRampItem(4, QColor("#41ab5d")),
          QgsColorRampShader.ColorRampItem(5, QColor("#006d2c")),
          QgsColorRampShader.ColorRampItem(6, QColor("#00441b"))]

renderer = QgsPalettedRasterRenderer(layer.dataProvider(), 1,
                                     QgsPalettedRasterRenderer.colorTableToClassData(colors))
layer.setRenderer(renderer)

extent = layer.extent()
width = layer.width()
height = layer.height()
renderer = layer.renderer()
provider = layer.dataProvider()
crs = layer.crs()

pipe = QgsRasterPipe()
pipe.set(provider.clone())
pipe.set(renderer.clone())

file_writer = QgsRasterFileWriter(FilePaths.result_path_with_color)
file_writer.writeRaster(pipe, width, height, extent, crs)

Utils.print_duration("Total:", start_program_time)
Utils.save_result_as_jpg()
print("Done!")
