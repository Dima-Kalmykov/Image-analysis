import time
import numpy as np
from PyQt5.QtCore import QFileInfo
from PyQt5.QtGui import QColor
from scipy import stats
from osgeo import gdal
from osgeo import ogr
import pickle
from skimage.segmentation import slic
from skimage import exposure
from qgis.core import (QgsRasterLayer,
                       QgsColorRampShader,
                       QgsPalettedRasterRenderer,
                       QgsRasterPipe,
                       QgsRasterFileWriter)


def segment_features(pixels_in_segment: np.ndarray) -> list:
    features = []
    pixels_count, bands_number = pixels_in_segment.shape

    for band_number in range(bands_number):
        pixels_statistic = stats.describe(pixels_in_segment[:, band_number])
        band_statistic = list(pixels_statistic.minmax) + list(pixels_statistic)[2:]

        if pixels_count == 1:
            # in this case the variance = nan, change it 0.0
            band_statistic[3] = 0.0

        features += band_statistic

    return features


# file_name = 'C:/temp/clipped.tif'
file_name_without = 'C:/temp/clipped2.tif'

driver_tiff = gdal.GetDriverByName("GTiff")
# source_dataset = gdal.Open(file_name)
source_dataset_without = gdal.Open(file_name_without)

# bands_count = source_dataset.RasterCount
bands_count_without = source_dataset_without.RasterCount

band_data = []
band_data_without = []
# print("bands", bands_count, "rows", source_dataset.RasterYSize,
#       "columns", source_dataset.RasterXSize)
# for i in range(1, bands_count + 1):
#     band = source_dataset.GetRasterBand(i).ReadAsArray()
#     band_data.append(band)

for i in range(1, bands_count_without + 1):
    band_without = source_dataset_without.GetRasterBand(i).ReadAsArray()
    band_data_without.append(band_without)

# band_data = np.dstack(band_data)
band_data_without = np.dstack(band_data_without)

# normalized_image = exposure.rescale_intensity(band_data)
normalized_image_without = exposure.rescale_intensity(band_data_without)

start_segmentation_time = time.time()

# Save segments to raster. (рандомное разбиение)
# segments = slic(normalized_image, n_segments=50000, compactness=0.1, start_label=0)
segments_without = slic(normalized_image_without, n_segments=50000, compactness=0.1, start_label=0)
# segments = quickshift(img, convert2lab=False, ratio=0.99, max_dist=5)
print("Segments complete in ", time.time() - start_segmentation_time)

# segment_ids = np.unique(segments)
segment_ids_without = np.unique(segments_without)
# print(len(segment_ids))

objects = []
objects_without = []
object_ids = []
object_ids_without = []

print(len(segment_ids_without))

# for segment_id in segment_ids:
#     if segment_id % 1000 == 0:
#         print("Create object for id = ", segment_id)
#     segment_pixels = normalized_image[segments == segment_id]
#     object_features = segment_features(segment_pixels)
#     objects.append(object_features)
#     object_ids.append(segment_id)

for segment_id in segment_ids_without:
    if segment_id % 1000 == 0:
        print("Create object for id = ", segment_id)
    segment_pixels = normalized_image_without[segments_without == segment_id]
    object_features = segment_features(segment_pixels)
    objects_without.append(object_features)
    object_ids_without.append(segment_id)

# Result of segmentation
# segments_file_path = 'C:/temp/naip/segments_final.tif'
# segments_dataset = driver_tiff.Create(segments_file_path, source_dataset.RasterXSize,
#                                       source_dataset.RasterYSize, 1, gdal.GDT_Float32)
#
# segments_dataset.SetGeoTransform(source_dataset.GetGeoTransform())
# segments_dataset.SetProjection(source_dataset.GetProjectionRef())
# segments_dataset.GetRasterBand(1).WriteArray(segments)
# segments_dataset = None

train_file_path = "C:/temp/landsat/train.shp"
train_dataset = ogr.Open(train_file_path)
raster_layer = train_dataset.GetLayer()

# Memory driver to create dataset in memory.
# driver_mem = gdal.GetDriverByName("MEM")
# target_dataset = driver_mem.Create('', source_dataset.RasterXSize, source_dataset.RasterYSize, 1, gdal.GDT_UInt16)
# target_dataset.SetGeoTransform(source_dataset.GetGeoTransform())
# target_dataset.SetProjection(source_dataset.GetProjection())
# options = ["ATTRIBUTE=id"]
# gdal.RasterizeLayer(target_dataset, [1], raster_layer, options=options)
#
# ground_truth = target_dataset.GetRasterBand(1).ReadAsArray()
# classes = np.unique(ground_truth)[1:]
# print('classes values', classes)
#
# segments_per_class = {}

# for point_class in classes:
#     segments_of_class = segments[ground_truth == point_class]
#     segments_per_class[point_class] = set(segments_of_class)
#     print("Training segments for class", point_class, ":", len(segments_of_class))
#
# intersection = set()
# accum = set()
# for class_segment in segments_per_class.values():
#     intersection |= accum.intersection(class_segment)
#     accum |= class_segment
#
# assert len(intersection) == 0, "Segments represent multiple classes"
#
# train_image = np.copy(segments)
# threshold = train_image.max() + 1
# for point_class in classes:
#     class_label = threshold + point_class
#     for segment_id in segments_per_class[point_class]:
#         train_image[train_image == segment_id] = class_label
#
# train_image[train_image <= threshold] = 0
# train_image[train_image > threshold] -= threshold
#
# training_objects = []
# training_labels = []
#
# for point_class in classes:
#     class_train_object = [value for index, value in enumerate(objects) if
#                           segment_ids[index] in segments_per_class[point_class]]
#     training_labels += [point_class] * len(class_train_object)
#     training_objects += class_train_object
#     print("Training objects for class", point_class, ":", len(class_train_object))
#
# classifier_svm = svm.SVC()
# classifier_svm.fit(training_objects, training_labels)
# predicted_svm = classifier_svm.predict(objects)
# classifier = RandomForestClassifier(n_jobs=1)
# classifier.fit(training_objects, training_labels)
classifier = pickle.load(open('finalized_model.sav', 'rb'))
print("Fitting random forest classifier")
predicted = classifier.predict(objects_without)
print("Predicted =", predicted, "len = ", len(predicted))
print("Predicting classification")

# segments_copy = np.copy(segments)
segments_copy_without = np.copy(segments_without)
for segment_id, point_class in zip(segment_ids_without, predicted):
    # for segment_id, klass in zip(segment_ids, predicted_svm):
    segments_copy_without[segments_copy_without == segment_id] = point_class
print('Prediction applied to numpy array')
# mask = np.sum(normalized_image, axis=2)
mask = np.sum(normalized_image_without, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0
segments_copy_without = np.multiply(segments_copy_without, mask)
segments_copy_without[segments_copy_without < 0] = -9999.0

print('Saving classification to raster with gdal')
result_path = "C:/temp/naip/classified.tif"
result_dataset = driver_tiff.Create(result_path, source_dataset_without.RasterXSize,
                                    source_dataset_without.RasterYSize,
                                    1, gdal.GDT_Float32)

result_dataset.SetGeoTransform(source_dataset_without.GetGeoTransform())
result_dataset.SetProjection(source_dataset_without.GetProjection())
result_dataset.GetRasterBand(1).SetNoDataValue(-9999.0)
result_dataset.GetRasterBand(1).WriteArray(segments_copy_without)
result_dataset = None

fileInfo = QFileInfo(result_path)
baseName = fileInfo.baseName()
layer = QgsRasterLayer(result_path, baseName)

pcolor = []

pcolor.append(QgsColorRampShader.ColorRampItem(1, QColor("#d2ca97")))
pcolor.append(QgsColorRampShader.ColorRampItem(2, QColor("#f7f7f7")))
pcolor.append(QgsColorRampShader.ColorRampItem(3, QColor("#a1d99b")))
pcolor.append(QgsColorRampShader.ColorRampItem(4, QColor("#41ab5d")))
pcolor.append(QgsColorRampShader.ColorRampItem(5, QColor("#006d2c")))
pcolor.append(QgsColorRampShader.ColorRampItem(6, QColor("#00441b")))

renderer = QgsPalettedRasterRenderer(layer.dataProvider(), 1,
                                     QgsPalettedRasterRenderer.colorTableToClassData(pcolor))
layer.setRenderer(renderer)

extent = layer.extent()
width, height = layer.width(), layer.height()
renderer = layer.renderer()
provider = layer.dataProvider()
crs = layer.crs().toWkt()
pipe = QgsRasterPipe()
pipe.set(provider.clone())
pipe.set(renderer.clone())
file_writer = QgsRasterFileWriter("C:/temp/naip/classified_res.tif")
file_writer.writeRaster(pipe,
                        width,
                        height,
                        extent,
                        layer.crs())

print("Done!")
