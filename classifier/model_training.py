import time
import numpy as np
from osgeo import gdal
from osgeo import ogr
from skimage.segmentation import slic
from sklearn import svm
import pickle

from file_paths import FilePaths
from utils import Utils

start_program_time = time.time()

file_path_to_source_data = 'C:/temp/clipped.tif'
driver_tiff = gdal.GetDriverByName("GTiff")

source_dataset = gdal.Open(file_path_to_source_data)

image = Utils.get_image(source_dataset)

start_segmentation_time = time.time()
segments = slic(image, n_segments=50000, compactness=0.1, start_label=0)
Utils.print_duration("Segmentation done:", start_segmentation_time)

segment_ids = np.unique(segments)
start_pixel_bypass_time = time.time()
objects, object_ids = Utils.get_objects_and_ids(segment_ids, segments, image)
Utils.print_duration("Pixel bypass done:", start_pixel_bypass_time)

train_file_path = "C:/temp/landsat/train.shp"
train_dataset = ogr.Open(train_file_path)
raster_layer = train_dataset.GetLayer()

# Memory driver to create dataset in memory.
driver_mem = gdal.GetDriverByName("MEM")
target_dataset = driver_mem.Create('', source_dataset.RasterXSize, source_dataset.RasterYSize, 1, gdal.GDT_UInt16)
target_dataset.SetGeoTransform(source_dataset.GetGeoTransform())
target_dataset.SetProjection(source_dataset.GetProjection())

options = ["ATTRIBUTE=id"]
gdal.RasterizeLayer(target_dataset, [1], raster_layer, options=options)

ground_truth = target_dataset.GetRasterBand(1).ReadAsArray()
classes = np.unique(ground_truth)[1:]

segments_per_class = {}

for point_class in classes:
    segments_of_class = segments[ground_truth == point_class]
    segments_per_class[point_class] = set(segments_of_class)

intersection = set()
accum = set()
for class_segment in segments_per_class.values():
    intersection |= accum.intersection(class_segment)
    accum |= class_segment

assert len(intersection) == 0, "Segments represent multiple classes"

train_image = np.copy(segments)
threshold = train_image.max() + 1

for point_class in classes:
    class_label = threshold + point_class
    for segment_id in segments_per_class[point_class]:
        train_image[train_image == segment_id] = class_label

train_image[train_image <= threshold] = 0
train_image[train_image > threshold] -= threshold

training_objects = []
training_labels = []

for point_class in classes:
    class_train_object = [value for index, value in enumerate(objects) if
                          segment_ids[index] in segments_per_class[point_class]]

    training_labels += [point_class] * len(class_train_object)
    training_objects += class_train_object

classifier = svm.SVC()
classifier.fit(training_objects, training_labels)

pickle.dump(classifier, open(FilePaths.classifier_path, 'wb'))
Utils.print_duration("Total:", start_program_time)
print("Training done!")
