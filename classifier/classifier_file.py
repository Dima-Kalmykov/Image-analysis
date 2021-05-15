import pickle
import time
from typing import (Tuple,
                    List)

import numpy as np
from osgeo import gdal
from qgis.core import QgsPalettedRasterRenderer
from skimage.segmentation import slic

from Backend.classifier.qgis_helper import QGISHelper
from Backend.utils.file_paths import FilePaths
from Backend.utils.utils import Utils


class ImageClassifier:

    @staticmethod
    def save_colorized_tif(file_number: int, colors) -> None:
        """
        Save colored tif to file.
        :param colors: colors
        :param file_number: id of file
        """
        layer = QGISHelper.get_layer(file_number)

        colors = QGISHelper.get_color_schema(colors)

        renderer = QgsPalettedRasterRenderer(layer.dataProvider(), 1,
                                             QgsPalettedRasterRenderer.colorTableToClassData(colors))
        layer.setRenderer(renderer)

        layer_info = QGISHelper.get_layer_info(layer)

        QGISHelper.write_colorized_tif_to_file(layer_info, file_number)

    @staticmethod
    def save_colorless_tif(file_number: int,
                           driver_tiff: gdal.Driver,
                           source_dataset: gdal.Dataset,
                           segments_copy: np.ndarray) -> None:
        """
        Save colorless '.tif' to file.
        :param file_number: id of file
        :param driver_tiff: driver fot '.tif' files
        :param source_dataset: dataset
        :param segments_copy: segments
        """
        file_path_to_result = FilePaths.RESULT_PATH_WITHOUT_COLOR + str(file_number) + ".tif"

        result_dataset = driver_tiff.Create(file_path_to_result,
                                            source_dataset.RasterXSize,
                                            source_dataset.RasterYSize,
                                            1,
                                            gdal.GDT_Float32)

        result_dataset.SetGeoTransform(source_dataset.GetGeoTransform())
        result_dataset.SetProjection(source_dataset.GetProjection())
        result_dataset.GetRasterBand(1).SetNoDataValue(-9999.0)
        result_dataset.GetRasterBand(1).WriteArray(segments_copy)

    @staticmethod
    def apply_mask(image: np.ndarray, segments_copy: np.ndarray) -> None:
        """
        Apply mask to segments.
        :param image: image
        :param segments_copy: segments
        """
        mask = np.sum(image, axis=2)
        mask[mask > 0.0] = 1.0
        mask[mask == 0.0] = -1.0

        segments_copy = np.multiply(segments_copy, mask)
        segments_copy[segments_copy < 0] = -9999.0

    @staticmethod
    def load_main_tools(path: str) -> Tuple[np.ndarray, gdal.Driver, gdal.Dataset]:
        """
        Load main tools for algorithm.
        :param path: path to dataset
        :return:
        """
        file_path_to_source_dataset = path
        driver_tiff = gdal.GetDriverByName("GTiff")
        source_dataset = gdal.Open(file_path_to_source_dataset)
        image = Utils.get_image(source_dataset)

        return image, driver_tiff, source_dataset

    @staticmethod
    def pixel_bypass(segments: np.ndarray,
                     image: np.ndarray,
                     id: int) -> Tuple[List[List[np.uint8]], np.ndarray]:
        """
        Pixel bypass.
        :param id: file id
        :param segments: segments
        :param image: image
        :return: objects with extra info and segment ids
        """
        start_pixel_bypass_time = time.time()
        segment_ids = np.unique(segments)

        objects, object_ids = Utils.get_objects_and_ids(segment_ids, segments, image, id)
        Utils.print_duration("Pixel bypass done:", start_pixel_bypass_time)

        return objects, segment_ids

    @staticmethod
    def get_segments_copy(segments: np.ndarray,
                          segment_ids: np.ndarray,
                          predicted: np.ndarray) -> np.ndarray:
        """
        Get segment's copy.
        :param segments:  segments
        :param segment_ids: segment ids
        :param predicted: predicted
        :return: the copy of segments
        """
        segments_copy = np.copy(segments)

        for segment_id, point_class in zip(segment_ids, predicted):
            segments_copy[segments_copy == segment_id] = point_class

        return segments_copy

    @staticmethod
    def get_classifier(classifier_type: str) -> str:
        """
        Get path to model.
        :param classifier_type: classifier type
        :return: path to model
        """
        result_path = ""
        if classifier_type == 'rbf':
            result_path = FilePaths.CLASSIFIER_PATH_RBF
        if classifier_type == 'linear':
            result_path = FilePaths.CLASSIFIER_PATH_LINEAR
        if classifier_type == 'poly':
            result_path = FilePaths.CLASSIFIER_PATH_POLY
        if classifier_type == 'sigmoid':
            result_path = FilePaths.CLASSIFIER_PATH_SIGMOID

        return pickle.load(open(result_path, 'rb'))

    def classify(self, path: str, file_number: int, colors, classifier_type: str) -> None:
        """
        Classify image.
        :param classifier_type: classifier type
        :param colors: colors
        :param path: path to image
        :param file_number: id o file
        """
        start_program_time = time.time()

        image, driver_tiff, source_dataset = self.load_main_tools(path)

        start_segmentation_time = time.time()
        segments = slic(image, n_segments=50000, compactness=0.1, start_label=0)
        Utils.print_duration(f"Segmentation for file with id = {file_number} done:", start_segmentation_time)

        objects, segment_ids = self.pixel_bypass(segments, image, file_number)

        classifier = self.get_classifier(classifier_type)

        start_prediction_time = time.time()
        predicted = classifier.predict(objects)

        segments_copy = self.get_segments_copy(segments, segment_ids, predicted)
        self.apply_mask(image, segments_copy)

        Utils.print_duration(f"Prediction for file with id = {file_number} done:", start_prediction_time)

        self.save_colorless_tif(file_number, driver_tiff, source_dataset, segments_copy)

        self.save_colorized_tif(file_number, colors)

        Utils.print_duration("Total:", start_program_time)
        Utils.save_colorized_tif_as_jpg(file_number)
        print(f"Classification for file with id = {file_number} done!")


# path_to_file = "C:/temp/clipped.tif"
# file_number_1 = 2
# colors_1 = ['#FFFF00', '#0000FF', '#00FF00', '#FF6347', '#800080', '#FFA500']
# kernel_1 = 'rbf'
# ImageClassifier().classify(path_to_file, file_number_1, colors_1, kernel_1)
# naip_fn = "C:/temp2/дата/train/Chicago.tif"
# driver = gdal.GetDriverByName("GTiff")
# naip_ds = gdal.Open(naip_fn)
# band_data = []
# print('bands', naip_ds.RasterCount, 'rows', naip_ds.RasterYSize, 'columns', naip_ds.RasterXSize)
# n_bands = naip_ds.RasterCount
# for i in range(1, n_bands + 1):
#     band = naip_ds.GetRasterBand(i).ReadAsArray()
#     band_data.append(band)
#
# band_data = np.dstack(band_data)