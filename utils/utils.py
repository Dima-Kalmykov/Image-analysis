from PIL import Image
from scipy import stats
from osgeo import gdal
from skimage import exposure
from typing import List, Tuple

import time
import numpy as np

from utils.file_paths import FilePaths


class Utils:

    @staticmethod
    def save_result_as_jpg(file_number) -> None:
        """
        Save '.tif' file as '.jpg'.
        """
        result_path = FilePaths.result_path_with_color
        out_path = result_path + str(file_number) + ".jpg"
        print(result_path)
        image = Image.open(result_path + str(file_number) + ".tif")
        image.thumbnail(image.size)
        rgb_image = image.convert('RGB')
        rgb_image.save(out_path, "JPEG", quiality=100)

    @staticmethod
    def print_duration(message: str, start_time: time) -> None:
        """
        Print duration of period, which starts with 'start_time'.
        :param message: auxiliary message
        :param start_time: start time for measuring
        """
        end_time = time.time()
        total = end_time - start_time
        minutes = total // 60
        seconds = total - minutes * 60
        print(message, " minutes = ", minutes, ", seconds = ", round(seconds, 4), sep="", end="\n\n")

    @staticmethod
    def get_segment_features(pixels_in_segment: np.ndarray) -> List[np.uint8]:
        """
        Get info about pixels in given segment.
        :param pixels_in_segment:  pixels in segment
        :return: list of extra data about segment
        """
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

    @staticmethod
    def get_image(source_dataset: gdal.Dataset) -> np.ndarray:
        """
        Get image as array of pixels.
        :param source_dataset: dataset
        :return: list of pixels
        """
        bands_count = source_dataset.RasterCount

        band_data = []

        for i in range(1, bands_count + 1):
            band = source_dataset.GetRasterBand(i).ReadAsArray()
            band_data.append(band)

        band_data = np.dstack(band_data)

        image = exposure.rescale_intensity(band_data)
        return image

    @staticmethod
    def get_objects_and_ids(segment_ids: np.ndarray,
                            segments: np.ndarray,
                            image: np.ndarray) -> Tuple[List[List[np.uint8]], List[np.int64]]:
        """
        Get additional information about each segment.
        :param segment_ids: list of segment id
        :param segments: list of pixels in segment
        :param image: total list of pixels
        :return: tuple of objects with extra info and its id
        """
        objects = []
        object_ids = []

        for segment_id in segment_ids:
            segment_pixels = image[segments == segment_id]
            object_features = Utils.get_segment_features(segment_pixels)
            objects.append(object_features)
            object_ids.append(segment_id)

        return objects, object_ids
