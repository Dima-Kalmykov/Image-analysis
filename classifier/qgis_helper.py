from typing import List

from PyQt5.QtCore import QFileInfo
from PyQt5.QtGui import QColor
from qgis.core import (QgsRasterLayer,
                       QgsColorRampShader,
                       QgsRasterFileWriter)

from classifier.layer_model import LayerInfo
from utils.file_paths import FilePaths


class QGISHelper:

    @staticmethod
    def get_layer_info(layer: QgsRasterLayer) -> LayerInfo:
        """
        Get info about given layer.
        :param layer: raster layer
        :return: object with layer information
        """
        layer_info = LayerInfo(layer)

        return layer_info

    @staticmethod
    def get_layer(file_number: int) -> QgsRasterLayer:
        """
        Get layer by file number.
        :param file_number: id of '.tif' file
        :return: raster layer
        """
        file_path_to_result = FilePaths.result_path_without_color + str(file_number) + ".tif"

        file_info = QFileInfo(file_path_to_result)
        base_name = file_info.baseName()
        layer = QgsRasterLayer(file_path_to_result, base_name)

        return layer

    @staticmethod
    def get_color_schema() -> List[QgsColorRampShader.ColorRampItem]:
        """
        Get color schema.
        :return: color schema
        """
        color_schema = [
            QgsColorRampShader.ColorRampItem(1, QColor("#d2ca97")),
            QgsColorRampShader.ColorRampItem(2, QColor("#f7f7f7")),
            QgsColorRampShader.ColorRampItem(3, QColor("#a1d99b")),
            QgsColorRampShader.ColorRampItem(4, QColor("#41ab5d")),
            QgsColorRampShader.ColorRampItem(5, QColor("#006d2c")),
            QgsColorRampShader.ColorRampItem(6, QColor("#00441b"))
        ]

        return color_schema

    @staticmethod
    def write_colorized_tif_to_file(layer_info: LayerInfo, file_number: int) -> None:
        """
        Write layer to file.
        :param layer_info: info about layer
        :param file_number: id of file
        """
        new_path = FilePaths.result_path_with_color + str(file_number) + ".tif"

        file_writer = QgsRasterFileWriter(new_path)
        file_writer.writeRaster(layer_info.pipe,
                                layer_info.width,
                                layer_info.height,
                                layer_info.extent,
                                layer_info.crs)
