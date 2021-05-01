from qgis.core import (QgsRasterPipe,
                       QgsRasterLayer)


class LayerInfo:

    extent = None
    width = None
    height = None
    renderer = None
    provider = None
    crs = None
    pipe = None

    def __init__(self, layer: QgsRasterLayer):
        self.extent = layer.extent()
        self.width = layer.width()
        self.height = layer.height()
        self.renderer = layer.renderer()
        self.provider = layer.dataProvider()
        self.crs = layer.crs()

        self.init_pipe()

    def init_pipe(self) -> None:
        """
        Initialize 'pipe' field.
        """
        self.pipe = QgsRasterPipe()
        self.pipe.set(self.provider.clone())
        self.pipe.set(self.renderer.clone())
