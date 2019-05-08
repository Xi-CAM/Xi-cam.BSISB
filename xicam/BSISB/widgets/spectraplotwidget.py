from pyqtgraph import PlotWidget
from xicam.core import msg
from xicam.core.data import NonDBHeader
import numpy as np


class SpectraPlotWidget(PlotWidget):
    def __init__(self, *args, **kwargs):
        super(SpectraPlotWidget, self).__init__(*args, **kwargs)
        self._data = None

    def setHeader(self, header: NonDBHeader, field: str, *args, **kwargs):
        self.header = header
        self.field = field
        # make lazy array from document
        data = None
        try:
            data = header.meta_array(field)
        except IndexError:
            msg.logMessage('Header object contained no frames with field ''{field}''.', msg.ERROR)

        if data is not None:
            # kwargs['transform'] = QTransform(1, 0, 0, -1, 0, data.shape[-2])
            self._data = data

    def showSpectra(self, x, y):
        if self._data is not None:
            self.clear()
            self.plot(self._data[:, x, y])
