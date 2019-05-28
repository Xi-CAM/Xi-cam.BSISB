from pyqtgraph import PlotWidget, TextItem
from xicam.core import msg
from xicam.core.data import NonDBHeader
import numpy as np
from pyqtgraph import InfiniteLine
from qtpy.QtCore import Signal
from lbl_ir.data_objects.ir_map import val2ind


class SpectraPlotWidget(PlotWidget):
    sigEnergyChanged = Signal(object)

    def __init__(self, *args, **kwargs):
        super(SpectraPlotWidget, self).__init__(*args, **kwargs)
        self._data = None
        self.positionmap = dict()
        self.wavenumbers = None
        self.line = InfiniteLine(movable=True)
        self.line.setPen((255, 255, 0, 200))
        self.line.setZValue(100)
        self.line.sigPositionChanged.connect(self.sigEnergyChanged)
        self.line.sigPositionChanged.connect(self.getEnergy)
        self.addItem(self.line)

    def getEnergy(self, lineobject):
        self.energy = lineobject.value()
        idx = val2ind(self.energy, self.wavenumbers)
        self.txt.setHtml(f'<div style="text-align: center"><span style="color: #FFF; font-size: 12pt">\
                            X = {self.energy: .2f}, Y = {self._y[idx]: .4f}</div>')

    def setHeader(self, header: NonDBHeader, field: str, *args, **kwargs):
        self.header = header
        self.field = field
        # get wavenumbers
        spectraEvent = next(header.events(fields=['spectra']))
        self.wavenumbers = spectraEvent['wavenumbers']
        # make lazy array from document
        data = None
        try:
            data = header.meta_array(field)
        except IndexError:
            msg.logMessage('Header object contained no frames with field ''{field}''.', msg.ERROR)

        if data is not None:
            # kwargs['transform'] = QTransform(1, 0, 0, -1, 0, data.shape[-2])
            self._data = data

    def showSpectra(self, i=0):
        if self._data is not None:
            self.clear()
            self._y = self._data[i]
            self._ymax = max(self._y)
            self.plot(self.wavenumbers, self._y)
            self.getViewBox().invertX(True)

    def plot(self, *args, **kwargs):
        self.plotItem.plot(*args, **kwargs)
        self.addItem(self.line)
        self.txt = TextItem(
            html=f'<div style="text-align: center"><span style="color: #FFF; font-size: 12pt">X = {0}, Y = {0}</div>',
            anchor=(0, 0))
        self.txt.setPos(1500, 0.95 * self._ymax)
        self.addItem(self.txt)
