from pyqtgraph import PlotWidget, TextItem, PlotDataItem
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
        self._meanSpec = True # whether current spectrum is a mean spectrum
        self.line = InfiniteLine(movable=True)
        self.line.setPen((255, 255, 0, 200))
        self.line.setValue(650)
        self.line.sigPositionChanged.connect(self.sigEnergyChanged)
        self.line.sigPositionChanged.connect(self.getEnergy)
        self.addItem(self.line)
        self.cross = PlotDataItem([650], [0], symbolBrush=(255, 0, 0), symbolPen=(255, 0, 0), symbol='+',
                                  symbolSize=20)
        self.cross.setZValue(100)
        self.addItem(self.cross)
        self.getViewBox().invertX(True)
        self.spectrumInd = 0
        self.selectedPixels = None
        self._y = None

    def getEnergy(self, lineobject):
        if self._y is not None:
            x_val = lineobject.value()
            idx = val2ind(x_val, self.wavenumbers)
            x_val = self.wavenumbers[idx]
            y_val = self._y[idx]
            if not self._meanSpec:
                txt_html = f'<div style="text-align: center"><span style="color: #FFF; font-size: 12pt">\
                                Spectrum #{self.spectrumInd}</div>'
            else:
                txt_html = f'<div style="text-align: center"><span style="color: #FFF; font-size: 12pt">\
                                 {self._mean_title}</div>'

            txt_html += f'<div style="text-align: center"><span style="color: #FFF; font-size: 12pt">\
                                 X = {x_val: .2f}, Y = {y_val: .4f}</div>'
            self.txt.setHtml(txt_html)
            self.cross.setData([x_val], [y_val])

    def setHeader(self, header: NonDBHeader, field: str, *args, **kwargs):
        self.header = header
        self.field = field
        # get wavenumbers
        spectraEvent = next(header.events(fields=['spectra']))
        self.wavenumbers = spectraEvent['wavenumbers']
        self.N_w = len(self.wavenumbers)
        self.rc2ind = spectraEvent['rc_index']
        # make lazy array from document
        data = None
        try:
            data = header.meta_array(field)
        except IndexError:
            msg.logMessage(f'Header object contained no frames with field {field}.', msg.ERROR)

        if data is not None:
            # kwargs['transform'] = QTransform(1, 0, 0, -1, 0, data.shape[-2])
            self._data = data

    def showSpectra(self, i=0):
        if (self._data is not None) and (i < len(self._data)):
            self.clear()
            self._meanSpec = False
            self.spectrumInd = i
            self.plot(self.wavenumbers, self._data[i])

    def getSelectedPixels(self, selectedPixels):
        self.selectedPixels = selectedPixels
        # print(selectedPixels)

    def showMeanSpectra(self):
        self._meanSpec = True
        self.clear()
        if self.selectedPixels is not None:
            n_spectra = len(self.selectedPixels)
            tmp = np.zeros((n_spectra, self.N_w))
            for j in range(n_spectra):  # j: jth selected pixel
                row_col = tuple(self.selectedPixels[j])
                tmp[j, :] = self._data[self.rc2ind[row_col]]
            self._mean_title = f'ROI mean of {n_spectra} spectra'
        else:
            n_spectra = len(self._data)
            tmp = np.zeros((n_spectra, self.N_w))
            for j in range(n_spectra):
                tmp[j, :] = self._data[j]
            self._mean_title = f'Total mean of {n_spectra} spectra'
        if n_spectra > 0:
            meanSpec = np.mean(tmp, axis=0)
        else:
            meanSpec = np.zeros_like(self.wavenumbers) + 1e-3
        self.plot(self.wavenumbers, meanSpec)

    def plot(self, x, y, *args, **kwargs):
        # set up infinity line and get its position
        self.plotItem.plot(x, y, *args, **kwargs)
        self.addItem(self.line)
        self.addItem(self.cross)
        x_val = self.line.value()
        if x_val == 0:
            y_val = 0
        else:
            idx = val2ind(x_val, self.wavenumbers)
            x_val = self.wavenumbers[idx]
            y_val = y[idx]

        if not self._meanSpec:
            txt_html = f'<div style="text-align: center"><span style="color: #FFF; font-size: 12pt">\
                            Spectrum #{self.spectrumInd}</div>'
        else:
            txt_html = f'<div style="text-align: center"><span style="color: #FFF; font-size: 12pt">\
                             {self._mean_title}</div>'

        txt_html += f'<div style="text-align: center"><span style="color: #FFF; font-size: 12pt">\
                             X = {x_val: .2f}, Y = {y_val: .4f}</div>'
        self.txt = TextItem(html=txt_html, anchor=(0, 0))
        ymax = max(y)
        self._y = y
        self.txt.setPos(1500, 0.95 * ymax)
        self.cross.setData([x_val], [y_val])
        self.addItem(self.txt)
