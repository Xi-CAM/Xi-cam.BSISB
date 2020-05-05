from qtpy.QtCore import Qt
from pyqtgraph import PlotWidget, TextItem, PlotDataItem, mkPen
from xicam.core import msg
from xicam.core.data import NonDBHeader
import numpy as np
from pyqtgraph import InfiniteLine
from qtpy.QtCore import Signal
from lbl_ir.data_objects.ir_map import val2ind
from xicam.BSISB.widgets.mapviewwidget import toHtml


class SpectraPlotWidget(PlotWidget):
    sigEnergyChanged = Signal(object)

    def __init__(self, linePos=650, txtPosRatio=0.3, invertX=True, *args, **kwargs):
        """
        A widget to display a 1D spectrum
        :param linePos: the initial position of the InfiniteLine
        :param txtPosRatio: a coefficient that determines the relative position of the textItem
        :param invertX: whether to invert X-axis
        """
        super(SpectraPlotWidget, self).__init__(*args, **kwargs)
        self._data = None
        assert (txtPosRatio >= 0) and (txtPosRatio <= 1), 'Please set txtPosRatio value between 0 and 1.'
        self.txtPosRatio = txtPosRatio
        self.positionmap = dict()
        self.wavenumbers = None
        self._meanSpec = True  # whether current spectrum is a mean spectrum
        self.line = InfiniteLine(movable=True)
        self.line.setPen((255, 255, 0, 200))
        self.line.setValue(linePos)
        self.line.sigPositionChanged.connect(self.sigEnergyChanged)
        self.line.sigPositionChanged.connect(self.getEnergy)
        self.addItem(self.line)
        self.cross = PlotDataItem([linePos], [0], symbolBrush=(255, 0, 0), symbolPen=(255, 0, 0), symbol='+',
                                  symbolSize=20)
        self.cross.setZValue(100)
        self.addItem(self.cross)
        self.getViewBox().invertX(invertX)
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
                txt_html = toHtml(f'Spectrum #{self.spectrumInd}')
            else:
                txt_html = toHtml(f'{self._mean_title}')

            txt_html += toHtml(f'X = {x_val: .2f}, Y = {y_val: .4f}')
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
            txt_html = toHtml(f'Spectrum #{self.spectrumInd}')
        else:
            txt_html = toHtml(f'{self._mean_title}')

        txt_html += toHtml(f'X = {x_val: .2f}, Y = {y_val: .4f}')
        self.txt = TextItem(html=txt_html, anchor=(0, 0))
        ymax = np.max(y)
        self._y = y
        r = self.txtPosRatio
        self.txt.setPos(r * x[-1] + (1 - r) * x[0], 0.95 * ymax)
        self.cross.setData([x_val], [y_val])
        self.addItem(self.txt)

class baselinePlotWidget(SpectraPlotWidget):
    def __init__(self):
        super(baselinePlotWidget, self).__init__()
        self.line.setValue(800)
        self.txt = TextItem('', anchor=(0, 0))
        self.cross = PlotDataItem([800], [0], symbolBrush=(255, 255, 0), symbolPen=(255, 255, 0),
                                  symbol='+',symbolSize=20)
        self.line.sigPositionChanged.connect(self.getMu)
        self._mu = None

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
        ymax = np.max(y)
        self._y = y
        r = self.txtPosRatio
        self.txt.setPos(r * x[-1] + (1 - r) * x[0], 0.95 * ymax)
        self.cross.setData([x_val], [y_val])
        self.addItem(self.txt)

    def getMu(self):
        if self._mu is not None:
            x_val = self.line.value()
            if x_val == 0:
                y_val = 0
            else:
                idx = val2ind(x_val, self._x)
                x_val = self._x[idx]
                y_val = self._mu[idx]
            txt_html = f'<div style="text-align: center"><span style="color: #FFF; font-size: 12pt">\
                                                 X = {x_val: .2f}, Y = {y_val: .4f}</div>'
            self.txt.setHtml(txt_html)
            self.cross.setData([x_val], [y_val])

    def addDataCursor(self, x, y):
        self.addItem(self.line)
        self.addItem(self.cross)
        ymax = np.max(y)
        self.txt.setText('')
        r = self.txtPosRatio
        self.txt.setPos(r * x[-1] + (1 - r) * x[0], 0.95 * ymax)
        self.addItem(self.txt)
        self.getMu()

    def clearAll(self):
        # remove legend
        _legend = self.plotItem.legend
        if (_legend is not None) and (_legend.scene() is not None):
            _legend.scene().removeItem(_legend)
        self.clear()

    def plotBase(self, dataGroup, plotType='raw'):
        """
        make plots for Larch Group object
        :param dataGroup: Larch Group object
        :return:
        """
        # add legend
        self.plotItem.addLegend(offset=(-1, -1))
        x = self._x = dataGroup.energy # self._x, self._mu for getEnergy
        y = self._mu = dataGroup.specTrim
        n = len(x) # array length
        self._y = None  # disable getEnergy func
        if plotType == 'raw':
            self.plotItem.plot(x, y, name='Raw', pen=mkPen('w', width=2))
        elif plotType == 'base':
            self.plotItem.plot(x, y, name='Raw', pen=mkPen('w', width=2))
            self.plotItem.plot(x, dataGroup.rubberBaseline, name='Rubberband baseline', pen=mkPen('g', style=Qt.DotLine, width=2))
            self.plotItem.plot(dataGroup.wav_anchor, dataGroup.spec_anchor, symbol='o', symbolPen='r', symbolBrush=0.5)
        elif plotType == 'rubberband':
            y = self._mu = dataGroup.rubberDebased
            self.plotItem.plot(x, y, name='Rubberband debased', pen=mkPen('r', width=2))
        elif plotType == 'deriv2':
            y = self._mu = dataGroup.rubberDebased # for data cursor
            scale, offset = self.alignTwoCurve(dataGroup.rubberDebased[n//4:n*3//4], dataGroup.deriv2[n//4:n*3//4])
            deriv2Scaled = dataGroup.deriv2 * scale + offset
            ymin, ymax = np.min(y), np.max(y)
            self.getViewBox().setYRange(ymin, ymax, padding=0.1)
            self.plotItem.plot(x, y, name='Rubberband debased', pen=mkPen('r', width=2))
            self.plotItem.plot(x, deriv2Scaled, name='2nd derivative (scaled)', pen=mkPen('g', width=2))
        elif plotType == 'deriv4':
            y = self._mu = dataGroup.rubberDebased
            scale, offset = self.alignTwoCurve(dataGroup.rubberDebased[n//4:n*3//4], dataGroup.deriv4[n//4:n*3//4])
            deriv4Scaled = dataGroup.deriv4 * scale + offset
            ymin, ymax = np.min(y), np.max(y)
            self.getViewBox().setYRange(ymin, ymax, padding=0.1)
            self.plotItem.plot(x, y, name='Rubberband debased', pen=mkPen('r', width=2))
            self.plotItem.plot(x, deriv4Scaled, name='4th derivative (normed, scaled)', pen=mkPen('g', width=2))
        # add infinityline, cross
        self.addDataCursor(x, y)

    def alignTwoCurve(self, y1, y2):
        """
        Align the scale of y2 to that of y1
        :param y1: the main curve
        :param y2: the curve to be aligned
        :return:
        scale: scale factor
        offset: y offset
        """
        y1Range, y2Range = np.max(y1) - np.min(y1), np.max(y2) - np.min(y2)
        scale = y1Range / y2Range
        y = y2 * scale
        offset = np.max(y1) - np.max(y)
        return scale, offset