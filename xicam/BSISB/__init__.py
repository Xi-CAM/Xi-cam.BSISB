from functools import partial
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import pyqtgraph as pg
import numpy as np
from xicam.core.data import NonDBHeader
from xicam.BSISB.widgets.mapviewwidget import MapViewWidget
from xicam.BSISB.widgets.spectraplotwidget import SpectraPlotWidget
from xicam.BSISB.widgets.factorizationwidget import FactorizationWidget

from xicam.core import msg
from xicam.plugins import GUIPlugin, GUILayout
from xicam.gui.widgets.tabview import TabView


class MapView(QSplitter):
    sigROIpixels = Signal(object)

    def __init__(self, header: NonDBHeader = None, field: str = 'primary', ):
        super(MapView, self).__init__()
        # layout set up
        self.setOrientation(Qt.Vertical)
        self.imageview = MapViewWidget()
        self.spectra = SpectraPlotWidget()

        self.imageview_and_toolbar = QWidget()
        self.centerlayout = QHBoxLayout()
        self.imageview_and_toolbar.setLayout(self.centerlayout)
        self.lefttoolbar = QToolBar()
        self.lefttoolbar.setOrientation(Qt.Vertical)
        self.roiButton = QToolButton()
        self.roiButton.setText('ROI')
        self.roiButton.setCheckable(True)
        self.lefttoolbar.addWidget(self.roiButton)
        self.centerlayout.addWidget(self.lefttoolbar)
        self.centerlayout.addWidget(self.imageview)

        self.addWidget(self.imageview_and_toolbar)
        self.addWidget(self.spectra)

        # readin header
        self.imageview.setHeader(header, field='image')
        self.spectra.setHeader(header, field='spectra')
        self.header = header

        sideLen = 3
        self.roi = pg.PolyLineROI(positions=[[0, 0], [sideLen, 0], [sideLen, sideLen], [0, sideLen]], closed=True)
        self.imageview.view.addItem(self.roi)
        self.roiState = self.roi.getState()
        self.roi.hide()

        # Connect signals
        self.imageview.sigShowSpectra.connect(self.spectra.showSpectra)
        self.spectra.sigEnergyChanged.connect(self.imageview.setEnergy)
        self.roiButton.clicked.connect(self.roiClicked)
        self.roi.sigRegionChangeFinished.connect(self.selectMapROI)

    def roiClicked(self):
        if self.roiButton.isChecked():
            self.roi.show()
        else:
            self.roi.hide()
            self.roi.setState(self.roiState)

    def getImgShape(self, imgShape):
        self.row, self.col = imgShape[0], imgShape[1]

    def selectMapROI(self):
        if self.roiButton.isChecked():
            x = np.linspace(0, self.col - 1, self.col)
            y = np.linspace(self.row - 1, 0, self.row)
            X, Y = np.meshgrid(x, y)
            xPos = self.roi.getArrayRegion(X, self.imageview.imageItem)
            xPos = np.round(xPos[xPos > 0])
            yPos = self.roi.getArrayRegion(Y, self.imageview.imageItem)
            yPos = np.round(yPos[yPos > 0])

            # extract x,y coordinate from selected region
            selectedPixels = np.zeros((len(xPos), 2), dtype='int')
            for i, (row, col) in enumerate(zip(yPos, xPos)):
                selectedPixels[i, :] = [row, col]
            self.sigROIpixels.emit(selectedPixels)
        else:
            self.sigROIpixels.emit(None)


class BSISB(GUIPlugin):
    name = 'BSISB'

    def __init__(self, *args, **kwargs):
        # Data model
        self.headermodel = QStandardItemModel()

        # Selection model
        self.selectionmodel = QItemSelectionModel(self.headermodel)

        self.PCA_widget = FactorizationWidget(self.headermodel, self.selectionmodel)
        self.NMF_widget = FactorizationWidget(self.headermodel, self.selectionmodel)

        # update headers list when a tab window is closed
        self.headermodel.rowsRemoved.connect(partial(self.PCA_widget.setHeader, 'spectra'))
        self.headermodel.rowsRemoved.connect(partial(self.NMF_widget.setHeader, 'volume'))

        # Setup tabviews
        self.imageview = TabView(self.headermodel, self.selectionmodel, MapView, 'image')

        self.stages = {"BSISB": GUILayout(self.imageview),
                       "PCA": GUILayout(self.PCA_widget),
                       "NMF": GUILayout(self.NMF_widget)}
        super(BSISB, self).__init__(*args, **kwargs)

    def appendHeader(self, header: NonDBHeader, **kwargs):
        # init item
        item = QStandardItem(header.startdoc.get('sample_name', '????'))
        item.header = header
        item.selectedPixels = None

        self.headermodel.appendRow(item)
        self.headermodel.dataChanged.emit(QModelIndex(), QModelIndex())
        # read out image shape
        imageEvent = next(header.events(fields=['image']))
        imgShape = imageEvent['imgShape']

        # get current MapView widget
        currentMapView = self.imageview.currentWidget()
        # transmit imgshape to currentMapView
        currentMapView.getImgShape(imgShape)
        # get xy coordinates of ROI selected pixels
        currentMapView.sigROIpixels.connect(self.appendPixelSelection)

        self.PCA_widget.setHeader(field='spectra')
        self.NMF_widget.setHeader(field='volume')
        # print(imgShape)

    def appendPixelSelection(self, selectedPixels):
        # get current widget and append selectedPixels to item
        currentItemIdx = self.imageview.currentIndex()
        self.headermodel.item(currentItemIdx).selectedPixels = selectedPixels

        # print('map=',currentItemIdx,'\n',selectedPixels,'\n')
