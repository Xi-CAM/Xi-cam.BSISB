from functools import partial
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import pyqtgraph as pg
import numpy as np
import sys
from xicam.core.data import NonDBHeader
from xicam.BSISB.widgets.mapviewwidget import MapViewWidget
from xicam.BSISB.widgets.spectraplotwidget import SpectraPlotWidget
from xicam.BSISB.widgets.factorizationwidget import FactorizationWidget

from xicam.plugins import GUIPlugin, GUILayout
from xicam.gui.widgets.tabview import TabView
from pyqtgraph.parametertree import ParameterTree, Parameter



class MapView(QSplitter):
    sigROIpixels = Signal(object)

    def __init__(self, header: NonDBHeader = None, field: str = 'primary', ):
        super(MapView, self).__init__()
        # layout set up
        self.setOrientation(Qt.Vertical)
        self.imageview = MapViewWidget()
        self.spectra = SpectraPlotWidget()

        self.imageview_and_toolbar = QSplitter()
        self.imageview_and_toolbar.setOrientation(Qt.Horizontal)
        self.toolbar_and_param = QSplitter()
        self.toolbar_and_param.setOrientation(Qt.Vertical)
        #define tool bar
        self.toolBar = QWidget()
        self.gridlayout = QGridLayout()
        self.toolBar.setLayout(self.gridlayout)
        # self.toolBar.setOrientation(Qt.Vertical)
        #add tool bar buttons
        self.roiButton = QToolButton()
        self.roiButton.setText('manROI')
        self.roiButton.setCheckable(True)
        self.roiMeanButton = QToolButton()
        self.roiMeanButton.setText('ROImean')
        self.maskButton = QToolButton()
        self.maskButton.setText('autoROI')
        self.maskButton.setCheckable(True)
        self.gridlayout.addWidget(self.roiButton, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.roiMeanButton, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.maskButton, 0, 1, 1, 1)

        self.parameterTree = ParameterTree()
        self.parameter = Parameter(name='Threshhold', type='group',
                                   children=[{'name': 'Amide II',
                                              'value': 0,
                                              'type': 'float'}])
        self.parameter.child('Amide II').setOpts(step=0.1)
        self.parameterTree.setParameters(self.parameter, showTop=False)
        self.parameterTree.setHeaderLabels(['Params','Value'])
        self.parameterTree.setIndentation(0)

        # Assemble widgets
        self.toolbar_and_param.addWidget(self.toolBar)
        self.toolbar_and_param.addWidget(self.parameterTree)
        self.toolbar_and_param.setSizes([1000, 1]) #adjust initial splitter size
        self.imageview_and_toolbar.addWidget(self.toolbar_and_param)
        self.imageview_and_toolbar.addWidget(self.imageview)
        self.imageview_and_toolbar.setSizes([1, 1000])#adjust initial splitter size
        self.addWidget(self.imageview_and_toolbar)
        self.addWidget(self.spectra)
        self.setSizes([1000, 1000])  # adjust initial splitter size

        # readin header
        self.imageview.setHeader(header, field='image')
        self.spectra.setHeader(header, field='spectra')
        self.header = header
        # init pixel selection dict
        self.allSelection = {'ROI': None, 'Mask': None}

        #setup ROI item
        sideLen = 10
        self.roi = pg.PolyLineROI(positions=[[0, 0], [sideLen, 0], [sideLen, sideLen], [0, sideLen]], closed=True)
        self.imageview.view.addItem(self.roi)
        self.roiState = self.roi.getState()
        self.roi.hide()

        # Connect signals
        self.imageview.sigShowSpectra.connect(self.spectra.showSpectra)
        self.spectra.sigEnergyChanged.connect(self.imageview.setEnergy)
        self.roiButton.clicked.connect(self.roiClicked)
        self.roi.sigRegionChangeFinished.connect(self.selectMapROI)
        self.sigROIpixels.connect(self.spectra.getSelectedPixels)
        self.roiMeanButton.clicked.connect(self.spectra.showMeanSpectra)
        self.maskButton.clicked.connect(self.showAutoMask)
        self.parameter.child('Amide II').sigValueChanged.connect(self.showAutoMask)

    def roiClicked(self):
        if self.roiButton.isChecked():
            self.imageview.arrow.hide()
            self.roi.show()
        else:
            self.roi.hide()
            self.roi.setState(self.roiState)
        self.selectMapROI()

    def getImgShape(self, imgShape):
        self.row, self.col = imgShape[0], imgShape[1]
        #set up X,Y grid
        x = np.linspace(0, self.col - 1, self.col)
        y = np.linspace(self.row - 1, 0, self.row)
        self.X, self.Y = np.meshgrid(x, y)
        # setup automask item
        self.mask = np.ones((self.row, self.col))
        self.autoMask = pg.ImageItem(self.mask, axisOrder="row-major", autoLevels=True, opacity=0.3)
        self.imageview.view.addItem(self.autoMask)
        self.autoMask.hide()

    def selectMapROI(self):
        if self.roiButton.isChecked():
            #get x,y positions list
            xPos = self.roi.getArrayRegion(self.X, self.imageview.imageItem)
            xPos = np.round(xPos[xPos > 0])
            yPos = self.roi.getArrayRegion(self.Y, self.imageview.imageItem)
            yPos = np.round(yPos[yPos > 0])

            # extract x,y coordinate from selected region
            selectedPixels = list(zip(yPos, xPos))
            self.intersectSelection('ROI', selectedPixels)
        else:
            self.intersectSelection('ROI', None) # no ROI, select all pixels

    def showAutoMask(self):
        if self.maskButton.isChecked():
            # update and show mask
            self.mask = self.imageview.makeMask([self.parameter['Amide II']])
            self.autoMask.setImage(self.mask)
            self.autoMask.show()
            # select pixels
            mask = self.mask.astype(np.bool)
            selectedPixels = list(zip(self.Y[mask], self.X[mask]))
            self.intersectSelection('Mask', selectedPixels)
        else:
            self.autoMask.hide()
            self.mask[:, :] = 1
            self.intersectSelection('Mask', None) # no mask, select all pixels

    def intersectSelection(self, selector, selectedPixels):
        # update pixel selection dict
        self.allSelection[selector] = selectedPixels
        if (self.allSelection['ROI'] is None) and (self.allSelection['Mask'] is None):
            self.sigROIpixels.emit(None) # no ROI, select all pixels
            return
        elif self.allSelection['ROI'] is None:
            allSelected = set(self.allSelection['Mask']) #de-duplication of pixels
        elif self.allSelection['Mask'] is None:
            allSelected = set(self.allSelection['ROI']) #de-duplication of pixels
        else:
            allSelected = set(self.allSelection['ROI']) & set(self.allSelection['Mask'])

        allSelected = np.array(list(allSelected), dtype='int')  # convert to array
        self.sigROIpixels.emit(allSelected)


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
        item = QStandardItem(header.startdoc.get('sample_name', '????') + '_' + str(self.headermodel.rowCount()))
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
