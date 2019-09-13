import os
from functools import partial
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import pickle
import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree, Parameter
import numpy as np
from xicam.core import msg
from xicam.core.data import NonDBHeader
from xicam.BSISB.widgets.uiwidget import MsgBox, uiSaveFile, uiGetFile
from xicam.BSISB.widgets.mapconvertwidget import mapToH5
from xicam.BSISB.widgets.xasimagewidget import xasImageView
from xicam.BSISB.widgets.mapviewwidget import MapViewWidget
from xicam.BSISB.widgets.xasimagewidget import xasSpectraWidget
from xicam.BSISB.widgets.factorizationwidget import FactorizationWidget
from xicam.plugins import GUIPlugin, GUILayout
from xicam.gui.widgets.tabview import TabView

class MapView(QSplitter):
    sigRoiPixels = Signal(object)
    sigRoiState = Signal(object)
    sigAutoMaskState = Signal(object)
    sigSelectMaskState = Signal(object)

    def __init__(self, header: NonDBHeader = None, field: str = 'primary', ):
        super(MapView, self).__init__()
        # layout set up
        self.setOrientation(Qt.Vertical)
        self.imageview = MapViewWidget()
        self.spectra = xasSpectraWidget()

        self.imageview_and_toolbar = QSplitter()
        self.imageview_and_toolbar.setOrientation(Qt.Horizontal)
        self.toolbar_and_param = QSplitter()
        self.toolbar_and_param.setOrientation(Qt.Vertical)
        #define tool bar
        self.toolBar = QWidget()
        self.gridlayout = QGridLayout()
        self.toolBar.setLayout(self.gridlayout)
        #add tool bar buttons
        self.roiBtn = QToolButton()
        self.roiBtn.setText('Manual ROI')
        self.roiBtn.setCheckable(True)
        self.roiMeanBtn = QToolButton()
        self.roiMeanBtn.setText('ROI Mean')
        self.autoMaskBtn = QToolButton()
        self.autoMaskBtn.setText('Auto ROI')
        self.autoMaskBtn.setCheckable(True)
        self.selectMaskBtn = QToolButton()
        self.selectMaskBtn.setText('Mark Select')
        self.selectMaskBtn.setCheckable(True)
        self.saveRoiBtn = QToolButton()
        self.saveRoiBtn.setText('Save ROI')
        self.saveRoiBtn.setCheckable(False)
        self.loadRoiBtn = QToolButton()
        self.loadRoiBtn.setText('Load ROI')
        self.loadRoiBtn.setCheckable(False)
        self.gridlayout.addWidget(self.roiBtn, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.autoMaskBtn, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.selectMaskBtn, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.roiMeanBtn, 1, 1, 1, 1)
        self.gridlayout.addWidget(self.saveRoiBtn, 2, 0, 1, 1)
        self.gridlayout.addWidget(self.loadRoiBtn, 2, 1, 1, 1)

        self.parameterTree = ParameterTree()
        self.parameter = Parameter(name='Threshhold', type='group',
                                   children=[{'name': 'Amide II',
                                              'value': 0,
                                              'type': 'float'},
                                             {'name': "ROI type",
                                              'values': ['+', '-'],
                                              'value': '+',
                                              'type': 'list'},
                                             ])
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

        #setup ROI item
        sideLen = 10
        self.roi = pg.PolyLineROI(positions=[[0, 0], [sideLen, 0], [sideLen, sideLen], [0, sideLen]], closed=True)
        self.imageview.view.addItem(self.roi)
        self.roiInitState = self.roi.getState()
        self.roi.hide()

        #constants
        self.path = os.path.expanduser('~/')
        self.pixSelection = {'ROI': None, 'Mask': None} # init pixel selection dict

        # Connect signals
        self.imageview.sigShowSpectra.connect(self.spectra.showSpectra)
        self.spectra.sigEnergyChanged.connect(self.imageview.setEnergy)
        self.roiBtn.clicked.connect(self.roiBtnClicked)
        self.roi.sigRegionChangeFinished.connect(self.roiSelectPixel)
        self.roi.sigRegionChangeFinished.connect(self.showSelectMask)
        self.sigRoiPixels.connect(self.spectra.getSelectedPixels)
        self.roiMeanBtn.clicked.connect(self.spectra.showMeanSpectra)
        self.autoMaskBtn.clicked.connect(self.showAutoMask)
        self.selectMaskBtn.clicked.connect(self.showSelectMask)
        self.saveRoiBtn.clicked.connect(self.saveRoi)
        self.loadRoiBtn.clicked.connect(self.loadRoi)
        self.parameter.child('Amide II').sigValueChanged.connect(self.showAutoMask)
        self.parameter.child('Amide II').sigValueChanged.connect(self.intersectSelection)
        self.parameter.child('ROI type').sigValueChanged.connect(self.intersectSelection)

    def roiBtnClicked(self):
        self.roiSelectPixel()
        if self.roiBtn.isChecked():
            self.imageview.cross.hide()
            self.roi.show()
            self.sigRoiState.emit((True, self.roi.getState()))
        else:
            self.roi.hide()
            self.roi.setState(self.roiInitState)
            self.sigRoiState.emit((False, self.roi.getState()))

    def saveRoi(self):
        parameterDict = {name: self.parameter[name] for name in self.parameter.names.keys()}
        roiStates = {'roiBtn': self.roiBtn.isChecked(), 'maskBtn': self.autoMaskBtn.isChecked(),
                    'roiState': self.roi.getState(), 'parameter': parameterDict}
        filePath, fileName, canceled = uiSaveFile('Save ROI state', self.path, "Pickle Files (*.pkl)")
        if not canceled:
            with open(filePath + fileName, 'wb') as f:
                pickle.dump(roiStates, f)
            MsgBox(f'ROI state file was saved! \nFile Location: {filePath + fileName}')

    def loadRoi(self):
        filePath, fileName, canceled = uiGetFile('Open ROI state file', self.path, "Pickle Files (*.pkl)")
        if not canceled:
            with open(filePath + fileName, 'rb') as f:
                roiStates = pickle.load(f)
            self.roiBtn.setChecked(roiStates['roiBtn'])
            self.roi.setState(roiStates['roiState'])
            if roiStates['roiBtn']:
                self.roi.show()
            self.autoMaskBtn.setChecked(roiStates['maskBtn'])
            self.selectMaskBtn.setChecked(True)
            self.showSelectMask()
            for k, v in roiStates['parameter'].items():
                self.parameter[k] = v
            MsgBox(f'ROI states were loaded from: \n{filePath + fileName}')
        else:
            return

    def roiMove(self, roi):
        roiState = roi.getState()
        self.roi.setState(roiState)

    def getImgShape(self, imgShape, rc2ind):
        self.row, self.col = imgShape[0], imgShape[1]
        self.rc2ind = rc2ind
        # determine whether spectra data is sparse
        if len(rc2ind) == self.row * self.col:
            self.isDenseImage = True
        else:
            self.isDenseImage = False
        #set up X,Y grid
        x = np.linspace(0, self.col - 1, self.col)
        y = np.linspace(self.row - 1, 0, self.row)
        self.X, self.Y = np.meshgrid(x, y)
        if self.isDenseImage:
            self.fullMap = list(zip(self.Y.ravel(), self.X.ravel()))
        else:
            self.fullMap = list(rc2ind.keys())
        # setup automask item
        self.autoMask = np.ones((self.row, self.col))
        self.autoMaskItem = pg.ImageItem(self.autoMask, axisOrder="row-major", autoLevels=True, opacity=0.3)
        self.imageview.view.addItem(self.autoMaskItem)
        self.autoMaskItem.hide()
        # setup selctmask item to mark selected pixels
        self.selectMask = np.ones((self.row, self.col))
        self.selectMaskItem = pg.ImageItem(self.selectMask, axisOrder="row-major", autoLevels=True, opacity=0.3,
                                      lut = np.array([[0, 0, 0], [255, 0, 0]]))
        self.imageview.view.addItem(self.selectMaskItem)
        self.selectMaskItem.hide()

    def roiSelectPixel(self):
        if self.roiBtn.isChecked():
            #get x,y positions list
            xPos = self.roi.getArrayRegion(self.X, self.imageview.imageItem)
            xPos = np.round(xPos[xPos > 0])
            yPos = self.roi.getArrayRegion(self.Y, self.imageview.imageItem)
            yPos = np.round(yPos[yPos > 0])

            # extract x,y coordinate from selected region
            selectedPixels = list(zip(yPos, xPos))
            self.intersectSelection('ROI', selectedPixels)
            self.sigRoiState.emit((True, self.roi.getState()))
        else:
            self.intersectSelection('ROI', None) # no ROI, select all pixels
            self.sigRoiState.emit((False, self.roi.getState()))

    def showSelectMask(self):
        if self.selectMaskBtn.isChecked():
            # update and show mask
            self.selectMaskItem.setImage(self.selectMask)
            self.selectMaskItem.show()
            self.sigSelectMaskState.emit((True, self.selectMask))
        else:
            self.selectMaskItem.hide()
            self.sigSelectMaskState.emit((False, self.selectMask))

    def showAutoMask(self):
        if self.autoMaskBtn.isChecked():
            # update and show mask
            self.autoMask = self.imageview.makeMask([self.parameter['Amide II']])
            self.autoMaskItem.setImage(self.autoMask)
            self.autoMaskItem.show()
            # select pixels
            mask = self.autoMask.astype(np.bool)
            selectedPixels = list(zip(self.Y[mask], self.X[mask]))
            self.intersectSelection('Mask', selectedPixels)
            self.sigAutoMaskState.emit((True, self.autoMask))
        else:
            self.autoMaskItem.hide()
            self.autoMask[:, :] = 1
            self.intersectSelection('Mask', None) # no mask, select all pixels
            self.sigAutoMaskState.emit((False, self.autoMask))

    def intersectSelection(self, selector, selectedPixels):
        # update pixel selection dict
        if (selector == 'ROI') or (selector == 'Mask'):
            self.pixSelection[selector] = selectedPixels
        # reverse ROI selection
        if (self.parameter['ROI type'] == '-') and (self.pixSelection['ROI'] is not None):
            roi_copy = self.pixSelection['ROI']
            reverseROI = set(self.fullMap) - set(self.pixSelection['ROI'])
            self.pixSelection['ROI'] = list(reverseROI)

        if (self.pixSelection['ROI'] is None) and (self.pixSelection['Mask'] is None):
            if self.isDenseImage:
                self.sigRoiPixels.emit(None) # no ROI, select all pixels
                self.selectMask = np.ones((self.row, self.col))
            else:
                allSelected = np.array(list(self.rc2ind.keys()), dtype='int')
                self.sigRoiPixels.emit(allSelected)  # no ROI, select all sparse rc2ind
                self.selectMask = np.zeros((self.row, self.col))
                self.selectMask[allSelected[:, 0], allSelected[:, 1]] = 1
                self.selectMask = np.flipud(self.selectMask)
            return
        elif self.pixSelection['ROI'] is None:
            allSelected = set(self.pixSelection['Mask']) #de-duplication of pixels
        elif self.pixSelection['Mask'] is None:
            allSelected = set(self.pixSelection['ROI']) #de-duplication of pixels
        else:
            allSelected = set(self.pixSelection['ROI']) & set(self.pixSelection['Mask'])

        if self.isDenseImage:
            allSelected = np.array(list(allSelected), dtype='int')  # convert to array
        else:
            allSelected &= set(self.rc2ind.keys())
            allSelected = np.array(list(allSelected), dtype='int')

        self.selectMask = np.zeros((self.row, self.col))
        if len(allSelected) > 0:
            self.selectMask[allSelected[:, 0], allSelected[:, 1]] = 1
            self.selectMask = np.flipud(self.selectMask)
        self.sigRoiPixels.emit(allSelected)
        # show SelectMask
        self.showSelectMask()
        #recover ROI selection
        if (self.parameter['ROI type'] == '-') and (self.pixSelection['ROI'] is not None):
             self.pixSelection['ROI'] = roi_copy


class BSISB(GUIPlugin):
    name = 'BSISB'

    def __init__(self, *args, **kwargs):

        self.xas = xasImageView()
        self.mapToH5 = mapToH5()
        # Data model
        self.headermodel = QStandardItemModel()

        # Selection model
        self.selectionmodel = QItemSelectionModel(self.headermodel)

        self.FA_widget = FactorizationWidget(self.headermodel, self.selectionmodel)

        # update headers list when a tab window is closed
        self.headermodel.rowsRemoved.connect(partial(self.FA_widget.setHeader, 'spectra'))

        # Setup tabviews and update map selection
        self.imageview = TabView(self.headermodel, self.selectionmodel, MapView, 'image')
        self.imageview.currentChanged.connect(self.updateTab)

        self.stages = {'Xas View': GUILayout(self.xas),
                       "ROI View": GUILayout(self.imageview),
                       "Factor Analysis": GUILayout(self.FA_widget)}
                       # "NMF": GUILayout(self.NMF_widget)}
        super(BSISB, self).__init__(*args, **kwargs)

    def appendHeader(self, header: NonDBHeader, **kwargs):
        # get fileName and update status bar
        fileName = header.startdoc.get('sample_name', '????')
        msg.showMessage(f'Opening {fileName}.h5')
        # init item
        item = QStandardItem(fileName + '_' + str(self.headermodel.rowCount()))
        item.header = header
        item.selectedPixels = None

        self.headermodel.appendRow(item)
        self.headermodel.dataChanged.emit(QModelIndex(), QModelIndex())

        # read out image shape
        imageEvent = next(header.events(fields=['image']))
        imgShape = imageEvent['imgShape']
        rc2ind = imageEvent['rc_index']

        # get current MapView widget
        currentMapView = self.imageview.currentWidget()
        # transmit imgshape to currentMapView
        currentMapView.getImgShape(imgShape, rc2ind)
        # get xy coordinates of ROI selected pixels
        currentMapView.sigRoiPixels.connect(partial(self.appendSelection, 'pixel'))
        currentMapView.sigRoiState.connect(partial(self.appendSelection, 'ROI'))
        currentMapView.sigAutoMaskState.connect(partial(self.appendSelection, 'autoMask'))
        currentMapView.sigSelectMaskState.connect(partial(self.appendSelection, 'select'))

        self.FA_widget.setHeader(field='spectra')
        for i in range(4):
            self.FA_widget.roiList[i].sigRegionChangeFinished.connect(self.updateROI)

    def appendSelection(self, sigCase, sigContent):
        # get current widget and append selectedPixels to item
        currentItemIdx = self.imageview.currentIndex()
        if sigCase == 'pixel':
            self.headermodel.item(currentItemIdx).selectedPixels = sigContent
        elif sigCase == 'ROI':
            self.headermodel.item(currentItemIdx).roiState = sigContent
            self.FA_widget.updateRoiMask()
        elif sigCase == 'autoMask':
            self.headermodel.item(currentItemIdx).maskState = sigContent
            self.FA_widget.updateRoiMask()
        elif sigCase == 'select':
            self.headermodel.item(currentItemIdx).selectState = sigContent
            self.FA_widget.updateRoiMask()

    def updateROI(self, roi):
        if self.selectionmodel.hasSelection():
            selectMapIdx = self.selectionmodel.selectedIndexes()[0].row()
        else:
            selectMapIdx = 0
        self.imageview.widget(selectMapIdx).roiMove(roi)

    def updateTab(self, tabIdx):
        if tabIdx >= 0:
            self.selectionmodel.select(self.headermodel.index(tabIdx, 0), QItemSelectionModel.ClearAndSelect)

