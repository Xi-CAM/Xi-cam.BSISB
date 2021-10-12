from functools import partial
from qtpy.QtCore import *
from qtpy.QtGui import *
from xicam.core import msg
from xicam.core.data import NonDBHeader
from xicam.BSISB.widgets.mapconvertwidget import mapToH5
from xicam.BSISB.widgets.mapviewwidget import MapViewWidget
from xicam.BSISB.widgets.spectramaproiwidget import MapView
from xicam.BSISB.widgets.spectraplotwidget import SpectraPlotWidget
from xicam.BSISB.widgets.factorizationwidget import FactorizationWidget
from xicam.BSISB.widgets.preprocesswidget import PreprocessWidget
from xicam.BSISB.widgets.clusteringwidget import ClusteringWidget
from xicam.plugins import GUIPlugin, GUILayout
from xicam.gui.widgets.tabview import TabView

class BSISBTabview(TabView):

    def __init__(self, *args, **kwargs):
        super(BSISBTabview, self).__init__(*args, **kwargs)

    def closeTab(self, i):
        newindex = self.currentIndex()
        if (i <= self.currentIndex()) and (newindex > 0):
            newindex -= 1

        self.removeTab(i)
        self.catalogmodel.removeRow(i)
        self.setCurrentIndex(newindex)
        self.selectionmodel.setCurrentIndex(self.catalogmodel.index(newindex, 0), QItemSelectionModel.Rows
                                            | QItemSelectionModel.ClearAndSelect)

class BSISB(GUIPlugin):
    name = 'BSISB'

    def __init__(self, *args, **kwargs):

        self.mapToH5 = mapToH5()
        # Data model
        self.headermodel = QStandardItemModel()

        # Selection model
        self.selectionmodel = QItemSelectionModel(self.headermodel)
        self.preprocess = PreprocessWidget(self.headermodel, self.selectionmodel)
        self.FA_widget = FactorizationWidget(self.headermodel, self.selectionmodel)
        self.clusterwidget = ClusteringWidget(self.headermodel, self.selectionmodel)

        # update headers list when a tab window is closed
        self.headermodel.rowsRemoved.connect(partial(self.FA_widget.setHeader, 'spectra'))

        # Setup tabviews and update map selection
        self.imageview = BSISBTabview(self.headermodel, self.selectionmodel, MapView, 'image')
        self.imageview.currentChanged.connect(self.updateTab)

        self.stages = {"MapToH5": GUILayout(self.mapToH5),
                       "Image View": GUILayout(self.imageview),
                       "Preprocess": GUILayout(self.preprocess),
                       "Decomposition": GUILayout(self.FA_widget),
                       "Clustering": GUILayout(self.clusterwidget)}
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

        self.preprocess.setHeader(field='spectra')
        self.FA_widget.setHeader(field='spectra')
        self.clusterwidget.setHeader(field='spectra')
        for i in range(4):
            self.FA_widget.roiList[i].sigRegionChangeFinished.connect(self.updateROI)
        self.clusterwidget.roi.sigRegionChangeFinished.connect(self.updateROI)

    def appendSelection(self, sigCase, sigContent):
        # get current widget and append selectedPixels to item
        currentItemIdx = self.imageview.currentIndex()
        if sigCase == 'pixel':
            self.headermodel.item(currentItemIdx).selectedPixels = sigContent
        elif sigCase == 'ROI':
            self.headermodel.item(currentItemIdx).roiState = sigContent
            self.FA_widget.updateRoiMask()
            self.clusterwidget.updateRoiMask()
        elif sigCase == 'autoMask':
            self.headermodel.item(currentItemIdx).maskState = sigContent
            self.FA_widget.updateRoiMask()
            self.clusterwidget.updateRoiMask()
        elif sigCase == 'select':
            self.headermodel.item(currentItemIdx).selectState = sigContent
            self.FA_widget.updateRoiMask()
            self.clusterwidget.updateRoiMask()

    def updateROI(self, roi):
        if self.selectionmodel.hasSelection():
            selectMapIdx = self.selectionmodel.selectedIndexes()[0].row()
        else:
            selectMapIdx = 0
        self.imageview.widget(selectMapIdx).roiMove(roi)

    def updateTab(self):
        # clean up all widgets
        self.preprocess.cleanUp()
