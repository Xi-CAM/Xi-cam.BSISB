from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
from functools import partial
from xicam.core.data import NonDBHeader
from xicam.BSISB.widgets.mapviewwidget import MapViewWidget
from xicam.BSISB.widgets.spectraplotwidget import SpectraPlotWidget
from xicam.BSISB.widgets.factorizationwidget import FactorizationWidget

from xicam.core import msg
from xicam.plugins import GUIPlugin, GUILayout
from xicam.gui.widgets.tabview import TabView

class MapView(QWidget):
    def __init__(self, header: NonDBHeader = None, field: str = 'primary', ):
        super(MapView, self).__init__()
        self.imageview = MapViewWidget()
        self.spectra = SpectraPlotWidget()
        self.setLayout(QVBoxLayout())
        self.imageview_and_toolbar = QWidget()
        self.centerlayout = QHBoxLayout()
        self.imageview_and_toolbar.setLayout(self.centerlayout)
        self.lefttoolbar = QToolBar()
        self.lefttoolbar.setOrientation(Qt.Vertical)
        self.zoombutton = QToolButton()
        self.zoombutton.setText('Zoom In')
        self.lefttoolbar.addWidget(self.zoombutton)
        self.centerlayout.addWidget(self.lefttoolbar)
        self.centerlayout.addWidget(self.imageview)
        self.layout().addWidget(self.imageview_and_toolbar)
        self.layout().addWidget(self.spectra)

        self.imageview.setHeader(header,field='image')
        self.spectra.setHeader(header,field='spectra')
        self.header = header

        # Connect signals
        self.imageview.sigShowSpectra.connect(self.spectra.showSpectra)
        self.spectra.sigEnergyChanged.connect(self.imageview.setEnergy)

class BSISB(GUIPlugin):
    name = 'BSISB'

    def __init__(self, *args, **kwargs):
        # Data model
        self.headermodel = QStandardItemModel()

        # Selection model
        self.selectionmodel = QItemSelectionModel(self.headermodel)

        self.PCA_widget = FactorizationWidget(self.headermodel, self.selectionmodel)
        self.NMF_widget = FactorizationWidget(self.headermodel, self.selectionmodel)

        self.headermodel.rowsRemoved.connect(partial(self.PCA_widget.setHeader,'spectra'))
        self.headermodel.rowsRemoved.connect(partial(self.NMF_widget.setHeader,'volume'))

        # Setup tabviews
        self.imageview = TabView(self.headermodel, self.selectionmodel, MapView, 'image')

        self.stages = {"BSISB": GUILayout(self.imageview),
                       "PCA": GUILayout(self.PCA_widget),
                       "NMF": GUILayout(self.NMF_widget)}
        super(BSISB, self).__init__(*args, **kwargs)



    def appendHeader(self, header: NonDBHeader, **kwargs):
        item = QStandardItem(header.startdoc.get('sample_name', '????'))
        item.header = header
        self.headermodel.appendRow(item)
        self.headermodel.dataChanged.emit(QModelIndex(), QModelIndex())

        # self.imageview.setHeader(header, field='image')
        # self.spectra.setHeader(header, field='spectra')
        self.PCA_widget.setHeader(field='spectra')
        self.NMF_widget.setHeader(field='volume')

