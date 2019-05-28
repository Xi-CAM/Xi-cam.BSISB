from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy import uic 
import pyqtgraph as pg
from xicam.core.data import NonDBHeader
from xicam.gui.widgets.dynimageview import DynImageView
from xicam.BSISB.widgets.mapviewwidget import MapViewWidget
from xicam.BSISB.widgets.spectraplotwidget import SpectraPlotWidget
from xicam.BSISB.widgets.factorizationwidget import FactorizationWidget

from xicam.core import msg
from xicam.plugins import GUIPlugin, GUILayout

class BSISB(GUIPlugin):
    name = 'BSISB'

    def __init__(self, *args, **kwargs):
        self.imageview = MapViewWidget()
        self.spectra = SpectraPlotWidget()

        self.lefttoolbar = QToolBar()
        self.lefttoolbar.setOrientation(Qt.Vertical)
        self.zoombutton = QToolButton()
        self.zoombutton.setText('Zoom In')
        self.lefttoolbar.addWidget(self.zoombutton)

        self.centerwidget = QWidget()
        self.centerlayout = QHBoxLayout()
        self.centerwidget.setLayout(self.centerlayout)
        self.centerlayout.addWidget(self.lefttoolbar)
        self.centerlayout.addWidget(self.imageview)

        self.factorizationwidget = FactorizationWidget()

        # Connect signals
        self.imageview.sigShowSpectra.connect(self.spectra.showSpectra)
        self.spectra.sigEnergyChanged.connect(self.imageview.setEnergy)

        self.stages = {"BSISB": GUILayout(self.centerwidget, bottom=self.spectra),
                       "NMF": GUILayout(self.factorizationwidget)}
        super(BSISB, self).__init__(*args, **kwargs)

    def appendHeader(self, header: NonDBHeader, **kwargs):
        self.imageview.setHeader(header, field= 'image')
        self.spectra.setHeader(header, field='spectra')
        self.factorizationwidget.setHeader(header, field='spectra')

