import os
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn import cluster
from functools import partial
from qtpy.QtWidgets import *
from qtpy.QtCore import Qt, QItemSelectionModel, Signal
from qtpy.QtGui import QStandardItemModel, QStandardItem, QFont
from pyqtgraph.parametertree import ParameterTree, Parameter
from xicam.core import msg
from lbl_ir.data_objects.ir_map import val2ind
from xicam.BSISB.widgets.spectraplotwidget import SpectraPlotWidget
from xicam.BSISB.widgets.imshowwidget import SlimImageView
from xicam.BSISB.widgets.uiwidget import MsgBox, YesNoDialog


class ClusteringParameters(ParameterTree):
    sigParamChanged = Signal(object)

    def __init__(self):
        super(ClusteringParameters, self).__init__()

        self.parameter = Parameter(name='params', type='group',
                                   children=[{'name': "Embedding",
                                              'values': ['PCA', 'UMAP'],
                                              'value': 'UMAP',
                                              'type': 'list'},
                                             {'name': "Components",
                                              'value': 2,
                                              'type': 'int'},
                                             {'name': "Neighbors",
                                              'value': 15,
                                              'type': 'int'},
                                             {'name': "Min Dist",
                                              'value': 0.1,
                                              'type': 'float'},
                                             {'name': "Metric",
                                              'values': ['euclidean', 'manhattan', 'correlation'],
                                              'value': 'euclidean',
                                              'type': 'list'},
                                             {'name': "Normalization",
                                              'values': ['L2', 'L1', 'None'],
                                              'value': 'L2',
                                              'type': 'list'},
                                             {'name': "Wavenumber Range",
                                              'value': '400, 4000',
                                              'type': 'str'},
                                             {'name': "Clusters",
                                              'value': 3,
                                              'type': 'int'},
                                             {'name': "X Component",
                                              'values': [1, 2, 3, 4],
                                              'value': 1,
                                              'type': 'list'},
                                             {'name': "Y Component",
                                              'values': [1, 2, 3, 4],
                                              'value': 2,
                                              'type': 'list'}
                                             ])
        self.setParameters(self.parameter, showTop=False)
        self.setIndentation(0)
        # change Fonts
        self.fontSize = 12
        font = QFont("Helvetica [Cronyx]", self.fontSize)
        boldFont = QFont("Helvetica [Cronyx]", self.fontSize, QFont.Bold)
        self.header().setFont(font)
        for item in self.listAllItems():
            if hasattr(item, 'widget'):
                item.setFont(0, boldFont)
                item.widget.setFont(font)
                item.displayLabel.setFont(font)
                item.widget.setMaximumHeight(40)

        # connect signals
        self.parameter.child('Embedding').sigValueChanged.connect(self.updateMethod)
        self.parameter.child('Components').sigValueChanged.connect(self.setComponents)

    def updateMethod(self):
        """
        Toggle parameter menu based on embedding method
        :return: None
        """
        if self.parameter["Embedding"] == 'UMAP':
            self.parameter.child('Neighbors').show()
            self.parameter.child('Min Dist').show()
            self.parameter.child('Metric').show()
            self.parameter.child('Normalization').hide()
        else:
            self.parameter.child('Neighbors').hide()
            self.parameter.child('Min Dist').hide()
            self.parameter.child('Metric').hide()
            self.parameter.child('Normalization').show()

    def setComponents(self):
        N = self.parameter['Components']
        for entry in ['X Component', 'Y Component']:
            param = self.parameter.child(entry)
            param.setLimits(list(range(1, N + 1)))


class ScatterPlotWidget(SpectraPlotWidget):
    def __init__(self):
        super(ScatterPlotWidget, self).__init__(invertX=False)
        self.scene().sigMouseClicked.connect(self.getMousePos)
        self.line.hide()

    def getMousePos(self, event):
        pos = event.pos()
        if self.getViewBox().sceneBoundingRect().contains(pos):
            mousePoint = self.getViewBox().mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()


class ClusteringWidget(QSplitter):
    def __init__(self, headermodel, selectionmodel):
        super(ClusteringWidget, self).__init__()
        self.headermodel = headermodel
        self.mapselectmodel = selectionmodel
        self.selectMapidx = 0

        # split between cluster image and scatter plot
        self.image_and_scatter = QSplitter()
        # split between image&scatter and spec plot, vertical split
        self.leftsplitter = QSplitter()
        self.leftsplitter.setOrientation(Qt.Vertical)
        # split between params, buttons and map list, vertical split
        self.rightsplitter = QSplitter()
        self.rightsplitter.setOrientation(Qt.Vertical)

        self.clusterImage = SlimImageView()
        self.clusterScatterPlot = ScatterPlotWidget()
        self.rawSpecPlot = SpectraPlotWidget()
        self.clusterMeanPlot = SpectraPlotWidget()

        # ParameterTree
        self.parametertree = ClusteringParameters()
        self.parameter = self.parametertree.parameter

        # buttons layout
        self.buttons = QWidget()
        self.buttonlayout = QGridLayout()
        self.buttons.setLayout(self.buttonlayout)
        # set up buttons
        self.fontSize = 12
        font = QFont("Helvetica [Cronyx]", self.fontSize)
        self.computeBtn = QPushButton()
        self.computeBtn.setText('Compute clusters')
        self.computeBtn.setFont(font)
        self.removeBtn = QPushButton()
        self.removeBtn.setText('Remove spectra')
        self.removeBtn.setFont(font)
        # add all buttons
        self.buttonlayout.addWidget(self.computeBtn)
        self.buttonlayout.addWidget(self.removeBtn)

        # Headers listview
        self.headerlistview = QListView()
        self.headerlistview.setModel(headermodel)
        self.headerlistview.setSelectionModel(selectionmodel)  # This might do weird things in the map view?
        self.headerlistview.setSelectionMode(QListView.SingleSelection)

        # assemble widgets
        self.image_and_scatter.addWidget(self.clusterImage)
        self.image_and_scatter.addWidget(self.clusterScatterPlot)
        self.leftsplitter.addWidget(self.image_and_scatter)
        self.leftsplitter.addWidget(self.rawSpecPlot)
        self.leftsplitter.addWidget(self.clusterMeanPlot)
        self.leftsplitter.setSizes([200, 50, 50])
        self.rightsplitter.addWidget(self.parametertree)
        self.rightsplitter.addWidget(self.buttons)
        self.rightsplitter.addWidget(self.headerlistview)
        self.rightsplitter.setSizes([200, 50, 50])
        self.addWidget(self.leftsplitter)
        self.addWidget(self.rightsplitter)
        self.setSizes([500, 100])

        # Connect signals
        self.computeBtn.clicked.connect(self.computeCluster)
        self.removeBtn.clicked.connect(self.removeSpec)

    def computeCluster(self):
        # get current map idx
        if not self.isMapOpen():
            return
        # Select wavenumber region
        wavROIList = []
        for entry in self.parameter['Wavenumber Range'].split(','):
            try:
                wavROIList.append(val2ind(int(entry), self.wavenumbers))
            except:
                continue
        if len(wavROIList) % 2 == 0:
            wavROIList = sorted(wavROIList)
            wavROIidx = []
            for i in range(len(wavROIList) // 2):
                wavROIidx += list(range(wavROIList[2 * i], wavROIList[2 * i + 1] + 1))
        else:
            msg.logMessage('"Wavenumber Range" values must be in pairs', msg.ERROR)
            MsgBox('Clustering computation aborted.', 'error')
            return
        self.wavenumbers_select = self.wavenumbers[wavROIidx]
        self.N_w = len(self.wavenumbers_select)
        # get current dataset
        n_spectra = len(self.data)
        self.dataset = np.zeros((n_spectra, self.N_w))
        for i in range(n_spectra):
            self.dataset[i, :] = self.data[i][wavROIidx]
        # get parameters and compute embedding
        n_components = self.parameter['Components']
        if self.parameter['Embedding'] == 'UMAP':
            n_neighbors = self.parameter['Neighbors']
            metric = self.parameter['Metric']
            min_dist = np.clip(self.parameter['Min Dist'], 0, 1)
            self.umap = UMAP(n_neighbors=n_neighbors,
                              min_dist=min_dist,
                              n_components=n_components,
                              metric=metric)
            self.low_dim = self.umap.fit_transform(self.dataset)
        elif self.parameter['Embedding'] == 'PCA':
            pass

    def removeSpec(self):
        pass

    def setHeader(self, field: str):
        self.headers = [self.headermodel.item(i).header for i in range(self.headermodel.rowCount())]
        self.field = field
        self.wavenumberList = []
        self.imgShapes = []
        self.rc2indList = []
        self.ind2rcList = []
        self.dataSets = []

        # get wavenumbers, imgShapes, rc2ind
        for header in self.headers:
            dataEvent = next(header.events(fields=[field]))
            self.wavenumberList.append(dataEvent['wavenumbers'])
            self.imgShapes.append(dataEvent['imgShape'])
            self.rc2indList.append(dataEvent['rc_index'])
            self.ind2rcList.append(dataEvent['index_rc'])
            # get raw spectra
            data = None
            try:  # spectra datasets
                data = header.meta_array('spectra')
            except IndexError:
                msg.logMessage('Header object contained no frames with field ''{field}''.', msg.ERROR)
            if data is not None:
                self.dataSets.append(data)

    def isMapOpen(self):
        if not self.mapselectmodel.selectedIndexes():  # no map is open
            return False
        else:
            self.selectMapidx = self.mapselectmodel.selectedIndexes()[0].row()
            # get current data
            self.wavenumbers = self.wavenumberList[self.selectMapidx]
            self.rc2ind = self.rc2indList[self.selectMapidx]
            self.ind2rc = self.ind2rcList[self.selectMapidx]
            self.imgShape = self.imgShapes[self.selectMapidx]
            self.data = self.dataSets[self.selectMapidx]
            return True
