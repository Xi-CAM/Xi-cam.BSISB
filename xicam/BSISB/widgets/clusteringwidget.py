from functools import partial
import numpy as np
from lbl_ir.data_objects.ir_map import val2ind
from matplotlib import cm
from pyqtgraph import TextItem, mkBrush, mkPen
from pyqtgraph.parametertree import ParameterTree, Parameter
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import *
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from umap import UMAP
from xicam.BSISB.widgets.mapviewwidget import MapViewWidget, toHtml
from xicam.BSISB.widgets.spectraplotwidget import SpectraPlotWidget
from xicam.BSISB.widgets.uiwidget import MsgBox
from xicam.core import msg


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
                                              'value': 3,
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
                                              'values': [1, 2, 3],
                                              'value': 1,
                                              'type': 'list'},
                                             {'name': "Y Component",
                                              'values': [1, 2, 3],
                                              'value': 2,
                                              'type': 'list'}
                                             ])
        self.setParameters(self.parameter, showTop=False)
        self.setIndentation(0)
        self.parameter.child('Normalization').hide()
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
        for entry in ['Components', 'Clusters', 'X Component', 'Y Component']:
            self.parameter.child(entry).sigValueChanged.connect(partial(self.updateClusterParams, entry))

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

    def updateClusterParams(self, name):
        self.sigParamChanged.emit(name)


class ScatterPlotWidget(SpectraPlotWidget):
    sigScatterClicked = Signal(object)

    def __init__(self):
        super(ScatterPlotWidget, self).__init__(invertX=False, linePos=0)
        # self.scene().sigMouseClicked.connect(self.setCrossPos)
        self.scene().sigMouseMoved.connect(self.moveCrossPos)
        self.line.hide()
        self.scatterData = None
        self.nbr = None

    def setCrossPos(self, event):
        pos = event.pos()
        if (self.getViewBox().sceneBoundingRect().contains(pos)) and (self.scatterData is not None):
            mousePoint = self.getViewBox().mapToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            _, ind = self.nbr.kneighbors(np.array([[x, y]]))
            self.addItem(self.cross)
            self.cross.setData(self.scatterData[ind[0], 0], self.scatterData[ind[0], 1])
            self.sigScatterClicked.emit(ind[0, 0])

    def moveCrossPos(self, pos):
        if (self.getViewBox().sceneBoundingRect().contains(pos)) and (self.scatterData is not None):
            mousePoint = self.getViewBox().mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            _, ind = self.nbr.kneighbors(np.array([[x, y]]))
            self.addItem(self.cross)
            self.cross.setData(self.scatterData[ind[0], 0], self.scatterData[ind[0], 1])
            self.sigScatterClicked.emit(ind[0, 0])

    def clickFromImage(self, ind):
        self.cross.setData([self.scatterData[ind, 0]], [self.scatterData[ind, 1]])

    def getNN(self):
        msg.showMessage('Training NearestNeighbors model in scatter plot.')
        self.nbr = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.scatterData)
        msg.showMessage('NearestNeighbors model training is finished.')


class ClusterSpectraWidget(SpectraPlotWidget):
    def __init__(self):
        super(ClusterSpectraWidget, self).__init__(txtPosRatio=0.35)
        self._x = None
        self.ymax, self.zmax = 0, 100
        self._plots = []

    def getEnergy(self):
        if self._y is not None:
            x_val = self.line.value()
            idx = val2ind(x_val, self._x)
            x_val = self._x[idx]
            y_val = self._y[idx]
            txt_html = toHtml(f'X = {x_val: .2f}, Y = {y_val: .4f}')
            self.txt.setHtml(txt_html)
            self.cross.setData([x_val], [y_val])

    def setColors(self, colorLUT):
        self.colorLUT = colorLUT.copy()
        self.colorLUT[0, :] = np.ones(3) * 255

    def plotClusterSpectra(self):
        if self._data is not None:
            if self._plots:
                self.clearAll()
                self._plots = []
            self.ymax = 0
            self.plotItem.addLegend(offset=(1, 1))
            self.nSpectra = len(self._data)
            for i in range(self.nSpectra):
                name = 'Cluster #' + str(i)
                tmp = self.plot(self.wavenumbers, self._data[i], pen=mkPen(self.colorLUT[i], width=2), name=name)
                tmp.curve.setClickable(True)
                tmp.curve.sigClicked.connect(partial(self.curveHighLight, i))
                self._plots.append(tmp)
            self.addItem(self.txt)

    def curveHighLight(self, k):
        for i in range(self.nSpectra):
            if i == k:
                self._plots[i].setPen(mkPen(self.colorLUT[k], width=6))
                self._plots[i].setZValue(50)
            else:
                self._plots[i].setPen(mkPen(self.colorLUT[i], width=2))
                self._plots[i].setZValue(0)
        self._x, self._y = self._plots[k].getData()
        ymin, ymax = np.min(self._y), np.max(self._y)
        self.getViewBox().setYRange(ymin, ymax, padding=0.1)
        r = self.txtPosRatio
        self.txt.setPos(r * self._x[-1] + (1 - r) * self._x[0], ymax)
        self.getEnergy()

    def plot(self, x, y, *args, **kwargs):
        # set up infinity line and get its position
        plot_item = self.plotItem.plot(x, y, *args, **kwargs)
        self.addItem(self.line)
        self.addItem(self.cross)
        x_val = self.line.value()
        idx = val2ind(x_val, x)
        x_val = x[idx]
        y_val = y[idx]
        txt_html = toHtml(f'X = {x_val: .2f}, Y = {y_val: .4f}')
        self.txt = TextItem(html=txt_html, anchor=(0, 0))
        self.txt.setZValue(self.zmax - 1)
        self.cross.setData([x_val], [y_val])
        self.cross.setZValue(self.zmax)
        ymax = np.max(y)
        if ymax > self.ymax:
            self.ymax = ymax
        self._x, self._y = x, y
        r = self.txtPosRatio
        self.txt.setPos(r * x[-1] + (1 - r) * x[0], self.ymax)
        return plot_item


class ClusteringWidget(QSplitter):
    def __init__(self, headermodel, selectionmodel):
        super(ClusteringWidget, self).__init__()
        self.headermodel = headermodel
        self.mapselectmodel = selectionmodel
        # init some values
        self.selectMapidx = 0
        self.embedding = None
        self.labels = None
        self.mean_spectra = None

        # split between cluster image and scatter plot
        self.image_and_scatter = QSplitter()
        # split between image&scatter and spec plot, vertical split
        self.leftsplitter = QSplitter()
        self.leftsplitter.setOrientation(Qt.Vertical)
        # split between params, buttons and map list, vertical split
        self.rightsplitter = QSplitter()
        self.rightsplitter.setOrientation(Qt.Vertical)

        self.clusterImage = MapViewWidget()
        self.clusterScatterPlot = ScatterPlotWidget()
        self.rawSpecPlot = SpectraPlotWidget()
        self.clusterMeanPlot = ClusterSpectraWidget()

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
        # add all buttons
        self.buttonlayout.addWidget(self.computeBtn)

        # Headers listview
        self.headerlistview = QListView()
        self.headerlistview.setModel(headermodel)
        self.headerlistview.setSelectionModel(selectionmodel)  # This might do weird things in the map view?
        self.headerlistview.setSelectionMode(QListView.SingleSelection)
        # add title to list view
        self.mapListWidget = QWidget()
        self.listLayout = QVBoxLayout()
        self.mapListWidget.setLayout(self.listLayout)
        mapListTitle = QLabel('Maps list')
        mapListTitle.setFont(font)
        self.listLayout.addWidget(mapListTitle)
        self.listLayout.addWidget(self.headerlistview)

        # assemble widgets
        self.image_and_scatter.addWidget(self.clusterImage)
        self.image_and_scatter.addWidget(self.clusterScatterPlot)
        self.leftsplitter.addWidget(self.image_and_scatter)
        self.leftsplitter.addWidget(self.rawSpecPlot)
        self.leftsplitter.addWidget(self.clusterMeanPlot)
        self.leftsplitter.setSizes([200, 50, 50])
        self.rightsplitter.addWidget(self.parametertree)
        self.rightsplitter.addWidget(self.buttons)
        self.rightsplitter.addWidget(self.mapListWidget)
        self.rightsplitter.setSizes([300, 50, 50])
        self.addWidget(self.leftsplitter)
        self.addWidget(self.rightsplitter)
        self.setSizes([500, 100])

        # Connect signals
        self.computeBtn.clicked.connect(self.computeEmbedding)
        self.clusterImage.sigShowSpectra.connect(self.rawSpecPlot.showSpectra)
        self.clusterImage.sigShowSpectra.connect(self.showClusterMean)
        self.clusterImage.sigShowSpectra.connect(self.clusterScatterPlot.clickFromImage)
        self.clusterScatterPlot.sigScatterClicked.connect(self.rawSpecPlot.showSpectra)
        self.clusterScatterPlot.sigScatterClicked.connect(self.showClusterMean)
        self.clusterScatterPlot.sigScatterClicked.connect(self.setImageCross)
        self.parametertree.sigParamChanged.connect(self.updateClusterParams)
        self.mapselectmodel.selectionChanged.connect(self.updateMap)

    def computeEmbedding(self):
        # get current map idx
        if not self.isMapOpen():
            return
        msg.showMessage('Compute embedding.')
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
                             metric=metric,
                             random_state=0)
            self.embedding = self.umap.fit_transform(self.dataset)
        elif self.parameter['Embedding'] == 'PCA':
            # normalize and mean center
            if self.parameter['Normalization'] == 'L1':  # normalize
                data_norm = Normalizer(norm='l1').fit_transform(self.dataset)
            elif self.parameter['Normalization'] == 'L2':
                data_norm = Normalizer(norm='l2').fit_transform(self.dataset)
            else:
                data_norm = self.dataset
            # subtract mean
            data_centered = StandardScaler(with_std=False).fit_transform(data_norm)
            # Do PCA
            self.PCA = PCA(n_components=n_components)
            self.PCA.fit(data_centered)
            self.embedding = self.PCA.transform(data_centered)
        # save embedding to standardModelItem
        self.item.embedding = self.embedding
        # update cluster map
        self.computeCluster()

    def computeCluster(self):
        # check if embeddings exist
        if self.embedding is None:
            return
        msg.showMessage('Compute clusters.')
        # get num of clusters
        n_clusters = self.parameter['Clusters']
        # set colorLUT
        self.colorLUT = cm.get_cmap('viridis', n_clusters).colors[:, :3] * 255
        # compute cluster
        cluster_object = KMeans(n_clusters=n_clusters, random_state=0).fit(self.embedding)
        self.labels = cluster_object.labels_
        # update cluster image
        self.cluster_map = self.labels.reshape(self.imgShape[0], self.imgShape[1])
        self.cluster_map = np.flipud(self.cluster_map)
        self.clusterImage.setImage(self.cluster_map, levels=[0, n_clusters - 1])
        # self.clusterImage.setImage(self.cluster_map)
        self.clusterImage._image = self.cluster_map
        self.clusterImage.rc2ind = self.rc2ind
        self.clusterImage.row, self.clusterImage.col = self.imgShape[0], self.imgShape[1]
        self.clusterImage.txt.setPos(self.clusterImage.col, 0)
        self.clusterImage.cross.show()
        # update cluster mean
        mean_spectra = []
        for ii in range(n_clusters):
            sel = (self.labels == ii)
            this_mean = np.mean(self.dataset[sel, :], axis=0)
            mean_spectra.append(this_mean)
        self.mean_spectra = np.vstack(mean_spectra)
        self.clusterMeanPlot.setColors(self.colorLUT)
        self.clusterMeanPlot._data = self.mean_spectra
        self.clusterMeanPlot.wavenumbers = self.wavenumbers_select
        self.clusterMeanPlot.plotClusterSpectra()
        # update scatter plot
        self.updateScatterPlot()

    def updateScatterPlot(self):
        if (self.embedding is None) or (self.labels is None):
            return
        # get scatter x, y values
        self.clusterScatterPlot.scatterData = self.embedding[:,
                                              [self.parameter['X Component'] - 1, self.parameter['Y Component'] - 1]]
        # get colormapings
        brushes = [mkBrush(self.colorLUT[x, :]) for x in self.labels]
        # make plots
        if hasattr(self, 'scatterPlot'):
            self.clusterScatterPlot.plotItem.clearPlots()
        self.scatterPlot = self.clusterScatterPlot.plotItem.plot(self.clusterScatterPlot.scatterData, pen=None,
                                                                 symbol='o', symbolBrush=brushes)
        self.clusterScatterPlot.getViewBox().autoRange(padding=0.1)
        self.clusterScatterPlot.getNN()

    def updateClusterParams(self, name):
        if name == 'Components':
            self.computeEmbedding()
        elif name == 'Clusters':
            self.computeCluster()
        elif name in ['X Component', 'Y Component']:
            self.updateScatterPlot()

    def updateMap(self):
        # get current map idx
        if not self.mapselectmodel.selectedIndexes():  # no map is open
            return
        else:
            self.selectMapidx = self.mapselectmodel.selectedIndexes()[0].row()
            # get current item
            self.item = self.headermodel.item(self.selectMapidx)
            if hasattr(self.item, 'embedding'):
                # load embedding
                self.embedding = self.item.embedding
                self.computeCluster()
            else:
                # reset custer image and plots
                self.cleanUp()

    def showClusterMean(self, i):
        if self.mean_spectra is None:
            return
        self.clusterMeanPlot.curveHighLight(self.labels[i])

    def setImageCross(self, ind):
        row, col = self.ind2rc[ind]
        # update cross
        self.clusterImage.cross.setData([col + 0.5], [self.imgShape[0] - row - 0.5])
        # update text
        self.clusterImage.txt.setHtml(toHtml(f'Point: #{ind}', size=8)
                                      + toHtml(f'X: {col}', size=8)
                                      + toHtml(f'Y: {row}', size=8)
                                      + toHtml(f'Val: {self.clusterImage._image[self.imgShape[0] - row - 1, col] :d}',
                                               size=8))

    def cleanUp(self):
        if hasattr(self, 'imgShape'):
            img = np.zeros((self.imgShape[0], self.imgShape[1]))
            self.clusterImage.setImage(img=img)
        if hasattr(self, 'scatterPlot'):
            self.clusterScatterPlot.plotItem.clearPlots()
            self.clusterScatterPlot.scatterData = None
        self.rawSpecPlot.clearAll()
        self.rawSpecPlot._data = None
        self.clusterMeanPlot.clearAll()
        self.clusterMeanPlot._data = None

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
        self.cleanUp()

    def isMapOpen(self):
        if not self.mapselectmodel.selectedIndexes():  # no map is open
            return False
        else:
            self.selectMapidx = self.mapselectmodel.selectedIndexes()[0].row()
            # get current data
            self.item = self.headermodel.item(self.selectMapidx)
            self.currentHeader = self.headers[self.selectMapidx]
            self.wavenumbers = self.wavenumberList[self.selectMapidx]
            self.rc2ind = self.rc2indList[self.selectMapidx]
            self.ind2rc = self.ind2rcList[self.selectMapidx]
            self.imgShape = self.imgShapes[self.selectMapidx]
            self.data = self.dataSets[self.selectMapidx]
            self.rawSpecPlot.setHeader(self.currentHeader, 'spectra')
            return True
