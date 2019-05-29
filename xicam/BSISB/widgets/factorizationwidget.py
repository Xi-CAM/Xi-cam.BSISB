from qtpy.QtWidgets import QSplitter, QGridLayout, QWidget
from xicam.core import msg
from .mapviewwidget import MapViewWidget
from pyqtgraph import PlotWidget, ImageView, mkPen
from qtpy.QtCore import Qt
from xicam.core.data import NonDBHeader
from functools import partial
from qtpy.QtCore import Signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from pyqtgraph.parametertree import ParameterTree, Parameter


class FactorizationParameters(ParameterTree):
    sigPCA = Signal(object)

    def __init__(self):
        super(FactorizationParameters, self).__init__()
        self.parameter = Parameter(name='params', type='group',
                                   children=[{'name': "Number of Components",
                                              'value': 4,
                                              'type': 'int'},
                                             {'name': "Calculate",
                                              'type': 'action'},
                                             {'name': "Map 1 Component Index",
                                              'values': [1, 2, 3, 4],
                                              'value': 1,
                                              'type': 'list'},
                                             {'name': "Map 2 Component Index",
                                              'values': [1, 2, 3, 4],
                                              'value': 2,
                                              'type': 'list'},
                                             {'name': "Map 3 Component Index",
                                              'values': [1, 2, 3, 4],
                                              'value': 3,
                                              'type': 'list'},
                                             {'name': "Map 4 Component Index",
                                              'values': [1, 2, 3, 4],
                                              'value': 4,
                                              'type': 'list'},
                                             {'name': "Save results",
                                              'type': 'action'},
                                             ])

        self.setParameters(self.parameter, showTop=False)

        self.parameter.child('Calculate').sigActivated.connect(self.calculate)
        self.parameter.child('Save results').sigActivated.connect(self.saveResults)
        self.parameter.child('Number of Components').sigValueChanged.connect(self.setNumComponents)

    def setHeader(self, header: NonDBHeader, field: str):
        self.header = header
        self.field = field

        spectraEvent = next(header.events(fields=['spectra']))
        self.wavenumbers = spectraEvent['wavenumbers']
        self.imgShape = spectraEvent['imgShape']

        data = None
        try:
            data = header.meta_array(field)
        except IndexError:
            msg.logMessage('Header object contained no frames with field ''{field}''.', msg.ERROR)

        if data is not None:
            # kwargs['transform'] = QTransform(1, 0, 0, -1, 0, data.shape[-2])
            self._data = data

    def setNumComponents(self):
        N = self.parameter['Number of Components']

        for i in range(4):
            param = self.parameter.child(f'Map {i + 1} Component Index')
            param.setLimits(list(range(1, N+1)))

    def calculate(self):
        N = self.parameter['Number of Components']

        if hasattr(self, '_data'):
        #mean center
            ss = StandardScaler(with_std=False)
            n_spectra = len(self._data)
            n_feature = len(self._data[0])
            data = np.zeros((n_spectra, n_feature))
            for i in range(n_spectra):
                data[i,:] = self._data[i]
            data_centered = ss.fit_transform(data)
            # Do PCA
            self.pca = PCA(n_components=N)
            self.pca.fit(data_centered)
            self.data_pca = self.pca.transform(data_centered)
            # emit PCA and transformed data : data_PCA
            self.sigPCA.emit((self.pca, self.data_pca))
        else:
            print('data matrix not available.')

    def saveResults(self):
        if hasattr(self, 'pca') and hasattr(self, 'data_pca'):
            df_pca_components = pd.DataFrame(self.pca.components_, columns=self.wavenumbers)
            df_data_pca = pd.DataFrame(self.data_pca)
            df_pca_components.to_csv('pca_components.csv')
            df_data_pca.to_csv('data_pca.csv')
            print('PCA components successfully saved!')
        else:
            print('No PCA components available.')


class FactorizationWidget(QSplitter):
    def __init__(self):
        super(FactorizationWidget, self).__init__()

        self.gridwidget = QWidget()
        self.gridlayout = QGridLayout()
        self.gridwidget.setLayout(self.gridlayout)
        self.display = QSplitter()

        self.componentSpectra = PlotWidget()
        self._plotLegends = self.componentSpectra.addLegend()
        self._colors = ['r', 'g', 'b', 'y', 'c', 'm', 'w'] # color for plots
        self.componentSpectra.getViewBox().invertX(True)

        # self.spectraROI = PlotWidget()
        self.NWimage = ImageView()
        self.NEimage = ImageView()
        self.SWimage = ImageView()
        self.SEimage = ImageView()
        self._imageDict = {0 : 'NWimage', 1 : 'NEimage', 2 : 'SWimage', 3 : 'SEimage'}
        for i in range(4):
            eval('self.' + self._imageDict[i] + '.setPredefinedGradient("flame")')
            eval('self.' + self._imageDict[i] + '.view.invertY(False)')
            eval('self.' + self._imageDict[i] + '.imageItem.setOpts(axisOrder="row-major")')

        self.parametertree = FactorizationParameters()
        self.parameter = self.parametertree.parameter
        self.parametertree.sigPCA.connect(self.showComponents)
        for i in range(4):
            self.parameter.child(f'Map {i + 1} Component Index').sigValueChanged.connect(partial(self.updateComponents, i))

        self.addWidget(self.display)
        self.addWidget(self.parametertree)
        self.display.addWidget(self.gridwidget)
        self.display.addWidget(self.componentSpectra)
        # self.display.addWidget(self.spectraROI)
        self.gridlayout.addWidget(self.NWimage, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.NEimage, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.SWimage, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.SEimage, 1, 1, 1, 1)

        self.setOrientation(Qt.Horizontal)
        self.display.setOrientation(Qt.Vertical)

    def updateComponents(self, i):
        # i is imageview number
        # component_index is the PCA component index
        component_index = self.parameter[f'Map {i + 1} Component Index']
        # update scoreplots on view i
        if hasattr(self, '_data_pca'):
            img = self._data_pca[:, component_index-1].reshape(self.imgShape[0], self.imgShape[1])
            eval('self.' + self._imageDict[i] + '.setImage(img=img)')

        # update PCA components
        if hasattr(self, '_plots'):
            # update plots
            name = 'PCA' + str(component_index)
            self._plots[i].setData(self.wavenumbers, self._pca.components_[component_index - 1, :], name=name)
            # update legend label
            sample, label = self._plotLegends.items[i]
            label.setText(name)

    def showComponents(self, pca_obj):
        # clear plots and legends
        self.componentSpectra.clear()
        for sample, label in self._plotLegends.items[:]:
            self._plotLegends.removeItem(label.text)

        self._pca, self._data_pca = pca_obj[0], pca_obj[1]

        self._plots = []
        for i in range(4):
            component_index = self.parameter[f'Map {i + 1} Component Index']
            # show loading plots
            name = 'PCA' + str(component_index)
            tmp = self.componentSpectra.plot(self.wavenumbers, self._pca.components_[component_index-1,:], name=name,
                                             pen=mkPen(self._colors[i], width=2))
            self._plots.append(tmp)
            # show score plots
            img = self._data_pca[:, component_index-1].reshape(self.imgShape[0], self.imgShape[1])
            eval('self.' + self._imageDict[i] + '.setImage(img=img)')

        #update the last image and loading plots as a recalculation complete signal
        N = self.parameter['Number of Components']
        self.parameter.child(f'Map 4 Component Index').setValue(N)

    def setHeader(self, header: NonDBHeader, field: str):
        self.header = header
        self.field = field
        # get wavenumbers
        spectraEvent = next(header.events(fields=['spectra']))
        self.wavenumbers = spectraEvent['wavenumbers']
        self.imgShape = spectraEvent['imgShape']

        self.parametertree.setHeader(header=header, field=field)

