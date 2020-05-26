from qtpy.QtWidgets import *
from xicam.core import msg
from pyqtgraph import PlotWidget, PlotDataItem, TextItem, mkPen, InfiniteLine, ImageItem, PolyLineROI
from qtpy.QtCore import Qt, QItemSelectionModel
from qtpy.QtGui import QStandardItemModel, QFont
from functools import partial
from qtpy.QtCore import Signal
from pymcr.mcr import McrAR
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xicam.BSISB.widgets.uiwidget import MsgBox
from xicam.BSISB.widgets.imshowwidget import SlimImageView
from lbl_ir.data_objects.ir_map import val2ind
from lbl_ir.tasks.preprocessing import data_prep
from lbl_ir.tasks.NMF.multi_set_analyses import aggregate_data
from lbl_ir.io_tools import read_map

from pyqtgraph.parametertree import ParameterTree, Parameter


class FactorizationParameters(ParameterTree):
    sigPCA = Signal(object)

    def __init__(self, headermodel: QStandardItemModel, selectionmodel: QItemSelectionModel):
        super(FactorizationParameters, self).__init__()
        self.headermodel = headermodel
        self.selectionmodel = selectionmodel

        self.parameter = Parameter(name='params', type='group',
                                   children=[{'name': "Method",
                                              'values': ['PCA', 'NMF', 'MCR'],
                                              'value': 'PCA',
                                              'type': 'list'},
                                             {'name': "Components",
                                              'value': 4,
                                              'type': 'int'},
                                             {'name': "Calculate",
                                              'type': 'action'},
                                             {'name': "Map 1 Component",
                                              'values': [1, 2, 3, 4],
                                              'value': 1,
                                              'type': 'list'},
                                             {'name': "Map 2 Component",
                                              'values': [1, 2, 3, 4],
                                              'value': 2,
                                              'type': 'list'},
                                             {'name': "Map 3 Component",
                                              'values': [1, 2, 3, 4],
                                              'value': 3,
                                              'type': 'list'},
                                             {'name': "Map 4 Component",
                                              'values': [1, 2, 3, 4],
                                              'value': 4,
                                              'type': 'list'},
                                             {'name': "Wavenumber Range",
                                              'value': '800,1800',
                                              'type': 'str'},
                                             {'name': "Normalization",
                                              'values': ['L2', 'L1', 'None'],
                                              'value': 'L2',
                                              'type': 'list'},
                                             {'name': "C regressor",
                                              'values': ['OLS', 'NNLS'],
                                              'value': 'OLS',
                                              'type': 'list'},
                                             {'name': "Save results",
                                              'type': 'action'}
                                             ])

        self.setParameters(self.parameter, showTop=False)
        self.setIndentation(0)
        #constants
        self.method = 'PCA'
        self.field = 'spectra'
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
        self.parameter.child('Calculate').sigActivated.connect(self.calculate)
        self.parameter.child('Save results').sigActivated.connect(self.saveResults)
        self.parameter.child('Components').sigValueChanged.connect(self.setNumComponents)
        self.parameter.child('Method').sigValueChanged.connect(self.setMethod)
        self.parameter.child('C regressor').hide()

    def setMethod(self):
        if self.parameter['Method'] == 'PCA':
            self.method = 'PCA'
            self.field = 'spectra'
            self.parameter.child('Normalization').setToDefault()
            self.parameter.child('Normalization').show()
            self.parameter.child('C regressor').hide()
        elif self.parameter['Method'] == 'NMF':
            self.method = 'NMF'
            self.field = 'volume'
            self.parameter.child('Normalization').hide()
            self.parameter.child('C regressor').hide()
        elif self.parameter['Method'] == 'MCR':
            self.method = 'MCR'
            self.field = 'spectra'
            self.parameter.child('Normalization').hide()
            self.parameter.child('C regressor').show()

    def setHeader(self, wavenumbers, imgShapes, rc2indList, ind2rcList):
        # get all headers selected
        # headers = [self.headermodel.itemFromIndex(i).header for i in self.selectionmodel.selectedRows()]
        self.headers = [self.headermodel.item(i).header for i in range(self.headermodel.rowCount())]

        self.wavenumbers = wavenumbers
        self.N_w = len(self.wavenumbers)
        self.imgShapes = imgShapes
        self.rc2indList = rc2indList
        self.ind2rcList = ind2rcList
        self._dataSets = {'spectra': [], 'volume': []}

        for header in self.headers:
            data = None
            try:  # spectra datasets
                data = header.meta_array('spectra')
            except IndexError:
                msg.logMessage('Header object contained no frames with field ''{field}''.', msg.ERROR)
            if data is not None:
                self._dataSets['spectra'].append(data)
            # NMF path sets
            volumeEvent = next(header.events(fields=['volume']))
            path = volumeEvent['path'] # readin filepath
            self._dataSets['volume'].append(path)

    def setNumComponents(self):
        N = self.parameter['Components']

        for i in range(4):
            param = self.parameter.child(f'Map {i + 1} Component')
            param.setLimits(list(range(1, N + 1)))

    def calculate(self):

        N = self.parameter['Components']

        if hasattr(self, '_dataSets'):
            wavROIList = []
            for entry in self.parameter['Wavenumber Range'].split(','):
                try:
                    wavROIList.append(val2ind(int(entry), self.wavenumbers))
                except:
                    continue
            # Select wavenumber region
            if len(wavROIList) % 2 == 0:
                wavROIList = sorted(wavROIList)
                wavROIidx = []
                for i in range(len(wavROIList) // 2):
                    wavROIidx += list(range(wavROIList[2 * i], wavROIList[2 * i + 1] + 1))
            else:
                msg.logMessage('"Wavenumber Range" values must be in pairs', msg.ERROR)
                MsgBox('Factorization computation aborted.', 'error')
                return

            self.wavenumbers_select = self.wavenumbers[wavROIidx]
            # get map ROI selected region
            self.selectedPixelsList = [self.headermodel.item(i).selectedPixels for i in
                                       range(self.headermodel.rowCount())]
            self.df_row_idx = []  # row index for dataframe data_fac

            msg.showMessage('Start computing', self.method + '. Image shape:', str(self.imgShapes))
            self.dataRowSplit = [0]  # remember the starting/end row positions of each dataset
            if self.field == 'spectra':  # PCA workflow
                self.N_w = len(self.wavenumbers_select)
                self._allData = np.empty((0, self.N_w))

                for i, data in enumerate(self._dataSets['spectra']):  # i: map idx
                    if self.selectedPixelsList[i] is None:
                        n_spectra = len(data)
                        tmp = np.zeros((n_spectra, self.N_w))
                        for j in range(n_spectra):
                            tmp[j, :] = data[j][wavROIidx]
                            self.df_row_idx.append((self.ind2rcList[i][j], j))
                    else:
                        n_spectra = len(self.selectedPixelsList[i])
                        tmp = np.zeros((n_spectra, self.N_w))
                        for j in range(n_spectra):  # j: jth selected pixel
                            row_col = tuple(self.selectedPixelsList[i][j])
                            tmp[j, :] = data[self.rc2indList[i][row_col]][wavROIidx]
                            self.df_row_idx.append((row_col, self.rc2indList[i][row_col]))

                    self.dataRowSplit.append(self.dataRowSplit[-1] + n_spectra)
                    self._allData = np.append(self._allData, tmp, axis=0)

                if len(self._allData) > 0:
                    if self.method == 'PCA':
                        self.data_fac_name = 'data_PCA' # define pop up plots labels
                        # normalize and mean center
                        if self.parameter['Normalization'] == 'L1':# normalize
                            data_norm = Normalizer(norm='l1').fit_transform(self._allData)
                        elif self.parameter['Normalization'] == 'L2':
                            data_norm = Normalizer(norm='l2').fit_transform(self._allData)
                        else:
                            data_norm = self._allData
                        #subtract mean
                        data_centered = StandardScaler(with_std=False).fit_transform(data_norm)
                        # Do PCA
                        self.PCA = PCA(n_components=N)
                        self.PCA.fit(data_centered)
                        self.data_PCA = self.PCA.transform(data_centered)
                        # pop up plots
                        self.popup_plots()
                    elif self.method == 'MCR':
                        self.data_fac_name = 'data_MCR'  # define pop up plots labels
                        # Do ICA to find initial estimate of ST matrix
                        self.ICA = FastICA(n_components=N)
                        self.ICA.fit(self._allData)
                        # Do MCR
                        self.MCR = McrAR(max_iter=100, c_regr=self.parameter['C regressor'], st_regr='NNLS',
                                         tol_err_change=1e-6, tol_increase=0.5)
                        self.MCR.fit(self._allData, ST=self.ICA.components_)
                        self.MCR.components_ = self.MCR.ST_opt_
                        self.data_MCR = self.MCR.C_opt_
                        #test ICA
                        # self.MCR = self.ICA
                        # self.data_MCR = self.ICA.transform(self._allData)
                        # pop up plots
                        self.popup_plots()
                else:
                    msg.logMessage('The data matrix is empty. No PCA is performed.', msg.ERROR)
                    MsgBox('The data matrix is empty. No PCA is performed.', 'error')
                    self.PCA, self.data_PCA = None, None
                    self.MCR, self.data_MCR = None, None
                # emit PCA and transformed data
                if self.method == 'PCA':
                    self.sigPCA.emit((self.wavenumbers_select, self.PCA, self.data_PCA, self.dataRowSplit))
                elif self.method == 'MCR':
                    self.sigPCA.emit((self.wavenumbers_select, self.MCR, self.data_MCR, self.dataRowSplit))

            elif self.field == 'volume':  # NMF workflow
                data_files = []
                wav_masks = []
                row_idx = np.array([], dtype='int')
                self.allDataRowSplit = [0]  # row split for complete datasets

                for i, file in enumerate(self._dataSets['volume']):
                    ir_data, fmt = read_map.read_all_formats(file)
                    n_spectra = ir_data.data.shape[0]
                    self.allDataRowSplit.append(self.allDataRowSplit[-1] + n_spectra)
                    data_files.append(ir_data)
                    ds = data_prep.data_prepper(ir_data)
                    wav_masks.append(ds.decent_bands)
                    # row selection
                    if self.selectedPixelsList[i] is None:
                        row_idx = np.append(row_idx, np.arange(self.allDataRowSplit[-2], self.allDataRowSplit[-1]))
                        for k, v in self.rc2indList[i].items():
                            self.df_row_idx.append((k, v))
                    else:
                        n_spectra = len(self.selectedPixelsList[i])
                        for j in range(n_spectra):
                            row_col = tuple(self.selectedPixelsList[i][j])
                            row_idx = np.append(row_idx, self.allDataRowSplit[-2] +
                                                self.rc2indList[i][row_col])
                            self.df_row_idx.append((row_col, self.rc2indList[i][row_col]))

                    self.dataRowSplit.append(self.dataRowSplit[-1] + n_spectra)  # row split for ROI selected rows

                # define pop up plots labels
                self.data_fac_name = 'data_NMF'

                if len(self.df_row_idx) > 0:
                    # aggregate datasets
                    ir_data_agg = aggregate_data(self._dataSets['volume'], data_files, wav_masks)
                    col_idx = list(set(wavROIidx) & set(ir_data_agg.master_wmask))
                    self.wavenumbers_select = self.wavenumbers[col_idx]
                    ir_data_agg.data = ir_data_agg.data[:, col_idx]
                    ir_data_agg.data = ir_data_agg.data[row_idx, :]
                    # perform NMF
                    self.NMF = NMF(n_components=N)
                    self.data_NMF = self.NMF.fit_transform(ir_data_agg.data)
                    # pop up plots
                    self.popup_plots()
                else:
                    msg.logMessage('The data matrix is empty. No NMF is performed.', msg.ERROR)
                    MsgBox('The data matrix is empty. No NMF is performed.', 'error')
                    self.NMF, self.data_NMF = None, None
                # emit NMF and transformed data : data_NMF
                self.sigPCA.emit((self.wavenumbers_select, self.NMF, self.data_NMF, self.dataRowSplit))

    def popup_plots(self):
        # component variance ratio plot
        if self.method == 'PCA':
            plt.plot(getattr(self, self.method).explained_variance_ratio_, 'o-b')
            ax = plt.gca()
            ax.set_ylabel('Explained variance ratio', fontsize=16)
            ax.set_xlabel('Component number', fontsize=16)
            ax.set_xticks(np.arange(self.parameter['Components']))
        # loadings plot
        plt.figure()
        labels = []
        for i in range(getattr(self, self.method).components_.shape[0]):
            labels.append(self.method + str(i + 1))
            plt.plot(self.wavenumbers_select, getattr(self, self.method).components_[i, :], '-',
                     label=labels[i])
        loadings_legend = plt.legend(loc='best')
        plt.setp(loadings_legend, draggable=True)
        plt.xlim([max(self.wavenumbers_select), min(self.wavenumbers_select)])
        ax = plt.gca()
        ax.set_xlabel('Wavenumber$(cm^{-1})$', fontsize=16)
        # matrix plot
        groupLabel = np.zeros((self.dataRowSplit[-1], 1))
        for i in range(len(self.dataRowSplit) - 1):
            groupLabel[self.dataRowSplit[i]:self.dataRowSplit[i + 1]] = int(i)

        df_scores = pd.DataFrame(np.append(getattr(self, self.data_fac_name), groupLabel, axis=1),
                                 columns=labels + ['Group label'])
        grid = sns.pairplot(df_scores, vars=labels, hue="Group label")
        # change legend properties
        legend_labels = []
        for i in range(self.headermodel.rowCount()):
            if (self.selectedPixelsList[i] is None) or (self.selectedPixelsList[i].size > 0):
                legend_labels.append(self.headermodel.item(i).data(0))
        for t, l in zip(grid._legend.texts, legend_labels): t.set_text(l)
        plt.setp(grid._legend.get_texts(), fontsize=14)
        plt.setp(grid._legend.get_title(), fontsize=14)
        plt.setp(grid._legend, bbox_to_anchor=(0.2, 0.95), frame_on=True, draggable=True)
        plt.setp(grid._legend.get_frame(), edgecolor='k', linewidth=1, alpha=1)
        plt.show()

    def saveResults(self):
        if (hasattr(self, 'PCA') and self.PCA is not None) or (hasattr(self, 'NMF') and self.NMF is not None)\
                or (hasattr(self, 'MCR') and self.MCR is not None):
            name = self.method
            df_fac_components = pd.DataFrame(getattr(self, name).components_, columns=self.wavenumbers_select)
            df_data_fac = pd.DataFrame(getattr(self, self.data_fac_name), index=self.df_row_idx)
            df_fac_components.to_csv(name + '_components.csv')
            df_data_fac.to_csv(name + '_data.csv')
            np.savetxt(name + '_mapRowSplit.csv', np.array(self.dataRowSplit), fmt='%d', delimiter=',')
            MsgBox(name + ' components successfully saved!')
        else:
            MsgBox('No factorization components available.')

class ComponentPlotWidget(PlotWidget):
    def __init__(self, *args, **kwargs):
        super(ComponentPlotWidget, self).__init__(*args, **kwargs)
        self.line = InfiniteLine(pos=800, movable=True)
        self.line.sigPositionChanged.connect(self.getEnergy)
        self.addItem(self.line)
        self.cross = PlotDataItem([800], [0], symbolBrush=(255, 255, 255), symbolPen=(255, 255, 255), symbol='+', symbolSize=25)
        self.addItem(self.cross)
        self.txt = TextItem()
        self.getViewBox().invertX(True)
        self._x, self._y = None, None
        self.ymax, self.zmax = 0, 100

    def getEnergy(self):
        if self._y is not None:
            x_val = self.line.value()
            idx = val2ind(x_val, self._x)
            x_val = self._x[idx]
            y_val = self._y[idx]
            txt_html = f'<div style="text-align: center"><span style="color: #FFF; font-size: 12pt">\
                                                 X = {x_val: .2f}, Y = {y_val: .4f}</div>'
            self.txt.setHtml(txt_html)
            self.cross.setData([x_val], [y_val])

    def plot(self, x, y, *args, **kwargs):
        # set up infinity line and get its position
        plot_item = self.plotItem.plot(x, y, *args, **kwargs)
        self.addItem(self.line)
        self.addItem(self.cross)
        x_val = self.line.value()
        idx = val2ind(x_val, x)
        x_val = x[idx]
        y_val = y[idx]
        txt_html = f'<div style="text-align: center"><span style="color: #FFF; font-size: 12pt">\
                                     X = {x_val: .2f}, Y = {y_val: .4f}</div>'
        self.txt.setHtml(txt_html)
        self.txt.setZValue(self.zmax - 1)
        self.cross.setData([x_val], [y_val])
        self.cross.setZValue(self.zmax)
        ymax = max(y)
        if ymax > self.ymax:
            self.ymax = ymax
        self._x, self._y = x, y
        self.txt.setPos(x[0], 0.95 * self.ymax)
        self.addItem(self.txt)
        return plot_item

class FactorizationWidget(QSplitter):
    def __init__(self, headermodel, selectionmodel):
        super(FactorizationWidget, self).__init__()
        self.headermodel = headermodel
        self.selectionmodel = selectionmodel
        self.selectionmodel.selectionChanged.connect(self.updateMap)
        self.selectionmodel.selectionChanged.connect(self.updateRoiMask)
        self.selectMapIdx = 0

        self.rightsplitter = QSplitter()
        self.rightsplitter.setOrientation(Qt.Vertical)
        self.gridwidget = QWidget()
        self.gridlayout = QGridLayout()
        self.gridwidget.setLayout(self.gridlayout)
        self.display = QSplitter()

        # self.componentSpectra = PlotWidget()
        self.componentSpectra = ComponentPlotWidget()
        self._plotLegends = self.componentSpectra.addLegend()
        self._colors = ['r', 'g', 'm', 'y', 'c', 'b', 'w']  # color for plots

        # self.spectraROI = PlotWidget()
        self.NWimage = SlimImageView()
        self.NEimage = SlimImageView()
        self.SWimage = SlimImageView()
        self.SEimage = SlimImageView()
        # setup ROI item
        sideLen = 10
        self.roiList = []
        self.maskList = []
        self.selectMaskList = []
        self._imageDict = {0: 'NWimage', 1: 'NEimage', 2: 'SWimage', 3: 'SEimage'}
        for i in range(4):
            getattr(self, self._imageDict[i]).setPredefinedGradient("viridis")
            getattr(self, self._imageDict[i]).getHistogramWidget().setMinimumWidth(5)
            getattr(self, self._imageDict[i]).view.invertY(True)
            getattr(self, self._imageDict[i]).imageItem.setOpts(axisOrder="row-major")
            # set up roi item
            roi = PolyLineROI(positions=[[0, 0], [sideLen, 0], [sideLen, sideLen], [0, sideLen]], closed=True)
            roi.hide()
            self.roiInitState = roi.getState()
            self.roiList.append(roi)
            # set up mask item
            maskItem = ImageItem(np.ones((1,1)), axisOrder="row-major", autoLevels=True, opacity=0.3)
            maskItem.hide()
            self.maskList.append(maskItem)
            # set up select mask item
            selectMaskItem = ImageItem(np.ones((1, 1)), axisOrder="row-major", autoLevels=True, opacity=0.3,
                                          lut = np.array([[0, 0, 0], [255, 0, 0]]))
            selectMaskItem.hide()
            self.selectMaskList.append(selectMaskItem)
            # set up image title
            getattr(self, self._imageDict[i]).imageTitle = TextItem()
            getattr(self, self._imageDict[i]).view.addItem(roi)
            getattr(self, self._imageDict[i]).view.addItem(maskItem)
            getattr(self, self._imageDict[i]).view.addItem(selectMaskItem)
            getattr(self, self._imageDict[i]).view.addItem(getattr(self, self._imageDict[i]).imageTitle)

        self.parametertree = FactorizationParameters(headermodel, selectionmodel)
        self.parameter = self.parametertree.parameter
        self.parametertree.sigPCA.connect(self.showComponents)
        for i in range(4):
            self.parameter.child(f'Map {i + 1} Component').sigValueChanged.connect(
                partial(self.updateComponents, i))

        self.addWidget(self.display)
        self.addWidget(self.rightsplitter)
        self.display.addWidget(self.gridwidget)
        self.display.addWidget(self.componentSpectra)
        # self.display.addWidget(self.spectraROI)
        self.gridlayout.addWidget(self.NWimage, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.NEimage, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.SWimage, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.SEimage, 1, 1, 1, 1)

        self.setOrientation(Qt.Horizontal)
        self.display.setOrientation(Qt.Vertical)

        # Headers listview
        self.headerlistview = QListView()
        self.headerlistview.setModel(headermodel)
        self.headerlistview.setSelectionModel(selectionmodel)  # This might do weird things in the map view?
        self.headerlistview.setSelectionMode(QListView.SingleSelection)
        # add title to list view
        self.fontSize = 12
        font = QFont("Helvetica [Cronyx]", self.fontSize)
        self.mapListWidget = QWidget()
        self.listLayout = QVBoxLayout()
        self.mapListWidget.setLayout(self.listLayout)
        mapListTitle = QLabel('Maps list')
        mapListTitle.setFont(font)
        self.listLayout.addWidget(mapListTitle)
        self.listLayout.addWidget(self.headerlistview)

        # adjust right splitter
        self.rightsplitter.addWidget(self.parametertree)
        self.rightsplitter.addWidget(self.mapListWidget)
        self.rightsplitter.setSizes([200, 50])

    def updateRoiMask(self):
        if self.selectionmodel.hasSelection():
            self.selectMapIdx = self.selectionmodel.selectedIndexes()[0].row()
        elif self.headermodel.rowCount() > 0:
            self.selectMapIdx = 0
        else:
            return
        # update roi
        try:
            roiState = self.headermodel.item(self.selectMapIdx).roiState
            for i in range(4):
                if roiState[0]: #roi on
                    self.roiList[i].show()
                else:
                    self.roiList[i].hide()
                # update roi state
                self.roiList[i].blockSignals(True)
                self.roiList[i].setState(roiState[1])
                self.roiList[i].blockSignals(False)
        except Exception:
            for i in range(4):
                self.roiList[i].hide()
        # update automask
        try:
            maskState = self.headermodel.item(self.selectMapIdx).maskState
            for i in range(4):
                self.maskList[i].setImage(maskState[1])
                if maskState[0]:  # automask on
                    self.maskList[i].show()
                else:
                    self.maskList[i].hide()
        except Exception:
            pass
        # update selectMask
        try:
            selectMaskState = self.headermodel.item(self.selectMapIdx).selectState
            for i in range(4):
                self.selectMaskList[i].setImage(selectMaskState[1])
                if selectMaskState[0]:  # selectmask on
                    self.selectMaskList[i].show()
                else:
                    self.selectMaskList[i].hide()
        except Exception:
            pass

    def updateComponents(self, i):
        # i is imageview/window number
        # component_index is the PCA component index
        component_index = self.parameter[f'Map {i + 1} Component']
        # update scoreplots on view i
        if hasattr(self, '_data_fac') and (self._data_fac is not None):
            # update map
            self.drawMap(component_index, i)

        # update PCA components
        if hasattr(self, '_plots'):
            # update plots
            name = self.parameter['Method'] + str(component_index)
            self._plots[i].setData(self.wavenumbers, self._fac.components_[component_index - 1, :], name=name)
            # update legend label
            sample, label = self._plotLegends.items[i]
            label.setText(name)

    def updateMap(self):
        if self.selectionmodel.hasSelection():
            self.selectMapIdx = self.selectionmodel.selectedIndexes()[0].row()
        elif self.headermodel.rowCount() > 0:
            self.selectMapIdx = 0
        else:
            return

        if hasattr(self, '_data_fac') and (self._data_fac is not None):
            if len(self._dataRowSplit) < self.selectMapIdx + 2:  # some maps are not included in the factorization calculation
                msg.logMessage('One or more maps are not included in the factorization dataset. Please click "calculate" to re-compute factors.',
                    msg.ERROR)
            else:
                for i in range(4):
                    component_index = self.parameter[f'Map {i + 1} Component']
                    # update map
                    self.drawMap(component_index, i)
        elif hasattr(self, 'imgShapes') and (self.selectMapIdx < len(self.imgShapes)):  #clear maps
            for i in range(4):
                img = np.zeros((self.imgShapes[self.selectMapIdx][0], self.imgShapes[self.selectMapIdx][1]))
                getattr(self, self._imageDict[i]).setImage(img=img)

    def showComponents(self, fac_obj):
        # get map ROI selected region
        self.selectedPixelsList = [self.headermodel.item(i).selectedPixels for i in range(self.headermodel.rowCount())]
        # clear plots and legends
        self.componentSpectra.plotItem.clearPlots()
        self.componentSpectra.ymax = 0
        for sample, label in self._plotLegends.items[:]:
            self._plotLegends.removeItem(label.text)

        self.wavenumbers, self._fac, self._data_fac, self._dataRowSplit = fac_obj[0], fac_obj[1], fac_obj[2], fac_obj[3]

        if self._fac is not None:
            self._plots = []
            for i in range(4):
                component_index = self.parameter[f'Map {i + 1} Component']
                name = self.parameter['Method'] + str(component_index)
                # show loading plots
                tmp = self.componentSpectra.plot(self.wavenumbers, self._fac.components_[component_index - 1, :], name=name,
                                                 pen=mkPen(self._colors[i], width=2))
                tmp.curve.setClickable(True)
                tmp.curve.sigClicked.connect(partial(self.curveHighLight, i))
                self._plots.append(tmp)
                # show score plots
                self.drawMap(component_index, i)
            # update the last image and loading plots as a recalculation complete signal
            N = self.parameter['Components']
            self.parameter.child(f'Map 4 Component').setValue(N)
        # clear maps
        else:
            tab_idx = self.headermodel.rowCount() - 1
            if tab_idx >= 0:
                for i in range(4):
                    img = np.zeros((self.imgShapes[tab_idx][0], self.imgShapes[tab_idx][1]))
                    getattr(self, self._imageDict[i]).setImage(img=img)

    def drawMap(self, component_index, i):
        # i is imageview/window number
        data_slice = self._data_fac[self._dataRowSplit[self.selectMapIdx]:self._dataRowSplit[self.selectMapIdx + 1],
                     component_index - 1]
        # draw map
        if self.selectedPixelsList[self.selectMapIdx] is None:  # full map
            img = data_slice.reshape(self.imgShapes[self.selectMapIdx][0], self.imgShapes[self.selectMapIdx][1])
        elif self.selectedPixelsList[self.selectMapIdx].size == 0:  # empty ROI
            img = np.zeros((self.imgShapes[self.selectMapIdx][0], self.imgShapes[self.selectMapIdx][1]))
        else:
            img = np.zeros((self.imgShapes[self.selectMapIdx][0], self.imgShapes[self.selectMapIdx][1]))
            img[self.selectedPixelsList[self.selectMapIdx][:, 0], self.selectedPixelsList[self.selectMapIdx][:,
                                                               1]] = data_slice
        img = np.flipud(img)
        getattr(self, self._imageDict[i]).setImage(img=img)
        # set imageTitle
        imageTitle = getattr(self, self._imageDict[i]).imageTitle
        title = self.parameter['Method'] + str(component_index)
        imageTitle.setHtml(f'<div style="text-align: center"><span style="color: #FFF; font-size: 8pt">{title}</div>')
        imageTitle.setPos(0, -5)

    def curveHighLight(self, k):
        for i in range(4):
            if i == k:
                self._plots[i].setPen(mkPen(self._colors[k], width=6))
                self._plots[i].setZValue(50)
            else:
                self._plots[i].setPen(mkPen(self._colors[i], width=2))
                self._plots[i].setZValue(0)
        self.componentSpectra._x, self.componentSpectra._y = self._plots[k].getData()
        self.componentSpectra.getEnergy()

    def setHeader(self, field: str):

        self.headers = [self.headermodel.item(i).header for i in range(self.headermodel.rowCount())]
        self.field = field
        wavenum_align = []
        self.imgShapes = []
        self.rc2indList = []
        self.ind2rcList = []

        # get wavenumbers, imgShapes
        for header in self.headers:
            dataEvent = next(header.events(fields=[field]))
            self.wavenumbers = dataEvent['wavenumbers']
            wavenum_align.append(
                (round(self.wavenumbers[0]), len(self.wavenumbers)))  # append (first wavenum value, wavenum length)
            self.imgShapes.append(dataEvent['imgShape'])
            self.rc2indList.append(dataEvent['rc_index'])
            self.ind2rcList.append(dataEvent['index_rc'])

        # init maps
        if len(self.imgShapes) > 0:
            self.showComponents((self.wavenumbers, None, None, None))

        if wavenum_align and (wavenum_align.count(wavenum_align[0]) != len(wavenum_align)):
            MsgBox('Length of wavenumber arrays of displayed maps are not equal. \n'
                   'Perform PCA or NMF on these maps will lead to error.','warn')

        self.parametertree.setHeader(self.wavenumbers, self.imgShapes, self.rc2indList, self.ind2rcList)



