from qtpy.QtWidgets import QSplitter, QGridLayout, QWidget
from .mapviewwidget import MapViewWidget
from pyqtgraph import PlotWidget, ImageView
from qtpy.QtCore import Qt
from xicam.core.data import NonDBHeader
from functools import partial

from pyqtgraph.parametertree import ParameterTree, Parameter


class FactorizationParameters(ParameterTree):
    def __init__(self):
        super(FactorizationParameters, self).__init__()
        self.parameter = Parameter(name='params', type='group',
                                   children=[{'name': "Number of Components",
                                              'type': 'int'},
                                             {'name': "Recalculate",
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
                                              'type': 'list'}, ])

        self.setParameters(self.parameter, showTop=False)

        self.parameter.child('Recalculate').sigActivated.connect(self.recalculate)
        for i in range(4):
            self.parameter.child(f'Map {i + 1} Component Index').sigValueChanged.connect(partial(self.showComponent, i))

    def showComponent(self, i):
        # (i is imageview number)
        component_index = self.parameter[f'Map {i + 1} Component Index']

        # display component on view i

    def setNumComponents(self, N):
        i = 1
        for i in range(4):
            param = self.parameter.child(f'Map {i} Component Index')
            param.setValues(list(range(1, N + 1)))
            param.setValue(param.value if param.value <= N else N)

    def recalculate(self):
        N = self.parameter['Number of Components']
        self.setNumComponents(N)

        # Do some processing

        # Display images


class FactorizationWidget(QSplitter):
    def __init__(self):
        super(FactorizationWidget, self).__init__()

        self.gridwidget = QWidget()
        self.gridlayout = QGridLayout()
        self.gridwidget.setLayout(self.gridlayout)
        self.display = QSplitter()

        self.spectraROI = PlotWidget()
        self.componentSpectra = PlotWidget()
        self.NWimage = ImageView()
        self.NEimage = ImageView()
        self.SWimage = ImageView()
        self.SEimage = ImageView()
        self.parametertree = FactorizationParameters()
        self.parameter = self.parametertree.parameter

        self.addWidget(self.display)
        self.addWidget(self.parametertree)
        self.display.addWidget(self.gridwidget)
        self.display.addWidget(self.spectraROI)
        self.display.addWidget(self.componentSpectra)
        self.gridlayout.addWidget(self.NWimage, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.NEimage, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.SWimage, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.SEimage, 1, 1, 1, 1)

        self.setOrientation(Qt.Horizontal)
        self.display.setOrientation(Qt.Vertical)

    def setHeader(self, header: NonDBHeader, field: str):
        self.header = header
        self.field = field
