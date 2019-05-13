from xicam.gui.widgets.dynimageview import DynImageView
from xicam.core import msg
from xicam.core.data import NonDBHeader
import numpy as np
from qtpy.QtCore import Signal

class MapViewWidget(DynImageView):
    sigShowSpectra = Signal(int)

    def __init__(self, *args, **kwargs):
        super(MapViewWidget, self).__init__(*args, **kwargs)
        self.scene.sigMouseMoved.connect(self.showSpectra)

    def showSpectra(self, pos):
        if self.view.sceneBoundingRect().contains(
                pos):  # Note, when axes are added, you must get the view with self.view.getViewBox()
            mousePoint = self.view.mapSceneToView(pos)
            x, y = int(mousePoint.x()), int(mousePoint.y())
            idx = x + y * self.n_col
            self.sigShowSpectra.emit(idx)

    def setHeader(self, header: NonDBHeader, field: str, *args, **kwargs):
        self.header = header
        self.field = field
        # make lazy array from document
        data = None
        try:
            data = header.meta_array(field)
            self.n_row = data.shape[1]
            self.n_col = data.shape[2]
        except IndexError:
            msg.logMessage('Header object contained no frames with field ''{field}''.', msg.ERROR)

        if data is not None:
            # kwargs['transform'] = QTransform(1, 0, 0, -1, 0, data.shape[-2])
            self.setImage(img=data, *args, **kwargs)
            # test
            # data2 = np.zeros((17,32))
            # data2[3,31] = 0.05
            # data2[13, 0] = 1
            # self.setImage(img=data2, *args, **kwargs)
