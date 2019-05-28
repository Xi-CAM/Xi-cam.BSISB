from xicam.gui.widgets.dynimageview import DynImageView
from xicam.core import msg
from xicam.core.data import NonDBHeader
import numpy as np
from qtpy.QtCore import Signal
from lbl_ir.data_objects.ir_map import val2ind

class MapViewWidget(DynImageView):
    sigShowSpectra = Signal(int)

    def __init__(self, *args, **kwargs):
        super(MapViewWidget, self).__init__(*args, **kwargs)
        # self.scene.sigMouseMoved.connect(self.showSpectra)
        self.scene.sigMouseClicked.connect(self.showSpectra)

    def setEnergy(self, lineobject):
        E = lineobject.value()
        # map E to index
        i = val2ind(E, self.wavenumbers)
        # print('E:', E, 'wav:', self.wavenumbers[i])
        self.setCurrentIndex(i)

    def showSpectra(self, event):

        pos = event.pos()
        if self.view.sceneBoundingRect().contains(pos):  # Note, when axes are added, you must get the view with self.view.getViewBox()
            mousePoint = self.view.mapSceneToView(pos)
            x, y = int(mousePoint.x()), int(mousePoint.y())
            ind = self.rc2ind[(y,x)]
            # print(x, y, ind, x + y * self.n_col)
            self.sigShowSpectra.emit(ind)


    def setHeader(self, header: NonDBHeader, field: str, *args, **kwargs):
        self.header = header
        self.field = field

        imageEvent = next(header.events(fields=['image']))
        self.rc2ind = imageEvent['rc_index']
        self.wavenumbers = imageEvent['wavenumbers']
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

    def updateImage(self, autoHistogramRange=True):
        super(MapViewWidget, self).updateImage(autoHistogramRange)
        self.ui.roiPlot.setVisible(False)

    def setImage(self, img, **kwargs):
        super(MapViewWidget, self).setImage(img, **kwargs)
        self.ui.roiPlot.setVisible(False)