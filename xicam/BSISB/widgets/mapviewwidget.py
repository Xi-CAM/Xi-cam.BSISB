import numpy as np
from xicam.gui.widgets.dynimageview import DynImageView
from xicam.core import msg
from xicam.core.data import NonDBHeader
from pyqtgraph import ArrowItem, TextItem
from qtpy.QtCore import Signal
from lbl_ir.data_objects.ir_map import val2ind

class MapViewWidget(DynImageView):
    sigShowSpectra = Signal(int)

    def __init__(self, *args, **kwargs):
        super(MapViewWidget, self).__init__(*args, **kwargs)
        # self.scene.sigMouseMoved.connect(self.showSpectra)
        self.scene.sigMouseClicked.connect(self.showSpectra)
        self.view.invertY(True)

        # add arrow
        self.arrow = ArrowItem(angle=60, headLen=15, tipAngle=45, baseAngle=30, brush = (200, 80, 20))
        self.arrow.setPos(0, 0)
        self.view.addItem(self.arrow)
        self.arrow.hide()
        #add txt
        self.txt = TextItem('', anchor=(0, 0))
        self.addItem(self.txt)

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
            y = self.row - y - 1
            try:
                ind = self.rc2ind[(y,x)]
                self.sigShowSpectra.emit(ind)
                # print(x, y, ind, x + y * self.n_col)

                #update arrow
                self.arrow.setPos(x + 0.5, self.row - y - 0.5)
                self.arrow.show()
                # update text
                self.txt.setHtml(f'<div style="text-align: center"><span style="color: #FFF; font-size: 8pt">X: {x}</div>\
            <div style="text-align: center"><span style="color: #FFF; font-size: 8pt">Y: {y}</div>\
            <div style="text-align: center"><span style="color: #FFF; font-size: 8pt">Point: #{ind}</div>')
            except Exception:
                self.arrow.hide()


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
            self.row = data.shape[1]
            self.col = data.shape[2]
            self.txt.setPos(self.col, 0)
        except IndexError:
            msg.logMessage('Header object contained no frames with field ''{field}''.', msg.ERROR)

        if data is not None:
            # kwargs['transform'] = QTransform(1, 0, 0, -1, 0, data.shape[-2])
            self.setImage(img=data, *args, **kwargs)
            self._data = data

    def updateImage(self, autoHistogramRange=True):
        super(MapViewWidget, self).updateImage(autoHistogramRange)
        self.ui.roiPlot.setVisible(False)

    def setImage(self, img, **kwargs):
        super(MapViewWidget, self).setImage(img, **kwargs)
        self.ui.roiPlot.setVisible(False)

    def makeMask(self, thresholds):
        peak1550 = val2ind(1550, self.wavenumbers)
        thr1550 = thresholds[0]
        mask = self._data[peak1550] > thr1550
        mask = mask.astype(np.int)
        return mask
