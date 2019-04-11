from xicam.gui.widgets.dynimageview import DynImageView
from xicam.core import msg
from xicam.core.data import NonDBHeader

class MapViewWidget(DynImageView):
    def setHeader(self, header: NonDBHeader, field: str, *args, **kwargs):
        self.header = header
        self.field = field
        # make lazy array from document
        data = None
        try:
            data = header.meta_array(field)
        except IndexError:
            msg.logMessage('Header object contained no frames with field ''{field}''.', msg.ERROR)

        if data:
            # kwargs['transform'] = QTransform(1, 0, 0, -1, 0, data.shape[-2])
            self.setImage(img=data, *args, **kwargs)