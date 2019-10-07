from xicam.gui.widgets.imageviewmixins import BetterButtons
import numpy as np


class SlimImageView(BetterButtons):
    def __init__(self, invertY=True):
        super(SlimImageView, self).__init__()
        # Shrink LUT
        self.getHistogramWidget().setMinimumWidth(1)
        # set up layout
        self.ui.gridLayout.addWidget(self.resetAxesBtn, 2, 2, 1, 1)
        self.ui.gridLayout.addWidget(self.resetLUTBtn, 3, 2, 1, 1)
        self.ui.gridLayout.addWidget(self.ui.graphicsView, 0, 0, 4, 1)
        # set up colorbar
        self.setPredefinedGradient("viridis")
        self.view.invertY(invertY)
        self.imageItem.setOpts(axisOrder="row-major")
        # Setup late signal
        self.sigTimeChangeFinished = self.timeLine.sigPositionChangeFinished

    def quickMinMax(self, data):
        """
        Estimate the min/max values of *data* by subsampling. MODIFIED TO USE THE 99TH PERCENTILE instead of max.
        """
        if data is None:
            return 0, 0
        ax = np.argmax(data.shape)
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(None, None, max(1, int(data.size // 1e4)))
        data = data[sl]
        return (np.nanmin(data), np.nanpercentile(np.where(data < np.nanmax(data), data, np.nanmin(data)), 99))


