from xicam.plugins.DataHandlerPlugin import DataHandlerPlugin, start_doc
import functools
from lbl_ir.io_tools.read_map import read_all_formats


class MapFilePlugin(DataHandlerPlugin):
    name = 'BSISB Map File'

    DEFAULT_EXTENTIONS = ['.map']

    descriptor_keys = ['object_keys']

    def __call__(self, *args, **kwargs):
        data, fmt = read_all_formats(self.path)
        return data.imageCube

    def __init__(self, path):
        super(MapFilePlugin, self).__init__()
        self.path = path


