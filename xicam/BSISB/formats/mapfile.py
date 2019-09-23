from xicam.plugins.datahandlerplugin import DataHandlerPlugin, start_doc, descriptor_doc, embedded_local_event_doc
import functools
from lbl_ir.data_objects.ir_map import ir_map
import uuid
import h5py
from functools import lru_cache
import numpy as np
import time

class MapFilePlugin(DataHandlerPlugin):
    name = 'BSISB Map File'

    DEFAULT_EXTENTIONS = ['.h5']

    descriptor_keys = ['object_keys']

    def __call__(self, *args, E=None, i=None):
        if E is None and i is not None:
            # return spectra
            return self.h5[self.root_name + 'data/spectra'][i,:]

        elif E is not None and i is None:
            # return image or volume
            return np.flipud(self.h5[self.root_name + 'data/image/image_cube'][:,:,E])
        
        else:
            raise ValueError(f'Handler could not extract data given kwargs: { dict(E=E, i=i) }')

        # data, fmt = read_all_formats(self.path)
        # return data.imageCube

    @lru_cache(maxsize=1)
    def __init__(self, path):
        super(MapFilePlugin, self).__init__()
        self.path = path
        self.h5 = h5py.File(self.path, 'r')
        self.root_name = list(self.h5.keys())[0] + '/'

    def parseDataFile(self, *args, **kwargs):
        return dict()

    # TODO: spectra <-> image slice mapping in both events and descriptors

    @classmethod
    def getVolumeDescriptor(cls, path, start_uid):
        uid = uuid.uuid4()
        return descriptor_doc(start_uid, uid, {})

    @classmethod
    def getVolumeEvents(cls, path, descriptor_uid):
        with h5py.File(path, 'r') as f:
            root_name = list(f.keys())[0] + '/'
            n = f[root_name + 'data/image/image_cube'].shape[2]
            wavenumbers = f[root_name + 'data/wavenumbers'][:]
            ind_rc_map = f[root_name + 'data/image/ind_rc_map'][:, :]
            ind2rc = {x[0]: tuple(x[1:]) for x in ind_rc_map}
            rc2ind = {tuple(x[1:]): x[0] for x in ind_rc_map}
            imgMask = f[root_name + 'data/image/image_mask'][:, :]
            imgShape = (imgMask.shape[0], imgMask.shape[1])

        for i in range(n):
            yield embedded_local_event_doc(descriptor_uid, 'volume', cls, (path,), resource_kwargs={'E': i},
                                       metadata = {'path': path, 'wavenumbers': wavenumbers, 'rc_index': rc2ind, 'index_rc': ind2rc, 'imgShape': imgShape})

    @classmethod
    def getImageDescriptor(cls, path, start_uid):
        uid = uuid.uuid4()
        return descriptor_doc(start_uid, uid, {})

    @classmethod
    def getImageEvents(cls, path, descriptor_uid):
        # get a h5 object
        # find the imagecube
        # get imagecube shape
        # iterate over E dimension
        with h5py.File(path, 'r') as f:
            root_name = list(f.keys())[0] + '/'
            # get number of frames
            n = f[root_name + 'data/image/image_cube'].shape[2]
            wavenumbers = f[root_name + 'data/wavenumbers'][:]
            ind_rc_map = f[root_name + 'data/image/ind_rc_map'][:, :]
            ind2rc = {x[0]: tuple(x[1:]) for x in ind_rc_map}
            rc2ind = {tuple(x[1:]): x[0] for x in ind_rc_map}
            imgMask = f[root_name + 'data/image/image_mask'][:, :]
            imgShape = (imgMask.shape[0], imgMask.shape[1])
            
        for i in range(n):
            yield embedded_local_event_doc(descriptor_uid, 'image', cls, (path,), resource_kwargs={'E': i},
                                           metadata={'wavenumbers': wavenumbers, 'rc_index': rc2ind, 'index_rc': ind2rc, 'imgShape': imgShape})

    @classmethod
    def getSpectraDescriptor(cls, path, start_uid):
        uid = uuid.uuid4()
        return descriptor_doc(start_uid, uid, {})

    @classmethod
    def getSpectraEvents(cls, path, descriptor_uid):
        # get a h5 object
        # find the spectra data
        # get spectra index
        # iterate over rows
        with h5py.File(path, 'r') as f:
            root_name = list(f.keys())[0] + '/'
            # get number of rows
            n = f[root_name + 'data/spectra'].shape[0]
            wavenumbers = f[root_name + 'data/wavenumbers'][:]
            ind_rc_map = f[root_name + 'data/image/ind_rc_map'][:,:]
            ind2rc = {x[0]: tuple(x[1:]) for x in ind_rc_map}
            rc2ind = {tuple(x[1:]): x[0] for x in ind_rc_map}
            mask = f[root_name + 'data/image/image_mask'][:, :]
            imgShape = (mask.shape[0], mask.shape[1])

        for i in range(n):
            yield embedded_local_event_doc(descriptor_uid, 'spectra', cls, (path,), resource_kwargs={'i': i},
                                           metadata={'wavenumbers':wavenumbers, 'rc_index': rc2ind, 'index_rc': ind2rc, 'imgShape':imgShape})

    @classmethod
    def ingest(cls, paths):
        paths = cls.reduce_paths(paths)

        # TODO: handle multiple paths
        path = paths[0]

        start_uid = str(uuid.uuid4())

        volume_descriptor = cls.getVolumeDescriptor(path, start_uid)
        image_descriptor = cls.getImageDescriptor(path, start_uid)
        spectra_descriptor = cls.getSpectraDescriptor(path, start_uid)

        return {'start': cls._setTitle(cls.getStartDoc(paths, start_uid), paths),
                'descriptors': [volume_descriptor, image_descriptor, spectra_descriptor],
                'events': list(cls.getVolumeEvents(path, volume_descriptor['uid'])) +
                          list(cls.getImageEvents(path, image_descriptor['uid'])) +
                          list(cls.getSpectraEvents(path, spectra_descriptor['uid'])),
                'stop': cls.getStopDoc(paths, start_uid)}
