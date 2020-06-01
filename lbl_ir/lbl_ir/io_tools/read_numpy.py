import numpy as np
from lbl_ir.data_objects import ir_map


def read_npy(filename, wavenumbers=None, data_type="absorbance", sample_info=None):
    if sample_info is None:
        sample_info = ir_map.sample_info()

    data = np.load(filename)

    if wavenumbers is None:
        wavenumbers = np.arange(data.shape[2])

    image_grid_param = [0, 0, 1, 1]
    image_mask = np.ones(data.shape[0:2]) > 0.5

    this_ir_map = ir_map.ir_map(wavenumbers=wavenumbers,
                                sample_info=sample_info,
                                data_type=data_type)
    this_ir_map.add_image_cube(data, image_mask, image_grid_param)
    return this_ir_map

def read_npz(filename, data_type="absorbance", sample_info=None):
    if sample_info is None:
        sample_info = ir_map.sample_info()

    npzFile = np.load(filename)

    for k in npzFile.files:
        if 'energy' in k:
            wavenumbers = npzFile[k]
        else:
            data = npzFile[k]

    image_grid_param = [0, 0, 1, 1]
    image_mask = np.ones(data.shape[0:2]) > 0.5

    this_ir_map = ir_map.ir_map(wavenumbers=wavenumbers,
                                sample_info=sample_info,
                                data_type=data_type)
    this_ir_map.add_image_cube(data, image_mask, image_grid_param)
    return this_ir_map