import os
import numpy as np
import h5py
from lbl_ir.data_objects import ir_map


def get_grid_info(coords):
    xsorted = sorted(set(coords[:,0]))
    ysorted = sorted(set(coords[:,1]))
    x0, xmax, y0, ymax = xsorted[0], xsorted[-1], ysorted[0], ysorted[-1]
    dx = np.mean(np.diff(xsorted))
    dy = np.mean(np.diff(ysorted))
    if dx > dy:
        step = dy / 2
    else:
        step = dx / 2
    Nx = int(round((xmax - x0)/step) + 1)
    Ny = int(round((ymax - y0)/step) + 1)
    return x0, y0, step, Nx, Ny

def read_xasH5(filePath):
    xasTypes = []
    energy = {}
    dataSets = {}
    coords = {}
    xas_maps = {}

    with h5py.File(filePath, 'r') as f:
        xasSpectra = f['xas/']
        for k in xasSpectra:
            xasTypes.append(k)
            for k1 in xasSpectra[k]:
                k1 = '/' + k1
                spectra = xasSpectra[k + k1 + '/raw'][:, :, :]
                energy[k] = spectra[0, 0, :]
                dataSets[k] = spectra[:, 1, :]
                dataSets[k] = np.where(dataSets[k] != np.inf, dataSets[k], 0) #filter np.inf

        samples = f['maps/samples/']
        for i, k in enumerate(samples):
            coords[xasTypes[i]] = samples[k + '/xas_coords'][:, :]

        for _type in xasTypes:
            assert coords[_type].shape[0] == dataSets[_type].shape[0], 'xas and coords sample sequences were mis-aligned.'

            fileName = os.path.basename(filePath)
            sample_info = ir_map.sample_info(fileName[:-3])
            xas_maps[_type] = ir_map.ir_map(wavenumbers=energy[_type], sample_info=sample_info)
            xas_maps[_type].add_data(spectrum=dataSets[_type], xy=coords[_type])
            x0, y0, step, Nx, Ny = get_grid_info(coords[_type])
            xas_maps[_type].to_image_cube(Nx, Ny, x0, y0, step, step)

        return xas_maps

