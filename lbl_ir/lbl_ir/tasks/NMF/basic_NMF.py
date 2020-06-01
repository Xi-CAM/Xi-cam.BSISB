from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import pyqtgraph as pq

import sys
import numpy as np
from lbl_ir.io_tools import read_map
from lbl_ir.tasks.preprocessing import data_prep


def simple_NMF(ir_map, wmask, smask,components=40):
    """
    A tools to do NMF on a selected region in a map

    :param ir_map: Image data cube
    :param wmask: wavelength mask
    :param smask: spatial mask
    :param components: number of components
    :return: spatial maps of components and associated spectra
    """
    ori_shape = ir_map.imageCube.shape
    data = ir_map.imageCube[:,:,wmask]
    data = data.reshape( ori_shape[0]*ori_shape[1], data.shape[2] )
    position_selection = np.where(smask.flatten()>0.5)[0]
    data = data[ position_selection, : ]


    NMF_obj = NMF(n_components=components)
    W = NMF_obj.fit_transform( data )
    H = NMF_obj.components_

    maps = []
    for nn in range(components):
        this_map = np.zeros(ori_shape[0:2])
        this_map = this_map.flatten()
        this_map[position_selection] = W[:,nn]
        this_map = this_map.reshape( ori_shape[0:2])
        maps.append(this_map)

    return maps, H
