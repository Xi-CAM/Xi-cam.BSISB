import matplotlib.pyplot as plt
#import pyqtgraph as pq


import sys
import numpy as np
from lbl_ir.io_tools import read_map

from scipy.stats import iqr
import scipy.signal
import scipy.ndimage

from numba import jit


@jit#(nopython=True)
def band_score_numba(map,subsample=1,band=10):
    Nx,Ny,Nwav = map.shape
    result = np.zeros((Nwav,Nwav))
    norma  = np.zeros((Nwav,Nwav))
    for ii in range(Nwav):
        for jj in range( ii+1 ,min(ii+band,Nwav)):
            slab_0 = map[::subsample,::subsample,ii].flatten()
            slab_1 = map[::subsample,::subsample,jj].flatten()
            cc = np.corrcoef(slab_0, slab_1)[0][1]
            norma[ii,jj]=1.0
            norma[jj,ii]=1.0
            result[ii,jj]=cc
            result[jj,ii]=cc
    result = np.sum( result, axis=1)
    norma  = np.sum( norma, axis=1)
    return result/norma


class data_prepper(object):
    """
    Prep the data for further data analyses.
    We try to detect systematic issues, such as bad bands and synchortron noise spikes


    """
    def __init__(self, data_map, band_limit=0.99, threshold=6.0,band=5, additional_selection = None):
        self.data_map    = data_map
        self.waves       = self.data_map.wavenumbers
        self.threshold   = threshold
        self.additional_selection = additional_selection
        self.band_limit   = band_limit
        self.band_scores  = None
        self.bad_bands    = None
        self.decent_bands = None
        self.score_bands(band)

    def score_bands(self,band=10):
        # we need to loop over every single frame and detect spikes
        self.band_scores  = band_score_numba( self.data_map.imageCube,subsample=1,band=band )
        self.bad_bands    = np.where( self.band_scores < self.band_limit )[0]
        self.decent_bands = np.where( self.band_scores >= self.band_limit)[0]

    def wavenumber_mask(self):
        # we have to merge these auto score bands with the manual selection
        selections = self.band_scores >= self.band_limit
        tmp = np.zeros( len(self.waves) ) > 1.0
        if self.additional_selection is not None:
            for this_sel in self.additional_selection:
                pair = np.sort( this_sel )
                sel1 = (self.waves > pair[0]) & (self.waves<pair[1])
                tmp = tmp | sel1
        return np.where(selections&tmp)[0]

    def plot_bad_bands(self):
        for band in self.bad_bands:
            slab = self.data_map.imageCube[:,:,band]
            #pq.image(slab, title='wavenumber: %7.2f'%self.data_map.wavenumbers[band])

    def plot_decent_bands(self):
        for ii in range(0,1600,10):
            band = self.decent_bands[ii]
            slab = self.data_map.imageCube[:,:,band]
            pq.image(slab, title='wavenumber: %7.2f'%self.data_map.wavenumbers[band])



if __name__ == "__main__":
    print(sys.argv)
    data,fmt = read_map.read_all_formats( sys.argv[1] )
    ds = data_prepper(data)
    ds.plot_bad_bands()
    #ds.plot_decent_bands()
    plt.plot(data.wavenumbers, ds.band_scores);
    plt.savefig('tmp.png')
    input("Press enter to exit")





