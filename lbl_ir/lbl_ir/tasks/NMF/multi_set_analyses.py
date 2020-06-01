from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
#import pyqtgraph as pq


import sys
import numpy as np
from lbl_ir.io_tools import read_map
from lbl_ir.tasks.preprocessing import data_prep, transform
#from umap import UMAP


class aggregate_data(object):
    def __init__(self, names, data, wmask, components=40):
        self.names = names
        self.Nset  = len(names)
        self.wmask = wmask
        self.master_wmask = self.get_wavenumber_intersection()
        self.wavenumbers = data[0].wavenumbers[self.master_wmask]
        self.data, self.dims, self.labels = self.get_data(data)
        del data
        self.components = components
        print(self.dims)


    def get_wavenumber_intersection(self):
        result = self.wmask[0]
        for ii in range(1,self.Nset):
            result = np.intersect1d(result,self.wmask[ii])
        return result

    def get_data(self,data):
        result = []
        dims = []
        labels = []
        for kk,cube in enumerate(data):
            sel_cube = cube.imageCube[:,:, self.master_wmask ]
            dim = sel_cube.shape
            sel_cube = sel_cube.reshape( dim[0]*dim[1],dim[2] )
            result.append( sel_cube )
            labs = np.zeros( dim[0]*dim[1] ) + kk
            labels.append(labs)
            dims.append( dim[:2] )
        result = np.vstack( result )
        labels = np.concatenate(labels).flatten()
        return transform.to_absorbance(result,False)[0], dims, labels

    def splitter(self,X,to_map=False):
        sets = []
        for kk in range( self.Nset ):
            sel = self.labels == kk
            sel = np.where(sel)[0]
            tmp = X[sel,:]
            if to_map:
                tmp = tmp.reshape( self.dims[kk] )
            sets.append( tmp )
        return sets

    


def multi_set_umap(agg_data, fraction=0.25):
    umap_object = UMAP(n_components=2, n_neighbors=5)
    Nobs = agg_data.data.shape[0]
    these_ones = np.arange(Nobs)
    np.random.shuffle(these_ones)
    these_ones = these_ones[:int(Nobs*fraction)]
    these_ones = np.sort(these_ones)

    low_dim = umap_object.fit_transform( agg_data.data[these_ones,:] )
    low_dim_all = umap_object.transform( agg_data.data)
    low_dim_all_split = agg_data.splitter(low_dim_all,False)

    for set in low_dim_all_split:
        plt.plot(set[:,0],set[:,1],'.' , markersize=1.5 )
    plt.savefig('UMAP.png')

def multi_set_NMF(agg_data,components=40):
    NMF_obj = NMF(n_components=components)
    W = NMF_obj.fit_transform( agg_data.data )
    H = NMF_obj.components_
    spectra = []
    maps    = []
    for ii in range(components):
        Ws = W[:,ii].reshape(-1,1)
        Ws_split = agg_data.splitter( Ws,True )
        spectra.append(H[ii,:])
        maps.append( Ws_split )
    return maps, spectra, agg_data.wavenumbers


def single_set_NMF(data, wav_mask, components):
    A = transform.fix_data( data.data[:,wav_mask] )
    NMF_obj = NMF(n_components=components)
    W = NMF_obj.fit_transform( A )
    H = NMF_obj
    return data.wavenumbers[wav_mask],W,H



if __name__ == "__main__":
    data_files = []
    wav_masks  = []
    names = ['../../../ir_data/test_data.map']
    if len(names)>1:
        for fname in names:
            data,fmt = read_map.read_all_formats( fname )
            data_files.append( data )
            ds = data_prep.data_prepper(data)
            wav_masks.append( ds.decent_bands )

        ad = aggregate_data(names, data_files, wav_masks)
        #multi_set_umap( ad )
        multi_set_NMF( ad )
    else:
        data,fmt = read_map.read_all_formats( names[0] )
        print(fmt)
        wmask = data_prep.data_prepper(data).decent_bands
        wavs,W,H = single_set_NMF(data, wmask, 4 )
        for i in range(4):
            plt.plot(wavs, H.components_[i], label='NMF'+str(i))
            plt.xlim([4000, 500])
            plt.legend()
 
