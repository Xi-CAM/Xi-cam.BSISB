from scipy.sparse.linalg import svds, eigsh
from scipy.linalg.interpolative import svd as isvd


import os
import numpy as np
import matplotlib.pyplot as plt
import time

from mpi4py import MPI
import h5py

from tqdm import tqdm


def test_function( N, M , noise=0.001):
    """
    Generate some test data we can use. The resulting matrix should have a rank of 4.
    """

    x = np.linspace(-1,1,M)
    p0 = x*0+1.0
    p1 = x
    p2 = 0.5*(3*x*x-1)
    p3 = 0.5*(5*x*x*x-3*x)

    result = []
    for ii in range(N):
        tmp = np.random.uniform(-1,1,4)
        tmp = tmp[0]*p0 + tmp[1]*p1 + tmp[2]*p2 + tmp[3]*p3
        tmp = tmp + np.random.normal(0,1.0,M)*noise
        result.append(tmp)
    result = np.vstack(result)
    
    return result

def invert_permutation(p):
    """Given a permutation, provide an array that undoes the permutation.
    """
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


class batched_SVD(object):
    """Compute an SVD of all data, but in small batches to overcome memory issues.

    Parameters:
    ----------

    data:  A pointer to a data object, say a N,M matrix.
           N experimental observations of dimension M

    N_max: The maximum number of entries in sub block of data

    k_singular: The number of singular values to consider

    randomize : A flag which determines if the data will be split in a random fashion. True by default


    Attributes:
    -----------
    self.order     : The order in which the data will be examined

    self.inv_order : The inverse of the above array 

    self.N_split   : The number of batches of data

    self.parts     : A list of selection arrays

    self.partial_svd_u  :  A list of (truncated) svd matrices (U) from individual batches of data 

    self.partial_svd_s  :  A list of (truncated) svd matrices (S) from individual batches of data

    self.partial_svd_vt :  A list of (truncated) svd matrices (V^T) from individual batches of data

    self.partial_bases  :  A list of (truncated) svd matrices (SV^T) from individual batches of data


    Examples:
    ---------

    data =  test_function(10000,3000,1.1)
    bSVD = batched_SVD(data, 2000, 5, randomize=True) 
    u,s,vt = bSVD.go_svd()

    """

    def __init__(self, data, N_max, k_singular, randomize=True):
        self.data = data
        self.N_max = N_max
        self.k_singular = k_singular

        self.randomize = randomize
        
        self.order = np.arange( self.data.shape[0] ) 
        if self.randomize:
            np.random.shuffle( self.order  )
 
        self.N_split = int( np.floor(self.data.shape[0] / self.N_max) ) +1
        self.parts = np.array_split( self.order, self.N_split)

        # if we have these numbers in order, we can use them as slices in a hdf5 setting
        # this is only requiered when we randomize the lot
        if self.randomize:
            tmp = []
            for part in self.parts:
                part = np.sort(part)
                tmp.append(part)
            self.parts = tmp 
            self.order = np.concatenate(self.parts)
        self.inv_order = invert_permutation(self.order)      
 
        # here the partial svds are stored 
        self.partial_svd_u  = []
        self.partial_svd_s  = []
        self.partial_svd_vt = []
        self.partial_bases  = []


    def SVD_on_chunk(self, this_chunk):
        """Perform an SVD on a subset of the data

        Parameters:
        -----------

        this_chunk:  a list of indices that maps back to the self.data array 
        """

        tmp_data = self.data[ this_chunk, : ]

        u,s,v = isvd(tmp_data.astype('float64'), self.k_singular) 
        vt = v.transpose()
        self.partial_svd_u.append( u )
        self.partial_svd_s.append( s )
        self.partial_svd_vt.append( vt )
        self.partial_bases.append( np.diag(s).dot(vt) )


    def get_u_given_basis(self, sigma, v_transpose, this_chunk):
        inv_bases = v_transpose.transpose().dot( np.diag( 1.0 / sigma ) )
        tmp_data = self.data[ this_chunk, : ] 
        new_u = tmp_data.dot( inv_bases )
        return new_u

    def go_svd(self):
        """ Seperate SVD's for individual data chunks are computed and combined into a single, best estimate
        of the matrix S and V^T. The data is subsequently revisited to get a new estimate of the matrix U.  

        The estimate of U on the basis of the chunked SVD approach is possible, but not as accurate. 
        """

        for this_chunk in self.parts:
            self.SVD_on_chunk( this_chunk )
       
        # do an SVD on the SVD results of the individual chunks. 
        tmp_bases = np.vstack( self.partial_bases )
        ub,sb,vb = isvd(tmp_bases.astype('float64'), self.k_singular)
        vbt = vb.transpose()
        ubs = np.array_split(ub,self.N_split)
        new_s  = sb
        new_vt = vbt
        
        # revisit the data to get the U matrix again
        new_us = []
        for part in self.parts: 
            this_new_u = self.get_u_given_basis(new_s, new_vt, part)
            new_us.append(this_new_u)
        
        # stack it up and unmix the data
        new_us = np.vstack(new_us)
        if self.randomize:
            new_us = new_us[ self.inv_order ]
        return new_us, sb, vbt 


class parallel_SVD_MPI(object):
    """
    THSI NEEDS WORK!!!
    An MPI based version of an SVD method, splitting the data among a number of cores.
    The outline is a bit different than the batched_SVD version.

    The idea is to first split all data equally across all cores. In each core, we compute 
    the an SVD using the procedure coded up for the batched_SVD approach, and end up with a
    SVD estimate on each core. We subsequently pool all these SVD's and reestimate it on 
    the root node, and scatter these functions back to all subsequent nodes. Subsequently, 
    we need to compute U on on each node for all data associated with this node. The final 
    results are stored in a pointer, which is either a hdf5 file or a numpy array. 


    Parameters:
    ----------

    data:  A pointer to a data object, say a N,M matrix.
           N experimental observations of dimension M

    N_max: The maximum number of entries in sub block of data

    k_singular: The number of singular values to consider

    MPI_COMM_WORLD: an instance of MPI.COMM_WORLD

    randomize : A flag which determines if the data will be split in a random fashion. True by default

    Attributes:
    -----------    

    go_svd()  : this function performs the svd and returns the SVD results for each single core, including 
                an array that allows one to map the data back to the original order in which it was presented.

    Example:
    --------

    



    """


    def __init__(self, data, N_max, k_singular, MPI_COMM_WORLD, randomize=True, selection = None ):
        # randomize the order in which we analyze the data
        self.randomize  = randomize

        # sort out MPI stuff
        self.mpi_comm   = MPI_COMM_WORLD
        self.mpi_rank   = self.mpi_comm.Get_rank()
        self.mpi_size   = self.mpi_comm.Get_size()

        # data etc
        self.data       = data       # the data
        self.k_singular = k_singular # the number of singular vectors
        self.N_max      = N_max      # the maximum number of data points per chunk per core

        # split the data amond cores 
        self.N_obs, self.N_dim = self.data.shape
        if selection is not None:
            self.Nobs = len(selection)
        self.order  = np.arange(self.N_obs)
        self.selection = selection
        if self.selection is not None:
            self.order  = self.order[selection] 
        self.inv_order = None

        if self.randomize:
            np.random.shuffle( self.order )
        self.rank_splits       = np.array_split( self.order, self.mpi_size )

        if self.randomize:
            tmp = []
            for part in self.rank_splits:
                part = np.sort(part) # we want this ordered from low to high to enable slicing an hdf5 array 
                tmp.append(part)
            self.rank_splits = tmp
            self.order = np.concatenate(self.rank_splits)
        self.inv_order = invert_permutation(self.order)
        self.inv_order_split = np.array_split( self.inv_order, self.mpi_size )
        self.mpi_comm.Barrier()

    def go_svd(self):
        """Do the SVD ihn each core, on several chunks.
        """

        partial_u     = []
        partial_s     = []
        partial_vt    = []
        partial_bases = []

        rank_selection = self.rank_splits[  self.mpi_rank ]
        N_chunks = int( len(rank_selection) / self.N_max ) + 1
        chunks = np.array_split( rank_selection, N_chunks )
        u = None
        s = None
        vt = None
        this_rank_bases = None
        for chunk in tqdm(chunks, position=self.mpi_rank):
            partial_data = self.data[ chunk, : ] 
            u,s,vt = isvd(partial_data.astype('float64'), self.k_singular)
            partial_u.append(  u  )
            partial_s.append(  s  )
            partial_vt.append( vt )

            partial_bases.append( np.diag(s).dot(vt) ) 
        # now that we have the partial svd results, we bnring stuff together
        if N_chunks > 1:
            all_bases = np.vstack( partial_bases )
            uc,sc,vct = isvd(all_bases.astype('float64'), self.k_singular)
            # now we need to pass this guy to the main rank
            this_rank_bases = np.diag(sc).dot(vct)
        else:
            this_rank_bases = np.diag(s).dot(vt)
        self.mpi_comm.Barrier()


        gathered_bases = None
        if self.mpi_rank == 0:
            gathered_bases = np.empty( [self.mpi_size, self.k_singular, self.N_dim], dtype='d' )
        gathered_bases = self.mpi_comm.gather( this_rank_bases, root=0 )

        final_sg  = np.zeros( [ self.k_singular ], dtype='float32'  )
        final_vgt = np.zeros( [ self.k_singular, self.N_dim], dtype='float32'  )

        if self.mpi_rank == 0:
            gathered_bases = np.vstack( gathered_bases )
            ug,final_sg,final_vgt = isvd(gathered_bases.astype('float64'), self.k_singular)
        # we now need to scatter back the sg and vgt matrices
        self.mpi_comm.Barrier()
        self.mpi_comm.Bcast(  final_sg, root=0 )
        self.mpi_comm.Barrier()
        self.mpi_comm.Bcast( final_vgt , root=0 ) 
        self.mpi_comm.Barrier()
        # now that we have the final and best sigma and Vt, we need to go back to the data and 
        # reestimate the the U matrices
        inv_multi = final_vgt.transpose().dot( np.diag(final_sg) )
        chunks_of_u = []
        for chunk in chunks:
            partial_data = self.data[ chunk, : ]    
            this_u = partial_data.dot( inv_multi )
            chunks_of_u.append( this_u)

        chunks_of_u = np.vstack( chunks_of_u )
        # we need to return this, including a placement array
        return chunks_of_u, final_sg, final_vgt, self.inv_order_split






def tst_batched():
    N = 10000
    M = 2000
    K = 200
    P = 4

    data =  test_function(N,M,0.0001)
    print("Data constructed")
    e0 = time.time()
    u,s,vt = isvd(data.astype('float64'), P)
    vt = vt.transpose()
    e1 = time.time()
 
    bSVD = batched_SVD(data, K, P, randomize=False)
    e4 = time.time()
    us, ss, vst = bSVD.go_svd()
    e5 = time.time()

    assert np.std( (s-ss)/ss ) < 1e-3
    print("Singular values match between batch and full approach")

    # checking the reconstructions
    da = us.dot(np.diag(ss).dot( vst ))
    dc =  u.dot(np.diag(s).dot(  vt  ))
    delta_both = np.std( (da-dc) )
    assert np.abs( delta_both ) < 1e-2


    print('Reconstruction Error is similar in batched versus full approach')
    print ('Time for full: %4.2f  batched: %4.2f'%(e1-e0, e5-e4))

    # Now we want to do this using a random order or data

    bSVD = batched_SVD(data, K, P, randomize=True)
    e4 = time.time()
    us, ss, vst = bSVD.go_svd()
    e5 = time.time()

    assert np.std( (s-ss)/s ) < 1e-3
    print("Singular values match between randomized batch and full approach")
    delta_both = np.std( (dc - da) ) 
    assert np.abs( delta_both ) < 1e-2 
    print('Reconstruction Error is decent')
    print ('Time for full: %4.2f  batched: %4.2f'%(e1-e0, e5-e4))


def tst_MPI():
    N = 10000 # number of observations
    M = 2000  # dimension of an observations
    P = 4     # number of singular values

    # make the data
    mpi_comm = MPI.COMM_WORLD
    if mpi_comm.Get_rank() == 0:
        data =  test_function(N,M,0.0001)

        e0 = time.time()
        u,s,vt = isvd(data.astype('float64'),P)
        e1 = time.time()
        print('Standard svd takes %12.3f seconds'%(e1-e0))
        print('\n') 

        e0 = time.time()
        # lets quickly see if this works in the batched setup
        bSVD = batched_SVD(data, 1000, P, randomize=True)
        ub, sb, vbt = bSVD.go_svd()
        e1 = time.time()
        print('Single core batched version with data in memory takes %12.3f seconds'%(e1-e0) )


        # write this to an hdf5 file
        print('Creating h5 data file')
        f = h5py.File('test_data.h5','w')
        dset = f.create_dataset("data", data=data, dtype='float32')
        f.close()
        del data

        # now we read the data
        f = h5py.File('test_data.h5','r')
        data = f['/data']
        print(data.shape)
        e0 = time.time()
        # lets quickly see if this works in the batched setup
        bSVD = batched_SVD(data, 1000, P, randomize=False)
        ub, sb, vbt = bSVD.go_svd()
        e1 = time.time()
        print('Single core batched version while reading data from HDF5 takes %12.3f seconds'%( e1-e0 ) ) 
        print(sb)
        f.close()

    mpi_comm.Barrier()
    f = h5py.File('test_data.h5','r' ) # lets see if we can get away with this 
    data = f['data']
    e2 = time.time()
    print('build it')
    mSVD = parallel_SVD_MPI( data, 1000, P, mpi_comm, False, None) 
    print('go')
    u,s,v,sel = mSVD.go_svd()
    e3 = time.time()
    print('Core %i with batches version takes %12.3f seconds, while reading data from hdf5'%( mpi_comm.rank , e3-e2) )
    print(s, mpi_comm.rank )

    #lets read the data into memory
    print('Reading data into memory')
    data2 = f['data'].value #[:,:]
    print(data2.shape)
    print( 'done') 

    mpi_comm.Barrier()
    e2 = time.time()
    mSVD = parallel_SVD_MPI( data2, 500, P, mpi_comm, True)
    u,s,v,sel = mSVD.go_svd()
    e3 = time.time()
    print('Core %i with batches version takes %12.3f seconds, with data in memory'%(mpi_comm.rank , e3-e2) )

    mpi_comm.Barrier()
    selection = np.arange(1000)
    print('Testing setup with an included selection array')
    data2[1000,:]=0.
    mSVD = parallel_SVD_MPI( data2, 50000, P, mpi_comm, False, selection = selection)
    SVDs = batched_SVD.batched_SVD( )  
    up,sp,vp, order_split = mSVD.go_svd()
    uf,sf,vf = isvd(data2[0:1000,:].astype('float64'), P)
    assert np.mean( ( np.abs(sp-sf)/sf ) ) < 2e-2

    mpi_comm.Barrier()
    if mpi_comm.Get_rank() == 0:
        # now remove this file again
        print('Removing h5 data file')
        os.remove('test_data.h5')
        # done


    


if __name__ == "__main__":
    tst_batched()
