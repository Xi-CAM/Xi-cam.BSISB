import numpy as np
import h5py
import datetime
import sys
import os
import matplotlib.pyplot as plt  

def val2ind(val, an_array):
    return np.argmin(abs(an_array-val), axis=0)

class sample_info(object):
    """
    A simple class that contains sample info.

    Arguments:
    ----------

    sample_id            : A string that identifies the sample, spaces will be
                           substituted for underscores.
                           Specifying this is mandatory.
    
    sample_meta_data     : More verbose description of the data. Having this 
                           structured isn't a bad idea, but not enforced at this 
                           level. Not requiered but highly encouraged.

    sample_date          : The date at which the data was taken. Default is 
                           today.

    Attributes:
    -----------

    show(out)            : prints the contents of id, meta data and date to out.
                           If out is None, it defaults to sys.stdout 

    Examples:
    ---------

    si = sample_info(sample_id = 'C_elegans', 
                     sample_date='2004_04_18', 
                     sample_meta_data='Some details.')
    si.show()    

    """
    def __init__(self, sample_id="Unknown", sample_meta_data="None", sample_date=None):
        assert ' ' not in sample_id
        self.sample_id        = sample_id

        self.sample_meta_data = sample_meta_data
        if sample_date is None:
            sample_date =     str(datetime.date.today().year)+ \
                          '_'+str(datetime.date.today().month)+ \
                          '_'+str(datetime.date.today().day)

        self.sample_date      = sample_date

    def show(self, out=None):
        if out is None:
            out = sys.stdout
        print("Sample_id   : %s "%self.sample_id, file=out)
        print("Sample date : %s "%self.sample_date, file=out )
        print("Sample description:\n ", self.sample_meta_data, file=out )


class ir_map(object):
    """ A simple data object that contains IR data.

    Arguments:
    ----------
    wavenumbers : An array of wavenumbers

    sample_info : A sample_info object
    
    filename : The hdf5 filename where data will be read from. Required for 
                  the 'hdf5' mode.

    data_type   : Either 'transmission', 'reflection', or 'absorbance'(default)

    _mode        : Choice between 'memory' or 'hdf5'
                  If 'memory', the dataset currenly resides in memory
                  If 'hdf5', the dataset currenly resides in an hdf5 file 

    _N_obs       : The number of rows in the 2D spectral matrix.  

    Attributes:
    -----------
    As above plus

    add_data(self, spectrum, xy)  : add data, provide a numpy array of spectra
                                    and xy positions 

    _allocate_space(self)         : this is a function that allocates space for
                                    an hdf5 file. no need to call it yourself.

    write_as_hdf5(self, filename) : Writes in-memory data as an hdf5 file.

    """

    def __init__(self, 
                 wavenumbers = [],
                 sample_info = sample_info(), 
                 filename = None, 
                 data_type = 'absorbance',
                 with_image_cube = False, 
                 with_factorization = False):
        self.wavenumbers = wavenumbers
        self.N_w         = len(wavenumbers)
        self.sample_info = sample_info
        self._h5_filename = filename
        self.xy          = np.empty( (0,2) )
        self.data        = np.empty( (0,self.N_w) )
        self._N_obs      = 0
        self._with_image_cube = with_image_cube
        self._with_factorization = with_factorization
        
        assert data_type in ['transmission', 'reflection', 'absorbance']
        self.data_type   = data_type
        
        if self._h5_filename is not None: # hdf5 mode, load data from hdf5 file
            self._mode = 'hdf5'
            
            self._h5= h5py.File(self._h5_filename,'r')
            with self._h5:
                self._root = list(self._h5.keys())[0] #get sample root group name
                self.sample_info.sample_id = self._h5[self._root+'/info/sample_id'][()]
                self.sample_info.sample_meta_data = self._h5[self._root+'/info/sample_meta_data'][()]
                self.sample_info.sample_date = self._h5[self._root+'/info/sample_date'][()]
        else: # memory mode, data is in memory
            self._mode = 'memory'
            self._h5 = None # this stays None until we allocate space  
            
#            assert self.sample_info.sample_id != "Unknown", "Missing 'sample_info' keyword parameter" 
            self._root = str(self.sample_info.sample_id) + '_' + str(self.sample_info.sample_date)

    def add_data(self, spectrum=None, xy=None, ind=[]):
        """When working in memory mode, append the data in memory into the ir_map object.
        
           When working in hdf5 mode, load the data from the hdf5 file. 
           
        Arguments:
        
        ----------
        spectrum : The 2D spectral data matrix 
        
        xy       : An array of xy positions
        
        ind      : An array of indices for selecting specific rows in the 2D spectra matrix. 
                  If it is not given(default), full spectra matrix with all data points are loaded 
                  into the ir_map object, 1D int array
        
        """ 
        if self._mode == 'memory':
            assert spectrum is not None, "please provide a spectrum matrix"
            assert xy is not None, "please provide a xy position array"
            
            if len(ind) == 0:
                self.xy   = np.append(self.xy, xy, axis = 0)
                self.data = np.append(self.data, spectrum, axis = 0)
            else:
                self.xy   = np.append(self.xy, xy[ind,:], axis = 0)
                self.data = np.append(self.data, spectrum[ind,:], axis = 0)
                                    
        if self._mode == 'hdf5':
            self._h5= h5py.File(self._h5_filename,'r')
            with self._h5:
                self.wavenumbers = self._h5[self._root+'/data/wavenumbers'][:]
                if len(ind) == 0:
                    self.data = self._h5[self._root+'/data/spectra'][:,:]
                    self.xy   = self._h5[self._root+'/data/xy'][:,:]
                else:
                    self.data = self._h5[self._root+'/data/spectra'][ind,:]
                    self.xy   = self._h5[self._root+'/data/xy'][ind,:]  

    def add_image_cube(self, imageCube=None, imageMask=None, image_grid_param=None, ind=[]):
        """When working in memory mode, load the image cube data in memory into the ir_map object
           and flatten the image cube to 2d spectrum matrix.
        
           When working in hdf5 mode, load the data from the hdf5 file and flatten the image cube 
           to 2d spectrum matrix. 
           
        Arguments:
        
        ----------
        imageCube   : The spectral image cube, 3D float array 

        imageMask   : An image mask where non-blank pixels = True, 2D bool array
        
        image_grid_param : [x0, y0, dx, dy], 1D float list or array
        
        ind         : An array of indices for selecting wavenumber range. If it is not given(default), 
                      full spectrum are loaded into the ir_map object, 1D int array
        """ 
        self._with_image_cube = True
        
        if self._mode == 'memory':
            assert imageCube is not None, "please provide an image cube"
            assert imageMask is not None, "please provide an image mask matrix"
            assert image_grid_param is not None, "please provide image grid parameters : [x0, y0, dx, dy]"
            
            self.imageMask = imageMask
            self.image_grid_param = image_grid_param
            self.N_y, self.N_x = imageMask.shape[0], imageMask.shape[1]
            
            if len(ind) == 0:# read in full spectra
                self.imageCube = imageCube
            else:# read in partial spectra
                 self.imageCube = imageCube[:,:,ind]
                 assert len(ind) <= len(self.wavenumbers), "The selected wavenumber indices is longer than the full wavenumber range"
                 self.wavenumbers = self.wavenumbers[ind]
                 self.N_w = len(self.wavenumbers)
            # convert image cube to 2d data matrix and load the data into self.data, self.xy
            self.flatten_image_cube(imageCube, imageMask, image_grid_param)
                 
        if self._mode == 'hdf5':
            self._h5= h5py.File(self._h5_filename,'r')
            with self._h5:
                self.imageMask = self._h5[self._root+'/data/image/image_mask'][:,:]
                self.image_grid_param = self._h5[self._root+'/data/image/image_grid_param'][:]
                self.wavenumbers = self._h5[self._root+'/data/wavenumbers'][:]
                if len(ind) == 0:# read in full spectra
                     self.imageCube = self._h5[self._root+'/data/image/image_cube'][:,:,:]
                else:# read in partial spectrum
                     self.imageCube = self._h5[self._root+'/data/image/image_cube'][:,:,ind] 
                     assert len(ind) <= len(self.wavenumbers), "The selected wavenumber indices is longer than the full wavenumber range"
                     self.wavenumbers = self.wavenumbers[ind]
                self.N_w = len(self.wavenumbers)
                # convert image cube to 2d data matrix and load the data into self.data, self.xy    
                self.flatten_image_cube(self.imageCube, self.imageMask, self.image_grid_param)
    
    def add_factorization(self, component=None, component_coef=None, prefix='PCA', ind=[]):
        """When working in memory mode, load the factorized components data in memory into the ir_map object.
        
           When working in hdf5 mode, load the data from the hdf5 file. 
           
        Arguments:
        
        ----------
        component       : The spectral image cube, 3D float array 

        component_coef  : An image mask where non-blank pixels = True, 2D bool array
        
        prefix          : Name of the factorized components, e.g. PCA, MCR
        
        ind             : An array of indices for selecting specific rows in the 2D spectra matrix. 
                          If it is not given(default), full spectra matrix with all data points are loaded 
                          into the ir_map object, 1D int array
        """ 
        self._with_factorization = True
        self._factor_prefix = prefix + '_'
        
        if self._mode == 'memory':
            assert component is not None, "please provide a component matrix"
            assert component_coef is not None, "please provide a component_coef matrix"
            
            self.component = component
            self.N_component = component.shape[0]
            if len(ind) == 0:# read in all data
                self.component_coef = component_coef
            else:# read in partial data points
                self.component_coef = component_coef[ind,:]
        
        if self._mode == 'hdf5':
            self._h5= h5py.File(self._h5_filename,'r')
            with self._h5:
                self.component = self._h5[self._root + '/data/factorization/' + self._factor_prefix + 'component'][:,:]
                self.N_component = self.component.shape[0]
                if len(ind) == 0:  # read in all data
                    self.component_coef = self._h5[self._root + '/data/factorization/' + self._factor_prefix + 'component_coef'][:,:]
                else: # read in partial data points
                    self.component_coef = self._h5[self._root + '/data/factorization/' + self._factor_prefix + 'component_coef'][ind,:]
        # check the component dimensions match self.data dimensions
        assert self.component.shape[1] == self.data.shape[1], "number of wavenumbers in component does not match that of spectra matrix"
        assert self.component_coef.shape[0] == self.data.shape[0], "number of rows in component_coef does not match that of spectra matrix"
        
    def _allocate_space(self):
        
        self._h5= h5py.File(self._h5_filename,'w')
        data_group = self._h5.create_group( self._root )
        data_group.create_dataset( 'data/xy', 
                                   (self._N_obs, 2), 
                                   dtype='float32') # we just allocate space
        data_group.create_dataset('data/wavenumbers', 
                                  data = self.wavenumbers, 
                                  dtype='float32') # this we can keep in memory
        data_group.create_dataset('data/spectra', 
                                  (self._N_obs,self.N_w), 
                                  dtype='float32') # we just allocate space

        dt = h5py.special_dtype(vlen=str)
        data_group.create_dataset('info/sample_id',
                                  data = self.sample_info.sample_id,
                                  dtype= dt  )

        data_group.create_dataset('info/sample_meta_data',
                                  data = self.sample_info.sample_meta_data,
                                  dtype= dt  )

        data_group.create_dataset('info/sample_date',
                                  data = self.sample_info.sample_date,
                                  dtype= dt  ) 
        if self._with_image_cube:
            data_group.create_dataset('data/image/image_cube', 
                                      (self.N_y, self.N_x, self.N_w), 
                                      dtype='float32') # we just allocate space
            data_group.create_dataset('data/image/image_mask', 
                                      (self.N_y, self.N_x), 
                                      dtype='bool') # we just allocate space
            data_group.create_dataset('data/image/ind_rc_map', 
                                      (self._N_obs, 3), 
                                      dtype='int') # we just allocate space
            data_group.create_dataset('data/image/image_grid_param', 
                                      data = self.image_grid_param,
                                      dtype='float32') # this we keep in memory
                                      
        if self._with_factorization:
            data_group.create_dataset('data/factorization/' + self._factor_prefix + 'component', 
                                      (self.N_component, self.N_w), 
                                      dtype='float32') # we just allocate space
            data_group.create_dataset('data/factorization/' + self._factor_prefix + 'component_coef', 
                                      (self._N_obs, self.N_component), 
                                      dtype='float32') # we just allocate space

    def write_as_hdf5(self, filename):
        """Save the object out as an hdf5 file. 
        
        Arguments:
        
        ----------
        filename : The hdf5 filename where data will be written to. 
        
        """
        # prevent overwriting the existing hdf5 files 
        assert self._h5_filename != filename, \
        "The given hdf5 filename already exists. Please provide a different filename"
            
        self._h5_filename = filename
        self._N_obs = self.data.shape[0]
        self._allocate_space( )
        
        with self._h5:
            self._h5[self._root + '/data/xy'][:,:] = self.xy
            self._h5[self._root + '/data/spectra'][:,:] = self.data
            
            if self._with_image_cube: #save image cube
                self._h5[self._root + '/data/image/image_cube'][:,:,:] = self.imageCube
                self._h5[self._root + '/data/image/image_mask'][:,:] = self.imageMask
                self._h5[self._root + '/data/image/ind_rc_map'][:,:] = self.ind_rc_map
                self._h5[self._root + '/data/image/image_grid_param'][:] = self.image_grid_param
                
            if self._with_factorization: #save factorization
                self._h5[self._root + '/data/factorization/' + self._factor_prefix +'component'][:,:] = self.component
                self._h5[self._root + '/data/factorization/' + self._factor_prefix +'component_coef'][:,:] = self.component_coef
        
        print(f'Data is saved as an HDF5 file. Filename : {filename}')
            
    def to_image_cube(self, N_x=64, N_y=64, x0=0, y0=0, dx=1, dy=1):
        """Transform the spectra matrix to 3D image cube. 
        
           The third dimension of the image cube is the spectra data.  
        
        Arguments:
        ----------
        (x0, y0)    : The starting location of the ir image. 
        
        (N_x, N_y)  : The size of the image.
        
        (dx, dy)    : The step size of the image.
        
        Returns:
        --------
        imageCube   : The spectral image cube, 3D float array 

        imageMask   : An image mask where non-blank pixels = True, 2D bool array
        
        pointCounts : A mask that shows measurement counts in each pixel, 2D int array
        """
        
        self._with_image_cube = True
        
        self.N_x = N_x
        self.N_y = N_y
        self.image_grid_param = np.array([x0, y0, dx, dy], dtype='float32')
        
        self.imageCube = np.zeros((self.N_y, self.N_x, self.N_w), dtype='float32')
        self.pointCounts = np.zeros((self.N_y, self.N_x), dtype='int')
        ind_rc_map = np.zeros((self.xy.shape[0], 3), dtype='int') # ind to row-col mapping [i, row, col]
        
        x = np.arange(self.N_x)*dx + x0
        y = np.arange(self.N_y)*dy + y0
        
        for i in range(self.xy.shape[0]):
            ind_rc_map[i, 0] = i
            ind_rc_map[i, 1] = val2ind(self.xy[i, 1], y)  # align y coordinate to get row
            ind_rc_map[i, 2] = val2ind(self.xy[i, 0], x)  # align x coordinate to get col
            self.pointCounts[ind_rc_map[i, 1], ind_rc_map[i, 2]] += 1 # count how many measurements fall in a pixel 
            self.imageCube[ind_rc_map[i, 1], ind_rc_map[i, 2], :] += self.data[i,:]# add all spectra that fall in a pixel
        
        self.imageCube /= np.where(self.pointCounts != 0, self.pointCounts,1)[:,:,np.newaxis]# get average spectra per pixel
        self.imageMask = self.pointCounts.astype('bool') # convert to boolean matrix
        self.ind_rc_map = ind_rc_map
        
        return self.imageCube, self.imageMask, self.pointCounts
    
    def flatten_image_cube(self, imageCube, imageMask, image_grid_param):
        """Transform a 3D image cube into a spectra matrix using imageMask to filter out blank data points, 
        and load the matrix into self.data, load the xy positions of the data points into self.xy
        
        Arguments:
        ----------
        imageCube        : The spectral image cube, 3D float array 

        imageMask        : An image mask where non-blank pixels = True, 2D bool array
        
        image_grid_param : [x0, y0, dx, dy], 1D float list or array
        """
        self.N_x = N_x = imageCube.shape[1]
        self.N_y = N_y = imageCube.shape[0]
        x0 = image_grid_param[0]
        y0 = image_grid_param[1]
        dx = image_grid_param[2]
        dy = image_grid_param[3]
        # set up xy grid and use imageMask to pull out non-blank pixel xy-coordinate
        x = np.linspace(x0, x0+dx*(N_x-1), N_x)
        y = np.linspace(y0, y0+dy*(N_y-1), N_y)
        xv, yv = np.meshgrid(x, y)
        xy_grid = np.zeros((N_y, N_x, 2))
        xy_grid[:,:,0] = xv
        xy_grid[:,:,1] = yv
        
        # set up image grid and use imageMask to pull out non-blank pixel row, col position
        x = np.arange(N_x)
        y = np.arange(N_y)
        X, Y = np.meshgrid(x,y)
        ind_rc_map = np.zeros((len(X[imageMask]), 3), dtype='int') # ind to row-col mapping [i, row, col]
        for i, (r,c) in enumerate(zip(Y[imageMask], X[imageMask])):
            ind_rc_map[i,:] = [i, r, c]
        
        self.xy = xy_grid[imageMask,:]
        self.data = imageCube[imageMask,:]
        self.ind_rc_map = ind_rc_map

if __name__ == "__main__":
   si = sample_info(sample_id = 'C_elegans')
   si.show()   
   #prespare sample data
   np.random.seed(3)
   N_wav = 100
   N_obs = 100
   waves = np.linspace(500,4000,N_wav)
   data = np.random.uniform(0,1, (N_obs, N_wav) )    
   xy   = np.random.uniform(-5,5, (N_obs, 2) )
   imageCube = data.reshape(-1, 10, N_wav)
   imageMask = np.random.random((N_obs//10, 10)) > 0.5
   x0, y0, dx, dy = 0, 0, 1, 1
   image_grid_param = [x0, y0, dx, dy]
   component = np.random.random((3, N_wav))
   component_coef = np.random.random((imageMask.sum(), 3))
   # test loading 2d spectra matrix and writing into hdf5 file
   ir_data = ir_map( waves, si)
   ir_data.add_data( data, xy )
   ir_data.write_as_hdf5('tst_file.h5')
   
   # test loading 2d spectra matrix from a hdf5 file
   ir_data2 = ir_map(filename ='tst_file.h5' )
   ir_data2.add_data() #load full data matrix
   print(ir_data2.data.shape)
   ir_data2.add_data(ind=np.arange(20)) #load partial data matrix
   print(ir_data2.data.shape)
   os.remove('tst_file.h5')
   
   # test loading image cube data 
   ir_data3 = ir_map( waves, si)     
   ir_data3.add_image_cube(imageCube, imageMask, image_grid_param)
   assert ir_data3.data.shape[0] == imageMask.sum(), "number of rows in ir_data3.data doesn't match non-blank pixels in imageMask"
   r , c= np.where(imageMask)
   assert np.all(ir_data3.xy[:,0] == c), "x coordinates in ir_data3.xy doesn't match that of non-blank pixels in imageMask"
   assert np.all(ir_data3.xy[:,1] == r), "y coordinates in ir_data3.xy doesn't match that of non-blank pixels in imageMask"
   
   # loading factorization component
   ir_data3.add_factorization(component, component_coef)
   
   # writing into hdf5 file
   ir_data3.write_as_hdf5('tst_file2.h5')
   # show hdf5 file dataset structure
   lst=[]
   with h5py.File('tst_file2.h5','r') as f:
       root_name = list(f.keys())[0]
       h = f[root_name]
       f.visit(lst.append)
       ind_rc_map = f[root_name + '/data/image/ind_rc_map'][:,:]
   print(*lst, sep='\n')
   print(ind_rc_map[:20, :])
   plt.imshow(imageMask)
   
   # test loading image cube and factorization components from a hdf5 file
   ir_data4 = ir_map(filename='tst_file2.h5')
   ir_data4.add_image_cube()
   ir_data4.add_factorization()
   print(ir_data4.data.shape)
   print(ir_data4.component.shape)
   print(ir_data4.component_coef.shape)
   
   os.remove('tst_file2.h5')
    
   print('OK')
