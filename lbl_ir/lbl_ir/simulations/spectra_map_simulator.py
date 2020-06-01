# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 16:14:47 2019

@author: Liang Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from lbl_ir.data_objects.ir_map import sample_info, ir_map

def lorentzian(x,x0,gamma=10):
    
    return 1/(np.power((x-x0)/gamma,2)+1)

def gaussian(x,x0=0,sigma=10):

    return 1/np.exp(np.power((x-x0)/sigma,2))

class spectra_map_simulator:

    """Generate a simulated spectral map

    Parameters:
    -----------
    NbaseSpectra: int, optional
        The number of basis spectrum. Default is 3.
    
    Nclusters: int, optional
        The number of data point clusters in the spectral map. Default is 4.
        
    ptsPerCluster: int, optional
        The number of data points per cluster. Default is 50.
        
    Nx: int, optional
        The pixel numbers of x-axis. Default is 64.
    
    Ny: int, optional
        The pixel numbers of y-axis. Default is 64.
        
    cov: int or float matrix, optional
        The covariance matrix of 2D gaussian distribution used to generate random data point clusters. 
        Default is [[20, 10], [10, 25]].
        
    sigma: int or float, optional
        The standard deviation of 1D gaussian distribution used to generate spectral weights. Default is 10.
        
    startWavenumber: int or float, optional
        The beginning wavenumber of spectral range. Default is 400.
        
    endWavenumber: int or float, optional
        The ending wavenumber of spectral range. Default is 4000.
        
    Nwavenumber: int, optional
        The number of wavenumber values in the spectrum. Default is 1600.
        
    random_state: int, optional
        The seed of the pseudo random number generator to use when generating random numbers. Default to 17.

    Attributes:
    --------
    data : float matrix
        The final spectral data matrix. Number of rows equals number of non-zero data points; 
        Number of columns equals number of wavenumbers. 
    
    densityMatCondense : float matrix
        The distribution weight matrix of each basis spectrum (only stores non-zero data points, see nonZeroInd).
        Number of rows equals number of non-zero data points; 
        Number of columns equals number of basis spectrum.
        
    spectraMat : float matrix
        The basis spectra matrix. Number of rows equals number of basis spectrum; 
        Number of columns equals number of wavenumbers.
        
    wavenumber : float array
        The wavenumber values of the spectrum (x-axis).
        
    nonZeroInd : int array
        The linear index of all non-zero data points in the flatterned 2D map. 

    Examples:
    ---------
    >>> s = spectra_map_simulator(random_state=3)
    >>> s.spectra_map_gen()
    >>> print(s.data.shape)
    (446, 1600)
    """
    
    def __init__(self, NbaseSpectra=3, Nclusters=4, ptsPerCluster=50, Nx=64, cov=[[20, 10], [10, 25]], sigma=10, 
                 startWavenumber=400, endWavenumber=4000, Nwavenumber=1600, random_state=17):
        
        self.NbaseSpectra = NbaseSpectra
        self.Nclusters = Nclusters
        self.ptsPerCluster = ptsPerCluster
        self.Nx = Nx
        self.Ny = Nx
        self.cov = cov
        self.sigma = sigma
        self.startWavenumber = startWavenumber
        self.endWavenumber = endWavenumber
        self.Nwavenumber = Nwavenumber
        self.random_state = random_state
    
    def cluster_placement(self, random_state=17):
        """Generate simulated data point locations and weights in a 2D map
        
        Returns:
        --------
        points : float matrix 
            The data point location matrix. The first two columns are x-y coordiantes of data points. 
            The third column are the weight coefficients of data points.
    
        densityVec : float array
            The 1D array generated from flattened 2D weight coefficients matrix (densityMat)
        """

        np.random.seed(seed=random_state)
        centroids = np.random.randint(int(self.Nx*0.8), size=(self.Nclusters,2))
        self.points = np.zeros((self.ptsPerCluster*self.Nclusters, 3))
        densityMat = np.zeros((self.Ny, self.Nx))

        for i in range(self.Nclusters):
            self.points[i*self.ptsPerCluster:(i+1)*self.ptsPerCluster, :2] = np.random.multivariate_normal(centroids[i,:], self.cov, self.ptsPerCluster)
            self.points[i*self.ptsPerCluster:(i+1)*self.ptsPerCluster, 2] = gaussian(np.sqrt(np.power(self.points[i*self.ptsPerCluster:(i+1)*self.ptsPerCluster, :2]
                                                                                       -centroids[i,:],2).sum(axis = 1)),sigma=self.sigma)    

        for i in self.points:
            idx = tuple(np.clip(i[1::-1].astype(int), 0, self.Nx-1))
            densityMat[idx] += i[2]

        densityVec = densityMat.flatten()

        return densityVec
    
    def spectrum_gen(self, Npeaks=3, firstPeakPosition=1200, peakWidth=[30,60,150], random_state=17):
        """Generate a simulated spectrum
        
        Parameters:
        -----------
        Npeaks: int, optional
            The number of peaks. Default is 3.
            
        firstPeakPosition: int, optional
            The position of the first peak. Default is 1200.
            
        peakWidth: int or float list, optional
            The list of peak widths. Default is [30,60,150].
        
        Returns:
        --------
        wavenumber : float array 
            The wavenumber values of the spectrum (x-axis).
    
        spectrum : float array
            The transmission/reflection/absorption coefficients of the spectrum (y-axis)
        """
    
        if len(peakWidth) != Npeaks:
            raise Exception("The number of peak width values doesn't match the number of peaks")

        np.random.seed(seed=random_state)
        self.wavenumber = np.linspace(self.startWavenumber,self.endWavenumber,self.Nwavenumber)
        peakPositions = np.linspace(firstPeakPosition,self.endWavenumber,Npeaks,endpoint=False)
        spectrum = np.zeros(len(self.wavenumber))
        weights = np.random.rand(Npeaks)

        for i in range(Npeaks):
            singlePeak = lorentzian(self.wavenumber,peakPositions[i],peakWidth[i])
            spectrum += singlePeak * weights[i]
        spectrum /= weights.sum()

        return spectrum
    
    def spectra_map_gen(self, Npeaks = [3,4,5], positions = [800, 1000, 1200]):
        """Generate a spectral map data cube
        
        Parameters:
        -----------
        Npeaks: int list, optional
            The list of number of peaks. Default is [3,4,5].
            
        positions: int or float list, optional
            The list of first peak positions. Default is [800, 1000, 1200].
            
        Returns
        -------
        self : object
        """
        
        if len(positions) != len(Npeaks):
            raise Exception("The number of first peak positions doesn't match the number of basis spectrum")
            
        np.random.seed(seed=self.random_state)
        random_seeds = np.random.randint(50, size=len(Npeaks))
        
        densityVecs = np.zeros((self.Nx*self.Ny,self.NbaseSpectra)) # component coefficients
        self.spectraMat = np.zeros((self.NbaseSpectra, self.Nwavenumber)) # components spectra matrix

        for i in range(self.NbaseSpectra):
        
            densityVecs[:,i] = self.cluster_placement(random_state = random_seeds[i])

            np.random.seed(seed=random_seeds[i])
            peakWidth = np.random.randint(4,20, Npeaks[i])*10
            self.spectraMat[i,:] = self.spectrum_gen(Npeaks=Npeaks[i], firstPeakPosition = positions[i], 
                                                             peakWidth=peakWidth, random_state = random_seeds[i])

        self.nonZeroInd = ~np.all(densityVecs==0, axis=1)
        self.densityVecsCondense = densityVecs[self.nonZeroInd] # remove all zero rows
        self.data = self.densityVecsCondense.dot(self.spectraMat) # matrix mulplication 
        
        mask = np.zeros((self.Ny, self.Nx), dtype='bool').flatten()
        mask[self.nonZeroInd] = True
        self.mask = mask.reshape(self.Ny, self.Nx)
    
    def save(self, sample_id='simulated_dataset'):
        """Save the simulated dataset as an hdf5 file. 
        
        Arguments:
        
        ----------
        sample_id : A string that identifies the sample, spaces will be substituted for underscores.
        
        """
        si = sample_info(sample_id=sample_id, sample_meta_data=f"This dataset has {self.NbaseSpectra} components and map size is {self.Ny} * {self.Nx}")
        si.show() 
        
        self.spectra_map_gen() # generate simulated dataset
        y, x = np.where(self.mask)
        self.xy = np.c_[x,y]

        ir_data = ir_map(self.wavenumber, si, with_factorization=True)
        ir_data.add_data(spectrum=self.data, xy=self.xy)
        ir_data.to_image_cube()
        ir_data.add_factorization(component=self.spectraMat, component_coef=self.densityVecsCondense, prefix='MCR')
        ir_data.write_as_hdf5(f'{sample_id}.h5')
        
    def load(self, filename='simulated_dataset.h5'):
        """load the simulated dataset from an hdf5 file. 
        
        Arguments:
        
        ----------
        filename : The hdf5 filename where data will be read from. 
        """
        ir_data = ir_map(filename=filename)
        ir_data.add_image_cube()
        ir_data.add_factorization(prefix='MCR')
        
        return ir_data
        
    def plot_spectra_map(self):
        """Plot the spectral distribution map and basis spectra
         
        """
        
        self.spectra_map_gen() # generate simulated dataset
        
        plt.figure(figsize=(12, 6))
        for i in range(self.NbaseSpectra):
            
            n_row = 2
            
            plt.subplot(n_row,3,i+1)
            densityMat = np.zeros((self.Ny, self.Nx)).flatten()
            densityMat[self.nonZeroInd] = self.densityVecsCondense[:,i]
            densityMat = densityMat.reshape(self.Ny, self.Nx)
            plt.imshow(np.flipud(densityMat))
            plt.title(f'Distribution of Component {i+1}')
            plt.colorbar()
            plt.clim([0, 2])
            
            if n_row == 3:
                plt.subplot(n_row,3,i+7)
                plt.imshow(np.flipud(densityMat>0))

            plt.subplot(n_row,3,i+4)
            plt.subplots_adjust(hspace=0.3)
            plt.plot(self.wavenumber, self.spectraMat[i,:])
            plt.title(f'Spectrum of Component {i+1}')
            plt.xlim([4000,400])
            
        return None

if __name__ == "__main__":
    
    s = spectra_map_simulator(random_state=3)
    s.plot_spectra_map()