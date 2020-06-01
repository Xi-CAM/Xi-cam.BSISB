"""
Here are some specific worm processing tools.

"""

import numpy as np
import matplotlib.pyplot as plt

# reading an omnic map
from lbl_ir.io_tools.Omnic_PyMca5.OmnicMap import OmnicMap

# doing SVD
from lbl_ir.math_tools.batched_SVD import batched_SVD

# for segmentating of the image
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_fill_holes
from scipy.ndimage import gaussian_filter

from skimage.morphology import skeletonize
from skimage import draw
import skimage
import pandas
from skan import csr
from scipy.ndimage.morphology import distance_transform_edt



class little_maker( object ):
    def __init__(self, filename, k_singular=10, z_threshold =14):
        self.filename    = filename
        self.k_singular  = k_singular
        self.z_threshold = z_threshold

        self.omnic_object = OmnicMap(filename) 
        if self.omnic_object.info['OmnicInfo'] is not None:
            start_wav = self.omnic_object.info['OmnicInfo']['First X value']
            stop_wav  = self.omnic_object.info['OmnicInfo']['Last X value']
            n_wav     = self.omnic_object.info['OmnicInfo']['Number of points']
        else:   
            start_wav = 699
            stop_wav  = 3999
            n_wav     = self.omnic_object.data.shape[-1]
        self.waves = np.linspace(start_wav, stop_wav, n_wav ) # I'm not sure where to find this in the data object!
        
        self.data_shape = self.omnic_object.data.shape
        self.flattened_data = self.omnic_object.data.reshape(-1, self.data_shape[2] )

        self.full_frame_svd()
        self.z_scores = self.get_background_map()

    def full_frame_svd(self,N_max=1000):
        svd_obj = batched_SVD( self.flattened_data, 
                               N_max=N_max, 
                               k_singular=self.k_singular, 
                               randomize=True)
        self.U,self.S,self.VT = svd_obj.go_svd()


    def skeletonize(self, threshold):
        fin_mask = (self.z_scores > threshold).astype(int)
        skel = skeletonize( fin_mask ).astype(int)
        sk_obj = csr.Skeleton(skel, spacing=1, keep_images=True )

        path_lengths     = sk_obj.path_lengths()
        this_one         = np.argmax(path_lengths)
        path_coordinates = sk_obj.path_coordinates(this_one).astype(int)

        pruned_skel      = np.zeros( fin_mask.shape )
        for pair in path_coordinates:        
            pruned_skel[ pair[0], pair[1] ] = 1
        return pruned_skel, path_coordinates[::-1,:]

    def blobber(self):
        None

    def extract_extrema(self, N_pixels=20, radius=20, threshold=14):
        skel, coords = self.skeletonize(threshold)
        side_0_coords = coords[ 0:N_pixels ]
        dxy = side_0_coords[0,:]- side_0_coords[-1,:] 
        angle_0 = np.arctan2(dxy[1], dxy[0])
        print(angle_0*180.0/3.14)
        print( side_0_coords )
        nimage = skimage.transform.rotate( self.z_scores , -angle_0*180.0/np.pi, clip=False, resize=True )
        plt.imshow(nimage); plt.show()

        side_1_coords = coords[ -(N_pixels): ]
        dxy = side_1_coords[0,:]- side_1_coords[-1,:]
        angle_1 = np.arctan2(dxy[1], dxy[0])
        print(angle_1*180.0/3.14)
        print( side_1_coords )
        nimage = skimage.transform.rotate( self.z_scores , -angle_1*180.0/np.pi, clip=False, resize=True )
        plt.imshow(nimage); plt.show() 



        



    def get_background_map(self, percentile=75.0, safety1=10, safety2=10, window_MM=6):
        """
        We first use a local roughness check to determine edges. We do this on the mean images, 
        but could easly use the full hypercube if need be.
        """


        # get a mean image
        mean_img    = self.U[:,0].reshape( self.data_shape[0:2] )
        mean_img    = mean_img - np.mean(mean_img)    

        # lets padd the image to avoid FFT periodicity issues
        X,Y = mean_img.shape
        # we need to fill this with some decent values
        fill_sigma = np.std(mean_img)
        fill_mean  = np.median( mean_img )
        new_img = np.random.normal( fill_mean, fill_sigma, (X+2*safety1,Y+2*safety1))
        # blur it a bit
        new_img = gaussian_filter( new_img, sigma=10)

        # paste the real image in here 
        new_img[ safety1:safety1+X,safety1:safety1+Y ] = mean_img
        mean_img = new_img+0


        # define a windowing function of MM by MM pixels
        MM     = window_MM 
        kernel = np.zeros( mean_img.shape )
        kernel[0:MM,0:MM] = 1.0

        # compute a local mean
        FT_img    = np.fft.fft2(mean_img)
        FT_kernel = np.fft.fft2(kernel)
        local_mean= np.fft.ifft2( FT_img* FT_kernel.conjugate() ).real / ( MM*MM)

        # compute a local variance
        mean_img_sq = mean_img * mean_img 
        FT_img_sq = np.fft.fft2(mean_img_sq)
        local_var = np.fft.ifft2( FT_img_sq* FT_kernel.conjugate() ).real / (MM*MM)
        local_var = local_var - local_mean*local_mean
        local_sigma = np.sqrt( local_var )

        ##############################################################
        ##           Now we build a rough mask
        ##############################################################
        threshold = np.percentile( local_sigma.flatten(), percentile )
        sel = local_sigma > threshold
        mask = np.zeros( mean_img.shape )
        mask[sel] = 1.0

        # The morphological operators need some room on the sides 
        safety2 = 0
        V,W = mask.shape
        nV = V+safety2*2
        nW = W+safety2*2
        new_mask = np.zeros( (nV,nW) )
        # place the mask
        new_mask[safety2:safety2+V, safety2:safety2+W ] = mask

        BM=5
        structure = np.ones((BM,BM))
        closed = binary_closing(new_mask,structure)

        done=False
        BM=5
        while not done:
            BM = BM + 1
            structure = np.ones((BM,BM))
            new_closed = binary_closing(closed,structure).astype(int)
            delta = np.sum( np.abs(new_closed.astype(int) - closed.astype(int)) )
            if delta == 0:
                done = True
            if BM >20:
                done = True
            closed = new_closed+0

        new_mask = closed+0
        
        # for some reason, the whole mask is shifted along one axis. This has likely to do with some 
        # origin definition of the structuring elements, but lets solve this using an FFT based shift calculation
        # The origin option in the scipy morphology toolbox kill my kernel

        mask = new_mask[safety2:safety2+V, safety2:safety2+W]
        ft_mask = np.fft.fft2(mask)
        ft_mean = np.fft.fft2(np.abs(mean_img) )
        TF = np.fft.ifft2( ft_mean*ft_mask.conjugate() ).real
        dX,dY = np.meshgrid( np.arange(mask.shape[1]), np.arange(mask.shape[0]) )
        here = np.argmax( TF )
        dX = dX.flatten()[here]
        dY = dY.flatten()[here]

        # no wild shifts please
        if dX > 10:
            dX = 0
        if dY > 10:
            dY = 0

        mask = np.roll( mask, dX, axis=0)
        mask = np.roll( mask, dY, axis=1)
        # lift out the section of the mask that we are interested in
        mask = mask[safety1:safety1+X, safety1:safety1+Y]


        ######################################################
        # Here we build a more fine tuned mask / z_score map
        ######################################################


        # now we use this mask to define the background
        bg_sel = mask.flatten() < 0.5
        background = self.U[bg_sel,:]
        mean_bg = np.mean( background, axis=0)
        var_covar = np.cov( (background-mean_bg).transpose() )
        inv_vcv = np.linalg.pinv( var_covar )
        t = self.U - mean_bg
        z_scores = []
        for tt in t:
            z = tt.reshape(1,-1)
            z_scores.append( np.sqrt( z.dot(inv_vcv).dot(z.transpose()) ) )
        z_scores = np.array(z_scores).reshape( self.data_shape[0:2] )
        return z_scores

 





