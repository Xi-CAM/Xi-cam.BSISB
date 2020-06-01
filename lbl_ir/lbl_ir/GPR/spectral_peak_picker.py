import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys

from lbl_ir.io_tools.read_map import read_all_formats
from lbl_ir.tasks.preprocessing.transform import to_absorbance
from lbl_ir.GPR.GPR_peaks import peak_picker


class spectral_peak_picker(object):
    def __init__(self,
                 wavenumbers,
                 spectrum,
                 sigma=None,
                 window=5,
                 peak_height_threshold=0.1,
                 peak_separation_threshold=3,
                 peak_prominence=0.001,
                 ):
        """

        :param wavenumbers: The wavenumbers
        :param spectrum: The IR spectrum (Absorbance)
        :param sigma: The associated standard deviation. If None, a suitable default will be chosen.
        :param window: A parameter that determines the window size when fitting peak using a GPR approach.
                       The default should be fine.
        :param peak_height_threshold: minimum peak height
        :param peak_separation_threshold: minimum distance between peaks
        :param peak_prominence: peak prominance
        """

        self.wavenumbers                = wavenumbers
        self.Nwavs                      = len(wavenumbers)
        self.spectrum                   = spectrum
        self.sigma                      = sigma
        self.window                     = window
        self.peak_height_threshold      = peak_height_threshold
        self.peak_separation_threshold  = peak_separation_threshold
        self.peak_prominence            = peak_prominence

        # first pass: find the peak using a simple approach
        self.raw_peaks_indx, self.peak_props    = find_peaks(self.spectrum,
                                                             height=self.peak_height_threshold,
                                                             distance=self.peak_separation_threshold,
                                                             prominence=self.peak_prominence )
        self.prominances = self.peak_props['prominences']

    def refine_peaks(self, level=0.05, sigma_multi=0.0005):
        these_peaks = self.prominances > level
        these_peak_indx = self.raw_peaks_indx[ these_peaks ]
        # now loop over these peaks and do the GPR stuff
        peak_locations = []
        peak_sigmas    = []
        peak_vals      = []
        val_sigmas     = []
        ok_flags       = []
        for ii in these_peak_indx:
            min_indx = max(ii-self.window,0)
            max_indx = min(ii+self.window+1,self.Nwavs)
            these_waves = self.wavenumbers[min_indx:max_indx]
            these_specs = self.spectrum[min_indx:max_indx] 
            obs_sigma   = np.sqrt(np.abs( these_specs ))*sigma_multi



            obj = peak_picker(X=these_waves.reshape(-1,1),Y=these_specs.reshape(-1,1),sY=obs_sigma)
            x_start = np.mean(these_waves.flatten())

            peak, sigma, val, val_sig = obj.peak_and_std_via_resample(x_start = x_start )
            if abs(peak - x_start) > 3*(these_waves[1]- these_waves[0]):
                peak = x_start
                sigma = 2*(these_waves[1]- these_waves[0])
                val = self.spectrum[ii]
                val_sig = -1
                ok_flags.append(False)
            else:
                ok_flags.append(True)

            peak_locations.append(peak)
            peak_sigmas.append(sigma)
            peak_vals.append(val)
            val_sigmas.append(val_sig)
        return np.array(peak_locations).flatten(), \
               np.array(peak_sigmas).flatten(), \
               np.array(peak_vals).flatten(), \
               np.array(val_sigmas).flatten(), \
               np.array(ok_flags).flatten()











def tst(filename):
    map, fmt = read_all_formats(filename)
    data,bg = to_absorbance( map.data,map.wavenumbers )
    waves = map.wavenumbers
    spec = np.mean(data, axis=0)
    #plt.plot(spec);plt.show()
    obj = spectral_peak_picker(waves, spec, sigma=0.05, peak_height_threshold=0.1)
    peaks, sigma, val, vs, ok  = obj.refine_peaks( 0.005)
    plt.plot(waves, spec,'.-', markersize=4)
    plt.plot(peaks,val,'x',markersize=8);plt.show()
    plt.plot( peaks, sigma, '.', markersize=4); plt.show()




if __name__ =="__main__":
    tst(sys.argv[1])
