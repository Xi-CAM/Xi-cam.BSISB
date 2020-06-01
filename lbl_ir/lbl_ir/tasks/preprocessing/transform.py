import numpy as np
from lbl_ir.tasks.baseline import rubberband

def to_absorbance(data, wavenumbers, eps=1e-5, rubber_band = False):
    tmp = data /100.0
    sel_low = tmp < eps
    sel_high = tmp > 1-eps
    tmp[sel_low] = eps
    tmp[sel_high] = 1.0-eps
    result = -np.log(tmp)
    bg = result*0
    if rubber_band:
        shape = result.shape
        for ii in range(shape[0]):
            for jj in range(shape[1]):
                tmp = result[ii,jj,:]
                this_bg = rubberband.rubberband( wavenumbers, tmp )
                bg[ii,jj,:]=this_bg
        return result, bg
    else:
        return result,None
        
