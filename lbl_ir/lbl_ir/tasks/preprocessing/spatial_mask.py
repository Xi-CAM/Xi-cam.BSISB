import numpy as np

def peak_ratio_mask(ir_map, wave1, wave2, bg_threshold=1e-2, peak_threshold=0.05,operator = '<', delta=5):
    """
    Build a spatial mask on the basis of peak ratios.
    Data need to be in transmission mode.

    :param ir_map: image cube
    :param wave1: first wavenumber
    :param wave2: second wavenumber
    :param bg_threshold: threshold of background
    :param peak_threshold: threshold of peak
    :param operator: > or <
    :param delta: selects band size around provided peak values.
    :return:
    """
    sel1 = (ir_map.wavenumbers > wave1-delta) & (ir_map.wavenumbers < wave1+delta)
    sel2 = (ir_map.wavenumbers > wave2-delta) & (ir_map.wavenumbers < wave2+delta)
    map1 = np.mean( ir_map.imageCube[:,:,sel1], axis=2 )
    map2 = np.mean( ir_map.imageCube[:,:,sel2], axis=2 )

    mask1 = map1 > bg_threshold
    mask2 = map2 > bg_threshold
    bg_mask = mask1*mask2
    

    fin = map1 / map2
    if operator == '<':
        fin = fin < peak_threshold
    if operator == '>':
        fin = fin > peak_threshold
    fin = fin & bg_mask
    mask = np.zeros( ir_map.imageCube.shape[0:2] )
    mask[fin] = 1.0 
    return mask



    


