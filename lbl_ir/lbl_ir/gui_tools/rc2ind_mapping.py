import numpy as np


def getRC2Ind(imgShape, imageMask=None):
    """
    In a 2D image, get a dictionary mapping from (row, col) to linear index i of flattened image, and vice versa,
    :param imgShape: the shape of the image
    :param imageMask: if the image is sparse, use imageMask to generate the mapping
    :return:
    """
    if imageMask == None:
        imageMask = np.ones(imgShape) > 0
    N_x, N_y = imgShape[1], imgShape[0]
    x = np.arange(N_x)
    y = np.arange(N_y)
    X, Y = np.meshgrid(x, y)
    ind_rc_map = np.zeros((imgShape[0] * imgShape[1], 3), dtype='int')
    for i, (r, c) in enumerate(zip(Y[imageMask], X[imageMask])):
        ind_rc_map[i, :] = [i, r, c]
    # make a dictionary
    ind2rc = {x[0]: tuple(x[1:]) for x in ind_rc_map}
    rc2ind = {tuple(x[1:]): x[0] for x in ind_rc_map}
    return ind2rc, rc2ind