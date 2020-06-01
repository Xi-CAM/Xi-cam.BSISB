import numpy as np

from lbl_ir.math_tools.batched_SVD import batched_SVD



def svd(data, k_singular, N_max=10000):
    svd_obj = batched_SVD( data.data,
                           N_max=N_max,
                           k_singular=k_singular,
                           randomize=True)
    U,S,VT = svd_obj.go_svd()
    return U,S,VT

    

