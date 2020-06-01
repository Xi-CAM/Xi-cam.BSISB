import matplotlib.pyplot as plt

import sys
import numpy as np
from lbl_ir.io_tools import read_map
from lbl_ir.tasks.preprocessing.svd_data import svd




def run(filename, k_singular=200):
    data, format = read_map.read_all_formats( filename  )
    U,S,VT = svd(data, k_singular )
    plt.semilogy(S,'.-'); plt.show()


if __name__ == "__main__":
    run(sys.argv[1])
