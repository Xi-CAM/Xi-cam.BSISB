import h5py
import sys

from lbl_ir.io_tools import read_omnic
from lbl_ir.data_objects import ir_map

def read_all_formats(filename, sample_info=None):

    ok = False
    format = None
    if not ok:
        try:
            data = read_omnic.read_and_convert(filename, sample_info=None)
            ok= True
            format = "Omnic"
        except: pass

    if not ok:
        try: 
            data = ir_map.ir_map(filename=filename)
            ok = True
            format = "hdf5"
            
            with h5py.File(filename,'r') as f:
                root_name = list(f.keys())[0]
               # if there is an image group, load imagecube and data, otherwise load data group
                if 'image' in f[root_name + '/data']:
                    data.add_image_cube()
                else:
                    data.add_data()
               # if there is an factorization group, load factorization data
                if 'factorization' in f[root_name + '/data']:
                    data.add_factorization()
        except: pass 
  


    if not ok:
        print("Could not read file; Check input or file formats and header integrity.") 
    else:
        return data, format



if __name__ == "__main__":
    data,fmt = read_all_formats(sys.argv[1])
    print("Data read in with format", fmt)
