import numpy as np
import sys

from lbl_ir.io_tools.Omnic_PyMca5 import OmnicMap
from lbl_ir.data_objects import ir_map


def read_and_convert(filename, start_wav=None, stop_wav=None, data_type="absorbance", sample_info=None):
    if sample_info is None:
        sample_info = ir_map.sample_info()

    omnic_object       = OmnicMap.OmnicMap( filename ) 
    wavenumbers = None
    if wavenumbers is None:
        n_wav = omnic_object.data.shape[2]
        if (start_wav is None) or ( stop_wav is None) :
            if omnic_object.info['OmnicInfo'] is not None:
                start_wav = omnic_object.info['OmnicInfo']['First X value']
                stop_wav  = omnic_object.info['OmnicInfo']['Last X value']
                n_wav     = omnic_object.info['OmnicInfo']['Number of points']
            else:
                raise ValueError('There is an issue with the Ominc file or its parser. Make sure that the meta info from the header is present and parsed correctly.')

    wavenumbers      = np.linspace( start_wav, stop_wav, n_wav )
    image_grid_param = omnic_object.info['OmnicInfo']['Mapping stage parameters'] 
    image_mask       = np.ones( omnic_object.data.shape[0:2]) > 0.5 

    # build the basis object
    this_ir_map = ir_map.ir_map( wavenumbers = wavenumbers,
                                 sample_info = sample_info,
                                 data_type   = data_type 
                               )       
    this_ir_map.add_image_cube(omnic_object.data, image_mask, image_grid_param) 
    return this_ir_map

if __name__ == "__main__":
    
    map_name = '190519_N2_L2w1_mp2' + '.map'
    file_name =  '../../ir_data/' + map_name
    sample_id = file_name
    sample_id = sample_id.replace("../","")
    sample_id = sample_id.replace(".map","")
    sample_id = sample_id.replace("/","_")
    sample_id = sample_id.replace(" ","_")
    print(sample_id) 

    sample_info = ir_map.sample_info( sample_id = sample_id, sample_meta_data='Hello World')
    data = read_and_convert(file_name, sample_info=sample_info)
    data.write_as_hdf5('../../ir_data/' + map_name + '.h5')

 
