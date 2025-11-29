"""
Script in order to demonstrate processing of MINFLUX data: filtering, analysis and visualization.

@author: Thomas Arne Hensel, 2023
"""
import os
import numpy as np

from lib.config import DATA_DIR
from lib.data_handling.mf_parser import process_batch

if __name__=='__main__':
    base_in = os.path.join(DATA_DIR,'MINFLUXStatic/')
    #base_in = os.path.join(DATA_DIR,'MINFLUXDynamic/')
    
    parser_type = 'otto'
    
    bootstrap_dicts = [
        dict(mode='photons', chunk_size=3E3, overlap=0., bin_size=3E3,max_chunks=5E0)|dict(estimator='min-poly', plot=False, output=None)
        #,dict(mode='photons', chunk_size=4E3, overlap=0., bin_size=4E3,max_chunks=5E3)|dict(estimator='min-poly', plot=False, output=None)
        #,dict(mode='photons', chunk_size=5E3, overlap=0., bin_size=5E3,max_chunks=5E3)|dict(estimator='min-poly', plot=False, output=None)
        #,dict(mode='photons', chunk_size=1E4, overlap=0., bin_size=1E4,max_chunks=5E3)|dict(estimator='min-poly', plot=False, output=None)
        ]
    new_dir = process_batch(base_in,parse=True,parser_type=parser_type,filter=True,postfilter=True,process=False,bootstrap_dicts=bootstrap_dicts,postprocess=False,max_files=np.inf)    
