"""
Evaluate data necessary to reproduce the MINFLUX experiment figure of dynamic systems.

copyright: @Thomas Hensel, 2024
"""

import os
import shutil
from copy import copy
script_name = os.path.basename(__file__)

import numpy as np
import matplotlib as mpl
import numpy as np

from lib.data_handling.mf_parser import process_batch
from lib.data_handling.data_analysis import MinfluxAnalysis
import lib.utilities as ut
from lib.constants import LAMBDA
from lib.plotting.style import Style
from lib.config import DATA_DIR, ROOT_DIR


mpl.use('Agg')

if __name__=='__main__':
    relevant_files = []
    base_dir = os.path.join(ROOT_DIR,'src/figures/DynamicDataFigure/data')
        
    # dynamic Nanoruler analysis
    base_in = os.path.join(DATA_DIR,'MINFLUXDynamic/raw')    
    parser_type = 'otto'
    bootstrap_dicts = [dict(mode='tuple', chunk_size=40, overlap=0.7, bin_size=5E3,max_chunks=1E3)|dict(estimator='min-poly', plot=False, output=None)]#0.6
    new_dir = process_batch(base_in,parse=True,parser_type=parser_type,filter=True,postfilter=True,process=True,bootstrap_dicts=bootstrap_dicts,postprocess=True,max_files=np.inf,visualize=True)

    files = ut.BaseFunc().find_files(new_dir, lambda file: (file.endswith('.pkl') and ('all-postprocessing-results' in file)),max_depth=2,max_files=1)
    dir_n,f_n = os.path.split(files[0])
    new_n = 'dynamic-minflux-data.pkl'
    new_path = os.path.join(dir_n,new_n)
    os.rename(files[0],new_path)
    relevant_files+=[new_path]

    try:
        for file in relevant_files:
            if os.path.isfile(file):
                #print(file)
                shutil.copy(file, base_dir)
    except:
        raise Exception('Failed to copy dynamic MINFLUX results!')