"""
"""
import os
import numpy as np
import time as time
import shutil

from src.scripts.ProcessLineScan import Processor
from lib.data_handling.mf_parser import process_batch
import lib.utilities as ut
from lib.config import ROOT_DIR, DATA_DIR

if __name__ == '__main__':    
    relevant_files = []
    base_dir = os.path.join(ROOT_DIR, 'src/figures/ControlMethodFigure/data')

    base_in = os.path.join(DATA_DIR, 'LineScans/raw/50nm')

    p = Processor()
    methods = ['MIN-POLY']#['FOURIER', 'HARMONIC', 'CORR','MIN-QUAD','MAX-QUAD','MIN-POLY']#'MLE','WINDOW','KAPPA'
    new_dir = p.process_data(input_dir = base_in,  agnostic=True, filter=True, max_filter=np.inf, analyze=True, methods = methods, max_analyze=np.inf, postprocess=True, max_post=np.inf, visualize=False, collect_artefacts=False)

    name = 'GQ50001_phase_scans_v00078_20230327'
    files = ut.BaseFunc().find_files(os.path.join(new_dir,'filtered/50nm'), lambda file: name in file,max_depth=2,max_files=10)
    relevant_files.append(files)

    relevant_files = ut.BaseFunc().flatten_list(relevant_files)
    try:
        for file in relevant_files:
            if os.path.isfile(file):
                #print(file)
                shutil.copy(file, base_dir)
    except:
        raise Exception('Failed to copy static MINFLUX results!')