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
    base_dir = os.path.join(ROOT_DIR, 'src/figures/StaticDataFigure/data')

    # Line Scan Analysis
    if False:
        base_in = os.path.join(DATA_DIR, 'LineScans/')

        p = Processor()
        methods = ['FOURIER', 'HARMONIC', 'CORR','MIN-QUAD','MAX-QUAD','MIN-POLY']#['FOURIER', 'HARMONIC', 'CORR','MIN-QUAD','MAX-QUAD','MIN-POLY']#'MLE','WINDOW','KAPPA'
        new_base_dir = p.process_data(input_dir = base_in,  agnostic=True, filter=False, max_filter=np.inf, analyze=True, methods = methods, max_analyze=np.inf, postprocess=True, max_post=np.inf, visualize=False, collect_artefacts=False)

        files = ut.BaseFunc().find_files(new_base_dir, lambda file: (file.endswith('.pkl') and 'meta_results' in file))
        dir_n,f_n = os.path.split(files[0])
        new_n = 'line-scan-data.pkl'
        new_path = os.path.join(dir_n,new_n)
        os.rename(files[0],new_path)
        relevant_files+=[new_path]

    if True:        
        base_in = os.path.join(DATA_DIR, 'MINFLUXStatic/')
        parser_type = 'otto'    
        bootstrap_dicts = [
            dict(mode='photons', chunk_size=5E3,max_chunks=30,overlap=0.,bin_size=5E3)|dict(estimator='min-poly', plot=False, output=None)
            ]
        new_dir = process_batch(base_in,parse=False,parser_type=parser_type,filter=False,postfilter=False,process=True,bootstrap_dicts=bootstrap_dicts,postprocess=True,visualize=False,max_files=np.inf)    
        #new_dir = base_in

        files = ut.BaseFunc().find_files(new_dir, lambda file: (file.endswith('.pkl') and ('all-postprocessing-results' in file)),max_depth=2,max_files=1)
        dir_n,f_n = os.path.split(files[0])
        new_n = 'static-minflux-data.pkl'
        new_path = os.path.join(dir_n,new_n)
        os.rename(files[0],new_path)
        relevant_files+=[new_path]

    try:
        for file in relevant_files:
            if os.path.isfile(file):
                #print(file)
                shutil.copy(file, base_dir)
    except:
        raise Exception('Failed to copy static MINFLUX results!')