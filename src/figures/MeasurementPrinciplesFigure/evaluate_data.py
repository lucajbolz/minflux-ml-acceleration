"""
Evaluate data needed to reproduce the measurement principles figure.
"""
import os

import numpy as np
import time as time
import shutil

from src.scripts.ProcessLineScan import Processor
from lib.data_handling.mf_parser import process_batch
import lib.utilities as ut
from lib.config import ROOT_DIR, DATA_DIR

#------ Carry out simulation ------------------
if __name__=='__main__':
    base_dir = os.path.join(ROOT_DIR, 'src/figures/MeasurementPrinciplesFigure/data')
    if False:
        base_in = os.path.join(DATA_DIR, 'LineScans/raw/50nm/')
        p = Processor()
        try:#select and copy exemplary file
            name = 'GQ50001_phase_scans_v00078_20230327'
            files = ut.BaseFunc().find_files(base_in, lambda file: ut.BaseFunc().match_pattern(file, '.tif', match='partial'), max_files=np.inf)
            source_tif = [file  for file in files if name in file][0]
            if os.path.isfile(source_tif):
                shutil.copy(source_tif, base_dir)
                dir_n,f_n = os.path.split(source_tif)
                old_path = os.path.join(base_dir,f_n)
                new_n = 'source.tif'
                new_path = os.path.join(base_dir,new_n)
                os.rename(old_path,new_path) 
        except:
            print('unable to copy new local results.')

    if True:
        # process the 20nm batch of static MINFLUX data, use the pre-parsed and pre-filtered data!
        base_in = os.path.join(DATA_DIR, 'MINFLUXStatic/')
        parser_type = 'otto'
        bootstrap_dicts = [
            dict(mode='photons', chunk_size=None, overlap=0., bin_size=None,max_chunks=1)|dict(estimator='min-quad', plot=False)
            ]        
        new_dir = process_batch(base_in,parse=False,parser_type=parser_type,filter=False,postfilter=False,process=True,processing_key='20nm',bootstrap_dicts=bootstrap_dicts,postprocess=False,max_files=np.inf)
        
        try:# copy source files, select specific Nanoruler to plot here...
            name = '-mmGQ20001_MF_scans_i1_14-'
            candidates = []
            candidates.append(ut.BaseFunc().find_files(base_in, lambda file: file.endswith('.pkl'), max_files=np.inf))
            candidates.append(ut.BaseFunc().find_files(new_dir, lambda file: file.endswith('.pkl'), max_files=np.inf))
            candidates = ut.BaseFunc().flatten_list(candidates)
            files = [file for file in candidates if (name in file) & (('parsed' in file)|('filtered' in file)|('local-results' in file))]
            for file in files:
                if os.path.isfile(file):
                    shutil.copy(file, base_dir)
        except:
            raise Exception('unable to copy new local results.')