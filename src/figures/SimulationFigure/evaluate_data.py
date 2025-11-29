"""
Script to analyze the data necessary for the MINFLUX experiment figure.

copyright: @Thomas Hensel, 2023
"""

import os
import shutil
import copy
import numpy as np
import matplotlib as mpl
import numpy as np

from lib.simulation.MINFLUXMonteCarlo import MCSimulation
import lib.utilities as ut
from lib.config import ROOT_DIR


mpl.use('Agg')

if __name__=='__main__':

    base_dir = os.path.join(ROOT_DIR,'src/figures/SimulationFigure/data')

    p0 = dict(
            I0=20
            ,M0=2
            ,P0=2
            ,N0=50
            ,K0=300
            ,d0=30
            ,gamma0=60
            ,r0=0.0
            ,alpha0=1.
            ,sig_alpha0=.0
            ,beta0=0.1
            ,L0=30
            ,config='line'
            )
    d0 = dict(
        mode='photons'
        ,chunk_size=5000
        ,overlap=0.0
        ,bin_size=5000
        ,max_chunks=1
        ,plot=False
        ,output=None
        )
    d0 = d0|dict(solver='powell')
    configs=['grid','line','polygon']
    
    params = []
    for c in configs:
        for d in range(1,31,1):
            for m in [2,3,4,5]:#range(2,11,2):
                t = copy.deepcopy(p0)
                t['config'] = c
                t['d0'] = d
                t['M0'] = m
                t['I0'] = p0['I0']/np.sqrt(m)
                params.append(t)

    bootstrap_dicts = []
    for n in [500]:#[100,200,300,400,500,600,700,800,900,1000,2000,3000]:#,1000,2000,3000,4000,5000]:
        t = copy.deepcopy(d0)
        t['chunk_size'] = n
        t['bin_size'] = n
        bootstrap_dicts.append(t)
        
    new_n = 'multiflux.pkl'
    new_dir = MCSimulation().run(params=params,bootstrap_dicts=bootstrap_dicts)
    # overwrite analysis results
    try:
        result = ut.BaseFunc().find_files(os.path.join(new_dir,'post_processed/'), lambda file: ut.BaseFunc().match_pattern(file, 'all-postprocessing-results', match='partial'), max_files=np.inf)[0]
        if os.path.isfile(result):
            dir_n,f_n = os.path.split(result)
            new_path = os.path.join(dir_n,new_n)
            os.rename(result,new_path)
            shutil.copy(new_path, base_dir)
    except:
        print('unable to copy new local results.')