"""
Description: Simulation of Phase Scans and MINFLUX measurements
"""
import multiprocessing
import numpy as np
import time as time
from lib.simulation.experiment import *

import lib.utilities as ut

def generate_data(name='LineScan/',params = {},**kwargs):
    exp = ExperimentGenerator().get_default(**params)
    exp.perform(mode='real')
    #Converter().reshape_record(exp,export=False,show=True)
    if kwargs['folder'] is None:
        kwargs['folder'] = 'testLineScan'
    file_str, todays_dir = ut.Labeler().stamp('experiment')
    output_dir = todays_dir + name
    exp.save(filename=file_str+'.yaml', folder = output_dir + kwargs['folder']+'/')
    return output_dir

def func(name,arg1,arg2):
    func = generate_data(name=name,params=arg1,**arg2)
    return func

def execute_simulation(names,args1,args2):
    #for n, a1, a2 in zip(names,args1,args2):
    #    output = func(n,a1,a2)
    with multiprocessing.Pool() as pool:
        print(f"using a parallel pool with {pool._processes} processes")
        output = pool.starmap(func, zip(names,args1,args2))
    return list(set(output))


if __name__== '__main__':
    rng = np.random.default_rng()
    #------ Carry out simulation ------------------
    names, param_dicts, kwarg_dicts = [], [], []
    name = ut.Labeler().id_generator() + '-LineScanSimulation'
    name = ut.Labeler().id_generator() + '-MINFLUXSimulation'
    for d in range(0,21,50):
        kwargs = dict(folder=f'/{d}nm')
        #params = {'type':'line-scan','block_size':1, 'iteration_depth':1,'repetitions':30}
        params = {'type':'minflux','L':150.,'repetitions':20}

        for j in range(1,3):
            p0 = np.array([[0.,0.]])# * 2 * (rng.random((1,2))-0.5)# np.array([[-40.,0.]])
            vec = (rng.random((2,))-0.5)##np.ones((2,))#
            n = vec/np.linalg.norm(vec)
            p1 = p0 - float(d)*n/2
            p2 = p0 + float(d)*n/2
            mol_pos = np.concatenate([p1,p2],axis=0)

            params = params | dict(
                beta=5E-1
                ,blinking=False
                #,bleach_idx=np.array([10,20])
                ,bleach_idx=np.array([700,1400])#for minflux: keep in mind that we have some pre-localization steps, i.e. more scan units...
                ,photon_budget=[np.inf, np.inf]
                ,molecule_brightness=35
                ,mol_pos=mol_pos
                )
            param_dicts.append(params)
            kwarg_dicts.append(kwargs)
            names.append(name)
            j+=1
    
    execute_simulation(names,param_dicts,kwarg_dicts)