"""
Module that provides functionality in order to filter data.

@author: Thomas Arne Hensel, 2023
"""
import os, multiprocessing
import numpy as np
from collections import Counter
from shutil import copyfile
from functools import partial
import copy as copy
import pandas as pd

from lib.data_handling.data_preprocessing import Extractor
from lib.data_handling.data_converter import Constructor
from lib.data_handling.utilities import Converter
from lib.plotting.artefacts import Artefacts, Filter_Figures
import lib.utilities as ut

class Filter:
    """
    Class for filtering data based on certain criteria.

    Attributes:
    - ext (Extractor): Extractor instance for data extraction.
    - artefacts (Artefacts): Artefacts instance for storing figures and other artifacts.
    - collect_artefacts (bool): Flag indicating whether to collect and store artifacts.

    Methods:
    - filter_data(in_folder, base_out, max_files=100, batchsize=1, agnostic=True, fast_return=False):
      Iterate through a nested directory and filter data based on certain criteria.
    """

    def __init__(self, collect_artefacts=True)->None:
        """
        Constructor for the Filter class.

        :param collect_artefacts: Flag indicating whether to collect and store artifacts.
        :type collect_artefacts: bool, optional, default is True.
        """
        self.ext = Extractor()
        self.artefacts = Artefacts()
        self.collect_artefacts = collect_artefacts
        pass

    def filter_data(self, in_folder, base_out, max_files=100, batchsize=1, agnostic=True, fast_return=False):
        """
        Iterate through a nested directory and filter data based on certain criteria.

        :param in_folder: Root of the input directory.
        :type in_folder: str
        :param base_out: Root of the output directory.
        :type base_out: str
        :param max_files: Maximum number of files to be analyzed in one directory.
        :type max_files: int, optional, default is 100.
        :param batchsize: Size of moving average.
        :type batchsize: int, optional, default is 1.
        :param agnostic: Specify whether construction is agnostic or not.
        :type agnostic: bool, optional, default is True.
        :param fast_return: Optional direct return of experiment object.
        :type fast_return: bool, optional, default is False.
        """
        try: # try to find source files, if operated on processed data
            processable_files = ut.BaseFunc().find_files(in_folder, lambda file: (file.endswith('.tif') and 'source' in file), max_files=max_files )
        except:
            pass
        if len(processable_files)==0: # in case no sourcefiles are found, check for other data, e.g. raw traces or yaml files
            processable_files = ut.BaseFunc().find_files(in_folder, lambda file: ((file.endswith('.tif') or file.endswith('.tiff')) and 'c003' in file) or file.endswith('.yaml'), max_files=max_files )
        
        task = partial(filter_single_file, base_out=base_out, batchsize=batchsize, agnostic=agnostic, fast_return=fast_return, collect_artefacts=self.collect_artefacts)
        
        #task(processable_files[0])
        #for file in processable_files:
        #    task(file)
        num_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_cores) as pool:
            print(f"using a parallel pool with {pool._processes} processes")
            reasons = pool.map(task, processable_files)                 
        
        if self.collect_artefacts:
            fig, _ = Filter_Figures().fig_filter_statistics(reasons)
            self.artefacts.add_figures([fig],['filter_results'])
            self.artefacts.save_figures(meta={}, out_dir = base_out)
            df = pd.DataFrame.from_dict(Counter(reasons), orient='index', columns=['count'])
            df.to_csv(base_out+'filter-results.csv',index=True)
        pass

def filter_single_file(full_file_path, base_out, batchsize = 1, agnostic = True, fast_return =False, collect_artefacts=True):
    """
    Filter a single file based on certain criteria.

    :param full_file_path: Full path to the input file.
    :type full_file_path: str
    :param base_out: Root of the output directory.
    :type base_out: str
    :param batchsize: Size of moving average.
    :type batchsize: int, optional, default is 1.
    :param agnostic: Specify whether construction is agnostic or not.
    :type agnostic: bool, optional, default is True.
    :param fast_return: Optional direct return of experiment object.
    :type fast_return: bool, optional, default is False.
    :param collect_artefacts: Flag indicating whether to collect and store artifacts.
    :type collect_artefacts: bool, optional, default is True.
    :return: Reason for rejection or None if the file is accepted.
    :rtype: str
    """
    curr_constructor = Constructor(full_file_path,collect_artefacts=collect_artefacts)
    experiments = curr_constructor.get_experiments(batchsize=batchsize, agnostic=agnostic, fast_return=fast_return)
    # check if experiments comply with filter criteria
    try:
        curr_reason = filter_experiments(experiments)
    except:
        print('error during filtering of file')
        curr_reason = 'filter'
    n_f, fname = os.path.split(full_file_path)
    f_base, f_extension = os.path.splitext(fname)
    curr_folder = os.path.split(n_f)[1] +'/'
    new_dir = base_out + curr_folder# folder of one batch/size of nanorulers
    os.makedirs(new_dir,exist_ok=True)
    if curr_reason=='None':
        os.makedirs(new_dir + f_base + '/', exist_ok=True)
        copyfile(full_file_path, new_dir + f_base + '/' + 'source' + f_extension) # copy original data into new directory
        ut.Exporter().write_yaml(experiments, filename='experiments.yaml', out_dir = new_dir + f_base + '/') # save experiments
        try:
            curr_constructor.ext.artefacts.save_figures(meta={}, out_dir= new_dir + f_base + '/')
        except:
            print('error in figure production')
    else:
        os.makedirs(new_dir + 'REJECTED_' + f_base + '/', exist_ok=True)
        copyfile(full_file_path, new_dir + 'REJECTED_' + f_base + '/' + 'source' + f_extension)
        ut.Exporter().write_yaml(experiments, filename='experiments.yaml', out_dir = new_dir + 'REJECTED_' + f_base + '/')
        try:
            curr_constructor.ext.artefacts.save_figures(meta={}, out_dir= new_dir + 'REJECTED_' + f_base + '/')
        except:
            print('error in figure production of rejected experiment')
    return curr_reason

def filter_experiments(experiments):
    """
    Check whether experiments comply with filter criteria.

    :param experiments: List of Experiment objects.
    :type experiments: list
    :return: Reason for rejection or None if the experiments are accepted.
    :rtype: str
    """
    reason = None
    # TODO: check whether tif as specs from param-file, avoid import of Extractor() above!
    if experiments is None or np.any([exp is None for exp in experiments]):
        return 'invalid_experiment'
    n = len(experiments)
    if n!=3: # check number of segments
        reason = 'n_seg'
        return reason
    imarrays = []
    for exp in experiments:
        new_exp = copy.deepcopy(exp)
        imarray = Converter().reshape_record(new_exp,export=False).photons
        imarrays.append(imarray)
    
    shapes, means, line_means = [], [], []
    for arr in imarrays:
        shapes.append(arr.shape)
        means.append(np.mean(arr))
        line_means.append(np.mean(arr,axis=1)) 
    if np.any([t[1] != 160 for t in shapes]): #TODO: flexibly set pixel count
        reason = 'shape'
        pass
    else:
        if np.any([t[0] < 1 for t in shapes]):
            reason = 'len_seg'
            pass
        else:             
            # check if 1M signal is 1/2 2M signal
            ratio = means[0]/(2*(means[1]))
            if ratio > 1.5 or ratio < 0.5:
                reason = 'ratio_N'
                pass
            else:
                # check homogeneity of background/variance/fluctuation
                std = np.array([np.std(m) for m in line_means])
                if np.any(std > 10*(np.sqrt([np.mean(line_mean)/np.sqrt(len(line_mean)) for line_mean in line_means]))): # check whether std deviation of line_means is within 3 sigma (re-scaled by number of lines that have been averaged)
                    reason = 'var_N'
                    pass
                else: # save experiments (configs, records) in output_folder_filtered if provided, otherwise input_folder_filtered
                    reason = 'None'
    return reason

class MinfluxFilter:
    """
    Class for filtering Minflux data based on certain criteria.

    Attributes:
    - ext (Extractor): Extractor instance for data extraction.
    - artefacts (Artefacts): Artefacts instance for storing figures and other artifacts.
    - collect_artefacts (bool): Flag indicating whether to collect and store artifacts.

    Methods:
    - __init__(collect_artefacts=True): Constructor for the MinfluxFilter class.

    """
    
    def __init__(self, collect_artefacts=True)->None:
        """
        Constructor for the MinfluxFilter class.

        :param collect_artefacts: Flag indicating whether to collect and store artifacts.
        :type collect_artefacts: bool, optional, default is True.
        """
        self.ext = Extractor()
        self.artefacts = Artefacts()
        self.collect_artefacts = collect_artefacts
        pass