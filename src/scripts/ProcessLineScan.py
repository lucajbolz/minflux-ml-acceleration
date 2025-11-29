"""
Script in order to process LineScan data: filtering, analysis and visualization.

@author: Thomas Arne Hensel, 2023
"""
import os
import shutil
from pathlib import Path
import numpy as np
from itertools import chain

import lib.utilities as ut
from lib.utilities import FunctionLogger
from lib.data_handling.data_filtering import Filter
from lib.data_handling.data_analysis import EvaluationFacade
from lib.data_handling.data_postprocessing import PostProcessorFacade, MetaAnalysis
from lib.config import DATA_DIR

class Processor(FunctionLogger):

    def __init__(self, table_name = 'phase_scan_processing', data_base='log.sqlite', comment='test'):
        super().__init__(table_name = table_name, data_base=data_base, comment=comment)
        pass

    @FunctionLogger()
    def process_data(self, input_dir = '', agnostic=True, filter=True, max_filter=1, analyze=True, methods= [], max_analyze=1, postprocess=True, max_post=1, visualize=True, collect_artefacts=True, comment = 'test'):
        return self._process_data(input_dir = input_dir, agnostic=agnostic, filter=filter, max_filter=max_filter, analyze=analyze, methods=methods, max_analyze=max_analyze, postprocess=postprocess, max_post=max_post, visualize=visualize, collect_artefacts=collect_artefacts, comment = comment)

    def _process_data(self, input_dir = '', agnostic=True, filter=True, max_filter=1, analyze=True, methods= [], max_analyze=1, postprocess=True, max_post=1, visualize=True, collect_artefacts=True, comment = 'test'):
        """
        This method takes data from the input directory and copies it to the new session directory.
        If filter is True, the method assumes that the input is raw data and applies the filter routine.
        If filter is false, the method assumes already filtered data and does not apply the filter.
        The analyze option allows to analyze/fit the filtered data (which is hence known to be of sufficient quality).
        Filtered data can be analyzed again and again by choosing a directory with filtered data, set filter=False and analyze=True.
        :param input_base_dir: parent directory of data
        :param filter: filter data-> input must be raw data, or do not filter-> assume that input is already filtered
        :param analyze: apply fit to data
        """
        if not os.path.exists(input_dir):
            raise Exception('Input directory does not exist!')
        dir_str, todays_dir = ut.Labeler().stamp('session')
        new_base_dir = todays_dir + dir_str + '/'
        
        if filter:
            os.makedirs(new_base_dir + 'filtered/',exist_ok=True)
            filter = Filter(collect_artefacts=collect_artefacts)
            filter.filter_data(input_dir, new_base_dir + 'filtered/', max_files=max_filter, batchsize=1, agnostic=agnostic, fast_return=False)
        else: # assume that input is filtered
            # check if 'filtered/ is in str, i.e. we are looking at a subdirectory 'filtered/' of an input directory. Or, if we look at the base directory:
            dirs = [set(Path(x[0]).parts) for x in os.walk(input_dir)]
            parts = list(set().union(*dirs))
            # extract filename and ground truth size from folder name
            p = Path(input_dir)
            #parts = list(p.parts)
            pattern = 'filtered'
            if not np.any([ut.BaseFunc().match_pattern(string, pattern) for string in parts]):
                raise Exception('Please run the filter, since no filtered files could be found.')
            else:
                if analyze:
                    self.files_copied = 0
                    def ignore_func(d,files):
                        if self.files_copied>=max_analyze:
                            return files
                        keep = [f for f in files if (not (Path(d)/Path(f)).is_file() or f.endswith('.yaml'))]
                        relevant_files = [f for f in keep if ((Path(d)/Path(f)).is_file() and f.endswith('.yaml'))]
                        self.files_copied += len(relevant_files)
                        p = set(files)-set(keep)
                        return p
                    with ut.MultithreadedCopier(max_threads=16) as copier:
                        shutil.copytree(input_dir + 'filtered/', new_base_dir + 'filtered/', copy_function=copier.copy, ignore=ignore_func)
                    self.files_copied = 0
                else:
                    if postprocess:
                        #ignore_func = ut.FileIgnoreCounter(max_post,'.json')
                        #ignore_func = lambda d, files: [f for i,f in enumerate(files) if ((Path(d) / Path(f)).is_file() and not f.endswith('.json') and not i<=max_post)] # copy all result-files
                        self.files_copied = 0
                        def ignore_func(d,files):
                            if self.files_copied>=max_post:
                                return files
                            keep = [f for f in files if (not (Path(d)/Path(f)).is_file() or f.endswith('.json'))]
                            relevant_files = [f for f in keep if ((Path(d)/Path(f)).is_file() and f.endswith('.json'))]
                            self.files_copied += len(relevant_files)
                            p = set(files)-set(keep)
                            return p
                        with ut.MultithreadedCopier(max_threads=16) as copier:
                            shutil.copytree(input_dir + 'filtered/', new_base_dir + 'filtered/', copy_function=copier.copy, ignore=ignore_func)
                        self.files_copied = 0
        if analyze:
            os.makedirs(new_base_dir + 'analysis/',exist_ok=True)
            eval = EvaluationFacade(collect_artefacts=collect_artefacts, max_files=max_analyze)
            eval.evaluate_data(new_base_dir + 'filtered/', methods=methods, agnostic=agnostic)
        if postprocess:
            os.makedirs(new_base_dir + 'analysis/',exist_ok=True)
            postprocessor = PostProcessorFacade(collect_artefacts=collect_artefacts)
            postprocessor.postprocess_data(new_base_dir + 'filtered/', new_base_dir + 'analysis/', max_files=max_post)
        else:
            if visualize:
                try:
                    ignore_func = lambda d, files: [f for f in files if (Path(d) / Path(f)).is_file() and not f.endswith('.pkl')] # copy all result-files
                    with ut.MultithreadedCopier(max_threads=16) as copier:
                        shutil.copytree(input_dir + 'analysis/', new_base_dir + 'analysis/', copy_function=copier.copy, ignore=ignore_func)
                except:
                    dirs = [[set(Path(i).parts) for i in x[2]] for x in os.walk(input_dir+'analysis/')]
                    parts = list(set().union(*chain(*dirs)))
                    pattern = 'results'
                    if not np.any([ut.BaseFunc().match_pattern(string, pattern) for string in parts]):
                        raise Exception('Please run the analysis and/or postprocessing, since no result-files could be found.')
                    else:
                        print('unknown error while copying postprocessed results')
        if visualize:
            # create visualization figure of data set
            if postprocess:
                MetaAnalysis().meta_analysis(new_base_dir)
            else:
                MetaAnalysis().meta_analysis(input_dir)
        return new_base_dir
    

if __name__ == '__main__':
    
    base_in = os.path.join(DATA_DIR,'LineScans/raw/50nm')
    #base_in = '/mnt/d/artefacts/2024/02/05/7OEU4R-session-0da898d0/'

    # filter all the data and plot statistics about rejection reasons
    p = Processor(table_name = 'test', data_base='log.sqlite', comment='test analysis')

    methods = ['MIN-POLY']#['FOURIER', 'HARMONIC', 'CORR','MIN-QUAD','MIN-POLY']#'MLE','WINDOW' ,'MAX-QUAD'
    p.process_data(input_dir = base_in,  agnostic=True, filter=True, max_filter=np.inf, analyze=True, methods = methods, max_analyze=np.inf, postprocess=True, max_post=np.inf, visualize=True, collect_artefacts=False)