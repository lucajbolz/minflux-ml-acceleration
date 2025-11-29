"""
Module that provides functionality in order to analyze an experiment.
Includes functionality for pre-processing data and analyzing the data afterwards.
"""
import os, multiprocessing

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize, differential_evolution, LinearConstraint
from scipy.interpolate import griddata
from sklearn.neighbors import KernelDensity
import pandas as pd
import numbers
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import copy as copy
from pathlib import Path
import pandas as pd

import lib.utilities as ut
from lib.constants import *
from lib.plotting.artefacts import Artefacts, Residual_Figures
from lib.plotting.style import Style
from lib.data_handling.utilities import Converter

def study_single_file(full_file_path, methods=['QUAD'], agnostic=True, collect_artefacts=True):
    """
    Analyze a single file (experiments.yaml).

    This function analyzes an experiments.yaml file, applying different methods to study and collect data.
    The analysis results are then stored in a queue for later saving.

    :param str full_file_path:
        The absolute path of the experiments.yaml file to be analyzed.
    :param list methods:
        The list of method(s) to use for analysis. Default is ['QUAD'].
    :param bool agnostic:
        Flag to determine if the analysis should be agnostic. Default is True.
    :param bool collect_artefacts:
        Flag to determine if the collected artefacts should be saved. Default is True.

    :return:
        None. This function adds the analyzed result as a dictionary to a queue for later saving.
    :rtype:
        None

    :raises Exception:
        If the study of a specific type could not be performed.
        If the artefacts from the study could not be saved.

    :Example:
        study_single_file('/path/to/experiments.yaml', methods=['QUAD'], agnostic=True, collect_artefacts=True)

    .. note::
        The function prints diagnostic information to the console during execution.

    .. warning::
        This function may raise exceptions if specific study types fail or if artefacts cannot be saved.

    .. seealso::
        :class:`Study`: The class used to perform the analysis.
    """
    n_f, fname = os.path.split(full_file_path)
    print(n_f)
    pattern = 'REJECTED'
    if not np.any([ut.BaseFunc().match_pattern(string, pattern) for string in list(Path(full_file_path).parts)]):
        experiments = ut.Importer().load_yaml(full_file_path)
        for method in methods:
            study = Study(collect_artefacts=collect_artefacts, study_type=method)
            try:
                curr_experiments = copy.deepcopy(experiments)
                study.perform(curr_experiments, agnostic=agnostic)
            except Exception as e:
                print(f'Study of type {method} could not be performed on file {full_file_path}. Error: {e}')
            try:
                study.artefacts.save_figures(out_dir=n_f + '/')
                ut.Exporter().write_json(study.results.__dict__, filestr=n_f + '/' + method + f'_fitting_results.json')
            except Exception as e:
                print(f'Artefacts from study could not be saved. Error: {e}')
    pass

class EvaluationFacade:
    """
    Wrapper to provide convenient access to data evaluation.
    """

    def __init__(self, collect_artefacts=True, max_files=1):
        """
        Initialize the EvaluationFacade object.

        :param bool collect_artefacts:
            Flag indicating whether to collect artifacts. Default is True.
        :param int max_files:
            Maximum number of files to consider for evaluation. Default is 1.
        """
        self.results = {}
        self.collect_artefacts = collect_artefacts
        self.it_files = 0
        self.max_files = max_files

    def evaluate_data(self, in_folder, methods=['QUAD'], agnostic=True):
        """
        Evaluate data in a given folder.

        :param str in_folder:
            The path to the input folder.
        :param list methods:
            The list of evaluation methods. Default is ['QUAD'].
        :param bool agnostic:
            Flag indicating whether the evaluation should be done agnostically. Default is True.
        :return: None
        :rtype: None

        :raises Exception:
            If the evaluation of a specific file fails.

        :Example:
            evaluation_facade = EvaluationFacade()
            evaluation_facade.evaluate_data('/path/to/data', methods=['QUAD'], agnostic=True)
        """
        processable_files = ut.BaseFunc().find_files(
            in_folder, lambda file: 'experiments.yaml' in file, max_files=self.max_files
        )
        task = partial(study_single_file, methods=methods, agnostic=agnostic, collect_artefacts=self.collect_artefacts)

        with multiprocessing.Pool() as pool:
            print(f"Using a parallel pool with {pool._processes} processes")
            results = pool.map(task, processable_files)

class Study:
    """
    Class to cluster the analysis of an ensemble of experiment objects.
    """

    def __init__(self, study_type='QUAD', collect_artefacts=True):
        """
        Initialize the Study object.

        :param str study_type:
            The type of study to perform. Default is 'QUAD'.
        :param bool collect_artefacts:
            Flag indicating whether to collect artifacts. Default is True.
        """
        self.seg_idcs = None  # indices from segmentation
        self.artefacts = Artefacts()
        self.collect_artefacts = collect_artefacts
        self.results = Results()
        self.method = study_type
        self.results.method = self.method  # save method used for study

    def perform(self, experiments, agnostic=True):
        """
        Perform an evaluation of the conducted experiment.

        :param list experiments:
            List of experiment objects to evaluate.
        :param bool agnostic:
            Whether to forget everything or employ prior knowledge (effect only on MLE).

        :return: None
        :rtype: None

        :raises Exception:
            If the experiment object is invalid for non-agnostic analysis.

        :Example:
            study = Study()
            study.perform(experiments_list, agnostic=True)
        """
        self.results.agnostic = agnostic

        if experiments is None:
            return None
        ordered_exp = []
        for i in range(3):
            for exp in experiments:
                if not agnostic and ('sig_photons' not in exp.record.__dict__ or 'bgr_photons' not in exp.record.__dict__):
                    raise Exception('Invalid experiment object for non-agnostic analysis!')
                elif exp.setup.sample.config.n_molecules == i:
                    ordered_exp.append(exp)
                else:
                    continue
        experiments = ordered_exp

        routines = AnalysisRoutines(agnostic=agnostic, collect_artefacts=self.collect_artefacts)

        if self.method == 'MLE':
            routines.MLE_analysis(experiments)
        elif self.method in ['MIN-POLY', 'MIN-QUAD', 'MAX-QUAD']:
            routines.POLY_analysis(experiments, self.method)
        elif self.method in ['CORR', 'FOURIER', 'HARMONIC']:
            routines.HARMONIC_analysis(experiments, self.method)
        elif self.method in ['WINDOW']:
            routines.WINDOW_analysis(experiments, self.method)
        elif self.method in ['KAPPA']:
            routines.KAPPA_analysis(experiments, self.method)

        self.results.append(routines.results)
        self.artefacts.append(routines.artefacts)

        # get metrics
        # evaluation = Evaluation()
        self._get_metrics(experiments)
        # self.results.append(evaluation.results)
        pass


    def _get_metrics(self, experiments):
        """
        Calculates metrics for several experiments to evaluate the success of the experiment, mainly in terms of photon numbers.

        :param list experiments:
            List of experiments.
        :return:
            None. Modifies self.results.

        :Example:
            study = Study()
            study._get_metrics(experiments_list)
        """
        N0_bgr = jnp.nansum(experiments[0].record.photons)
        N0_K = experiments[0].record.photons.shape[0]
        for exp in experiments:
            record = exp.record
            N_tot = jnp.nansum(getattr(record, 'photons', 0))
            N_sig = jnp.nansum(getattr(record, 'sig_photons', 0))  # sig and bgr are lists of emissions from individual molecules: concatenate lists to apply nansum!
            N_bgr = jnp.nansum(getattr(record, 'bgr_photons', 0))
            K = getattr(record, 'photons', 0).shape[0]  # number of measurement sites
            if N_sig < 1 or N_bgr < 1:
                N_bgr = N0_bgr / N0_K * K
                N_sig = N_tot - N_bgr
            avg_SBR = N_sig / N_bgr
            # SBR = ut.BaseFunc()._moving_average(record.sig_photons,10)/ut.BaseFunc()._moving_average(record.bgr_photons,10)
            metrics = {'N_tot': N_tot, 'N_sig': N_sig, 'N_bgr': N_bgr, 'avg_SBR': avg_SBR, 'K': K, 'block_size': exp.measurement.config.block_size, 'seg_idcs': exp.measurement.config.seg_idcs}
            for key in metrics.keys():
                setattr(self.results, f'{key}_{exp.setup.sample.config.n_molecules}M', metrics[key])
        pass


class AnalysisRoutines:
    """
    Class to handle various analysis routines.

    :param bool agnostic:
        Flag indicating whether the analysis should be agnostic. Default is True.
    :param bool collect_artefacts:
        Flag indicating whether to collect artifacts. Default is True.
    """

    def __init__(self, agnostic=True, collect_artefacts=True) -> None:
        """
        Initialize the AnalysisRoutines object.

        :param bool agnostic:
            Flag indicating whether the analysis should be agnostic. Default is True.
        :param bool collect_artefacts:
            Flag indicating whether to collect artifacts. Default is True.
        """
        self.agnostic = agnostic
        self.collect_artefacts = collect_artefacts
        self.fit_routine = Fit(agnostic=agnostic, collect_artefacts=collect_artefacts)
        self.results = Results()
        self.artefacts = Artefacts()
        return None

    def MLE_analysis(self, experiments):
        """
        Perform a Maximum Likelihood Estimation (MLE) on distance parameters.

        This method estimates distance parameters using MLE based on the provided experiments.
        It calculates background estimates, kappa values, and performs fitting for different molecule counts.

        :param list experiments:
            List of experiments for analysis.
        :return:
            None. Modifies self.results and self.artefacts with analysis results.
        
        :Example:
            routines = AnalysisRoutines()
            routines.MLE_analysis(experiments_list)
        """
        N_bgr = self._get_background_estimate(experiments)
        self.results.N_bgr = N_bgr
        
        kappa_1M = self._get_kappa_estimate(experiments,{'param_dict':{'N_bgr': N_bgr},'scope':'global', 'estimator':'min-poly', 'agnostic':True})
        self.results.kappa_1M = kappa_1M

        param_dict = {'N_bgr': N_bgr,'kappa_1M':kappa_1M}#
        estimator = 'MLE'
        scope0, scope1, scope2 = 'global', 'global', 'global'
        fit_dict = {'1M': {'param_dict': param_dict, 'scope': scope1, 'estimator': estimator, 'agnostic': self.agnostic},
                    '2M': {'param_dict': param_dict, 'scope': scope2, 'estimator': estimator, 'agnostic': self.agnostic}}
    
        for idx, exp in enumerate(experiments):
            try_fit = False
            if exp.setup.sample.config.n_molecules == 0 and '0M' in fit_dict.keys():
                tmp_fit_dict = fit_dict['0M']
                try_fit = True
            elif exp.setup.sample.config.n_molecules == 1 and '1M' in fit_dict.keys():
                tmp_fit_dict = fit_dict['1M']
                try_fit = True
            elif exp.setup.sample.config.n_molecules == 2 and '2M' in fit_dict.keys():
                tmp_fit_dict = fit_dict['2M']
                try_fit = True
            else:
                continue
            
            if try_fit:
                sol_dict, artefacts = self.fit_routine.do_fit(exp, tmp_fit_dict, check_residuals=True)  # find minimum via quadratic estimator
                self.artefacts.append(artefacts)
                for key in sol_dict.keys():
                    setattr(self.results, f'{key}_{exp.setup.sample.config.n_molecules}M', sol_dict[key])  
        pass


    def POLY_analysis(self, experiments, method):
        """
        Perform NALM analysis of a sorted list of experiments, estimating distance via center of mass shift after bleaching steps.
        Done via quadratic approximation near the minimum, extracting the position of the minimum.
        Combined with an estimate from the quadratic estimator.

        :param list experiments:
            Sorted list of experiments for analysis.
        :param str method:
            The analysis method, one of ['MIN-POLY', 'MIN-QUAD', 'MAX-QUAD'].
        :return:
            None. Modifies self.results and self.artefacts with analysis results.
        
        :Example:
            routines = AnalysisRoutines()
            routines.POLY_analysis(experiments_list, method='MIN-POLY')
        """
        N_bgr = self._get_background_estimate(experiments)
        self.results.N_bgr = N_bgr
        
        #kappa_1M = self._get_kappa_estimate(experiments,{'param_dict':{'N_bgr': N_bgr},'scope':'global', 'estimator':'min-poly', 'agnostic':True})
        #self.results.kappa_1M = kappa_1M

        param_dict = {'N_bgr': N_bgr}#,'kappa_1M':kappa_1M
        if method=='MIN-POLY':
            estimator = 'min-poly'
            scope0,scope1,scope2 = 'global','local','local'
            #scope0,scope1,scope2 = 'global','global','global'
            fit_dict = {'1M': {'param_dict':param_dict,'scope':scope1, 'estimator':estimator, 'agnostic':self.agnostic},'2M': {'param_dict':param_dict,'scope':scope2, 'estimator':estimator, 'agnostic':self.agnostic}}
        elif method=='MIN-QUAD':
            estimator = 'min-quad'
            scope0,scope1,scope2 = 'global','local','local'
            #scope0,scope1,scope2 = 'local','global','global'
            #fit_dict = {'0M': {'param_dict':param_dict,'scope':scope0, 'estimator':estimator, 'agnostic':self.agnostic},'1M': {'param_dict':param_dict,'scope':scope1, 'estimator':estimator, 'agnostic':self.agnostic},'2M': {'param_dict':param_dict,'scope':scope2, 'estimator':estimator, 'agnostic':self.agnostic}}
            fit_dict = {'1M': {'param_dict':param_dict,'scope':scope1, 'estimator':estimator, 'agnostic':self.agnostic},'2M': {'param_dict':param_dict,'scope':scope2, 'estimator':estimator, 'agnostic':self.agnostic}}
        elif method=='MAX-QUAD':
            estimator = 'max-quad'
            scope0,scope1,scope2 = 'global','global','global'
            #fit_dict = {'0M': {'param_dict':param_dict,'scope':scope0, 'estimator':estimator, 'agnostic':self.agnostic},'1M': {'param_dict':param_dict,'scope':scope1, 'estimator':estimator, 'agnostic':self.agnostic},'2M': {'param_dict':param_dict,'scope':scope2, 'estimator':estimator, 'agnostic':self.agnostic}}
            fit_dict = {'1M': {'param_dict':param_dict,'scope':scope1, 'estimator':estimator, 'agnostic':self.agnostic},'2M': {'param_dict':param_dict,'scope':scope2, 'estimator':estimator, 'agnostic':self.agnostic}}
                
        for idx, exp in enumerate(experiments):
            try_fit = False
            if exp.setup.sample.config.n_molecules==0 and '0M' in fit_dict.keys():
                tmp_fit_dict = fit_dict['0M']
                try_fit=True
            elif exp.setup.sample.config.n_molecules==1 and '1M' in fit_dict.keys():
                tmp_fit_dict = fit_dict['1M']
                try_fit=True
            elif exp.setup.sample.config.n_molecules==2 and '2M' in fit_dict.keys():
                tmp_fit_dict = fit_dict['2M']
                try_fit=True
            else:
                continue
            
            if try_fit:
                sol_dict, artefacts = self.fit_routine.do_fit(exp,tmp_fit_dict,check_residuals=True) # find minimum via quadratic estimator
                self.artefacts.append(artefacts)
                for key in sol_dict.keys():
                    setattr(self.results,f'{key}_{exp.setup.sample.config.n_molecules}M', sol_dict[key])  
        pass

    def HARMONIC_analysis(self, experiments, method):
        """
        Perform harmonic analysis on a list of experiments.

        :param list experiments:
            List of experiments for harmonic analysis.
        :param str method:
            The harmonic analysis method, one of ['CORR', 'FOURIER', 'HARMONIC'].
        :return:
            None. Modifies self.results and self.artefacts with analysis results.

        :Example:
            routines = AnalysisRoutines()
            routines.HARMONIC_analysis(experiments_list, method='CORR')
        """
        if method=='CORR':
            estimator = 'correlate'
        elif method=='FOURIER':
            estimator = 'fourier'
        elif method=='HARMONIC':
            estimator = 'harmonic'

        N_bgr = self._get_background_estimate(experiments)
        self.results.N_bgr = N_bgr

        #kappa_1M = self._get_kappa_estimate(experiments,{'param_dict':{'N_bgr': N_bgr},'scope':'global', 'estimator':estimator, 'agnostic':True})
        #self.results.kappa_1M = kappa_1M

        param_dict = {'N_bgr': N_bgr, 'molecule_brightness': 1.0}
        
        fit_dict = {'0M': {'param_dict':param_dict},'1M': {'param_dict':param_dict,'scope':'global', 'estimator':estimator, 'agnostic':self.agnostic},'2M': {'param_dict':param_dict,'scope':'global', 'estimator':estimator, 'agnostic':self.agnostic}}
        sol_dict = {}
        for exp in experiments:#start with 0M experiment and proceed to 1M, 2M
            record = exp.record
            if exp.setup.sample.config.n_molecules==0 and '0M' in fit_dict.keys():
                continue
                # background has been estimated before
            if exp.setup.sample.config.n_molecules==1 and '1M' in fit_dict.keys():
                tmp_fit_dict = copy.deepcopy(fit_dict)['1M']
                if not tmp_fit_dict['agnostic']:
                    tmp_fit_dict['param_dict'] = {}#{'pos': 0.98 * exp.setup.sample.config.mol_pos}
                sol_dict, artefacts = self.fit_routine.do_fit(exp,tmp_fit_dict)
            if exp.setup.sample.config.n_molecules==2 and '2M' in fit_dict.keys():
                tmp_fit_dict = copy.deepcopy(fit_dict)['2M']
                if not tmp_fit_dict['agnostic']:
                    tmp_fit_dict['param_dict'] = {}#{'pos': 0.98 * exp.setup.sample.config.mol_pos}
                sol_dict, artefacts = self.fit_routine.do_fit(exp,tmp_fit_dict)
            for key in sol_dict.keys():
                setattr(self.results,f'{key}_{exp.setup.sample.config.n_molecules}M', sol_dict[key])
            self.artefacts.append(artefacts)
        pass

    def KAPPA_analysis(self, experiments, method, fixed_curvature=False):
        """
        Perform NALM analysis of a sorted list of experiments, estimating distance via center of mass shift after bleaching steps.
        Done via quadratic approximation near the minimum, extracting the position of the minimum.
        Combined with an estimate from the quadratic estimator.

        :param list experiments:
            Sorted list of experiments for analysis.
        :param str method:
            The analysis method.
        :param bool fixed_curvature:
            Flag indicating whether to use a fixed curvature value. Default is False.
        :return:
            None. Modifies self.results with analysis results.

        :Example:
            routines = AnalysisRoutines()
            routines.KAPPA_analysis(experiments_list, method='your_method', fixed_curvature=False)
        """
        N_bgr = self._get_background_estimate(experiments)
        self.results.N_bgr = N_bgr
        
        kappa_1M = self._get_kappa_estimate(experiments,{'param_dict':{'N_bgr': N_bgr},'scope':'global', 'estimator':'min-quad', 'agnostic':True})
        self.results.kappa_1M = kappa_1M
        kappa_2M = self._get_kappa_estimate(experiments,{'param_dict':{'N_bgr': N_bgr},'scope':'global', 'estimator':'min-quad', 'agnostic':True},n=2,kap0=kappa_1M)
        self.results.kappa_2M = kappa_2M
        pass

    def WINDOW_analysis(self, experiments, estimator='harmonic'):
        """
        Use a sliding window of the data to perform an analysis on a list of experiments for different spatial areas around the minimum,
        i.e. extract photons from different regions of a full line scan.

        :param list experiments:
            List of experiments for window analysis.
        :param str method:
            The window analysis method.
        :return:
            None. Modifies self.results and self.artefacts with analysis results.

        :Example:
            routines = AnalysisRoutines()
            routines.WINDOW_analysis(experiments_list)
        """
        N_bgr = self._get_background_estimate(experiments)
        self.results.N_bgr = N_bgr

        kappa_1M = self._get_kappa_estimate(experiments,{'param_dict':{'N_bgr': N_bgr},'scope':'global', 'estimator':estimator, 'agnostic':True})
        self.results.kappa_1M = kappa_1M

        windows = 16
        window_positions = np.linspace(0,1,windows)#numbers between 0 and 1
        
        sol_dict = {}
        for exp in experiments:#start with 0M experiment and proceed to 1M, 2M
            if exp.setup.sample.config.n_molecules==2:
                fit_dict = {'param_dict':{'N_bgr': N_bgr},'scope':'global', 'estimator':estimator, 'agnostic':self.agnostic}
                preloc_dict, _ = self.fit_routine.do_fit(exp,fit_dict) # pre-localization to retrieve lateral offset and just fit amplitude and y-offset
                phi0 = np.array([preloc_dict['raw_sol'][0][2],preloc_dict['raw_sol'][1][2]])
                for i,pos in enumerate(window_positions):
                    param_dict = {'N_bgr': N_bgr, 'window_position': pos,'phi0':phi0}
                    fit_dict = {'param_dict':param_dict,'scope':'global', 'estimator':estimator, 'agnostic':self.agnostic}
                    if not fit_dict['agnostic']:
                        fit_dict['param_dict'] = {}#{'pos': 0.98 * exp.setup.sample.config.mol_pos}
                    #sol_dict = sol_dict | self.do_fit(exp,tmp_fit_dict)
                    tmp_sol_dict, artefacts = self.fit_routine.do_fit(exp,fit_dict)
                    #tmp_sol_dict['window_pos'] = [pos, pos]
                    if i==0:
                        sol_dict = tmp_sol_dict
                    else:
                        sol_dict = ut.BaseFunc().join_dict_values(sol_dict, tmp_sol_dict)
                    self.artefacts.append(artefacts)
            else:
                continue
        for key in sol_dict.keys():
            setattr(self.results,f'{key}_{exp.setup.sample.config.n_molecules}M', sol_dict[key])
        pass

    def _get_fixed_curvature(self, exp, fit_dict):
        """
        Obtain the average curvature of the minimum in the experiment.

        :param exp:
            Experiment object for curvature estimation.
        :param dict fit_dict:
            Fit dictionary containing the parameters for the fit.
        :return:
            Array of curvatures, one for each axis.
        """
        glob_sol, _ = self.fit_routine.do_fit(exp, fit_dict, check_residuals=False)
        return [sol[1] for sol in glob_sol['raw_sol']]


    def _get_background_estimate(self, experiments):
        """
        Estimate background from the 0M experiment and set the corresponding parameter in the parameter dictionary.

        :param list experiments:
            List of experiments for background estimation.
        :return:
            Numpy array representing the estimated background.
        """
        for exp in experiments:
            # setattr(self.results, f'beta_{exp.setup.sample.config.n_molecules}M', exp.setup.sample.config.beta)
            if exp.setup.sample.config.n_molecules == 0:
                record = Converter().reshape_record(exp, export=False, show=False)
                counts = Converter().partition_record(record, exp.measurement.config.block_size,
                                                    seg_idcs=exp.measurement.config.seg_idcs, truncate=False).photons
                N_bgr = np.average(counts, axis=(1, 2))
            else:
                continue
        return N_bgr

    
    def _get_kappa_estimate(self, experiments, fit_dict, n=1, kap0=None):
        """
        Estimate the quality of the minimum from the 1M experiment and set the corresponding parameter in the parameter dictionary.

        :param list experiments:
            List of experiments for kappa estimation.
        :param dict fit_dict:
            Fit dictionary containing the parameters for the fit.
        :param int n:
            Number of molecules in the experiment. Default is 1.
        :param numpy.ndarray kap0:
            Initial value for kappa. Default is None.
        :return:
            Numpy array representing the estimated kappa.
        """
        if kap0 is None:
            kap0 = np.ones((2,))

        def kap(sol, i):
            kap = (sol[0] - fit_dict['param_dict']['N_bgr'][i]) / sol[1]
            # kap = max((sol[0] - fit_dict['param_dict']['N_bgr'][i]) / sol[1], kap0[i])
            return kap

        for exp in experiments:
            # setattr(self.results, f'beta_{exp.setup.sample.config.n_molecules}M', exp.setup.sample.config.beta)
            if exp.setup.sample.config.n_molecules == n:
                try:
                    sol_dict, _ = self.fit_routine.do_fit(exp, fit_dict, check_residuals=False)
                    partitioned_sol = Converter().partition_array(np.array(sol_dict['raw_sol']), exp.measurement.config.block_size,
                                                                seg_idcs=exp.measurement.config.seg_idcs, truncate=False)
                    kappa_1M = np.average([[kap(sol, i) for sol in axis] for i, axis in enumerate(partitioned_sol)], axis=1)
                except:
                    kappa_1M = np.ones((2,))
            else:
                continue
        return kappa_1M


class Fit:
    """
    Class that contains fit routine to estimate positions of molecules from experimental record.
    Every new fit needs the initialization of a new fit routine to clear results.
    """
    def __init__(self, agnostic=True, collect_artefacts=True) -> None:
        """
        Initialize the Fit class.

        :param agnostic: Flag indicating whether the fit should be agnostic.
        :type agnostic: bool
        :param collect_artefacts: Flag indicating whether to collect artifacts during fit.
        :type collect_artefacts: bool
        """
        self.collect_artefacts = collect_artefacts
        self.agnostic = agnostic
        pass

    def do_fit(self, exp, fit_dict, check_residuals=True):
        """
        Method to fit experiments either line-wise or globally. Results are accessible via self.results.

        :param exp: Experimental data.
        :type exp: Experiment
        :param fit_dict: Dictionary containing fit parameters.
        :type fit_dict: dict
        :param check_residuals: Flag indicating whether to check residuals after fitting.
        :type check_residuals: bool
        :return: Tuple containing solution dictionary and artifacts.
        :rtype: tuple
        """
        curr_fit_dict = copy.deepcopy(fit_dict)
        param_dict, scope, estimator, agnostic = curr_fit_dict['param_dict'], curr_fit_dict['scope'], curr_fit_dict[
            'estimator'], curr_fit_dict['agnostic']
        artefacts = Artefacts()
        if scope == 'global':
            sol_dict, fit_arr = self._do_global_fit(exp, param_dict=param_dict, estimator=estimator)
        elif scope == 'local':
            sol_dict, fit_arr = self._do_local_fit(exp, param_dict=param_dict, estimator=estimator)
        if check_residuals:
            chi2, artefacts = self.check_residuals(exp, fit_arr, estimator, scope)
            sol_dict = sol_dict | {'chi2': chi2}
        return sol_dict, artefacts

    def _do_global_fit(self, experiment, param_dict={}, estimator=None):
        """
        Fit one experiment globally (not line-wise). Suitable for 1M and 2M experiments.

        :param experiment: Experimental data.
        :type experiment: Experiment
        :param param_dict: Dictionary containing fit parameters.
        :type param_dict: dict
        :param estimator: Estimation method.
        :type estimator: str

        :raises Exception: If no estimator is provided.

        :return: Tuple containing solution dictionary and fit array.
        :rtype: tuple
        """
        exp = copy.deepcopy(experiment)
        sol_dict = {}
        if estimator is None:
            raise Exception('No estimator provided!')
        else:
            # collapse record into one line
            record = Converter().reshape_record(exp, export=False, show=False)
            record = Converter().partition_record(record, exp.measurement.config.block_size)
            lines = len(record.photons[0])
            # average all record attributes
            rec_dict = record.__dict__
            for key in rec_dict.keys():
                rec_dict[key] = np.concatenate(np.average(rec_dict[key], axis=1))
                setattr(record, key, rec_dict[key])
            exp.record = record
            exp.measurement.config.seg_idcs[:] = 0
            exp.measurement.config.block_size = 1

            sol_dict, fit_arr = self._do_local_fit(exp, param_dict=param_dict, estimator=estimator)
            sol_dict['N_fit'] = [N * lines for N in sol_dict['N_fit']]
        return sol_dict, fit_arr


    def _do_local_fit(self, experiment, param_dict={}, estimator='quadratic', max_lines=np.inf):
        """
        Find the molecule(s)' position(s) of an experiment. Perform a line-wise fit
        by fitting each line of the record separately with MLE, quadratic, and FE.

        :param experiment: The experiment to be fitted.
        :type experiment: Experiment
        :param param_dict: Dictionary of parameter dictionaries for each line.
                        {'0': {'FWHM': 1., ...}, '1': ...}
        :type param_dict: dict
        :param estimator: List of estimators to be applied.
        :type estimator: str
        :param max_lines: Maximum number of lines to fit (default is infinity).
        :type max_lines: int
        :return: Dictionary of fit results for each line.
        :rtype: dict
        """
        exp = copy.deepcopy(experiment)
        record = Converter().reshape_record(exp, export=False, show=False)
        start, end = (
            (-exp.measurement.config.seg_idcs % exp.measurement.config.block_size)[0],
            (exp.measurement.config.seg_idcs % exp.measurement.config.block_size)[1]
        )

        lines = record.photons.shape[0]
        rec_dict = record.__dict__
        fit_arr = copy.deepcopy(record.photons)  # record template to store fitted values.
        fit_arr[:] = np.nan

        lines = record.photons.shape[0]
        lines = int(np.min(np.array([lines, max_lines])))
        sol_dict = {}

        for line in range(lines):
            tmp_exp = copy.deepcopy(exp)
            par_dict = copy.deepcopy(param_dict)

            for key in par_dict.keys():
                par_dict = self._dict_to_line_mapping(
                    line, lines, start, end, exp.measurement.config.block_size, par_dict, param_dict, key
                )

            for key in rec_dict.keys():  # pick a line from one axis
                siz = np.array([el.size for el in rec_dict[key]])
                if np.all(siz != 0):
                    s = rec_dict[key][line]  # take line of each attribute (x0, c0,...)
                    setattr(tmp_exp.record, key, s)
                else:
                    continue

            if estimator in ['min-quad', 'min-poly', 'max-quad']:
                sol, masked_model = self.get_taylor_estimate(tmp_exp, estimator, param_dict=par_dict)
            elif estimator in ['fourier', 'correlate', 'harmonic']:
                sol, masked_model = self.get_harmonic_estimate(tmp_exp, estimator, param_dict=par_dict)
            elif estimator == 'MLE':
                sol, masked_model = self.get_MLE(tmp_exp, estimator, param_dict=par_dict)

            fit_arr[line] = masked_model

            tmp_sol = copy.deepcopy(sol)
            for key in sol.keys():
                if key not in sol_dict.keys():
                    sol_dict[key] = [tmp_sol[key]]
                else:
                    sol_dict[key].append(tmp_sol[key])

        return sol_dict, fit_arr


    def _line_to_nan_mapping(self, dict, key, line, start, end, lines, block_size):
        """
        Remove redundant information depending on axis of line.

        Substitutes value of local fit by nan if it is not a fit of the corresponding axis.

        :param dict: The dictionary containing fit information.
        :type dict: dict
        :param key: The key corresponding to the axis.
        :type key: str
        :param line: The line number.
        :type line: int
        :param start: The start index.
        :type start: int
        :param end: The end index.
        :type end: int
        :param lines: The total number of lines.
        :type lines: int
        :param block_size: The block size.
        :type block_size: int
        :return: The modified dictionary.
        :rtype: dict
        """
        dict = copy.deepcopy(dict)
        if 0 <= line < start:  # x, set y=nan
            dict[key][:, 1] = np.nan
        elif start <= line < 2 * start:  # y
            dict[key][:, 0] = np.nan
        elif 2 * start <= line < lines - 2 * end:
            if (line - 2 * start) % (2 * block_size) < block_size:  # x
                dict[key][:, 1] = np.nan
            else:
                dict[key][:, 0] = np.nan  # y
        elif lines - 2 * end <= line < lines - end:  # x
            dict[key][:, 1] = np.nan
        elif lines - end <= line < lines:  # y
            dict[key][:, 0] = np.nan
        return dict


    def check_residuals(self, exp, fit_arr, estimator, scope):
        """
        Evaluate residuals with respect to the original full model of fit with optional visualization.

        The estimate might have been obtained from a different model, e.g., quadratic or Fourier estimate.

        :param exp: The experimental data.
        :type exp: Experiment
        :param fit_arr: The array containing fitted values.
        :type fit_arr: np.ndarray
        :param estimator: The estimation method used.
        :type estimator: str
        :param scope: The scope of the fit ('global' or 'local').
        :type scope: str
        :return: Chi2 value in each axis normalized via pixel number, i.e., average chi2/pixel of axis.
        :rtype: list
        """
        seg_idcs = exp.measurement.config.seg_idcs
        block_size = exp.measurement.config.block_size
        record = Converter().reshape_record(exp, export=False, show=False)  # original record

        if len(fit_arr.shape) == 1:
            fit_exp = copy.deepcopy(exp)
            fit_exp.record.photons = fit_arr
            fit_rec = Converter().reshape_record(fit_exp, export=False, show=False)
        else:
            fit_rec = copy.deepcopy(record)
            fit_rec.photons = fit_arr

        # partition record and set counts
        fit_arrays = Converter().partition_record(fit_rec, block_size, seg_idcs=seg_idcs, truncate=False).photons
        data_arrays = Converter().partition_record(record, block_size, seg_idcs=seg_idcs, truncate=False).photons

        if scope == 'global':
            data_arrays = [np.average([array], axis=1) for array in data_arrays]

        res_arrays = [(data_arrays[i] - fit_arrays[i]) for i in range(len(data_arrays))]
        chi2 = [(data_arrays[i] - fit_arrays[i]) ** 2 / fit_arrays[i] for i in range(len(data_arrays))]
        chi2 = [np.nansum(arr, axis=1) / np.sum(~np.isnan(arr), axis=1) for arr in chi2]
        chi2 = [np.array([ut.BaseFunc().remove_trailing_zeros(arr)]) for arr in chi2]
        chi2 = Converter().join_array(chi2, block_size, seg_idcs=seg_idcs)
        chi2 = chi2.flatten().tolist()

        artefacts = Artefacts()
        if self.collect_artefacts:
            fig, axs = Residual_Figures().fig_check_residuals(data_arrays, fit_arrays, res_arrays)
            artefacts.add_figures([fig], [estimator + f'_residuals_{exp.setup.sample.config.n_molecules}M' +
                                        ut.Labeler().id_generator(size=3)])
        return chi2, artefacts

    
    def get_MLE(self, exp, estimator, param_dict={}, show=False):
        """
        Perform Maximum Likelihood Estimation (MLE) to obtain fit results for the given experiment.

        :param exp: The experimental data.
        :type exp: Experiment
        :param estimator: The estimator to be used for MLE.
        :type estimator: str
        :param param_dict: Dictionary of additional parameters for the estimation (default is an empty dictionary).
        :type param_dict: dict
        :param show: Flag indicating whether to display the fitting results (default is False).
        :type show: bool
        :return: Dictionary containing the MLE fit results and the masked model.
        :rtype: dict
        """
        tmp_est = 'harmonic'
        M = exp.setup.sample.config.n_molecules
        counts = exp.record.photons
        K = len(counts)
        sol_dict = {}
        n_p = min(int(exp.measurement.config.num_positions/2 - 1), 18)#15
        window_idx = Estimators()._find_ext(estimator, np.arange(K), w=counts)           
        for i in range(2):
            if i>0 and M==0:
                continue
            window_idx = min(K-n_p-1, max(n_p+1, int(window_idx))) 
            window = counts[window_idx-n_p:window_idx+n_p+1]
            while window.size < 2*n_p:
                n_p -= 1
                window = counts[window_idx-n_p:window_idx+n_p+1]
                if n_p < 10:
                    continue
            l = int(len(window)/2)
            xdata = (2*np.pi/K) * np.arange(-l,l+1)
            ydata = window
            sol, success = Estimators().get_MLE(xdata, ydata, tmp_est, param_dict=param_dict)#m0,m1,b
            window_idx = sol[2] * K/(2*np.pi) + window_idx # - n_p

        if tmp_est=='min-poly':
            fit_vals = Estimators().min_poly_model(xdata,sol)
        elif tmp_est=='min-quad':
            fit_vals = Estimators().min_quad_model(xdata,sol)
        elif tmp_est=='max-quad':
            fit_vals = Estimators().max_quad_model(xdata,sol)
        elif tmp_est=='harmonic':
            fit_vals = Estimators().harmonic_model(xdata,sol)
        masked_model = ut.BaseFunc().nan_mask_list(counts, window, fit_vals)
        
        if show and M>0:          
            fig, ax = plt.subplots()
            ax.plot(xdata, fit_vals, 'r--')
            ax.scatter(xdata,ydata, label='data')
            ax.xlabel('x')
            ax.ylabel('y')
            ax.legend()
            plt.show()
            t=1
            plt.close()

        sol_dict['N_fit'] = np.sum(window)
        sol_dict['raw_sol'] = sol
        sol_dict['min_idx'] = window_idx
        sol_dict['Lambda0'] = exp.setup.instrument.config.Lambda0
        sol_dict['L'] = exp.measurement.config.L
        sol_dict['num_pos'] = exp.measurement.config.num_positions
        sol_dict['success'] = success
        return sol_dict, masked_model
    
    def get_taylor_estimate(self, exp, estimator, param_dict={}, show=False):
        """
        Obtain fit results using Taylor series expansion-based estimation.

        :param exp: The experimental data.
        :type exp: Experiment
        :param estimator: The estimator to be used for Taylor series expansion.
        :type estimator: str
        :param param_dict: Dictionary of additional parameters for the estimation (default is an empty dictionary).
        :type param_dict: dict
        :param show: Flag indicating whether to display the fitting results (default is False).
        :type show: bool
        :return: Dictionary containing the fit results and the masked model.
        :rtype: dict
        """
        M = exp.setup.sample.config.n_molecules
        counts = exp.record.photons
        K = len(counts)
        sol_dict = {}
        n_p = min(int(exp.measurement.config.num_positions/2 - 1), 18)
        window_idx = Estimators()._find_ext(estimator, np.arange(K), w=counts)           
        for i in range(2):
            if i>0 and M==0:
                continue
            window_idx = min(K-n_p-1, max(n_p+1, int(window_idx))) 
            window = counts[window_idx-n_p:window_idx+n_p+1]
            while window.size < 2*n_p:
                n_p -= 1
                window = counts[window_idx-n_p:window_idx+n_p+1]
                if n_p < 10:
                    continue
            l = int(len(window)/2)
            xdata = (2*np.pi/K) * np.arange(-l,l+1)
            ydata = window
            sol, success, fit_vals = Estimators().get_taylorE(xdata, ydata, estimator, param_dict=param_dict)#m0,m1,b
            window_idx = sol[2] * K/(2*np.pi) + window_idx # - n_p

        masked_model = ut.BaseFunc().nan_mask_list(counts, window, fit_vals)
        
        if show and M>0:          
            fig, ax = plt.subplots()
            ax.plot(xdata, fit_vals, 'r--')
            ax.scatter(xdata,ydata, label='data')
            ax.xlabel('x')
            ax.ylabel('y')
            ax.legend()
            plt.show()
            t=1
            plt.close()

        sol_dict['N_fit'] = np.sum(window)
        sol_dict['raw_sol'] = sol
        sol_dict['min_idx'] = window_idx
        sol_dict['Lambda0'] = exp.setup.instrument.config.Lambda0
        sol_dict['L'] = exp.measurement.config.L
        sol_dict['num_pos'] = exp.measurement.config.num_positions
        sol_dict['success'] = success
        return sol_dict, masked_model
    
    def get_harmonic_estimate(self, exp, estimator, param_dict={}, show=False):
        """
        Obtain fit results using harmonic estimation.

        :param exp: The experimental data.
        :type exp: Experiment
        :param estimator: The estimator to be used for harmonic estimation ('fourier', 'correlate', 'harmonic').
        :type estimator: str
        :param param_dict: Dictionary of additional parameters for the estimation (default is an empty dictionary).
        :type param_dict: dict
        :param show: Flag indicating whether to display the fitting results (default is False).
        :type show: bool
        :return: Dictionary containing the fit results and the masked model.
        :rtype: dict
        """
        counts = exp.record.photons
        a = exp.measurement.config.L
        b = exp.measurement.config.num_positions
        scale_fac = exp.setup.instrument.config.Lambda0/(4*np.pi)

        
        if show and exp.setup.sample.config.n_molecules<1:
            show = False
        if 'window_position' in param_dict.keys():
            pos = param_dict['window_position'] # fractional position within window range
            window_size = 40
            n_p = min(int(exp.measurement.config.num_positions/2 - 1), int(window_size/2))
            window_range = len(counts)-int(window_size+2)
        else:
            pos = 0 # choose full line for fit
            window_size = len(counts)
            n_p = int(window_size/2)
            window_range = 0
        window_idx = int(window_size/2+window_range*pos) # window center
        phot_window = counts[window_idx-n_p:window_idx+n_p+1]
        xdata = (2*np.pi/b) * np.arange(0,len(counts))[window_idx-n_p:window_idx+n_p+1]
        # do fourier trafo or correlation and estimate phase and amplitude
        if estimator=='fourier':
            phase, amplitude = Estimators().get_FE(counts, exp.measurement.config.L, exp.measurement.config.num_positions, show=show)
            offset = np.mean(counts)
            success = True
        elif estimator=='correlate': # how to handle background here?
            phase, amplitude = Estimators().get_CE(counts,show=show)
            offset = np.mean(counts)
            success = True
        elif estimator=='harmonic':
            sol, success = Estimators().get_HE(xdata, phot_window, param_dict=param_dict,show=False)
            offset, amplitude, phase = sol
        
        sol = [offset, amplitude, phase]

        fit_vals = Estimators().harmonic_model(xdata,sol)
        masked_model = ut.BaseFunc().nan_mask_list(counts, phot_window, fit_vals)
        
        sol_dict = {}
        sol_dict['N_fit'] = np.sum(counts)
        sol_dict['raw_sol'] = sol
        sol_dict['min_idx'] = phase*b/(2*np.pi)
        sol_dict['window_idx'] = window_idx
        sol_dict['Lambda0'] = exp.setup.instrument.config.Lambda0
        sol_dict['L'] = exp.measurement.config.L
        sol_dict['num_pos'] = exp.measurement.config.num_positions
        sol_dict['success'] = success
        return sol_dict, masked_model
    
    def _dict_to_line_mapping(self, line, lines, start, end, block_size, line_dict, glob_dict, key):
        """
        Map values in each axis to the corresponding x or y line.

        :param line: Current line number.
        :type line: int
        :param lines: Maximum number of lines.
        :type lines: int
        :param start: Size of the starting block.
        :type start: int
        :param end: Size of the ending block.
        :type end: int
        :param block_size: Block size.
        :type block_size: int
        :param line_dict: Current parameter dictionary for the fit in the respective axis/line.
        :type line_dict: dict
        :param glob_dict: Dictionary with two-dimensional key.
        :type glob_dict: dict
        :param key: Key for which values are assigned to the current axis/line.
        :type key: str
        :return: Updated dictionary with assigned values for the current axis/line.
        :rtype: dict
        """
        dict1 = copy.deepcopy(line_dict)
        dict2 = copy.deepcopy(glob_dict)
        try:
            s=dict2[key].size
        except:
            try:
                s = len(dict2[key])
            except:
                try:
                    assert isinstance(dict2[key], numbers.Number)
                    s=1
                except:
                    print("couldn't retrieve size of dict entry!")
        if s==2:
            if 0<= line < start: #x
                dict1[key]=dict2[key][0]
            elif start<= line < 2*start: # y
                dict1[key]=dict2[key][1]
            elif 2*start <= line < lines-2*end:
                if (line-2*start)%(2*block_size) < block_size: # x
                    dict1[key]=dict2[key][0]
                else:
                    dict1[key]=dict2[key][1] # y
            elif lines-2*end<=line<lines-end: # x
                dict1[key]=dict2[key][0]
            elif lines-end <= line < lines: # y
                dict1[key]=dict2[key][1]
        elif s==1:
            if type(dict2[key]) is list:
                dict1[key]=dict2[key][0]
            elif isinstance(dict2[key], numbers.Number):
                dict1[key]=dict2[key]
        return dict1

class Estimators:
    """Class that provides different estimators to evaluate the shape of the intensity minimum."""

    def __init__(self):
        pass

    def get_taylorE(self, xdata, ydata, estimator, param_dict={}):
        """
        Estimate parameters using the Taylor expansion of the full harmonic model.

        :param xdata: Independent variable data.
        :type xdata: array-like
        :param ydata: Dependent variable data.
        :type ydata: array-like
        :param estimator: The type of estimator ('min-poly', 'min-quad', 'max-quad').
        :type estimator: str
        :param param_dict: Dictionary of additional parameters for the estimation (default is an empty dictionary).
        :type param_dict: dict
        :return: Tuple containing the estimated parameters, success flag, and the fitted model.
        :rtype: tuple
        """
        # get initial guess via simple quadratic fit
        params0 = None#[cmin+2e-3, cmin+1e-3, 0.]
        linear_constraint = LinearConstraint([[1, 0, 0], [0, 1, 0], [0,0,1]], [0.,0.,-np.pi/2], [np.inf, np.inf, np.pi/2])#m0,m1,b
        sol = self.get_initial_guess(xdata,ydata,estimator,params0=params0,**param_dict,constr=[linear_constraint])#,constr=[linear_constraint]
        
        if estimator=='min-poly':
            #model = self.min_poly_model
            #model = self.min_hexa_model
            model = self.harmonic_model
        elif estimator=='min-quad':
            model = self.min_quad_model
        elif estimator=='max-quad':
            model = self.max_quad_model
        #objective = lambda params: self.lsqs_objective(xdata,ydata,model,params,**param_dict)
        objective = lambda params: self.loglike_objective(xdata,ydata,model,params,**param_dict)
        objective_jac = lambda params: self.objective_jac(objective, params)

        options = {'ftol': 1e-5, 'eps': 1e-3}
        success = False
        try:
            sol1 = minimize(objective, sol, method='SLSQP',options=options, constraints=[linear_constraint])# , constraints=[linear_constraint]
            if not sol1.success:
                try:
                    sol1 = minimize(objective, sol, method='nelder-mead')
                except:
                    pass
            if not sol1.success:
                try:
                    sol1 = minimize(objective, sol, method='SLSQP', jac=objective_jac,options=options, constraints=[linear_constraint])
                except:
                    pass
            if sol1.success:
                sol = sol1.x
                success = True
        except:
            pass
        return sol, success, model(xdata,sol,**param_dict)

    def get_MLE(self, xdata, ydata, estimator, param_dict={}):
        """
        Maximum Likelihood Estimation (MLE) estimator for distance estimate.

        :param xdata: Independent variable data.
        :type xdata: array-like
        :param ydata: Dependent variable data.
        :type ydata: array-like
        :param estimator: The type of estimator ('min-poly', 'min-quad', 'max-quad', 'harmonic').
        :type estimator: str
        :param param_dict: Dictionary of additional parameters for the estimation (default is an empty dictionary).
        :type param_dict: dict
        :return: Tuple containing the estimated parameters and a success flag.
        :rtype: tuple
        """
        cmin = param_dict.get('N_bgr',1E-9)
        k1 = param_dict.get('kappa_1M',1.)
        
        # get initial guess via simple quadratic fit
        params0 = None#[cmin+2e-3, cmin+1e-3, 0.]
        sol = self.get_initial_guess(xdata,ydata,estimator,params0=params0,**param_dict)
        
        if estimator=='min-poly':
            model = self.min_poly_model
        elif estimator=='min-quad':
            model = self.min_quad_model
        elif estimator=='max-quad':
            model = self.max_quad_model
        elif estimator=='harmonic':
            model = self.harmonic_model
        objective = lambda params: self.loglike_objective(xdata,ydata,model,params,**param_dict)
        objective_jac = lambda params: self.objective_jac(objective, params)

        def wrap_f(f, params, Nbgr, kappa):
            gamma, delta, phi = params
            alpha = kappa+np.sqrt(kappa**2-1)
            m0 = Nbgr + gamma*(1+alpha**2)*2
            m1 = 2*alpha*np.sqrt(2)*gamma*np.sqrt(1+np.cos(delta))
            b = phi
            param_vec = np.array([m0,m1,b])
            return f(param_vec)
        
        wrapped_objective = lambda params: wrap_f(objective, params, cmin, k1)
        wrapped_jac = lambda params: wrap_f(objective_jac, params, cmin, k1)

        options = {'ftol': 1e-5, 'eps': 1e-5}
        linear_constraint = LinearConstraint([[1, 0, 0], [0, 1, 0], [0,0,1]], [0,0,min(xdata)], [np.inf, np.pi/4., max(xdata)])#gamma, delta, phi0
        params0 = [(sol[0]-cmin)/(1+(k1+np.sqrt(k1**2-1))**2),30,sol[2]]
        try:
            sol1 = minimize(wrapped_objective, params0, method='SLSQP', jac=wrapped_jac, constraints=[linear_constraint],options=options)
            if sol1.success:
                sol = sol1.x
                alpha = k1+np.sqrt(k1**2-1)
                m0 = cmin + sol[0]*(1+alpha**2)*2
                m1 = 2*alpha*np.sqrt(2)*sol[0]*np.sqrt(1+np.cos(sol[1]))
                b = sol[2]
                sol = np.array([m0,m1,b])
        except:
            pass
        return sol, sol1.success

    def get_CE(self, counts,show=False):
        """
        Correlative estimator to determine phase shift and amplitude of harmonic signal.

        This estimator is independent of the wavelength but assumes to process a full period of the signal.
        By correlating it with a harmonic signal in that period, it extracts the phase shift and amplitude that fit the signal best.

        :param counts: Array containing the signal data.
        :type counts: array-like
        :param show: Flag to display plots of the original data, pure cosine, and residuals (default is False).
        :type show: bool
        :return: Tuple containing the phase shift and amplitude.
        :rtype: tuple
        """
        def get_cos_params(samples):
            N = len(samples)
            x = np.linspace(-np.pi, np.pi, N, endpoint=True)
            template = np.exp(1j * x)
            corr = 2 / N * template@samples
            R = np.abs(corr)
            phi = np.log(corr).imag
            return R, phi
        A, phi = get_cos_params(counts-np.mean(counts))

        if show:
            fig, ax = plt.subplots(1,2,figsize=(12,8))
            cos_data = np.mean(counts) + A * np.cos(np.arange(len(counts))/(8*np.pi) - (phi-np.pi))
            ax[0].plot(np.arange(len(counts)), counts,'b', label='Noisy original data')
            ax[0].plot(np.arange(len(counts)), cos_data, c='green',label='pure cosine')
            ax[0].legend()
            ax[1].plot(np.arange(len(counts)),counts-cos_data,c='g',label='residuals with pure cosine')
            ax[1].legend()
            plt.show()
            plt.close()
        return phi%(2*np.pi), A

    def get_HE(self, xdata, ydata, param_dict={}, show=False):
        """
        Get harmonic estimator, i.e. simple sinusoidal fit of phase scan.

        :param xdata: Array containing the phase data.
        :type xdata: array-like
        :param ydata: Array containing the photon counts.
        :type ydata: array-like
        :param param_dict: Dictionary of additional parameters for the estimator (default is an empty dictionary).
        :type param_dict: dict
        :param show: Flag to display plots of the original data, pure cosine, and residuals (default is False).
        :type show: bool
        :return: Tuple containing the solution vector and a success flag.
        :rtype: tuple
        """
        if 'phi0' in param_dict.keys():
            b0 = param_dict['phi0']
        else:
            b0 = np.mean(xdata)
        linear_constraint = LinearConstraint([[1, 0, 0], [0, 1, 0], [0,0,1]], [0,0,0], [np.inf, np.inf, 2*np.pi])#m0,m1,b
        # get initial guess via simple quadratic fit
        params0 = [np.mean(ydata), 0.8*np.mean(ydata) , b0]
        model = self.harmonic_model
        objective = lambda params: self.loglike_objective(xdata,ydata,model,params)
        objective_jac = lambda params: self.objective_jac(objective, params)

        options = {'ftol': 1e-5, 'eps': 1e-5}
        try:
            sol = minimize(objective, params0, jac=objective_jac,method='SLSQP',options=options, constraints=[linear_constraint])#
            if not sol.success:
                sol_vec = params0
            else:
                sol_vec = sol.x
        except:
            pass
        if show:
            fig, ax = plt.subplots(1,2,figsize=(12,8))
            cos_data = model(xdata,sol)
            ax[0].plot(xdata, ydata,'b', label='Noisy original data')
            ax[0].plot(xdata, cos_data, c='green',label='pure cosine')
            ax[0].legend()
            ax[1].plot(xdata, ydata-cos_data,c='g',label='residuals with pure cosine')
            ax[1].legend()
            plt.show()
            plt.close()
        return sol_vec, sol.success

    def get_FE(self, counts, L, K,show=False):
        """
        Get 1D Fourier estimator (only for one full phase scan line!).
        
        Call only on 1M and 2M experiments!

        :param counts: Array containing photon counts.
        :type counts: array-like
        :param L: Length of the signal.
        :type L: float
        :param K: Number of samples in the signal.
        :type K: int
        :param show: Flag to display plots of the full FFT spectrum, cleaned signal, and residuals (default is False).
        :type show: bool
        :return: Tuple containing the phase and amplitude of the selected frequency.
        :rtype: tuple
        """
        N = counts.size #batch size
        dx = L/K * N
        fft_y = np.fft.fft(counts,n=N, norm='ortho') # scale forward and backward with 1/sqrt(N)

        f = np.linspace(0, N/dx, N, endpoint=False)
        frequency = f[:N // 2]
        amplitude = np.abs(fft_y)[:N // 2] * 2/np.sqrt(N) # half of the amplitudes, real spectrum
        phase = np.angle(fft_y)[:N // 2] # phase shifts
        
        cutoff = 5*np.mean(amplitude)# TODO: cutoff or biggest one?
        amplitudes = amplitude[np.where((cutoff < amplitude))]
        frequencies = frequency[np.where((cutoff < amplitude))]
        phases = phase[np.where((cutoff < amplitude))]

        if show:
            # plot the full fft spectrum
            fig, ax = plt.subplots(1,3,figsize=(12,8))
            ax[0].set_ylabel('Amplitude')
            ax[0].set_xlabel('Lambda [nm]')
            ax[0].plot(1/(N*frequency), amplitude,'.',label='spectrum')
            ax[0].plot(1/(N*frequencies), amplitudes,'x',c='r',label='selected frequency')
            ax[0].legend()

            fft_y[np.where((2*np.abs(fft_y)/np.sqrt(N)<cutoff))]=0
            cleaned_signal = np.real(np.fft.ifft(fft_y,norm='ortho'))
            cleaned_signal=np.sum(cleaned_signal.reshape(len(counts),-1),axis=1)
            cos_data = np.mean(counts) - amplitudes[1] * np.cos(np.arange(len(counts)) * L/K *2*np.pi * frequencies[1] + phases[1] * LAMBDA/2/(4*np.pi))
            ax[1].plot(np.arange(len(counts)), counts,'b', label='Noisy original data')
            ax[1].plot(np.arange(len(counts)), cleaned_signal,'r--', label='FFT cleaned signal')
            ax[1].plot(np.arange(len(counts)), cos_data, c='green',label='pure cosine')
            ax[1].legend()
            ax[2].plot(np.arange(len(counts)),counts-cleaned_signal,c='r',label='residuals with cleaned data')
            ax[2].plot(np.arange(len(counts)),counts-cos_data,c='g',label='residuals with pure cosine')
            ax[2].legend()
            plt.show()
            plt.close()
        phase = phases[1]
        amp = amplitudes[1]
        return (np.pi-phase)%(2*np.pi), amp#

    def min_hexa_model(self, x, params,**kwargs):
        """
        Model based on a sixth-order polynomial.

        :param x: Input values.
        :type x: array_like

        :param params: Model parameters (m0, m1, b).
        :type params: tuple

        :param kwargs: Additional keyword arguments.
        :type kwargs: dict

        :return: Model values.
        :rtype: array_like
        """
        m0, m1, b = params
        r = m0 + m1*(-1 + 0.5*((x-b)**2) - ((x-b)**4)/24. + ((x-b)**6)/720.)
        return r
    
    def min_poly_model(self, x, params,**kwargs):
        """
        Model based on a 4-th order polynomial.

        :param x: Input values.
        :type x: array_like

        :param params: Model parameters (m0, m1, b).
        :type params: tuple

        :param kwargs: Additional keyword arguments.
        :type kwargs: dict

        :return: Model values.
        :rtype: array_like
        """
        m0, m1, b = params
        r = m0 + m1*(-1 + 0.5*((x-b)**2) - ((x-b)**4)/24.)
        return r
      
    def min_quad_model(self, x, params, **kwargs):
        """
        Model based on a quadratic polynomial.

        :param x: Input values.
        :type x: array_like

        :param params: Model parameters (m0, m1, b).
        :type params: tuple

        :param kwargs: Additional keyword arguments.
        :type kwargs: dict

        :return: Model values.
        :rtype: array_like
        """
        m0, m1, b = params
        r = m0 + m1*(0.5*((x-b)**2) - 1)
        return r
    
    def max_quad_model(self, x, params, **kwargs):
        """
        Model based on a quadratic polynomial to fit a maximum.

        :param x: Input values.
        :type x: array_like

        :param params: Model parameters (m0, m1, b).
        :type params: tuple

        :param kwargs: Additional keyword arguments.
        :type kwargs: dict

        :return: Model values.
        :rtype: array_like
        """
        m0, m1, b = params
        r = m0 - m1*(0.5*((x-b)**2) + 1)
        return r
    
    def harmonic_model(self, x, params,**kwargs):
        """
        Harmonic model.

        :param x: Input values.
        :type x: array_like

        :param params: Model parameters (m0, m1, b).
        :type params: tuple

        :param kwargs: Additional keyword arguments.
        :type kwargs: dict

        :return: Model values.
        :rtype: array_like
        """
        m0, m1, b = params
        r = m0 + m1 * jnp.cos(x+jnp.pi-b) # negative sine to shift minimum into center
        return r
    
    def lsqs_objective(self, x, y, model, params, **kwargs):
        """
        Least squares objective function.

        :param x: Input values.
        :type x: array_like

        :param y: Target values.
        :type y: array_like

        :param model: Model function.
        :type model: callable

        :param params: Model parameters.
        :type params: array_like

        :param kwargs: Additional keyword arguments.
        :type kwargs: dict

        :return: Objective value.
        :rtype: float
        """
        val = model(x, params)
        weights = kwargs.get('weights', np.ones(len(y)))
        return jnp.sum((val - y) ** 2)


    def loglike_objective(self, x, y, model, params, **kwargs):
        """
        Log-likelihood objective function.

        :param x: Input values.
        :type x: array_like

        :param y: Target values.
        :type y: array_like

        :param model: Model function.
        :type model: callable

        :param params: Model parameters.
        :type params: array_like

        :param kwargs: Additional keyword arguments.
        :type kwargs: dict

        :return: Objective value.
        :rtype: float
        """
        arg = model(x, params)
        weights = kwargs.get('weights', np.ones(len(y)))
        return -jnp.sum((y * jnp.log(arg) - arg) * weights)


    def objective_jac(self, objective, params):
        """
        Compute the Jacobian of an objective function.

        :param objective: Objective function.
        :type objective: callable

        :param params: Model parameters.
        :type params: array_like

        :return: Jacobian matrix.
        :rtype: array_like
        """
        f = lambda params: objective(params)  # objective is already evaluated on a model and data.
        J = jax.jacobian(f)
        return J(params)

    
    def objective_hess(self, x, y, objective, params, **kwargs):
        """
        Compute the Hessian matrix of an objective function.

        :param x: Input values.
        :type x: array_like

        :param y: Target values.
        :type y: array_like

        :param objective: Objective function.
        :type objective: callable

        :param params: Model parameters.
        :type params: array_like

        :param kwargs: Additional keyword arguments.
        :type kwargs: dict

        :return: Hessian matrix.
        :rtype: array_like
        """
        f = lambda params: objective(params, x, y, **kwargs)
        H = jax.hessian(f)
        return H(params)


    def get_initial_guess(self, x, y, estimator, params0=None, constr=None, **kwargs):
        """
        Retrieve an initial guess for model parameters.

        :param x: Input values.
        :type x: array_like

        :param y: Target values.
        :type y: array_like

        :param estimator: Estimation method.
        :type estimator: str

        :param params0: Initial guess for parameters.
        :type params0: array_like, optional

        :param constr: Constraints on parameters.
        :type constr: LinearConstraint, optional

        :param kwargs: Additional keyword arguments.
        :type kwargs: dict

        :return: Initial guess for parameters.
        :rtype: array_like
        """
        if estimator in ['min-quad', 'min-poly', 'harmonic']:
            model = self.min_quad_model
        elif estimator in ['max-quad']:
            model = self.max_quad_model
        objective = lambda params: self.lsqs_objective(x, y, model, params, **kwargs)
        objective_jac = lambda params: self.objective_jac(objective, params)

        if constr is None:
            low, up = [0., 0., min(x)], [np.inf, np.inf, max(x)]
            constr = LinearConstraint([[1, 0, 0], [0, 1, 0], [0, 0, 1]], low, up)

        if params0 is None:
            params0 = self._estimate_initial_values(x, y)

        try:
            bnds = ((0., 800.), (0., 800.), (min(x), max(x)))
            options = {'ftol': 1e-3, 'eps': 1e-3}
            sol1 = minimize(objective, params0, method='SLSQP', constraints=constr, options=options)
            popt = sol1.x
        except:
            popt = params0

        return popt


    def _estimate_initial_values(self, x, y):
        """
        Estimate initial values for model parameters.

        :param x: Input values.
        :type x: array_like

        :param y: Target values.
        :type y: array_like

        :return: Initial values for parameters.
        :rtype: array_like
        """
        mean_x = np.mean(x)
        variance_x = np.var(x)
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        y_q = np.quantile(y, [0, 0.33])
        data_within_quantile = y[(y >= y_q[0]) & (y <= y_q[1])]
        y_offset = np.mean(data_within_quantile)

        initial_m1 = y_range / (2 * variance_x)
        initial_m0 = initial_m1 + y_offset
        initial_b = mean_x
        initial_m1 = min(initial_m1, initial_m0 - 10**-6)  # make sure m1 is smaller than m0
        return [initial_m0, initial_m1, initial_b]

        
    def _find_ext(self, estimator, p, w=None):
        """
        Find the extremum in 1D via Kernel Density Estimation (KDE) and differential evolution.

        :param estimator: Estimator to decide whether to find a maximum or minimum.
        :type estimator: str

        :param p: Array of positions.
        :type p: array_like

        :param w: Photons used as weights for the KDE.
        :type w: array_like, optional

        :return: Position of the maximum density.
        :rtype: array_like
        """
        dim = len(p.shape)
        if dim==1:
            w = ut.BaseFunc()._moving_average(w, n=10) # smooth weights via moving average
            p = ut.BaseFunc()._moving_average(p, n=10)
            pos = p.reshape(-1, 1)
            # instantiate and fit the KDE model
            kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
            if estimator=='max-quad':
                weights = w
            else:
                weights = 1/(w+1e-4)
            kde.fit(pos,sample_weight=weights)
            log_density = lambda x: -kde.score_samples(x.reshape(-1, 1))
            xmin, xmax = pos.min(),pos.max()
            bnds = [(xmin,xmax)] # TODO: generalize to d dim
            #kernel = stats.gaussian_kde(p, weights=w)
            #bnds = ((-10, 10), (-10, 10)) # TODO: generalize to d dim
            #f = lambda x: -kernel(np.array(x))
        if dim==2:
            xmin, xmax, ymin, ymax = pos[0].min(),pos[0].max(), pos[1].min(),pos[1].max()
            bnds = ((xmin,xmax), (ymin, ymax)) # TODO: generalize to d dim
            raise Exception('Wrong dimensioanlity!')
            #bnds = ((-10, 10), (-10, 10)) # TODO: generalize to d dim
            #f = lambda x: -kernel(np.array([x[0],x[1]]))
        opt = differential_evolution(log_density,bounds=bnds) # find global minimum
        #opt2 = minimize(log_density,opt.x,method='SLSQP',bounds=bnds) # optional: 2nd step minimization with higher precision
        return opt.x
    
class Results:
    def __init__(self)->None:
        pass

    def append(self,obj):
        """
        Merge two objects by appending their attributes.

        *Attributes have to be lists!*

        :param obj: The object to be appended.
        :type obj: object

        :return: None
        """
        param_dict = obj.__dict__
        eigen_dict = self.__dict__
        result_dict = ut.BaseFunc().join_dict_values(eigen_dict, param_dict)
        for key in result_dict.keys():
            setattr(self, key, result_dict[key])
        pass


class MinfluxAnalysis:

    def __init__(self):
        pass

    def fit_chunk(self,input_df, estimator='min-quad', plot=False, output=None,**kwargs):
        """
        Fit a chunk of data using the specified estimator.

        :param input_df: Input DataFrame containing 'photons', 'pos', 'weights', and 'time' columns.
        :type input_df: pd.DataFrame
        :param estimator: Estimation method (default is 'min-quad').
        :type estimator: str
        :param plot: Flag indicating whether to plot the fit results (default is False).
        :type plot: bool
        :param output: Output file or path for the plot (default is None).
        :type output: str or None
        :param kwargs: Additional keyword arguments to be passed to the estimator.
        :type kwargs: dict

        :return: Series containing fit results.
        :rtype: pd.Series
        """
        chunk_df = input_df.copy()
        photons = chunk_df['photons'].to_numpy().copy()
        positions = chunk_df['pos'].to_numpy().copy()
        weights = chunk_df['weights'].to_numpy().copy()
        
        if np.mean(photons)>3:
            if estimator=='mle':
                sol, success, fit_vals = Estimators().get_MLE(positions*4*np.pi/LAMBDA, photons, estimator, param_dict={'weights':weights})
            sol, success, fit_vals = Estimators().get_taylorE(positions*4*np.pi/LAMBDA, photons, estimator, param_dict={'weights':weights}) #param_dict={'N_bgr':chunk['N_bgr'].to_numpy()[0]}
            chi2 = np.sqrt(np.average((photons-fit_vals)**2/fit_vals**2))
            chunk = pd.Series({'chi2':chi2,'a0':sol[0],'a1':sol[1],'x0':sol[2],'success':success,'N_fit':np.sum(photons),'N_avg': np.nanmean(photons),'time':chunk_df['time'].mean()})
            if plot:
                self._plot_fit(input_df,fit_vals,output)
        else:
            chunk = pd.Series({'chi2':0.1,'a0':1.,'a1':1.,'x0':0.,'success':True,'N_fit':np.sum(photons),'N_avg': np.nanmean(photons),'time':chunk_df['time'].mean()})
        return chunk
    
    def _plot_fit(self,df,fit_vals,output=None):
        """
        Plot the fit and residuals of the given DataFrame.

        :param df: DataFrame containing 'photons', 'pos', 'weights', 'fit', and 'residuals' columns.
        :type df: pd.DataFrame
        :param fit_vals: Fit values obtained from the fitting procedure.
        :type fit_vals: np.ndarray
        :param output: Output file or path for saving the plot.
        :type output: str or None
        """
        photons = df['photons'].to_numpy().copy()
        positions = df['pos'].to_numpy().copy()
        weights = df['weights'].to_numpy().copy()

        artefacts=Artefacts()
        assert output is not None
        df['fit'] = fit_vals
        df['residuals'] = df['photons'] - df['fit']
        sorted_df = df.sort_values(by='pos')
        fit_x = sorted_df.pos.to_numpy()
        fit_y = sorted_df.fit.to_numpy()
        fit_res = sorted_df.residuals.to_numpy()
        fig, ax =plt.subplots(2,2,figsize=(10,10),sharex=True)
        fig.suptitle('Fit and Residuals')
        ax[0,0].scatter(positions,photons,color='r',alpha=0.6,marker='x',label='data')
        ax[0,0].scatter(fit_x,fit_y,color='b',marker='.',alpha=0.6,label='fit')
        ax[0,0].legend()
        ax[0,0].set_ylabel('photon counts (1)')
        ax[0,0].set_ylim(0,50)
        bins = np.arange(min(fit_x),max(fit_x),step=.5)
        hist, edges = np.histogram(fit_x,bins=bins,weights=fit_res,density=True)
        ax[0,1].plot(edges[:-1],hist,c='k',label='residuals')
        ax[0,1].legend()
        ax[0,1].set_ylabel('normalized residuals (a.u.)')
        ax[0,1].set_ylim(-15,15)
        sns.kdeplot(data=df,x='pos',y='photons',ax=ax[1,0],fill=True,levels=20,cmap="YlOrBr")
        ax[1,0].scatter(fit_x,fit_y,color='b',marker='.',alpha=0.6,label='fit')
        ax[1,0].set_ylim(0,50)
        sns.kdeplot(data=df,x='pos',y='residuals',ax=ax[1,1],fill=True,levels=20,cmap="YlOrBr")
        ax[1,1].set_ylim(-15,15)
        artefacts.add_figures([fig], ['residuals'])
        artefacts.save_figures(out_dir = output)
        pass
    
    def assign_chunk_id(self, df, mode='tuple', chunk_size=50, max_chunks=10, overlap=0.,bin_size=50,**kwargs):
        """
        Assign chunk IDs to the given DataFrame based on the specified mode and chunking parameters.

        :param df: Input DataFrame containing relevant data.
        :type df: pd.DataFrame
        :param mode: Chunking mode ('tuple' or 'photons').
        :type mode: str, optional
        :param chunk_size: Size of each chunk.
        :type chunk_size: int, optional
        :param max_chunks: Maximum number of chunks.
        :type max_chunks: int, optional
        :param overlap: Overlap percentage between chunks.
        :type overlap: float, optional
        :param bin_size: Bin size for assigning bin IDs to chunks.
        :type bin_size: int, optional
        :param kwargs: Additional keyword arguments.
        :return: DataFrame with assigned chunk IDs.
        :rtype: pd.DataFrame
        """
        try:
            if bin_size is None and chunk_size is None:
                if mode=='photons':
                    bin_size = min(2E4,df['photons'].sum()) - 1
                    chunk_size = bin_size - 1
                if mode=='tuple':
                    bin_size = min(2E3,df['tuple'].nunique()) - 1
                    chunk_size = bin_size - 1
            elif bin_size is None and not chunk_size is None:
                bin_size = chunk_size
            try:
                assert bin_size>=chunk_size, 'Invalid combination of chunk_size and bin_size!'
            except AssertionError as e:
                bin_size = chunk_size
            data = df.copy()
            data['tuple'] = data['tuple']-data['tuple'].min()
            chunk_size = int(chunk_size)
            bin_size = int(bin_size)
            max_chunks = int(max_chunks)
            overlap= round(overlap,2) #targeted overlap of the individual chunks in %

            # create overlapping chunks
            overlapping_df = pd.DataFrame(columns=['chunk_id', 'tuple','time'])# Create an empty dataframe to store overlapping chunks
            
            for chunk_id,start in enumerate(range(0, max_chunks*chunk_size, max(1,int(chunk_size * (1 - overlap))))):
                end = start + chunk_size
                if mode=='photons':# select chunks by number of photons with size photons
                    #TODO: this doesnt work!
                    cum_data = data.groupby(['tuple'], group_keys=False).agg({'photons': 'sum'}).cumsum().reset_index().copy()
                    minP, maxP = cum_data['photons'].min(), cum_data['photons'].max()
                    if (end>maxP)&(chunk_id==0):
                        print('Chunksize exceeds number of available photons. Consider smaller chunks.')
                    if ((start>maxP) | (end>maxP) | (end>max_chunks*chunk_size)) & (not (minP==maxP)):
                        break
                    relevant_part = cum_data.loc[(cum_data['photons'] >= start) & (cum_data['photons'] < end)].copy()
                elif mode=='tuple':# select chunks by number of tuples
                    minP, maxP = data['tuple'].min(), data['tuple'].max()
                    if (end>maxP)&(chunk_id==0):
                        print('Chunksize exceeds number of available tuples. Consider smaller chunks.')
                    if ((start>maxP) | (end>maxP) | (end>max_chunks*chunk_size)) & (not (minP==maxP)):
                        break
                    relevant_part = data.loc[(data['tuple'] >= start) & (data['tuple'] < end)].copy()
                chunk = data.loc[(data['tuple'].isin(relevant_part.tuple))].copy()# Extract data within the current chunk
                if not chunk.empty:
                    chunk['chunk_id'] = chunk_id# Assign 'chunk_id' to the rows
                    chunk['time'] = chunk['tuple'].mean().astype(int)+df['tuple'].min().astype(int)#time in ms, since one triple takes 1ms to record
                    overlapping_df = pd.concat([overlapping_df, chunk])# Concatenate the chunk to the resulting dataframe
                if chunk_id >= max_chunks:
                    break
            overlapping_df.reset_index(drop=True, inplace=True)# Reset the index of the resulting dataframe
            
            #assign bin_ids to chunks
            mapFrame = df.copy()
            if mode=='photons':
                # select chunks by number of photons with size photons
                tmp_df = mapFrame.groupby('tuple')['photons'].sum().reset_index().rename(columns={'photons':'mean'})
                tmp_df['cumsum'] = tmp_df['mean'].cumsum()
                mapFrame = mapFrame.merge(tmp_df,on=['tuple'],how='left')
                mapFrame['bin_id'] = mapFrame['cumsum'].apply(lambda x: int(x/bin_size))
            elif mode=='tuple' or mode=='tuple-rand':
                # select chunks by number of tuples
                mapFrame['bin_id'] = mapFrame['tuple'].apply(lambda x: int(x/bin_size)).copy()
   
            mapFrame['tuple'] = mapFrame['tuple'].astype(int)
            tuple_bin_map = mapFrame.set_index('tuple')['bin_id'].to_dict()
            overlapping_df['bin_id'] = overlapping_df['time'].map(tuple_bin_map)
        except:
            raise Exception('Chunking failed!')
        try:
            overlapping_df['bin_id']
        except:
            raise Exception('Did not find any bins in overlap!')
        return overlapping_df.loc[~np.isnan(overlapping_df['bin_id'])]
    
    def grid_data(self,df,num_points=100):
        """
        Grid the given DataFrame to interpolate and create a new DataFrame with a specified number of points.

        :param df: Input DataFrame containing 'pos' and 'photons' columns.
        :type df: pd.DataFrame
        :param num_points: Number of points for the new grid.
        :type num_points: int, optional
        :return: Gridded DataFrame with interpolated 'photons' values.
        :rtype: pd.DataFrame
        """
        df_unique = df.groupby('pos')['photons'].mean().reset_index()
        num_points = min(len(df_unique['pos']),num_points)
        grid = np.linspace(df_unique['pos'].min(), df_unique['pos'].max(), num_points)
        photons_grid = griddata(df_unique['pos'].to_numpy(), df_unique['photons'].to_numpy(), grid, method='cubic')
        df_grid = pd.DataFrame({'pos': grid, 'photons': photons_grid}) # new gridded dataframe
        return df_grid