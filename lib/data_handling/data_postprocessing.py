"""
Module that provides functionality in order to postprocess data.

@author: Thomas Arne Hensel, 2023
"""

import os, multiprocessing
from functools import partial
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from pathlib import Path

import lib.utilities as ut
from lib.constants import *
from lib.data_handling.utilities import Converter
from lib.plotting.artefacts import Artefacts, Meta_Analysis_Figures
from lib.data_handling.data_converter import Converter

def postprocess_single_file(full_file_path):
    """
    Post process fitting results from an experiment.

    :param full_file_path: Full path to the fitting results file.
    :type full_file_path: str
    :return: Dictionary containing processed results.
    :rtype: dict
    """
    n_f, fname = os.path.split(full_file_path)
    result_dict = ut.Importer().load_json(full_file_path)
    results,success = PostProcessor().process_results(n_f, result_dict)
    if success:
        ut.Exporter().write_json(results, filestr = n_f + '/' + results['method'] + '_processed_results.json')
    results = results | {'file': f'{n_f}'}
    return {f'{full_file_path}': results}

def process_group(group):
    """
    Process a group by calculating the normalized distance 'd_norm'.

    :param group: Pandas DataFrame group.
    :type group: pandas.DataFrame
    :return: Processed group.
    :rtype: pandas.DataFrame
    """
    try:
        group['d_norm'] = group.groupby(['state_id', 'chunk_id'])['d'].transform(lambda x: np.linalg.norm(x))
    except:
        pass
    return group

class PostProcessorFacade:
    """
    Class to post-process data.

    Operates only on results of a study, no reference to the experiment object.
    """

    def __init__(self,collect_artefacts=True):
        """
        Initialize the PostProcessorFacade.

        :param collect_artefacts: Whether to collect artefacts during postprocessing.
        :type collect_artefacts: bool
        """
        self.results_file = ut.Labeler().stamp('results')[0] + '.json'
        self.collect_artefacts = collect_artefacts
        pass

    def postprocess_data(self, input_dir, output_dir, max_files=10**4):
        """
        Multi-threaded postprocessing of the data.

        :param input_dir: Root directory of the input data.
        :type input_dir: str
        :param output_dir: Root directory of the output data.
        :type output_dir: str
        :param max_files: Maximum number of files to process.
        :type max_files: int
        """
        processable_files = ut.BaseFunc().find_files(input_dir, lambda file: '_fitting_results.json' in file, max_files = max_files)
        task = partial(postprocess_single_file)
        #for file in processable_files:
        #    task(file)
        results = []
        with multiprocessing.Pool() as pool:
            results = pool.map(task, processable_files)

        #save post-processed results dictionary in output_dir ('analysis')
        results_str = output_dir + ut.Labeler().stamp('fitting_results')[0] + '.json'
        res_dict = {}
        for result in results:
            res_dict = res_dict | result
        ut.Exporter().write_json(res_dict, filestr = results_str)
        df = self._convert_results_to_df(res_dict)
        df.to_pickle(output_dir + ut.Labeler().stamp('meta_results')[0] + '.pkl')       
        pass

    def _convert_results_to_df(self, result_dict):
        """
        Convert post-processed results dictionary to a DataFrame.

        :param result_dict: Post-processed results dictionary.
        :type result_dict: dict
        :return: DataFrame containing post-processed results.
        :rtype: pandas.DataFrame
        """
        new_df = pd.DataFrame(result_dict.values())
        new_df['visibility_1M']=new_df.loc[:,'kappa_1M'].apply(lambda x: 1/np.asarray(x,dtype='float64'))
        new_df['visibility_2M']=new_df.loc[:,'kappa_2M'].apply(lambda x: 1/np.asarray(x,dtype='float64'))
        new_df['d_norm']=new_df.loc[:,'distances'].apply(lambda x: np.linalg.norm(np.asarray(x,dtype='float64'),axis=0))
        new_df['d_norm_NALM']=new_df.loc[:,'d_NALM'].apply(lambda x: np.linalg.norm(np.asarray(x,dtype='float64'),axis=0))
        new_df['N_fit']=new_df.loc[:,'N_fit_2M'].apply(lambda x: np.sum(np.asarray(x,dtype='int'),axis=0))
        new_df['gt']=new_df.loc[:,'ground_truth']
        new_df['trace'] = 0
        new_df['target'] = 0
        return new_df

class PostProcessor:
    """
    Class to post-process data.

    Operates only on results of a study, no reference to the experiment object.
    """

    def __init__(self):
        pass

    def process_results(self, dir, result_dict):
        """
        Method to post-process results from a study.
        1. Cluster positions and retrieve means.
        2. Compare result to ground truth if available.
        3. Calculate CRB either for true or for estimated position.

        :param dir: Directory of results file. Look here for source to calculate CRB.
        :type dir: str
        :param result_dict: Dictionary containing results from the study.
        :type result_dict: dict
        :return: Processed results and success status.
        :rtype: tuple(dict, bool)
        """
        p = Path(dir)
        parts = list(p.parts)
        filtered_strings = [string for string in parts if ut.BaseFunc().match_pattern(string, 'nm', match='partial')]
        ground_truth = filtered_strings[0][:-2]

        results = {'method': result_dict['method'], 'ground_truth': int(ground_truth)}
        #TODO: processing of MLE analysis results
        if result_dict['method']=='WINDOW':
            results['window_idx_2M'] = np.array(Converter().partition_array(np.array(result_dict['window_idx_2M']), result_dict['block_size_2M'], seg_idcs=np.array(result_dict['seg_idcs_2M']), truncate=False))
        def get_kap2(sol,i,kap1):
            kap = (sol[0]-result_dict['N_bgr'][i])/sol[1]
            return kap
        try:
            Lambda0 = result_dict['Lambda0_2M'][0]
            N_tot_2M = result_dict['N_tot_2M']
            def dist(kappa1,kappa2):
                kappa1 = max(1.,kappa1)
                if kappa1<1 or kappa2<kappa1:
                    return np.nan
                else:
                    return Lambda0/(4*np.pi) * np.arccos(2*(kappa1/kappa2)**2 - 1)
            points_1M, points_2M = [], []
            kappa_1M = np.array([[1],[1]])
            kappa_2M = kappa_1M
            try: # calc 1M pos
                lat_offset = [a/b * min_idx - a/2 for a,b,min_idx in zip(result_dict['L_1M'], result_dict['num_pos_1M'], result_dict['min_idx_1M'])] # find true idx and convert to spatial coordinates [nm]
                partitioned_offset = Converter().partition_array(np.array(lat_offset), result_dict['block_size_1M'], seg_idcs=np.array(result_dict['seg_idcs_1M']), truncate=False)
                points = [np.array([x,y]).reshape((1,2)) for x,y in zip(*partitioned_offset)]
                points_1M = points
                N_fit_1M = np.array(Converter().partition_array(np.array(result_dict['N_fit_1M']), result_dict['block_size_1M'], seg_idcs=np.array(result_dict['seg_idcs_1M']), truncate=False))
                chi2_1M = np.array(Converter().partition_array(np.array(result_dict['chi2_1M']), result_dict['block_size_1M'], seg_idcs=np.array(result_dict['seg_idcs_1M']), truncate=False))#Converter().partition_array(np.array(result_dict['chi2_2M']), result_dict['block_size_2M'], seg_idcs=np.array(result_dict['seg_idcs_2M']), truncate=False)

                #results['pos_1M'] = points
                if not 'kappa_1M' in result_dict.keys():
                    partitioned_sol = Converter().partition_array(np.array(result_dict['raw_sol_1M']), result_dict['block_size_1M'], seg_idcs=np.array(result_dict['seg_idcs_1M']), truncate=False)
                    kappa_1M = np.array([[(sol[0]-result_dict['N_bgr'][i])/sol[1] for sol in axis] for i, axis in enumerate(partitioned_sol)])
                else:
                    kappa_1M = np.array(result_dict['kappa_1M']).reshape(-1,1)
            except:
                print('could not postprocess/retrieve 1M positions')
            try: # calc 2M pos
                lat_offset = [a/b * min_idx - a/2 for a,b,min_idx in zip(result_dict['L_2M'], result_dict['num_pos_2M'], result_dict['min_idx_2M'])]
                partitioned_offset = Converter().partition_array(np.array(lat_offset), result_dict['block_size_2M'], seg_idcs=np.array(result_dict['seg_idcs_2M']), truncate=False)
                points = [np.array([x,y]).reshape((1,2)) for x,y in zip(*partitioned_offset)]
                points_2M = points
                partitioned_sol = Converter().partition_array(np.array(result_dict['raw_sol_2M']), result_dict['block_size_2M'], seg_idcs=np.array(result_dict['seg_idcs_2M']), truncate=False)
                kappa_2M = np.array([[get_kap2(sol,i,kap1=kappa_1M) for sol in axis] for i, axis in enumerate(partitioned_sol)])
                N_fit_2M = np.array(Converter().partition_array(np.array(result_dict['N_fit_2M']), result_dict['block_size_2M'], seg_idcs=np.array(result_dict['seg_idcs_2M']), truncate=False))
                chi2_2M = np.array(Converter().partition_array(np.array(result_dict['chi2_2M']), result_dict['block_size_2M'], seg_idcs=np.array(result_dict['seg_idcs_2M']), truncate=False))
            except:
                print('could not retrieve 2M positions')
            try: # continue to do NALM processing, if possible:
                d_NALM = np.array([np.nan,np.nan])
                nan_pos_1M = [np.any(np.isnan(pos)) for pos in points_1M]
                nan_pos_2M = [np.any(np.isnan(pos)) for pos in points_2M]
                valid_1M_pos = [points_1M[i] for i, val in enumerate(nan_pos_1M) if not val]
                valid_2M_pos = [points_2M[i] for i, val in enumerate(nan_pos_2M) if not val]
                if len(valid_1M_pos)>0 and len(valid_2M_pos)>0:
                        points_1M = np.asarray(valid_1M_pos)
                        points_2M = np.asarray(valid_2M_pos)
                        min_pos_1M = np.transpose(points_1M,axes=(1,0,2))
                        min_pos_2M = np.transpose(points_2M,axes=(1,0,2))
                        chi2_weights_1M = np.array([[chi for i,chi in enumerate(axis) if not nan_pos_1M[i]] for axis in chi2_1M]).T
                        chi2_weights_2M = np.array([[chi for i,chi in enumerate(axis) if not nan_pos_2M[i]] for axis in chi2_2M]).T
                        N_weights_1M = np.array([[n for i,n in enumerate(axis) if not nan_pos_1M[i]] for axis in N_fit_1M]).T
                        N_weights_2M = np.array([[n for i,n in enumerate(axis) if not nan_pos_2M[i]] for axis in N_fit_2M]).T
                        min_pos_avg_1M = np.average(min_pos_1M, axis=1, weights=np.repeat([N_weights_1M/chi2_weights_1M],1,axis=0))
                        min_pos_avg_2M = np.average(min_pos_2M, axis=1, weights=np.repeat([N_weights_2M/chi2_weights_2M],1,axis=0))
                        d_NALM = 2*np.abs(min_pos_avg_1M - min_pos_avg_2M).reshape(2,) # NALM distance for each axis
            except:
                print('Could not calculate NALM distance')
            try: # estimate 2M distance
                kap1 = np.hstack(len(kappa_2M[0])*[kappa_1M])#TODO: what if kappa_1M is longer than 1, i.e. was locally determined?
                kap2 = kappa_2M
                dist_est = [[dist(k1,k2) for k1,k2 in zip(ax1,ax2)] for ax1,ax2 in zip(kap1,kap2)]

                xpoints,ypoints = [[[com-d/2,com+d/2] for com,d in zip(axis1,axis2)] for axis1,axis2 in zip(partitioned_offset,dist_est)]
                points_2M = [np.array([[x[0],y[0]],[x[1],y[1]]]) for x,y in zip(xpoints,ypoints)]            

                nan_pos = [np.any(np.isnan(pos)) for pos in points_2M]
                points_2M = [points_2M[i] for i, val in enumerate(nan_pos) if not val]
                if len(points_2M)>0:
                    points = np.asarray(points_2M)
                    if points.size>4:
                        clustered_pos = self.get_even_clusters(points)
                    else:
                        clustered_pos = points.reshape(2,1,2)
                    chi2_weights = np.array([[chi for i,chi in enumerate(axis) if not nan_pos[i]] for axis in chi2_2M]).T # list of chi2 values for each fit (each line in each axis)
                    N_weights = np.array([[n for i,n in enumerate(axis) if not nan_pos[i]] for axis in N_fit_2M]).T
                    #chi2_weights = [chi2_weights[i] for i, val in enumerate(nan_pos) if not val]
                    cluster_avg = np.average(clustered_pos, axis=1, weights=np.repeat([N_weights/chi2_weights],2,axis=0))
                    cluster_com = np.average(cluster_avg,axis=0) # center of mass of two molecules
            except:
                print('distance estimate failed')
                return results, False
            results = results | {'kappa_1M' : kappa_1M, 'kappa_2M' : kappa_2M, 'distances': dist_est, 'chi_1M':chi2_1M, 'chi_2M':chi2_2M,'chi2_weights': chi2_2M, 'd_NALM': d_NALM, 'N_fit_2M':N_fit_2M, 'N_tot_2M':N_tot_2M}
        except:
            return results, False
        # load source yaml and compare to ground truth
        CRB_dict = {'avg_sig':np.nan}
        segments = ut.BaseFunc().find_files(dir, lambda file: (file.endswith('.yaml') and 'experiments' in file), max_files=10)
        if len(segments)>0:
            experiments = ut.Importer().load_yaml(segments[0])
            try:
                X = experiments[0].setup.sample.config.mol_pos
            except:
                try:
                    X = cluster_avg
                except:
                    X = None
        results = results | CRB_dict
        return results, True

    def get_even_clusters(self, pos, anchor=None):
        """
        Method for K-means clustering with even cluster-size.

        :param pos: Positions in shape (K, M, dim) for N estimates of M molecules in dim dimensions.
        [ [[x1, y1, ...], [x2, y2, ...], ..., [xM, yM, ...]], [], ..., [n-th estimate] ]
        :type pos: numpy.ndarray
        :param anchor: Anchor point for clustering (default is None).
        :type anchor: numpy.ndarray, optional
        :return: Ordered positions in shape (M, K, dim).
        :rtype: numpy.ndarray
        """
        Kn, M, dim = pos.shape
        pos = pos.reshape((Kn * M, dim))
        n_clusters = M # TODO: allow different number of clusters than number of molecules
        cluster_size = int(np.ceil(len(pos)/n_clusters))
        model = KMeans(n_clusters, random_state=42, n_init=5)
        model.fit(pos)
        #print(pos, flush=True)
        centers = model.cluster_centers_
        
        centers = centers.reshape(-1, 1, pos.shape[-1]).repeat(cluster_size, 1).reshape(-1, pos.shape[-1])
        distance_matrix = cdist(pos, centers)
        clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size
        centers = clusters
        new_pos = np.empty((M,Kn,dim))

        for i in range(0,n_clusters):
            new_pos[i]=pos[centers==i]
        return new_pos
    

class MetaAnalysis:

    def __init__(self):
        pass

    def meta_analysis(self, new_base_dir):
        """
        Visualize overarching statistics and batch-dependent characteristics of all analyzed data.

        :param new_base_dir: The base directory for saving analysis results.
        :type new_base_dir: str
        """
        file = ut.BaseFunc().find_files(new_base_dir, lambda file: ut.BaseFunc().match_pattern(file, '.pkl', match='partial'),max_files=1)[0]
        df = pd.read_pickle(file)
        artefacts = Artefacts()
        try:
            figures, names = Meta_Analysis_Figures().fig_power_balance(df)
            artefacts.add_figures(figures, names)
            artefacts.save_figures(out_dir = new_base_dir + 'analysis/')
        except:
            print('Failed to create visibility figure')
        try:
            figures, names = Meta_Analysis_Figures().fig_correlations(df)
            artefacts.add_figures(figures, names)
            artefacts.save_figures(out_dir = new_base_dir + 'analysis/')
        except:
            print('Failed to generate figure of correlations')
        try:
            figures, names = Meta_Analysis_Figures().fig_correlation_matrix(df,'d_norm')
            artefacts.add_figures(figures, names)
            artefacts.save_figures(out_dir = new_base_dir + 'analysis/')
        except:
            print('Failed to generate correlation matrix')
        try:
            figures, names = Meta_Analysis_Figures().fig_estimated_distances(df)
            artefacts.add_figures(figures, names)
            artefacts.save_figures(out_dir = new_base_dir + 'analysis/')
        except:
            print('Failed to generate figure of distance etsimates vs ground truth')
        try:
            figures, names = Meta_Analysis_Figures().fig_individual_batch_histograms(df)
            artefacts.add_figures(figures,names)
            artefacts.save_figures(out_dir = new_base_dir + 'analysis/')
        except:
            print('Failed to generate figure of individual batches')                
        pass