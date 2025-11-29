"""
Module that provides functionality in order to parse minflux data.
"""
import os, multiprocessing
import json
from functools import partial
import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
import shutil as sh
from pathlib import Path
import itertools
from copy import deepcopy
import ruptures as rpt # change point detection
from sklearn.cluster import DBSCAN
from k_means_constrained import KMeansConstrained

import lib.utilities as ut
from lib.constants import *
from lib.data_handling.utilities import Converter
from lib.data_handling.data_analysis import MinfluxAnalysis
from lib.plotting.artefacts import Figures, Artefacts, Minflux_Figures

def parse_single_file(path,base_in=None,base_out=None,parser=None):
    """
    Parse a single file using the specified parser.

    This function reads a file, transforms its contents using a parser, and saves the resulting DataFrame as a CSV file.

    :param path: The path to the file to be parsed.
    :type path: str
    :param base: The base input directory (unused in the current implementation).
    :type base: str, optional
    :param base_out: The base output directory where the parsed CSV file will be saved.
    :type base_out: str, optional
    :param parser: The parser object used to read and transform the file.
    :type parser: Parser, optional
    """
    try:
        # read file
        original_df = parser._load_as_frame(path)
        # create dataframe
        new_df = parser._transform_frame(original_df)
        # save dataframe as csv
        parser._save_frame(new_df,path,base_in,base_out)
    except:
        print(f'could not parse {path}')
    pass

class MF_ParserFacade:
    """
    Facade class for parsing Minflux files in a specified directory.

    This class provides a high-level interface for parsing Minflux files in a given directory using a MinfluxParser.

    :param type: The type of Minflux parser to be used.
    :type type: str, optional
    """

    def __init__(self,type=None):
        """
        Initialize the Minflux Parser Facade.

        :param type: The type of Minflux parser to be used.
        :type type: str, optional
        """
        self.type=type
        self.parser = MinfluxParser(type=self.type)
        pass

    def parse_directory(self,input,output,max_files=3):
        """
        Parse Minflux files in the specified directory.

        This method parses Minflux files in the input directory using a MinfluxParser. It performs multithreaded parsing
        of files and then multi-threaded post-processing on the list of files.

        :param input: The input directory containing Minflux files to be parsed.
        :type input: str
        :param output: The output directory where parsed CSV files will be saved.
        :type output: str
        :param max_files: The maximum number of files to be processed.
        :type max_files: int, optional
        """
        if not os.path.exists(input):
            raise Exception('Input directory does not exist!')
        # multithreaded parsing of files
        processable_files = self.parser._find_files(input,max_files=max_files)
        #multi-thread post-processing on the list of files
        task = partial(parse_single_file,base_in=input,base_out=output+'parsed/',parser=self.parser)
        #for file in processable_files:
        #    task(file)
        results = []
        with multiprocessing.Pool() as pool:
            results = pool.map(task, processable_files)
        pass
class MinfluxParser:
    """
    Parser class for Minflux data files.

    This class provides methods to find, load, transform, and save Minflux data files of different types.

    :param type: The type of Minflux parser to be used.
    :type type: str, optional
    """

    def __init__(self,type=None):
        """
        Initialize the Minflux Parser.

        :param type: The type of Minflux parser to be used.
        :type type: str, optional
        :raises Exception: If the type of Minflux parser is not specified.
        """
        self.type=type
        if type is None:
            raise Exception('Specify type of Minflux Parser!')
        pass

    def _find_files(self,input_dir,max_files=10):
        """
        Find Minflux data files in the specified directory.

        This method finds all available Minflux data files based on the specified type.

        :param input_dir: The input directory containing Minflux data files.
        :type input_dir: str
        :param max_files: The maximum number of files to be processed.
        :type max_files: int, optional
        :return: List of Minflux data files.
        :rtype: List[str]
        """
        if self.type=='philip':
            pattern = '.JSON'
        elif self.type=='otto':
            pattern='.txt'
        elif self.type=='thomas':
            pattern='.yaml'
        #processable_files = ut.BaseFunc().find_files(input_dir, lambda file: ut.BaseFunc().match_pattern(file, pattern, match='partial'), max_files = max_files)
        processable_files = ut.BaseFunc().find_files(input_dir, lambda file: file.endswith(pattern), max_files = max_files)
        return processable_files

    def _load_as_frame(self,path):
        """
        Load Minflux data file as a DataFrame.

        This method reads the content of a Minflux data file and returns it as a DataFrame.

        :param path: The path to the Minflux data file.
        :type path: str
        :return: DataFrame containing the Minflux data.
        :rtype: pandas.DataFrame
        """
        if self.type=='otto':
            df = self._read_otto(path)
        elif self.type=='philip':
            df = self._read_philip(path)
        elif self.type=='thomas':
            df = self._read_thomas(path)
        return df

    def _transform_frame(self,df):
        """
        Transform the Minflux data frame.

        This method transforms the Minflux data frame to return position and counts.

        :param df: The original Minflux data frame.
        :type df: pandas.DataFrame
        :return: Transformed Minflux data frame.
        :rtype: pandas.DataFrame
        """
        if self.type=='otto':
            new_df = self._transform_otto(df)
        elif self.type=='philip':
            new_df = self._transform_philip(df)
        elif self.type=='thomas':
            new_df = self._transform_thomas(df)
        return new_df

    def _save_frame(self,df,path,base_in,base_out):
        """
        Save the Minflux data frame.

        This method labels and saves the Minflux data frame as a pickle file.

        :param df: The Minflux data frame to be saved.
        :type df: pandas.DataFrame
        :param path: The path to the original Minflux data file.
        :type path: str
        :param base: The base input directory.
        :type base: str
        :param base_out: The base output directory.
        :type base_out: str
        """
        n_f, fname = os.path.split(path)
        f_base, f_extension = os.path.splitext(fname)
        unique_name, _ = ut.Labeler().stamp(f_base)
        df['experiment_id'] = unique_name
        
        curr_dir = os.path.dirname(path)
        diff_path = os.path.relpath(curr_dir, base_in)
        new_dir = os.path.join(base_out, diff_path) + '/'
        
        os.makedirs(new_dir + unique_name + '/', exist_ok=True) # create a directory for this file
        sh.copyfile(path, new_dir + unique_name + '/' + 'source' + f_extension) # copy original data into new directory
        df.to_pickle(new_dir + unique_name + '/' + 'parsed.pkl')

        #artefacts = Artefacts()
        #figures, names = Minflux_Figures().fig_COM_movement(df)
        #artefacts.add_figures(figures, names)
        #artefacts.save_figures(out_dir = new_dir + unique_name + '/')
        pass

    def _transform_otto(self,df):
        """
        Transform the Minflux Otto data frame.

        This method selects the last iteration, extracts position and photon count values, and creates a new DataFrame
        with merged values for both X and Y axes.

        :param df: The original Minflux Otto data frame.
        :type df: pandas.DataFrame
        :return: Transformed Minflux data frame.
        :rtype: pandas.DataFrame
        """
        # select last iteration
        minL_y = df['Ly/2 (nm)'].min()
        minL_x = df['Lx/2 (nm)'].min()
        df = df[np.isclose(df['Lx/2 (nm)'], minL_x,atol=1) * np.isclose(df['Ly/2 (nm)'], minL_y,atol=1)].reset_index(drop=True).copy()

        # Create empty lists to store the values and their corresponding tuple indices
        x_pos_with_index = []
        y_pos_with_index = []

        # Iterate over the rows of the DataFrame and assign tuple indices
        for index, row in df.iterrows():
            x_values = [row['xM-pos (nm)'] + row['Lx/2 (nm)'], row['xM-pos (nm)'], row['xM-pos (nm)'] - row['Lx/2 (nm)']]
            y_values = [row['yM-pos (nm)'] + row['Ly/2 (nm)'], row['yM-pos (nm)'], row['yM-pos (nm)'] - row['Ly/2 (nm)']]
            
            # Append the values with their corresponding tuple indices
            x_pos_with_index.extend(zip(x_values, [index] * len(x_values)))
            y_pos_with_index.extend(zip(y_values, [index] * len(y_values)))

        # Extract the separate values and tuple indices
        x_pos, x_idx = zip(*x_pos_with_index)
        y_pos, y_idx = zip(*y_pos_with_index)

        # create a new dataframe with the merged values
        #x_pos = list(itertools.chain.from_iterable(zip(df['xM-pos (nm)']+df['Lx/2 (nm)'],df['xM-pos (nm)'],df['xM-pos (nm)']-df['Lx/2 (nm)'])))
        #y_pos = list(itertools.chain.from_iterable(zip(df['yM-pos (nm)']+df['Ly/2 (nm)'],df['yM-pos (nm)'],df['yM-pos (nm)']-df['Ly/2 (nm)'])))
        
        x_photons = list(itertools.chain.from_iterable(zip(df['photons at xL-Lx/2'],df['photons at xL'],df['photons at xL+Lx/2'])))
        y_photons = list(itertools.chain.from_iterable(zip(df['photons at yL-Ly/2'],df['photons at yL'],df['photons at yL+Ly/2'])))
        trace = list(itertools.chain.from_iterable(zip(df['trace'],df['trace'],df['trace'])))
        
        pos = list(x_pos) + list(y_pos)
        photons = x_photons + y_photons
        assert len(pos)==len(photons)
        ground_truth = np.full(len(pos),df['gt'].mean())
        batch = np.full(len(pos),df['batch'].mean())
        trace_index = trace*2
        axis_index = np.concatenate([np.full(len(x_pos),int(0)),np.full(len(y_pos),int(1))])
        tuple_index = list(x_idx) + list(y_idx)
        new_df = pd.DataFrame({'pos': pos,'photons': photons,'gt':ground_truth,'batch':batch,'axis': axis_index, 'tuple': tuple_index, 'trace':trace_index})
        return new_df
    
    def _transform_philip(self,df):
        """
        Transform the Minflux Philip data frame.

        This method creates a new DataFrame with merged values for both X and Y axes.

        :param df: The original Minflux Philip data frame.
        :type df: pandas.DataFrame
        :return: Transformed Minflux data frame.
        :rtype: pandas.DataFrame
        """
        # create a new dataframe with the merged values
        Lx = df.loc[0,'L2']*10**3
        x_pos = list(itertools.chain.from_iterable(zip(df['x_mean']*10**3-Lx,df['x_mean']*10**3,df['x_mean']*10**3+Lx)))
        y_pos = list(itertools.chain.from_iterable(zip(df['x_mean']*10**3-Lx,df['x_mean']*10**3,df['x_mean']*10**3+Lx)))
        x_photons = list(itertools.chain.from_iterable(zip(df['cl'],df['cm'],df['cr'])))
        y_photons = list(itertools.chain.from_iterable(zip(df['cl'],df['cm'],df['cr'])))
        new_df = pd.DataFrame({'x': x_pos,'y':y_pos,'x_photons':x_photons,'y_photons':y_photons})
        return new_df
    
    def _transform_thomas(self,df):
        """
        Transform the Minflux Thomas data frame.

        This method creates a new DataFrame with merged values for both X and Y axes and additional parameters.

        :param df: The original Minflux Thomas data frame.
        :type df: pandas.DataFrame
        :return: Transformed Minflux data frame.
        :rtype: pandas.DataFrame
        """
        df.loc[:,'trace'] = 0
        # add tuple index
        pos = list(df.x) + list(df.y)
        photons = list(df.x_photons) + list(df.y_photons)
        assert len(pos)==len(photons)
        ground_truth = np.full(len(pos),df['gt'].mean())
        trace_index = list(df.trace)*2
        axis_index = np.concatenate([np.full(len(df.x),int(0)),np.full(len(df.y),int(1))])
        tuple_index = list(df.index//3)*2
        batch = np.full(len(pos),df['batch'].mean())
        new_df = pd.DataFrame({'pos': pos,'photons': photons,'gt':ground_truth,'batch':batch,'axis': axis_index, 'tuple': tuple_index, 'trace':trace_index})
        
        #calculate MINFLUX parameter L
        L = new_df.groupby(['axis', 'tuple'])['pos'].apply(lambda x: np.abs(x.max() - x.min()))
        L.name = 'L'  # Name the new Series
        # Merge the calculated values with the original DataFrame
        new_df = new_df.merge(L, left_on=['axis', 'tuple'], right_index=True)
        # Find the minimal value of 'L'
        min_L = new_df['L'].min()
        # Select only the rows where 'L' is minimal
        tol = 1
        last_iter_rows = new_df[np.isclose(new_df['L'], min_L, atol=tol)].copy()
        return last_iter_rows

    def _read_philip(self,file_path):
        """
        Read Minflux Philip data from a JSON file.

        :param file_path: Path to the Minflux Philip JSON file.
        :type file_path: str
        :return: Processed Minflux Philip data frame.
        :rtype: pandas.DataFrame
        """
        # Load JSON data from file
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        # Extract unique 'name' values
        unique_names = list(set(item['name'] for item in json_data))

        # Create an empty DataFrame
        df = pd.DataFrame()

        # Iterate over unique 'name' values and collect corresponding 'value' in columns
        for name in unique_names:
            values = [item['value'] for item in json_data if item['name'] == name]
            df[name] = pd.Series(values)
        return df

    def _read_otto(self,file_path):
        """
        Read Minflux Otto data from a text file.

        :param file_path: Path to the Minflux Otto text file.
        :type file_path: str
        :return: Processed Minflux Otto data frame.
        :rtype: pandas.DataFrame
        """
        p = Path(file_path)
        parts = list(p.parts)
        filtered_strings = [string for string in parts if ut.BaseFunc().match_pattern(string, 'nm', match='partial')]
        ground_truth = filtered_strings[0][:-2]
        try:
            filtered_strings = [string for string in parts if ut.BaseFunc().match_pattern(string, 'batch', match='partial')]
            batch = filtered_strings[0][-1:]
        except:
            batch = 0
        with open(file_path, 'r') as f:
            # read header
            header = f.readline().strip().split('\t')
            num_traces = int(float(header[0].split('size')[1].replace(',', '.')))
            trace_length = int(float(header[1].replace(',', '.')))
            num_repeats = int(float(header[2].replace(',', '.')))
            
            # read data
            data = []
            for line in f:
                data.append(line.replace(',', '.').strip().split('\t'))
        f.close()

        # reshape data
        data_df = pd.DataFrame(data, columns=['xM-pos (nm)', 'yM-pos (nm)', 'zM-pos (nm)', 'exposures per dim', 
                                        'photons at xL-Lx/2', 'photons at xL', 'photons at xL+Lx/2',
                                        'photons at yL-Ly/2', 'photons at yL', 'photons at yL+Ly/2', 
                                        'galvo pos x (um)', 'galvo pos y (um)', 'blank', 'Lx/2 (nm)',
                                        'Ly/2 (nm)', 'Lz/2 (nm)', 'photon threshold', 'time per loc (s)',
                                        'blank'])
        data_df = data_df.astype('float')
        
        #data_np = data_df.values.reshape(num_traces, trace_length, num_repeats, -1)

        # create index column
        index = []
        for i in range(num_traces):
            #index += list(range(i*trace_length*num_repeats, (i+1)*trace_length*num_repeats))
            index += list(i*np.ones((trace_length*num_repeats)))
        index = list(map(int, index))
        #data_df = pd.DataFrame(data_np.reshape(num_traces, -1), columns=data_df.columns)
        data_df['trace'] = index
        data_df['gt'] = [int(ground_truth)]*len(index)
        data_df['batch'] = [int(batch)]*len(index)

        # shift 'xM-pos (nm)' column and pad first row with zeroes
        data_df['xM-pos (nm)'] = data_df['xM-pos (nm)'].shift(1)
        data_df['yM-pos (nm)'] = data_df['yM-pos (nm)'].shift(1)
        data_df['zM-pos (nm)'] = data_df['zM-pos (nm)'].shift(1)
        #data_df['xM-pos (nm)'] = data_df['xM-pos (nm)'] - data_df['xM-pos (nm)'].shift(1)
        #data_df['yM-pos (nm)'] = data_df['yM-pos (nm)'] - data_df['yM-pos (nm)'].shift(1)
        #data_df['zM-pos (nm)'] = data_df['zM-pos (nm)'] - data_df['zM-pos (nm)'].shift(1)
        
        data_df.iloc[0, 0:3] = 0
        return data_df.iloc[::2].reset_index(drop=True)#take only every second row to drop irrelevant detector
    
    def _read_thomas(self,path):
        """
        Read Minflux Thomas data from a YAML file.

        :param path: Path to the Minflux Thomas YAML file.
        :type path: str
        :return: Processed Minflux Thomas data frame.
        :rtype: pandas.DataFrame
        """
        exp = ut.Importer().load_yaml(path)
        record = deepcopy(exp.record)
        reshaped_record = Converter().reshape_record(exp,export=False,show=False)
        partitioned_record = Converter().partition_record(reshaped_record, exp.measurement.config.block_size, seg_idcs=None, truncate=False)
        x_photons, y_photons = [phot.flatten() for phot in partitioned_record.photons]
        x_pos = partitioned_record.phi[0][:,:,0,0].flatten()*exp.setup.instrument.config.Lambda0/(4*np.pi)
        y_pos = partitioned_record.phi[1][:,:,0,0].flatten()*exp.setup.instrument.config.Lambda0/(4*np.pi)
        d = np.linalg.norm(np.diff(exp.setup.sample.config.mol_pos,axis=0))

        p = Path(path)
        parts = list(p.parts)
        filtered_strings = [string for string in parts if ut.BaseFunc().match_pattern(string, 'nm', match='partial')]
        try:
            filtered_strings = [string for string in parts if ut.BaseFunc().match_pattern(string, 'batch', match='partial')]
            batch = filtered_strings[0][-1:]
        except:
            batch = 0

        df = pd.DataFrame({'x': x_pos,'y':y_pos,'x_photons': x_photons,'y_photons':y_photons,'gt':[d]*x_pos.shape[0],'batch':[int(batch)]*x_pos.shape[0]})
        return df
    
def filter_single_minflux_file(path,base_in=None,base_out=None,bin_size=None):
    """
    Filter a single Minflux file.

    :param path: Path to the Minflux file.
    :type path: str
    :param base: Base input directory, default is None.
    :type base: str, optional
    :param base_out: Base output directory, default is None.
    :type base_out: str, optional
    :param bin_size: Size of bins for filtering, default is None.
    :type bin_size: int, optional
    :return: DataFrame with mean Minflux data.
    :rtype: pandas.DataFrame
    """
    # read file
    try:
        parsed_df = pd.read_pickle(path)
        # create dataframe
        filtered_df, segmented_df = MinfluxFilter().segment_frame(parsed_df)
        filtered_df = MinfluxFilter().filter_frame(filtered_df,bin_size=bin_size)
        try:
            MinfluxFilter()._save_frame(filtered_df,segmented_df,path,base_in,base_out)
        except:
            pass
        mean_df = filtered_df.groupby(['gt','experiment_id','segment_id'],group_keys=False).mean().reset_index(drop=False).copy()
        mean_df['photons_norm'] = mean_df['photons']/mean_df['photons'].max()
        mean_df['file'] = path
        mean_df['experiment_id'] = filtered_df.loc[:,'experiment_id'].iloc[0]
    except:
        return pd.DataFrame({})
    return mean_df

class MinfluxFilterFacade:

    def __init__(self):
        pass

    def filter_directory(self,input,output,max_files=10):
        """
        Filter files in the input directory and save filtered results.

        :param input: Input directory path.
        :type input: str
        :param output: Output directory path.
        :type output: str
        :param max_files: Maximum number of files to process, default is 10.
        :type max_files: int, optional
        """
        if not os.path.exists(input):
            raise Exception('Input directory does not exist!')
        # multithreaded filtering of files
        processable_files = ut.BaseFunc().find_files(input, lambda file: (ut.BaseFunc().match_pattern(file, 'parsed.pkl', match='partial')), max_files = max_files)
        out = output+'filtered/'
        # filtering via state assignment    
        task = partial(filter_single_minflux_file,base_in=input,base_out=out)
        #for file in processable_files:
        #    task(file)
        results = []
        with multiprocessing.Pool() as pool:
            results = pool.map(task, processable_files)
        results_df = pd.concat(results)
        results_df.to_pickle(out + 'preliminary-filter-results.pkl')
        pass
    
    def post_filtering(self, input, output):
        """
        Post-process filtered Minflux data and infer state means.
        Based on the clear identification of states via traces with bleaching steps,
        we now proceed to label other traces with the knowledge of the state means.

        :param input: Input directory path.
        :type input: str
        :param output: Output directory path.
        :type output: str
        """
        os.makedirs(output, exist_ok=True) # create a directory for this file
        # load preliminary filter results, where state ids were assigned to clear cases
        try:
            preliminary_filter_results = ut.BaseFunc().find_files(input, lambda file: (ut.BaseFunc().match_pattern(file, 'preliminary-filter-results.pkl', match='partial')), max_files = 1)[0]
            results_df = pd.read_pickle(preliminary_filter_results)
        except:
            raise Exception('Could not find preliminary filter results.')
        
        try: # infer state means from counts
            clustered_df = MinfluxFilter()._find_state_clusters(results_df)
            labeled_df = MinfluxFilter()._map_clusters_on_states(clustered_df)
            grid = sns.displot(clustered_df, x="photons", col="gt", row="batch",hue='cluster_label',binwidth=1, height=3, facet_kws=dict(margin_titles=True))
            Figures().save_fig(grid.figure, 'state-cluster', out_path=output)
            grid = sns.displot(labeled_df, x="photons", col="gt", row="batch",hue='state_id',binwidth=1, height=3, facet_kws=dict(margin_titles=True))
            Figures().save_fig(grid.figure, 'state-means', out_path=output)
        except:
            raise Exception('State clustering failed.')
        
        # check if some traces could be identified although the clustering failed
        try:
            remainder = results_df.loc[(results_df['file'].apply(lambda x: not x in labeled_df['file'].unique()))&(~np.isnan(results_df['state_id']))].copy()
            if not remainder.empty:
                remainder.loc[:,'valid'] = True
                labeled_df = pd.concat([labeled_df,remainder])
        except:
            print('Post state assignment of remaining files failed.')

        """try: # infer state_ids
            results_df = self._assign_state_ids(results_df, labeled_df)
        except:
            print('Post Filtering: State identification Failed.')
            pass
        results_df = results_df.loc[~np.isnan(results_df['state_id'])]#activate only if you want to use very strict filter...
        results_df.to_pickle(output + 'filter-results.pkl')"""

        labeled_df.to_pickle(output + 'filter-results.pkl')
        return output
    
    

class MinfluxFilter:
    """
    Class for filtering and processing Minflux data frames.
    """

    def __init__(self):
        """
        Initialize MinfluxFilter object.
        """
        pass

    def segment_frame(self,df):
        """
        Segment the input data frame.

        :param df: Input data frame.
        :type df: pandas.DataFrame
        :return: Tuple containing filtered and segmented data frames.
        :rtype: tuple(pandas.DataFrame, pandas.DataFrame)
        """
        segmented_df = self._segment_frame(df)            
        #segmented_df, success, model = MarkovSegmentation().segment_frame(df)
        filtered_df = segmented_df.loc[~np.isnan(segmented_df.segment_id)].copy()
        filtered_df = filtered_df.groupby(['trace','segment_id'],group_keys=False).apply(self.remove_tuple_outliers,column_name='photons',threshold=2.)
        return filtered_df, segmented_df

    def filter_frame(self,segmented_df,bin_size=None):
        """
        Filter the segmented data frame.

        :param segmented_df: Segmented data frame.
        :type segmented_df: pandas.DataFrame
        :param bin_size: Size of bins for spatial binning, default is None.
        :type bin_size: float, optional
        :return: Binned data frame.
        :rtype: pandas.DataFrame
        """
        # state_id pre-assignment
        filtered_df = segmented_df.groupby(['trace','experiment_id'],group_keys=True).apply(lambda df: self._state_id_pre_assignment(df)).reset_index().copy()
        filtered_df.drop(columns=['level_2'],inplace=True)
        # if we bin points spatially -> loose time information...
        binned_df = filtered_df.groupby(['trace','axis','segment_id'],group_keys=False).apply(self.bin_frame,bin_size=bin_size).reset_index(drop=True)
        return binned_df
    
    def _state_id_pre_assignment(self, df):
        """
        Pre-assign state IDs based on mean values.

        :param df: Input data frame.
        :type df: pandas.DataFrame
        :return: Data frame with pre-assigned state IDs.
        :rtype: pandas.DataFrame
        """
        # method to find unique traces that can be easily recognized as 1M, 2M or 3M traces.
        df['state_id'] = np.nan
        mean_df = df.groupby(['segment_id'],group_keys=False).mean().reset_index(drop=False).copy()
        mean_df['photons_norm'] = mean_df['photons']/mean_df['photons'].max()
        norm_seg = mean_df.photons_norm.to_numpy()
        success = False
        a = .1
        lim_up = 30.
        sM_lim = 10.#single molecules are rarely brighter than 10...
        def test0(norms, counts):
            # ideal trace: three bleaching steps and suitable segments
            t0 = np.all(len(norm_seg)==3)
            t1 = np.all(((1-a <= norms) & (norms <= 1.)) | ((.5-a <= norms) & (norms <= .5+a)) | ((.0 <= norms) & (norms <= a)))
            t2 = np.any(((1-a <= norms) & (norms <= 1.))) & np.any(((.5-a <= norms) & (norms <= .5+a))) & np.any(((.0 <= norms) & (norms <= a)))
            t3 = np.all(counts<=lim_up)
            ds = np.all(np.diff(norms)<0)#make sure steps decend
            return t0 & t1 & t2 & t3 & ds
        def test1(norms, counts):
            # trace with two segments, 2M and 1M part.
            t0 = np.all(len(norm_seg)==2)
            t1 = np.all(((1-a <= norms) & (norms <= 1.)) | ((.5-a <= norms) & (norms <= .5+a)))
            t2 = np.any(((1-a <= norms) & (norms <= 1.))) & np.any(((.5-a <= norms) & (norms <= .5+a)))
            t3 = np.all(counts<=lim_up)&np.all(counts>=3) # make sure there is no background
            ds = np.all(np.diff(norms)<0)
            return t0 & t1 & t2 & t3 & ds
        def test2(norms, counts):
            # trace with two segments, 1M and 0M part.
            t0 = np.all(len(norm_seg)==2)
            t1 = np.all((norms == 1.) | ((0 <= norms) & (norms <= a))) # make sure all values lie in this interval
            t2 = np.any((norms == 1.)) & np.any(((0 <= norms) & (norms <= a))) # make sure there is at least one value in each interval
            t3 = np.all(counts<=sM_lim)#make sure its not a 2M trace where both molecules bleached (almost) at the same time...
            ds = np.all(np.diff(norms)<0)
            return t0 & t1 & t2 & t3 & ds
        def test3(norms, counts):
            # trace with two segments, 2M and 0M part.
            t0 = np.all(len(norm_seg)==2)
            t1 = np.all((norms == 1.) | ((0 <= norms) & (norms <= a))) # make sure all values lie in this interval
            t2 = np.any((norms == 1.)) & np.any(((0 <= norms) & (norms <= a))) & np.any(((sM_lim <= counts))) # make sure there is at least one value in each interval
            t3 = np.all(counts<=lim_up)
            ds = np.all(np.diff(norms)<0)
            return t0 & t1 & t2 & t3 & ds
        
        if test0(norm_seg,mean_df.photons.to_numpy()):
            # 2M -> 1M -> 0M
            df.loc[df['segment_id'].isin(mean_df.loc[mean_df.photons_norm.between(1-a, 1.0)]['segment_id']),'state_id'] = 2
            df.loc[df['segment_id'].isin(mean_df.loc[mean_df.photons_norm.between(0.5-a, .5+a)]['segment_id']),'state_id'] = 1
            df.loc[df['segment_id'].isin(mean_df.loc[mean_df.photons_norm.between(0., a)]['segment_id']),'state_id'] = 0
            success=True
        elif test1(norm_seg,mean_df.photons.to_numpy()):
            # 2M -> 1M
            df.loc[df['segment_id'].isin(mean_df.loc[mean_df.photons_norm.between(1-a, 1.0)]['segment_id']),'state_id'] = 2
            df.loc[df['segment_id'].isin(mean_df.loc[mean_df.photons_norm.between(0.5-a, .5+a)]['segment_id']),'state_id'] = 1
            success=True
        elif test2(norm_seg,mean_df.photons.to_numpy()):
            # 1M -> 0M
            df.loc[df['segment_id'].isin(mean_df.loc[mean_df.photons_norm.between(1-a, 1.0)]['segment_id']),'state_id'] = 1
            df.loc[df['segment_id'].isin(mean_df.loc[mean_df.photons_norm.between(0., a)]['segment_id']),'state_id'] = 0
            success=True
        elif test3(norm_seg,mean_df.photons.to_numpy()):
            # 2M -> 0M
            df.loc[df['segment_id'].isin(mean_df.loc[mean_df.photons_norm.between(1-a, 1.0)]['segment_id']),'state_id'] = 2
            df.loc[df['segment_id'].isin(mean_df.loc[mean_df.photons_norm.between(0., a)]['segment_id']),'state_id'] = 0
            success=True
        df['success'] = success
        return df
    
    def bin_frame(self,df,bin_size=1):
        """
        Bin the input data frame.

        :param df: Input data frame.
        :type df: pandas.DataFrame
        :param bin_size: Size of bins, default is 1.
        :type bin_size: float, optional
        :return: Reduced and binned data frame.
        :rtype: pandas.DataFrame
        """
        # get positions and counts from segmented_df
        positions = df['pos'].to_numpy()
        if bin_size is None:
            reduced_df = df.copy()
            reduced_df['weights'] = np.ones(len(df))
        else:
            # bin the positions and counts
            bin_indices = ((positions - positions.min()) / bin_size).astype(int)
            df['bin_idx'] = bin_indices

            # now bin filtered dataframe
            groups = df.groupby('bin_idx',group_keys=False)
            reduced_df = groups.mean()
            reduced_df['weights'] = (groups['pos'].count()/groups['pos'].count().sum())**(1/5)
            reduced_df = reduced_df.reset_index().drop(columns=['bin_idx'])
        return reduced_df
    
    def remove_tuple_outliers(self,group,column_name='photons',threshold=1.5):
        """
        Remove outliers within each bin using the IQR method.

        :param group: Grouped data frame.
        :type group: pandas.DataFrame
        :param column_name: Name of the column to filter, default is 'photons'.
        :type column_name: str, optional
        :param threshold: IQR threshold for outlier removal, default is 1.5.
        :type threshold: float, optional
        :return: Data frame with outliers removed.
        :rtype: pandas.DataFrame
        """
        tuple_counts = group['tuple'].value_counts()
    
        # Extract tuples with exactly three points
        valid_tuples = tuple_counts[tuple_counts == 6].index#3 in each axis...
        
        # Filter the DataFrame to retain only valid tuples
        group = group[group['tuple'].isin(valid_tuples)]

        df = group.groupby(['tuple','experiment_id'], group_keys=False).mean().reset_index()
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        valid_tuples = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)].tuple.to_numpy()
        # remove fractional tuples, i.e. where one point is missing.
        return group.loc[group['tuple'].isin(valid_tuples)]
    
    def _save_frame(self,filtered_df,segmented_df,path,base_in,base_out):
        """
        Save filtered and segmented data frames.

        :param filtered_df: Filtered data frame.
        :type filtered_df: pandas.DataFrame
        :param segmented_df: Segmented data frame.
        :type segmented_df: pandas.DataFrame
        :param path: File path.
        :type path: str
        :param base: Base input directory.
        :type base: str
        :param base_out: Base output directory.
        :type base_out: str
        """
        # label and save as pkl
        curr_dir = os.path.dirname(path)
        diff_path = os.path.relpath(curr_dir, base_in)
        new_dir = os.path.join(base_out, diff_path) + '/'
        fname = ''#optional:stamp again....
        
        os.makedirs(new_dir + '/', exist_ok=True) # create a directory for this file
        filtered_df.to_pickle(new_dir + fname + 'filtered.pkl')
        segmented_df.to_pickle(new_dir + fname + 'segmented.pkl')

        #artefacts = Artefacts()
        #figures, names = CPSegmentation_Figures().fig_diagnostics(segmented_df, filtered_df)
        #artefacts.add_figures(figures, names)
        #artefacts.save_figures(out_dir = new_dir)
        pass

    def _segment_frame(self,df):
        """
        Segment the input data frame.

        :param df: Input data frame.
        :type df: pandas.DataFrame
        :return: Segmented data frame.
        :rtype: pandas.DataFrame
        """
        # find 2M segment
        new_df = df.copy()
        new_df.loc[:,'segment_id'] = np.nan
        new_df = new_df.groupby(['trace'],group_keys=False).apply(self._find_bkps)
        segment_df = new_df.loc[np.isnan(new_df.loc[:,'segment_id'])==False,:].copy()
        return segment_df
    
    def _find_bkps(self,df):
        """
        Find breakpoints in the input data frame.

        :param df: Input data frame.
        :type df: pandas.DataFrame
        :return: Data frame with segment IDs.
        :rtype: pandas.DataFrame
        """
        segmented_df = df.copy()
        # change point detection on mean
        grouped_df = df.groupby(['tuple','experiment_id'], group_keys=False).mean().reset_index()
        signal = np.asarray(grouped_df.photons.to_numpy(),dtype='int')
        #signal = df['photons'].to_numpy()
        algo_c = rpt.KernelCPD(kernel="rbf", min_size=100,jump=10).fit(
            signal
        )  # written in C
        penalty_value = 10  # beta
        bkps = [0]
        try:
            bkps += algo_c.predict(pen=penalty_value)
        except:
            bkps += [len(signal)]
        bkps = [p+grouped_df.tuple.min() for p in bkps]#adjust for minimum tuple

        #adjust breakpoints to cut away some tuples before and after
        window_size = 25#number of tuples to discard
        start_bkps, stop_bkps = [], []
        for i in range(len(bkps)-1):
            p1 = bkps[i]
            p2 = bkps[i+1]
            start_bkps.append(min(p1+2*window_size,p2))
            stop_bkps.append(max(p1,p2-2*window_size))

        # assign ids to segments
        lengths = [stop-start for start,stop in zip(start_bkps,stop_bkps)]

        segmented_df.loc[:,'segment_id'] = np.nan
        for i,length in enumerate(lengths):
            id = i
            indices = segmented_df[segmented_df['tuple'].between(start_bkps[i], stop_bkps[i])].index
            if length<7: # require at least 7 triples
                id = np.nan # check length of segments and sort out too short segments:
            segmented_df.loc[segmented_df.index.isin(indices),'segment_id'] = id
        segmented_df = segmented_df.loc[~np.isnan(segmented_df.segment_id)].copy()
        #for state in np.unique(states):
        #    tuples = np.asarray(grouped_df.loc[np.where(new_states == state)[0]].tuple.to_numpy(),dtype='int')
        #    segmented_df.loc[segmented_df['tuple'].isin(tuples), 'state_id'] = state
        return segmented_df
    
    def _find_state_clusters(self,df):
        """
        Find state clusters in the input data frame.

        :param df: Input data frame.
        :type df: pandas.DataFrame
        :return: Clustered data frame.
        :rtype: pandas.DataFrame
        """
        df_cutoff = df.loc[(df['photons']>4) & (df['photons']<60)].copy()
        grouped = df_cutoff.groupby(['gt', 'batch'])

        # Define the DBSCAN parameters (you may need to tune these)
        eps = .5  # The maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples = 10  # The number of samples in a neighborhood for a point to be considered a core point

        # Define a function to apply DBSCAN within each group
        def apply_dbscan(group):
            X = group[['photons']]
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            group['cluster_label'] = dbscan.fit_predict(X)
            return group
        
        def applyKMeans(group):
            try:
                try:
                    init = group.groupby('state_id')['photons'].mean().sort_values().values
                    if init.size!=2:
                        init = 'k-means++'
                    else:
                        init = init.reshape(-1,1)
                except:
                    init = 'k-means++'
                X = group[['photons']]
                clf = KMeansConstrained(
                    n_clusters=2
                    ,size_min=min_samples
                    ,random_state=0
                    ,init=init
                    ,n_init=10
                    )
                group['cluster_label'] = clf.fit_predict(X)
            except:
                group['cluster_label'] = -1
            return group

        clustered_df = grouped.apply(applyKMeans).reset_index(drop=True)
        df_bgr = df.loc[(df['photons']<2)].copy()
        df_bgr['cluster_label'] = 2
        clustered_df = pd.concat([df_bgr,clustered_df]).reset_index(drop=True)
        # Apply DBSCAN clustering to each group
        #clustered_df = grouped.apply(apply_dbscan).reset_index(drop=True)
        return clustered_df
    
    def _map_clusters_on_states(self,df):
        """
        Map clusters on states in the input data frame.

        :param df: Input data frame.
        :type df: pandas.DataFrame
        :return: Data frame with mapped clusters.
        :rtype: pandas.DataFrame
        """
        df.loc[:,'state_id'] = np.nan
        
        raw_df = df.loc[(df['cluster_label']!=-1)].copy()
        cluster_means = raw_df.groupby(['gt','batch','cluster_label'])['photons'].mean().reset_index().copy()
        assessed_batches = cluster_means.groupby(['gt','batch']).apply(self._check_cluster_number)

        try:
            assert np.any(assessed_batches.valid), 'No valid batches found!'
        except AssertionError as e:
             print(f"Assertion failed: {e}")
        
        merged = raw_df.merge(assessed_batches,on=['gt','batch'],how='left')
        merged = merged.loc[merged['valid']==True]
        valid_cluster_means = merged.groupby(['gt','batch','cluster_label'])['photons'].mean().reset_index().copy()
        labels = valid_cluster_means.groupby(['gt','batch']).apply(self._label_cluster)
        cluster_state_map = labels.set_index(['gt','batch','cluster_label'])['state_id'].to_dict()
        # Create a tuple of the multi-index columns
        merged['multi_index'] = list(zip(merged['gt'], merged['batch'], merged['cluster_label']))
        # Map the multi-index to 'state_id' using cluster_state_map
        merged['state_id'] = merged['multi_index'].map(cluster_state_map)
        # Drop the temporary 'multi_index' column if needed
        labeled_df = merged.drop(['multi_index','cluster_label'], axis=1).copy()
        return labeled_df.loc[~np.isnan(labeled_df.state_id)]

    def _check_cluster_number(self,df):
        """
        Check the number of clusters in the input data frame.

        :param df: Input data frame.
        :type df: pandas.DataFrame
        :return: Series with validity information.
        :rtype: pandas.Series
        """
        if df['cluster_label'].nunique() == 3:
            return pd.Series({'valid':True})
        else:
            return pd.Series({'valid':False})
    
    def _label_cluster(self,df):
        """
        Label clusters in the input data frame.

        :param df: Input data frame.
        :type df: pandas.DataFrame
        :return: Data frame with labeled clusters.
        :rtype: pandas.DataFrame
        """
        df.loc[df['photons']<=2,'state_id'] = 0
        df.loc[df['photons']==df['photons'].max(),'state_id'] = 2
        df.loc[(df['photons']<=.7 * df['photons'].max()) & (df['photons']>=.3 * df['photons'].max()),'state_id'] = 1
        return df
    
    
class MinfluxProcessorFacade:
    """
    Facade class for processing Minflux data using the MinfluxProcessor.

    This class provides a high-level interface for processing multiple Minflux data files.

    :ivar global_filter_results: Global filter results data frame.
    :vartype global_filter_results: pandas.DataFrame
    :ivar base: Base input directory.
    :vartype base: str
    :ivar base_out: Base output directory.
    :vartype base_out: str
    :ivar bootstrap_dicts: List of bootstrap dictionaries.
    :vartype bootstrap_dicts: list
    """
        
    def __init__(self):
        pass

    def process_minflux(self, input, output,max_files=2,key='',bootstrap_dicts=[]):
        """
        Process multiple Minflux data files.

        This method processes multiple Minflux data files using parallel processing.
        It relies on the MinfluxProcessor class for individual file processing.

        :param input: Input directory containing Minflux data files.
        :type input: str
        :param output: Output directory for processed files.
        :type output: str
        :param max_files: Maximum number of files to process, default is 2.
        :type max_files: int, optional
        :param key: A key to filter files during processing, e.g. to select a certain batch. Default is an empty string.
        :type key: str, optional
        :param bootstrap_dicts: List of bootstrap dictionaries, default is an empty list.
        :type bootstrap_dicts: list, optional
        """
        global_filter_results = ut.BaseFunc().find_files(input, lambda file: (ut.BaseFunc().match_pattern(file, 'filter-results.pkl', match='partial')), max_files = 1)[0]
        global_filter_results = pd.read_pickle(global_filter_results)
        global_filter_results = global_filter_results.reset_index(drop=True)
        processable_files = ut.BaseFunc().find_files(input, lambda file: (('filtered.pkl' in file) and (key in file)), max_files=max_files)
        output = output+'processed/'
        os.makedirs(output, exist_ok=True)
        task = partial(MinfluxProcessor().process_single_file,global_filter_results=global_filter_results,base_in=input,base_out=output,bootstrap_dicts=bootstrap_dicts)
        results = []
        #for file in processable_files:
        #    results.append(task(file))
        with multiprocessing.Pool() as pool:
            results = pool.map(task, processable_files)
        results_df = pd.concat(results)
        results_df.to_pickle(output + 'fitting-results.pkl')
        pass

class MinfluxProcessor:
    """
    Class for processing Minflux data.

    This class provides methods for processing individual Minflux data files and performing local analysis.

    :ivar bootstrap_dicts: List of bootstrap dictionaries.
    :vartype bootstrap_dicts: list
    """

    def __init__(self):
        pass

    def process_single_file(self,path,global_filter_results=None,base_in=None,base_out=None,bootstrap_dicts=[]):
        """
        Process a single Minflux data file.

        This method processes a single Minflux data file, performs global filtering based on provided results,
        and conducts local analysis using bootstrap dictionaries.

        :param path: Path to the Minflux data file.
        :type path: str
        :param global_filter_results: Global filter results data frame.
        :type global_filter_results: pandas.DataFrame, optional
        :param base: Base input directory.
        :type base: str, optional
        :param base_out: Base output directory.
        :type base_out: str, optional
        :param bootstrap_dicts: List of bootstrap dictionaries.
        :type bootstrap_dicts: list, optional
        :return: Local analysis results data frame.
        :rtype: pandas.DataFrame
        """
        # TODO: sanity check of bootstrap_dictionary
        try:
            try:
                assert not global_filter_results is None, 'Global filter results are None!'
            except AssertionError as e:
                raise Exception(f'Assertion failed: {e}')

            df = pd.read_pickle(path)
            curr_id = df.loc[:,'experiment_id'].iloc[0]
            
            valid_ids = list(set(global_filter_results['experiment_id'].values))
            if ~np.any([curr_id==idd for idd in valid_ids]):
                return pd.DataFrame({})
            else:
                filtered_results = global_filter_results.loc[global_filter_results['experiment_id'].apply(lambda x: x==curr_id)]
                df = df.merge(filtered_results.loc[:,['segment_id','state_id']],how='left',on=['segment_id'],suffixes=('', '_to_merge'))
                df['state_id'] = df['state_id'].fillna(df['state_id_to_merge'])
                df.drop(columns={'state_id_to_merge'},inplace=True)

                curr_dir = os.path.dirname(path)
                diff_path = os.path.relpath(curr_dir, base_in)
                new_dir = os.path.join(base_out, diff_path) + '/'
                
                _, fname = os.path.split(path)            
                os.makedirs(new_dir + '/', exist_ok=True)

                #--------------------
                #local analysis
                local_results = pd.DataFrame({})
                for i,i_dict in enumerate(bootstrap_dicts):
                    try:
                        i_dict = i_dict|dict(output=new_dir)
                        #MinfluxAnalysis().assign_chunk_id(df, **i_dict)
                        chunked_df = df.groupby(['trace','segment_id','state_id'],group_keys=False).apply(MinfluxAnalysis().assign_chunk_id, **i_dict).copy()
                        new_rows = chunked_df.groupby(['experiment_id','trace','segment_id','state_id','axis','chunk_id','bin_id'],group_keys=False).apply(MinfluxAnalysis().fit_chunk,**i_dict)
                        new_rows.loc[:,'gt'] = df['gt'].mean()
                        new_rows.loc[:,'batch'] = df['batch'].mean()
                        new_rows.loc[:,'file'] = path
                        new_rows = new_rows.assign(**i_dict)
                        local_results = pd.concat([local_results, new_rows], ignore_index=False)
                    except:
                        continue
                fname = 'local-results'
                local_results.reset_index(inplace=True)
                local_results.to_pickle(new_dir + '/' + ut.Labeler().stamp(fname)[0] + '.pkl')
                return local_results
        except:
            return pd.DataFrame({})

def _load_processed_file(file):
    """
    Load a processed file.

    This function loads a processed file and categorizes it as 'global' or 'local'.

    :param file: Path to the processed file.
    :type file: str
    :return: Tuple containing the category ('global' or 'local') and the loaded DataFrame.
    :rtype: tuple
    """
    try:
        tmp_df = pd.read_pickle(file)
        if ut.BaseFunc().match_pattern(file, 'global', match='partial'):
            return ('global', tmp_df)
        elif ut.BaseFunc().match_pattern(file, 'local', match='partial'):
            return ('local', tmp_df)
    except:
        pass

def process_group(group):
    """
    Process a group.

    This function processes a group by calculating the Euclidean norm of the 'd' column within the group.

    :param group: DataFrame group to be processed.
    :type group: pandas.DataFrame
    :return: Processed DataFrame group.
    :rtype: pandas.DataFrame
    """
    try:
        group['d_norm'] = group.groupby(['trace', 'chunk_size','bin_size', 'segment_id', 'chunk_id'])['d'].transform(lambda x: np.linalg.norm(x))
    except:
        pass
    return group
        
class MinfluxPostProcessing:
    def __init__(self):
        pass

    def post_process_minflux(self,input_dir,output_dir,visualize=True):
        """
        Perform post-processing on Minflux data.

        This method performs post-processing on Minflux data, including filtering, calibration, and visualization.

        :param input_dir: Input directory containing processed Minflux data.
        :type input_dir: str
        :param output_dir: Output directory for post-processed data and visualizations.
        :type output_dir: str
        :param visualize: Flag to enable or disable visualization.
        :type visualize: bool, optional
        """
        os.makedirs(output_dir+'post_processed/', exist_ok=True)
        results_file = ut.BaseFunc().find_files(os.path.join(input_dir,'processed/'), lambda file: ut.BaseFunc().match_pattern(file, '.pkl', match='partial') & ('fitting-results' in file), max_files = 1)[0]
        results_df = pd.read_pickle(results_file)
        
        artefacts = Artefacts()
        try:
            #----------------------
            # initial setp of dataframe
            results = results_df.reset_index()                
            results['gt'] = results['gt'].round(0).astype(int)
            results['batch'] = results['batch'].round(0).astype(int)
            ##results['target'] = results['target'].round(0).astype(int)
            ##results['method'] = 'minflux'
            ##results['calibration'] = 'global'
            results['label'] = (results['gt'].astype(str) + '.' + results['batch'].astype(str)).astype(float)
            available_targets = results['chunk_size'].unique()
            print(f"Available targets: {available_targets}")
            original_data = results.copy()
            collected_results = pd.DataFrame({})
            for target in available_targets:
                results = original_data.copy()
                if target is None:
                    label = 'None'
                    # Replace invalid values with the placeholder in key_columns
                    key_columns = ['chunk_size','bin_size']
                    placeholder = 'InvalidPlaceholder'
                    for col in key_columns:
                        results[col] = results[col].replace({None: placeholder})
                else:
                    label = int(target)
                    results = results.loc[(results['chunk_size']==target)]#
                print(f"Chosen target: {target}")
                os.makedirs(output_dir+f'post_processed/{label}/', exist_ok=True)

                

                #----------------------------
                # filter out failed fits (chunk-wise) via residuals
                key_columns = ['file', 'trace', 'chunk_size', 'bin_size', 'segment_id', 'chunk_id']
                mask = ((results['chi2'] > np.inf*.55) | (results['chi2'] < 0.)) & (results['state_id']!=0)# Create a boolean mask to identify rows to remove
                combinations_to_remove = results.loc[mask, key_columns].drop_duplicates()# Get the key columns as tuples for combinations to remove
                filtered_results = results.merge(combinations_to_remove, on=key_columns, how='left', indicator=True)# Filter the DataFrame using merge
                filtered_results = filtered_results[filtered_results['_merge'] == 'left_only'].drop(columns='_merge')

                #-----------------------
                #get average background
                average_background = filtered_results.loc[filtered_results['state_id']==0].groupby(['gt', 'batch', 'axis'])['N_avg'].median().reset_index()
                results = filtered_results.merge(average_background, on=['gt', 'batch', 'axis'], how='left', suffixes=('', '_glob')).copy()
                local_background = filtered_results.loc[filtered_results['state_id']==0].groupby(['file','axis'])['N_avg'].median().reset_index()
                results = results.merge(local_background, on=['file', 'axis'], how='left', suffixes=('', '_loc'))
                results['N0'] = results['N_avg_loc']
                results['N0'] = results['N0'].fillna(results['N_avg_glob'])
                results['N0'] = results['N0'].fillna(1.)#if no background is available, set to 1 count (realistic...)
                results.drop(columns={'N_avg_glob','N_avg_loc'},inplace=True)
                
                vis_df = results.copy()              
                vis_df['v'] = vis_df['a1']/(vis_df['a0']-vis_df['N0'])# get background corrected visibilities
                noise = np.random.normal(loc=0.0, scale=0.02, size=len(vis_df.loc[vis_df['state_id']==1,'v']))  # Test for misscalibration
                #vis_df.loc[vis_df['state_id']==1,'v'] += noise

                # determine local and global v0 for calibration
                filtered_vis_df = vis_df.loc[vis_df['state_id']==1].copy() # take only 1M segments
                if not filtered_vis_df.empty:
                    try:
                        # calibration via visibility of single molecule trace
                        local_average_visibility = filtered_vis_df.groupby(['file', 'axis']).apply(lambda x: np.nanmedian(x['v'])).reset_index().rename(columns={0: 'v0'})
                        global_average_visibility = filtered_vis_df.groupby(['gt', 'batch', 'axis']).apply(lambda x: np.nanmedian(x['v'])).reset_index().rename(columns={0: 'v0'})
                        vis_df = vis_df.merge(local_average_visibility, on=['file', 'axis'], how='left').copy()
                        vis_df = vis_df.merge(global_average_visibility, on=['gt', 'batch', 'axis'], how='left', suffixes=('', '_glob')).copy()
                        
                        vis_df['v0'].fillna(vis_df['v0_glob'], inplace=True)#
                        #vis_df['v0'].fillna(.98, inplace=True)
                        vis_df.drop(columns={'v0_glob'},inplace=True)
                    except:
                        print('Local calibration of initial visibility failed. Proceed with standard calibration.')
                else:
                    #zz
                    vis_df['v0'] = 1.0
                vis_df.loc[vis_df['state_id']==0,'v'] = 0 #avoid infinities from arccos

                #-----------------------------                
                # filter out rulers with bad calibration/bad quality of minimum, if applicable
                key_columns = ['file','trace']
                mask = (vis_df['v0']<0.9)&(vis_df['state_id']==1)# Create a boolean mask to identify rows to remove
                combinations_to_remove = vis_df.loc[mask, key_columns].drop_duplicates()# Get the key columns as tuples for combinations to remove
                filtered_vis_df = vis_df.merge(combinations_to_remove, on=key_columns, how='left', indicator=True)# Filter the DataFrame using merge
                vis_df = filtered_vis_df[filtered_vis_df['_merge'] == 'left_only'].drop(columns='_merge').copy()

                """#---------------------------------------------
                # determine v1 as 2M visibilities
                filtered_vis_df = vis_df.loc[vis_df['state_id']==2.].copy() # take only 2M segments
                
                # filter out unphysical visibilities
                key_columns = ['batch','file','trace','chunk_size','bin_size', 'segment_id', 'chunk_id']
                mask = (filtered_vis_df['v']<0.5)|(filtered_vis_df['v']>1.)# Create a boolean mask to identify rows to remove
                combinations_to_remove = filtered_vis_df.loc[mask, key_columns].drop_duplicates()# Get the key columns as tuples for combinations to remove
                filtered_vis_df = filtered_vis_df.merge(combinations_to_remove, on=key_columns, how='left', indicator=True)# Filter the DataFrame using merge
                filtered_vis_df = filtered_vis_df[filtered_vis_df['_merge'] == 'left_only'].drop(columns='_merge')"""

                # calculate median 2M visibility per segment and axis
                average_visibility = vis_df.groupby(['file','axis'])['v'].median().reset_index().rename(columns={'v':'v1'})#.apply(lambda x: ut.BaseFunc().weighted_median(x['v'], 1/x['chi2'])).reset_index().rename(columns={0: 'v1'})
                new_filtered_vis_df = vis_df.merge(average_visibility, on=['file', 'axis'], how='left',indicator=True)
                vis_med_df = new_filtered_vis_df[new_filtered_vis_df['_merge'] == 'both'].drop(columns='_merge').copy()

                # include some more visibility values if they are within a certain tolerance
                tolerance = 5E-3
                mask =  (vis_df['state_id']!=0) & (vis_df['v'] > vis_df['v0']) & ((vis_df['v']-vis_df['v0'])<tolerance)
                vis_df.loc[mask, 'v0'] = vis_df.loc[mask, 'v']


                #----------------------------
                # calculate distances
                dist_df = vis_df.copy()
                dist_med_df = vis_med_df.copy()
                arg = 2 * (dist_df['v'] / dist_df['v0']) ** 2 - 1
                dist_df['d'] = LAMBDA/(4*np.pi) * np.arccos(arg.astype('float'))
                arg = 2 * (dist_med_df['v1'] / dist_med_df['v0']) ** 2 - 1
                dist_med_df['d'] = LAMBDA/(4*np.pi) * np.arccos(arg.astype('float'))

                #---------------------------
                # take the norm of calculated distances
                target_df = dist_df.copy()
                target_df.loc[target_df['state_id']==1,'d'] = 0
                
                
                file_groups = target_df.loc[target_df['state_id']==2].groupby(['file'],group_keys=False)#target_df['state_id']!=0 for origami
                list_of_groups = [group for name, group in file_groups]
                
                task = partial(process_group)
                #for group in list_of_groups:
                #    task(group)
                num_cores = multiprocessing.cpu_count()
                with multiprocessing.Pool(processes=num_cores) as pool:
                    processed_groups = pool.map(task, list_of_groups)
                dist_df = pd.concat(processed_groups, ignore_index=True)

                # ------------------------------------
                # filter out failed distance estimates
                key_columns = ['file', 'trace', 'chunk_size','bin_size', 'segment_id', 'chunk_id']
                mask = (np.isnan(dist_df['d_norm'])) | (dist_df['d_norm'].isna()) | (dist_df['d_norm']>=100) | ((dist_df['state_id']==2) & (dist_df['d_norm']<=0.05))# Create a boolean mask to identify rows to remove
                combinations_to_remove = dist_df.loc[mask, key_columns].drop_duplicates()# Get the key columns as tuples for combinations to remove
                filtered_dist_df = dist_df.merge(combinations_to_remove, on=key_columns, how='left', indicator=True)# Filter the DataFrame using merge
                norm_df = filtered_dist_df[filtered_dist_df['_merge'] == 'left_only'].drop(columns='_merge')

                # take median if more than one estimate has been obtained per ruler
                vis_df_median = vis_df.groupby(['gt','file','label','trace','chunk_size','bin_size','segment_id','state_id','axis','estimator'],group_keys=False)[['v','v0','N_avg']].median().reset_index()
                vis_df_median['gt'] = vis_df_median['gt'].round(0).astype(int)
                norm_df_median = norm_df.groupby(['gt','file','label','trace','chunk_size','bin_size','segment_id','state_id','axis','estimator'],group_keys=False)[['d','d_norm']].median().reset_index()
                norm_df_median['gt'] = norm_df_median['gt'].round(0).astype(int)

                #--------------------------------
                # do the visualization
                if visualize:
                    # global diagnostics (on median data)
                    figures, names = Minflux_Figures().fig_visibility(vis_df_median)
                    artefacts.add_figures(figures, names)
                    figures, names = Minflux_Figures().fig_count_histogram(vis_df_median)
                    artefacts.add_figures(figures, names)
                    figures, names = Minflux_Figures().fig_distance_histogram(norm_df_median.loc[norm_df_median['state_id']==2])
                    artefacts.add_figures(figures, names)
                    artefacts.save_figures(out_dir = output_dir+f'post_processed/{label}/')

                    # local diagnostics
                    figures, names = Minflux_Figures().fig_distance_wrt_time(norm_df.loc[norm_df['state_id']==2])
                    artefacts.add_figures(figures, names)
                    #figures, names = Minflux_Figures().fig_tracking(norm_df)
                    #artefacts.add_figures(figures, names)
                    figures, names = Minflux_Figures().fig_distance_histogram(norm_df.loc[norm_df['state_id']==2])
                    artefacts.add_figures(figures, names)
                    figures, names = Minflux_Figures().fig_distance_precision(norm_df.loc[norm_df['state_id']==2])
                    artefacts.add_figures(figures, names)
                    artefacts.save_figures(out_dir = output_dir+f'post_processed/{label}/')

                fname = 'postprocessing-results'
                norm_df.to_pickle(output_dir+f'post_processed/{label}/' + ut.Labeler().stamp(fname)[0] + '.pkl')
                collected_results = pd.concat([collected_results,norm_df])
            fname = 'all-postprocessing-results'
            collected_results.to_pickle(output_dir+f'post_processed/' + ut.Labeler().stamp(fname)[0] + '.pkl')
        except:
            pass
        pass
        
    def _load_processed_files(self, processable_files):
        """
        Load processed files.

        This method loads processed files in parallel and categorizes them as 'global' or 'local'.

        :param processable_files: List of processed file paths.
        :type processable_files: list
        :return: Tuple containing global and local results DataFrames.
        :rtype: tuple
        """        
        # Create a multiprocessing pool with the same number of processes as the available CPU cores
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        # Map the process_file function to each file in processable_files
        results = pool.map(_load_processed_file, processable_files)
        # Close the pool to no longer accept new jobs
        pool.close()
        # Wait for all processes to finish
        pool.join()
        # Separate the results into global and local results
        global_results = pd.concat([result[1] for result in results if result[0] == 'global'])
        local_results = pd.concat([result[1] for result in results if result[0] == 'local'])
        
        return global_results, local_results
    
def process_batch(base_in, parse=True, parser_type='otto', filter=True, postfilter=True,process=True, processing_key='', bootstrap_dicts={}, postprocess=True, max_files=np.inf, visualize=True):
    """
    Process a batch of Minflux data.

    This function orchestrates the entire process of parsing, filtering, processing, and post-processing a batch of Minflux data.

    :param base: Base input directory containing Minflux data.
    :type base: str
    :param parse: Flag to enable or disable parsing.
    :type parse: bool, optional
    :param parser_type: Type of Minflux parser to use.
    :type parser_type: str, optional
    :param filter: Flag to enable or disable filtering.
    :type filter: bool, optional
    :param postfilter: Flag to enable or disable post-filtering.
    :type postfilter: bool, optional
    :param process: Flag to enable or disable processing.
    :type process: bool, optional
    :param bootstrap_dicts: Dictionary or list of dictionaries containing bootstrap parameters.
    :type bootstrap_dicts: dict or list, optional
    :param postprocess: Flag to enable or disable post-processing.
    :type postprocess: bool, optional
    :param max_files: Maximum number of files to process.
    :type max_files: int, optional
    :param visualize: Flag to enable or disable visualization during post-processing.
    :type visualize: bool, optional
    :return: Path to the output directory containing post-processed data.
    :rtype: str
    """
    dir_str, todays_dir = ut.Labeler().stamp('MINFLUX-session')
    new_base_dir = todays_dir + dir_str + '/'
    if not os.path.exists(new_base_dir):
        os.makedirs(new_base_dir, exist_ok=True)
    
    if parse:
        parser = MF_ParserFacade(type=parser_type)
        parser.parse_directory(base_in,new_base_dir,max_files=max_files)

    if filter:
        if parse:
            input = new_base_dir
        else:
            input = base_in
            #TODO: check if folder contains parsed data
        f=MinfluxFilterFacade()
        f.filter_directory(input+'parsed/',new_base_dir,max_files=max_files)
    
    if postfilter:
        if filter:
            input = new_base_dir
        else:
            input = base_in
            # Remove the destination directory if it exists
            files = ut.BaseFunc().find_files(input, lambda file: 'filter-results.pkl' in file, max_files=np.inf, max_depth=0)
            files += ut.BaseFunc().find_files(input, lambda file: file.endswith('.pdf'), max_files=np.inf,max_depth=0)
            files += ut.BaseFunc().find_files(input, lambda file: file.endswith('.png'), max_files=np.inf,max_depth=0)
            for file in files:
                if os.path.isfile(file):
                    os.remove(file)
        f = MinfluxFilterFacade()
        f.post_filtering(input+'filtered/', input+'filtered/')
    
    if process:
        if filter:
            input = new_base_dir+'filtered/'
        else:
            input = base_in+'filtered/'
            #TODO: check for filtered data
        processor = MinfluxProcessorFacade()
        processor.process_minflux(input,new_base_dir,max_files=max_files,key=processing_key,bootstrap_dicts=bootstrap_dicts)

    if postprocess:
        if process:
            input = new_base_dir
        else:
            input = base_in
            #TODO: check if processed data is available
        postprocessor = MinfluxPostProcessing()
        postprocessor.post_process_minflux(input,new_base_dir,visualize=visualize)
    return new_base_dir