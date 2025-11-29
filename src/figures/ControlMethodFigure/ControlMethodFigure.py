"""

Code to generate the figure illustrating the control method of obtaining
inter-fluorophore distances from bleaching steps.

copyright: @Thomas Arne Hensel, 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from lib.plotting.style import Style
from lib.plotting.artefacts import NALM_Figures
from lib.data_handling.data_converter import Constructor, Converter
import lib.utilities as ut

def make_figure(plot_style='nature',color_scheme='default',show=True):
    s = Style(style=plot_style,color_scheme=color_scheme)
    base_dir = os.path.abspath(os.path.split(__file__)[0])
    # load result dictionary 
    file = ut.BaseFunc().find_files(base_dir, lambda file: file.endswith('.json') and 'processed' in file, max_files=1)[0]
    result_dict = ut.Importer().load_json(file)

    file = ut.BaseFunc().find_files(base_dir, lambda file: file.endswith('.json') and 'fitting' in file, max_files=1)[0]
    result_dict = result_dict | ut.Importer().load_json(file)

    # load imarray
    file = ut.BaseFunc().find_files(base_dir, lambda file: file.endswith('.tif') and 'source' in file, max_files=1)[0]
    curr_constructor = Constructor(file,collect_artefacts=False)
    
    experiments = curr_constructor.get_experiments(batchsize=1, agnostic=True, fast_return=False)
    glob_imarray = []
    for exp in experiments:
        if exp.setup.sample.config.n_molecules>0:
            imarray = Converter().reshape_record(exp,export=False,show=False).photons
            x_imarray, y_imarray = Converter().partition_array(imarray, curr_constructor.ext.block_size)
            glob_imarray.append([x_imarray,y_imarray])
    imarray = np.concatenate(glob_imarray,axis=1)
    
    min_pos_1M = Converter().partition_array(np.asarray(result_dict['min_idx_1M']), curr_constructor.ext.block_size)
    min_pos_2M = Converter().partition_array(np.asarray(result_dict['min_idx_2M']), curr_constructor.ext.block_size)
    result_dict['min_idx'] = np.concatenate([min_pos_2M,min_pos_1M],axis=1)
    result_dict['com_2M'] = np.median(min_pos_2M,axis=1)
    result_dict['com_1M'] = np.median(min_pos_1M,axis=1)
    curv_1M = [sol[1] for sol in result_dict['raw_sol_1M']]
    curv_2M = [sol[1] for sol in result_dict['raw_sol_2M']]
    curv_1M = Converter().partition_array(np.asarray(curv_1M), curr_constructor.ext.block_size)
    curv_2M = Converter().partition_array(np.asarray(curv_2M), curr_constructor.ext.block_size)
    result_dict['curvature'] = np.concatenate([curv_2M,curv_1M],axis=1)
    result_dict['median_curvature'] = [np.median(curv_2M,axis=1),np.median(curv_1M,axis=1)]

    min_depth_1M = [sol[0]-sol[1] for sol in result_dict['raw_sol_1M']]
    min_depth_2M = [sol[0]-sol[1] for sol in result_dict['raw_sol_2M']]
    min_depth_1M = Converter().partition_array(np.asarray(min_depth_1M), curr_constructor.ext.block_size)
    min_depth_2M = Converter().partition_array(np.asarray(min_depth_2M), curr_constructor.ext.block_size)
    result_dict['min_depth'] = np.concatenate([min_depth_2M,min_depth_1M],axis=1)

    fig, axs = NALM_Figures().fig_NALM_analysis(result_dict, imarray)

    if show:
        plt.show()
    return fig


if __name__=='__main__':
    fig = make_figure(show=True)
    #path_to_figure = Figures().save_fig(fig, 'ControlMethod')