import os, sys
script_name = os.path.basename(__file__)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

import lib.utilities as ut
from lib.plotting.artefacts import Figures
from lib.plotting.style import Style

def plot_corr(dataframe, key):
    df = dataframe
    # Create a new dataframe with the unique labels as columns
    new_df = df.groupby(['method','ground_truth'])[key].apply(list).reset_index(name='aggregated_values')
    new_df.set_index(['method','ground_truth'], inplace=True)
    pivot_df = new_df['aggregated_values'].apply(pd.Series).transpose()
    # Compute the correlation matrix of the 'aggregated_values' column
    corr = pivot_df.corr()

    fig, ax = Figures()._get_std_layout_single('Cross-correlations between methods')
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Create a heatmap of the correlation matrix
    sns.heatmap(corr,
        annot=False,
        mask=mask,
        cmap=sns.diverging_palette(230, 20, as_cmap=True),
        vmin=-1.0, vmax=1.0,
        square=True, ax=ax['1'])
    return fig

def make_figure(plot_style='nature',color_scheme='default',show=True):
    s = Style(style=plot_style,color_scheme=color_scheme)

    base_dir = os.path.split(__file__)[0]
    
    file = ut.BaseFunc().find_files(base_dir, lambda file: ut.BaseFunc().match_pattern(file, 'line-scan-data', match='partial'),max_files=1)[0] # load pkl
    df = pd.read_pickle(file)
    df['d_norm'] = df['d_norm'].apply(lambda x: np.nanmean(x))
    df['N_fit'] = df['N_fit'].apply(lambda x: np.nanmean(x))
    df['error'] = df['d_norm']-df['ground_truth']
    df['NALM_error'] = df['d_norm_NALM']-df['ground_truth']

    df.loc[:,'d_norm'] = df['d_norm'].apply(lambda x: np.nanmean(x))
    sub_df = df.loc[(df['method']=='MIN-POLY')&(df['d_norm']<100)&(df['d_norm_NALM']<100)].copy()

    unique_sizes = sub_df['ground_truth'].unique()
    fig = plot_corr(df, 'd_norm')
    
    if show:
        plt.show()
    return fig

if __name__=='__main__':
    fig = make_figure(show=True)
    #Figures().save_fig(fig, 'cross-correlations-extended-data-figure',meta={'generating script': script_name})