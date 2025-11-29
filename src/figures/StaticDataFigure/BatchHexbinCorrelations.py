"""
Experimental line-scan data.

copyright: @Thomas Hensel, 2023
"""

import os
script_name = os.path.basename(__file__)
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

import lib.utilities as ut
from lib.plotting.style import Style
from lib.plotting.artefacts import Figures

mpl.use('TkAgg')

def make_figure(plot_style='nature',color_scheme='default',show=True):
    s = Style(style=plot_style,color_scheme=color_scheme)
    base_dir = os.path.split(__file__)[0]
    
    file = ut.BaseFunc().find_files(base_dir, lambda file: ut.BaseFunc().match_pattern(file, 'line-scan-data', match='partial'),max_files=1)[0] # load pkl
    df = pd.read_pickle(file)
    df['d_norm'] = df['d_norm'].apply(lambda x: np.nanmean(x))
    df['N_fit'] = df['N_fit'].apply(lambda x: np.nanmean(x))
    df['error'] = df['d_norm']-df['ground_truth']
    df['NALM_error'] = df['d_norm_NALM']-df['ground_truth']

    def hexbin(x, y, color, **kwargs):
        cmap = sns.light_palette(color, as_cmap=True)
        plt.hexbin(x, y, gridsize=40, cmap=cmap, **kwargs)

    def joint(x,y,**kwargs):
        sns.jointplot(x=x, y=y, kind="hex", color="#4CB391")

    def my_hist(x, **kwargs):
        ax0 = plt.gca()
        ax = ax0.twinx()
        sns.despine(ax=ax, left=True, top=True, right=False)
        ax.yaxis.tick_right()
        ax.set_ylabel('Counts')
        ax.hist(x, **kwargs)

    df.loc[:,'d_norm'] = df['d_norm'].apply(lambda x: np.nanmean(x))
    unique_sizes = df['ground_truth'].unique()

    figs, names = [], []
    for i, size in enumerate(unique_sizes):
        sub_df = df.loc[(df['method']=='MIN-POLY')&(df['d_norm']<100)&(df['d_norm_NALM']<100)&(df['ground_truth']==size)].copy()

        variables = ["d_norm", "d_norm_NALM",'ground_truth']
        g = sns.PairGrid(sub_df, vars=variables, aspect=1, diag_sharey=False)

        g.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
        

        lims = [(0, 120)]*10
        tick_inc = [20]*10

        for ax,(ylims, xlims),(yticks, xticks) in zip(g.axes.flat, 
                                                    itertools.product(lims, lims),
                                                    itertools.product(tick_inc, tick_inc)
                                                    ):
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xticks))
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(yticks))
            ax.set_xlabel('test')

        g.map_upper(hexbin, extent=[-20, 130, -20, 130])#
        g.map_lower(sns.kdeplot,linewidths=0.5,color=s.c10,zorder=0)
        g.map_lower(sns.scatterplot,s=5,color=s.c20,zorder=5)
        g.map_diag(my_hist,color=s.c20,zorder=0,bins=np.arange(0,100,2),alpha=0.5)#
        g.map_diag(sns.kdeplot,lw=0.5,color=s.c10,zorder=5)        

        sns.despine(offset=5, trim=True)
        g.tight_layout()
        figs.append(g)
        names.append(f'BatchHexbinCorrelations{size}nm')

    if show:
        plt.show()
    return figs, names

if __name__=='__main__':
    figs, names = make_figure(show=True)
    for fig, name in zip(figs,names):
        continue
        path_to_figure = Figures().save_fig(fig, name,meta={'generating script': script_name})