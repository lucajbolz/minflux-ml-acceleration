import os, sys
script_name = os.path.basename(__file__)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

import lib.utilities as ut
from lib.plotting.style import Style

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

    # Create a grid of stacked histograms
    num_rows = len(unique_sizes)
    num_cols = 3  # Two histograms: 'error' and 'NALM_error' stacked together

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6*num_rows),sharex='all')

    # Define the maximum number of rows per figure
    max_rows = 2

    # Create a grid of stacked histograms for each figure
    num_figures = int(np.ceil(len(unique_sizes) / max_rows))
    figs, names = [], []
    for fig_num in range(num_figures):
        start_idx = fig_num * max_rows
        end_idx = (fig_num + 1) * max_rows
        sizes_figure = unique_sizes[start_idx:end_idx]

        # Determine the number of rows and columns for the current figure
        num_rows = max(2,len(sizes_figure))
        num_cols = 3

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(7.086, 2.5 * num_rows),sharex='all',sharey='col',constrained_layout=True)

        # Iterate through unique sizes and plot histograms for the current figure
        for i, size in enumerate(sizes_figure):
            # Filter the dataframe for the current size
            current_size_df = sub_df[sub_df['ground_truth'] == size]
            xmin,xmax = int(min(current_size_df.d_norm)),int(max(current_size_df.d_norm))
            bin_width = 2
            bins = 50

            if size<50:
                legend_pos0 = (1.,1.)
                loc0 = 'upper right'
                legend_pos1 = (1.,0.05)
                loc1 = 'lower right'
            else:
                legend_pos0 = (0.,1.)
                loc0 = 'upper left'
                legend_pos1 = (0.,1.)
                loc1 = 'upper left'

            # histogram and KDE
            sns.histplot(current_size_df,x='d_norm',binwidth=2,binrange=(0,100),ax=axes[i, 0],color=s.c20,label=f'd={size} nm')
            axes[i, 0].set_ylabel("d (nm)")
            axes[i, 0].set_ylabel("Counts")
            axes[i,0].legend(bbox_to_anchor=legend_pos0, loc=loc0 ,fontsize=7)

            sns.histplot(current_size_df,x='d_norm',ax=axes[i, 1], color=s.c20, binwidth=2, binrange=(0,100), stat="density",
                element="step", fill=False, cumulative=True, common_norm=False,label='Minimum')
            sns.histplot(current_size_df,x='d_norm_NALM',ax=axes[i, 1], color=s.c10, binwidth=2, binrange=(0,100), stat="density",
                element="step", fill=False, cumulative=True, common_norm=False,label='Control')
            axes[i, 1].set_ylim(0,1)
            axes[i, 1].set_ylabel("d (nm)")
            axes[i, 1].set_ylabel("Density")
            axes[i,1].legend(bbox_to_anchor=legend_pos1, loc=loc1, fontsize=7)
            
            sns.scatterplot(current_size_df,x='d_norm',y='d_norm_NALM',ax=axes[i, 2],color=s.c20,s=5,legend=False)
            sns.kdeplot(current_size_df,x='d_norm',y='d_norm_NALM',ax=axes[i, 2],color=s.c20,alpha=0.5, linewidths=0.5,legend=False)
            axes[i, 2].set_ylim(0,100)
            axes[i, 2].set_xlabel("d (nm)")
            axes[i, 2].set_ylabel("Control (nm)")

            for ax in axes[i]:
                ax.set_xlabel("d (nm)")
                ax.set_xlim(0,100)
                ax.set_xticks(np.arange(0,101,10))
                ax.set_xticks(np.arange(0,101,5),minor=True)
                s.hide_axes(axes=[ax],dirs=['top','right'])
                s.drop_axes(axes=[ax],dirs=['left','bottom'])
        figs.append(fig)
        names.append(f'BatchHistogramsCumulativePMF{fig_num}')
    if show:
        plt.show()
    return figs, names

if __name__=='__main__':
    figs, names = make_figure(show=True)
    for fig, name in zip(figs,names):
        continue
        path_to_figure = Figures().save_fig(fig, name,meta={'generating script': script_name})