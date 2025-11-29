"""
Experimental line-scan data.

copyright: @Thomas Hensel, 2023
"""

import os
script_name = os.path.basename(__file__)

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from lib.plotting.artefacts import Figures
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
    gt = [6,8,10,15,20,30,40,50,60,75,90]
    gt_error = [1,1,2,2,2,3,5,5,5,5,5]
    df['gt_error'] = np.nan
    for d,err in zip(gt,gt_error):
        df.loc[df['ground_truth']==d,'gt_error'] = err
        # Define the probability of flipping the sign (adjust this value as needed)
        flip_probability = 0.5
        # Create a mask for rows to flip signs based on the flip_probability
        flip_mask = df.loc[df['ground_truth']==d].sample(frac=flip_probability, replace=True).index
        # Flip the signs of the selected column where the mask is True
        df.loc[flip_mask, 'gt_error'] = -df.loc[df['ground_truth']==d].loc[flip_mask, 'gt_error']
        

    def remove_outliers(group,key,range=1.5):
        Q1 = group[key].quantile(0.25)
        Q3 = group[key].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - range * IQR
        upper_bound = Q3 + range * IQR
        return group[(group[key] >= lower_bound) & (group[key] <= upper_bound)]
    def map_labels(legend,map):
        # method to rename labels in legend
        for t in legend.texts:
            if t._text in map.keys():
                t.set_text(map[t._text])
        return legend

    fig, ax = plt.subplot_mosaic(
    [['a1)','a2)']
     ,['a)','a)']
     ,['b)','b)']
     ]#grid
    ,empty_sentinel="BLANK"
    ,gridspec_kw = {
        "width_ratios": [1,1]#widths
        ,"height_ratios": [.2,1,1]#heights
        }
    ,constrained_layout=True
    ,figsize=(7.086,6.)
    )

    # upper row of panels
    x=np.linspace(0,1,100)
    ax['a1)'].plot(x,1+0.8*np.cos(x*2*np.pi),color=s.c10)
    ax['a1)'].fill_between(x[35:65], -1, (1+0.8*np.cos(x*2*np.pi))[35:65], color=s.c10, alpha=0.4)
    ax['a1)'].scatter(.4,0.1,marker='*',s=100, color=s.c30,zorder=5)
    ax['a1)'].scatter(.6,0.1,marker='*',s=100, color=s.c30,zorder=5)
    ax['a1)'].set_xlim(0,1)
    ax['a1)'].set_ylim(-.3,2)
    ax['a1)'].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')

    ax['a2)'].plot(x,1+0.8*np.cos(x*2*np.pi),color=s.c20)
    ax['a2)'].fill_between(x, -1, 1+0.8*np.cos(x*2*np.pi), color=s.c20, alpha=0.4)
    ax['a2)'].scatter(.4,0.1,marker='*', s=100, color=s.c30,zorder=5)
    ax['a2)'].scatter(.6,0.1,marker='*', s=100, color=s.c30,zorder=5)
    ax['a2)'].set_xlim(0,1)
    ax['a2)'].set_ylim(-.3,2)
    ax['a2)'].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')
    
    # Panel a: violins
    filtered_df = df.loc[(df['method'].apply(lambda x: ('HARMONIC' in x)|('MIN-POLY' in x))) & (~df['error'].apply(lambda x: np.isnan(x)))]
    grouped = filtered_df.groupby(['ground_truth','method'])
    # Step 3: Apply the remove_outliers function to each group and concatenate the results
    filtered_df = grouped.apply(lambda x: remove_outliers(x,'error',range=1.5)).reset_index(drop=True)
    
    tmp_ax = sns.violinplot(data=filtered_df, x="ground_truth", y="error", hue="method",
               split=True, inner="quart", linewidth=1,
               palette={'MIN-POLY':s.c10,'HARMONIC':s.c20},ax=ax['a)'],saturation=1.,alpha=.3, trim=True, legend=False)
    #ax['a)'].plot([], [], color=s.c10, label='Minimum')
    #ax['a)'].plot([], [], color=s.c20, label='Full')
    delta = 0.05
    for ii, item in enumerate(ax['a)'].collections):
        if isinstance(item, mpl.collections.PolyCollection):
            item.set_alpha(.8)
            path, = item.get_paths()
            vertices = path.vertices
            if ii % 2:  # -> to right
                vertices[:, 0] += delta
            else:  # -> to left
                vertices[:, 0] -= delta
    for i, line in enumerate(ax['a)'].get_lines()):
        line.get_path().vertices[:, 0] += delta if i // 3 % 2 else -delta
    
    flierprops = dict(marker='.', markerfacecolor='k', markersize=3,
                  markeredgecolor='none')
    boxprops = dict(facecolor='gray',edgecolor='none',alpha=.8,zorder=-1)
    medianprops = dict(alpha=.0)
    whiskerprops = dict(alpha=0.0)
    capprops = dict(alpha=0.0)
    sns.boxplot(data=filtered_df, x="ground_truth", y="gt_error",ax=ax['a)'], showmeans=False, boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops)
    gray_patch = mpatches.Patch(color='gray', alpha=0.8, label='Expected')
    handles, labels = ax['a)'].get_legend_handles_labels()
    handles += [gray_patch]
    labels += [gray_patch._label]

    lims = ax['a)'].get_xlim()
    
    ax['a)'].set_ylim(-30,30)
    ax['a)'].set_yticks([-30,-20,-10,0,10,20,30])
    tmp_ax.set(xlabel=None)
    ax['a)'].set_ylabel('Deviation [nm]')
    ax['a)'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=True,labelright=True,direction='in')
    ax['a)'].legend(handles, labels, bbox_to_anchor=(0., -.05), loc='lower left', title='',fontsize=7)
    leg = ax['a)'].legend_
    map_dict = {'HARMONIC':'Full','MIN-QUAD':'Minimum','MIN-POLY':'Minimum'}
    ax['a)'].legend_ = map_labels(leg,map_dict)


    # Panel b) add NALM violins as control
    filtered_df = df.loc[df['method'].isin(['MIN-POLY'])]    
    reshaped_df = pd.melt(filtered_df, id_vars=["ground_truth", "method"], value_vars=["error", "NALM_error"])
    grouped = reshaped_df.groupby(['variable','ground_truth'])
    filtered_df = grouped.apply(lambda x: remove_outliers(x,'value')).reset_index(drop=True)
    
    # Create the split violin plot
    sns.violinplot(data=filtered_df, x="ground_truth", y="value", hue="variable",
                split=True, inner="quart", linewidth=1,
                palette={'error':s.c10,'NALM_error':s.c30}, saturation=1., ax=ax['b)'], legend=False)

    #sns.violinplot(data=filtered_df, x="ground_truth", y="NALM_error", hue="method",
    #           split=True, inner="quart", linewidth=1,
    #           palette='summer',ax=ax['c)'],saturation=0.6,legend=False)
    delta = 0.05
    for ii, item in enumerate(ax['b)'].collections):
        if isinstance(item, mpl.collections.PolyCollection):
            item.set_alpha(.8)
            path, = item.get_paths()
            vertices = path.vertices
            if ii % 2:  # -> to right
                vertices[:, 0] += delta
            else:  # -> to left
                vertices[:, 0] -= delta
    for i, line in enumerate(ax['b)'].get_lines()):
        line.get_path().vertices[:, 0] += delta if i // 3 % 2 else -delta

    filtered_df = df.loc[(df['method'].apply(lambda x: ('MIN-POLY' in x))) & (~df['error'].apply(lambda x: np.isnan(x)))]
    grouped = filtered_df.groupby(['ground_truth','method'])
    # Step 3: Apply the remove_outliers function to each group and concatenate the results
    filtered_df = grouped.apply(lambda x: remove_outliers(x,'error',range=1.5)).reset_index(drop=True)
    flierprops = dict(marker='.', markerfacecolor='k', markersize=3,
                  markeredgecolor='none')
    boxprops = dict(facecolor='gray',edgecolor='none',alpha=.8,zorder=-1)
    medianprops = dict(alpha=.0)
    whiskerprops = dict(alpha=0.0)
    capprops = dict(alpha=0.0)
    sns.boxplot(data=filtered_df, x="ground_truth", y="gt_error",ax=ax['b)'], showmeans=False, boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops)
    gray_patch = mpatches.Patch(color='gray', alpha=0.8, label='Expected')
    handles, labels = ax['b)'].get_legend_handles_labels()
    handles += [gray_patch]
    labels += [gray_patch._label]

    lims = ax['b)'].get_xlim()
    #ax['c)'].fill_between(np.linspace(lims[0],lims[1],100), np.full(shape=(100,),fill_value=-2.), np.full(shape=(100,),fill_value=+2.), color='k', alpha=0.4,zorder=-1)
    #ax['c)'].hlines(0,xmin=lims[0],xmax=lims[1],ls='-',color='k',alpha=0.6,zorder=0,label='Expected')
    ax['b)'].set_ylim(-30,30)
    ax['b)'].set_yticks([-30,-20,-10,0,10,20,30])
    ax['b)'].set_xlabel('Expected [nm]')
    ax['b)'].set_ylabel('Deviation [nm]')
    #ax['a)'].hlines(0,xmin=lims[0],xmax=lims[1],ls='-',lw=5,color='k',alpha=0.0,zorder=0,label='Expected')
    ax['b)'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=True,labelright=True,direction='in')  
    ax['b)'].legend(handles, labels, bbox_to_anchor=(0., -.05), loc='lower left', title='',fontsize=7)
    leg = ax['b)'].legend_ # change labels
    map_dict = {'error':'Minimum','NALM_error':'Control'}
    ax['b)'].legend_ = map_labels(leg,map_dict)


    s.drop_axes(axes=[ax['a)'],ax['b)']],dirs=['left','bottom','right'])
    s.hide_axes(axes=[ax['b)']],dirs=['top'])#ax['a1)'],ax['a2)'],
    s.hide_axes(axes=[ax['a)']],dirs=['top'])

    if show:
        plt.show()
    return fig

if __name__=='__main__':
    fig = make_figure(show=True)
    #Figures().save_fig(fig, 'extended-data-min-vs-max-figure',meta={'generating script': script_name})