"""
Experimental line-scan data.

copyright: @Thomas Hensel, 2023
"""
import os
import argparse
script_name = os.path.basename(__file__)

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import numpy as np

import lib.utilities as ut
from lib.plotting.style import Style
from lib.plotting.artefacts import Figures

import pylustrator
pylustrator.start()

def corrected_sizes(gt):
    d=gt
    if gt == 6:
        d = 6
    elif gt == 8:
        d = 8.4
    elif gt == 10:
        d = 10.2
    elif gt == 12:
        d = 12
    elif gt == 15:
        d = 14.3
    elif gt == 20:
        d=20.1
    elif gt == 25:
        d = 25
    elif gt==30:
        d = 31.8
    elif gt==40:
        d = 40.8
    elif gt==50:
        d = 50.9
    elif gt==60:
        d = 59
    elif gt==75:
        d = 74.5
    elif gt==90:
        d = 90.7   
    return d

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

def calculated_errors(d):
    if d < 10:
        err = 1
    elif d < 30:
        err = 2
    elif d < 40:
        err = 3
    elif d >= 40:
        err = 5
    return [d, d - err, d + err]

def get_errorbands(x):
    error_bands = [calculated_errors(d) for d in x]
    # Extract values for plotting
    x_values = [item[0] for item in error_bands]
    lower_err_values = [item[1] for item in error_bands]
    upper_err_values = [item[2] for item in error_bands]
    return x_values, lower_err_values, upper_err_values

def plotLineScanMin(axis,inset_pos,LineScanDataFrame,style):
    df = LineScanDataFrame.copy()
    s = style
    axis.set_aspect('equal')
    flierprops = dict(marker='.', markerfacecolor='k', markersize=3,
                markeredgecolor='none',alpha=.5)
    boxprops = dict(facecolor=s.c10,edgecolor=s.c10,alpha=1.)
    medianprops = dict(color = 'k', linewidth = 1.,alpha=1.)
    whiskerprops = dict(color=s.c10,alpha=1.)
    capprops = dict(color=s.c10,alpha=1.)

    filtered_df = df.loc[df['method'].isin(['MIN-POLY']) & (~df['error'].apply(lambda x: np.isnan(x)))].groupby(['ground_truth'],group_keys=False)

    grouped_mean_data = filtered_df['d_norm'].agg(list).apply(lambda x:x)
    mean = grouped_mean_data.to_list()
    grouped_pos_data = filtered_df['ground_truth'].agg(list).apply(lambda x: np.nanmean(x))
    pos = grouped_pos_data.to_list()
    grouped_std_data = filtered_df['d_norm'].agg(list).apply(lambda x: np.nanstd(x))
    std = grouped_std_data.to_list()

    #sns.boxplot(data=filtered_df.apply(lambda x:x), x="ground_truth", y="d_norm",ax=ax['c'], showmeans=False, boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops)
    boxplot = axis.boxplot(mean, positions=pos, notch=False, widths=len(pos)*[1.5], showmeans=False, meanline=True,patch_artist=True, boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops, manage_ticks=False)    
    axis.set_xlim(0,100)#2*[1.5]+7*[4.]
    axis.set_ylim(0,100)
    axis.set_xticks(np.arange(0,101,20))
    axis.set_yticks(np.arange(0,101,20))
    x = np.linspace(0,101,2)
    axis.plot(x,x,label='Expected Mean',color='k',alpha=.5,zorder=0)
    x_values, lower_err_values, upper_err_values = get_errorbands(x)
    axis.fill_between(x_values, lower_err_values, upper_err_values, color=s.c30, alpha=0.6,label='Expected Error',zorder=-1)
    
    #----------------------
    # inset cartoon of minimum
    x=np.linspace(0,1,100)
    axins = axis.inset_axes(inset_pos)
    axins.plot(x,1+0.8*np.cos(x*2*np.pi),color=s.c10)
    axins.fill_between(x[30:70], -1, (1+0.8*np.cos(x*2*np.pi))[30:70], color=s.c10, alpha=0.4)
    axins.scatter(.35,0.,marker='*',s=100, color=s.c30,edgecolors='k',zorder=5)
    axins.scatter(.65,0.,marker='*',s=100, color=s.c30,edgecolors='k',zorder=5)
    axins.set_xlim(0,1)
    axins.set_ylim(-.3,2)
    axins.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')

    axis.set_xlabel('Expected d (nm)')
    axis.set_ylabel('Estimated d (nm)')
    axis.tick_params(top=True, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=True,labelright=False,direction='in')
    red_patch = mpatches.Patch(color=s.c10, label='Scan (Min)')
    handles, labels = axis.get_legend_handles_labels()
    handles += [red_patch]
    labels += [red_patch._label]
    axis.legend(handles, labels, bbox_to_anchor=(1.0, .0), loc='lower right', frameon=True)
    
    s.hide_axes(axes=[axis],dirs=['top','right'])
    s.drop_axes(axes=[axis])
    pass

def plotMINFLUXData(axis,MINFLUXdf,inset_pos,style):
    axis.set_aspect('equal')
    s = style
    # add static MINFLUX results
    sub_data = MINFLUXdf.loc[~MINFLUXdf['gt'].isin([6.0])].groupby(['gt'],group_keys=False)
    #sub_data = MINFLUXdf.groupby(['gt'],group_keys=False)
    grouped_mean_data = sub_data.apply(lambda x:x).reset_index().groupby(['gt','file'],group_keys=False)['d_norm'].agg(list).apply(lambda x:np.nanmedian(x))#take median of each nanoruler
    mean = grouped_mean_data.reset_index().groupby('gt')['d_norm'].agg(list).apply(lambda x:x).to_list()
    #grouped_mean_data = sub_data['d_norm'].agg(list).apply(lambda x:x)
    #mean = grouped_mean_data.to_list()
    grouped_pos_data = sub_data['gt'].agg(list).apply(lambda x: np.nanmean(x))
    pos = grouped_pos_data.to_list()
    grouped_std_data = sub_data['d_norm'].agg(list).apply(lambda x: np.nanstd(x))
    std = grouped_std_data.to_list()

    flierprops = dict(marker='.', markerfacecolor='k', markersize=3,
                markeredgecolor='none',alpha=0.5)
    boxprops = dict(facecolor=s.c10,edgecolor=s.c10,alpha=1.)
    medianprops = dict(color = "k", linewidth = 1.,alpha=1.)
    whiskerprops = dict(color=s.c10,alpha=1.)
    capprops = dict(color=s.c10,alpha=1.)

    axis.set_xlim(0,40)#2*[1.5]+7*[4.]
    axis.set_ylim(0,40)
    boxplot = axis.boxplot(mean, positions=pos, notch=False, widths=len(pos)*[1.], showmeans=False, meanline=True, patch_artist=True, boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops, manage_ticks=False)    
    #errbar = axis.errorbar(pos, [np.median(m) for m in mean], std, fmt='o', capsize=3)
    lims = axis.get_xlim()
    x = np.linspace(lims[0],lims[1],20)
    axis.plot(x,x,label='Expexted Mean',color='k',alpha=.5,zorder=0)
    x_values, lower_err_values, upper_err_values = get_errorbands(x)
    axis.fill_between(x_values, lower_err_values, upper_err_values, color=s.c30, alpha=0.6,label='Expexted Error',zorder=-1)
    axis.set_xticks(np.arange(0,41,5))
    axis.set_xlabel('Expected d (nm)')
    axis.set_ylabel('d (nm)')
    axis.tick_params(top=True, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=True,labelright=False,direction='in')
    red_patch = mpatches.Patch(color=s.c10, label='MINFLUX')
    handles, labels = axis.get_legend_handles_labels()
    handles += [red_patch]
    labels += [red_patch._label]
    axis.legend(handles, labels, bbox_to_anchor=(1.0, .0), loc='lower right', frameon=True)

    # inset cartoon
    def partialCos(array):
        condition1 = (0.3 < array) & (array < 0.35)
        condition2 = (0.45 < array) & (array < 0.55)
        condition3 = (0.65 < array) & (array < 0.7)

        # Combine the conditions using logical OR (|)
        result = condition1 | condition2 | condition3
        y = 1+0.8*np.cos(x*2*np.pi)
        y[~result]=np.nan
        return y
    
    x=np.linspace(0,1,100)
    y = partialCos(x)
    axins = axis.inset_axes(inset_pos)
    axins.plot(x,y,color=s.c10)
    axins.fill_between(x, -1, y, color=s.c10, alpha=0.4)
    axins.scatter(.35,0.,marker='*',s=100, color=s.c30,edgecolors='k',zorder=5)
    axins.scatter(.65,0.,marker='*',s=100, color=s.c30,edgecolors='k',zorder=5)
    axins.set_xlim(0,1)
    axins.set_ylim(-.3,1)
    axins.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')

    s.hide_axes(axes=[axis],dirs=['top','right'])
    s.drop_axes(axes=[axis])
    pass

def plotControl(axis,inset_pos,LineScanDataFrame,style):
    df = LineScanDataFrame.copy()
    s = style
    axis.set_aspect('equal')
    # ------------------------------------------
    # correlation between estimate and NALM
    def get_errs(series):
        median = series.apply(lambda x:np.nanmedian(x))
        q25 = series.apply(lambda x: np.nanpercentile(x, 25))
        q75 = series.apply(lambda x: np.nanpercentile(x, 75))
        lower_err = median - q25
        upper_err = q75 - median
        err = [lower_err,upper_err]
        return median, err

    filtered_df = df.loc[df['method'].isin(['MIN-POLY']) & (~df['error'].apply(lambda x: np.isnan(x)))].groupby(['ground_truth'],group_keys=False)
    grouped_est = filtered_df.apply(lambda x: remove_outliers(x,'d_norm',range=4)).groupby(['ground_truth'])['d_norm'].agg(list)
    y_median, y_errs = get_errs(grouped_est)

    grouped_NALM = filtered_df.apply(lambda x: remove_outliers(x,'d_norm_NALM',range=4)).groupby(['ground_truth'])['d_norm_NALM'].agg(list)
    x_median, x_errs = get_errs(grouped_NALM)
    
    axis.set_xlim(0,100)
    axis.set_ylim(0,100)
    axis.errorbar(x_median, y_median, xerr=x_errs, yerr=0, fmt='',capsize=1.,label='Control',linestyle='none',lw=1.,color='k',ecolor='k')
    axis.errorbar(x_median, y_median, xerr=0, yerr=y_errs, fmt='',capsize=1.,label='Scan (Min)',linestyle='none',lw=1.,color=s.c10,ecolor=s.c10)
    x = np.linspace(0,100,5)
    axis.plot(x,x,label='Expected Mean',color='k',alpha=.5,zorder=-1)
    axis.set_xticks(np.arange(0,101,20))
    axis.set_yticks(np.arange(0,101,20))
    axis.set_xlabel('Control d (nm)')
    axis.set_ylabel('Estimated d (nm)')
    axis.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
    axis.legend(bbox_to_anchor=(1., .0), loc='lower right', frameon=True)
    

    #----------------------
    # inset cartoon of control
    x=np.linspace(0,1,100)
    axins = axis.inset_axes(inset_pos)
    axins.plot(x,1+0.8*np.cos(x*2*np.pi),color='k')
    axins.plot(x,.5*(1+np.cos((x-.15)*2*np.pi)),color='k',alpha=.5)
    axins.scatter(.35,0.,marker='*',s=100, color=s.c30,edgecolors='k',zorder=5)
    axins.scatter(.65,0.,marker='*',s=100, color=s.c30,edgecolors='k',zorder=5)
    axins.set_xlim(0,1)
    axins.set_ylim(-.3,2)
    axins.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')

    s.hide_axes(axes=[axis],dirs=['top','right'])
    s.drop_axes(axes=[axis])
    pass

def plotHistogram(fig,axis,MINFLUXdf,style):
    # create histogram
    s=style
    size = 8.4
    gs = axis.get_gridspec()
    nrow, ncol, start, stop = axis.get_subplotspec().get_geometry()
    start_row, start_col = divmod(start, ncol)
    stop_row, stop_col = divmod(stop, ncol)
    axis.remove()
    inner_grid = gs[start_row:stop_row+1,start_col:stop_col+1].subgridspec(2, 1, wspace=0.0, hspace=0.0,height_ratios=[.1,.9])
    axis = fig.add_subplot(inner_grid[1, 0])
    ax2 = fig.add_subplot(inner_grid[0, 0], sharex=axis)
    
    sub_data = MINFLUXdf.loc[(MINFLUXdf['gt']==size)].copy()#& (MINFLUXdf['label'].isin([8.0, 8.2]))
    #sub_data = MINFLUXdf.loc[~MINFLUXdf['gt'].isin([6.0, 10.2])].groupby(['gt'],group_keys=False)
    grouped_mean_data = sub_data.groupby(['gt','file'],group_keys=False)['d_norm'].agg(list).apply(lambda x:np.nanmean(x)).reset_index()#take mean of each nanoruler
    grouped_mean_data['method'] = 'mean'
    grouped_median_data = sub_data.groupby(['gt','file'],group_keys=False)['d_norm'].agg(list).apply(lambda x:np.nanmedian(x)).reset_index()#take median of each nanoruler
    grouped_median_data['method'] = 'median'
    df = pd.concat([grouped_mean_data,grouped_median_data], ignore_index=True).sort_values(['gt','file'])
    
    means = df.loc[df['method']=='mean'].groupby('gt')['d_norm'].agg(list).apply(lambda x:x).to_list()
    medians = df.loc[df['method']=='median'].groupby('gt')['d_norm'].agg(list).apply(lambda x:x).to_list()
    pos = df.loc[df['method']=='mean'].groupby('gt')['gt'].agg(list).apply(lambda x: np.nanmean(x)).to_list()

    median = [np.median(m) for m in medians]
    mean = [np.mean(m) for m in means]#df.loc[df['method']=='mean'].groupby('gt')["d_norm"].mean().values
    std = [np.std(m) for m in means]#df.loc[df['method']=='mean'].groupby('gt')["d_norm"].std().values
    IQR = [np.quantile(m,0.75) - np.quantile(m,0.25) for m in means]
    
    flierprops = dict(marker='.', markerfacecolor='k', markersize=3,
                markeredgecolor='none',alpha=0.5)
    boxprops = dict(facecolor=s.c10,edgecolor=s.c10,alpha=1.)
    medianprops = dict(color = "k", linewidth = 1.,alpha=1.)
    whiskerprops = dict(color=s.c10,alpha=1.)
    capprops = dict(color=s.c10,alpha=1.)

    #ax2.set_xlim(5,40)#2*[1.5]+7*[4.]
    #ax2.set_ylim(5,40)
    ax2.boxplot(means, notch=False, vert=False, showmeans=False, meanline=True, patch_artist=True, boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops, manage_ticks=False)    
    ax2.tick_params(top=False, labeltop=False, bottom=True, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')
    
    #counts, edges, _ = axis.hist(grouped_mean_data['d_norm'],color=s.c10,alpha=1.,density=False,orientation='vertical',bins=np.linspace(0,20,20),rwidth=1.)
    sns.histplot(x=df.loc[df['method']=='mean','d_norm'],color=s.c10, binwidth=.5, kde=True, stat='density', ax=axis)
    #ymax = np.max(counts)
    for me,med, st,iqr in zip(mean,median,std,IQR):
        #axis.axvline(m, color='k', linestyle='-',lw=1.)#label=f'Median: {median:.1f} nm'
        axis.text(.9, .95, r'Expected d='+f'{size:.1f} nm', transform=axis.transAxes,
                        fontweight='bold', verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='1.0', edgecolor='none', pad=1.0))
        axis.text(.9, .9, r'median d='+f'{med:.1f} nm', transform=axis.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='1.0', edgecolor='none', pad=1.0))
        axis.text(.9, .85, r'IQR='+f'{iqr:.1f} nm', transform=axis.transAxes,
                        color='k',verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='none', edgecolor='none', pad=1.0))
        axis.text(.9, .8, r'mean d='+f'{me:.1f} nm', transform=axis.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='1.0', edgecolor='none', pad=1.0))
        axis.text(.9, .75, r'$\sigma$='+f'{st:.1f} nm', transform=axis.transAxes,
                        color='k',verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='none', edgecolor='none', pad=1.0))
    #axis.annotate("", xytext=(mean-1-std, ymax/2), xy=(mean-1+std, ymax/2), xycoords=('data','data'),arrowprops=dict(arrowstyle="<->",linewidth=1.,color='white'))
    axis.set_xlim(0,30)
    axis.set_ylim(0,0.15)
    axis.set_yticks([0,.05,.1,0.15])
    #s.handle_ticks(axis,nx=4,ny=7)
    axis.set_xlabel('d (nm)')
    axis.set_ylabel('Probability')
    axis.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
    axis.legend()

    s.hide_axes(axes=[axis],dirs=['top','right'])
    s.hide_axes(axes=[ax2],dirs=['top','left','right'])
    s.drop_axes(axes=[axis,ax2])
    pass

def plotViolins(axis,inset_pos,LineScanDataFrame,style):
    df = LineScanDataFrame.copy()
    s = style
    #---------------------------
    # Panel e: Line-Scan violins
    filtered_df = df.loc[(df['method'].apply(lambda x: ('HARMONIC' in x)|('MIN-POLY' in x))) & (~df['error'].apply(lambda x: np.isnan(x)))]
    grouped = filtered_df.groupby(['ground_truth','method'])
    # Step 3: Apply the remove_outliers function to each group and concatenate the results
    filtered_df = grouped.apply(lambda x: remove_outliers(x,'error',range=1.5)).reset_index(drop=True)
    
    tmp_ax = sns.violinplot(data=filtered_df, x="ground_truth", y="error", hue="method",
            split=True, inner="quart", linewidth=1,
            palette={'MIN-POLY':s.c10,'HARMONIC':s.c20},ax=axis,saturation=1.,alpha=.3, trim=True, legend=False)

    #ax['a)'].plot([], [], color=s.c10, label='Minimum')
    #ax['a)'].plot([], [], color=s.c20, label='Full')
    delta = 0.05
    for ii, item in enumerate(axis.collections):
        if isinstance(item, matplotlib.collections.PolyCollection):
            item.set_alpha(.8)
            path, = item.get_paths()
            vertices = path.vertices
            if ii % 2:  # -> to right
                vertices[:, 0] += delta
            else:  # -> to left
                vertices[:, 0] -= delta
    for i, line in enumerate(axis.get_lines()):
        line.get_path().vertices[:, 0] += delta if i // 3 % 2 else -delta
    
    flierprops = dict(marker='.', markerfacecolor='k', markersize=3,
                markeredgecolor='none')
    boxprops = dict(facecolor=s.c30,edgecolor='none',alpha=1.,zorder=-1)
    medianprops = dict(alpha=.0)
    whiskerprops = dict(alpha=0.0)
    capprops = dict(alpha=0.0)
    sns.boxplot(data=filtered_df, x="ground_truth", y="gt_error",ax=axis, showmeans=False, boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops)
    yellow_patch = mpatches.Patch(color=s.c30, alpha=0.8, label='Expected Error')
    handles, labels = axis.get_legend_handles_labels()
    handles += [yellow_patch]
    labels += [yellow_patch._label]
    
    axis.set_ylim(-25,25)
    axis.set_yticks([-20,-10,0,10,20])
    tmp_ax.set(xlabel='Expected d (nm)')
    axis.set_ylabel('Deviation (nm)')
    axis.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=True,labelright=True,direction='in')
    axis.legend(handles, labels, bbox_to_anchor=(-.01, -.1), loc='lower left',frameon=True)
    leg = axis.legend_
    map_dict = {'HARMONIC':'Scan (Full)','MIN-QUAD':'Scan (Min)','MIN-POLY':'Scan (Min)'}
    axis.legend_ = map_labels(leg,map_dict)

    # inset cartoon: Full
    x=np.linspace(0,1,100)
    axins = axis.inset_axes(inset_pos)
    axins.plot(x,1+0.8*np.cos(x*2*np.pi),color=s.c20)
    axins.fill_between(x, -1, 1+0.8*np.cos(x*2*np.pi), color=s.c20, alpha=0.2)
    axins.scatter(.4,0.1,marker='*', s=100, color=s.c30,edgecolors='k',zorder=5)
    axins.scatter(.6,0.1,marker='*', s=100, color=s.c30,edgecolors='k',zorder=5)
    axins.set_xlim(0,1)
    axins.set_ylim(-.3,2)
    axins.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')

    s.hide_axes(axes=[axis],dirs=['top','right'])
    s.drop_axes(axes=[axis])
    pass

def make_figure(plot_style='nature',color_scheme='default',show=True):

    s = Style(style=plot_style,color_scheme=color_scheme)
    base_dir = os.path.split(__file__)[0]

    file = 'static-minflux-data'

    # get static MINFLUX data
    file = ut.BaseFunc().find_files(base_dir+'/data/', lambda file: ut.BaseFunc().match_pattern(file, file, match='partial'),max_files=1)[0] # load pkl
    MINFLUXdf = pd.read_pickle(file)
    target = 5000
    unique_targets = MINFLUXdf["chunk_size"].unique()
    print(f'Available targets: {unique_targets}')
    if not 5000 in unique_targets:
        target = max(unique_targets)
        print(f'Target not available. Switched to target={target}')
    print(f'Chosen target: {target}')
    MINFLUXdf = MINFLUXdf.loc[(MINFLUXdf['chunk_size']==target)&(~MINFLUXdf['label'].isin([8.1]))]#leave out batch 8.1 due to miserable quality of minimum during measurement
    
    # prepare data
    MINFLUXdf['gt_error'] = np.nan
    gt = [6,8,10,15,20,30,40,50,60,75,90]
    gt_error = [1,1,2,2,2,3,5,5,5,5,5]
    for d,err in zip(gt,gt_error):
        MINFLUXdf.loc[MINFLUXdf['gt']==d,'gt_error'] = err
        flip_probability = 0.5
        flip_mask = MINFLUXdf.loc[MINFLUXdf['gt']==d].sample(frac=flip_probability, replace=True).index
        MINFLUXdf.loc[flip_mask, 'gt_error'] = -MINFLUXdf.loc[MINFLUXdf['gt']==d].loc[flip_mask, 'gt_error']
    MINFLUXdf.loc[:,'gt'] = MINFLUXdf['gt'].map(lambda x: int(x)).copy()
    MINFLUXdf['gt'] = MINFLUXdf['gt'].map(corrected_sizes)
    MINFLUXdf['error'] = MINFLUXdf['d_norm']-MINFLUXdf['gt']


    fig, ax = plt.subplot_mosaic(
    [
        ['BLANK','BLANK']
        ,['a','b']
     ]#grid
    ,empty_sentinel="BLANK"
    ,gridspec_kw = {
        "width_ratios": [1.,1.]#widths
        ,"height_ratios": [.1,1]#heights
        }
    ,constrained_layout=True
    ,figsize=s.get_figsize(rows=1.1,cols=2,ratio=1)#
    ,sharex=False
    ,sharey=False
    )

    inset_pos = [0.01, 0.8, 0.5, 0.2]

    #fancyMINFLUXViolins(ax['a'],MINFLUXdf,inset_pos,s)
    plotMINFLUXData(ax['a'],MINFLUXdf,inset_pos,s)

    plotHistogram(fig,ax['b'],MINFLUXdf,s)

    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
    plt.figure(1).text(0.0091, 0.9483, 'a', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[0].new
    plt.figure(1).text(0.5518, 0.9483, 'b', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[1].new
    plt.figure(1).text(0.0782, 0.9205, 'MINFLUX', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[2].new
    #% end: automatic generated code from pylustrator
    if show:
        plt.show()
    return fig

if __name__=='__main__':
    fig = make_figure(show=True)
    #path_to_figure = Figures().save_fig(fig, 'StaticDataFigure',meta={'generating script': script_name})