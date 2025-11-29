"""
Module that provides plotting,
pre-layouted figures and
import/export of artefacts.
"""
import os
script_name = os.path.basename(__file__)

import git as git
import numpy as np
import numpy.ma as ma
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer
import scipy.stats as stats
import yaml as yaml
from PIL import Image, PngImagePlugin # Use PIL to save some image metadata
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import (TextArea, AnnotationBbox)
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages

import lib.utilities as ut
from lib.plotting.style import *

class Artefacts:
    """
    Class to deal with artefacts and save measurements in order to re-evaluate the data later
    or load an old experiment if needed.
    Saves parameters, measurement record (results) as json-files and plots as figures in various formats.
    """
    def __init__(self):
        self.figures = []
        pass

    def add_figures(self, figures, prefixes):
        """
        Add figures one by one to list of figures with prefixes and close them.
        """
        t = {'figures': np.array(list(zip(figures,prefixes)))}
        self.append(t)
        #for f,p in zip(figures,prefixes):
        #    self.figures.append([f,p])
        #    plt.close(f)
        pass

    def save_figures(self, meta={}, out_dir=None):
        """
        Save all present figures and empty list.
        """
        for fig in self.figures:
            try:
                Figures().save_fig(fig[0], short_title_str=fig[1], meta=meta, out_path=out_dir)
            except:
                print(f'Failed to save figure {fig[1]}')
        self.figures = []
        pass

    def append(self,obj):
        """
        Method to merge to objects via merging their attributes.
        *attributes have to be lists!*
        :param obj1: first object
        :param obj2: second object to be appended
        :return: pass
        """
        if type(obj) is type(None):
            pass
        else:
            if isinstance(obj,dict):
                obj_dict = obj
            else:
                try:
                    obj_dict = obj.__dict__
                except:
                    raise Exception('Failed to append artefacts.')
            eigen_dict = self.__dict__
            for key in obj_dict:
                size = getattr(obj_dict[key], "size", None)
                if size is None:
                    try:
                        size = len(obj_dict[key])
                    except:
                        continue
                    if size==0:
                        continue
                elif not key in eigen_dict.keys() or len(eigen_dict[key])==0:
                    eigen_dict[key] = obj_dict[key]
                else:
                    eigen_dict[key] = np.concatenate((eigen_dict[key],obj_dict[key]))
                setattr(self, key, eigen_dict[key])
        pass   


class Figures:
    """
    Class that provides some routines to create figures.
    """

    def __init__(self)->None:
        pass

    def save_fig(self, fig, short_title_str, meta={}, out_path=None,make_pdf=False):
        """
        Method to time-stamp, name and save plots.
        :param fig: figure object to be saved.
        :param short_title_str: string used to create label.
        :param meta: optional dictionary of meta-data
        :param out_path: string, full path to output destination, *without* file extension.
        """
        if not isinstance(fig, plt.Figure):
            try:
                fig = fig.figure
            except:
                print(f'Cannot access figure of {short_title_str}')
            
        file_str, todays_dir= ut.Labeler().stamp(short_title_str)
        if out_path is None:
            out_path = todays_dir + file_str
        else:
            out_path = os.path.join(out_path,file_str)
        repo = git.Repo(search_parent_directories=True) # extract the git hash of the current build, to put it into the plots
        sha = repo.head.object.hexsha
        METADATA = {'GitHash': f'{sha}', 'Author' : 'Thomas Arne Hensel'} | meta # join two dicts
        fig.savefig(out_path + '.png', dpi=300, bbox_inches='tight')
        im = Image.open(out_path + '.png')
        meta = PngImagePlugin.PngInfo()
        for x in METADATA:
            meta.add_text(x, METADATA[x])
        im.save(out_path + '.png', "png", pnginfo=meta)
        if make_pdf:
            try:
                with PdfPages(out_path + '.pdf') as pdf:
                    d = pdf.infodict() # set the file's metadata via the PdfPages object:
                    d['Title'] = f'{sha}'
                    d['Author'] = 'Thomas Arne Hensel'
                    #d['Subject'] = 
                    #d['Keywords'] = param_str
                    #pdf.attach_note(param_str)
                    pdf.savefig(fig)
            except:
                print(f'could not print pdf of {short_title_str}')
        plt.close(fig)
        return out_path + '.png'
    
    def _get_std_layout_single(self,suptitle):
        fig, axs = plt.subplot_mosaic(
                [
                    ["1"]
                ]
            ,empty_sentinel="BLANK"
            ,gridspec_kw = {
                "width_ratios": [1.]
                ,"height_ratios": [1.]
                }
            ,constrained_layout=True
            ,figsize=(8,8))#width x height
        fig.suptitle(suptitle,fontsize=20)
        return fig, axs
    
class CPSegmentation_Figures(Figures):
    """Class to generate diagnostic figures for the classical changepoint segmentation
    """

    def __init__(self):
        self.s = Style()
        pass

    def fig_diagnostics(self, segmented_df, filtered_df):
        figures = []
        names = []

        # for diagnostics: visualize removed outliers
        merged_df = segmented_df.merge(filtered_df, how='outer', indicator=True)
        unshared_rows = merged_df[merged_df['_merge'] != 'both'].drop(columns=['_merge'])
        outlier_df = unshared_rows.groupby('tuple', group_keys=False).mean().reset_index()

        #unshared_rows = merged_df[merged_df['_merge'] != 'both'].drop(columns=['_merge'])
        #unshared_tuples = unshared_rows['tuple'].unique()
        #shared_rows_with_respect_to_tuple = merged_df[merged_df['tuple'].isin(unshared_tuples) & (merged_df['_merge'] == 'both')]


        data = segmented_df.groupby('tuple', group_keys=False).mean().reset_index()
        segment_ids = data.segment_id.unique().astype(int)
        X = data.photons.to_numpy().reshape(-1,1)

        # plot model states over time
        fig, ax = plt.subplots()
        grouped_df = segmented_df.groupby('tuple',group_keys=False).mean().reset_index()
        for seg_id in segment_ids:
            tmp_df = grouped_df.loc[grouped_df['segment_id'].isin([seg_id])]
            ax.plot(tmp_df.tuple,tmp_df.photons, ".-", ms=6, mfc="orange",alpha=.5,label=f'segment: {seg_id}')
        ax.scatter(outlier_df.tuple,outlier_df.photons,marker='x',s=100,c='r',label='outliers')
        ax.set_title('Trace segmented via CPD')
        ax.set_xlabel('Tuple Index')
        ax.set_ylabel('Counts')
        ax.legend()
        figures.append(fig)
        names.append('segmented-trace')

        data = filtered_df.groupby(['axis','tuple'], group_keys=False).mean('tuple').reset_index()
        data['photons'] = data.groupby(['axis'], group_keys=False).rolling(window=50).mean().photons.reset_index(drop=True)
        fig, ax = plt.subplots()
        [ax.plot(data.loc[np.where(data.axis==axs)[0]].tuple,data.loc[np.where(data.axis==axs)[0]].photons, ".-", ms=6, mfc="orange",alpha=.5) for axs in np.unique(data.axis)]
        ax.set_ylim(0,50)#np.nanmax(data.photons)+1)
        ax.set_title('Trace segmented via CPD')
        ax.set_xlabel('Tuple Index')
        ax.set_ylabel('Counts')
        ax.legend()
        figures.append(fig)
        names.append('average-trace')

        # get count histogram
        bins = np.asarray(sorted(np.unique(X.flatten())))
        fig, ax = plt.subplots()
        bins = np.arange(int(min(bins)),int(max(bins))+1)  # Calculate bin centers
        ax.hist(X, bins=bins, density=True, alpha=0.5, label='Data')
        ax.set_title('Histogram of Counts')
        ax.set_xlabel('X')
        ax.set_ylabel('Probability')
        figures.append(fig)
        names.append('count_histogram')

        # save plot of photons in last iteration
        segments = list(map(int,set(segmented_df['segment_id'])))
        space = np.linspace(0, 1, 2*len(segments), endpoint=True)
        colors = [self.s.cmap_seq(x) for x in space]
        fig, ax =plt.subplots(1,len(segments)+1,figsize=(5*(len(segments)),5))
        for i,segment_id in enumerate(segments):
            tmp_df = segmented_df.loc[segmented_df['segment_id']==segment_id]
            tmp_df2 = filtered_df.loc[filtered_df['segment_id']==segment_id]
            ax[i].scatter(tmp_df.loc[tmp_df['axis']==0]['pos'],tmp_df.loc[tmp_df['axis']==0]['photons'],marker='x',s=1,alpha=0.6,color=colors[int(i)],label='x raw')
            ax[i].scatter(tmp_df.loc[tmp_df['axis']==1]['pos'],tmp_df.loc[tmp_df['axis']==1]['photons'],marker='x',s=1,alpha=0.6,color=colors[-max(int(i),1)],label='y raw')
            ax[i].scatter(tmp_df2.loc[tmp_df2['axis']==0]['pos'],tmp_df2.loc[tmp_df2['axis']==0]['photons'],marker='o',s=1,alpha=1,color=colors[-max(int(i),1)],label='x filtered')
            ax[i].scatter(tmp_df2.loc[tmp_df2['axis']==1]['pos'],tmp_df2.loc[tmp_df2['axis']==1]['photons'],marker='o',s=1,alpha=1,color=colors[int(i)],label='y filtered')
            ax[i].legend()
        fig.delaxes(ax[-1])
        figures.append(fig)
        names.append('counts-in-segments')
        return figures, names


class Residual_Figures(Figures):
    """Figures to visualize residuals of fit. Until now only wrt to full model (harmonic)
    """
    def __init__(self):
        self.s = Style()
        Figures.__init__(self)
        pass

    def fig_check_residuals(self, data_arrays, fit_arrays, res_arrays):
        """
        Create a figure for the residuals.
        :param max_lines: number of lines to be plotted in each axis.
        """
        fig, axs = plt.subplot_mosaic(
            [
                ["resX",'resY','CB']
                ,['avgX','avgY','BLANK']
                ,['dataX','dataY','BLANK']
            ]
        ,empty_sentinel="BLANK"
        ,gridspec_kw = {
            "width_ratios": [1,1,.1]
            ,"height_ratios": [1,.5,.5]
            }
        ,constrained_layout=False
        ,figsize=(7.086,7.086)
        )
        fig.suptitle("Residuals of fit",fontsize=10)
        axs['resX'].pcolormesh(res_arrays[0],cmap=self.s.cmap_seq,linewidth=0,rasterized=True)
        axs['resX'].set_ylabel('t (lines)')
        axs['resX'].tick_params(top=True, labeltop=False, bottom=True, labelbottom=False,left=True,labelleft=True,right=True,labelright=False,direction='in')
        im = axs['resY'].pcolormesh(res_arrays[1],cmap=self.s.cmap_seq,linewidth=0,rasterized=True)
        axs['resY'].tick_params(top=True, labeltop=False, bottom=True, labelbottom=False,left=True,labelleft=False,right=True,labelright=True,direction='in')
        axs['resY'].sharey(axs['resX'])
        axs['avgX'].plot(np.nanmean(res_arrays[0],axis=0),c=self.s.c20,label=r'$\bar{y}_{res}$')
        axs['avgX'].tick_params(top=True, labeltop=False, bottom=True, labelbottom=False,left=True,labelleft=True,right=True,labelright=False,direction='in')
        axs['avgX'].legend()
        axs['avgY'].plot(np.nanmean(res_arrays[1],axis=0),c=self.s.c20,label=r'$\bar{y}_{res}$')
        axs['avgY'].tick_params(top=True, labeltop=False, bottom=True, labelbottom=False,left=True,labelleft=False,right=True,labelright=True,direction='in')
        axs['avgY'].legend()
        axs['dataX'].plot(np.nanmean(data_arrays[0],axis=0),c=self.s.c20,label=r'$\bar{y}$')
        axs['dataX'].plot(np.nanmedian(fit_arrays[0],axis=0),c=self.s.c10,label=r'$\bar{y}_{fit}$')
        axs['dataX'].tick_params(top=True, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=True,labelright=False,direction='in')
        axs['dataX'].legend()
        
        axs['dataY'].plot(np.nanmean(data_arrays[1],axis=0),c=self.s.c20,label=r'$\bar{y}$')
        axs['dataY'].plot(np.nanmedian(fit_arrays[1],axis=0),c=self.s.c10,label=r'$\bar{y}_{fit}$')
        axs['dataY'].tick_params(top=True, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=False,right=True,labelright=True,direction='in')
        axs['dataY'].legend()
        fig.colorbar(im, cax=axs['CB'])
        for label, a in axs.items(): # label the panels from a) to f)

            if label != 'CB':
                a.sharex(axs['avgX'])
        return fig, axs

class Minflux_Figures(Figures):
    
    def __init__(self):
        self.s = Style()
        Figures.__init__(self)
        pass

    def fig_visibility(self, df):
        """
        Method to visualize the analysis of a dataset in terms of kappa related to quality of minimum.
        """
        figures = []
        names = []

        try:

            groups = df.groupby(['chunk_size'])

            for group in groups:
                method = group[0]
                data = group[1]
                name = f'{method}-minimum-quality'
                
                # create layout
                name = f"{method}_minimum_quality"
                fig, axs = plt.subplot_mosaic(
                        [
                            ["hist_1M","hist_2M"]
                            ,["scatter_1M","scatter_2M"]
                        ]
                    ,empty_sentinel="BLANK"
                    ,gridspec_kw = {
                        "width_ratios": [1.,1.]
                        ,"height_ratios": [1.,1.]
                        }
                    ,constrained_layout=True
                    ,figsize=(7.068,4.))#width x height
                fig.suptitle('Quality of minimum',fontsize=20)

                data.loc[data['state_id']==1,'visibility_delta'] = data['v'] - 1
                data.loc[data['state_id']==2,'visibility_delta'] = data['v'] - data['v0']

                if not data.loc[data['state_id']==1].empty:
                    sns.barplot(data=data.loc[data['state_id']==1], x="label", y="v", hue="axis",ax=axs['hist_1M'],palette={0:self.s.c10,1:self.s.c20})
                    sns.stripplot(
                        data=data.loc[data['state_id']==1], x="label", y="visibility_delta", hue="axis",
                        dodge=True, alpha=.25, zorder=1, legend=False,ax=axs['scatter_1M'],palette={0:self.s.c10,1:self.s.c20}
                    )
                    sns.pointplot(
                        data=data.loc[data['state_id']==1], x="label", y="visibility_delta", hue="axis", estimator='median', errorbar=None,
                        join=False, dodge=.8 - .8 / 2, markers="d", scale=.5, ax=axs['scatter_1M'],palette={0:self.s.c11,1:self.s.c21}
                    )
                    
                if not data.loc[data['state_id']==2].empty:
                    sns.barplot(data=data.loc[data['state_id']==2], x="label", y="v", hue="axis",ax=axs['hist_2M'],palette={0:self.s.c10,1:self.s.c20})
                    sns.stripplot(
                        data=data.loc[data['state_id']==2], x="label", y="visibility_delta", hue="axis",
                        dodge=True, alpha=.25, zorder=1, legend=False,ax=axs['scatter_2M'],palette={0:self.s.c10,1:self.s.c20}
                    )
                    sns.pointplot(
                        data=data.loc[data['state_id']==2], x="label", y="visibility_delta", hue="axis", estimator='median', errorbar=None,
                        join=False, dodge=1 - 1. / 2, markers="d", scale=.5,ax=axs['scatter_2M'],palette={0:self.s.c11,1:self.s.c21}
                    )
                axs['hist_1M'].set_ylim(.95,1.)
                axs['hist_2M'].set_ylim(.95,1.)

                axs['hist_1M'].set_ylabel(r'$\nu_{1M}$ (1)')
                axs['hist_2M'].set_ylabel(r'$\nu_{2M}$ (1)')
                axs['scatter_1M'].set_ylabel(r'$\nu_{1M}-1$ (1)')
                axs['scatter_2M'].set_ylabel(r'$\nu_{2M}-\nu_{1M}$ (1)')
                axs['scatter_1M'].set_ylim(-0.1,0.01)
                axs['scatter_2M'].set_ylim(-0.1,0.01)

                for ax in axs.values():
                    ax.legend(fontsize=7)
                    ax.set_xlabel(r'ground truth (nm)')

                figures.append(fig)
                names.append(name)
                plt.close(fig)
        except:
            pass
        return figures, names
    
    def fig_count_histogram(self, df):
        """
        Method to visualize the analysis of a dataset in terms of kappa related to quality of minimum.
        """
        figures = []
        names = []

        try:
            groups = df.groupby(['chunk_size'])

            for group in groups:
                method = group[0]
                data = group[1]
                name = f'{method}-count-histogram'

                grid = sns.displot(data, x="N_avg", col="label", row="axis",hue='state_id',binwidth=1, height=3, facet_kws=dict(margin_titles=True))
                axes = grid.axes.flatten()
                for ax in axes:
                    ax.set_xlim(0,40)

                figures.append(grid.figure)
                names.append(name)
        except:
            pass
        return figures, names
    
    def fig_distance_histogram(self, df):
        """
        Method to visualize the analysis of a dataset in terms of kappa related to quality of minimum.
        """
        figures = []
        names = []

        try:
            groups = df.groupby(['chunk_size'])

            for group in groups:
                method = group[0]
                data = group[1]
                
                name = f'{method}-distance-axis-histogram'
                try:
                    grid = sns.displot(data, x="d", col="label",hue='axis',binwidth=.5, height=3, facet_kws=dict(margin_titles=True))
                    axes = grid.axes.flatten()
                    for ax in axes:
                        col_name = float(ax.title._text.split('=')[-1].strip())
                        for hue_name, sub_data in data.groupby(["axis"]):
                            filtered_data = sub_data[sub_data["label"] == col_name]
                            median = filtered_data["d"].median()  # Calculate median for each group
                            mean = filtered_data["d"].mean() 
                            color = sns.color_palette()[int(hue_name[0])]
                            ax.axvline(median, color=color, linestyle='--', label=f'Median: {median:.1f}')
                            ax.axvline(mean, color=color, linestyle='-.', label=f'Mean: {mean:.1f}')
                            ax.set_xlim(0,100)
                        ax.legend()
                    figures.append(grid.figure)
                    names.append(name)
                except:
                    print('Could not produce distance histogram wrt axis')

                try:
                    name = f'{method}-distance-norm-histogram'
                    grid = sns.displot(data.loc[data['axis']==0], x="d_norm", col="label",col_wrap=3,binwidth=.5, height=3, facet_kws=dict(margin_titles=True))
                    axes = grid.axes.flatten()
                    for ax in axes:
                        col_name = float(ax.title._text.split('=')[-1].strip())
                        hues = data.loc[data['axis']==0,'estimator'].nunique()
                        for hue_idx, (hue_name, sub_data) in enumerate(data.loc[data['axis']==0].groupby(["estimator"])):
                            filtered_data = sub_data.loc[sub_data["label"] == col_name]
                            median = filtered_data["d_norm"].median()  # Calculate median for each group
                            mean = filtered_data["d_norm"].mean() 
                            idx = (hue_idx+1)/(hues+1) * len(sns.color_palette())
                            color = sns.color_palette()[int(idx)]
                            ax.axvline(median, color=color, linestyle='--', label=f'Median: {median:.1f}')
                            ax.axvline(mean, color=color, linestyle='-.', label=f'Mean: {mean:.1f}')
                            ax.set_xlim(0,100)
                        ax.legend()
                    figures.append(grid.figure)
                    names.append(name)
                except:
                    print('Could not produce distance norm histogram')
        except:
            pass
        return figures, names
    
    def fig_distance_precision(self, df):
        """
        Method to visualize the precision of distance estimates.
        """
        figures = []
        names = []

        try:
            name = f'distance-precision'
            fig, ax = plt.subplot_mosaic(
                [
                    ["a","b"]
                ]
            ,empty_sentinel="BLANK"
            ,gridspec_kw = {
                "width_ratios": [1,1]
                ,"height_ratios": [1]
                }
            ,constrained_layout=True
            ,figsize=self.s.get_figsize(cols = 2, rows=1,ratio=.8)#(7.086,6.)
            )

            sns.boxplot(x="label", y="d_norm",hue="chunk_size",data=df,ax=ax['a'])
            statistics_df = df.groupby(['label','chunk_size']).d_norm.describe().loc[:,['mean','std']].reset_index()
            sns.pointplot(data=statistics_df, x='chunk_size', y='std', hue='label', ax=ax['b'],palette="dark",markers="d",join=False)
            sns.despine(offset=10, trim=True)
            
            figures.append(fig)
            names.append(name)
        except:
            print('could not produce figure for precision')
        return figures, names
    
    def fig_COM_movement(self,df):
        figures = []
        names = []
        try:
            groups = df.groupby(['trace'])
            for group in groups:
                fig, ax = plt.subplots()
                method = group[0]
                data = group[1]
                name = f'com'

                tuple_means = data.groupby(['tuple','axis']).mean().reset_index()
                pivoted = tuple_means.pivot(index='tuple', columns='axis', values=['pos'])
                pivoted.reset_index(inplace=True)
                pivoted.columns = ['tuple', 'posx', 'posy']
                window_size = 50  # Adjust the window size as needed
                averaged = pivoted.rolling(window=window_size,on='tuple').mean()

                sns.scatterplot(pivoted,x='posx', y='posy',ax=ax, s=2, alpha=0.7)#,color=s.c10
                sns.scatterplot(averaged,x='posx', y='posy',ax=ax, s=3, alpha=1.)
                
                figures.append(fig)
                names.append(name)
        except:
            pass
        return figures, names
    
    def fig_tracking(self,df):
        figures = []
        names = []
        try:
            groups = df.groupby(['file','trace'])
            for group in groups:
                fig, ax = plt.subplots()
                method = group[0]
                data = group[1]
                name = f'com'
                sns.scatterplot(x=data.loc[data['axis']==0,'x0'].to_numpy(), y=data.loc[data['axis']==1,'x0'].to_numpy())
                figures.append(fig)
                names.append(name)

        except:
            pass
        return figures, names
    
    def fig_distance_wrt_time(self,df):
        plt.close()
        figures = []
        names = []
        try:
            groups = df.groupby(['chunk_size'],group_keys=False)

            for group in groups:
                method = group[0]
                data = group[1]
                name = f'{method}-distance-wrt-time'

                sorted_df = data.groupby(['gt','batch','file'],group_keys=False).apply(lambda group: group.sort_values(['state_id','chunk_id'])).copy()
                sorted_df = sorted_df.groupby(['file','chunk_id']).mean('d_norm').reset_index()
                fig, ax = plt.subplots()
                sns.lineplot(data=sorted_df,x=sorted_df.index, y="d_norm", hue="label",markers=True, dashes=False, ax=ax)

                indexer = FixedForwardWindowIndexer(window_size=10)
                for file, file_group in sorted_df.groupby('file'):
                    start_index = file_group.index.min()
                    end_index = file_group.index.max()
                    ax.axvline(x=start_index, color='k', linestyle='--')
                    ax.axvline(x=end_index, color='r', linestyle='--')
                    rolling_mean = file_group['d_norm'].rolling(indexer,min_periods=1).mean()
                    sns.lineplot(data=rolling_mean, x=file_group.index, y=rolling_mean, color="red", linewidth=2, ax=ax)
                figures.append(fig)
                names.append(name)
        except:
            pass
        return figures, names
    
    def fig_pos_brightness(self, df):
        plt.close()
        figures = []
        names = []
        x, y, zx, zy = df.loc[df.axis==0, 'x0'], df.loc[df.axis==1, 'x0'], df.loc[df.axis==0, 'N_avg'], df.loc[df.axis==1, 'N_avg']#df.groupby(df.index // 2).mean('N_avg')['N_avg']

        name = f'{df.method[0]}-counts-wrt-position'
        fig, ax = plt.subplot_mosaic(
        [
            ['a','b','c']
            ,['d','e','f']
        ]#grid
        ,empty_sentinel="BLANK"
        ,gridspec_kw = {
            "width_ratios": [1,1,1]#widths
            ,"height_ratios": [1,1]#heights
            }
        ,constrained_layout=True
        ,figsize=self.s.get_figsize(rows=2,cols=2,ratio=1)#
        )

        ax['a'].tricontourf(x, y, zx, cmap='coolwarm')
        ax['b'].tricontourf(x, zx, y, cmap='coolwarm')
        ax['c'].tricontourf(y, zx, x, cmap='coolwarm')

        ax['d'].tricontourf(x, y, zy, cmap='coolwarm')
        ax['e'].tricontourf(x, zy, y, cmap='coolwarm')
        ax['f'].tricontourf(y, zy, x, cmap='coolwarm')

        
        figures.append(fig)
        names.append(name)
        return figures, names

class Meta_Analysis_Figures(Figures):
    """Figures for meta analysis on all analyzed data, e.g. statistics over batches etc.
    """
    def __init__(self):
        Figures.__init__(self)
        self.s = Style()
        pass

    def fig_correlations(self, df):
        """Method to plot correlations between several methods.
        
        If no method is provided (default), only method intern correlations are sought for,
        i.e. between different measured quantities.
        """
        figures = []
        names = []
        
        groups = df.groupby('method')

        for group in groups:
            method = group[0]
            data = group[1]
            name = f'{method}-correlations'
            
            # create layout
            fig, axs = plt.subplot_mosaic(
                    [
                        ['d_vs_NALM','dxy_vs_NALM']
                    ]
                ,empty_sentinel="BLANK"
                ,gridspec_kw = {
                    "width_ratios": [1.,1.]
                    ,"height_ratios": [1.]
                    }
                ,constrained_layout=True
                ,figsize=(10,5))#width x height
            fig.suptitle(f"{method} distance correlations",fontsize=20)

            reduced_df = data.drop(columns=['kappa_1M','kappa_2M','visibility_1M','visibility_2M','N_fit_2M','N_tot_2M','chi2_weights','file','avg_sig'])
            reshaped_df = reduced_df.explode(['distances','d_NALM']).reset_index(drop=True)
            reshaped_df['axis'] = reshaped_df.index % 2
            reshaped_df['distances'] = reshaped_df['distances'].apply(lambda x: np.nanmean(x))
            reshaped_df['d_norm'] = reshaped_df['d_norm'].apply(lambda x: np.nanmean(x))
            reshaped_df['N_fit'] = reshaped_df['N_fit'].apply(lambda x: np.nanmean(x))

            sns.scatterplot(x="d_norm_NALM", y="d_norm",
                size="N_fit",
                sizes=(1, 8), linewidth=0,
                data=reshaped_df.loc[reshaped_df.index%2==0], ax=axs['d_vs_NALM'])
            
            sns.scatterplot(x="d_NALM", y="distances",
                size="N_fit",
                hue='axis',
                palette="ch:r=-.2,d=.3_r",
                sizes=(1, 8), linewidth=0,
                data=reshaped_df, ax=axs['dxy_vs_NALM'])
           
            [ax.set_xlim(0,100) for ax in axs.values()]
            [ax.set_ylim(0,100) for ax in axs.values()]
            [ax.get_shared_x_axes().join(list(axs.values())[0], ax) for ax in axs.values()]
            [ax.get_shared_y_axes().join(list(axs.values())[0], ax) for ax in axs.values()]
            figures.append(fig)
            names.append(name)
            plt.close()
        return figures, names

    def fig_correlation_matrix(self,df,key):
        # Analyse cross-correlations between methods for different quantities:
        def plot_corr(dataframe, key):
            df = dataframe
            # Create a new dataframe with the unique labels as columns
            new_df = df.groupby('method')[key].apply(list).reset_index(name='aggregated_values')
            new_df.set_index('method', inplace=True)
            pivot_df = new_df['aggregated_values'].apply(pd.Series).transpose()
            # Compute the correlation matrix of the 'aggregated_values' column
            corr = pivot_df.corr()

            fig, ax = self._get_std_layout_single('Cross-correlations between methods')
            # Generate a mask for the upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))
            # Create a heatmap of the correlation matrix
            sns.heatmap(corr,
                mask=mask,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0, vmax=1.0,
                square=True, ax=ax['1'])
            return fig
            
        figures = []
        names = []

        fig = plot_corr(df, key)
        figures.append(fig)
        names.append(f'cross-correlations-of-{key}')
        plt.close(fig)     
        return figures, names


    def fig_individual_batch_histograms(self,df):
        """
        Create a figure for each analyzed batch with distance histogram, scattered points etc.
        returns a list of figures, one for each analyzed batch.
        """
        figures = []
        names = []

        def hexbin(x, y, color, **kwargs):
            cmap = sns.light_palette(color, as_cmap=True)
            plt.hexbin(x, y, gridsize=40, cmap=cmap, **kwargs)

        df.loc[:,'d_norm'] = df['d_norm'].apply(lambda x: np.nanmean(x))

        with sns.axes_style("dark"):
            g = sns.FacetGrid(df, hue="method", row="method", col='ground_truth', height=4)
        g.map(hexbin, "d_norm_NALM", "d_norm", extent=[0, 100, 0, 100])
        figures.append(g.figure)
        names.append("hexbin_batch_histograms")

        g = sns.displot(
            df, x="d_norm", col="ground_truth", col_wrap=4,
            binwidth=1, height=3, facet_kws=dict(margin_titles=True),
        )
        figures.append(g.figure)
        names.append('phase_scan_distance_norm_histograms')

        df.loc[:,'distances'] = df['distances'].transform(lambda x: np.asarray([np.nanmedian(axes_list) for axes_list in x]))
        df.loc[:,'chi2_weights'] = df['chi2_weights'].transform(lambda x: np.asarray([np.nanmedian(axes_list) for axes_list in x]))
        exploded_df = df.explode(['distances','chi2_weights'])
        exploded_df['axis'] = exploded_df.index%2
        exploded_df.reset_index(inplace=True)
        name = f"phase_scan_distance_histograms_wrt_axes"
        g = sns.displot(
            exploded_df, x="distances", col="ground_truth", hue='axis', col_wrap=4,
            binwidth=1, weights='chi2_weights',height=3, facet_kws=dict(margin_titles=True),
        )
        figures.append(g.figure)
        names.append(name)
        return figures, names

    def fig_power_balance(self, df):
        """
        Method to visualize the analysis of a dataset in terms of kappa related to quality of minimum.
        """
        figures = []
        names = []

        groups = df.groupby('method')

        for group in groups:
            method = group[0]
            data = group[1]
            name = f'{method}-minimum-quality'
            
            # create layout
            name = f"{method}_minimum_quality"
            fig, axs = plt.subplot_mosaic(
                    [
                        ["hist_1M","hist_2M"]
                        ,["scatter_1M","scatter_2M"]
                    ]
                ,empty_sentinel="BLANK"
                ,gridspec_kw = {
                    "width_ratios": [1.,1.]
                    ,"height_ratios": [1.,1.]
                    }
                ,constrained_layout=True
                ,figsize=(7.068,4.))#width x height
            fig.suptitle('Quality of minimum',fontsize=20)

            reduced_df = data.drop(columns=['kappa_1M','kappa_2M','d_norm','distances','d_norm_NALM','N_fit','N_fit_2M','N_tot_2M','chi2_weights','d_NALM','file','avg_sig'])
            reshaped_df = reduced_df.explode(['visibility_1M','visibility_2M']).reset_index(drop=True)
            reshaped_df['axis'] = reshaped_df.index % 2
            reshaped_df['visibility_1M'] = reshaped_df['visibility_1M'].apply(lambda x: np.nanmean(x))
            reshaped_df['visibility_2M'] = reshaped_df['visibility_2M'].apply(lambda x: np.nanmean(x))
            reshaped_df['visibility_delta_1M'] = reshaped_df['visibility_1M'] - 1.
            reshaped_df['visibility_delta_2M'] = reshaped_df['visibility_2M'] - reshaped_df['visibility_1M']
            # melt group for x and y axis
            sns.barplot(data=reshaped_df, x="ground_truth", y="visibility_1M", hue="axis",ax=axs['hist_1M'],palette={0:self.s.c10,1:self.s.c20})
            sns.barplot(data=reshaped_df, x="ground_truth", y="visibility_2M", hue="axis",ax=axs['hist_2M'],palette={0:self.s.c10,1:self.s.c20})
            
            
            #scatter x-y of 1M and 2M
            # Show each observation with a scatterplot
            sns.stripplot(
                data=reshaped_df, x="ground_truth", y="visibility_delta_1M", hue="axis",
                dodge=True, alpha=.25, zorder=1, legend=False,ax=axs['scatter_1M'],palette={0:self.s.c10,1:self.s.c20}
            )
            # Show the conditional means, aligning each pointplot in the
            # center of the strips by adjusting the width allotted to each
            # category (.8 by default) by the number of hue levels
            sns.pointplot(
                data=reshaped_df, x="ground_truth", y="visibility_delta_1M", hue="axis",
                join=False, dodge=.8 - .8 / 2,
                markers="d", scale=.75, errorbar=None,estimator='median',ax=axs['scatter_1M'],palette={0:self.s.c11,1:self.s.c21}
            )

            sns.stripplot(
                data=reshaped_df, x="ground_truth", y="visibility_delta_2M", hue="axis",
                dodge=True, alpha=.25, zorder=1, legend=False,ax=axs['scatter_2M'],palette={0:self.s.c10,1:self.s.c20}
            )
            sns.pointplot(
                data=reshaped_df, x="ground_truth", y="visibility_delta_2M", hue="axis",
                join=False, dodge=1 - 1. / 2,
                markers="d", scale=.75, errorbar=None,estimator='median',ax=axs['scatter_2M'],palette={0:self.s.c11,1:self.s.c21}
            )

            axs['hist_1M'].set_ylabel(r'$\nu_{1M}$ (1)')
            axs['hist_1M'].set_ylim(.95,1)
            axs['hist_2M'].set_ylabel(r'$\nu_{2M}$ (1)')
            axs['hist_2M'].set_ylim(.8,1)
            axs['scatter_1M'].set_ylabel(r'$\nu_{1M}-1$ (1)')
            axs['scatter_1M'].set_ylim(-.2,.1)
            axs['scatter_2M'].set_ylabel(r'$\nu_{2M}-\nu_{1M}$ (1)')
            axs['scatter_2M'].set_ylim(-.2,.1)

            figures.append(fig)
            names.append(name)
            plt.close(fig)
        return figures, names
    
    def fig_estimated_distances(self, df):
        """
        Method to visualize the analysis of a dataset in terms of estimated distances vs ground truth.
        """
        figures = []
        names = []
        
        groups = df.groupby('method')

        for group in groups:
            method = group[0]
            data = group[1]
            name = f'{method}-distance-estimates'

            fig, ax = self._get_std_layout_single("Estimated distance vs ground truth")
            
            flierprops = dict(marker='.', markerfacecolor='k', markersize=3,
                  markeredgecolor='none')
            boxprops = dict(facecolor=self.s.c10,edgecolor=self.s.c10,alpha=.8)
            medianprops = dict(color = "k", linewidth = 1.,alpha=.8)
            whiskerprops = dict(color=self.s.c10,alpha=.8)
            capprops = dict(color=self.s.c10,alpha=.8)

            filtered_df = data.loc[~data['d_norm'].apply(lambda x: np.isnan(x).any())]
            grouped_mean_data = filtered_df.groupby('ground_truth')['d_norm'].agg(list).apply(lambda x:x)
            mean = grouped_mean_data.to_list()
            grouped_pos_data = filtered_df.groupby('ground_truth')['ground_truth'].agg(list).apply(lambda x: int(np.nanmean(x)))
            pos = grouped_pos_data.to_list()

            try:
                assert len(mean)==len(pos)
                assert len(mean)>0
                boxplot = ax['1'].boxplot(mean, positions=pos, notch=False, widths=1.5, showmeans=False, meanline=True,patch_artist=True, boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops, manage_ticks=False)
            except:
                pass
            ax['1'].plot([], [], color=self.s.c10, label='Minimum')
            ax['1'].set_xlim(0,100)
            ax['1'].set_ylim(0,100)
            lims = ax['1'].get_xlim()
            #ax['c2)'].set_aspect('equal', 'box')
            ax['1'].plot(np.linspace(lims[0],lims[1],5),np.linspace(lims[0],lims[1],5),label='Expected',color='k',alpha=.5,zorder=-1)
            ax['1'].set_xticks(np.arange(0,101,10))
            ax['1'].set_xlabel('Expected [nm]')
            ax['1'].set_ylabel('Estimated [nm]')
            ax['1'].tick_params(top=True, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=True,labelright=False,direction='in')
            ax['1'].legend(bbox_to_anchor=(1., -.05), loc='lower right', title=method)
            
            figures.append(fig)
            names.append(name)
            plt.close(fig)
        return figures, names

class Filter_Figures(Figures):
    """Figures related to the filtering procedure"""
    def __init__(self):
        Figures.__init__(self)
        self.s = Style()
        pass

    def fig_filter_statistics(self,reasons):
        counts = Counter(reasons)
        values = list(counts.values())
        labels = list(counts.keys())
        fig, ax = plt.subplots()
        fig.suptitle('Reasons to reject a trace')
        ax.bar(labels, values, .5, color='g')
        return fig, ax

class NALM_Figures(Figures):
    """
    Class that contains the figures for the NALM analysis
    """
    def __init__(self):
        Figures.__init__(self)
        self.s = Style()
        pass

    def fig_NALM_analysis(self, result_dict, imarray):
        # pad/truncate imarray if necessary
        #pre-calculations:
        line_avg = np.average(imarray,axis=2) #line average of trace

        min_pos = result_dict['min_idx'] #position of minimum
        N_bgr = result_dict['N_bgr'] #average background counts
        min_depth = result_dict['min_depth'] #parabel offset for each line
        dist_est = result_dict['distances']  # distance-estimates for each line in each axis
        d_avg, d_std = np.nanmean(dist_est,axis=1),np.nanstd(dist_est,axis=1) #weighted average distance and weighted average standard deviation
        com = result_dict['com_2M'] #center of mass of 2M trace
        d_NALM = result_dict['d_NALM'] #NALM distance
        curvature = result_dict['curvature'] # line-wise curvature
        median_curvature = result_dict['median_curvature'] # median curvature for each axis and each segment.
        
        # create actual figure:
        fig, axs = plt.subplot_mosaic(
                [
                    ["x_im","y_im","pos"]
                    ,["x_counts","y_counts","curv"]
                    ,["x_min_depth","y_min_depth","curv"]
                    ,["dx","dy","curv"]
                    ,['dist', 'dist', 'dist']
                ]
            ,empty_sentinel="BLANK"
            ,gridspec_kw = {
                "width_ratios": [1.,1.,1.]
                ,"height_ratios": [3.,1.,1.,1.,1.]
                }
            ,constrained_layout=True
            ,figsize=(7,10))#width x height
        #fig.suptitle("NALM analysis of phase-scan")

        # X-PLOTs
        #axs["x_im"].set_title('x trace')
        x = np.arange(1,len(min_pos[0])+1)
        X, Y = np.meshgrid(x, np.linspace(0,160, imarray[0].shape[-1]))
        axs["x_im"].set_title('x-axis')

        axs["x_im"].pcolormesh(X,Y, imarray[0].T,cmap=self.s.cmap_seq,linewidth=0,rasterized=True)
        axs["x_im"].plot(x, min_pos[0],c=self.s.c10)
        axs["x_im"].set_ylabel('pixel [2 nm]')
        axs['x_im'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=False,left=True,labelleft=True,right=False,labelright=False,direction='in')
        axs["x_im"].set_xlim(min(x),max(x))

        axs["x_counts"].plot(range(1,len(line_avg[0])+1),line_avg[0],c=self.s.c20)
        axs["x_counts"].axhline(N_bgr[0],ls='--',c='gray')
        axs["x_counts"].sharex(axs['x_im'])
        axs["x_counts"].set_ylabel(r'$\bar{N}$ [1]')
        axs['x_counts'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=False,left=True,labelleft=True,right=False,labelright=False,direction='in')
        
        axs["x_min_depth"].plot(range(1,len(min_depth[0])+1),min_depth[0],c=self.s.c20)
        axs["x_min_depth"].axhline(N_bgr[0],ls='--',c='gray')
        axs["x_min_depth"].sharex(axs['x_im'])
        axs["x_min_depth"].set_ylabel(r'$N_0$ [N]')
        axs['x_min_depth'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=False,left=True,labelleft=True,right=False,labelright=False,direction='in')
        
        axs["dx"].plot(range(1,len(dist_est[0])+1), dist_est[0],c=self.s.c20)
        axs['dx'].axhline(d_avg[0],c=self.s.c10,ls='solid')
        axs['dx'].axhline((d_avg+d_std)[0],c=self.s.c10,alpha=0.3,ls='solid')
        axs['dx'].axhline((d_avg-d_std)[0],c=self.s.c10,alpha=0.3,ls='solid')
        axs["dx"].sharex(axs['x_im'])
        axs["dx"].set_ylabel(r'$\delta x$ [nm]')
        axs["dx"].set_xlabel('line No [1]')
        axs['dx'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
        
        # Y-PLOTs
        axs["y_im"].set_title('y-axis')
        axs["y_im"].pcolormesh(X,Y, imarray[1].T,cmap=self.s.cmap_seq,linewidth=0,rasterized=True)
        axs["y_im"].plot(np.arange(1,len(min_pos[1])+1), min_pos[1],c=self.s.c10) # convert min pos from lateral offset [nm] to pixel
        axs["y_im"].sharey(axs['x_im'])
        axs['y_im'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=False,left=True,labelleft=False,right=False,labelright=False,direction='in')
        axs["y_im"].set_xlim(min(x),max(x))

        axs["y_counts"].plot(range(1,len(line_avg[1])+1),line_avg[1],c=self.s.c20)
        axs["y_counts"].axhline(N_bgr[1],ls='--',c='gray')
        axs["y_counts"].sharex(axs['x_im'])
        axs["y_counts"].sharey(axs['x_counts'])
        axs['y_counts'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=False,left=True,labelleft=False,right=False,labelright=False,direction='in')
        
        axs["y_min_depth"].plot(range(1,len(min_depth[1])+1),min_depth[1],c=self.s.c20)
        axs["y_min_depth"].axhline(N_bgr[1],ls='--',c='gray')
        axs["y_min_depth"].sharex(axs['y_im'])
        axs["y_min_depth"].sharey(axs['x_min_depth'])
        axs['y_min_depth'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=False,left=True,labelleft=False,right=False,labelright=False,direction='in')
        
        axs["dy"].plot(range(1,len(dist_est[1])+1),dist_est[1],c=self.s.c20)
        axs['dy'].axhline(d_avg[1],c=self.s.c10,ls='solid')
        axs['dy'].axhline((d_avg + d_std)[1],c=self.s.c10, alpha=0.3,ls='solid')
        axs['dy'].axhline((d_avg - d_std)[1],c=self.s.c10, alpha=0.3,ls='solid')
        axs["dy"].sharex(axs['y_im'])
        axs["dy"].sharey(axs['dx'])
        axs["dy"].set_xlabel('line No [1]')
        axs['dy'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=False,right=False,labelright=False,direction='in')

        # POSITION and CURVATURE
        # Define lines for rectangles
        offset = com
        width = d_avg
        x_min, y_min = (offset-width/4)
        x_max, y_max = (offset+width/4)
        dx, dy = d_std

        x_outer = [x_min-dx, x_min-dx, x_max+dx, x_max+dx, x_min-dx]
        y_outer = [y_min-dy, y_max+dy, y_max+dy, y_min-dy, y_min-dy]
        x_middle = [x_min, x_min, x_max, x_max, x_min]
        y_middle = [y_min, y_max, y_max, y_min, y_min]
        x_inner = [x_min+dx, x_min+dx, x_max-dx, x_max-dx, x_min+dx]
        y_inner = [y_min+dy, y_max-dy, y_max-dy, y_min+dy, y_min+dy]

        d = np.linalg.norm(dist_est,axis=0)
        d_nalm = np.linalg.norm(d_NALM)
        med_dist_est = np.nanmedian(d)

        # Add NALM rectangle
        axs['pos'].plot(x_inner, y_inner,c=self.s.c10,alpha=0.3,ls='solid')
        axs['pos'].plot(x_middle, y_middle,c=self.s.c10,ls='solid',label=r'd$_{NALM}$'+f'={round(d_nalm,1)} nm\n')
        axs['pos'].plot(x_outer, y_outer,c=self.s.c10,alpha=0.3,ls='solid')
        axs['pos'].scatter(min_pos[0],min_pos[1],label=r'd$_{min}$'+ f'={round(med_dist_est,1)} nm',alpha=.7) # center and scatter-plot positions at scan-origin

        axs['pos'].set_xlim(x_min-10, x_max+10)
        axs['pos'].set_ylim(y_min-10, y_max+10)
        axs["pos"].set_xlabel('x [pixel]')
        axs["pos"].set_ylabel('y [pixel]')
        axs['pos'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=False,right=True,labelright=True,direction='in')
        axs['pos'].legend()

        max_c = np.max(curvature)
        med_curv = median_curvature
        norm_curv = curvature/med_curv[1][:,np.newaxis]
        norm_med_curv = med_curv/med_curv[1][np.newaxis,:]
        axs['curv'].scatter(norm_curv[0], norm_curv[1],label='local curvature')
        axs['curv'].scatter(norm_med_curv.T[0],norm_med_curv.T[1],label='median curvature')
        axs['curv'].plot(np.linspace(0,2,2),np.linspace(0,2,2),color='k',label='line of equal\ncurvature')
        axs["curv"].set_xlabel(r'$\hat{\kappa}_x$ [1]')
        axs["curv"].set_ylabel(r'$\hat{\kappa}_y$ [1]')
        axs['curv'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=False,right=True,labelright=True,direction='in')
        axs['curv'].legend()
        
        
        axs["dist"].plot(range(1,len(d)+1),d)
        axs['dist'].axhline(med_dist_est,c=self.s.c10,ls='solid')
        axs['dist'].axhline(med_dist_est+np.nanstd(d),c=self.s.c10,ls='solid', alpha=0.3)
        axs['dist'].axhline(med_dist_est-np.nanstd(d),c=self.s.c10,ls='solid', alpha=0.3)
        axs["dist"].set_xlabel('line No. [1]')
        axs["dist"].set_ylabel(r'$\delta x$ [nm]')
        return fig, axs

class Trace_Figures(Figures):
    """
    Figures for visualization of traces.
    """
    def __init__(self):
        Figures.__init__(self)
        pass

    def fig_segmented_trace(self, line_avgs, seg_idcs):
        """Display line average and detected bleaching steps
        """
        fig, ax = plt.subplots(figsize=(7.086,3))
        fig.suptitle('Bleaching steps and segments',fontsize=10)
        self.plot_segmented_trace(fig, ax, line_avgs, seg_idcs)
        return fig, ax

    def fig_raw_trace(self,imarray,x_imarray,y_imarray):
        """Pcolormesh plot of raw trace , combined and in each axis
        """
        fig, ax = plt.subplots(1,3,figsize=(7.086,3))
        #fig.suptitle('extracted trace')
        self.plot_imarray(fig, ax[0], imarray)
        ax[0].set_title('raw trace',fontsize=10)
        ax[0].set_xlabel('pixel')
        ax[0].set_ylabel('t (lines)')
        self.plot_imarray(fig, ax[1], x_imarray)
        ax[1].set_title('x trace',fontsize=10)
        ax[1].set_xlabel('pixel')
        self.plot_imarray(fig, ax[2], y_imarray)
        ax[2].set_title('y trace',fontsize=10)
        ax[2].set_xlabel('pixel')
        #plt.show()
        return fig, ax
    
    def plot_segmented_trace(self, fig, ax, line_avgs, seg_idcs):
        ax.plot(line_avgs[0],c=self.s.c10,alpha=.3,label='x')
        ax.plot(line_avgs[1],c=self.s.c20,alpha=.3,label='y')
        for pair in seg_idcs:
            ax.fill_between(pair, 0, np.max(line_avgs), color='gray', alpha=0.5, transform=ax.get_xaxis_transform())
            ax.plot(range(pair[0],pair[1]+1),line_avgs[0][pair[0]:pair[1]+1],c=self.s.c10,zorder=5)
            ax.plot(range(pair[0],pair[1]+1),line_avgs[1][pair[0]:pair[1]+1],c=self.s.c20,zorder=5)
        ax.set_xlabel('t (lines)')
        ax.set_ylabel('counts (1)')
        ax.legend()
        return fig, ax
    
    def plot_imarray(self, fig, ax, imarray):
        ax.pcolormesh(imarray,cmap=self.s.cmap_b,linewidth=0,rasterized=True)
        ax.set_xlim(0,160)
        ax.set_xticks(np.arange(0,161,40))
        return fig, ax