"""
Reproduce the MINFLUX experiment figure of dynamic systems.

copyright: @Thomas Hensel, 2024
"""

import os
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np

import lib.utilities as ut
from lib.constants import LAMBDA
from lib.plotting.style import Style
from lib.plotting.artefacts import Figures
from lib.config import DATA_DIR, ROOT_DIR

import pylustrator
pylustrator.start()

def plot_timeaveraged_trace(axis,filtered_df,name,s):
    curr_df = filtered_df.loc[(filtered_df['state_id']==2)&(filtered_df['file'].apply(lambda x: name in x))].copy()
    rolling_df = curr_df.groupby(['axis'],group_keys=False).rolling(window=10,min_periods=1).mean(['COM','pos1','pos2']).reset_index()#10
    sub_df = pd.melt(rolling_df.loc[:,['axis','chunk_id','bin_id','COM','pos1','pos2']],id_vars=['axis','chunk_id','bin_id'],value_vars=['COM','pos1','pos2']).sort_values(['bin_id','chunk_id']).reset_index(drop=True)
    dfx, dfy = sub_df.loc[sub_df['axis']==0],sub_df.loc[sub_df['axis']==1]
    x_data, y_data, comx, comy = dfx.loc[dfx['variable'].apply(lambda x: ('pos1' in x)|('pos2' in x)),'value'].values, dfy.loc[dfy['variable'].apply(lambda x: ('pos1' in x)|('pos2' in x)),'value'].values, dfx.loc[dfx['variable'].apply(lambda x: 'COM' in x),'value'].values, dfy.loc[dfy['variable'].apply(lambda x: 'COM' in x),'value'].values
    
    xi, yi = np.mgrid[-35:35:400j, -35:35:400j]# Create a meshgrid for the x and y positions
    xy = np.vstack([xi.ravel(), yi.ravel()])# Combine the x and y positions into a single array
    kde = gaussian_kde(np.vstack([x_data,y_data]),bw_method=.07)
    #kde = gaussian_kde(np.vstack([comx,comy]),bw_method=.02)#0.07
    zi = kde(xy).reshape(xi.shape)
    pcol = axis.pcolormesh(xi,yi,zi,cmap=s.cmap_seq,linewidth=0,rasterized=False)
    axis.set_xlim(-30,30)
    axis.set_ylim(-30,30)
    axis.set_xticks(np.arange(-30,31,10))
    axis.set_yticks(np.arange(-30,31,10))
    axis.set_aspect('equal')
    axis.set_xlabel('x (nm)')
    axis.set_ylabel('y (nm)')
    axis.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')

    # add inset with COM trajectory
    """with plt.style.context('dark_background'):
        axins = axis.inset_axes([0.03, 0.67, 0.3, 0.3])
        axins.pcolormesh(xi,yi,zi,cmap=s.cmap_seq,linewidth=0,rasterized=False,alpha=.6)
        axins.plot(comx,comy,zorder=1,c='w',lw=1,alpha=1)
        axins.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=False,labelleft=False,right=True,labelright=True,direction='in')
        #axins.set_title('COM')
        axins.set_xlim(-20,20)
        axins.set_ylim(-20,20)
        axins.set_xticks([-20,0,20])
        axins.set_yticks([-20,0,20])
        axins.set_aspect('equal')"""

def plot_d_trace(axis,df,s):
    tmp_df = df.groupby('gt').apply(lambda x: x).reset_index(drop=True).copy()
    tmp_df.loc[:,'std'] = tmp_df.groupby('gt')['d_norm'].rolling(window=4,min_periods=1).std().reset_index()['d_norm']#5
    tmp_df.loc[:,'d_norm'] = tmp_df.groupby('gt')['d_norm'].rolling(window=3,min_periods=1).mean().reset_index()['d_norm']#5
    for gt,cols in zip(tmp_df['gt'].unique(),[[s.c20,s.c21],[s.c10,s.c11]]):
        sub_df = tmp_df.loc[(tmp_df['gt']==gt)&(tmp_df['axis']==0)]#select one axis since we only plot dnorm
        axis.fill_between(sub_df['time'].values, sub_df['d_norm'].values-sub_df['std'].values, sub_df['d_norm'].values+sub_df['std'].values, color=cols[1], alpha=0.5)
        axis.plot(sub_df['time'].values,sub_df['d_norm'].values,label=f'{gt}',c=cols[0])
    axis.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
    axis.set_ylabel('d (nm)')
    axis.set_xlabel('Time (ms)')
    axis.set_yticks([0,15,30,45])
    axis.set_xlim(0,2500)
    axis.set_xticks(np.arange(0,2501,500))
    axis.legend()
    axis.get_legend().remove()

def plot_hist(axis,df,s):
    sns.histplot(data=df, y="d_norm",binwidth=1, ax=axis,hue="gt", stat='probability', palette={30:s.c10,20:s.c30,15:s.c20}, common_norm=False, kde=True)
    axis.set_xlabel('Probability')
    axis.set_ylabel('d (nm)')
    axis.yaxis.set_label_position("right")
    axis.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=False,labelleft=False,right=True,labelright=True,direction='in')
    axis.set_xticks([0,.1,.2,.3,.4])
    axis.set_xlim(0,.4)
    patch1 = mpatches.Patch(color=s.c10, label='32 nm')
    patch2 = mpatches.Patch(color=s.c20, label='15 nm')
    handles = [patch1,patch2]
    labels = [patch1._label,patch2._label]
    axis.legend(handles, labels, bbox_to_anchor=(.98, -.05), title=r'Expected $d_2$:', loc='lower right', frameon=True)
    


def make_figure(plot_style='nature',color_scheme='default',show=True):
    
    s = Style(style=plot_style,color_scheme=color_scheme)
    base_dir = os.path.join(ROOT_DIR,'src/figures/DynamicDataFigure/data')

    #-------------------------------
    # plot d wrt t:
    # Load fitting results
    file = ut.BaseFunc().find_files(base_dir, lambda file: ut.BaseFunc().match_pattern(file, 'dynamic-minflux-data.pkl', match='partial'),max_files=1)[0]
    df = pd.read_pickle(file)
    names = ['GQ30003_MF_scans_i1_circle_movement_27','GQ15_MF_scans_modulated_i1_mp2_3']# select exemplary nanorulers
    pairs = [names[i:i + 2] for i in range(0, len(names), 2)]

    for idx,pair in enumerate(pairs):
        print(pair)
        # start figure
        fig, ax = plt.subplot_mosaic(
        [
            ['BLANK','BLANK','BLANK']
            ,['a1','a2','a3']
            ,['BLANK','BLANK','BLANK']
            ,['b1','b2','b3']
            ,['BLANK','BLANK','BLANK']
            ,['c1','c1','c3']
        ]#grid
        ,empty_sentinel="BLANK"
        ,gridspec_kw = {
            "width_ratios": [.3,.3,.3]#widths
            ,"height_ratios": [.1,1,.05,1,.01,.5]#heights
            }
        ,constrained_layout=True
        ,figsize=s.get_figsize(rows=2.5,cols=3,ratio=1)#
        ,sharex=False
        ,sharey=False
        )


        file = ut.BaseFunc().find_files(base_dir, lambda file: ut.BaseFunc().match_pattern(file, 'circle1', match='partial'),max_files=1)[0]
        img = Image.open(file)
        ax['a1'].imshow(img)
        ax['a1'].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')

        file = ut.BaseFunc().find_files(base_dir, lambda file: ut.BaseFunc().match_pattern(file, 'circle2', match='partial'),max_files=1)[0]
        img = Image.open(file)
        ax['a2'].imshow(img)
        ax['a2'].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')

        file = ut.BaseFunc().find_files(base_dir, lambda file: ut.BaseFunc().match_pattern(file, 'lissajou1', match='partial'),max_files=1)[0]
        img = Image.open(file)
        ax['b1'].imshow(img)
        ax['b1'].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')

        file = ut.BaseFunc().find_files(base_dir, lambda file: ut.BaseFunc().match_pattern(file, 'lissajou2', match='partial'),max_files=1)[0]
        img = Image.open(file)
        ax['b2'].imshow(img)
        ax['b2'].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')
        
        filtered_df = df.loc[df['file'].apply(lambda x: np.any([name in x for name in pair]))].copy()
        filtered_df['COM'] = filtered_df['x0']*LAMBDA/(4*np.pi)
        axis_means = filtered_df.groupby(['file','axis'])['COM'].transform('mean')
        filtered_df.loc[:,'COM'] = filtered_df['COM'] - axis_means
        filtered_df['pos1'] = filtered_df['COM'] - filtered_df['d']/2
        filtered_df['pos2'] = filtered_df['COM'] + filtered_df['d']/2        

        # continuous distance measurement
        plot_d_trace(ax['c1'],filtered_df,s)
        plot_hist(ax['c3'],filtered_df,s)
        ax['c3'].sharey(ax['c1'])        

        #--------------------------
        # 2D trace: time-averaged localizations
        plot_timeaveraged_trace(ax['a3'],filtered_df,pair[0],s)
        ax['a3'].text(0.23, .32, r'$d_1$', transform=ax['a3'].transAxes,color='white',zorder=10,
                        verticalalignment='top',bbox=dict(facecolor='none', edgecolor='none', pad=1.0))
        ax['a3'].annotate("", xy=(0.2, 0.44), xytext=(0.47, 0.13),xycoords=('axes fraction','axes fraction'),arrowprops=dict(arrowstyle="<->",color='white'),color='white')
        
        ax['a3'].text(0.63, .27, r'$d_2$', transform=ax['a3'].transAxes,color='white',zorder=10,
                        verticalalignment='top',bbox=dict(facecolor='none', edgecolor='none', pad=1.0))
        ax['a3'].annotate("", xy=(0.45, 0.12), xytext=(0.9, 0.5),xycoords=('axes fraction','axes fraction'),arrowprops=dict(arrowstyle="<->",color='white'),color='white')
        
        ax['a3'].text(0.02, .98, r'$d_1=30$ nm', transform=ax['a3'].transAxes,color='white',zorder=10,
                        verticalalignment='top',horizontalalignment='left',bbox=dict(facecolor='none', edgecolor='none', pad=1.0))
        ax['a3'].text(0.02, .9, r'$d_2=32$ nm', transform=ax['a3'].transAxes,color='white',zorder=10,
                        verticalalignment='top',horizontalalignment='left',bbox=dict(facecolor='none', edgecolor='none', pad=1.0))

        plot_timeaveraged_trace(ax['b3'],filtered_df,pair[1],s)
        ax['b3'].text(0.35, .2, r'$d_1$', transform=ax['b3'].transAxes,color='white',zorder=10,
                        verticalalignment='top',bbox=dict(facecolor='none', edgecolor='none', pad=1.0))
        ax['b3'].annotate("", xy=(0.22, 0.23), xytext=(0.59, 0.23),xycoords=('axes fraction','axes fraction'),arrowprops=dict(arrowstyle="<->",color='white'),color='white')
        
        ax['b3'].text(0.63, .3, r'$d_2$', transform=ax['b3'].transAxes,color='white',zorder=10,
                        verticalalignment='top',bbox=dict(facecolor='none', edgecolor='none', pad=1.0))
        ax['b3'].annotate("", xy=(0.5, 0.25), xytext=(0.8, 0.4),xycoords=('axes fraction','axes fraction'),arrowprops=dict(arrowstyle="<->",color='white'),color='white')

        ax['b3'].text(0.02, .98, r'$d_1=20$ nm', transform=ax['b3'].transAxes,color='white',zorder=10,
                        verticalalignment='top',bbox=dict(facecolor='none', edgecolor='none', pad=1.0))
        ax['b3'].text(0.02, .9, r'$d_2=15$ nm', transform=ax['b3'].transAxes,color='white',zorder=10,
                        verticalalignment='top',bbox=dict(facecolor='none', edgecolor='none', pad=1.0))

        #-----------------------
        # post script
        s.drop_axes(axes=[ax['a1'],ax['a2'],ax['a3'],ax['b1'],ax['b2'],ax['b3'],ax['c1'],ax['c3']],dirs=['left','bottom'])#ax['b'],
        s.hide_axes(axes=[ax['a3'],ax['b3'],ax['c1']],dirs=['top','right'])
        s.hide_axes(axes=[ax['c3']],dirs=['top'])
        s.hide_axes(axes=[ax['a1'],ax['a2'],ax['b1'],ax['b2']])
        
        #% start: automatic generated code from pylustrator
        plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
        import matplotlib as mpl
        getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
        plt.figure(1).text(0.0266, 0.9609, 'a', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[0].new
        plt.figure(1).text(0.3601, 0.9609, 'b', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[1].new
        plt.figure(1).text(0.6319, 0.9609, 'c', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[2].new
        plt.figure(1).text(0.0266, 0.5962, 'd', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[3].new
        plt.figure(1).text(0.3601, 0.5962, 'e', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[4].new
        plt.figure(1).text(0.6319, 0.5962, 'f', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[5].new
        plt.figure(1).text(0.0266, 0.2521, 'g', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[6].new
        plt.figure(1).text(0.6319, 0.2521, 'h', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[7].new
        #% end: automatic generated code from pylustrator
        if show:
            plt.show()
            plt.close()
    return fig


if __name__=='__main__':
    fig = make_figure(show=True)
    #path_to_figure = Figures().save_fig(fig, 'MINFLUXDynamicFigure',meta={'generating script': script_name})