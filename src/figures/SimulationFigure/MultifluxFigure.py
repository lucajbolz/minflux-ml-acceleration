"""
Main Text Multiflux Figure with only one panel, mixed cases

copyright: @Thomas Hensel, 2023
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np

import lib.utilities as ut
from lib.plotting.style import Style
from lib.plotting.artefacts import Figures
from lib.constants import LAMBDA
from lib.simulation.MINFLUXMonteCarlo import MCSimulation
from lib.config import ROOT_DIR

import pylustrator
pylustrator.start()

def prepare_axis(fig,axis,style):
    """Prepare axis with twin axis and correct style of the spines.
    """
    s=style

    axis.set_ylim(0,1)
    axis.spines['left'].set_color(s.c10)
    axis.set_ylabel('rel. error')
    axis.set_xlabel('d (nm)')
    axis.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
    axis.tick_params(axis='y',colors=s.c10)

    tw = axis.twinx()
    tw.set_ylim(0,1)
    tw.spines['right'].set_color(s.c20)
    tw.yaxis.tick_right()
    tw.set_ylabel('rel. bias')
    tw.yaxis.set_label_position("right")
    tw.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=True,labelright=True,direction='in')
    tw.tick_params(axis='y',colors=s.c20)

    s.drop_axes(axes=[axis,tw])
    s.hide_axes(axes=[axis],dirs=['top','right'])
    s.hide_axes(axes=[tw],dirs=['top','left'])
    return axis,tw

def create_cartoon(ax,m,config,n,colors):
    points = MCSimulation()._generate_points(1, m, 1, config=config).reshape((m,2))*LAMBDA/(4*np.pi)
    if config=='polygon':
        circle = plt.Circle((0,0), 1/2, color='k', fill=False,zorder=-1)
        ax.add_patch(circle)
        tx = ax.text(0.5, .85, r'$\o=d$', transform=ax.transAxes,
            verticalalignment='top',horizontalalignment='center',color='k',zorder=10
            ,bbox=dict(facecolor='white', edgecolor='none', pad=1.,alpha=.9)
            )
    if config=='line':
        tx = ax.text(0.5, .85, r'$|x_i - x_{i+1}|=d$', transform=ax.transAxes,
            verticalalignment='top',horizontalalignment='center',color='k',zorder=10
            ,bbox=dict(facecolor='white', edgecolor='none', pad=1.,alpha=.9)
            )
    ax.scatter(points.T[0], points.T[1], color=colors[n], marker='o',s=10,alpha=.8,zorder=2)

    major_ticks = np.arange(-10,10,1)
    minor_ticks = np.arange(-10,10,1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.7)
    
    ax.set_ylim(-3,3)
    ax.set_xlim(-3,3)
    ax.set_aspect('equal')
    ax.tick_params(which='both',top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')
    tx = ax.text(0.5, .05, f'm={m}', transform=ax.transAxes,
            verticalalignment='bottom',horizontalalignment='center',color='k',zorder=10
            ,bbox=dict(facecolor='white', edgecolor='none', pad=1.,alpha=.9)
            )
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    if (n<=1):
        ax.tick_params(which='major',top=True, labeltop=True, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')
        ax.xaxis.set_label_position("top")
        ax.set_xlabel('x (d)')
        ax.xaxis.label.set_color('none')
        ax.tick_params(axis='x', labelcolor="none")
        if n%2==0:
            ax.tick_params(which='major',top=True, labeltop=True, bottom=False, labelbottom=False,left=True,labelleft=True,right=False,labelright=False,direction='in')
            ax.set_ylabel('y (d)')
    if (n>1):
        ax.tick_params(which='major',top=False, labeltop=False, bottom=True, labelbottom=True,left=False,labelleft=False,right=False,labelright=False,direction='in')
        ax.set_xlabel('x (d)')
        if n%2==0:
            ax.tick_params(which='major',top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
            ax.set_ylabel('y (d)')
    
    bb = tx.get_window_extent().transformed(ax.transData.inverted())
    rect = plt.Rectangle((bb.x0, bb.y0), bb.width, bb.height,
                        facecolor="white", alpha=.9, zorder=5)
    #ax.add_patch(rect)

def plot_data(fig,axs,df,configs,style):
    s = style
    colors = ['k', s.c20, s.c30, s.c10, 'c', 'm', 'y', 'g', 'purple', 'brown']
    markers = ['.', 'o', 's', '^', 'D', 'v', '>', 'x', '+', '*']
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
    
    df = df.assign(relRMSE=df['rmse'] / df['d0'], reld0=df['d0']*2/LAMBDA)
    ns = df['chunk_size'].unique()
    n = ns[np.where((500<=ns) & (ns<1000))].min()
    df = df.loc[df['chunk_size']==n].copy()
    #df.loc[:,'std'] = df.groupby('M0')['relRMSE'].rolling(window=3,min_periods=1).std().reset_index()['relRMSE']

    # plot relative error for cases M = 2 (line), 3 (poly), 5 (poly and line)
    ax = axs['0']
    
    sub_df = df.loc[(df['M0'].apply(lambda x: np.any([m in [x] for m in [2]])))&(df['config']=='line')]
    x = sub_df.groupby('reld0')['reld0'].mean().values
    y = sub_df.groupby('reld0')['relRMSE'].mean().rolling(window=indexer,min_periods=1).mean().values
    ax.plot(x,y,c=colors[0])
    create_cartoon(axs['000'],2,'line',0,colors)

    sub_df = df.loc[(df['M0'].apply(lambda x: np.any([m in [x] for m in [3]])))&(df['config']=='polygon')]
    x = sub_df.groupby('reld0')['reld0'].mean().values
    y = sub_df.groupby('reld0')['relRMSE'].mean().rolling(window=indexer,min_periods=1).mean().values
    ax.plot(x,y,c=colors[1])
    create_cartoon(axs['001'],3,'polygon',1,colors)

    sub_df = df.loc[(df['M0'].apply(lambda x: np.any([m in [x] for m in [5]])))&(df['config']=='line')]
    x = sub_df.groupby('reld0')['reld0'].mean().values
    y = sub_df.groupby('reld0')['relRMSE'].mean().rolling(window=indexer,min_periods=1).mean().values
    ax.plot(x,y,c=colors[2])
    create_cartoon(axs['010'],5,'line',2,colors)

    sub_df = df.loc[(df['M0'].apply(lambda x: np.any([m in [x] for m in [5]])))&(df['config']=='polygon')]
    x = sub_df.groupby('reld0')['reld0'].mean().values
    y = sub_df.groupby('reld0')['relRMSE'].mean().rolling(window=indexer,min_periods=1).mean().values
    ax.plot(x,y,c=colors[3])
    create_cartoon(axs['011'],5,'polygon',3,colors)
    

    ax.set_yscale("log", base=10)
    ax.set_ylim(1E-2,1E0)
    ax.set_xlim(0,.1)
    ax.set_ylabel(r'$\sigma/d$')
    ax.yaxis.set_label_position("right")
    ax.set_xlabel(r'd ($\lambda$)')
    #ax.get_legend().remove()
    ax.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=False,right=True,labelright=True,direction='in')
    s.drop_axes(axes=[ax])
    s.hide_axes(axes=[ax],dirs=['top'])
    
    tw = ax.twiny()
    tw.set_xlim(0,.1*LAMBDA/2)
    tw.set_xlabel('d (nm)')
    tw.xaxis.set_label_position("top")
    tw.tick_params(top=True, labeltop=True, bottom=True, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')
    tw.tick_params(axis='x',colors='k')#s.c20)
    s.hide_axes(axes=[tw],dirs=['bottom','left','right'])
    

def make_figure(plot_style='nature',color_scheme='default',show=True):

    s = Style(style=plot_style,color_scheme=color_scheme)
    base_dir = os.path.join(ROOT_DIR,'src/figures/SimulationFigure')

    file = ut.BaseFunc().find_files(os.path.join(base_dir,'data'), lambda file: ut.BaseFunc().match_pattern(file, 'multiflux.pkl', match='partial'),max_files=1)[0] # load pkl
    df = pd.read_pickle(file)
    n_cases = 1

    m = [
    ['.',[[str(n) + str(j) + str(l) for l in range(2)] for j in range(2)],'.',str(n)]
    for n in range(n_cases)
    ]
    mosaic = []
    for l in m:
        mosaic.append(['.','.','.','.'])
        mosaic.append(l)
        
    
    gridspec_kw = {
        "width_ratios": [.1,.7,.06,1]
        ,"height_ratios": len(m)*[.1,1]
        }

    fig, axs = plt.subplot_mosaic(mosaic
    ,constrained_layout=True
    #,layout='compressed'
    #,layout='tight'
    ,empty_sentinel="."
    ,gridspec_kw = gridspec_kw
    ,figsize=s.get_figsize(scale=.9,rows=.8,cols=2,ratio=1)#
    ,sharex=False
    ,sharey=False
    )

    configs = df['config'].unique()
    plot_data(fig,axs,df,configs,s)

    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
    plt.figure(1).text(0.0079, 0.9162, 'a', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[0].new
    plt.figure(1).text(0.4137, 0.9162, 'b', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[1].new
    #% end: automatic generated code from pylustrator

    if show:
        plt.show()
    return fig

if __name__=='__main__':
    fig = make_figure(show=True)
    #path_to_figure = Figures().save_fig(fig, 'MultifluxFigure')