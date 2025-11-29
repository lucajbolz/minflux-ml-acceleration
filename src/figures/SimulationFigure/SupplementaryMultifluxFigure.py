"""
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
    for i,c in enumerate(configs):
        for k in range(2):
            for l in range(2):
                m0 = sorted(df['M0'].unique())[k*2+l]

                ax = axs[f'{i}']
                sub_df = df.loc[(df['M0'].apply(lambda x: np.any([m in [x] for m in [m0]])))&(df['config']==c)]
                
                x = sub_df.groupby('reld0')['reld0'].mean().values
                y = sub_df.groupby('reld0')['relRMSE'].mean().rolling(window=indexer,min_periods=1).mean().values
                #std = sub_df.groupby('reld0')['relRMSE'].std().rolling(window=5,min_periods=1).mean().values
                #ax.fill_between(x, y-std, y+std, color=colors[k*2+l], alpha=0.5)
                ax.plot(x,y,c=colors[k*2+l])
                
                ax.set_yscale("log", base=10)
                ax.set_ylim(1E-2,1E0)
                ax.set_xlim(0,.1)
                ax.set_ylabel(r'$\sigma/d$ (1)')
                #ax.get_legend().remove()
                ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
                s.drop_axes(axes=[ax])
                s.hide_axes(axes=[ax],dirs=['top','right'])
                
                tw = ax.twiny()
                tw.set_xlim(0,.1*LAMBDA/2)
                if i==0:
                    tw.set_xlabel('d (nm)')
                    tw.xaxis.set_label_position("top")
                tw.tick_params(top=True, labeltop=True, bottom=True, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')
                tw.tick_params(axis='x',colors='k')#s.c20)
                #s.drop_axes(axes=[tw])
                s.hide_axes(axes=[tw],dirs=['bottom','left','right'])
                
                if i < len(configs)-1:
                    ax.set(xlabel=None)
                else:
                    ax.set_xlabel(r'd ($\lambda$)')

                index = str(i) + str(k) + str(l)
                ax = axs[index]
                points = MCSimulation()._generate_points(1, m0, 1, config=c).reshape((m0,2))*LAMBDA/(4*np.pi)
                circle = plt.Circle((0,0), 1/2, color='k', fill=False,zorder=-1)
                #ax.add_patch(circle)
                ax.scatter(points.T[0], points.T[1], color=colors[k*2+l], marker='o',s=10,zorder=2)

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
                if (k*2+l<=1):
                    tx = ax.text(0.5, .05, f'M={m0}', transform=ax.transAxes,
                        verticalalignment='bottom',horizontalalignment='center',color='k',zorder=10
                        ,bbox=dict(facecolor='white', edgecolor='none', pad=1.,alpha=.9)
                        )
                    ax.xaxis.set_major_locator(MultipleLocator(2))
                    ax.tick_params(which='major',top=True, labeltop=True, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')
                    if (i == 0):
                        ax.set_xlabel('x (d)')
                        ax.xaxis.set_label_position("top")
                if (k*2+l>1):
                    tx = ax.text(0.5, .95, f'M={m0}', transform=ax.transAxes,
                        verticalalignment='top',horizontalalignment='center',color='k',zorder=10
                        ,bbox=dict(facecolor='white', edgecolor='none', pad=1.,alpha=.9)
                        )
                    ax.xaxis.set_major_locator(MultipleLocator(2))
                    ax.tick_params(which='major',top=False, labeltop=False, bottom=True, labelbottom=True,left=False,labelleft=False,right=False,labelright=False,direction='in')
                    if (i == len(configs)-1):
                        ax.set_xlabel('x (d)')
                bb = tx.get_window_extent().transformed(ax.transData.inverted())
                rect = plt.Rectangle((bb.x0, bb.y0), bb.width, bb.height,
                                 facecolor="wheat", alpha=1., zorder=5)
                #ax.add_patch(rect)
                

           
        
        #sub_df = df.loc[(df['M0']==m)&(df['chunk_size']==n)].groupby('d0')
        #x = sub_df['reld0'].median().to_numpy()
        #y = sub_df['relRMSE'].median().to_numpy()
        #std = sub_df['relRMSE'].std().to_numpy()
        #ax1.errorbar(x,y,yerr=0,ls='-',marker=markers[i],c=colors[i],label=f'M={m}')

    

    # Contour plot on the second axis
    #gs = axis2.get_gridspec()
    #nrow, ncol, start, stop = axis2.get_subplotspec().get_geometry()
    #start_row, start_col = divmod(start, ncol)
    #stop_row, stop_col = divmod(stop, ncol)
    #axis2.remove()
    #inner_grid = gs[start_row:stop_row+1,start_col:stop_col+1]#.subgridspec(1, 1, wspace=0.0, hspace=0.0, height_ratios=[1])
    #axis2 = fig.add_subplot(inner_grid,projection='3d')

    """for i, d in enumerate(sorted(df['d0'].unique())):
        sub_df = df.loc[(df['d0']==d)]
        M0_values = sorted(sub_df['M0'].unique())
        chunk_size_values = sorted(sub_df['chunk_size'].unique())
        smallest_chunk_sizes = sub_df[sub_df['relRMSE']-.5 <= 0].groupby('M0')['chunk_size'].min().reset_index()
        
        M0, chunk_size = np.meshgrid(M0_values, chunk_size_values)
        relRMSE_values = sub_df.pivot_table(index='chunk_size', columns='M0', values='relRMSE').values

        x, y = smallest_chunk_sizes['M0'].values, smallest_chunk_sizes['chunk_size'].values
        axis2.plot(x,y,color=colors[i])"""

        #axis2.contourf(M0, relRMSE_values,chunk_size, zdir='y', offset=d, cmap='coolwarm',alpha=.5)
    #axis2.set(xlim=(np.min(M0_values), np.max(M0_values)), zlim=(np.min(chunk_size_values), np.max(chunk_size_values)), ylim=(np.min(df['d0'].unique()), np.max(df['d0'].unique())),xlabel='X', ylabel='Y', zlabel='Z')
    
    """ds = df['d0'].unique()
    d = ds[np.where(15<ds)].min()
    sub_df = df.loc[(df['d0']==d)]
    M0_values = sorted(sub_df['M0'].unique())
    chunk_size_values = sorted(sub_df['chunk_size'].unique())
    smallest_chunk_sizes = sub_df[sub_df['relRMSE']-.5 <= 0].groupby('M0')['chunk_size'].min().reset_index()

    # Create a grid of M0 and chunk_size values
    M0, chunk_size = np.meshgrid(M0_values, chunk_size_values)
    relRMSE_values = sub_df.pivot_table(index='chunk_size', columns='M0', values='relRMSE').values

    ax2.contourf(M0, chunk_size, relRMSE_values, cmap='viridis')
    x, y = smallest_chunk_sizes['M0'].values, smallest_chunk_sizes['chunk_size'].values
    ax2.plot(x,y,c=s.c10)
    ax2.set_xlabel('No. Molecules (1)')
    ax2.set_ylabel('No. Photons (1)')
    ax2.set_title(f'd={d}')
    ax2.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')"""
    
    

def make_figure(plot_style='nature',color_scheme='default',show=True):
    s = Style(style=plot_style,color_scheme=color_scheme)
    base_dir = os.path.join(ROOT_DIR,'src/figures/SimulationFigure')

    file = ut.BaseFunc().find_files(os.path.join(base_dir,'data'), lambda file: ut.BaseFunc().match_pattern(file, 'multiflux.pkl', match='partial'),max_files=1)[0] # load pkl
    df = pd.read_pickle(file)
    n_cases = df['config'].nunique()

    m = [
    [str(n), [[str(n) + str(j) + str(l) for l in range(2)] for j in range(2)]]
    for n in range(n_cases)
    ]
    mosaic = []
    for l in m:
        mosaic.append(['.','.'])
        mosaic.append(l)
        
    
    gridspec_kw = {
        "width_ratios": [1,.6]
        ,"height_ratios": len(m)*[.3,1]
        }

    fig, axs = plt.subplot_mosaic(mosaic
    ,constrained_layout=True
    #,layout='compressed'
    #,layout='tight'
    ,empty_sentinel="."
    ,gridspec_kw = gridspec_kw
    ,figsize=s.get_figsize(scale=.7,rows=2.3,cols=1.7,ratio=1)#
    ,sharex=False
    ,sharey=False
    )

    configs = df['config'].unique()
    plot_data(fig,axs,df,configs,s)

    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
    plt.figure(1).text(0.1932, 0.8461, 'Structure: Grid', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[0].new
    plt.figure(1).text(0.1932, 0.5369, 'Structure: Line', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[1].new
    plt.figure(1).text(0.1932, 0.2080, 'Structure: Polygon', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[2].new
    plt.figure(1).text(0.0328, 0.9495, 'a', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[3].new
    plt.figure(1).text(0.0328, 0.6189, 'b', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[4].new
    plt.figure(1).text(0.0328, 0.2974, 'c', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[5].new
    #% end: automatic generated code from pylustrator

    if show:
        plt.show()
    return fig

if __name__=='__main__':
    fig = make_figure(show=True)
    #path_to_figure = Figures().save_fig(fig, 'MultifluxSIFigure')