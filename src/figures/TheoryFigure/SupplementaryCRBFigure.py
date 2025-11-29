"""
Plot the Cramer Rao Bound for the separation estimation of two fluorophores, supplementary material.

copyright: @Thomas Hensel, 2024
"""

import os
script_name = os.path.basename(__file__)

import numpy as np
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt

from lib.constants import *
from lib.plotting.style import Style

import pylustrator
pylustrator.start()

def mysqrt(x): return np.sqrt((1.+0j)*x)

def minCRB(dx,l,N,v):
    """CRB at minimum in units of Lambda of the standing wave (lambda/2 of light!)"""
    DX = dx * 2*np.pi
    L = l * 2*np.pi

    aux0=2.*(1.-((2.**-0.5)*(v*((mysqrt((1.+(np.cos(DX)))))*(np.cos((0.5*\
    L)))))))
    aux1=((1.+aux0)-((2.**-0.5)*(v*(mysqrt((1.+(np.cos(DX))))))))*(((1/np.\
    sin(DX))**2))
    aux2=(-2.*(1.+(np.cos(L))))/(-2.+((mysqrt(2.))*(v*((mysqrt((1.+(np.\
    cos(DX)))))*(np.cos((0.5*L)))))))
    aux3=((v**-2.)*((1.+(np.cos(DX)))*aux1))/(((1.-(v*(mysqrt((((np.cos((\
    0.5*DX)))**2))))))**-1.)+aux2)
    output=2.*((mysqrt(2.))*(mysqrt((aux3/N))))

    sig=output/(2*np.pi)
    return sig

def find_min_d(v, l, n):
    def objective_func(d,l,n,v):
        err = np.abs(minCRB(d,l,n,v)/d)
        return np.abs(err - .5)
    linear_constraint = LinearConstraint([1], [1E-4], [.2])#m0,m1,b
    root_res = minimize(objective_func, args=(l,n,v),x0=.05, method='SLSQP',constraints=[linear_constraint])
    ds_roots = root_res.x
    return ds_roots
    
def find_min_v(d, l, n):
    def objective_func(v,d,l,n):
        err = np.abs(minCRB(d,l,n,v)/d)
        return np.abs(err - .5)
    linear_constraint = LinearConstraint([1], [.2], [1.])
    root_res = minimize(objective_func, args=(d,l,n),x0=.99, method='SLSQP',constraints=[linear_constraint])
    ds_roots = root_res.x
    return ds_roots

def plot_relCRBN(axis, s):
    x=list(map(lambda x: round(x),np.arange(10,100)))+list(map(lambda x: round(x),np.arange(1E2,1E3,20)))
    dx=.05
    L = 0.1
    y1 = np.abs(minCRB(dx,L,x,1.0))
    y2 = np.abs(minCRB(dx,L,x,.95))
    axis.plot(x, y1/dx,c=s.c10,label=f'd={dx}, L={L}, '+r'$\nu=1.0$')
    axis.plot(x, y2/dx,c=s.c10,alpha=.3,label=f'd={dx}, L={L}, '+r'$\nu=.95$')

    axis.set_xscale('log',base=10)
    axis.set_yscale('log',base=10)
    axis.set_xlim(np.min(x),np.max(x))
    axis.set_ylim(1E-2,1E0)
    axis.set_xlabel(r'N (photons)')
    axis.set_ylabel(r'$\sigma_{CRB}/d$')
    axis.legend(bbox_to_anchor=(0.01, 0.01), loc='lower left',frameon=True)#, title= f'N={round(N,0)}'
    axis.set_xticks([10,100,1E3])
    #axis.set_yticks([1E-2,1E-1,1E0,1E1,1E2,1E3])

def plot_relCRBSBR(axis, s):
    x = np.linspace(1,100,100)#SBR
    dx = .05#dx in units of Lambda
    L = 0.1
    N = 1000
    v = x/(x+1)
    y1 = np.abs(minCRB(dx,L,N,1.0 * v))
    y2 = np.abs(minCRB(dx,L,N,.95 * v))
    axis.plot(x, y1/dx,c=s.c10,label=f'd={dx}, N={N}, L={L}, '+r'$\nu_0=1.0$')
    axis.plot(x, y2/dx,c=s.c10,alpha=.3,label=f'd={dx}, N={N}, L={L}, '+r'$\nu_0=.95$')

    axis.set_xscale('log',base=10)
    axis.set_yscale('log',base=10)
    axis.set_xlim(np.min(x),np.max(x))
    axis.set_ylim(1E-2,1E0)
    axis.set_xlabel(r'SBR')
    axis.set_ylabel(r'$\sigma_{CRB}/d$')
    axis.legend(bbox_to_anchor=(0.01, 0.01), loc='lower left',frameon=True)#, title= f'N={round(N,0)}'
    #axis.set_xticks([1,10,100,1E3])
    #axis.set_yticks([1E-2,1E-1,1E0,1E1,1E2,1E3])

def plot_resdN(axis,s):
    x = list(map(lambda x: round(x),np.arange(10,100)))+list(map(lambda x: round(x),np.arange(1E2,1E3,20)))
    L=.1
    y1 = np.asarray([find_min_d(1.,L,n) for n in x]).flatten()#d in units of Lambda
    y2 = np.asarray([find_min_d(.95,L,n) for n in x]).flatten()#d in units of Lambda
    axis.fill_between(x=x,y1=0, y2=y1, facecolor='gray', alpha=0.2)
    axis.plot(x,y1,c=s.c10,label=f'L={L}, '+r'$v=1$')
    axis.plot(x,y2,c=s.c10,alpha=.3,label=f'L={L}, '+ r'$v=.95$')

    axis.set_ylim(1E-3,1E-1)
    axis.set_xlim(1E1,1E3)
    axis.set_xscale('log',base=10)
    axis.set_yscale('log',base=10)
    #axis.set_yticks([0.004,.2])
    #axis.set_xticks([1E0,1E2,5E2,1E3])
    axis.set_xlabel(r'N (photon counts)')
    axis.set_ylabel(r'resolvable d $(\lambda)$')
    axis.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right',frameon=True)

    #data for table
    x = [10,50,100,5E2,1E3]
    y = np.asarray([find_min_d(1.,.1,n) for n in x]).flatten()#d in units of Lambda
    data = [['d (nm)', r'$N$']]  # Header row
    data.extend(zip(list(map(round,y*LAMBDA)), list(map(lambda x: round(x,0),x))))
    # Corrected code for setting cell backgrounds to white
    num_rows = len(data)
    num_cols = len(data[0])
    cell_colors = [['w'] * num_cols for _ in range(num_rows)]
    # Create the table using the corrected cellColors parameter
    table = axis.table(cellText=data, loc='lower left', colWidths=[0.18] * num_cols, cellLoc='center', cellColours=cell_colors,zorder=5)
    table.auto_set_font_size(False)
    table.set_fontsize(5)

def plot_resdSBR(axis,s):
    x = np.linspace(1,100,100)#SBR
    v = x/(x+1)
    N=1000
    L=.1
    y1 = np.asarray([find_min_d(vi,L,N) for vi in v]).flatten()#d in units of Lambda
    y2 = np.asarray([find_min_d(.95*vi,L,N) for vi in v]).flatten()#d in units of Lambda
    axis.fill_between(x=x,y1=0, y2=y1, facecolor='gray', alpha=0.2)
    axis.plot(x,y1,c=s.c10,label=f'L={L}, N={N}, '+r'$v_0=1$')
    axis.plot(x,y2,c=s.c10,alpha=.3,label=f'L={L}, N={N}, '+r'$v_0=.95$')

    axis.set_ylim(1E-2,1E-1)
    axis.set_xlim(1,1E2)
    axis.set_xscale('log',base=10)
    axis.set_yscale('log',base=10)
    #axis.set_yticks([0.004,.2])
    #axis.set_xticks([1,10,1E2,5E2,1E3])
    axis.set_xlabel(r'SBR')
    axis.set_ylabel(r'resolvable d $(\lambda)$')
    axis.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right',frameon=True)


    #data for table
    x = np.array([1,5,10,20,50,100])
    v = x/(x+1)
    y = np.asarray([find_min_d(vi,.1,N) for vi in v]).flatten()#d in units of Lambda
    data = [['d (nm)', 'SBR']]  # Header row
    data.extend(zip(list(map(round,y*LAMBDA)), list(map(lambda x: round(x,0),x))))
    # Corrected code for setting cell backgrounds to white
    num_rows = len(data)
    num_cols = len(data[0])
    cell_colors = [['w'] * num_cols for _ in range(num_rows)]
    # Create the table using the corrected cellColors parameter
    table = axis.table(cellText=data, loc='lower left', colWidths=[0.18] * num_cols, cellLoc='center', cellColours=cell_colors,zorder=5)
    table.auto_set_font_size(False)
    table.set_fontsize(5)


def make_figure(plot_style='nature',color_scheme='default',show=True):
    s = Style(style=plot_style,color_scheme=color_scheme)
    N=10**2

    fig, ax = plt.subplot_mosaic(
            [
                ['.','.']
                ,['relCRB(N)','resd(N)']
                ,['.','.']
                ,['relCRB(SBR)','resd(SBR)']
            ]
        ,empty_sentinel='.'
        ,gridspec_kw = {
            "width_ratios": [1,1]
            ,"height_ratios": [.1,1,.1,1]
            }
        ,constrained_layout=True
        ,figsize=s.get_figsize(cols=2, rows=2.2,ratio=1)#(7.086,6.)
        )
    #fig.suptitle("Comparison of CRB for max and min (1D)")

    #*************************************************************************************

    plot_relCRBN(ax['relCRB(N)'],s)

    plot_relCRBSBR(ax['relCRB(SBR)'],s)

    plot_resdN(ax['resd(N)'],s)

    plot_resdSBR(ax['resd(SBR)'],s)

    axes = []
    for label, a in ax.items():
        axes.append(a)
    s.drop_axes(axes=axes,dirs=['bottom','left'])
    s.hide_axes(axes=axes,dirs=['top','right'])

    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
    plt.figure(1).text(0.0130, 0.9712, 'a', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[0].new
    plt.figure(1).text(0.5278, 0.9712, 'b', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[1].new
    plt.figure(1).text(0.0130, 0.4787, 'c', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[2].new
    plt.figure(1).text(0.5278, 0.4787, 'd', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[3].new
    #% end: automatic generated code from pylustrator

    if show:
        plt.show()
    return fig

if __name__=='__main__':
    fig = make_figure(show=True)
    #path_to_figure = Figures().save_fig(fig, 'theory-figure',meta={'generating script': script_name})