"""
Reproduce the figure illustrating the linescan principle (SI).

copyright: @Thomas Hensel, 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')

from lib.constants import *
from lib.plotting.style import *
from lib.plotting.artefacts import Figures

def psf(x,FWHM,a):
    f = a*np.exp(-4*np.log(2)*x**2/FWHM**2)
    return f

def doughnut(x,FWHM,a):
    return a * 4*np.exp(1)*np.log(2)*x**2/FWHM**2 * psf(x,FWHM,1)

def model(dx,phi):
    """
    dx and PHI in units of lambda: 0.1 Lambda etc"""
    DX = dx * 4*np.pi
    PHI = phi * 4*np.pi + np.pi
    return .2*(4 + 2*np.sqrt(2*(1+np.cos(DX)))*np.cos(PHI))

def make_figure(plot_style='nature',color_scheme='default',show=True):
    
    s = Style(style=plot_style,color_scheme=color_scheme)
    # Generate data for the Gaussian functions
    d=0.25
    x = np.linspace(-1, 1, 1000)
    d0 = 0.
    d1 = d
    d2 = 2*d
    FWHM = 300

    fig, ax = plt.subplot_mosaic(
            [
                ["a)","b)"]
            ]
        ,empty_sentinel="BLANK"
        ,gridspec_kw = {
            "width_ratios": [1,1]
            ,"height_ratios": [1]
            }
        ,constrained_layout=True
        ,figsize=(7.086,3.5)
        ,sharex=False,sharey=True
        )
    #-------------------------------------
    # psf1 and phase scan
    x = np.linspace(-1,1,200) #space in lambda
    ax['a)'].plot(x,max(model(0,x))*psf(x,1,1),c='gray',alpha=1,ls='-',label='envelope')#plot envelope
    ax['a)'].plot(x,model(0,.5*(x))*psf(x,1,1),c=s.c10,alpha=1,ls='-',label=r'$\phi=0.0$ FWHM')
    ax['a)'].plot(x,model(0,.5*(x+.1))*psf(x,1,1),c=s.c10,alpha=.5,ls='-',label=r'$\phi=0.1$ FWHM')
    ax['a)'].scatter([-d1/2,d1/2],[0.07,0.07],marker='*',s=100,c=s.c30,zorder=3)
    ax['a)'].set_xlim(-1,1)
    ax['a)'].set_xticks([-1,0,1])
    ax['a)'].set_ylim(-.05,1.8)
    ax['a)'].set_yticks([-.05,1.8])
    ax['a)'].set_xlabel(r'x (FWHM)')
    ax['a)'].set_ylabel(r'I (a.u.)')
    ax['a)'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=False,right=False,labelright=False,direction='in')
    ax['a)'].legend(bbox_to_anchor=(1.02, 0.02), loc='lower right',frameon=True)
    
    #-------------------------------
    # harmonic response of system
    phi=x
    ax['b)'].plot(phi,model(0,phi/2),c=s.c10,alpha=1,ls='-',label=r'I$(\phi,\,d=0.0$ FWHM)')
    ax['b)'].plot(phi,model(0.2/2,phi/2),c=s.c10,alpha=.6,ls='-',label=r'I$(\phi,\,d=0.2$ FWHM)')
    ax['b)'].plot(phi,model(0.4/2,phi/2),c=s.c10,alpha=.3,ls='-',label=r'I$(\phi,\,d=0.4$ FWHM)')
    ax['b)'].set_xlim(-1,1)
    ax['b)'].set_xticks([-1,0,1])
    ax['b)'].set_ylim(-.05,1.8)
    ax['b)'].set_yticks([-.05,1.8])
    ax['b)'].set_xlabel(r'$\phi$ ($\pi$)')
    ax['b)'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=False,right=False,labelright=False,direction='in')
    ax['b)'].legend(bbox_to_anchor=(1.02, 0.02), loc='lower right', title= f'',frameon=True)

    #----------------------------
    s.drop_axes(axes=[ax['a)'],ax['b)']],dirs=['bottom','left'])
    s.hide_axes(axes=[ax['a)'],ax['b)']],dirs=['top','right'])
    if show:
        plt.show()
    return fig

if __name__=='__main__':
    fig = make_figure(show=True)
    #path_to_figure = Figures().save_fig(fig, 'LineScanPrincipleFigureSI')