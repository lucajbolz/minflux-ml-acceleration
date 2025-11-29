"""
Plot description: phenomenological figure for paper: rayleigh ciretrion vs localization with a minimum
"""

import os
import argparse
script_name = os.path.basename(__file__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl

from lib.constants import *
from lib.plotting.style import Style
from lib.plotting.artefacts import Figures

import pylustrator
pylustrator.start()

from scipy.special import j1

def airy_pattern(x, y, wavelength=640, aperture_diameter=2e3):
    k = 2 * np.pi / wavelength
    r = np.sqrt(x**2 + y**2)
    theta = k * aperture_diameter / 1 * r
    pattern = (2 * j1(theta) / theta)**2
    return pattern / np.max(pattern)

def LG_mode(x, y, p, l, wavelength=1, aperture_diameter=1):
    k = 2 * np.pi / wavelength
    r = np.sqrt(x**2 + y**2)
    theta = k * aperture_diameter / 1 * r
    laguerre_poly = np.exp(1j * l * theta) * np.polynomial.laguerre.Laguerre(p)(2 * r**2 / aperture_diameter**2 - 1)
    pattern = (2 * j1(theta) / theta) * laguerre_poly
    return np.abs(pattern / np.max(pattern))

def HG_mode(x, y, m, n, wavelength=1, aperture_diameter=1):
    k = 2 * np.pi / wavelength
    r = np.sqrt(x**2 + y**2)
    x_term = np.polynomial.hermite.Hermite(m)(np.sqrt(2) * x / aperture_diameter)
    y_term = np.polynomial.hermite.Hermite(n)(np.sqrt(2) * y / aperture_diameter)
    pattern = np.exp(-(x**2 + y**2) / aperture_diameter**2) * x_term * y_term
    return np.abs(pattern / np.max(pattern))

def superposition_LG(x, y, weights, p_values, l_values, wavelength=1, aperture_diameter=1):
    result = np.zeros_like(x, dtype=np.complex128)
    for weight, p, l in zip(weights, p_values, l_values):
        result += weight * LG_mode(x, y, p, l, wavelength, aperture_diameter)
    return np.abs(result / np.max(result))

def superposition_HG(x, y, weights, m_values, n_values, wavelength=1, aperture_diameter=1):
    result = np.zeros_like(x, dtype=np.complex128)
    for weight, m, n in zip(weights, m_values, n_values):
        result += weight * HG_mode(x, y, m, n, wavelength, aperture_diameter)
    return np.abs(result / np.max(result))




def make_figure(plot_style='nature',color_scheme='default',show=True):
    s = Style(style=plot_style,color_scheme=color_scheme)

    # Generate data for the Gaussian functions
    d = 0.1
    phi = np.linspace(0,np.pi,4,endpoint=False)
    mosaic = []
    for i,p in enumerate(phi):
        mosaic.append([str(p),f'cbar{str(i)}'])

    fig = plt.figure(layout="constrained", figsize=s.get_figsize(scale=.4,cols=1.1,rows=4,ratio=1))

    ax = fig.subplot_mosaic(#
            mosaic
        ,empty_sentinel="BLANK"
        ,gridspec_kw = {
            #'top':.5
            #,'bottom':.0
            #,'left':.1
            #,"right": 0.5
            "width_ratios": [1,.1]
            ,"height_ratios": [1]*len(phi)
            }
        ,sharex=False,sharey=False
        )
    #fig.tight_layout(pad=3)

    #--------------------------
    #Differntial Intensities with respect to zero distance
    dx, dy = 5,5
    x = np.linspace(-dx/2,dx/2,500)
    y = np.linspace(-dy/2,dy/2,500)
    X,Y = np.meshgrid(x,y)
    N = 1E6

    fig1,ax1 = plt.subplots()
    ax1.pcolormesh(LG_mode(X, Y, 30,3,wavelength=1, aperture_diameter=.91)-LG_mode(X, Y, 1,1,wavelength=1, aperture_diameter=.91))
    ax1.set_aspect('equal')
    fig1.show()


    # zero-distance intensities
    zero_d = N*superposition_LG(X, Y, [1], [1], [1])#N*cum_do(X,Y,1E-5,0,FWHM)#np.random.poisson(N*cum_do(X,Y,1E-5,0,FWHM))
    # non-zero distance intensities
    diff_densities = []
    vmin = 0
    vmax = 0
    for p in phi:
        density_do = N*superposition_LG(X, Y, [1], [2], [1])#np.random.poisson(N*cum_do(X, Y, d,p,FWHM))
        # differential intensities
        diff_density = density_do - zero_d
        diff_densities.append(diff_density)

        # Determine the minimum and maximum values for the color scale
        if diff_density.min()<vmin:
            vmin = diff_density.min()
        if diff_density.max()>vmax:
            vmax = diff_density.max()
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    #norm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=vmin, vmax=vmax, base=10)

    for idx, p in enumerate(phi):
        axis = ax[str(p)]
        im = axis.pcolormesh(X,Y,diff_densities[idx],cmap=s.cmap_div,norm=norm)
        axis.scatter([-np.cos(p) * d / 2, +np.cos(p) * d / 2], [-np.sin(p)*d/2,np.sin(p)*d/2],marker='*',c=s.c30,s=2)
        axis.set_aspect('equal')
        axis.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
        #s.hide_axes(axes=[axis],dirs=['top','right'])

        # Create a colorbar axis
        cbar = fig.colorbar(im, cax=ax[f'cbar{str(idx)}'],location='right',drawedges=False,pad=0.0,format=lambda x, _: f"{x:.1f}",fraction=1.0)
        cbar.set_ticks([vmin, 0, vmax])  # Set the tick locations
        cbar.set_ticklabels(['min', '0', 'max'])

    if show:
        plt.show()
    return fig

if __name__=='__main__':
    fig = make_figure(show=True)
    #Figures().save_fig(fig, 'orientation-figure',meta={'generating script': script_name})