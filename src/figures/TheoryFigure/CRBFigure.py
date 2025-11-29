"""
Plot the Cramer Rao Bound for the separation estimation of two fluorophores.

copyright: @Thomas Hensel, 2024
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt

from lib.constants import *
from lib.plotting.style import Style

import pylustrator
pylustrator.start()

def h1M(x,xl):
    return np.sin(2*np.pi*(x-xl)/LAMBDA)**2

def dh1M(x,xl):
    return np.sin(2*np.pi*(x-xl)/LAMBDA) * np.cos(2*np.pi*(x-xl)/LAMBDA) * 2*np.pi/LAMBDA

def h2M(x1,x2,xl):
    return h1M(x1,xl) + h1M(x2,xl)

def dh2M(x1,x2,xl):
    return dh1M(x1,xl) + dh1M(x2,xl)

def signal(h):
    """
    cumulative mean signal of molecules with background and scaled to incident brightness
    """
    return ALPHA * DWELL_TIME * (BACKGROUND_RATE + h)

def dsignal(dh):
    return ALPHA * DWELL_TIME * dh

def F_info(x1,x2,xi,xj,xl):
    """
    ij-th matrix element of FIM for l-th measurement of molecules at x1,x2
    """
    return 1/signal(h2M(x1,x2,xl)) * dsignal(dh1M(xi,xl)) * dsignal(dh1M(xj,xl))

def F_trace(x1, x2, xl):
    """
    Trace of FIM for l-th measurement of molecules at x1, x2
    """
    return F_info(x1,x2,x1,x1,xl) + F_info(x1,x2,x2,x2,xl)

def quadCRB(xc,L,pos,N):
    """
    CRB for general measurement of two molecules at xc-L/2 and xc+L/2 in quadratic approximation.
    """
    xi1=pos[0]
    xi2=pos[1]
    sig1 = 0.25 * np.sqrt(
            (L**2 + 2*(2*xc**2 + xi1**2 + xi2**2 - 2*xc*(xi1+xi2)))\
            * (L**4 - 8*L**2*(xc-xi1)*(xc-xi2) + 4*(2*xc**2 + xi1**2 + xi2**2 - 2*xc*(xi1+xi2))**2
            - np.sqrt(
                (L**2 - 2*(2*xc**2 + xi1**2 + xi2**2 - 2*xc*(xi1+xi2)))**2\
                * (L**4 - 8*L**2*(xc-xi1)*(xc-xi2) + 4*(2*xc**2 + xi1**2 + xi2**2 - 2*xc*(xi1+xi2))**2
                )
            )
            )/(L**2*N*(xi1-xi2)**2)
    )
    sig2 = 0.25 * np.sqrt(
            (L**2 + 2*(2*xc**2 + xi1**2 + xi2**2 - 2*xc*(xi1+xi2)))\
            * (L**4 - 8*L**2*(xc-xi1)*(xc-xi2) + 4*(2*xc**2 + xi1**2 + xi2**2 - 2*xc*(xi1+xi2))**2
            + np.sqrt(
                (L**2 - 2*(2*xc**2 + xi1**2 + xi2**2 - 2*xc*(xi1+xi2)))**2\
                * (L**4 - 8*L**2*(xc-xi1)*(xc-xi2) + 4*(2*xc**2 + xi1**2 + xi2**2 - 2*xc*(xi1+xi2))**2
                )
            )
            )/(L**2*N*(xi1-xi2)**2)
    )
    return sig1, sig2

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

def maxCRB(dx,l,N,v):
    """CRB at maximum in units of Lambda"""
    DX = dx * 2*np.pi
    L = l * 2*np.pi

    aux0=2.*(1.+((2.**-0.5)*(v*((mysqrt((1.+(np.cos(DX)))))*(np.cos((0.5*\
    L)))))))
    aux1=(1.+(((2.**-0.5)*(v*(mysqrt((1.+(np.cos(DX)))))))+aux0))*(((1/np.\
    sin(DX))**2))
    aux2=(2.*(1.+(np.cos(L))))/(2.+((mysqrt(2.))*(v*((mysqrt((1.+(np.cos(\
    DX)))))*(np.cos((0.5*L)))))))
    aux3=((v**-2.)*((1.+(np.cos(DX)))*aux1))/(((1.+(v*(mysqrt((((np.cos((\
    0.5*DX)))**2))))))**-1.)+aux2)
    output=2.*((mysqrt(2.))*(mysqrt((aux3/N))))

    sig = output/(2*np.pi)
    return sig

def plot_resolvability(axis,N,s):
    # resolvability with respect to initial contrast
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

    y = np.linspace(.5,9999E-4,200,endpoint=True)
    x = np.asarray([find_min_d(v,.0,N) for v in y]).flatten()
    xL1 = np.asarray([find_min_d(v,.1,N) for v in y]).flatten()
    xL2 = np.asarray([find_min_d(v,.2,N) for v in y]).flatten()
    
    yv = np.asarray([2,5,10,20])/LAMBDA#d in units of Lambda
    xv = np.asarray([find_min_v(d,.0,N) for d in yv]).flatten()
    data = [['d (nm)', r'$\nu_0$']]  # Header row
    data.extend(zip(list(map(round,yv*LAMBDA)), list(map(lambda x: round(x,3),xv))))


    axis.fill_between(x=x,y1=0, y2=y, facecolor='gray', alpha=0.2)
    #ax['d)'].fill_between(x=x,y1=y, y2=0, hatch='XX', facecolor='none', edgecolor=s.c11, alpha=0.5)
    #ax['d)'].fill_between(x=x,y1=0,y2=1, where=(x>=0.9),hatch='\\\\', facecolor='none', edgecolor='k', alpha=0.3,label='experimentally relevant')
    axis.plot(x,y,c=s.c10)#label=r'L=0.0$\lambda$'

    # Corrected code for setting cell backgrounds to white
    num_rows = len(data)
    num_cols = len(data[0])
    cell_colors = [['w'] * num_cols for _ in range(num_rows)]

    # Create the table using the corrected cellColors parameter
    table = axis.table(cellText=data, loc='lower left', colWidths=[0.18] * num_cols, cellLoc='center', cellColours=cell_colors,zorder=5)
    table.auto_set_font_size(False)
    table.set_fontsize(5)

    axis.set_ylim(min(y),max(y))
    axis.set_xlim(0.0,.15)
    axis.set_yticks([0.5,.6,.7,.8,.9,1.])
    axis.set_xticks([0.0,.05,.1,.15])
    axis.set_xlabel(r'd ($\lambda$)')
    axis.set_ylabel(r'minimally required $\nu_0$')
    #axis.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right')

def plot_relCRBd(axis,N,s):
    dx=np.linspace(0,.3,200)#dx in units of lambda
    axis.plot(dx, np.abs(minCRB(dx,.0,N,1.0)/dx),c=s.c10,label=r'min, $\nu_0=1.0$')
    axis.plot(dx, np.abs(minCRB(dx,.0,N,.95)/dx),c=s.c10,alpha=.3,label=r'min, $\nu_0=.95$')
    axis.plot(dx, np.abs(maxCRB(dx,.0,N,1.0)/dx),c=s.c20,ls='-',label=r'max, $\nu_0=1.0$')
    
    #axis.axhline(y=.5,c='k',label=r'$\sigma_{CRB}/d=0.5$')
    #axis.fill_between(x=dx,y1=.5, y2=.01, facecolor=s.c11, alpha=0.2)

    axis.set_yscale('log',base=10)
    axis.set_xlim(dx.min(),dx.max())
    axis.set_ylim(1E-2,1E3)
    axis.set_xlabel(r'd ($\lambda$)')
    axis.set_ylabel(r'$\sigma_{CRB}/d$')
    axis.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right',frameon=True)#, title= f'N={round(N,0)}'
    axis.set_xticks([0,.1,.2,.3])
    axis.set_yticks([1E-2,1E-1,1E0,1E1,1E2,1E3])

def plot_relCRBL(axis,N,s):
    #------------------------------------------------------
    L = np.linspace(0,.3,100)
    dist = np.asarray([1E-2,5E-2,1E-1])#molecules distance in units of lambda
    for i,d in enumerate(dist):
        alpha = 1-i/len(dist)
        axis.plot(L,minCRB(d,L,N,1)/d,label=r'min, $d=$'+fr'{round(d,2)}$\lambda$',c=s.c10,alpha=alpha,ls='-')
        #axis.plot(L,maxCRB(d,L,N,1)/d,label=r'max, $d=$'+fr'{round(d,2)}$\lambda$',c=s.c20,alpha=alpha,ls='-') # 1st eigensigma for minimum
    axis.set_xlim(L.min(),L.max())
    axis.set_ylim(1E-2,1E0)
    axis.set_xlabel(r'L ($\lambda$)')
    axis.set_ylabel(r'$\sigma_{CRB}/d$')
    axis.set_yscale('log',base=10)
    axis.legend(bbox_to_anchor=(0.98, 0.02), loc='lower right',frameon=True)#, title= f'N={round(N,0)}'
    axis.set_xticks([0,.1,.2,.3])


def make_figure(plot_style='nature',color_scheme='default',show=True):
    s = Style(style=plot_style,color_scheme=color_scheme)
    N=10**2

    fig, ax = plt.subplot_mosaic(
            [
                ['.','.','.']
                ,['relCRB(d)','res','relCRB(L)']
            ]
        ,empty_sentinel='.'
        ,gridspec_kw = {
            "width_ratios": [1,1,1]
            ,"height_ratios": [.2,1]
            }
        ,constrained_layout=True
        ,figsize=s.get_figsize(cols=3, rows=1,ratio=1)#(7.086,6.)
        )
    #fig.suptitle("Comparison of CRB for max and min (1D)")

    #*************************************************************************************

    plot_relCRBd(ax['relCRB(d)'],N,s)

    plot_resolvability(ax['res'],N,s)

    plot_relCRBL(ax['relCRB(L)'],N,s)

    s.drop_axes(axes=[ax['relCRB(d)'],ax['res'],ax['relCRB(L)']],dirs=['bottom','left'])
    s.hide_axes(axes=[ax['relCRB(d)'],ax['res'],ax['relCRB(L)']],dirs=['top','right'])
    
    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
    plt.figure(1).ax_dict["res"].text(0.3970, 0.0363, 'unresolved', transform=plt.figure(1).ax_dict["res"].transAxes, fontsize=10.)  # id=plt.figure(1).ax_dict["res"].texts[0].new
    plt.figure(1).ax_dict["res"].text(0.3970, 0.7885, 'resolved', transform=plt.figure(1).ax_dict["res"].transAxes, fontsize=10.)  # id=plt.figure(1).ax_dict["res"].texts[1].new
    plt.figure(1).text(0.0274, 0.9131, 'a', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[0].new
    plt.figure(1).text(0.3623, 0.9131, 'b', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[1].new
    plt.figure(1).text(0.7011, 0.9131, 'c', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[2].new
    #% end: automatic generated code from pylustrator
    if show:
        plt.show()
    return fig

if __name__=='__main__':
    fig = make_figure(show=True)
    #path_to_figure = Figures().save_fig(fig, 'theory-figure')