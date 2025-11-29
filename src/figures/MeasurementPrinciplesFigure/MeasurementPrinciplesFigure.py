"""
Reproduce the measurement principle figure illustrating linescans and MINFLUX.

copyright: @Thomas Hensel, 2024
"""

import os
script_name = os.path.basename(__file__)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
from scipy.special import j1


from lib.data_handling.data_converter import Constructor, Converter
from lib.data_handling.data_analysis import Estimators
import lib.utilities as ut
from lib.constants import *
from lib.plotting.style import Style
from lib.plotting.artefacts import Figures

import pylustrator
pylustrator.start()

def airy_do_pattern(x, y, phi, wavelength=1, aperture_diameter=1):
    k = 2 * np.pi / wavelength
    r = np.sqrt(x**2 + y**2)
    theta = k * aperture_diameter / 1 * r
    pattern = np.sin(k*x-phi)**2 * (2 * j1(theta) / theta)**2
    return pattern / np.max(pattern)

def plotPSF(axis, string, density, X, Y, norm, style):
    s=style
    im = axis.pcolormesh(X,Y,density,norm=norm,cmap=s.cmap_seq,rasterized=True)
    axis.text(0.5, .98, string, transform=axis.transAxes,
                verticalalignment='top',horizontalalignment='center',color='white'
                )
    axis.set_aspect('equal')
    axis.tick_params(direction='in')
    s.drop_axes(axes=[axis])
    ss = axis.get_subplotspec()
    if ss.is_first_col():
        if ss.is_first_row():
            s.hide_axes(axes=[axis],dirs=['top','right','bottom'])
            axis.set_ylabel('y (FWHM)')
        else:
            s.hide_axes(axes=[axis],dirs=['top','right'])
            axis.set_ylabel('y (FWHM)')
            axis.set_xlabel('x (FWHM)')
    else:
        if ss.is_last_row():
            s.hide_axes(axes=[axis],dirs=['top','left','right'])
            axis.set_xlabel('x (FWHM)')
        else:
            s.hide_axes(axes=[axis])
    return im
    
def plotSchematics(axs,style):
    s=style
    def attCos(array):
        return .5*(1+0.8*np.cos(array*np.pi))
    
    def partialCos(array,noise=False):
        condition1 = (2*0.27 < array) & (array < 2*0.37)
        condition2 = (2*0.45 < array) & (array < 2*0.55)
        condition3 = (2*0.63 < array) & (array < 2*0.73)
        # Combine the conditions using logical OR (|)
        result = condition1 | condition2 | condition3
        y = attCos(array)
        if noise:
            y = np.random.poisson(20*y)/20
        y[~result]=np.nan
        return y
    
    x = 2*np.sort(np.random.random(1000))
    y = attCos(x)
    axs[0].plot(x,y,color='k')
    #ax['s1'].fill_between(x, -1, y, color=s.c10, alpha=0.4)
    axs[0].scatter(2*.45,0.,marker='*',s=100, color=s.c30,zorder=5)
    axs[0].scatter(2*.55,0.,marker='*',s=100, color=s.c30,zorder=5)
    axs[0].set_xticks([0,1,2])
    axs[0].set_yticks([0,1])
    axs[0].set_xlim(0,2)
    axs[0].set_ylim(-.2,1)
    axs[0].spines['right'].set_bounds(0,1)
    axs[0].spines['bottom'].set_bounds(0,2)
    axs[0].set_ylabel('counts (a.u.)')
    axs[0].yaxis.set_label_position("right")
    axs[0].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=False,labelleft=False,right=True,labelright=True,direction='in')
    axs[0].text(0.02, .0, 'Line Scan', transform=axs[0].transAxes,
                verticalalignment='bottom',horizontalalignment='left',color='k'
                )

    y = partialCos(x,noise=True)
    axs[1].scatter(x,y,color='k',s=1)
    #ax['s1'].fill_between(x, -1, y, color=s.c10, alpha=0.4)
    axs[1].scatter(2*.45,0.,marker='*',s=100, color=s.c30,zorder=5)
    axs[1].scatter(2*.55,0.,marker='*',s=100, color=s.c30,zorder=5)
    axs[1].set_xticks([0,1,2])
    axs[1].set_yticks([0,1])
    axs[1].set_xlim(0,2)
    axs[1].set_ylim(-.2,1)
    axs[1].spines['right'].set_bounds(0,1)
    axs[1].spines['bottom'].set_bounds(0,2)
    axs[1].set_xlabel(r'$\phi$ ($\pi$)')
    axs[1].set_ylabel('counts (a.u.)')
    axs[1].yaxis.set_label_position("right")
    axs[1].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=False,labelleft=False,right=True,labelright=True,direction='in')
    axs[1].text(0.02, .0, 'MINFLUX', transform=axs[1].transAxes,
            verticalalignment='bottom',horizontalalignment='left',color='k'
            )

    for ax in axs.flat:
        s.drop_axes(axes=[ax])
        s.hide_axes(axes=[ax], dirs=['top','left'])
    pass

def plotLineScanData(fig, axs, imarray,style):
    s = style
    x = np.linspace(0, 2, imarray.shape[1])
    y = np.linspace(0, 1, imarray.shape[0])
    X,Y = np.meshgrid(x,y)
    pcol = axs.pcolormesh(X,Y, imarray,cmap=s.cmap_seq,linewidth=0,rasterized=True)
    pcol.set_edgecolor('face')
    cbar = fig.colorbar(pcol, ax=axs,location='right',drawedges=False,pad=0.02)
    cbar.set_ticks([0,1])  # Set the tick locations
    cbar.set_ticklabels(['min', 'max'])  # Set the tick labels
    axs.set_ylim(0,1)
    axs.set_xlim(0,2)
    axs.set_ylabel('time (a.u.)')
    axs.set_xlabel(r'$\phi$ ($\pi$)')
    axs.set_yticks([0,1])
    axs.set_xticks([0,1,2])
    axs.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
    s.drop_axes(axes=[axs],dirs=['left','bottom'])
    s.hide_axes(axes=[axs],dirs=['top','right'])
    axs.invert_yaxis()

def plotLineScanTrace(axs, line_avgs, style):
    s = style
    norm = 2*line_avgs[0]#/max(line_avgs[0])
    axs.plot(norm,np.linspace(0,1,len(norm)),color=s.c10,lw=1.5)
    norm = 2*line_avgs[1]#/max(line_avgs[1])
    axs.plot(norm,np.linspace(0,1,len(norm)),color=s.c20,lw=1.5)
    axs.invert_yaxis()
    axs.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
    axs.set_xlim(0,1)
    axs.set_xticks([0,1])
    axs.set_ylim(1,0)
    axs.set_yticks([0,1])
    s.drop_axes(axes=[axs],dirs=['left','bottom'])
    s.hide_axes(axes=[axs],dirs=['top','right'])
    axs.set_xlabel(r'counts (a.u.)')
    axs.set_ylabel('time (a.u.)')
    pass

def plotLineScans(axs,x_imarray,y_imarray,style):
    s=style
    axs[0].plot(np.linspace(0,2,len(x_imarray[0])),x_imarray[0],color=s.c10,lw=1.5)
    axs[0].plot(np.linspace(0,2,len(y_imarray[0])),y_imarray[0],color=s.c20,lw=1.5)
    axs[0].axvline(x=1.07,ymin=0.,ymax=1.,color=s.c10,alpha=0.5)
    axs[0].axvline(x=0.755,ymin=0.,ymax=1.,color=s.c20,alpha=0.5)
    axs[0].sharex(axs[1])
    axs[0].set_ylim(0,1)
    axs[0].set_yticks([0,1])
    axs[0].tick_params(top=False, labeltop=False, bottom=True, labelbottom=False,left=False,labelleft=False,right=True,labelright=True,direction='in')
    axs[0].set_ylabel(r'counts (a.u.)')
    axs[0].yaxis.set_label_position("right")

    axs[1].plot(np.linspace(0,2,len(x_imarray[0])),x_imarray[60],color=s.c10,lw=1.5,label='x')
    axs[1].plot(np.linspace(0,2,len(y_imarray[0])),y_imarray[60],color=s.c20,lw=1.5,label='y')
    axs[1].axvline(x=1.,ymin=0.,ymax=1.,color=s.c10,alpha=0.5)
    axs[1].axvline(x=0.62,ymin=0.,ymax=1.,color=s.c20,alpha=0.5)
    axs[1].annotate("", xytext=(.63, .9), xy=(.8, .9), xycoords=('data','axes fraction'),arrowprops=dict(arrowstyle="<-",linewidth=1.))
    axs[1].annotate("", xytext=(1., .9), xy=(1.2, .9), xycoords=('data','axes fraction'),arrowprops=dict(arrowstyle="<-",linewidth=1.))
    axs[1].set_xlim(0,2)
    axs[1].set_xticks([0,1,2])
    axs[1].set_ylim(0,1)
    axs[1].set_yticks([0,1])
    axs[1].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=False,labelleft=False,right=True,labelright=True,direction='in')
    axs[1].set_ylabel(r'counts (a.u.)')
    axs[1].yaxis.set_label_position("right")
    axs[1].set_xlabel(r'$\phi$ ($\pi$)')
    axs[1].legend(bbox_to_anchor=(0.01, 0.5), loc='lower left')

    for ax in axs.flat:
        s.drop_axes(axes=[ax],dirs=['right','bottom'])
        s.hide_axes(axes=[ax], dirs=['top','left'])
    pass

def plotMINFLUXData(fig, axs,df,style):
    s=style
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=1)
    sub_df = df.groupby('axis').apply(lambda x: x.iloc[1::3]).reset_index(drop=True).groupby('axis',group_keys=False).rolling(window=indexer,min_periods=1).mean().reset_index(level='axis')
    sub_df.pos = sub_df.pos*2/LAMBDA+.8
    dfx = sub_df.loc[(sub_df['axis']==0)&(sub_df['state_id']==2)].reset_index(drop=True)
    dfy = sub_df.loc[(sub_df['axis']==1)&(sub_df['state_id']==2)].reset_index(drop=True)
    mean2 = np.asarray((dfx.pos.mean(),dfy.pos.mean()))
    dfx = sub_df.loc[(sub_df['axis']==0)&(sub_df['state_id']==1)].reset_index(drop=True)
    dfy = sub_df.loc[(sub_df['axis']==1)&(sub_df['state_id']==1)].reset_index(drop=True)
    mean1 = np.asarray((dfx.pos.mean(),dfy.pos.mean()))

    # calculate kde for 2 Molecule trace
    xi, yi = np.mgrid[0:2:100j, 0:2:100j]# Create a meshgrid for the x and y positions
    xy = np.vstack([xi.ravel(), yi.ravel()])# Combine the x and y positions into a single array
    dfx, dfy = sub_df.loc[(sub_df['axis']==0)&(sub_df['state_id']==2)].reset_index(drop=True),sub_df.loc[(sub_df['axis']==1)&(sub_df['state_id']==2)].reset_index(drop=True)
    df_weights = dfx.photons.to_numpy()+dfy.photons.to_numpy()
    kde_2M = gaussian_kde(np.vstack([dfx.pos, dfy.pos]), weights=df_weights,bw_method=.1)
    # calculate kde for 1 Molecule trace
    dfx, dfy = sub_df.loc[(sub_df['axis']==0)&(sub_df['state_id']==1)].reset_index(drop=True),sub_df.loc[(sub_df['axis']==1)&(sub_df['state_id']==1)].reset_index(drop=True)
    df_weights = dfx.photons.to_numpy()+dfy.photons.to_numpy()
    kde_1M = gaussian_kde(np.vstack([dfx.pos, dfy.pos]), weights=df_weights,bw_method=.1)
    # weighted and combined kernel density estimate on the meshgrid
    kde = lambda xy : kde_2M(xy)/np.sum(kde_2M(xy)) + .7*kde_1M(xy)/np.sum(kde_1M(xy))

    zi = kde(xy).reshape(xi.shape)
    pcol = axs.pcolormesh(xi,yi,zi,cmap=s.cmap_seq,linewidth=0,rasterized=True)
    pcol.set_edgecolor('face')
    cbar = fig.colorbar(pcol, ax=axs,location='right',drawedges=False,pad=0.02)
    cbar.set_ticks([min(zi.flatten()), max(zi.flatten())])  # Set the tick locations
    cbar.set_ticklabels(['min', 'max'])  # Set the tick labels
    
    # now create zoom-in inset: resample kde in subregion:
    xi, yi = np.mgrid[0.75:.87:50j, .82:.89:50j]
    xy = np.vstack([xi.ravel(), yi.ravel()])
    zi = kde(xy).reshape(xi.shape)
    axins = axs.inset_axes([0.32, 0.52, 0.65, 0.5])
    axins.spines['bottom'].set_color('white')
    axins.spines['top'].set_color('white') 
    axins.spines['right'].set_color('white')
    axins.spines['left'].set_color('white')
    axins.set_xlim(.7, .9)
    axins.set_ylim(.8, .9)
    axs.indicate_inset_zoom(axins, edgecolor="white")
    xmin, xmax = np.min((xi-.8)*LAMBDA/2), np.max((xi-.8)*LAMBDA/2)
    ymin, ymax = np.min((yi-.8)*LAMBDA/2), np.max((yi-.8)*LAMBDA/2)
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymin, ymax)
    pcol = axins.pcolormesh((xi-.8)*LAMBDA/2,(yi-.8)*LAMBDA/2,zi, cmap=s.cmap_seq,linewidth=0,rasterized=True)#
    
    pcol.set_edgecolor('face')
    axins.set_xlabel(r'$x$ (nm)',color='white')
    axins.set_ylabel(r'$y$ (nm)',color='white')
    axins.set_aspect('equal')
    axins.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in',colors='white')    

    # add the molecules
    mean1, mean2 = (mean1-np.array([.8,.8]))*LAMBDA/2, (mean2-np.array([.8,.8]))*LAMBDA/2
    or_vec = np.asarray(mean2)-np.asarray(mean1)
    ortho_vec = np.array([-or_vec[1], or_vec[0]])
    d = np.linalg.norm(or_vec)
    m1 = mean2 + or_vec + 5*ortho_vec/d
    m2 = mean2 - or_vec + 5*ortho_vec/d
    axins.scatter(m1[0],m1[1],marker='*',s=30,c=s.c30)
    axins.scatter(m2[0],m2[1],marker='*',s=30,c=s.c30)
    start, end = mean2 + 8*ortho_vec/d, mean1 + 8*ortho_vec/d
    axins.annotate("", xytext=start, xy=end, xycoords='data',arrowprops=dict(color='w',facecolor='w',arrowstyle="->",linewidth=1.))
    
    axs.set_xlabel(r'$\phi_x$ ($\pi$)')
    axs.set_ylabel(r'$\phi_y$ ($\pi$)')
    axs.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
    axs.set_xlim(0,2)
    axs.set_xticks([0,1,2])
    axs.set_ylim(0,2)
    axs.set_yticks([0,1,2])
    s.drop_axes(axes=[axs],dirs=['left','bottom'])
    s.hide_axes(axes=[axs],dirs=['top','right'])
    pass

def plotMINFLUXTrace(axs, df,style):
    s=style
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=50)
    sub_df = df.groupby(['axis'],group_keys=False).rolling(window=indexer,min_periods=1).mean().reset_index(level='axis').apply(lambda x: x/x.max(), axis=0)
    trace = sub_df.loc[sub_df['axis']==0].reset_index()#.head(28000).drop(columns='index')
    axs.plot(trace.photons,np.linspace(0,1,len(trace.index)),color=s.c10,lw=1.5)
    trace = sub_df.loc[sub_df['axis']==1].reset_index()#.head(28000).drop(columns='index')
    axs.plot(trace.photons,np.linspace(0,1,len(trace.index)),color=s.c20,lw=1.5)
    axs.invert_yaxis()
    axs.set_xlabel(r'counts (a.u.)')
    axs.set_ylabel('time (a.u.)')
    axs.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
    axs.set_xlim(0,1)
    axs.set_xticks([0,1])
    axs.set_ylim(1,0)
    axs.set_yticks([0,1])
    s.drop_axes(axes=[axs],dirs=['left','bottom'])
    s.hide_axes(axes=[axs],dirs=['top','right'])
    pass

def plotMINFLUXfits(axs, df, sol_df, style):
    s = style
    sol_df = sol_df.reset_index()

    # 2M, x
    norm_fac = 50
    sub_df = df.loc[(df.segment_id==0) & (df['axis']==0)]
    axs[0].scatter(sub_df.pos,sub_df.photons/norm_fac,color=s.c11,marker='.',s=1,alpha=.5)
    fit_x = np.linspace(min(sub_df.pos),max(sub_df.pos),100)
    sol = sol_df.loc[(sol_df.state_id==2) & (sol_df['axis']==0),['a0','a1','x0']].to_numpy()[0]
    minx_2M = sol[-1]*LAMBDA/(4*np.pi)
    fit_y = Estimators().min_poly_model(fit_x*4*np.pi/LAMBDA,sol)
    axs[0].plot(fit_x,fit_y/norm_fac,c=s.c10,ls='-',lw=1.5)#s.c10
    axs[0].axvline(x=minx_2M,ymin=0.,ymax=1.,color=s.c10,alpha=0.5)

    # 2M, y
    sub_df = df.loc[(df.segment_id==0) & (df['axis']==1)]
    axs[0].scatter(sub_df.pos,sub_df.photons/norm_fac,color=s.c21,marker='.',s=1)
    fit_x = np.linspace(min(sub_df.pos),max(sub_df.pos),100)
    sol = sol_df.loc[(sol_df.state_id==2) & (sol_df['axis']==1),['a0','a1','x0']].to_numpy()[0]
    miny_2M = sol[-1]*LAMBDA/(4*np.pi)
    fit_y = Estimators().min_poly_model(fit_x*4*np.pi/LAMBDA,sol)
    axs[0].plot(fit_x,fit_y/norm_fac,c=s.c20,ls='-',lw=1.5)
    axs[0].axvline(x=miny_2M,ymin=0.,ymax=1.,color=s.c20,alpha=0.5)

    # 1M, x
    sub_df = df.loc[(df.segment_id==1) & (df['axis']==0)]
    axs[1].scatter(sub_df.pos,sub_df.photons/norm_fac,color=s.c11,marker='.',s=1)
    fit_x = np.linspace(min(sub_df.pos),max(sub_df.pos),100)
    sol = sol_df.loc[(sol_df.state_id==1) & (sol_df['axis']==0),['a0','a1','x0']].to_numpy()[0]
    minx_1M = sol[-1]*LAMBDA/(4*np.pi)
    fit_y = Estimators().min_poly_model(fit_x*4*np.pi/LAMBDA,sol)
    axs[1].plot(fit_x,fit_y/norm_fac,c=s.c10,ls='-',lw=1.5,label='x')
    axs[1].axvline(x=minx_1M,ymin=0.,ymax=1.,color=s.c10,alpha=0.5)
    axs[1].annotate("", xytext=(minx_2M, .9), xy=(minx_1M, .9), xycoords=('data','axes fraction'),arrowprops=dict(facecolor='black',arrowstyle="->",linewidth=1.))
    
    # 1M, y
    sub_df = df.loc[(df.segment_id==1) & (df['axis']==1)]
    axs[1].scatter(sub_df.pos,sub_df.photons/norm_fac,color=s.c21,marker='.',s=1)
    fit_x = np.linspace(min(sub_df.pos),max(sub_df.pos),100)
    sol = sol_df.loc[(sol_df.state_id==1) & (sol_df['axis']==1),['a0','a1','x0']].to_numpy()[0]
    miny_1M = sol[-1]*LAMBDA/(4*np.pi)
    fit_y = Estimators().min_poly_model(fit_x*4*np.pi/LAMBDA,sol)
    axs[1].plot(fit_x,fit_y/norm_fac,c=s.c20,ls='-',lw=1.5,label='y')
    axs[1].axvline(x=miny_1M,ymin=0.,ymax=1.,color=s.c20,alpha=0.5)
    axs[1].annotate("", xytext=(miny_2M+7, .9), xy=(miny_1M, .9), xycoords=('data','axes fraction'),arrowprops=dict(facecolor='black',arrowstyle="->",linewidth=1.))

    axs[0].sharex(axs[1])
    axs[0].set_ylim(0,1)
    axs[0].set_yticks([0,1])
    axs[0].tick_params(top=False, labeltop=False, bottom=True, labelbottom=False,left=False,labelleft=False,right=True,labelright=True,direction='in')
    axs[0].set_ylabel(r'counts (a.u.)')
    axs[0].yaxis.set_label_position("right")

    axs[1].set_xlim(-30,45)
    axs[1].set_xticks([-30,-15,0,15,30,45])
    axs[1].set_ylim(0,1)
    axs[1].set_yticks([0,1])
    axs[1].set_ylabel(r'counts (a.u.)')
    axs[1].yaxis.set_label_position("right")
    axs[1].legend(bbox_to_anchor=(0.01, 0.5), loc='lower left')
    axs[1].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=False,labelleft=False,right=True,labelright=True,direction='in')
    axs[1].set_xlabel(r'$x$ (nm)')
    
    for ax in axs.flat:
        s.drop_axes(axes=[ax],dirs=['right','bottom'])
        s.hide_axes(axes=[ax], dirs=['top','left'])
    pass

def make_figure(plot_style='nature',color_scheme='default',show=True):
    s = Style(style=plot_style,color_scheme=color_scheme)
    base_dir = os.path.split(__file__)[0]

    #1st row: schematics of measurement principle
    #2nd row: LineScan principle
    #3rd row: MINFLUX principle
    fig = plt.figure(layout='constrained',figsize=s.get_figsize(rows=2.8,cols=2.5,ratio=1))#
    subfigs = fig.subfigures(2, 1, hspace=0.05, width_ratios=[1],height_ratios=[.3,.7])

    for f in subfigs:
        rect = f.patch
        rect.set_facecolor('none')

    #------------------------------------------
    PSFs = [
        ['x1','x2','x3']
        ,['y1','y2','y3']
    ]
    Principles = [
        ['LineScanPrinciple']
        ,['MINFLUXPrinciple']
    ]

    mosaic = [
            ['.','.','.']
            ,['scheme','psfs',Principles]
        ]
    axs = subfigs[0].subplot_mosaic(#fig.subplot_mosaic(#
        mosaic
    ,gridspec_kw = {
        "width_ratios": [.2,.6,.4]
        ,"height_ratios": [.1,1]
        }
    #,constrained_layout=True#'constrained'
    #,figsize=s.get_figsize(rows=3.1,cols=3,ratio=1)
    )
    

    #--------------
    # modify gridspec
    """gs = axs[0, -1].get_gridspec()
    for ax in axs[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[1:, -1])"""

    gs = axs['psfs'].get_gridspec()
    nrow, ncol, start, stop = axs['psfs'].get_subplotspec().get_geometry()
    start_row, start_col = divmod(start, ncol)
    stop_row, stop_col = divmod(stop, ncol)
    axs['psfs'].remove()
    inner_grid = gs[start_row:stop_row+1,start_col:stop_col+1].subgridspec(2, 4, wspace=0.0, hspace=0.0,height_ratios=[1,1],width_ratios=[.3,.3,.3,.03])

    # plot PSFs
    x = np.linspace(-.5,.5,200)
    y = np.linspace(-.5,.5,200)
    X,Y = np.meshgrid(x,y)
    phases = [0.8*np.pi,np.pi,1.2*np.pi]
    strings = [[r'$\mathbf{\phi_x=-L/2}$',r'$\mathbf{\phi_x=0}$',r'$\mathbf{\phi_x=+L/2}$'],[r'$\mathbf{\phi_y=-L/2}$',r'$\mathbf{\phi_y=0}$',r'$\mathbf{\phi_y=+L/2}$']]
    # get common norm
    density = airy_do_pattern(X, Y, np.pi)
    vmin = density.min()
    vmax = density.max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    for r in range(inner_grid._nrows):
        for c in range(inner_grid._ncols-1):
            string = strings[r][c]
            phi = phases[c]
            axis = subfigs[0].add_subplot(inner_grid[r, c])
            if r==0:
                density = airy_do_pattern(X, Y, phi)
            else:
                density = airy_do_pattern(Y, X, phi)
            im = plotPSF(axis,string, density, X, Y,norm,s)
    
    # add colorbar to gridspec of PSF plots
    cax = subfigs[0].add_subplot(inner_grid[:, -1])
    cbar = subfigs[0].colorbar(im, cax=cax,location='right',drawedges=False,pad=0.0,format=lambda x, _: f"{x:.1f}")
    cbar.set_ticks([vmin, vmax])  # Set the tick locations
    cbar.set_ticklabels(['min', 'max'])
    
    # plot schematic of microscope
    image = plt.imread(base_dir+'/data/setup.png')
    axs['scheme'].imshow(image)
    s.drop_axes(axes=[axs['scheme']])
    s.hide_axes(axes=[axs['scheme']])

    # plot Line Scan and MINFLUX schematic
    plotSchematics(np.array([axs['LineScanPrinciple'],axs['MINFLUXPrinciple']],dtype=object),s)

    
    #-------------------------------
    Lines = [
        ['Line2M']
        ,['Line1M']
        ]
    MINFLUX = [
        ['MINFLUX2M']
        ,['MINFLUX1M']
        ]

    mosaic = [
            ['.','.','.']
            ,['LineScanData','ScanTrace',Lines]
            ,['.','.','.']
            ,['MINFLUXData','MINFLUXTrace',MINFLUX]
        ]
    axs = subfigs[1].subplot_mosaic(#fig.subplot_mosaic(#
        mosaic
    ,gridspec_kw = {
        "width_ratios": [.35,.1,.55]
        ,"height_ratios": [.1,1,.2,1]
        }
    )

    rect = subfigs[1].patch
    rect.set_facecolor('none')

    #-------------------------
    # Second row: LineScans
    #-------------------------
    file = ut.BaseFunc().find_files(base_dir, lambda file: file.endswith('.tif'), max_files=1)[0]
    curr_constructor = Constructor(file,collect_artefacts=False)
    imarray = curr_constructor.ext.get_array_from_tif(file)
    imarray = imarray/(np.concatenate(imarray).max())
    experiments = curr_constructor.get_experiments(batchsize=1, agnostic=True, fast_return=False)
    x_imarray, y_imarray = Converter().partition_array(imarray, curr_constructor.ext.block_size)
    line_avgs = np.average([x_imarray,y_imarray],axis=2)
    seg_idcs = [exp.measurement.config.seg_idcs for exp in experiments]

    # Visualization of LineScan trace
    plotLineScanData(fig, axs['LineScanData'], imarray,s)
    # time-averaged counts
    plotLineScanTrace(axs['ScanTrace'], line_avgs, s)
    # single lines
    plotLineScans(np.array([axs['Line2M'],axs['Line1M']]),x_imarray,y_imarray,s)

    #-------------------
    # Third Row: MINFLUX principle
    #-------------------
    file = ut.BaseFunc().find_files(base_dir+'/data/', lambda file: file.endswith('filtered.pkl'), max_files=1)[0]
    df =  pd.read_pickle(file)
    df.drop(columns=['experiment_id'],inplace=True)
    
    # plot spatial MINFLUX trace
    plotMINFLUXData(fig, axs['MINFLUXData'],df,s)

    # time averaged MINFLUX counts
    file = ut.BaseFunc().find_files(base_dir+'/data/', lambda file: file.endswith('filtered.pkl'), max_files=1)[0]
    df =  pd.read_pickle(file)
    df.drop(columns=['experiment_id'],inplace=True)
    plotMINFLUXTrace(axs['MINFLUXTrace'], df, s)

    # quadratic fit--------------------
    file = ut.BaseFunc().find_files(base_dir, lambda file: ut.BaseFunc().match_pattern(file, 'local-results', match='partial'), max_files=1)[0]
    sol_df =  pd.read_pickle(file)
    plotMINFLUXfits(np.array([axs['MINFLUX2M'],axs['MINFLUX1M']],dtype=object), df, sol_df, s)
    
    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
    plt.figure(1).text(0.0083, 0.9731, 'a', transform=plt.figure(1).transFigure, fontsize=15., weight='bold',zorder=100)  # id=plt.figure(1).texts[0].new
    plt.figure(1).text(0.2090, 0.9731, 'b', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[1].new
    plt.figure(1).text(0.7073, 0.9731, 'c', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[2].new
    plt.figure(1).text(0.0083, 0.6633, 'd', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[3].new
    plt.figure(1).text(0.4282, 0.6633, 'e', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[4].new
    plt.figure(1).text(0.5698, 0.6633, 'f', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[5].new
    plt.figure(1).text(0.0083, 0.3129, 'g', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[6].new
    plt.figure(1).text(0.4282, 0.3129, 'h', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[7].new
    plt.figure(1).text(0.5821, 0.3129, 'i', transform=plt.figure(1).transFigure, fontsize=15., weight='bold')  # id=plt.figure(1).texts[8].new
    #% end: automatic generated code from pylustrator
    if show:
        plt.show()
    return fig


if __name__=='__main__':
    fig = make_figure(show=True)
    #Figures().save_fig(fig, 'measurement-principle-figure',meta={'generating script': script_name})

    