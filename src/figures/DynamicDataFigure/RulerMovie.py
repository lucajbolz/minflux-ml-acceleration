"""
Reproduce the MINFLUX experiment movie of dynamic systems.

copyright: @Thomas Hensel, 2024
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import pandas as pd
import numpy as np
from functools import partial

import lib.utilities as ut
from lib.constants import LAMBDA
from lib.plotting.style import Style, ImageHandler
from lib.config import DATA_DIR, ROOT_DIR

def generate_data(filtered_df, names, dx, dy, dt, time_values):
    points, comPoints, arrs = [], [], []
    sig = 4
    for tidx,time in enumerate(time_values):
        p = tidx/len(time_values)
        for nidx,name in enumerate(names):
            curr_df = filtered_df.loc[(filtered_df['state_id']==2)&(filtered_df['file'].apply(lambda x: name in x))].copy()
            rolling_df = curr_df.groupby(['axis'],group_keys=False).rolling(window=10,min_periods=1).mean(['COM','pos1','pos2']).reset_index()#10
            try:
                tmp_df = rolling_df.loc[(rolling_df['time']>=max(0,time-dt)) & (rolling_df['time']<=time)]
                sub_df = pd.melt(tmp_df.loc[:,['axis','chunk_id','bin_id','time','COM','pos1','pos2']],id_vars=['axis','chunk_id','bin_id','time'],value_vars=['COM','pos1','pos2']).sort_values(['bin_id','chunk_id','time']).reset_index(drop=True)
                dfx, dfy = sub_df.loc[sub_df['axis']==0],sub_df.loc[sub_df['axis']==1]
                x_data, y_data, comx, comy = dfx.loc[dfx['variable'].apply(lambda x: ('pos1' in x)|('pos2' in x)),'value'].values, dfy.loc[dfy['variable'].apply(lambda x: ('pos1' in x)|('pos2' in x)),'value'].values, dfx.loc[dfx['variable'].apply(lambda x: 'COM' in x),'value'].values, dfy.loc[dfy['variable'].apply(lambda x: 'COM' in x),'value'].values
                            
                x_bins = np.linspace(-dx/2, dx/2, num=241)  # Adjust the number of bins as needed
                y_bins = np.linspace(-dy/2, dy/2, num=241)
                zi, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=(x_bins, y_bins))
                sigma = sig
                # Define the size of padding
                padding_size = 2*sigma  # Adjust this based on your requirements
                # Step 1: Pad the zi array
                padded_zi = np.pad(zi, pad_width=padding_size, mode='constant', constant_values=0)
                # Step 2: Apply the Gaussian filter
                smoothed_zi = gaussian_filter(padded_zi, sigma=sigma)
                # Step 3: Remove the padding
                original_shape = zi.shape
                smoothed_zi = smoothed_zi[padding_size:padding_size+original_shape[0], padding_size:padding_size+original_shape[1]]
                new_zi = smoothed_zi                
            except:
                try:
                    new_zi = arrs[-2]
                except:
                    continue
            arrs.append(new_zi)
            
            # COM position
            try:
                tmp_df = curr_df.loc[(curr_df['time']<=max(0,time))].copy()
                rolling_df = tmp_df.groupby(['axis'],group_keys=False).rolling(window=10,min_periods=1).mean(['COM','pos1','pos2']).reset_index()
                sub_df = pd.melt(rolling_df.loc[:,['axis','chunk_id','bin_id','COM','pos1','pos2']],id_vars=['axis','chunk_id','bin_id'],value_vars=['COM','pos1','pos2']).sort_values(['bin_id','chunk_id']).reset_index(drop=True)
                dfx, dfy = sub_df.loc[sub_df['axis']==0],sub_df.loc[sub_df['axis']==1]
                x_data, y_data, comx, comy = dfx.loc[dfx['variable'].apply(lambda x: ('pos1' in x)|('pos2' in x)),'value'].values, dfy.loc[dfy['variable'].apply(lambda x: ('pos1' in x)|('pos2' in x)),'value'].values, dfx.loc[dfx['variable'].apply(lambda x: 'COM' in x),'value'].values, dfy.loc[dfy['variable'].apply(lambda x: 'COM' in x),'value'].values
            except:
                try:
                    x_data, y_data = points[-1]
                except:
                    continue
            points.append(np.vstack([x_data,y_data]))
            comPoints.append(np.vstack([comx,comy]))
    return arrs, points, comPoints

class AnimatedPlot:
    
    def __init__(self,style):
        self.s = style
        self.custom_handler = ImageHandler()
        pass
    
    def set_axis(self, current_time, axes, not_refresh_axes, dx, dy):
        for axis in axes.values():
            if not axis._label in not_refresh_axes:
                axis.clear()
        patch1 = mpatches.Patch(color='white', label='centre-of-mass')
        patch2 = mpatches.Circle((0, 0), 0.1, color=self.s.cmap_seq(.5),label='fluorophores')
        handles = [patch1,patch2]
        labels = [patch1._label,patch2._label]
        
        axes['a'].set_title(f'Time: {current_time} ms',loc='left')
        axes['a'].text(0.05, 0.9, r'd = 15 nm', horizontalalignment='left', verticalalignment='baseline', 
                transform=axes['a'].transAxes,fontsize=10, color='white')
        axes['a'].set_xlim(-dx/2,dx/2)
        axes['a'].set_ylim(-dy/2,dy/2)
        axes['a'].set_aspect('equal')
        axes['a'].set_xlabel('x (nm)')
        axes['a'].set_ylabel('y (nm)')
        axes['a'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')
        leg = axes['a'].legend(handles, labels, bbox_to_anchor=(1, .01), title='Localizations:', facecolor='none', labelcolor='white', loc='lower right', frameon=False)
        leg._legend_title_box._text.set_color('#FFFFFF')

        axes['b'].set_title(f'Time: {current_time} ms',loc='left')
        axes['b'].text(0.05, 0.9, r'd = 32 nm', horizontalalignment='left', verticalalignment='baseline', 
                transform=axes['b'].transAxes,fontsize=10, color='white')
        axes['b'].set_xlim(-dx/2,dx/2)
        axes['b'].set_ylim(-dy/2,dy/2)
        axes['b'].set_aspect('equal')
        axes['b'].set_xlabel('x (nm)')
        axes['b'].set_ylabel('y (nm)')
        axes['b'].yaxis.set_label_position("right")
        axes['b'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=False,labelleft=False,right=True,labelright=True,direction='in')
        leg = axes['b'].legend(handles, labels, bbox_to_anchor=(1, .01), title='Localizations:', facecolor='none', labelcolor='white', loc='lower right', frameon=False)
        leg._legend_title_box._text.set_color('#FFFFFF')
        
        patch1 = mpatches.Patch(color=self.s.c10, label='32 nm')
        patch2 = mpatches.Patch(color=self.s.c20, label='15 nm')
        handles = [patch1,patch2]
        labels = [patch1._label,patch2._label]

        axes['c'].set_xlim(0,2000)
        axes['c'].set_xticks([0,500,1000,1500,2000])
        axes['c'].set_ylim(0,45)
        axes['c'].set_yticks([0,15,30,45])
        axes['c'].set_xlabel('time (ms)')
        axes['c'].set_ylabel('d (nm)')
        axes['c'].legend(handles, labels, bbox_to_anchor=(1, 1), title='Expected size:', loc='upper right', frameon=True)
        axes['c'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=True,labelleft=True,right=False,labelright=False,direction='in')

        axes['d'].sharey(axes['c'])
        axes['d'].set_xlabel('Probability')
        axes['d'].set_ylabel('d (nm)')
        axes['d'].yaxis.set_label_position("right")
        axes['d'].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,left=False,labelleft=False,right=True,labelright=True,direction='in')
        axes['d'].set_xlim([0,.2])
        axes['d'].set_xticks([0,.1,.2])
        #axes['d'].legend()

        self.s.drop_axes(axes=[axes['a'],axes['b'],axes['c'],axes['d']])#ax['b'],
        self.s.hide_axes(axes=[axes['a'],axes['c']],dirs=['top','right'])
        self.s.hide_axes(axes=[axes['b']],dirs=['top','left'])
        self.s.hide_axes(axes=[axes['d']],dirs=['top'])
        pass

    def init(self, axes, dx, dy):
        self.set_axis(0, axes, [], dx, dy)
        pass

    # Define a function to update the plot for each time step
    def update(self,frame, axes, filtered_df, names, arrs, points, comPoints, dx, dy, dt, time_values):
        self.set_axis(time_values[frame],axes,[],dx,dy)
        cmap = mpl.colormaps['afmhot']
        xi, yi = np.mgrid[-dx/2:dx/2:240j, -dy/2:dy/2:240j]# Create a meshgrid for the x and y positions
        # Filter the data for the current time step
        for idx, axis, cols in zip(range(len(names)),[axes['a'],axes['b']],[[self.s.c10,self.s.c11],[self.s.c20,self.s.c21]]):
            # select data:
            zi = arrs[2*frame+idx]
            comx, comy = comPoints[2*frame+idx]
            x, y = points[2*frame+idx]
            alphas = np.linspace(0.,.8,len(x))
            colors = cmap(np.linspace(0.2,.6,len(x)))

            # draw kde/histogram
            try:
                axis.pcolormesh(xi,yi,zi,cmap='afmhot',linewidth=0,rasterized=False)
            except:
                pass
            
            # draw fading individual positions
            try:
                axis.scatter(x,y,zorder=1,c=colors,s=.2,alpha=alphas)
            except:
                pass

            # moving average of centre of mass
            try:
                axis.plot(comx,comy,zorder=10,c='w',lw=1.5,alpha=.9,label='COM')#,s=.5
            except:
                pass

            # line plot of distance estimate
            if idx==0:
                try:
                    tmp_df = filtered_df.loc[filtered_df['time']<=time_values[frame]].groupby('gt').apply(lambda x: x).reset_index(drop=True).copy()
                    tmp_df.loc[:,'std'] = tmp_df.groupby('gt')['d_norm'].rolling(window=5,min_periods=1).std().reset_index()['d_norm']#5
                    tmp_df.loc[:,'d_norm'] = tmp_df.groupby('gt')['d_norm'].rolling(window=2,min_periods=1).mean().reset_index()['d_norm']#5
                    for gt,cols in zip(tmp_df['gt'].unique(),[[self.s.c20,self.s.c21],[self.s.c10,self.s.c11]]):
                        sub_df = tmp_df.loc[(tmp_df['gt']==gt)&(tmp_df['axis']==0)]#select one axis since we only plot dnorm
                        axes['c'].fill_between(sub_df['time'].values, sub_df['d_norm'].values-sub_df['std'].values, sub_df['d_norm'].values+sub_df['std'].values, color=cols[1], alpha=0.5)
                        axes['c'].plot(sub_df['time'].values,sub_df['d_norm'].values,label=f'{gt}',c=cols[0])
                    axes['d'].hist([tmp_df.loc[tmp_df['gt']==30,'d_norm'],tmp_df.loc[tmp_df['gt']==15,'d_norm']],color=[self.s.c10,self.s.c20],alpha=1.,density=True,stacked=True,orientation='horizontal',bins=45)
                except:
                    pass

#---------------------------
# create and animate figure
#---------------------------
def make_animation(plot_style='nature',color_scheme='default',show=True, frames=10,time_bin=50):
    max_frames = frames
    dt = time_bin #time bin in ms
    dx, dy = 60,60#extent of plots in nm

    s = Style(style=plot_style,color_scheme=color_scheme)
    fig, axs = plt.subplot_mosaic(
        [
            ['a','a','b','b']
            ,['BLANK','BLANK','BLANK','BLANK']
            ,['c','c','c','d']
            ]#grid
        ,empty_sentinel="BLANK"
        ,gridspec_kw = {
            "width_ratios": [1,1,1,1]#widths
            ,"height_ratios": [1,.07,.3]#heights
            }
        ,constrained_layout=False
        ,figsize=s.get_figsize(rows=1.5,cols=2,ratio=1)#
        )

    base_dir = os.path.join(ROOT_DIR,'src/figures/DynamicDataFigure/data')
    file = ut.BaseFunc().find_files(base_dir, lambda file: ut.BaseFunc().match_pattern(file, 'dynamic-minflux-data.pkl', match='partial'),max_files=1)[0] # load pkl
    df = pd.read_pickle(file)

    # select exemplary measurements
    names = ['GQ15_MF_scans_modulated_i1_mp2_3','GQ30003_MF_scans_i1_circle_movement_27']
    filtered_df = df.loc[df['file'].apply(lambda x: np.any([name in x for name in names]))].copy()
    filtered_df['COM'] = filtered_df['x0']*LAMBDA/(4*np.pi)
    axis_means = filtered_df.groupby(['file','axis'])['COM'].transform('mean')
    filtered_df.loc[:,'COM'] = filtered_df['COM'] - axis_means
    filtered_df['pos1'] = filtered_df['COM'] - filtered_df['d']/2
    filtered_df['pos2'] = filtered_df['COM'] + filtered_df['d']/2
    filtered_df['time'] = filtered_df['time'] - filtered_df['time'].min()

    # Create a list of unique time values from your DataFrame
    unique_times = filtered_df['time'].nunique()
    time_values = filtered_df.loc[filtered_df['time']>dt,'time'].unique()
    time_values = np.sort(time_values)[::int(unique_times/max_frames)]
    unique_times = np.unique(time_values)

    arrs, points, comPoints = generate_data(filtered_df, names, dx, dy, dt, time_values)

            
    interval = 2*int(np.diff(time_values).mean()) #1000 / fps  # Convert fps to interval in milliseconds
    fps = max(1,int(1000/interval))  # Adjust as needed
    animator = AnimatedPlot(s)

    animator.set_axis(0,axs,[],dx,dy)
    animator.update(1, axs, filtered_df, names, arrs, points, comPoints, dx, dy, dt, time_values)

    update = partial(animator.update, axes=axs, filtered_df=filtered_df, names=names, arrs=arrs, points=points, comPoints=comPoints, dx=dx, dy=dy, dt=dt, time_values=time_values)

    ani = FuncAnimation(fig, update, init_func=partial(animator.init,axes=axs, dx=dx,dy=dy), frames=len(time_values), repeat=True, interval=interval)#
    if show:
        plt.show()
    plt.close()
    return ani, fps

if __name__=='__main__':
    ani, fps = make_animation(show=False)
    
    base_dir = os.path.join(ROOT_DIR,'src/figures/DynamicDataFigure')
    output_file = base_dir+'/DynamicRulerAnimationTest.gif'# Define the file name for the GIF
    ani.save(output_file, writer='pillow', fps=fps, dpi=400)# Save the animation as a GIF