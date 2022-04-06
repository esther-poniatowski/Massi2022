import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib import cm
import matplotlib.collections as mcoll
from matplotlib.collections import LineCollection as lc
from mpl_toolkits.mplot3d.art3d import Line3DCollection as lc3d
from scipy.interpolate import interp1d
from matplotlib.colors import colorConverter
import matplotlib.path as mpath
import matplotlib.colors as colors
from matplotlib.ticker import LinearLocator
from  matplotlib.legend import Legend


# -----------------------------------  
#### Colors ####
# -----------------------------------  

def discrete_colormap(n, cmap_name='rainbow', cmap=None, sample=None):
    '''Create an N-bin discrete colormap from the specified input map.'''
    if cmap is None:
        cmap = cm.get_cmap(cmap_name)
    if sample is None:
        colors = cmap(np.linspace(0, 1, n))
    else:
        colors = cmap(sample)
    return colors

def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=100):
    cmap = cm.get_cmap(cmap_name)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# -----------------------------------  
#### Plot frames ####
# -----------------------------------  

def hide_spines(ax, sides=['right', 'top']):
    for side in sides :
            ax.spines[side].set_visible(False)
            
def hide_ticks(ax, axis):
    if axis == 'x':
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
    if  axis == 'y':
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])

def hide_all(ax):
    hide_spines(ax, sides=['left', 'right', 'top', 'bottom'])
    hide_ticks(ax, 'x')
    hide_ticks(ax, 'y')
    
def center_axes(ax, axes=['x','y']):
    # Eliminate upper and right axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Make spines pass through zero of the other axis
    if 'x' in axes:
        ax.spines['bottom'].set_position('zero')
    if 'y' in axes:
        ax.spines['left'].set_position('zero')
    # Ticks protrude in both directions
    ax.xaxis.set_tick_params(direction='inout')
    ax.yaxis.set_tick_params(direction='inout')

def dashed_line(ax, axis='y', pos=0, label=''):
    if axis=='x':
        xlim = ax.get_xlim()
        ax.plot(xlim, (pos,pos), color='gray', linestyle='dashed', label=label)
    if axis=='y':
        ylim = ax.get_ylim()
        ax.plot((pos,pos), ylim, color='gray', linestyle='dashed', label=label)
        
def label_end_axis(ax, label, axis='x', fontsize=12, rot=0):
    if axis == 'x':
        label = ax.set_xlabel(label, ha='left', va = 'top', fontsize=fontsize)
        # get a tick and position label next to the last one
        ticklab = ax.xaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()
        xlim = ax.get_xlim()[1]
        ax.xaxis.set_label_coords(xlim, 0, transform=trans)
    if axis == 'y':
        label = ax.set_ylabel(label, ha='right', va = 'bottom', fontsize=fontsize, rotation=rot)
        ticklab = ax.yaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()
        ylim = ax.get_ylim()[1]
        ax.yaxis.set_label_coords(0, ylim, transform=trans)
        
# -----------------------------------   
#### Legends ####
# -----------------------------------  

def legend_out(ax, title=''):
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=title)

def collect_legend(ax, ax_leg):
    '''Collect the legends from ax and plots them on ax_leg.'''
    legends = [c for c in ax.get_children() if isinstance(c, Legend)]
    print(legends)
    ax_leg.legend(legends)
    ax_leg.axis('off') # hide the axes frame and the x/y labels
    