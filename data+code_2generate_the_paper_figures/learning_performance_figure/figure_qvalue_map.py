import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.spatial as scispa
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.colors import Normalize


import figures_utils as fig_utl
import parameters_MF_MB as PRMS
from figures_indiv import show_trajectory, extract_data, show_transitions
params = PRMS.params


def create_voronoid(params=params):
    centre_states = np.array(params['state_coords'])
    # Add 4 distant dummy points 
    centre_states = np.append(centre_states, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0)  
    # Create Voronoi
    vor = scispa.Voronoi(centre_states)
    return vor

def fill_voronoid(ax, Q, vor, cmap=plt.cm.viridis):
    '''Plots the voronoi structure of the environment filled with the max Q-values for each state.
    :param ax: Axes to plot on, matplotlib.pyplot.figure.subplots axes.
    :return: plt.cm.ScalarMappable instance plotted on the figure
    '''
    # Data : max Q-value for each state
    Qmax = np.max(Q, axis=1) # maximum Q-value for each state, array of length nS
    normalizer = Normalize(vmin=0, vmax=1) # normalization
    # Fill each state with a color corresponding to the normalized Q-value
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=cmap(normalizer(Qmax[r])), zorder=0)
    return plt.cm.ScalarMappable(cmap=cmap, norm=normalizer)

def mask_outside_polygon(poly_verts, ax=None):
    '''Plots a mask on the specified axis ("ax", defaults to plt.gca())
    such that all areas outside of the polygon specified by "poly_verts" are masked.  
    :param poly_verts: List of tuples of the verticies in the polygon in counter-clockwise order.
    :returns: matplotlib.patches.PathPatch instance plotted on the figure.
    '''
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Verticies of the plot boundaries in clockwise order
    bound_verts = [(xlim[0], ylim[0]), (xlim[0], ylim[1]), (xlim[1], ylim[1]), (xlim[1], ylim[0]), (xlim[0], ylim[0])]
    # Series of codes (1 and 2) to specify whether to draw a line or 
    # move the "pen" (so that there's no connecting line)
    bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
    poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]
    # Plot the masking patch
    path = mpath.Path(bound_verts + poly_verts, bound_codes + poly_codes)
    patch = mpatches.PathPatch(path, facecolor='white', edgecolor='none', zorder=2)
    patch = ax.add_patch(patch)
    # Reset the plot limits to their original extents
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return patch


def create_map(ax, map_path="map1.pgm", scale=1.0, offset=np.zeros(2)):
    '''Plots the map of the environment.
    :param map_path: contour, .pgm.'''
    img = mpimg.imread(map_path)
    # Get list of coordinates for the border
    coords = []
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i,j] == 0:
                coords.append((i*scale, j*scale))
    coords = np.array(coords)
    # Center data and adjust offset
    coords[:, 0] -= np.mean(coords[:, 0])
    coords[:, 1] -= np.mean(coords[:, 1])
    coords = coords + offset
    # Compute the contour
    hull = ConvexHull(coords)
    hull_points_x = [coords[v, 0] for v in hull.vertices]
    hull_points_x.append(coords[hull.vertices[0], 0])
    hull_points_y = [coords[v, 1] for v in hull.vertices]
    hull_points_y.append(coords[hull.vertices[0], 1])
    # Draw contour and white patch outside
    hull_points = list(zip(hull_points_x, hull_points_y))
    mask_outside_polygon(hull_points, ax=ax)
    ax.plot(hull_points_x, hull_points_y, 'k-')
    ax.axis("equal")
    ax.axis("off")


def plot_Qvalues_map(Q_rpls, H_rpls=None, H_trajs=None, params=params, r_state=0, epoch='learning',
                    norm_across_rep=True, deterministic=True, fig_title='Q-value map(s)', 
                    fontsize=14, leg=True, title_ax=True,
                    show_trans=False,
                    cmap_Q=plt.cm.Greys, cmap_traj='rainbow',
                    sz=50, sz_r=100, edgc='k', edgw=1, c='white', scale=3,
                    axes=None):
    '''Plots a Q value map and a trajectoryfor the specified replays.
    :param Q_rpls: Dictionary containing the Q matrices to be plotted for each replay type.
    :params H_rpls: Dictionary containnig the exploration or replayed transitions to be plotted for each replay type, in gradient.
    :params H_traj: Dictionary containing additional trajectories to be plotted, in dotted lines.
    '''
    centre_states = np.array(params['state_coords']) # coordinates of states centres (x,y)
    vor = create_voronoid(params)
    x_states, y_states, T = extract_data(deterministic, params=params)
    i_r = params['reward_states'][r_state]
    if epoch == 'learning':
        i_s0 = params['starting_points']['learning']
    if epoch == 'generalization':
        i_s0 = params['starting_points']['generalization'][0]
    norm = 1 # no normalization by default
    if norm_across_rep: # find the maximum Q across all replay types
        norm = max([np.max(Q_rpls[rep]) for rep in params['replay_refs']])
        if norm == 0:
            norm = 1

    n_plots = len(params['replay_refs'])
    fig, axes = plt.subplots(1, n_plots, figsize=(scale*n_plots,scale))
    if deterministic==True:
        add_title = ' - Deterministic environment'
    if deterministic==False:
        add_title = ' - Stochastic environment'
    else: # do not specify
        add_title = ''
    fig.suptitle(fig_title+add_title, y=1.15, fontsize=15)

    for i, rep in enumerate(params['replay_refs']):
        if n_plots > 1:
            ax = axes[i]
        else:
            ax = axes
        Q = Q_rpls[rep]/norm
        colormap = fill_voronoid(ax, Q, vor, cmap=cmap_Q)
        create_map(ax, map_path="Figures/map1.pgm", scale=0.08, offset=np.array([-0.2, 0.2]))
        scispa.voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='k')

        if show_trans:
            show_transitions(ax, x_states, y_states, T)
        if H_rpls is not None:
            show_trajectory(ax, H_rpls[rep], x_states, y_states, cmap_name=cmap_traj)
        if H_trajs is not None:
            show_trajectory(ax, H_trajs[rep], x_states, y_states, uniform_col=True)

        ax.scatter(centre_states[i_s0, 0], centre_states[i_s0, 1], label='Initial state', color=c, edgecolor=edgc, linewidth=edgw, s=sz, zorder=100)
        ax.scatter(centre_states[i_r, 0], centre_states[i_r, 1], label='Reward state', marker="*", color=c, edgecolor=edgc, linewidth=edgw, s=sz_r, zorder=100)
        ax.set_xlim(-1.5, 1.2)
        ax.set_ylim(-1, 1)

        if leg and i == 0: # add legend for remarkable states only on the first plot
            ax.legend(bbox_to_anchor=[0, 0], loc='center', fontsize=13)
        if title_ax:
            ax.set_title(params['replay_types'][rep], fontsize=14)

    # Set a common colorbar
    cbar = fig.colorbar(colormap, ax=axes, fraction=0.01)
    cbar.set_label('Maximum Q-value\nin each state (a.u.)', fontsize=12)
    plt.show()




