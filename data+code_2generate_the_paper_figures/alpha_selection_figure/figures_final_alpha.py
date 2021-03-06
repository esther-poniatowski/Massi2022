#!/usr/bin/env python

import numpy as np
import random
from collections import defaultdict

import matplotlib_latex_bridge as mlb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

import figures_utils as fig_utl
import parameters_MF_MB as PRMS
from analyzes_MF_MB import test_pairwise, test_groups
from figures_pop import curve_shaded, create_data_groups, compare_alpha_values_replays, compare_alpha_det_stoch_all,\
    plot_violins_replays
from figures_indiv import show_trajectory, extract_data
from figure_qvalue_map import create_voronoid, fill_voronoid, create_map
import scipy.spatial as scispa

from simulations_MF_MB import recover_data
params = PRMS.params


colors_replays = {0: 'royalblue',
                  1: 'orange',
                  2: 'forestgreen',
                  3: 'orchid',
                  4: 'k'} 




# ********** LEARNING CURVES and VIOLIN PLOTS *************** #

def figure_learning_curves_violin_plots(det=True, params=params, thres=0.05, log_scale=True,
                                        fontsize=10, fontsize_leg=10, fontsize_fig=11, scale=2, folder='Data/'):
    if det:
        env = '_D'
        suptitle = 'Deterministic environment'
    else:
        env = '_S'
        suptitle = 'Stochastic environment'
    # Recover data
    LC_pop = recover_data('LCl'+env, folder=folder)
    PRFl0_5f = recover_data('PRFl0_5f'+env, folder=folder)
    PRFl0_5l = recover_data('PRFl0_5l'+env, folder=folder)
    PRFl1_5f = recover_data('PRFl1_5f'+env, folder=folder)
    PRFl1_5l = recover_data('PRFl1_5l'+env, folder=folder)
    PRFS = [PRFl0_5f,PRFl0_5l,PRFl1_5f,PRFl1_5l]
    titles = ['Trials 1-5',
             'Trials 20-25',
             'Trials 26-30',
             'Trials 45-50']
    # Set the figure structure
    fig = plt.figure(figsize=(5*scale,4*scale), constrained_layout=True)
    gs_outer = fig.add_gridspec(3, 1, height_ratios=[1,0.1,1]) # learning curves above, violin plots below, one intermediary line for title
    gs_bottom = gs_outer[2].subgridspec(1, 4) # 4 boxes for 4 trial epochs on the second line
    fig.suptitle(suptitle, fontsize=fontsize_fig)
    # Learning curves
    ax_LC = fig.add_subplot(gs_outer[0,0]) # learning curves
    for rep in params['replay_refs'] :
        ylow = LC_pop['Q1'].loc[(LC_pop['Replay type']==rep)].to_numpy()
        y = LC_pop['Q2'].loc[(LC_pop['Replay type']==rep)].to_numpy()
        yup = LC_pop['Q3'].loc[(LC_pop['Replay type']==rep)].to_numpy()
        t = np.arange(1, len(y)+1)
        label = params['replay_types'][rep]
        color = colors_replays[rep]
        curve_shaded(ax_LC, t, y, ylow, yup, label=label, color=color)
    ax_LC.grid()
    ymin, ymax = ax_LC.get_ylim()
    for (tmin, tmax) in [(1, 5), (20, 25), (25, 30), (45, 50)]:
        ax_LC.vlines(tmin, ymin, ymax, linestyle='dashed', color='gray')
        ax_LC.vlines(tmax, ymin, ymax, linestyle='dashed', color='gray')
        ax_LC.axvspan(tmin, tmax, alpha=0.15, color='gray')
        ax_LC.text(tmin+(tmax-tmin)/2, ymax*1.1, '{}-{}'.format(tmin, tmax), ha='center')
    if log_scale:
        ax_LC.set_yscale('log', base=2)
    fig_utl.hide_spines(ax_LC)
    ax_LC.set_xlabel('Trials', fontsize=fontsize)
    ax_LC.set_ylabel('Number of actions\nto get to the reward', fontsize=fontsize)
    ax_LC.set_title('Learning curves', fontsize=fontsize)
    legend = ax_LC.legend(title='Replay types', fontsize=fontsize_leg)
    legend.get_title().set_fontsize(fontsize_leg)
    # Intermediate legend
    ax = fig.add_subplot(gs_outer[1,0]) 
    # ax.plot([0,0], [0,1]) # define range
    xmin, xmax = ax.get_xlim()
    ax.text((xmax-xmin)/2, 0,'Performance', ha='center')
    fig_utl.hide_ticks(ax, 'x')
    fig_utl.hide_ticks(ax, 'y')
    fig_utl.hide_spines(ax, ['top', 'bottom', 'left', 'right'])
    # Violin plots
    for col, (PRF, ttl) in enumerate(zip(PRFS, titles)):
        ax = fig.add_subplot(gs_bottom[0,col])
        Q1, Q2, Q3, data = create_data_groups(PRF, var_name='Mean', params=params)
        Hvalue, pvalue = test_groups(PRF, var_name='all', params=params, display=True)
        if pvalue < thres:
            stats = test_pairwise(PRF, var_name='all', thres=thres, params=params, display=False, test='conover')
        else:
            stats = []
        if col == 0:
            ylab = 'Number of actions\ntaken to reach reward'
        else:
            ylab = ''
        plot_violins_replays(ax, data, Q1, Q2, Q3, stats=stats, params=params, 
                            ylab=ylab, ax_title=ttl, fontsize=fontsize, fontsize_title=fontsize,
                            fontsize_leg=fontsize_leg, leg=False)
        print('here')

    plt.show()


    
# ********** SELECTION OF THE LEARNING RATE *************** #

def figure_alpha_selection(params=params,
                          fontsize=10, fontsize_leg=10, fontsize_fig=12, scale=2.5):
    # Recover data
    D_alpha_D = recover_data('D-alpha_D')
    D_alpha_S = recover_data('D-alpha_S')
    D_alpha_all = recover_data('D-alpha_summary')
    # Set the figure structure
    # fig = plt.figure(figsize=(5*scale,2*scale), constrained_layout=True)
    fig = mlb.figure_textwidth(widthp=1, ratio=1.5)
    gs = fig.add_gridspec(2, 1, height_ratios=[1,1])
    # fig.suptitle(r'Selection of the learning rate $\alpha$', fontsize=fontsize_fig)
    # For different replays
    ax0 = fig.add_subplot(gs[0,0]) 
    # ax0.set_title('Performance in the deterministic environment\nEach replay method', fontsize=fontsize)
    compare_alpha_values_replays(D_alpha_D, ax=ax0, params=params)
    # For all replays
    ax1 = fig.add_subplot(gs[1,0]) 
    # ax1.set_title('Performance in both environments\nAll replay methods', fontsize=fontsize)
    compare_alpha_det_stoch_all(D_alpha_all, ax=ax1, params=params)
    fig.savefig("alpha_selection.pdf")

    plt.show()



# ********** Q-VALUE PROPAGATION *************** #

def plot_Q_map(ax, Q, params=params,
               s0=params['starting_points']['learning'], s_rw=params['reward_states'][0],
               cmap_Q=plt.cm.Greys, sz=50, sz_r=100, edgc='k', edgw=1, c='white', scale=0.08):
    centre_states = np.array(params['state_coords'])
    vor = create_voronoid(params)
    colormap = fill_voronoid(ax, Q, vor, cmap=cmap_Q)
    create_map(ax, map_path="Figures/map1.pgm", scale=scale, offset=np.array([-0.2, 0.2]))
    scispa.voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='k')
    ax.scatter(centre_states[s0, 0], centre_states[s0, 1], label='Initial state', color=c, edgecolor=edgc,
               linewidth=edgw, s=sz, zorder=1000)
    ax.scatter(centre_states[s_rw, 0], centre_states[s_rw, 1], label='Reward state', marker="*", color=c,
               edgecolor=edgc, linewidth=edgw, s=sz_r, zorder=1000)
    ax.set_xlim(-1.5, 1.2)
    ax.set_ylim(-1, 1)
    ax.tick_params(labelcolor=(1.0, 1.0, 1.0, 0.0), top='off', bottom='off', left='off', right='off')
    return colormap


def figure_Qvalues_new(det=True, trials=[0, 1, 24, 25], params=params):

    # Recover and compute data
    if det:
        env = '_D'
        fig_title = 'Deterministic environment'
    else:
        env = '_S'
        fig_title = 'Stochastic environment'
    Dl_indiv = recover_data('Dl_indiv' + env, df=False)
    x_states, y_states, T = extract_data(deterministic=det, params=params)

    # setup grid
    num_trials = len(trials)
    fig = mlb.figure_textwidth(widthp=0.5, ratio=1/1.5)
    # fig = plt.figure(constrained_layout=True)
    # fig = plt.figure()
    # gs = fig.add
    gs = fig.add_gridspec(1 + 2 * num_trials, 1, figure=fig, height_ratios=[1] + [1, 1] * num_trials, wspace=0)
    print(gs[0])

    # top titles
    gs_titles = gs[0].subgridspec(1, 4)
    for i, rep in enumerate([0, 1, 2, 4]):
        ax = fig.add_subplot(gs_titles[0, i])
        ax.text(0.5, 0.5, params['replay_types'][rep].replace(" ", "\n"), ha='center', va='center')
        plt.plot([4, 3, 2, 1])
        # ax.set_axis_off()
        # fig_utl.hide_spines(ax, ['top', 'bottom', 'left', 'right'])
        # fig_utl.hide_ticks(ax, 'x')
        # fig_utl.hide_ticks(ax, 'y')

    for i, trial in enumerate(trials):

        # suptitle
        ax = fig.add_subplot(gs[2*i+1])
        ax.text(x=0.5, y=0.5, s=f"Trial {trial}", ha='center', va='center')
        # ax.set_axis_off()
        # fig_utl.hide_spines(ax, ['top', 'bottom', 'left', 'right'])
        # fig_utl.hide_ticks(ax, 'x')
        # fig_utl.hide_ticks(ax, 'y')

        # data for this trial
        Q_rpls = Dl_indiv[trial]['Q_repl']
        H_rpls = Dl_indiv[trial]['h_repl']
        H_trajs = Dl_indiv[trial]['h_explo']
        if trial < 25:
            s_rw = params['reward_states'][0]
        else:
            s_rw = params['reward_states'][1]

        # trajectories
        gs_traj = gs[2*i+2].subgridspec(1, 4)
        print(gs[2*i+2])
        print(gs_traj.get_subplot_params())
        print(gs_traj[0, 0])
        for k, rep in enumerate([0, 1, 2, 4]):

            Q = Q_rpls[rep]
            norm = np.max(Q)
            if norm == 0:
                norm = 1
            Q /= norm
            ax = fig.add_subplot(gs_traj[0, k])
            # ax.plot([1, 2, 3, 4])
            # fig_utl.hide_ticks(ax, 'x')
            # fig_utl.hide_ticks(ax, 'y')
            # colormap = plot_Q_map(ax, Q, params=params, s_rw=s_rw)
            # show_trajectory(ax, H_rpls[rep], x_states, y_states)
            # show_trajectory(ax, H_trajs[rep], x_states, y_states, uniform_col=True)

    plt.show()


def figure_Qvalues(det=True, trials=[0, 1, 24, 25], params=params):
    # Recover and compute data
    if det:
        env = '_D'
        fig_title = 'Deterministic environment'
    else:
        env = '_S'
        fig_title = 'Stochastic environment'
    Dl_indiv = recover_data('Dl_indiv' + env, df=False)
    x_states, y_states, T = extract_data(deterministic=det, params=params)

    # Set the figure structure
    n_rows = len(trials)
    n_cols = len(params['replay_refs'])
    fig = mlb.figure_textwidth(widthp=0.5, ratio=1/1.5)
    axes_trials = fig.subplots(n_rows, 1, sharey=True)
    # fig, axes_trials = plt.subplots(figsize=(n_cols * 1.1 * figscale, 1.3 * n_rows * figscale), nrows=n_rows, ncols=1,
    #                                 sharey=True)
    # fig.suptitle('Q-values propagation (maximum Q-values)\n' + fig_title, fontsize=fontsize_fig, y=1)
    for row, ax_trial in enumerate(axes_trials, start=1):
        ax_trial.set_title('Trial {}'.format(trials[row - 1]))
        fig_utl.hide_spines(ax_trial, ['top', 'bottom', 'left', 'right'])
        fig_utl.hide_ticks(ax_trial, 'x')
        fig_utl.hide_ticks(ax_trial, 'y')

    # Plot the Q value maps
    for row in range(1, n_rows + 1):
        t = trials[row - 1]  # corresponding trial
        Q_rpls = Dl_indiv[t]['Q_repl']
        H_rpls = Dl_indiv[t]['h_repl']
        H_trajs = Dl_indiv[t]['h_explo']
        if t < 25:
            s_rw = params['reward_states'][0]
        else:
            s_rw = params['reward_states'][1]

        for col in range(1, n_cols + 1):
            i_ax = (row - 1) * n_cols + col  # index of the plot
            rep = params['replay_refs'][col - 1]  # corresponding replay type
            ax = fig.add_subplot(n_rows, n_cols, i_ax)
            if row == 1:
                ax.set_title(params['replay_types'][rep].replace(" ", "\n"), y=1.15)
            # normalize Q matrix
            Q = Q_rpls[rep]
            norm = np.max(Q)
            if norm == 0:
                norm = 1
            Q /= norm
            colormap = plot_Q_map(ax, Q, params=params, s_rw=s_rw)
            # show_trajectory(ax, H_rpls[rep], x_states, y_states)
            # show_trajectory(ax, H_trajs[rep], x_states, y_states, uniform_col=True)
            # if i_ax == 1:  # add legend for remarkable states only on the first plot
            #     ax.legend(bbox_to_anchor=(0, 0), loc='center')
            # if i_ax == 2:
            #     custom_lines = [Line2D([0], [0], color='gray', linestyle='dashed', lw=1.5),
            #                     Line2D([0], [0], color='blue', lw=2.5)]
            #     labels = ['Explorated trajectory', 'Replayed transitions']
            #     ax.legend(custom_lines, labels, bbox_to_anchor=(0.2, 0), loc='center')

    # Set a common colorbar
    # cbar = fig.colorbar(colormap, ax=axes_trials[0], fraction=0.01)
    # cbar.set_label('Maximum Q-value\nin each state (normalized)')
    plt.show()


def figure_Qvalues_old(det=True, trials=[0,1,24,25], params=params,
                    figscale=2, fontsize=10, fontsize_fig=12,
                    norm_across_rep=False, norm_single_replay=True,
                    cmap_Q=plt.cm.Greys, cmap_traj='rainbow',
                    sz=50, sz_r=100, edgc='k', edgw=1, c='white', scale=3):
    # Recover and compute data
    if det:
        Dl_indiv = recover_data('Dl_indiv_D', df=False)
        fig_title = 'Deterministic environment'
    else:
        Dl_indiv = recover_data('Dl_indiv_S', df=False)
        fig_title = 'Stochastic environment'
    centre_states = np.array(params['state_coords'])
    vor = create_voronoid(params)
    x_states, y_states, T = extract_data(deterministic=det, params=params)
    i_s0 = params['starting_points']['learning']

    # Set the figure structure
    n_rows = len(trials)
    n_cols = len(params['replay_refs'])
    fig, big_axes = plt.subplots(figsize=(n_cols*1.1*figscale, 1.3*n_rows*figscale), nrows=n_rows, ncols=1, sharey=True) 
    fig.suptitle('Q-values propagation (maximum Q-values)\n'+fig_title,  y=1) # fontsize=fontsize_fig, y=1)
    for row, ax_trial in enumerate(big_axes, start=1):
        ax_trial.set_title('Trial {}'.format(trials[row-1]))#, fontsize=fontsize_fig)
        fig_utl.hide_spines(ax_trial, ['top', 'bottom', 'left', 'right'])
        fig_utl.hide_ticks(ax_trial, 'x')
        fig_utl.hide_ticks(ax_trial, 'y')
    
    # Plot the Q value maps
    for row in range(1,n_rows+1):
        t = trials[row-1] # corresponding trial
        Q_rpls = Dl_indiv[t]['Q_repl']
        H_rpls = Dl_indiv[t]['h_repl']
        H_trajs = Dl_indiv[t]['h_explo']
        if t < 25 :
            i_r = params['reward_states'][0]
        else:
            i_r = params['reward_states'][1]
        norm = 1 # no normalization by default
        if norm_across_rep: # maximum Q across all replay types
            norm = max([np.max(Q_rpls[rep]) for rep in params['replay_refs']])
            if norm == 0:
                norm = 1

        for col in range(1, n_cols+1):
            i_ax = (row-1)*n_cols + col # index of the plot
            rep = params['replay_refs'][col-1] # corresponding replay type
            ax = fig.add_subplot(n_rows,n_cols,i_ax)
            if row == 1:
                ax.set_title(params['replay_types'][rep], y=1.15)

            Q = Q_rpls[rep]
            if norm_single_replay:
                norm = np.max(Q)
                if norm == 0 :
                    norm = 1
            Q /= norm
            colormap = fill_voronoid(ax, Q, vor, cmap=cmap_Q)
            create_map(ax, map_path="Figures/map1.pgm", scale=0.08, offset=np.array([-0.2, 0.2]))
            scispa.voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='k')
            show_trajectory(ax, H_rpls[rep], x_states, y_states, cmap_name=cmap_traj)
            show_trajectory(ax, H_trajs[rep], x_states, y_states, uniform_col=True)
            ax.scatter(centre_states[i_s0, 0], centre_states[i_s0, 1], label='Initial state', color=c, edgecolor=edgc,
                       linewidth=edgw, s=sz, zorder=1000)
            ax.scatter(centre_states[i_r, 0], centre_states[i_r, 1], label='Reward state', marker="*", color=c,
                       edgecolor=edgc, linewidth=edgw, s=sz_r, zorder=1000)
            ax.set_xlim(-1.5, 1.2)
            ax.set_ylim(-1, 1)
            ax.tick_params(labelcolor=(1.0,1.0,1.0, 0.0), top='off', bottom='off', left='off', right='off')

            if i_ax == 0: # add legend for remarkable states only on the first plot
                ax.legend(bbox_to_anchor=[0, 0], loc='center')#, fontsize=13)
    
    # Set a common colorbar
    # for row, ax_trial in enumerate(big_axes, start=1):
    #     cbar = fig.colorbar(colormap, ax=big_axes, fraction=0.01)
    #     cbar.set_label('Maximum Q-value\nin each state (a.u.)', fontsize=12)
    plt.show()

