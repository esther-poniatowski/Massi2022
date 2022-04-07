#!/usr/bin/env python

import numpy as np
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

import figures_utils as fig_utl
import parameters_MF_MB as PRMS
from analyzes_MF_MB import test_pairwise, test_groups, split_before_after_change, identify_representative
from figures_pop import curve_shaded, create_data_groups, plot_violins_replays, compare_alpha_values_replays, compare_alpha_det_stoch_all, plot_Q_distribution
from figures_indiv import show_trajectory, extract_data
from figure_qvalue_map import create_voronoid, fill_voronoid, create_map
import scipy.spatial as scispa

from simulations_MF_MB import recover_data
params = PRMS.params


colors_replays = {0: 'royalblue',
                  1: 'orange',
                  2: 'forestgreen',
                  3: 'orchid',
                  4: 'k',
                  -1:'white'} 




# ********** LEARNING CURVES and VIOLIN PLOTS *************** #

def plot_violin_and_stats_on_ax(ax, PRF,  params=params, thres=0.05,
                                ylab='', ax_title='', fontsize=10, fontsize_title=10,
                                fontsize_leg=11, leg=False, log=True):
    Q1, Q2, Q3, data = create_data_groups(PRF, var_name='Mean', params=params)
    Hvalue, pvalue = test_groups(PRF, var_name='Mean', params=params, display=False)
    if pvalue < thres:
        stats = test_pairwise(PRF, var_name='Mean', thres=thres, params=params, display=False)
    else:
        stats = []
    plot_violins_replays(ax, data, Q1, Q2, Q3, stats=stats, params=params, 
                        ylab=ylab, ax_title=ax_title, fontsize=fontsize, fontsize_title=fontsize,
                        fontsize_leg=fontsize_leg, leg=False, log=log)

def figure_learning_curves_violin_plots(det=True, params=params, thres=0.05, log_scale=True,
                                        log=False, sharey=False,
                                        fontsize=10, fontsize_leg=10, fontsize_fig=11, scale=2):
    if det:
        env = '_D'
        suptitle = 'Deterministic environment'
    else:
        env = '_S'
        suptitle = 'Stochastic environment'
    # Recover data
    LC_pop = recover_data('LCl'+env)
    PRFl0_5f = recover_data('PRFl0_5f'+env)
    PRFl0_5l = recover_data('PRFl0_5l'+env)
    PRFl1_5f = recover_data('PRFl1_5f'+env)
    PRFl1_5l = recover_data('PRFl1_5l'+env)
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
    Axes_violin = []
    for col, (PRF, ttl) in enumerate(zip(PRFS, titles)):
        ax = fig.add_subplot(gs_bottom[0,col])
        Axes_violin.append(ax)
        if col == 0:
            ylab = 'Number of actions\n taken to get to the reward'
        else:
            ylab = ''
        plot_violin_and_stats_on_ax(ax, PRF,  params=params, thres=thres,
                                ylab=ylab, ax_title=ttl, fontsize=fontsize, fontsize_title=fontsize,
                                fontsize_leg=fontsize_leg, leg=False, log=log)
    if sharey:
        ymin = min([ax.get_ylim()[0] for ax in Axes_violin]) 
        ymax = max([ax.get_ylim()[1] for ax in Axes_violin]) 
        for ax, PRF, ttl in zip(Axes_violin, PRFS, titles):
            ax.clear()
            ax.set_ylim(ymin, ymax)
            plot_violin_and_stats_on_ax(ax, PRF,  params=params, thres=thres,
                                ylab=ylab, ax_title=ttl, fontsize=fontsize, fontsize_title=fontsize,
                                fontsize_leg=fontsize_leg, leg=False, log=log)
    plt.show()


    
# ********** SELECTION OF THE LEARNING RATE *************** #

def figure_alpha_selection(params=params,
                          fontsize=10, fontsize_leg=10, fontsize_fig=12, scale=2.5):
    # Recover data
    D_alpha_D = recover_data('D-alpha_D')
    D_alpha_S = recover_data('D-alpha_S')
    D_alpha_all = recover_data('D-alpha_summary')
    # Set the figure structure
    fig = plt.figure(figsize=(5*scale,2*scale), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1,1])
    fig.suptitle(r'Selection of the learning rate $\alpha$', fontsize=fontsize_fig)
    # For different replays
    ax0 = fig.add_subplot(gs[0,0]) 
    ax0.set_title('Performance in the deterministic environment\nEach replay method', fontsize=fontsize)
    compare_alpha_values_replays(D_alpha_D, ax=ax0, fontsize=fontsize, params=params)
    # For all replays
    ax1 = fig.add_subplot(gs[1,0]) 
    ax1.set_title('Performance in both environments\nAll replay methods', fontsize=fontsize)
    compare_alpha_det_stoch_all(D_alpha_all, ax=ax1, fontsize=fontsize, params=params)
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
    ax.scatter(centre_states[s0, 0], centre_states[s0, 1], label='Initial state', color=c, edgecolor=edgc, linewidth=edgw, s=sz, zorder=1000)
    ax.scatter(centre_states[s_rw, 0], centre_states[s_rw, 1], label='Reward state', marker="*", color=c, edgecolor=edgc, linewidth=edgw, s=sz_r, zorder=1000)
    ax.set_xlim(-1.5, 1.2)
    ax.set_ylim(-1, 1)
    ax.tick_params(labelcolor=(1.0,1.0,1.0, 0.0), top='off', bottom='off', left='off', right='off')
    return colormap 

def figure_Qvalues(det=True, trials=[0,1,24,25], params=params, 
                    figscale=2, fontsize=10, fontsize_fig=12):
    # Recover and compute data
    if det:
        env = '_D'
        fig_title = 'Deterministic environment'
    else:
        env = '_S'
        fig_title = 'Stochastic environment'
    Dl_indiv = recover_data('Dl_indiv'+env, df=False)
    x_states, y_states, T = extract_data(deterministic=det, params=params)

    # Set the figure structure
    n_rows = len(trials)
    n_cols = len(params['replay_refs'])
    fig, axes_trials = plt.subplots(figsize=(n_cols*1.1*figscale, 1.3*n_rows*figscale), nrows=n_rows, ncols=1, sharey=True) 
    fig.suptitle('Q-values propagation (maximum Q-values)\n'+fig_title, fontsize=fontsize_fig, y=1)
    for row, ax_trial in enumerate(axes_trials, start=1):
        ax_trial.set_title('Trial {}'.format(trials[row-1]), fontsize=fontsize_fig)
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
            s_rw = params['reward_states'][0]
        else:
            s_rw = params['reward_states'][1]

        for col in range(1, n_cols+1):
            i_ax = (row-1)*n_cols + col # index of the plot
            rep = params['replay_refs'][col-1] # corresponding replay type
            ax = fig.add_subplot(n_rows,n_cols,i_ax)
            if row == 1:
                ax.set_title(params['replay_types'][rep], y=1.15)
            # normalize Q matrix
            Q = Q_rpls[rep]
            norm = np.max(Q)
            if norm == 0 :
                norm = 1
            Q /= norm
            colormap = plot_Q_map(ax, Q, params=params, s_rw=s_rw)
            show_trajectory(ax, H_rpls[rep], x_states, y_states)
            show_trajectory(ax, H_trajs[rep], x_states, y_states, uniform_col=True)
            if i_ax == 1: # add legend for remarkable states only on the first plot
                ax.legend(bbox_to_anchor=(0,0), loc='center', fontsize=fontsize)
            if i_ax == 2:
                custom_lines = [Line2D([0], [0], color='gray', linestyle='dashed', lw=1.5),
                Line2D([0], [0], color='blue', lw=2.5)]
                labels = ['Explorated trajectory', 'Replayed transitions']
                ax.legend(custom_lines, labels, fontsize=fontsize, bbox_to_anchor=(0.2,0), loc='center')
    
    # Set a common colorbar
    cbar = fig.colorbar(colormap, ax=axes_trials[0], fraction=0.01)
    cbar.set_label('Maximum Q-value\nin each state (normalized)', fontsize=fontsize)
    plt.show()


# ********** HISTOGRAMS OF Q-VALUES *************** #

def figure_histograms(trial=24, det=True, params=params, log=True,
                    figscale=2, fontsize=10, fontsize_fig=12):
    # Recover and compute data
    if det:
        env = '_D'
        fig_title = 'Deterministic environment'
    else:
        env = '_S'
        fig_title = 'Stochastic environment'
    Qpop = recover_data('Qpop{}'.format(trial)+env, df=False)
    Dl0 = recover_data('Dl0'+env)
    LCl = recover_data('LCl'+env)
    LCl0, LCl1, _, _ = split_before_after_change(LCl, params=params)
    i_repr = identify_representative(Dl0, LCl0, params=params, show=False)
    # print(i_repr)
    Q_dict = dict(zip(params['replay_refs'], [None for rep in params['replay_refs']]))
    for rep in params['replay_refs']:
        Q = Qpop[rep][i_repr,:,:]
        Q_dict[rep] = Q.copy()
    if trial < 25:
        Qopt = recover_data('Qoptl0'+env, df=False)
        s_rw = params['reward_states'][0]
    else:
        Qopt = recover_data('Qoptl1'+env, df=False)
        s_rw = params['reward_states'][1]
    params['replay_refs'].append(-1)
    Q_dict[-1] = Qopt
    H = recover_data('Hpop'+str(trial)+env, df=False)
    # print(H['bins'])

    # Set the structure of the figure
    fig = plt.figure(figsize=(5*figscale, 2.6*figscale), constrained_layout=True)
    fig.suptitle('Q-value propagation on trial 24\n'+fig_title, y=1.15, fontsize=fontsize_fig)
    gs = fig.add_gridspec(2, 5, height_ratios=[1,1.3]) 
    
    # Plot Q maps
    for col in range(5):
        rep = params['replay_refs'][col] # corresponding replay type
        ax = fig.add_subplot(gs[0,col])
        ax.set_title(params['replay_types'][rep], y=1)
        Q = Q_dict[rep]
        norm = np.max(Q)
        if norm == 0 :
            norm = 1
        Q /= norm
        colormap = plot_Q_map(ax, Q, params=params, s_rw=s_rw)
        
    # Plot histograms
    Axes_hist = []
    for col in range(5):
        rep = params['replay_refs'][col] # corresponding replay type
        ax = fig.add_subplot(gs[1,col])
        Axes_hist.append(ax)
        plot_Q_distribution(H[rep], ax, color=colors_replays[rep], params=params, log=True)
    Axes_hist[0].set_ylabel('Distribution (log)', fontsize=fontsize)
    fig_utl.hide_spines(Axes_hist[0])
    ymax = max([max(H[rep]['Q3']) for rep in params['replay_refs'] if rep != -1])
    ymin = Axes_hist[0].get_ylim()[0]
    for ax in Axes_hist[1:]:
        fig_utl.hide_spines(ax, sides=['left','right','top'])
        fig_utl.hide_ticks(ax, 'y')
        ax.set_ylim(ymin,ymax)

    ax_up = fig.add_subplot(2,1,1)
    ax_up.set_title('Q-value maps of the most representative individual', y=1.6, fontsize=fontsize_fig)
    ax_down = fig.add_subplot(2,1,2)
    ax_down.set_title('Distributions of Q-values (median in the population)', y=1.85, fontsize=fontsize_fig)
    for ax in [ax_up, ax_down]:
        fig_utl.hide_spines(ax, ['top', 'bottom', 'left', 'right'])
        fig_utl.hide_ticks(ax, 'x')
        fig_utl.hide_ticks(ax, 'y')
        ax.patch.set_alpha(0)