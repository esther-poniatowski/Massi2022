#!/usr/bin/env python

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
import seaborn as sns

import figures_utils as fig_utl
import parameters_MF_MB as PRMS
from functions_MF_MB import V_from_Q
params = PRMS.params
from figures_pop import curve_shaded
from analyzes_MF_MB import split_before_after_change

check_reload = False # signal successful importation at the end of the file


colors_replays = {0: 'royalblue',
                  1: 'orange',
                  2: 'forestgreen',
                  3: 'orchid',
                  4: 'k'} 


# ********** AUXLIARY FUNCTIONS *************** #

def extract_data(deterministic, params=params):
    if deterministic:
        T = params['T_det']
    else:
        T = params['T_stoch']
    x_states = [params['state_coords'][s][0] for s in range(params['nS'])] # list of x coordinates of states
    y_states = [params['state_coords'][s][1] for s in range(params['nS'])] # list of y coordinates of states
    return x_states, y_states, T

def list_duplicates(seq):
    d = defaultdict(list)
    for i,item in enumerate(seq):
        d[item].append(i)
    duplicates = [(key,locs) for key,locs in d.items() if len(locs)>1]
    return duplicates

def discrete_colormap(cmap_name, n):
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, n))
    return colors


# ********** ENVIRONMENT *************** #

def show_transitions(ax, x_states, y_states, T, linestyle='solid', linewidth=2.5, params=params):
    for s1 in range(params['nS']): 
            for a in range(params['nA']):
                for s2 in range(params['nS']): 
                        if T[s1,a,s2] !=0 :
                            proba = T[s1,a,s2]
                            ax.plot([x_states[s1], x_states[s2]], [y_states[s1], y_states[s2]],
                                    linewidth=linewidth*proba, linestyle=linestyle, color='gray', zorder=0)

def show_states(ax, x_states, y_states, values=False, Q=None, maxV=None, label_states=False, params=params,
                sz = 400, sz_r = 600, edgc = 'k', edgw = 3, c = 'white'):
    s0 = params['s_start']
    sr = params['s_rw']
    X = [x_states[s] for s in range(params['nS']) if s!=s0 and s!=sr]
    Y = [y_states[s] for s in range(params['nS']) if s!=s0 and s!=sr]
    if not values:
        ax.scatter(X, Y, color=c, edgecolor=edgc, s=sz, zorder=1)
        ax.scatter(x_states[s0], y_states[s0], color=c, edgecolor=edgc, linewidth=edgw, s=sz, zorder=1)
        ax.scatter(x_states[sr], y_states[sr], marker="*", color=c, edgecolor=edgc, linewidth=edgw, s=sz_r, zorder=1)
    else:
        V = V_from_Q(Q, params) # value of each state
        if maxV is None:
            maxV = np.max(V)
        cmap = cm.get_cmap('binary')
        V_red = [V[s] for s in range(params['nS']) if s!=s0 and s!=sr]
        ax.scatter(X, Y, 
                    c=V_red, cmap=cmap, vmin=0, vmax=maxV,
                    edgecolor=edgc, s=sz, zorder=1)
        ax.scatter(x_states[s0], y_states[s0], 
                    c=V[s0], cmap=cmap, vmin=0, vmax=maxV,
                    edgecolor=edgc, linewidth=edgw, s=sz, zorder=1)
        ax.scatter(x_states[sr], y_states[sr], marker="*", 
                    c=V[sr], cmap=cmap, vmin=0, vmax=maxV,
                    edgecolor=edgc, linewidth=edgw, s=sz_r, zorder=1)
    if label_states:
        for s in range(params['nS']): 
            ax.text(x_states[s], y_states[s], s)

def show_environment(deterministic, title=None, values=False, Q=None, maxV=None, label_states=False, hatP=None, params=params):
    x_states, y_states, T = extract_data(deterministic, params)
    if title is None:
        if deterministic:
            title = 'Deterministic world'
        else:
            title = 'Stochastic world'
    if hatP is not None:
        T = hatP
        deterministic = False
        title = 'Estimated transition probabilities (MB agent)'
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_aspect('equal')
    show_transitions(ax, x_states, y_states, T, params=params)
    show_states(ax, x_states, y_states, values=values, Q=Q, maxV=maxV, label_states=label_states, params=params)
    ax.set_title(title, fontsize=15)
    plt.show()



# ********** TRAJECTORIES *************** #

def add_jitter(X, Y, jit = 0.05):
    np.random.seed(1)
    for t in range(len(X)-1):
            dx, dy = jit*np.random.random(2)
            X[t][1] += dx # x_states[s1] for transition t (arrival state)
            X[t+1][0] += dx # x_states[s0] for transition t+1 (departure state = previous arrival state)
            Y[t][1] += dy
            Y[t+1][0] += dy
    return X, Y

def show_trajectory(ax, h_trial, x_states, y_states, cmap_name='rainbow', jitter=False, label_states=False, params=params):
    if jitter:
        jit = 0.05
    else:
        jit = 0
    if len(h_trial) > 0: # h_trial != [] (no replay)
        if isinstance(h_trial[0],tuple): # already trajectory (replay types 0,1,2,4) 
            colors = discrete_colormap(cmap_name, len(h_trial))       
            X = [[x_states[s0], x_states[s1]] for s0, a, s1, r in h_trial]
            Y = [[y_states[s0], y_states[s1]] for s0, a, s1, r in h_trial]
            X, Y = add_jitter(X, Y, jit)
            for t in range(len(X)):
                ax.plot(X[t], Y[t], linewidth=2.5, color=colors[t], zorder=2)
        elif isinstance(h_trial[0],list): # list of sequences (replay type 3)
            colors = discrete_colormap(cmap_name, params['n_seq_d'])
            for i, seq in enumerate(h_trial):
                col = colors[i]
                X = [[x_states[s0], x_states[s1]] for s0, a, s1, r in seq]
                Y = [[y_states[s0], y_states[s1]] for s0, a, s1, r in seq]
                X, Y = add_jitter(X, Y, jit=0.1)
                for t in range(len(X)):
                    ax.plot(X[t], Y[t], linewidth=2.5, color=col, zorder=2)


# ************* GLOBAL FIGURES *************** #

def plot_trial_V(ax, h_trial, Q, deterministic, maxV=None, title=None, cmap_name='rainbow', params=params):
    x_states, y_states, T = extract_data(deterministic, params=params)
    ax.set_aspect('equal')
    fig_utl.hide_ticks(ax, 'x')
    fig_utl.hide_ticks(ax, 'y')
    show_states(ax, x_states, y_states, values=True, Q=Q, maxV=maxV, params=params)
    show_transitions(ax, x_states, y_states, T, linestyle='dashed', linewidth=1, params=params)
    show_trajectory(ax, h_trial, x_states, y_states, cmap_name=cmap_name, params=params)
    ax.set_title(title, fontsize=15)


def plot_evolution_V(H, H_Q, id_trials, deterministic, fig_title='', replay=False, fontsize=16, params=params):
    n_trials = len(id_trials)
    m = max([np.max(Q) for Q in H_Q])

    fig, axes = plt.subplots(nrows=1, ncols=n_trials, figsize=(5*n_trials,5))
    if not replay:
        fig.suptitle('Q-learning and Trajectories\n'+fig_title, fontsize=fontsize)
        cmap_name = 'rainbow'
    else:
        fig.suptitle('Replays\n'+fig_title, fontsize=fontsize)
        cmap_name = 'viridis'
    for i, trial in enumerate(id_trials):
        h = H[trial]
        Q = H_Q[trial]
        title = 'Trial ' + str(trial)
        if n_trials == 1:
            ax = axes
        else:
            ax = axes[i]
        plot_trial_V(ax, h, Q, deterministic=deterministic, title=title, maxV=m, cmap_name=cmap_name)
    # ax.colorbar()
    # fig_utl.legend_out(ax, title='Transition step')
    plt.show()


def plot_V_comparison_replays(Data, h_explo, Q_explo, deterministic=True, params=params, fontsize=16):
    n_plots = len(Data)
    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(5*n_plots,5))
   
    ax = axes[0]
    title = 'Q-learning and Trajectory'
    cmap_name = 'rainbow'
    h = h_explo
    Q = Q_explo
    plot_trial_V(ax, h, Q, deterministic=deterministic, title=title, cmap_name=cmap_name)

    m = max([np.max(Data[rep]['Q_upd']) for rep in range(1, n_plots)])
    for rep in range(1, n_plots):
        ax = axes[rep]
        cmap_name = 'viridis'
        title = params['replay_types'][rep]
        h = Data[rep]['h_repl']
        Q = Data[rep]['Q_upd']
        plot_trial_V(ax, h, Q, deterministic=deterministic, title=title, maxV=m, cmap_name=cmap_name)
    plt.show()



# ************* SANITY CHECKS *************** #

def plot_performance(Performance, title=''):
    fig, ax = plt.subplots()
    ax.plot(Performance, linestyle='dashed', marker='o', color='k')
    ax.set_xlabel('Trials')
    ax.set_ylabel('Number of actions taken to reach reward')
    fig_utl.hide_spines(ax)
    ax.set_title('Performance\n'+title)
    plt.show()

def plot_performance_replays(Perf0=None, Perf1=None, Perfg=None, 
                            Perf_pop0=None, Perf_pop1=None, Perf_popg=None,
                            params=params, scale=4, log_scale=True):
    ncols = len(params['replay_refs'])
    epochs = ['l0','l1','g']
    Perfs = dict(zip(epochs, [Perf0, Perf1, Perfg]))
    Perfs_pop = dict(zip(epochs, [Perf_pop0, Perf_pop1, Perf_popg]))
    titles = dict(zip(epochs, ['First learning phase', 'Learning with new reward location', 'Generalization with new starting point']))
    keys = [e for e, perf in zip(epochs,[Perf0, Perf1, Perfg]) if perf is not None]    

    for key in keys:
        title = titles[key]
        perf_pop = Perfs_pop[key]
        max_perfs_indiv = max([max(Perfs[key][rep]) for rep in params['replay_refs']])
        min_perfs_indiv = min([min(Perfs[key][rep]) for rep in params['replay_refs']])
        max_perf = max(np.max(perf_pop['Q3']), max_perfs_indiv)
        min_perf = min(np.min(perf_pop['Q1']), min_perfs_indiv)
        fig, axes = plt.subplots(1, ncols, figsize=(scale*ncols,scale))
        for col, rep in enumerate(params['replay_refs']):
            ax = axes[col]
            perf = Perfs[key][rep]
            ax.plot(perf, marker='o', color=colors_replays[rep], label='Individual')
            ylow = perf_pop['Q1'].loc[(perf_pop['Replay type']==rep)].to_numpy()
            y = perf_pop['Q2'].loc[(perf_pop['Replay type']==rep)].to_numpy()
            yup = perf_pop['Q3'].loc[(perf_pop['Replay type']==rep)].to_numpy()
            t = np.arange(0, len(y))
            curve_shaded(ax, t, y, ylow, yup, color=colors_replays[rep], linestyle='dashed', label='Median')
            ax.grid()
            if log_scale:
                ax.set_yscale('log', base=2)
            ax.set_ylim(min_perf-1,max_perf+2)
            ax.set_xlabel('Trials')
            if col == 0:
                ax.set_ylabel('Number of actions\ntaken to reach reward')
            fig_utl.hide_spines(ax)
            ax.set_title(params['replay_types'][rep], fontsize=14)
            ax.legend()
        fig.suptitle('Performance - '+titles[key], fontsize=15, y=1.02)
        plt.show()

def plot_visited_states(h_explo, h_r, title='', trial_nb=0, params=params):
    fig, ax = plt.subplots()

    s0_explo = [s0 for (s0, a, s1, r) in h_explo]
    ax.plot(np.arange(len(s0_explo)), s0_explo, linestyle='dashed', marker='o', label='Exploration', color='indianred')
    
    if isinstance(h_r[0],tuple): # already trajectory (replay types 0,1,2,4)
        s0_r = [s0 for (s0, a, s1, r) in h_r]
        ax.plot(len(s0_explo)+np.arange(len(s0_r)), s0_r, linestyle='dashed', marker='o', label='Replay', color='seagreen')
    
    elif isinstance(h_r[0],list): # list of sequences (replay type 3)
        seqs = [[s0 for (s0, a, s1, r) in seq] for seq in h_r]
        ymin = 0
        ymax = params['nS']+1
        i_0 = len(s0_explo)
        for i, s0_r in enumerate(seqs):
            if i == 0:
                ax.plot(i_0+np.arange(len(s0_r)), s0_r, linestyle='dashed', marker='o', label='Replay', color='seagreen')
            else:
                ax.plot(i_0+np.arange(len(s0_r)), s0_r, linestyle='dashed', marker='o', color='seagreen')
            i_0 += len(s0_r)
            ax.vlines(i_0, ymin, ymax, linestyle='dashed', color='gray')

    ax.set_xlabel('Transitions')
    ax.set_ylabel('State')
    fig_utl.legend_out(ax, title='Trial phase')
    fig_utl.hide_spines(ax)
    ax.set_title('Visited states - Trial {}\n'.format(trial_nb)+title)
    plt.show()
