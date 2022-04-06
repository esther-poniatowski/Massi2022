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

def show_trajectory(ax, h_trial, x_states, y_states, params=params, 
                    uniform_col=False, cmap_name='rainbow', jitter=0.05):
    if len(h_trial) > 0: # h_trial != [] (no replay)
        # 1) Define visual properties
        if uniform_col: # plot trajectories in gray, dashed
            linestyle = 'dashed'
            linewidth = 1.5
            zorder = 10
            colors = ['gray' for t in range(len(h_trial))]
        else: # plot trajectories with color gradient, solid
            linestyle = 'solid'
            linewidth = 2.5
            zorder = 100
            if isinstance(h_trial[0],tuple): 
            # already trajectory (replay types 0,1,2,4)
            # 1 transition = 1 color
                colors = discrete_colormap(cmap_name, len(h_trial))
            elif isinstance(h_trial[0],list): 
            # list of sequences (replay type 3)
            # 1 sequence = 1 color
                colors = discrete_colormap(cmap_name, params['n_seq_d'])
        # 2) Plot trajectories
        if isinstance(h_trial[0],tuple): # already trajectory (replay types 0,1,2,4)       
            X = [[x_states[s0], x_states[s1]] for s0, a, s1, r in h_trial]
            Y = [[y_states[s0], y_states[s1]] for s0, a, s1, r in h_trial]
            X, Y = add_jitter(X, Y, jit=jitter)
            for t in range(len(X)):
                ax.plot(X[t], Y[t], linewidth=linewidth, linestyle=linestyle, color=colors[t], zorder=zorder)
        elif isinstance(h_trial[0],list): # list of sequences (replay type 3)
            for i, seq in enumerate(h_trial):
                X = [[x_states[s0], x_states[s1]] for s0, a, s1, r in seq]
                Y = [[y_states[s0], y_states[s1]] for s0, a, s1, r in seq]
                X, Y = add_jitter(X, Y, jit=0.1)
                for t in range(len(X)):
                    ax.plot(X[t], Y[t], linewidth=linewidth, linestyle=linestyle, color=colors[i], zorder=zorder)



# ************* SANITY CHECKS *************** #

def plot_performance(Performance, title=''):
    fig, ax = plt.subplots()
    ax.plot(Performance, linestyle='dashed', marker='o', color='k')
    ax.set_xlabel('Trials')
    ax.set_ylabel('Number of actions taken to reach reward')
    fig_utl.hide_spines(ax)
    ax.set_title('Performance\n'+title)
    plt.show()

def plot_performance_replays(LC_indiv, LC_pop, params=params, scale=4, log_scale=True,
                            epoch='l0', titles={'l0':'First learning phase', 
                                                'l1':'Learning with new reward location', 
                                                'g':'Generalization with new starting point'}):
    ncols = len(params['replay_refs'])
    max_indiv = max([max(LC_indiv['Performance'].loc[LC_indiv['Replay type']==rep]) for rep in params['replay_refs']])
    min_indiv = min([min(LC_indiv['Performance'].loc[LC_indiv['Replay type']==rep]) for rep in params['replay_refs']])
    max_pop = max([max(LC_pop['Q3'].loc[LC_pop['Replay type']==rep]) for rep in params['replay_refs']])
    min_pop = min([min(LC_pop['Q1'].loc[LC_pop['Replay type']==rep]) for rep in params['replay_refs']])
    max_tot = min(max_indiv, max_pop)
    min_tot = min(min_indiv, min_pop)

    fig, axes = plt.subplots(1, ncols, figsize=(scale*ncols,scale))
    for col, rep in enumerate(params['replay_refs']):
        ax = axes[col]
        y_indiv = LC_indiv['Performance'].loc[LC_indiv['Replay type']==rep].to_numpy()
        ylow = LC_pop['Q1'].loc[(LC_pop['Replay type']==rep)].to_numpy()
        y = LC_pop['Q2'].loc[(LC_pop['Replay type']==rep)].to_numpy()
        yup = LC_pop['Q3'].loc[(LC_pop['Replay type']==rep)].to_numpy()
        t = np.arange(0, len(y))
        ax.plot(y_indiv, marker='o', color=colors_replays[rep], label='Individual')
        curve_shaded(ax, t, y, ylow, yup, color=colors_replays[rep], linestyle='dashed', label='Median')
        ax.grid()
        if log_scale:
            ax.set_yscale('log', base=2)
        ax.set_ylim(min_tot-1,max_tot+2)
        ax.set_xlabel('Trials')
        if col == 0:
            ax.set_ylabel('Number of actions\ntaken to reach reward')
        fig_utl.hide_spines(ax)
        ax.set_title(params['replay_types'][rep], fontsize=14)
        ax.legend()
    fig.suptitle('Performance - '+titles[epoch], fontsize=15, y=1.02)
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



