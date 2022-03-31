#!/usr/bin/env python

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr

import figures_utils as fig_utl
import parameters_MF_MB as PRMS
from functions_MF_MB import V_from_Q
from analyzes_MF_MB import test_pairwise
params = PRMS.params

check_reload = False # signal successful importation at the end of the file


colors_replays = {0: 'royalblue',
                  1: 'orange',
                  2: 'forestgreen',
                  3: 'orchid',
                  4: 'k'} 


# ********** AUXILIARY FUNCTIONS *************** #

def legend_replay_types(ax, params=params, fontsize=10):
    labels = [params['replay_types'][rep] for rep in params['replay_refs']]
    colors = [colors_replays[rep] for rep in params['replay_refs']]
    handles = [mpatches.Patch(facecolor=c, label=l) for l, c in zip(labels, colors)]
    legend = ax.legend(handles=handles, title="Replay types", loc=2, bbox_to_anchor=(1,1), fontsize=fontsize)
    legend.get_title().set_fontsize(fontsize)

def significant_difference_bars(ax, x1, x2, h=None, text='*',
                                dh=None, dh_scale=0.03,
                                color='k', linewidth=2):
    if h is None:
        h = ymax
    if dh is None:
        ymin, ymax = ax.get_ylim()
        dh = dh_scale*(ymax-ymin)
    ax.plot([x1,x2],[h,h], color=color, linewidth=linewidth)
    ax.plot([x1,x1],[h,h-dh], color=color, linewidth=linewidth)
    ax.plot([x2,x2],[h,h-dh], color=color, linewidth=linewidth)
    ax.text(min(x1,x2)+(max(x1,x2)-min(x1,x2))/2, h, text, 
        fontsize=15,  ha='center', va='center', zorder=10)


# ********** POPULATION DATA - LEARNING CURVES *************** #

def curve_shaded(ax, x, y, ylow, yup, label='',
                color='k', alpha=0.2, linestyle='solid',
                ms=5, mfc='w', mew=1.5):
    # ms : marker size
    # mec : markeredge color
    # mew : marker edge width
    # mfc : marker face color
    ax.plot(x, y, label=label, color=color, linestyle=linestyle,
            marker='o', ms=ms, mec=color, mfc='w', mew=1.5)
    ax.plot(x, yup, color=color, alpha=0.2, linestyle='dashed')
    ax.plot(x, ylow, color=color, alpha=0.2, linestyle='dashed')
    ax.fill_between(x, ylow, yup, alpha=0.2, facecolor=color)

def plot_learning_curves(Perf_pop, mode='median', log_scale=True, deterministic=True, save=True, fig_name='', fontsize=13, fontsize_leg=12, colors_replays=colors_replays, shaded=True, params=params):
    '''Convergence curves.''' 
    fig, ax = plt.subplots(figsize=(10,6))
    for rep in params['replay_refs'] :
        if mode=='mean':
            mean = Perf_pop['Mean'].loc[(Perf_pop['Replay type']==rep)].to_numpy()
            std = Perf_pop['STD'].loc[(Perf_pop['Replay type']==rep)].to_numpy()
            y = mean
            ylow = mean-std
            yup = mean+std
        elif mode=='median':
            ylow = Perf_pop['Q1'].loc[(Perf_pop['Replay type']==rep)].to_numpy()
            y = Perf_pop['Q2'].loc[(Perf_pop['Replay type']==rep)].to_numpy()
            yup = Perf_pop['Q3'].loc[(Perf_pop['Replay type']==rep)].to_numpy()
        t = np.arange(1, len(y)+1)
        label = params['replay_types'][rep]
        color = colors_replays[rep]
        if shaded :
            curve_shaded(ax, t, y, ylow, yup, label=label, color=color)
        else: # version error bars
            ax.errorbar(t, mean, yerr= std, label=label, color=color)
    ax.grid()
    if log_scale:
        ax.set_yscale('log', base=2)
    fig_utl.hide_spines(ax)
    ax.set_xlabel('Trials', fontsize=fontsize)
    ax.set_ylabel('Number of actions to get to the reward', fontsize=fontsize)
    if deterministic:
        fig_title = '\n Deterministic environment'
    else:
        fig_title = '\n Stochastic environment'
    ax.set_title('Learning curves'+fig_title, fontsize=fontsize)
    legend = ax.legend(title='Replay types', fontsize=fontsize_leg)
    legend.get_title().set_fontsize(fontsize_leg)
    if save:
        plt.savefig('Figures/learning_conv_'+fig_name)
    plt.show()


def plot_distribution_perf_trial(rep, t, Data, Perf, params=params):
    Dt = Data['Performance'].loc[(Data['Replay type']==rep)&(Data['Trial']==t)]
    h = sns.distplot(Dt, hist = False, kde = True, rug = True,
                 color = 'k', rug_kws={'color': 'gray'})
    m = Perf['Mean'].loc[(Perf['Replay type']==rep)&(Perf['Trial']==t)].to_numpy()[0]
    Q1 = Perf['Q1'].loc[(Perf['Replay type']==rep)&(Perf['Trial']==t)].to_numpy()[0]
    Q2 = Perf['Q2'].loc[(Perf['Replay type']==rep)&(Perf['Trial']==t)].to_numpy()[0]
    Q3 = Perf['Q3'].loc[(Perf['Replay type']==rep)&(Perf['Trial']==t)].to_numpy()[0]
    plt.axvline(m, color='k', linestyle='dashed', label='Mean')
    plt.axvline(Q1, color='red', linestyle='dashed', label='Q1')
    plt.axvline(Q2, color='orangered', linestyle='dashed', label='Q2')
    plt.axvline(Q3, color='orange', linestyle='dashed', label='Q3')
    plt.legend()
    fig_utl.hide_spines(plt.gca())
    plt.xlabel('Performance')
    plt.ylabel('Distribution')
    plt.title('{}, Trial {}'.format(params['replay_types'][rep], t))
    plt.show()


# ********** POPULATION DATA - SUMMARY STATISTICS *************** #

def create_data_groups(Data, var_name='Mean', params=params):
    # Compute means and STDS across groups
    Means = [np.mean(Data[var_name].loc[Data['Replay type']==rep]) for rep in params['replay_refs']]
    STD = [np.std(Data[var_name].loc[Data['Replay type']==rep]) for rep in params['replay_refs']]
    data_groups = [Data[var_name].loc[Data['Replay type']==rep] for rep in params['replay_refs']]
    return Means, STD, data_groups

def create_violin_plot(ax, data_group, x, mean, std,
                      color, label='', alpha=0.5):
    data_group = data_group.dropna() # remove None lines
    parts = ax.violinplot(data_group, positions=x,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('white')
        pc.set_alpha(alpha)  
    ax.bar(x, mean, yerr=std, alpha=0, align='center', ecolor='black', capsize=5)
    ax.scatter(x, mean, color='white', edgecolor='k', zorder=100)


def plot_violins_replays(ax, data, Means, STD, stats=[], params=params, 
                        ylab='', fig_title='', fontsize=13, 
                        fontsize_leg=11, leg=True, dh_scale=0.03):
    for rep in params['replay_refs'] :
        create_violin_plot(ax, data[rep], np.array([rep]), 
                        Means[rep], STD[rep],
                        colors_replays[rep], label=params['replay_types'][rep])
    ax.xaxis.set_ticklabels([])
    fig_utl.hide_spines(ax, sides=['right', 'top', 'bottom'])
    fig_utl.hide_ticks(ax, 'x')
    ax.set_title(fig_title)
    ax.set_xlabel('Replay types', fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)
    if leg:
        legend_replay_types(ax, params=params, fontsize=fontsize_leg)
    ymin, ymax = ax.get_ylim()
    dh = dh_scale*ymax
    heights = [ymax+i*dh*2 for i in range(len(stats))]
    texts = ['**' if stat[2]<0.001 else '*' for stat in stats]
    for h, stat, text in zip(heights, stats, texts):
        if 0 in stat:
            color='gray'
        else:
            color = 'k'
        significant_difference_bars(ax, stat[0], stat[1], h=h, text=text,
                                color=color, linewidth=2, dh=dh*0.9)


def plot_comparison_convergence(CV0, CV1, plot_stats=True, params=params, deterministic=True, save=True):
    Means0, STD0, data0 = create_data_groups(CV0, var_name='Convergence trial', params=params)
    Means1, STD1, data1 = create_data_groups(CV1, var_name='Convergence trial', params=params)
    if plot_stats:
        _, stats0 = test_pairwise(CV0, 'Convergence trial', params=params, display=False)
        _, stats1 = test_pairwise(CV1, 'Convergence trial', params=params, display=False)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    plot_violins_replays(axes[0], data0, Means0, STD0, stats=stats0, params=params, 
                        ylab='Trial number', fig_title='First learning phase', leg=False)
    plot_violins_replays(axes[1], data1, Means1, STD1, stats=stats1, params=params, 
                        ylab='', fig_title='Learning with new reward state', leg=True)
    fig.suptitle('Time to convergence')
    if deterministic:
        ttl = ' - Deterministic environment'
    else:
        ttl = ' - Stochastic environment'
    fig.suptitle('Convergence'+ttl)
    plt.show()

def plot_comparison_performance(Perf0, Perf1, plot_stats=True, params=params, deterministic=True, save=True):
    Means0, STD0, data0 = create_data_groups(Perf0, var_name='Mean', params=params)
    Means1, STD1, data1 = create_data_groups(Perf1, var_name='Mean', params=params)
    if plot_stats:
        _, stats0 = test_pairwise(Perf0, 'Mean', params=params, display=False)
        _, stats1 = test_pairwise(Perf1, 'Mean', params=params, display=False)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    plot_violins_replays(axes[0], data0, Means0, STD0, stats=stats0, params=params, 
                        ylab='Number of actions\ntaken to reach reward', fig_title='First learning phase', leg=False)
    plot_violins_replays(axes[1], data1, Means1, STD1, stats=stats1, params=params, 
                        ylab='', fig_title='Learning with new reward state', leg=True)
    if deterministic:
        ttl = ' - Deterministic environment'
    else:
        ttl = ' - Stochastic environment'
    fig.suptitle('Performance'+ttl)
    plt.show()

def plot_comparison_before_after(Perf_before, Perf_after, plot_stats=True, deterministic=True, figname='', save=True, params=params):
    Means0, STD0, data0 = create_data_groups(Perf_before, var_name='Mean', params=params)
    Means1, STD1, data1 = create_data_groups(Perf_after, var_name='Mean', params=params)
    if plot_stats:
        _, stats_before = test_pairwise(Perf_before, 'Mean', params=params, display=False)
        _, stats_after = test_pairwise(Perf_after, 'Mean', params=params, display=False)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    plot_violins_replays(axes[0], data0, Means0, STD0, stats=stats_before, params=params,  
                        ylab='Number of actions\ntaken to reach reward', fig_title='Before convergence', leg=False)
    plot_violins_replays(axes[1], data1, Means1, STD1, stats=stats_after, params=params,  
                        ylab='', fig_title='After convergence', leg=True)
    if deterministic:
        ttl = ' - Deterministic environment'
    else:
        ttl = ' - Stochastic environment'
    fig.suptitle('Performance'+ttl)
    plt.show()

def plot_comparison_generalization(Perfs, deterministic=True, figname='', save=True, params=params):
    fig, axes = plt.subplots(nrows=1, ncols=len(Perfs), figsize=(len(Perfs)*5,5))
    for ax, Perf in zip(Perfs, axes):
        Means, STD, data = create_data_groups(Perf, var_name='Mean', params=params)
    plot_violins_replays(axes[0], data0, Means0, STD0, params, 
                        ylab='Number of actions\ntaken to reach reward', fig_title='Before convergence', leg=False)
    plot_violins_replays(axes[1], data1, Means1, STD1, params, 
                        ylab='', fig_title='After convergence', leg=True)
    if deterministic:
        ttl = ' - Deterministic environment'
    else:
        ttl = ' - Stochastic environment'
    fig.suptitle('Performance'+ttl)
    plt.show()


def plot_learning_swarm(Perf_pop, params=params):
    ax = sns.swarmplot(x='Replay type', y='Mean', data=Perf_pop, palette=colors_replays)
    # alternative: catplot
    ax = sns.boxplot(x='Replay type', y='Mean', data=Perf_pop,
            showcaps=False,boxprops={'facecolor':'None'},
            showfliers=False,whiskerprops={'linewidth':0})
    plt.show()



# ********** POPULATION DATA - CORRELATIONS *************** #

def plot_correlations(Perf0, Perf1, params=params, label0='', label1=''):
    ax = plt.subplot(111)
    for rep in params['replay_refs']:
        d0 = Perf0['Mean'].loc[Perf0['Replay type']==rep]
        d1 = Perf1['Mean'].loc[Perf1['Replay type']==rep]
        corr, pval = spearmanr(a=d0, b=d1)
        ax.scatter(d0, d1, s=5, color=colors_replays[rep], label=params['replay_types'][rep]+r', $\rho = {}$, p-val = {}'.format(np.round(corr,decimals=1), np.round(pval, decimals=3)))
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)
    fig_utl.hide_spines(ax, sides=['right', 'top'])
    ax.set_title('Correlations in performances\nin different simulation epochs')
    ax.legend(bbox_to_anchor=(1,1))
    plt.show()

# ******************* OPTIMIZATION OF ALPHA PARAMETER ************************ #

def compare_alpha_values_replays(Data, deterministic=True, curve=False, fontsize=13, params=params):
    alpha_vals = params['alpha_vals']
    dx = (3/4)/len(params['replay_refs']) # space between bars for one alpha value
    x = np.array([k for k in range(len(alpha_vals))])
    dx_center = (2/3)/2 # position of label fo alpha value
    w = 0.55*dx
    if curve:
        figsize = (10,5)
    else :
        figsize = (15,5)

    fig, ax = plt.subplots(figsize=figsize)
    for i, rep in enumerate(params['replay_refs']) :
        label = params['replay_types'][rep]
        color = colors_replays[rep]
        ylow = Data['Q1'].loc[Data['Replay type']==rep].to_numpy()
        y = Data['Q2'].loc[Data['Replay type']==rep].to_numpy()
        yup = Data['Q3'].loc[Data['Replay type']==rep].to_numpy()
        if curve:
            curve_shaded(ax, alpha_vals, y, ylow, yup, color=color, label=label)
        else: # bar plot
            pos = x + i*dx
            error_low = y - ylow
            error_up = yup - y
            error = [error_low, error_up]
            ax.bar(x=pos, height=y, yerr=error, width=w, color=color, align='center', alpha=0.8, ecolor='black', capsize=5)
            ax.scatter(pos, y, color='white', edgecolor='k', zorder=100)
    
    if not curve :
        ax.set_xticks(x+dx_center)
        ax.set_xticklabels(alpha_vals)
        legend_replay_types(ax, params)
    else:
        ax.legend(title='Replay types')
    fig_utl.hide_spines(ax, sides=['right', 'top'])
    ax.set_xlabel(r'$\alpha$', fontsize=fontsize)
    ax.set_ylabel('Mean number of actions\n taken to reach reward\n across trials and individuals', fontsize=fontsize)
    if deterministic:
        fig_title = '\n Deterministic environment'
    else:
        fig_title = '\n Stochastic environment'
    ax.set_title(r'Performance as a function of the learning rate $\alpha$'+fig_title, fontsize=fontsize)
    plt.show()


def compare_alpha_det_stoch_all(Data, fontsize=13, params=params):
    alpha_vals = params['alpha_vals']
    Q1_det = Data['Q1 det'].to_numpy()
    Q2_det = Data['Mean det'].to_numpy()
    Q3_det = Data['Q3 det'].to_numpy()
    Q1_sto = Data['Q1 sto'].to_numpy()
    Q2_sto = Data['Mean sto'].to_numpy()
    Q3_sto = Data['Q3 sto'].to_numpy()
    Q1_tot = Data['Q1 tot'].to_numpy()
    Q2_tot = Data['Mean tot'].to_numpy()
    Q3_tot = Data['Q3 tot'].to_numpy()

    dx = 1/4
    w = 0.7*dx
    pos = []
    y = []
    ylow = []
    yup = []
    labs = []
    for x in np.arange(len(alpha_vals)):
        pos += [x-dx, x, x+dx]
        y += [Q2_det[x], Q2_sto[x], Q2_tot[x]]
        ylow += [Q1_det[x], Q1_sto[x], Q1_tot[x]]
        yup += [Q3_det[x], Q3_sto[x], Q3_tot[x]]
        labs += ['D', 'S', 'D+S']
    sample = np.array(pos)/max(pos)
    colors = fig_utl.discrete_colormap(len(alpha_vals), cmap_name='rainbow', sample=sample)
    error_low = np.array(y) - np.array(ylow)
    error_up = np.array(yup) - np.array(y)
    error = [error_low, error_up]
    i_opt = np.argwhere(Q2_tot==np.min(Q2_tot))[0][0]

    fig, ax = plt.subplots(figsize=(15,5))
    ax.bar(x=pos, height=y, yerr=error, width=w, color=colors, align='center', alpha=0.8, ecolor='k', capsize=5)
    ax.scatter(pos, y, color='white', edgecolor='k', zorder=100)
    fig_utl.hide_spines(ax, sides=['right', 'top'])
    for i, (x,h) in enumerate(zip(pos, yup)):
        ax.text(x, h+0.1*np.min(yup), labs[i], ha='center', fontsize=13)
    
    ax.scatter(0,0, alpha=0, label='D: Deterministic environment')
    ax.scatter(0,0, alpha=0, label='S: Stochastic environment')
    ax.scatter(0,0, alpha=0, label='D+S: All data')
    ax.scatter(i_opt, 1.5*Q3_tot[i_opt], marker='*', s=200, color='k', label='Chosen value\n(optimizing total performance)')
    ax.bar(x=pos[3*i_opt+2], height=y[3*i_opt+2], width=w, color=colors[3*i_opt+2], align='center', alpha=0.8, edgecolor='k', linewidth=2.5)
    ax.set_xticks(np.arange(len(alpha_vals)))
    ax.set_xticklabels(alpha_vals)
    ax.set_xlabel(r'$\alpha$', fontsize=13)
    ax.set_ylabel('Mean number of actions\n taken to reach reward\n across trials and individuals', fontsize=13)
    ax.set_title(r'Comparison of learning rates $\alpha$', fontsize=14)
    ax.legend(fontsize=12)
    plt.show()

