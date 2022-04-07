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
from analyzes_MF_MB import test_pairwise, test_groups
from simulations_MF_MB import recover_data
params = PRMS.params

check_reload = False # signal successful importation at the end of the file


colors_replays = {0: 'royalblue',
                  1: 'orange',
                  2: 'forestgreen',
                  3: 'orchid',
                  4: 'k',
                  -1:'white'} 
cmaps_replays = {0: 'Blues',
                  1: 'YlOrBr',
                  2: 'YlGn',
                  3: 'RdPu',
                  4: 'Greys'} 

# ********** AUXILIARY FUNCTIONS *************** #

def legend_replay_types(ax, params=params, fontsize=10):
    labels = [params['replay_types'][rep] for rep in params['replay_refs']]
    colors = [colors_replays[rep] for rep in params['replay_refs']]
    handles = [mpatches.Patch(facecolor=c, label=l) for l, c in zip(labels, colors)]
    legend = ax.legend(handles=handles, title="Replay types", loc=2, bbox_to_anchor=(1,1), fontsize=fontsize)
    legend.get_title().set_fontsize(fontsize)


def create_moustaches(ax, y, ylow, yup, x=None):
    if x is None:
        x = np.arange(len(y))
    error_low = np.array(y) - np.array(ylow)
    error_up = np.array(yup) - np.array(y)
    ax.bar(x=x, height=y, yerr=(error_low, error_up), alpha=0, align='center', ecolor='black', capsize=5)
    ax.scatter(x, y, color='white', edgecolor='k', zorder=100)


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

def plot_learning_curves(LC_pop, mode='median', log_scale=True, deterministic=True, fig_name='', fontsize=13, fontsize_leg=12, colors_replays=colors_replays, shaded=True, params=params):
    '''Convergence curves.''' 
    fig, ax = plt.subplots(figsize=(10,6))
    for rep in params['replay_refs'] :
        if mode=='mean':
            mean = LC_pop['Mean'].loc[(LC_pop['Replay type']==rep)].to_numpy()
            std = LC_pop['STD'].loc[(LC_pop['Replay type']==rep)].to_numpy()
            y = mean
            ylow = mean-std
            yup = mean+std
        elif mode=='median':
            ylow = LC_pop['Q1'].loc[(LC_pop['Replay type']==rep)].to_numpy()
            y = LC_pop['Q2'].loc[(LC_pop['Replay type']==rep)].to_numpy()
            yup = LC_pop['Q3'].loc[(LC_pop['Replay type']==rep)].to_numpy()
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
    plt.show()


def plot_distribution_perf_trial(rep, t, Data, LC, params=params):
    Dt = Data['Performance'].loc[(Data['Replay type']==rep)&(Data['Trial']==t)]
    h = sns.distplot(Dt, hist = False, kde = True, rug = True,
                 color = 'k', rug_kws={'color': 'gray'})
    m = LC['Mean'].loc[(LC['Replay type']==rep)&(LC['Trial']==t)].to_numpy()[0]
    Q1 = LC['Q1'].loc[(LC['Replay type']==rep)&(LC['Trial']==t)].to_numpy()[0]
    Q2 = LC['Q2'].loc[(LC['Replay type']==rep)&(LC['Trial']==t)].to_numpy()[0]
    Q3 = LC['Q3'].loc[(LC['Replay type']==rep)&(LC['Trial']==t)].to_numpy()[0]
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
    '''Splits data into groups according to their replay type.
    Removes NaN lines (non-converging times) if necessary.
    Compute median and quatiles (or means and STDS) across groups.'''
    data_groups = [Data[var_name].loc[Data['Replay type']==rep] for rep in params['replay_refs']]
    data_groups = [d.dropna() for d in data_groups] # remove None lines
    Q1, Q2, Q3 = [], [], []
    for d in data_groups:
        q1, q2, q3 = np.percentile(d, [25, 50, 75])
        Q1.append(q1)
        Q2.append(q2)
        Q3.append(q3)
    # Means = [np.mean(Data[var_name].loc[Data['Replay type']==rep]) for rep in params['replay_refs']]
    # STD = [np.std(Data[var_name].loc[Data['Replay type']==rep]) for rep in params['replay_refs']]
    return Q1, Q2, Q3, data_groups

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

def create_violin_plot(ax, data, alpha=0.5, params=params):
    parts = ax.violinplot(data, positions=[k for k in range(len(data))],
                          showmeans=False, showmedians=False, showextrema=False)
    for pc, rep in zip(parts['bodies'], params['replay_refs']):
        pc.set_facecolor(colors_replays[rep])
        pc.set_edgecolor('white')
        pc.set_alpha(alpha)

def plot_violins_replays(ax, data, Q1, Q2, Q3, stats=[], params=params, log=False,
                        ylab='', ax_title='', fontsize=13, fontsize_title=14,
                        fontsize_leg=11, leg=True, alpha=0.5, dh_scale = 0.03):
    # Violin plots
    create_violin_plot(ax, data, params=params)
    create_moustaches(ax, Q2,Q1,Q3)
    fig_utl.hide_spines(ax, sides=['right', 'top', 'bottom'])
    fig_utl.hide_ticks(ax, 'x')
    ax.set_title(ax_title, fontsize=fontsize_title)
    ax.set_xlabel('Replay types', fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)
    if leg:
        legend_replay_types(ax, params=params, fontsize=fontsize_leg)
    # Statistical differences
    texts = ['**' if stat[2]<0.001 else '*' for stat in stats]
    ymin, ymax = ax.get_ylim()
    dh = dh_scale*ymax
    heights = [ymax+i*dh*2 for i in range(len(stats))] # uniform spacing of statistical bars
    dh_list = [dh for i in range(len(stats))] # uniform cap size
    if log:
        ax.set_yscale('log', base=2)
        loghmin = np.log2(ymax)
        loghmax = loghmin + 0.25*(np.log(ymax)-np.log(ymin+1))
        heights = [2**y for y in np.linspace(loghmin, loghmax, len(stats))]
        h0 = 2**(loghmin - (loghmax-loghmin)/len(stats))
        dh_list = [(np.sqrt(heights[0]/h0)-1)*h0]+[(np.sqrt(heights[i+1]/heights[i])-1)*heights[i] for i in range(len(heights)-1)]
    for h, dh, stat, text in zip(heights, dh_list, stats, texts):
        x1 = params['replay_refs'].index(stat[0])
        x2 = params['replay_refs'].index(stat[1])
        if stat[0]==0: # comparison between no replay and replay
            color='gray'
        else: # comparison between two replay methods
            color = 'k'
        significant_difference_bars(ax, x1, x2, h=h, dh=dh, text=text, color=color, linewidth=2)

def plot_comparison(Data0, Data1, variable='perf', plot_stats=True, thres=0.05, params=params, deterministic=True, log=False):
    if variable == 'conv':
        var_name = 'Convergence trial'
        ylab = 'Trial number'
        ax_title0 = 'First learning phase'
        ax_title1 = 'Learning with new reward state'
        sup_title = 'Time to convergence'
    elif variable == 'perf':
        var_name = 'Mean'
        ylab = 'Number of actions\ntaken to reach reward'
        ax_title0 = 'First 5 trials'
        ax_title1 = 'Last 5 trials'
        sup_title = 'Performance'
    Q1_0, Q2_0, Q3_0, data0 = create_data_groups(Data0, var_name=var_name, params=params)
    Q1_1, Q2_1, Q3_1, data1 = create_data_groups(Data1, var_name=var_name, params=params)
    if plot_stats:
        Hvalue0, pvalue0 = test_groups(Data0, var_name, params=params, display=False)
        if pvalue0 < thres:
            stats0 = test_pairwise(Data0, var_name, thres=thres, params=params, display=False)
        else:
            stats0 = []
        Hvalue1, pvalue1 = test_groups(Data1, var_name, params=params, display=False)
        if pvalue1 < thres:
            stats1 = test_pairwise(Data1, var_name, thres=thres, params=params, display=False)
        else:
            stats1 = []
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    plot_violins_replays(axes[0], data0, Q1_0, Q2_0, Q3_0, stats=stats0, params=params, 
                        ylab=ylab, ax_title=ax_title0, leg=False, log=log)
    plot_violins_replays(axes[1], data1, Q1_1, Q2_1, Q3_1, stats=stats1, params=params, 
                        ylab='', ax_title=ax_title1, leg=True, log=log)
    if deterministic:
        ttl = ' - Deterministic environment'
    else:
        ttl = ' - Stochastic environment'
    fig.suptitle(sup_title+ttl)
    plt.show()



# ******************* OPTIMIZATION OF ALPHA PARAMETER ************************ #

def compare_alpha_values_replays(Data, ax=None, deterministic=True, fontsize=13, params=params):
    alpha_vals = params['alpha_vals']
    dx = (3/4)/len(params['replay_refs']) # space between bars for one alpha value
    x = np.array([k for k in range(len(alpha_vals))])
    dx_center = (2/3)/2 # position of label fo alpha value
    w = 0.55*dx
   
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,5))
        if deterministic:
            fig_title = '\n Deterministic environment'
        else:
            fig_title = '\n Stochastic environment'
        ax.set_title(r'Performance as a function of the learning rate $\alpha$'+fig_title, fontsize=fontsize)
        show = True
    else:
        show = False
    for i, rep in enumerate(params['replay_refs']) :
        label = params['replay_types'][rep]
        color = colors_replays[rep]
        ylow = Data['Q1'].loc[Data['Replay type']==rep].to_numpy()
        y = Data['Q2'].loc[Data['Replay type']==rep].to_numpy()
        yup = Data['Q3'].loc[Data['Replay type']==rep].to_numpy()
        pos = x + i*dx
        error_low = y - ylow
        error_up = yup - y
        error = [error_low, error_up]
        ax.bar(x=pos, height=y, yerr=error, width=w, color=color, align='center', alpha=0.8, ecolor='black', capsize=5)
        ax.scatter(pos, y, color='white', edgecolor='k', zorder=100)
        ax.set_xticks(x+dx_center)
        ax.set_xticklabels(alpha_vals)
        legend_replay_types(ax, params)
    fig_utl.hide_spines(ax, sides=['right', 'top'])
    ax.set_xlabel(r'$\alpha$', fontsize=fontsize)
    ax.set_ylabel('Mean number of actions\n taken to reach reward\n across trials and individuals', fontsize=fontsize)
    if show:
        plt.show()


def compare_alpha_det_stoch_all(Data, ax=None, params=params, fontsize=13, dx = 1/4, w_scale=0.7):
    alpha_vals = params['alpha_vals']
    Q1_det = Data['Q1 D'].to_numpy()
    Q2_det = Data['Mean D'].to_numpy()
    Q3_det = Data['Q3 D'].to_numpy()
    Q1_sto = Data['Q1 S'].to_numpy()
    Q2_sto = Data['Mean S'].to_numpy()
    Q3_sto = Data['Q3 S'].to_numpy()
    Q1_tot = Data['Q1 tot'].to_numpy()
    Q2_tot = Data['Mean tot'].to_numpy()
    Q3_tot = Data['Q3 tot'].to_numpy()

    w = w_scale*dx
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(15,5))
        ax.set_title(r'Comparison of learning rates $\alpha$', fontsize=fontsize)
        show = True
    else:
        show = False
    ax.bar(x=pos, height=y, yerr=error, width=w, color=colors, align='center', alpha=0.8, ecolor='k', capsize=5)
    ax.scatter(pos, y, color='white', edgecolor='k', zorder=100)
    fig_utl.hide_spines(ax, sides=['right', 'top'])
    for i, (x,h) in enumerate(zip(pos, yup)):
        ax.text(x, h+0.1*np.min(yup), labs[i], ha='center', fontsize=fontsize)
    
    ax.scatter(0,0, alpha=0, label='D: Deterministic environment')
    ax.scatter(0,0, alpha=0, label='S: Stochastic environment')
    ax.scatter(0,0, alpha=0, label='D+S: All data')
    ax.scatter(i_opt, 1.5*Q3_tot[i_opt], marker='*', s=200, color='k', label='Chosen value\n(optimizing total performance)')
    ax.bar(x=pos[3*i_opt+2], height=y[3*i_opt+2], width=w, color=colors[3*i_opt+2], align='center', alpha=0.8, edgecolor='k', linewidth=2.5)
    ax.set_xticks(np.arange(len(alpha_vals)))
    ax.set_xticklabels(alpha_vals)
    ax.set_xlabel(r'$\alpha$', fontsize=fontsize)
    ax.set_ylabel('Mean number of actions\n taken to reach reward\n across trials and individuals', fontsize=fontsize)
    ax.legend(bbox_to_anchor=(1,1), fontsize=fontsize)
    if show:
        plt.show()




# ******************* DISTRIBUTION OF Q-VALUES ************************ #

def plot_Q_distribution(H, ax, color='white', edgecolor='white', params=params, log=False, alpha=0.7):
    if type(H) is dict:
        h, hlow, hup = H['Q2'], H['Q1'], H['Q3']
        create_moustaches(ax, h, hlow, hup)
    else:
        h = H
        edgecolor = 'k'
    ax.bar(np.arange(len(h)), h, color=color, edgecolor=edgecolor, alpha=alpha)
    if log:
        ax.set_yscale('log', base=2)
    positions = np.linspace(0,1,len(h)) 
    ax.xaxis.set_ticks(np.arange(len(h)+1)-0.5)
    ax.xaxis.set_ticklabels(np.round(np.linspace(0,1,len(h)+1), decimals=2))
    ax.set_xlabel('Q-values')

def plot_Q_distributions_replays(H, ax=None, params=params, log=False):
    n_plots = len(params['replay_refs'])
    if ax is None:
        fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(5*n_plots,5), sharey=True)
        fig.suptitle('Q-value coefficients')
        show = True
    else:
        show = False
    for ax, rep in zip(axes, params['replay_refs']):
        plot_Q_distribution(H[rep], ax, color=colors_replays[rep], params=params, log=log)
        if rep==0:
            ax.set_ylabel('Distribution')
            fig_utl.hide_spines(ax)
        else:
            fig_utl.hide_spines(ax, sides=['left','right','top'])
            fig_utl.hide_ticks(ax, 'y')
        ax.set_title(params['replay_types'][rep])
    if show:
        plt.show()

def compare_distributions(EMD, Stats=None, params=params):
    # mask out the lower triangle
    mask =  np.tri(EMD.shape[0], k=-1)
    M = np.ma.array(EMD, mask=mask) 
    M = np.transpose(M)
    cmap = cm.get_cmap('YlOrRd', 100)
    cmap.set_bad('w')

    fig, ax = plt.subplots()
    im = ax.imshow(M, cmap=cmap)
    # major ticks
    ax.set_xticks(np.arange(0, M.shape[0])) 
    ax.set_yticks(np.arange(0, M.shape[1]))
    labels = [params['replay_types'][rep] for rep in params['replay_refs']]
    if M.shape[0] > len(params['replay_refs']): # also display distance to the optimal distribution
        labels.append(params['replay_types'][-1]) # 'Optimal policy'
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    # minor ticks
    ax.set_xticks(np.arange(-.5, M.shape[0]), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[1]), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2, zorder=100)
    fig_utl.hide_spines(ax, sides=['left','right','top','bottom'])
    ax.tick_params(axis=u'both', which=u'both',length=0)
    if Stats is not None:
        for i in range(M.shape[0]):
            for j in range(i,M.shape[1]):
                if Stats[i,j,1] < 0.001:
                    s = '**'
                elif Stats[i,j,1] < 0.05:
                    s = '*'
                else:
                    s = ''
                ax.text(i,j,s, ha='center', va='center')
    ax.set_title('Distances between distributions (EMD)')
    fig.colorbar(im)
    plt.show()

def plot_EMD_distance(EMD, params=params, fontsize=10):
    n_distrib = EMD.shape[0]
    maxEMD = np.max(EMD)
    minEMD = np.min(EMD)
    dx = 0.1*(maxEMD-minEMD)
    fig, axes = plt.subplots(n_distrib, n_distrib, figsize=(6,6))
    for j in range(n_distrib):
        for k in range(n_distrib):
            ax = axes[j,k]
            ax.set_xlim(minEMD-dx, maxEMD+dx)
            if k < j:
                data = EMD[j,k,:]
                h = sns.distplot(data, hist=False, kde=True, rug=True,
                             color='k', rug_kws={'color': 'gray'}, ax=ax)
                ax.axvline(np.median(data), color='k', linestyle='dashed')
                fig_utl.hide_spines(ax, sides=['left','right','top'])
                fig_utl.hide_ticks(ax, 'y')
                if k == n_distrib-1:
                    ax.set_ylabel('EMD')
            elif j == k:
                ax.axvline(0, color='k', linestyle='dashed')
                fig_utl.hide_spines(ax, sides=['left','right','top'])
                fig_utl.hide_ticks(ax, 'y')
            else:
                fig_utl.hide_spines(ax, sides=['left','right','top', 'bottom'])
                fig_utl.hide_ticks(ax, 'y')
                fig_utl.hide_ticks(ax, 'x')
    refs = params['replay_refs'].copy()
    if EMD.shape[0] > len(params['replay_refs']): # also display distance to the optimal distribution
        refs.append(-1)
    for j in range(n_distrib):
        ax = axes[j,0]
        props = dict(facecolor=colors_replays[refs[j]], alpha=0.5, boxstyle='round')
        ax.text(-0.1, 0.5, params['replay_types'][refs[j]], transform=ax.transAxes, 
                fontsize=fontsize, va='center', ha='right', bbox=props)
        ax = axes[n_distrib-1,j]
        props = dict(facecolor=colors_replays[refs[j]], alpha=0.5, boxstyle='round')
        ax.text(0.5, -0.25, params['replay_types'][refs[j]], transform=ax.transAxes, 
                fontsize=fontsize, ha='right', va='top', bbox=props, rotation=45)
    plt.show()



# ******************* FRECHET DISTANCES ************************ #

from mpl_toolkits.axes_grid1 import make_axes_locatable

def set_env(deterministic):
    if deterministic:
        env = '_D'
        ttl = 'Deterministic environment'
    else:
        env = '_S'
        ttl = 'Stochastic environment'
    return env, ttl

def compare_distances_trajectories(deterministic=True, params=params, scale=3):
    env, ttl = set_env(deterministic)
    n_plots = len(params['replay_refs'])
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots*scale*1.3, scale))
    fig.suptitle('Similarity between trajectories')
    for i, (ax, rep) in enumerate(zip(axes, params['replay_refs'])):
        DistMat_pop = recover_data('Frechet{}'.format(rep)+env, df=False)
        M = np.mean(DistMat_pop, axis=0)

        cmap = cmaps_replays[rep]
        im = ax.imshow(M, cmap=cmap)
        ax.set_xlabel('Trials')
        if i == 0:
            ax.set_ylabel('Trials')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        if i == n_plots -1:
            cbar.set_label('Mean Frechet distance\nbetween trajectories (a.u.)')
    plt.show()

def compare_distances_initial_trajectory(trial=0, deterministic=True, params=params):
    env, ttl = set_env(deterministic)
    fig, ax = plt.subplots()
    for rep in params['replay_refs']:
        DistMat_pop = recover_data('Frechet{}'.format(rep)+env, df=False)
        M = DistMat_pop[:,trial,:]
        q1, q2, q3 = np.percentile(M, [25, 50, 75], axis=0)
        t = np.arange(params['n_trials'])
        curve_shaded(ax, t, q2, q1, q3, label=params['replay_types'][rep], color=colors_replays[rep])
    ax.legend()
    plt.show()


# ******************* FACTORS EXPLAINING PERFORMANCE ************************ #

def create_groups_perfs(PRF, variable_name, params=params):
    data = dict(zip(params['replay_refs'], [None for rep in params['replay_refs']]))
    for rep in params['replay_refs']:
        data[rep] = PRF[variable_name].loc[PRF['Replay type']==rep].to_numpy()
    return data

def plot_correlations(data0, data1, params=params, label0='', label1='', title=''):
    ax = plt.subplot(111)
    for rep in params['replay_refs']:
        d0 = data0[rep]
        d1 = data1[rep]
        corr, pval = spearmanr(d0, d1)
        ax.scatter(d0, d1, s=5, color=colors_replays[rep], label=params['replay_types'][rep]+r', $\rho = {}$, p-val = {}'.format(np.round(corr,decimals=1), np.round(pval, decimals=3)))
        if pval < 0.05:
            x_m = np.mean(d0)
            y_m = np.mean(d1)
            y0 = y_m - x_m*corr
            xmin, xmax = np.min(d0), np.max(d0)
            ymin = y0 + corr*xmin
            ymax = y0 + corr*xmax
            ax.plot([xmin,xmax],[ymin,ymax], linestyle='dashed', color=colors_replays[rep])
    ymax = max([np.max(data1[rep]) for rep in params['replay_refs']])
    ymin = max([np.min(data1[rep]) for rep in params['replay_refs']])
    ax.set_ylim(ymin,ymax)
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)
    fig_utl.hide_spines(ax, sides=['right', 'top'])
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1,1))
    plt.show()

