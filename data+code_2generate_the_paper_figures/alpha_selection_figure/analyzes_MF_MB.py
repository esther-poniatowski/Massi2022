#!/usr/bin/env python
import json

import numpy as np
import random
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from bioinfokit.analys import stat
import pandas as pd
import sys
import pickle
import sys

import scikit_posthocs as sp


# import functions and parameters
import parameters_MF_MB as PRMS
params = PRMS.params
from simulations_MF_MB import save_data, recover_data


# ********** CONVERGENCE OF LEARNING *************** #

def cv_criterion(series, perc, window):
    '''Determines if and when convergence has been reached.
    :param series: time course of the performance metric (number of actions before getting the reward)
    :param window: number of steps to consider after the current time step
    :param perc: percentage (in [0, 1]), convergence occured if all the values in the window are between x*(1 +- perc)
    :return: array 
        cv_ind: trial index at which convergence occurred
        na_cv: nb of actions in this trial
        cum_na_cv: cumulative number of actions up to this trial
    '''
    b = False
    n = len(series)
    cv_ind = 0
    while (cv_ind < n) and not b:
        x = series[cv_ind]
        sup = x*(1 + perc)
        inf = x*(1 - perc)
        l = min((n - cv_ind - 1), window)
        i = 1
        while i <= l and sup >= series[cv_ind + i] >= inf:
            i = i + 1
        b = (i == l + 1)
        cv_ind = cv_ind + 1
    cv_ind = cv_ind - 1
    if cv_ind != (n - 1):
        na_cv = series[cv_ind]
        cum_na_cv = np.sum(series[0:cv_ind + 1])
        return cv_ind, na_cv, cum_na_cv
    else:
        return None, None, None

def compute_convergence(Data_pop, params=params):
    '''Determines convergence for each individual and replay type.
    :return CV_pop: dataframe
        Rows: 1 row = 1 individual (for 1 replay type)
        Columns : three columns (see cv_criterion)
    :return Non_cv: dictionary counting the non-converging individuals by type of replay
    '''
    replays = [r for r in params['replay_refs'] for i in range(params['n_individuals'])]
    individuals = [i for r in params['replay_refs'] for i in range(params['n_individuals'])]
    empty_data = [0 for i in range(len(individuals))]
    CV_pop = pd.DataFrame({'Replay type':replays,
                            'Individual':individuals,
                            'Convergence trial':empty_data,
                            'Performance at Convergence':empty_data,
                            'Time to Convergence':empty_data})
    Non_cv = dict(zip(params['replay_refs'], [0 for r in params['replay_refs']]))

    for rep in params['replay_refs']:
        for i in range(params['n_individuals']):
            perf = Data_pop['Performance'].loc[(Data_pop['Replay type']==rep)&(Data_pop['Individual']==i)] # extract performance time course of the individual
            cv_ind, na_cv, cum_na_cv = cv_criterion(perf.to_numpy(), params['perc'], params['window']) # determines if convergence has been reached
            CV_pop['Convergence trial'].loc[(CV_pop['Replay type']==rep)&(CV_pop['Individual']==i)] = cv_ind
            CV_pop['Performance at Convergence'].loc[(CV_pop['Replay type']==rep)&(CV_pop['Individual']==i)] = na_cv
            CV_pop['Time to Convergence'].loc[(CV_pop['Replay type']==rep)&(CV_pop['Individual']==i)] = cum_na_cv
            if cv_ind is None:
                Non_cv[rep] += 1
    return CV_pop, Non_cv


def display_convergence(NonCVl0_D, NonCVl1_D, NonCVl0_S, NonCVl1_S):
    print('Deterministic environment')
    print('Number of non-converging trials per type of replay:')
    print('Phase 0', NonCVl0_D)
    print('Phase 1', NonCVl1_D)
    print('---------------------------')
    print('Stochastic environment')
    print('Number of non-converging trials per type of replay:')
    print('Phase 0', NonCVl0_S)
    print('Phase 1', NonCVl1_S)
    print('---------------------------')


# ********** SPLITTING DATA ACCORDING TO SIMULATION EPOCHS *************** #

def split_before_after_change(Data, params=params):
    '''Splits the ata in two groups of trials : before and after the change of reward location.
    Update the corresponding number of trials by creating two distinct parameters dictionaries.'''
    Data0 = Data.loc[Data['Trial'] < params['trial_change']]
    Data1 = Data.loc[Data['Trial'] >= params['trial_change']]
    Data1['Trial'] -= params['trial_change'] # reset trial numbers to start from 0
    params0, params1 = params.copy(), params.copy()
    params0['n_trials'] = params['trial_change']
    params1['n_trials'] = params['n_trials'] - params['trial_change']
    return Data0, Data1, params0, params1

def split_selected_trials(Data, trials=(0,5), params=params):
    '''Splits the data corresponding to the interval of trials retained.'''
    Data_trials = Data.loc[(Data['Trial'] >= trials[0])&(Data['Trial'] < trials[1])]
    return Data_trials


# ********** SUMMARY STATISTICS FOR PERFORMANCE *************** #

def compute_performance_in_time(Data, params=params):
    '''Determines mean and STD of performance over the population along a simulation, for each replay type. 
    To be used for plotting learning curves.
    :param Data: dataframe, obtained from simulate_population()
    :return LC: dataframe ('learning curves')
        Rows: 
            1 row = 1 trial number
        Columns: 
            Replay type
            Trial
            Mean: mean number of actions taken by the population performing this replay type, on this trial
            STD: standard deviation
            Q1: first quartile
            Q2: median number of actions taken in the population
            Q3: third quartile
    '''
    replays = [r for r in params['replay_refs'] for t in range(params['n_trials'])]
    trials = [t for r in params['replay_refs'] for t in range(params['n_trials'])]
    empty_data = [0 for i in range(len(trials))]
    LC = pd.DataFrame({'Replay type':replays,
                        'Trial':trials,
                        'Mean':empty_data,
                        'STD':empty_data,
                        'Q1':empty_data,
                        'Q2':empty_data,
                        'Q3':empty_data})
    for rep in params['replay_refs']:
        print(f'rep" {rep}')
        for t in range(params['n_trials']):
            perf = Data['Performance'].loc[(Data['Replay type']==rep)&(Data['Trial']==t)].to_numpy()
            LC['Mean'].loc[(LC['Replay type']==rep)&(LC['Trial']==t)] = np.mean(perf)
            LC['STD'].loc[(LC['Replay type']==rep)&(LC['Trial']==t)] = np.std(perf)
            Q1, Q2, Q3 = np.percentile(perf, [25, 50, 75])
            LC['Q1'].loc[(LC['Replay type']==rep)&(LC['Trial']==t)] = Q1
            LC['Q2'].loc[(LC['Replay type']==rep)&(LC['Trial']==t)] = Q2
            LC['Q3'].loc[(LC['Replay type']==rep)&(LC['Trial']==t)] = Q3
    return LC

def compute_performance_across_population(Data, tmin=None, tmax=None, params=params):
    '''Determines mean and STD of performance over the population along a simulation, for each replay type. 
    To be used for plotting comparisons between groups.
    If tmin and tmax are given, only the trials between those boundary are used.
    :param Data:
    :param tmin: (resp tmax) number of first (resp. last) trial from which gathering data
    :return PRF: dataframe ('performance')
        Rows: 
            1 row = 1 individual on 1 replay type
        Columns:
            Replay type
            Individual
            Mean: mean number of actions taken by the individual over the selected trials.
            STD: standard deviation of the number of actions taken by this individual.
    '''
    if tmin is None:
        tmin = 0
    if tmax is None:
        tmax = params['n_trials']
    D = Data.loc[(Data['Trial']>=tmin)&(Data['Trial']<tmax)]

    replays = [r for r in params['replay_refs'] for i in range(params['n_individuals'])]
    individuals = [i for r in params['replay_refs'] for i in range(params['n_individuals'])]
    means = []
    stds = []
    all_values = []
    for rep in params['replay_refs']:
        for i in range(params['n_individuals']):
            perf = D['Performance'].loc[(D['Replay type']==rep)&(D['Individual']==i)]
            all_values.append(perf.tolist())
            means.append(np.mean(perf))
            stds.append(np.std(perf))
    return pd.DataFrame({'Replay type': replays,
                         'Individual': individuals,
                         'all': all_values,
                         'Mean': means,
                         'STD': stds})

def identify_representative(Data_pop, LC_pop, params=params, show=True):
    '''Identifies the individual most representative of the mean behavior, for all replays.
    This is the individual whose performance (learning curve) minimizes the difference with the median.'''
    D = np.zeros(params['n_individuals'])
    for i in range(params['n_individuals']):
        for rep in params['replay_refs']:
            Q2 = LC_pop['Q2'].loc[LC_pop['Replay type']==rep].to_numpy() # median LC of the population for this replay type
            data_i = Data_pop['Performance'].loc[(Data_pop['Individual']==i)&(Data_pop['Replay type']==rep)].to_numpy() # LC of the individual on this replay type
            D[i] += np.sum(np.abs(Q2-data_i)) # sum for successive replay types
    i_repr = np.argwhere(D==np.min(D))[0][0]
    if show:
        print('Most representative individual : ', i_repr)
        plt.scatter(np.arange(len(D)), D)
        plt.scatter(i_repr, D[i_repr])
        plt.xlabel('Individuals')
        plt.ylabel('Distance from the median performance\nover all replays')
        plt.show()
    return i_repr



# ********** STATISTICAL TESTS *************** #

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bioinfokit.analys import stat
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def test_groups(Data, var_name, params=params, display=True):
    '''The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal. 
    It is a non-parametric version of ANOVA.
    Note that rejecting the null hypothesis does not indicate which of the groups differs.
    Post hoc comparisons between groups are required to determine which groups are different.
    '''
    groups = [Data[var_name].loc[Data['Replay type']==rep].to_numpy() for rep in params['replay_refs']]
    if var_name == "all":
        groups_ma_giusto = []
        for gr in groups:
            tmp = np.array([json.loads(v) for v in gr], dtype=np.float)
            groups_ma_giusto.append(tmp.reshape(-1))
        print(f'prova {groups_ma_giusto}')
        groups = groups_ma_giusto
    # sys.exit(0)
    Hvalue, pvalue = stats.kruskal(*groups, nan_policy='omit')
    # Hvalue, pvalue = stats.f_oneway(*groups, axis=0)#, nan_policy='omit')
    # result_stat = statsmodels.stats.multicomp.pairwise_tukeyhsd(*groups, Data['Replay type'])
    if display:
        significant_05 = pvalue < 0.05
        significant_001 = pvalue < 0.001
        S = 'Not significant'
        if significant_05:
            S = ' < 0.05'
        if significant_001:
            S = ' < 0.001'
        print('Kruskal-Wallis result : H = {}, p-value = {}'.format(Hvalue, pvalue)+S)
    return Hvalue, pvalue


def test_pairwise(Data, var_name, thres=0.05, params=params, display=True, test='mannwhitney'):
    '''Perfoms Mann-Whitney pairwise comparisons between all groups.
    Note that the major difference between the Mann-Whitney U and the Kruskal-Wallis H is simply that the latter can accommodate more than two groups. 
    :param thres: p-value threshold for significance, 0.05 or 0.001
    '''
    print(Data)
    # sys.exit(0)
    pairs = list(combinations(params['replay_refs'], 2))
    Statistics = pd.DataFrame({'G0': [p[0] for p in pairs],
                              'G1': [p[1] for p in pairs],
                              'Uvalue': [None for p in pairs],
                              'pvalue': [0.0 for p in pairs]})
    if test == 'conover':
        pvalue = sp.posthoc_conover(Data, val_col='all', group_col='Replay type')
    else:
        significant = []
        for p in pairs: # select groups
            g0 = Data.loc[Data['Replay type']==p[0]]
            g1 = Data.loc[Data['Replay type']==p[1]] 
            g0 = g0.dropna() # remove None lines
            g1 = g1.dropna() # remove None lines
            if test == 'mannwhitney':
                Uvalue, pvalue = stats.mannwhitneyu(g0, g1)
            elif test == 'kruskal':
                Uvalue, pvalue = stats.kruskal(g0, g1)
            Statistics['Uvalue'].loc[(Statistics['G0']==p[0])&(Statistics['G1']==p[1])] = Uvalue
            Statistics['pvalue'].loc[(Statistics['G0']==p[0])&(Statistics['G1']==p[1])] = pvalue
            if pvalue < thres:
                significant.append((p[0], p[1], pvalue))
        if display:
            # prin gt(Statistics)
            print('Significant pairs :', significant)

    if test == 'conover':
        return pvalue
    else:
        return significant



# ********** OPTIMIZATION OF THE LEARNING RATE *************** #

def compare_alpha_replays(deterministic=True, params=params):
    '''Compare performance between replay types for each value of alpha.'''
    replays = [r for r in params['replay_refs'] for alpha in params['alpha_vals']]
    alpha_vec = [alpha for r in params['replay_refs'] for alpha in params['alpha_vals']]
    empty_data = [0 for obs in replays]
    D_D, D_S = None, None
    for det, env, D in zip([True, False],['_D','_S'], [D_D, D_S]):
        if det:
            print('Deterministic')
        else:
            print('Stochastic')
        Data = pd.DataFrame({'Replay type':replays,
                        'alpha':alpha_vec,
                        'Mean':empty_data,
                        'STD':empty_data,
                        'Q1':empty_data,
                        'Q2':empty_data,
                        'Q3':empty_data})
        print('Computing...')
        for alpha in params['alpha_vals']:
            print('alpha', alpha)
            data_raw = recover_data('D-alpha{}'.format(alpha)+env)
            data_avg = compute_performance_across_population(data_raw)
            for rep in params['replay_refs']:
                perf = data_avg['Mean'].loc[data_avg['Replay type']==rep].to_numpy()
                m = np.mean(perf)
                std = np.std(perf)
                Q1, Q2, Q3 = np.percentile(perf, [25, 50, 75])
                Data['Mean'].loc[(Data['Replay type']==rep)&(Data['alpha']==alpha)] = m
                Data['STD'].loc[(Data['Replay type']==rep)&(Data['alpha']==alpha)] = std
                Data['Q1'].loc[(Data['Replay type']==rep)&(Data['alpha']==alpha)] = Q1
                Data['Q2'].loc[(Data['Replay type']==rep)&(Data['alpha']==alpha)] = Q2
                Data['Q3'].loc[(Data['Replay type']==rep)&(Data['alpha']==alpha)] = Q3
        print('Done')
        save_data(Data, 'D-alpha'+env)
    return D_D, D_S

def optimize_alpha(params=params):
    '''Identify the alpha value for best performance.
    Criterion : the chosen alpha value is the one which obtains
    the minimal number of iterations to get to the reward,
    summing the deterministic and the stochastic case.'''
    alpha_vals = params['alpha_vals']
    empty_data = [0 for alpha in alpha_vals]
    Data = pd.DataFrame({'alpha':alpha_vals,
                        'Mean D':empty_data,
                        'STD D':empty_data,
                        'Q1 D':empty_data,
                        'Q2 D':empty_data,
                        'Q3 D':empty_data,
                        'Mean S':empty_data,
                        'STD S':empty_data,
                        'Q1 S':empty_data,
                        'Q2 S':empty_data,
                        'Q3 S':empty_data,
                        'Mean tot':empty_data,
                        'STD tot':empty_data,
                        'Q1 tot':empty_data,
                        'Q2 tot':empty_data,
                        'Q3 tot':empty_data})
    print('Computing...')
    for alpha in alpha_vals:
        print('alpha', alpha)
        # Deterministic case
        Data_all = recover_data('D-alpha{}_D'.format(alpha))
        Perf = compute_performance_across_population(Data_all, params=params)
        data_det = Perf['Mean'].to_numpy()
        Data['Mean D'].loc[Data['alpha']==alpha] = np.mean(data_det)
        Data['STD D'].loc[Data['alpha']==alpha] = np.std(data_det)
        Q1, Q2, Q3 = np.percentile(data_det, [25, 50, 75])
        Data['Q1 D'].loc[Data['alpha']==alpha] = Q1
        Data['Q2 D'].loc[Data['alpha']==alpha] = Q2
        Data['Q3 D'].loc[Data['alpha']==alpha] = Q3
        # Stochastic case
        Data_all = recover_data('D-alpha{}_S'.format(alpha))
        Perf = compute_performance_across_population(Data_all, params=params)
        data_sto = Perf['Mean'].to_numpy()
        Data['Mean S'].loc[Data['alpha']==alpha] = np.mean(data_sto)
        Data['STD S'].loc[Data['alpha']==alpha] = np.std(data_sto)
        Q1, Q2, Q3 = np.percentile(data_sto, [25, 50, 75])
        Data['Q1 S'].loc[Data['alpha']==alpha] = Q1
        Data['Q2 S'].loc[Data['alpha']==alpha] = Q2
        Data['Q3 S'].loc[Data['alpha']==alpha] = Q3
        # Total
        data_tot = np.concatenate((data_det,data_sto))
        Data['Mean tot'].loc[Data['alpha']==alpha] = np.mean(data_tot)
        Data['STD tot'].loc[Data['alpha']==alpha] = np.std(data_tot)
        Q1, Q2, Q3 = np.percentile(data_tot, [25, 50, 75])
        Data['Q1 tot'].loc[Data['alpha']==alpha] = Q1
        Data['Q2 tot'].loc[Data['alpha']==alpha] = Q2
        Data['Q3 tot'].loc[Data['alpha']==alpha] = Q3
    print('Done')
    save_data(Data, 'D-alpha_summary')
    return Data



# ********** QVALUES DISTRIBUTIONS *************** #
from sklearn.neighbors import KernelDensity
from scipy.stats.kde import gaussian_kde
from scipy import interpolate
from scipy import ndimage

def convert_Q_to_unidimensional(Q):
    return Q.reshape(Q.shape[0]*Q.shape[1],)


def compute_individual_histograms(Models, Qopt=None, params=params, nbins=10):
    '''Generates the histograms of Q-value distributions, for each individual.
    If Qopt is provided, the histogram of Q values of the optimal policy is also computed.
    :return H_indiv: Dictionary storing the histograms of all individuals, for each replay type.
                    Keys : replay types, -1 for the optimal policy.
                    Values: Matrices of dimension n_individuals*nbins.'''
    H_empty = np.zeros((params['n_individuals'], nbins))
    H_indiv = dict(zip(params['replay_refs'], [H_empty.copy() for rep in params['replay_refs']]))
    for rep in params['replay_refs']:
        for i in range(params['n_individuals']):
            Q = Models['Q'][rep][i,:,:]
            data = convert_Q_to_unidimensional(Q)
            hist, bin_edges = np.histogram(data, bins=nbins)
            H_indiv[rep][i,:] = hist
    if Qopt is not None:
        data = Qopt.reshape(Qopt.shape[0]*Qopt.shape[1],)
        hist, bin_edges = np.histogram(data, bins=nbins)
        H_indiv[-1] = hist
    H_indiv['bins'] = bin_edges
    return H_indiv

def compute_population_histogram(H_indiv, params=params):
    nbins = H_indiv[0].shape[1]
    H_pop = dict(zip(params['replay_refs'], [{'Q1':np.zeros(nbins), 'Q2':np.zeros(nbins), 'Q3':np.zeros(nbins)} for rep in params['replay_refs']]))
    for rep in params['replay_refs']:
        h = H_indiv[rep]
        q1, q2, q3 = np.percentile(h, [25, 50, 75], axis=0)
        H_pop[rep]['Q1'] = q1.copy()
        H_pop[rep]['Q2'] = q2.copy()
        H_pop[rep]['Q3'] = q3.copy()
    if -1 in H_indiv.keys():
        H_pop[-1] = H_indiv[-1]
    H_pop['bins'] = H_indiv['bins']
    return H_pop

def compute_individual_distances(Models, Qopt=None, params=params):
    '''Similarity between Q-value distributions of different replay types, for each individual.
    Metric: Wasserstein_distance (Earth mover's Distance).
            Represents the minimum amount of 'work' required to transform one distribution into the other, 
            where 'work' is measured as the amount of distribution weight that must be moved, multiplied by the distance it has to be moved.
            scipy.stats.wasserstein_distance() takes as argument the values observed in the (empirical) distribution (array_like).
    Statistical test : Two-sample Kolmogorov-Smirnov test.
            Compares the underlying continuous distributions F(x) and G(x) of two independent samples.
            scipy.stats.ks_2samp() takes as agruments two sets of sample observations assumed to be drawn from a continuous distribution (array_like).
    :return EMD: Distance matrix with EMD metric. 
            EMD[j,k,i] = distance between the distributions of Q values 
            of replays of indices j and k in replay_refs, for individual i.
    :return KS: Matrix storing the statistics and pvalues of the KS statistical test.
            KS[j,k,0,i] = statistic of the test for Q-value distributions 
            of replays of indices j and k in replay_refs, for individual i.
            KS[j,k,1,i] = idem for pvlaue.'''
    n = len(params['replay_refs'])
    if Qopt is not None:
        n += 1
    EMD = np.zeros((n, n, params['n_individuals']))
    KS = np.zeros((n, n, 2, params['n_individuals']))
    for i in range(params['n_individuals']):
        for j, rep1 in enumerate(params['replay_refs']):
            Q1 = Models['Q'][rep1][i,:,:]
            data1 = convert_Q_to_unidimensional(Q1)
            for k, rep2 in enumerate(params['replay_refs']):
                Q2 = Models['Q'][rep2][i,:,:]
                data2 = convert_Q_to_unidimensional(Q2)
                EMD[j,k,i] = stats.wasserstein_distance(data1, data2)
                KS[j,k,0,i], KS[j,k,1,i] = stats.ks_2samp(data1, data2)
            if Qopt is not None:
                data_opt = convert_Q_to_unidimensional(Qopt)
                EMD[j,n-1,i] = stats.wasserstein_distance(data1, data_opt)
                EMD[n-1,j,i] = EMD[j,n-1,i]
                EMD[n-1,n-1,i] = stats.wasserstein_distance(data_opt, data_opt)
                KS[j,n-1,0,i], KS[j,n-1,1,i] = stats.ks_2samp(data1, data_opt)
                KS[n-1,j,0,i] = KS[j,n-1,0,i]
                KS[n-1,j,1,i] = KS[j,n-1,1,i]
                KS[n-1,n-1,0,i], KS[n-1,n-1,1,i] = stats.ks_2samp(data_opt, data_opt)
    return EMD, KS


def compute_population_distances(EMD, KS, params=params):
    '''Computes the median EMD in the population for all replay types.
    Computes the Wilcoxon signed-rank test, which tests the null hypothesis 
    that the sample come from the null distribution. 
    It is a non-parametric version of the paired T-test.'''
    EMD_pop = np.median(EMD, axis=2) # median EMD distance across individuals
    Stat_emd = np.zeros((EMD_pop.shape[0],EMD_pop.shape[1],2))
    for j in range(EMD_pop.shape[0]):
        for k in range(EMD_pop.shape[1]):
            try:
                Wvalue, pvalue = Stat_emd[j,k] = stats.wilcoxon(EMD[j,k,:])
                Stat_emd[j,k,0] = Wvalue
                Stat_emd[j,k,1] = pvalue
            except ValueError:
                Stat_emd[j,k,0] = 0
                Stat_emd[j,k,1] = 1
    return EMD_pop, Stat_emd
