#!/usr/bin/env python

import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from bioinfokit.analys import stat
import pandas as pd
import sys
import pickle

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


def display_convergence(Non_cv_det0, Non_cv_det1, Non_cv_sto0, Non_cv_sto1):
    print('Deterministic environment')
    print('Number of non-converging trials per type of replay:')
    print('Phase 0', Non_cv_det0)
    print('Phase 1', Non_cv_det1)
    print('---------------------------')
    print('Stochastic environment')
    print('Number of non-converging trials per type of replay:')
    print('Phase 0', Non_cv_sto0)
    print('Phase 1', Non_cv_sto1)
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

def split_before_after_cv(Data, CV_pop, params=params):
    '''Splits the data in two groups of trials : before and after convergence has occurred.
    Criterion : mean trial at which convergence occurs in average for all replay types and across all individuals.'''
    CV_means = [np.mean(CV_pop['Convergence trial'].loc[CV_pop['Replay type']==rep]) for rep in params['replay_refs']]
    t_boundary = int(max(CV_means))
    Perf_before = compute_performance_across_population(Data, tmax=t_boundary, params=params)
    Perf_after = compute_performance_across_population(Data, tmin=t_boundary, params=params)
    return Perf_before, Perf_after, t_boundary


# ********** SUMMARY STATISTICS FOR PERFORMANCE *************** #

def compute_performance_in_time(Data, params=params):
    '''Determines mean and STD of performance over the population along a simulation, for each replay type. 
    To be used for plotting learning curves.
    :param Data: dataframe, obtained from simulate_population()
    :return Perf: dataframe
        Rows: 1 row = 1 trial number
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
    Perf = pd.DataFrame({'Replay type':replays,
                        'Trial':trials,
                        'Mean':empty_data,
                        'STD':empty_data,
                        'Q1':empty_data,
                        'Q2':empty_data,
                        'Q3':empty_data})
    for rep in params['replay_refs']:
        for t in range(params['n_trials']):
            perf = Data['Performance'].loc[(Data['Replay type']==rep)&(Data['Trial']==t)].to_numpy()
            Perf['Mean'].loc[(Perf['Replay type']==rep)&(Perf['Trial']==t)] = np.mean(perf)
            Perf['STD'].loc[(Perf['Replay type']==rep)&(Perf['Trial']==t)] = np.std(perf)
            Q1, Q2, Q3 = np.percentile(perf, [25, 50, 75])
            Perf['Q1'].loc[(Perf['Replay type']==rep)&(Perf['Trial']==t)] = Q1
            Perf['Q2'].loc[(Perf['Replay type']==rep)&(Perf['Trial']==t)] = Q2
            Perf['Q3'].loc[(Perf['Replay type']==rep)&(Perf['Trial']==t)] = Q3
    return Perf

def compute_performance_across_population(Data, tmin=None, tmax=None, params=params):
    '''Determines mean and STD of performance over the population along a simulation, for each replay type. 
    To be used for plotting comparisons between groups.
    If tmin and tmax are given, only a part of the dataset can be used (trials before or after convergence).
    :param Data:
    :param tmin: (resp tmax) number of first (resp. last) trial from which gathering data
    '''
    if tmin is None:
        tmin = 0
    if tmax is None:
        tmax = params['n_trials']
    trials_idx = np.arange(tmin, tmax+1)
    D = Data.loc[(Data['Trial']>=tmin)&(Data['Trial']<tmax)]

    replays = [r for r in params['replay_refs'] for i in range(params['n_individuals'])]
    individuals = [i for r in params['replay_refs'] for i in range(params['n_individuals'])]
    empty_data = [0 for i in range(len(individuals))]
    Perf = pd.DataFrame({'Replay type':replays,
                        'Individual':individuals,
                        'Mean':empty_data,
                        'STD':empty_data})
    for rep in params['replay_refs']:
        for i in range(params['n_individuals']):
            perf = D['Performance'].loc[(D['Replay type']==rep)&(D['Individual']==i)]
            Perf['Mean'].loc[(Perf['Replay type']==rep)&(Perf['Individual']==i)] = np.mean(perf)
            Perf['STD'].loc[(Perf['Replay type']==rep)&(Perf['Individual']==i)] = np.std(perf)
    return Perf

def identify_representative(Data_pop, Perf_pop, params=params):
    '''Identifies the individual most representative of the mean behavior, for all replays.
    This is the individual whose performance (learning curve) minimizes the difference with the median.'''
    D = np.zeros(params['n_individuals'])
    for i in range(params['n_individuals']):
        data_i = Data_pop['Performance'].loc[Data_pop['Individual']==i].to_numpy()
        Q2 = Perf_pop['Q2'].to_numpy()
        D[i] = np.sum(np.abs(Q2-data_i))
    i_repr = np.argwhere(D==np.min(D))[0][0]
    return i_repr


# ********** STATISTICAL TESTS *************** #

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bioinfokit.analys import stat
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def test_ANOVA(data, var_name, params=params):
    '''The p value obtained from ANOVA analysis is significant (p < 0.05), and therefore, we conclude that there are significant differences among treatments.
    F value is inversely related to p value and higher F value (greater than F critical value) indicates a significant p value.
    Function f_oneway takes groups as input and returns ANOVA F and p value'''
    groups = [data[var_name].loc[data['Replay type']==rep].to_numpy() for rep in params['replay_refs']]
    fvalue, pvalue = stats.f_oneway(*groups)
    significant_05 = pvalue < 0.05
    significant_001 = pvalue < 0.001
    S = 'Not significant'
    if significant_05:
        S = 'p < 0.05'
    if significant_001:
        S = 'p < 0.001'
    print('ANOVA result : '+S+', F = {}'.format(fvalue))
    return fvalue, pvalue


def test_pairwise(data, var_name, thres=0.05, params=params, display=True):
    '''Perfoms a Tukey pairwise comparison test between all groups.
    :param thres: p-value threshold for significance, 0.05 or 0.001
    '''
    data = data.dropna() # remove None lines
    tukey = pairwise_tukeyhsd(endog=data[var_name], groups=data['Replay type'], alpha=thres)
    res = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    significant = res.loc[res['reject']==True]
    pairs = [(g1,g2,pval) for g1,g2,pval in zip(significant['group1'], significant['group2'], significant['p-adj'])]
    if display:
        print(tukey)
        print('Significant differences between replays (> = better performance)')
        for g1, g2, pval in pairs:
            dm = list(res['meandiff'].loc[(res['group1']==g1)&(res['group2']==g2)])[0]
            if dm < 0:
                sign = ' < '
            else:
                sign = ' > '
            print(params['replay_types'][g1]+sign+params['replay_types'][g2]+', p-value : {}'.format(pval))
    return res, pairs



# ********** OPTIMIZATION OF THE LEARNING RATE *************** #

def compare_alpha_replays(deterministic=True, params=params):
    '''Compare performance between replay types for each value of alpha.'''
    replays = [r for r in params['replay_refs'] for alpha in params['alpha_vals']]
    alpha_vec = [alpha for r in params['replay_refs'] for alpha in params['alpha_vals']]
    empty_data = [0 for obs in replays]
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
        if deterministic:
            data_raw = recover_data('Data_det_a{}'.format(alpha))
        else:
            data_raw = recover_data('Data_sto_a{}'.format(alpha))
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
    return Data

def optimize_alpha(params=params):
    '''Identify the alpha value for best performance.
    Criterion : the chosen alpha value is the one which obtains
    the minimal number of iterations to get to the reward,
    summing the deterministic and the stochastic case)'''
    alpha_vals = params['alpha_vals']
    empty_data = [0 for alpha in alpha_vals]
    Data = pd.DataFrame({'alpha':alpha_vals,
                        'Mean det':empty_data,
                        'STD det':empty_data,
                        'Q1 det':empty_data,
                        'Q2 det':empty_data,
                        'Q3 det':empty_data,
                        'Mean sto':empty_data,
                        'STD sto':empty_data,
                        'Q1 sto':empty_data,
                        'Q2 sto':empty_data,
                        'Q3 sto':empty_data,
                        'Mean tot':empty_data,
                        'STD tot':empty_data,
                        'Q1 tot':empty_data,
                        'Q2 tot':empty_data,
                        'Q3 tot':empty_data})
    print('Computing...')
    for alpha in alpha_vals:
        print('alpha', alpha)
        # Deterministic case
        Data_all = recover_data('Data_det_a{}'.format(alpha))
        Perf = compute_performance_across_population(Data_all)
        data_det = Perf['Mean'].to_numpy()
        Data['Mean det'].loc[Data['alpha']==alpha] = np.mean(data_det)
        Data['STD det'].loc[Data['alpha']==alpha] = np.std(data_det)
        Q1, Q2, Q3 = np.percentile(data_det, [25, 50, 75])
        Data['Q1 det'].loc[Data['alpha']==alpha] = Q1
        Data['Q2 det'].loc[Data['alpha']==alpha] = Q2
        Data['Q3 det'].loc[Data['alpha']==alpha] = Q3
        # Stochastic case
        Data_all = recover_data('Data_sto_a{}'.format(alpha))
        Perf = compute_performance_across_population(Data_all)
        data_sto = Perf['Mean'].to_numpy()
        Data['Mean sto'].loc[Data['alpha']==alpha] = np.mean(data_sto)
        Data['STD sto'].loc[Data['alpha']==alpha] = np.std(data_sto)
        Q1, Q2, Q3 = np.percentile(data_sto, [25, 50, 75])
        Data['Q1 sto'].loc[Data['alpha']==alpha] = Q1
        Data['Q2 sto'].loc[Data['alpha']==alpha] = Q2
        Data['Q3 sto'].loc[Data['alpha']==alpha] = Q3
        # Total
        data_tot = np.concatenate((data_det,data_sto))
        Data['Mean tot'].loc[Data['alpha']==alpha] = np.mean(data_tot)
        Data['STD tot'].loc[Data['alpha']==alpha] = np.std(data_tot)
        Q1, Q2, Q3 = np.percentile(data_tot, [25, 50, 75])
        Data['Q1 tot'].loc[Data['alpha']==alpha] = Q1
        Data['Q2 tot'].loc[Data['alpha']==alpha] = Q2
        Data['Q3 tot'].loc[Data['alpha']==alpha] = Q3
    print('Done')
    return Data