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
import copy

# import functions and parameters
import parameters_MF_MB as PRMS
params = PRMS.params
from functions_MF_MB import trial_Q_learning, trial_exploration, trial_replays

check_reload = False # signal successful importation at the end of the file


# ******************* AUXILIARY FUNCtiONS ************************ #

def save_data(Data, file_name, df=True):
    if df:
        Data.to_csv('Data/'+file_name)
    else:
        with open('Data/'+file_name+'.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def recover_data(file_name, df=True):
    if df:
        Data = pd.read_csv('Data/'+file_name)
    else:
        with open('Data/'+file_name+'.pickle', 'rb') as handle:
            Data = pickle.load(handle)
    return Data



# ******************* SIMULATING INDIVIDUALS - LEARNING AND REPLAYS ************************ #

def set_seed(i=0):
    '''Initialize random seeds at the beginning of each simulation.'''
    np.random.seed(i)
    random.seed(i)

def simulate_individual(replay_type, params=params, i_indiv=0, Q=None, hatP=None, hatR=None, N=None, M_buffer=None, D_seq=None, Delta=None, save_memory=False, convergence=True, method_r='shuffle', method_p='predecessor', saved_trials=[]):
    '''Run a full simulation for a given type of replay, over n_trials.
    The seed can be set according to the individual identifyer, so that one individual produces the same trajectory on the first trial for each type of replay.
    :param replay_type: reference of the type of replay performed by the agent (int, see params['replay_types'])
    :param i_indiv: index referencing the individual
    :param Q: (idem for hatP, hatR, N) (MB agent) for generalization, previously stored model of the world
    :param save_memory: if True, the function will return a dictionary of detailed data to track the agent's behavior, if False it will only return minimal data
    :return Q0: Q-value matrix obtained at the end of learning (first phase) (numpy array dimension nS*na)
    :return hatP0: (MB agent) estimated transition matrix obtained at the end of learning (array nS*nA*nS)
    :return hatR0: (MB agent) estimated reward function obtained at the end of learning (array nS*nA)
    :return N0: (MB agent) numbers of each transition performed (array nS*nA*nS)
    :return Performance: number of actions taken by the individual in each trial to reach the reward
    :return Data: dictionary containing details about the agent's behavior in each trial (separately for exploration phases and replay phases)
        H_explo: history of *experienced* transitions (s0,a,s1,r) performed in each exploration trial (list of 'ntrials' lists of tuples)
        H_repl: history of *replayed* transitions (s0,a,s1,r) performed after each trial
        H_Qex: history of Q matrices obtained after each exploration trial (list of 'n_trials' matrices)
        H_Qr: idem after each replay phase
        H_dQ: history of the full Q matrices updates resulting from replays (Q_after - Q_before) (list of 'n_trials' matrices)
    '''
    # 1) Initialize empty data : storage variables for the simulation
    H_explo = [] # history of transitions performed during exploration
    H_repl = [] # history of replayed transitions
    H_Qex = [] # history of Q matrices after each exploration
    H_Qr = [] # history of Q matrices after replays
    H_dQ = [] # history of Q updates during the replay phase (full matrix, Qrepl - Q_explo)
    H_Q_trials = {} # Q matrices on the selected trials
    Performance = [] # number of actions taken in each trial
    Q0, hatP0, hatR0, N0 = None, None, None, None # avoid errors if n_trials < trial_change
    # 2) Simulation
    params['s_rw'] = params['reward_states'][0] # reset the initial rewarded state
    set_seed(i_indiv) # set seed for this individual
    for trial in range(params['n_trials']):
        # print('Trial',trial)
        if trial == params['trial_change']:
            params['s_rw'] = params['reward_states'][1] # change the rewarded state
            Q0, hatP0, hatR0, N0 = Q.copy(), hatP.copy(), hatR.copy(), N.copy() # save the current state of the model
        if trial in saved_trials:
            H_Q_trials[trial] = Q.copy()
        H, Q, hatP, hatR, N, M_buffer, D_seq, Delta, h_explo, h_repl, Q_explo, Q_repl = trial_Q_learning(replay_type, params, Q=Q, hatP=hatP, hatR=hatR, N=N, M_buffer=M_buffer, D_seq=D_seq, Delta=Delta, convergence=convergence, method_r=method_r, method_p=method_p)
        Performance.append(len(h_explo)) # save the number of actions taken in this trial
        if save_memory:
            H_explo.append(h_explo)
            H_repl.append(h_repl)
            H_Qex.append(Q_explo)
            H_Qr.append(Q_repl)
            H_dQ.append(Q_repl - Q_explo)
            Data = {'H_explo': H_explo,
                    'H_repl': H_repl,
                    'H_Qex': H_Qex,
                    'H_Qr': H_Qr,
                    'H_dQ': H_dQ,
                    'Performance': Performance}
        # if trial == 24:
        #     Q24 = Q.copy()
        # if trial in [25]:
        #     print('Trial', trial)
        #     print(len(h_explo), 'visits')
        #     print(h_explo)
        #     print('--------')
        #     print(len(h_repl), 'replays')
        #     print(h_repl)
        #     plt.imshow(Q_repl)
        #     plt.colorbar()
            plt.show()
    if save_memory:
        return Q0, hatP0, hatR0, N0, Data
    if len(saved_trials) != 0:
        return Q0, hatP0, hatR0, N0, H_Q_trials
    else:
        return Q0, hatP0, hatR0, N0, Performance

def extract_data(Data):
    H_explo = Data['H_explo']
    H_repl = Data['H_repl']
    H_Qex = Data['H_Qex']
    H_Qr = Data['H_Qr']
    H_dQ = Data['H_dQ']
    Performance = Data['Performance']
    return H_explo, H_repl, H_Qex, H_Qr, H_dQ, Performance

def get_individual_data_per_trial(i_indiv=0, params=params, Models0=None, convergence=True, method_r='shuffle', method_p='predecessor'):
    '''Get full data in exploration trials and replay phases for one *single* individual.
    Stores the full data, to compare replay strategies.
    :param i_indiv: Identifyer of the individual, to set the seed.
    :return Data_trials: List containing the behavior of the agent on each trial, for each replay type.
        Each element is a dictionary corresonding to one trial.
        Keys: Different data structures collected on one trial.
            h_explo: history of transitions over the exploration trial
            h_repl: history of replayed transitions after the trial
            Q_explo: Q matrix at the end of the trial
            Q_repl: Q matrix after the replay
            Q_upd: update of the Q matrix due to both exploration and replay.
        Values: Dictionary containing the data over the trial for each replay type.
            Keys: replay types.
            Values: data.
    :return Performance: Dictionary containing the performance of the individual (number of actions taken on each trial) for each replay type.
    :return Models: Dictionary containing the matrices describing the individual's learnt model of the world at the end of the simulation, for each replay type.
    '''
    # Initialize empty data to collect trajectories and Q matrices
    keys = ['h_explo', 'h_repl', 'Q_explo', 'Q_repl', 'Q_upd']
    Data_trials = [dict(zip(keys, [{} for k in keys])) for t in range(params['n_trials'])]
    Models = dict(zip(params['replay_refs'], [[] for rep in params['replay_refs']]))
    # Initialize empty data to collect the individual learning curves for different replays
    # in accordance with data structures storing the population performances
    replays = [r for r in params['replay_refs'] for t in range(params['n_trials'])]
    trials = [t for r in params['replay_refs'] for t in range(params['n_trials'])]
    empty_data = [0 for i in range(len(trials))]
    LC = pd.DataFrame({'Replay type':replays,
                        'Trial':trials,
                        'Performance':empty_data})    
    params0 = copy.deepcopy(params)
    for rep in params['replay_refs']: # for each type of replay
        if Models0 is None: # for learning
            Q, hatP, hatR, N, data = simulate_individual(replay_type=rep, params=params, i_indiv=i_indiv, 
                                                        convergence=convergence, method_r=method_r, method_p=method_p,
                                                        save_memory=True)
        else: # for generalization
            # change starting point and number of trials
            params0['s_start'] = params['starting_points']['generalization'][0]
            params0['n_trials'] = params['n_trials_sims']['generalization']
            # set Q0, hatP0, hatR0, N0 with the stored Models of the considered replay type
            Q0, hatP0, hatR0, N0 = Models0[rep]['Q'], Models0[rep]['hatP'], Models0[rep]['hatR'], Models0[rep]['N']
            # use those matrices and set replay_type=0 to prevent replays during generalization
            Q, hatP, hatR, N, data = simulate_individual(replay_type=0, params=params0, i_indiv=i_indiv, 
                                                        Q=Q0, hatP=hatP0, hatR=hatR0, N=N0,
                                                        save_memory=True)
        # save models (matrices)
        Models[rep] = {'Q':Q, 'hatP':hatP, 'hatR':hatR, 'N':N}
        # store trajectories and matrices according to the trial number
        H_explo, H_repl, H_Qex, H_Qr, H_dQ, perf = extract_data(data)
        for t in range(params0['n_trials']):
            Data_trials[t]['h_explo'][rep] = H_explo[t].copy()
            Data_trials[t]['h_repl'][rep] = H_repl[t].copy()
            Data_trials[t]['Q_explo'][rep] = H_Qex[t].copy()
            Data_trials[t]['Q_repl'][rep] = H_Qr[t].copy()
            Data_trials[t]['Q_upd'][rep] = H_dQ[t].copy()
        # save the learning curves
        LC['Performance'].loc[LC['Replay type']==rep] = perf.copy()
    return Data_trials, LC, Models


def collect_Q_matrices(params=params):
    Q_indiv = np.zeros((params['n_individuals'], params['nS'], params['nA']))
    Data_Q = dict(zip(params['replay_refs'], [Q_indiv.copy() for rep in params['replay_refs']]))
    for rep in params['replay_refs']:
        for i_indiv in range(params['n_individuals']):
            set_seed(i_indiv) # set seed for this individual
            Q, hatP, hatR, N, M_buffer, D_seq, Delta = None, None, None, None, None, None, None
            for trial in range(params['n_trials']):
                H, Q, hatP, hatR, N, M_buffer, D_seq, Delta, h_explo, h_repl, Q_explo, Q_repl = trial_Q_learning(replay_type=rep, params=params, Q=Q, hatP=hatP, hatR=hatR, N=N, M_buffer=M_buffer, D_seq=D_seq, Delta=Delta)
            Data_Q[rep][i_indiv,:,:] = Q.copy()
    return Data_Q



# ******************* SIMULATING POPULATION - LEARNING AND REPLAYS ************************ #

def initialize_empty_data(params=params):
    replays = [r for r in params['replay_refs'] for i in range(params['n_individuals']) for t in range(params['n_trials'])]
    individuals = [i for r in params['replay_refs'] for i in range(params['n_individuals']) for t in range(params['n_trials'])]
    trials = [t for r in params['replay_refs'] for i in range(params['n_individuals']) for t in range(params['n_trials'])]
    n_actions = [0 for obs in range(len(trials))]
    Data_pop = pd.DataFrame({'Replay type':replays,
                            'Individual':individuals,
                            'Trial':trials,
                            'Performance':n_actions})
    return Data_pop

def simulate_population(params=params, convergence=True, method_r='shuffle', method_p='predecessor'):
    '''Performs Q-learning phase for all replays. 
    Each replay is tested on n_individuals, each one performing n_trials.
    :return Data_pop: dataframe containing summary data
        Rows: 1 row = 1 trial (for 1 individual and 1 replay type)
        Columns:
            Replay type: type of replay for this trial
            Individual: individual performing this trial
            Trial: position of the trial along the learning phase of the individual
            Performance: number of actions taken in the trial before reaching the reward
    :return Model_pop: dictionary storing matrices of each individual obtained at the end of learning (phase 0)
        Keys: Different types of matrices making the agent model.
            Q: Q matrix at the end of the simulation (first learning phase, before change of reward location)
            hatP, hatR, hatN: idem than Q
        Values: Dictionary storing the matrices of all individuals for all replays.
            Keys: replay type
            Values: Q matrices of the 'n_individuals' tested on this replay type (array of dimension n_individuals*dim of the corresponding matrix).
    '''
    # Initialization of empty data
    Data_pop = initialize_empty_data(params)
    # For generalization 
    # keys: replays
    # values: Q matrices (resp P, R, N) of all individuals
    empty_Q = np.zeros((params['n_individuals'], params['nS'], params['nA']))
    empty_P = 0*np.ones((params['n_individuals'], params['nS'], params['nA'], params['nS']))
    empty_R = np.zeros((params['n_individuals'], params['nS'], params['nA']))
    empty_N =  np.zeros((params['n_individuals'], params['nS'], params['nA']))
    Q_pop = dict(zip(params['replay_refs'], [empty_Q.copy() for r in params['replay_refs']])) 
    P_pop = dict(zip(params['replay_refs'], [empty_P.copy() for r in params['replay_refs']]))
    R_pop = dict(zip(params['replay_refs'], [empty_R.copy() for r in params['replay_refs']]))
    N_pop = dict(zip(params['replay_refs'], [empty_N.copy() for r in params['replay_refs']]))
    print('LEARNING...')
    print('Replay | Individual')
    for rep in params['replay_refs']: # for each type of replay
        print('--------------------')
        for i in range(params['n_individuals']): # for each individual
            print(rep,'|',i)
            Q, hatP, hatR, N, Performance = simulate_individual(replay_type=rep, i_indiv=i, params=params, convergence=convergence, method_r=method_r, method_p=method_p)
            Data_pop['Performance'].loc[(Data_pop['Replay type']==rep)&(Data_pop['Individual']==i)] = np.array(Performance)
            Q_pop[rep][i,:,:] = Q.copy()
            P_pop[rep][i,:,:,:] = hatP.copy()
            R_pop[rep][i,:,:] = hatR.copy()
            N_pop[rep][i,:,:] = N.copy()
    print('LEARNING finished')
    Model_pop = {'Q':Q_pop, 'P':P_pop, 'R':R_pop, 'N':N_pop}
    return Data_pop, Model_pop


def simulate_generalization(params, Model_pop):
    ''' For generalization, please first execute set_simulation(params, 'generalization', gen_test)
    so that setting several parameters, including :
    - params['s_start'] = gen_test
    - setting params['replay_refs'] = [0] so that no replays are performed.
    '''
    Data_pop = initialize_empty_data(params)
    print('GENERALIZATION...')
    print('Replay | Individual')
    for rep in params['replay_refs']: # for each type of replay
        print('--------------------')
        # Extract the saved models for this type of replay
        Q_pop = Model_pop['Q'][rep]
        P_pop = Model_pop['P'][rep]
        R_pop = Model_pop['R'][rep]
        N_pop = Model_pop['N'][rep]
        for i in range(params['n_individuals']): # for each individual
            print(rep,'|',i)
            # Extract the model of this individual, to be given as arguments in simulate_individual()
            Q, hatP, hatR, N = Q_pop[i,:,:], P_pop[i,:,:,:], R_pop[i,:,:], N_pop[i,:,:]
            # Set replay_type to 0 in simulate_individual(), to prevent replay during generalization
            Q, hatP, hatR, N, Performance = simulate_individual(replay_type=0, params=params, i_indiv=i, Q=Q, hatP=hatP, hatR=hatR, N=N)
            Data_pop['Performance'].loc[(Data_pop['Replay type']==rep)&(Data_pop['Individual']==i)] = np.array(Performance)
    print('Done')
    return Data_pop



# ******************* SIMULATIONS - TESTiNG ALPHA PARAMETER ************************ #

def test_alpha(alpha_vals=np.linspace(0, 1, 10), params=params, convergence=True, method_r='shuffle', method_p='predecessor'):
    '''Generate data for testing the learning performance with different values of alpha'''
    replays = [r for r in params['replay_refs'] for alpha in alpha_vals for i in range(params['n_individuals'])]
    alpha_vec = [alpha for r in params['replay_refs'] for alpha in alpha_vals for i in range(params['n_individuals'])]
    individuals = [i for r in params['replay_refs'] for alpha in alpha_vals for i in range(params['n_individuals'])]
    n_actions = [0 for obs in range(len(individuals))]
    Data = pd.DataFrame({'Replay type':replays,
                        'alpha':alpha_vec,
                        'Individual':individuals,
                        'Performance':n_actions})
    print('TEST ALPHA...')
    print('Replay | alpha | Individual')
    for rep in params['replay_refs']: # for each type of replay
        print('--------------------')
        for alpha in alpha_vals:
            params['alpha'] = alpha # update the parameter in the dictionary
            for i in range(params['n_individuals']): # for each individual
                print(rep,'|', alpha, '|', i)
                Q, hatP, hatR, N, Performance = simulate_individual(replay_type=rep, i_indiv=i, params=params, convergence=convergence, method_r=method_r, method_p=method_p)
                m = np.mean(Performance) # mean number of actions across all trials, for this individual
                Data['Performance'].loc[(Data['Replay type']==rep)&(Data['alpha']==alpha)&(Data['Individual']==i)] = m
    print('TEST finished')
    return Data




# ******************************************************************************
if check_reload:
    print('Import done')


