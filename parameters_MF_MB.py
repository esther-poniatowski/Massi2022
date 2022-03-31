#!/usr/bin/env python

import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt

check_reload = False # signal successful importation at the end of the file


# ********************** SIMULATION PARAMETERS ************************ #

# Learning parameters
alpha = 0.6 # learning rate (previous : 0.56)
alpha_vals = np.round(0.1*np.arange(0,11), decimals=1) # test values for optimizing this parameter
beta = 15.0
beta_vals = {'learning':beta, 'generalization':10.0, 'fast_test':beta} # temperature for softmax
gamma = 0.9 # temporal discount factor

# Simulation parameters
n_individuals = 100 # number of individuals
n_individuals_sims = {'learning':n_individuals, 'generalization':n_individuals, 'fast_test':10} 
n_trials = 50 # number of trials
n_trials_sims = {'learning': n_trials, 'generalization':6, 'fast_test':10} 
trial_change = 25 # number of the trial on which the initial state is changed
# rewarded state is changed at trial 25 : 25 + 25 = 50 for both learning phases
s_start = 35 # initial state
starting_points = {'learning':s_start, 'generalization': [21, 34, 4], 'fast_test':s_start}
s_rw = 22 # initial reward state
reward_states = [s_rw, 4]

# Parameters for replays (general)
## All replay types with their labels
replay_types = {0: 'No replay', 
                1: 'Backward replay', 
                2: 'Random replay', 
                3: 'Most diverse replay',
                4: 'Prioritized sweeping'}
replay_refs = list(replay_types.keys()) # indices of the replays
n_types_replays = len(replay_types) - 1 # exclude "no replay"
RR = 20  # number of times each sequence is replayed
RSS = 90  # replayed sequences size = length of replayed sequences
# Parameters for most diverse sequence replays
RSS_d = 15  # length of replayed sequences
n_seq_d =  int(RSS/RSS_d) # nb of sequences which can be stored in memory, such that the memory size is equal with other replays
epsilon_d = 0.5 # minimal distance for a new sequence to be added in memory
# Parameters for prioritized sweeping
epsilon_p = 0.001 # threshold above which a cumulated change in Q-values requires another cycle of replay

# Parameters for convergence criterion
perc = 0.7 # boundary
window = 5 # window


# **************** STATES COORDINATES AND TRANSITION MATRIX ******************* #

def split_function(string, sep):  
    '''Splits a string in its parts.
    :param string: character string to be splitted
    :param sep: separator character indicating the splitting points
    :return l: list containing the elements of the splitted string.
    '''
    n = len(string)
    l = []
    i = 0
    while i < n:
        new_seq = ''
        while i < n and string[i] != sep:
            new_seq += string[i]
            i = i + 1
        l.append(new_seq)
        i = i + 1
    return l


def second_largest(numbers):
    '''Return the second largest element in a list
    :param numbers: list
    :return: list element
    '''
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None

def reading_transitions(max_len_tMatrix, nA, PATH="transitions.txt"):
    '''Imports the transition matrix M from the transition.txt file. 
    first dimension : s0 (all states)
    second dimension : 8 actions
    third dimension : s1 states (list of length 36)
    M[s0, a, s1] contains the probaility to end up in s1 when starting from s0 and taking action a
    :param max_len_tMatrix: possible max numebr of states (int)
    :param nA: number of actions (int)
    :param PATH: path of the transitions file (string)
    :return: Transition Matrix (numpy array)
    '''
    if os.path.exists(PATH):
        f = open(PATH, "r")
        lines = f.readlines()
        n = len(lines)
        Trans_Gazebo = np.zeros((max_len_tMatrix, nA, max_len_tMatrix))
        for k in range(n - 1):
            l = lines[k]
            l_split = split_function(l, ',')
            l_split[-1] = split_function(l_split[-1], '\n')[0]
            s0 = int(l_split[0])
            a = int(l_split[1])
            s1 = int(l_split[2])
            m = float(l_split[3])
            Trans_Gazebo[s0, a, s1] = m
    else:
        print("error: path not found")
    return Trans_Gazebo


def define_forbidden_state_action(T, len_T, nA):
    '''Defines the transitions matrix and output it as a .txt file.
    :param T: transition matrix from Gazebo (numpy array)
    :param len_T: len of T (int)
    :param nA: number of actions (int)
    :return: forbidden_state_action (numpy array, dimension s*a with 1 if dead-end and 0 otherwise)
    '''
    forbidden_state_action = np.zeros((len_T, nA))  # dead-end states/actions
    for s in range(len_T):
        for a in range(nA):
            if np.sum(T[s, a, :]) == 0: # 0 probability to reach any state from s by taking action a
                forbidden_state_action[s, a] = 1
    return forbidden_state_action

def define_transitions_matrix(Trans_Gazebo, len_T, nA, deterministic=True):
    '''Defines the transitions matrix and output it as a .txt file.
    For a deterministic transition matrix : keeping just new states with the greatest probability, when starting from s0 and taking action a (p=1, the others p=0)
    For a stochastic transition matrix : normalize the T matrix to get transition probabilities.
    :param Trans_Gazebo: transition matrix from Gazebo (numpy array)
    :param len_T: number of states desired (int)
    :param nA: number of actions (int)
    :param deterministic: bool, True for a deterministic environment, False for a stochastic one, as computed in Gazebo
    :return: T (numpy array)
    '''
    T = Trans_Gazebo.copy()
    if deterministic: # deterministic matrix
        filename = 'transitions_deterministic.txt'
        for s in range(len_T): 
            for a in range(nA):
                somme = np.sum(Trans_Gazebo[s, a, :])
                if somme != 0:
                    if second_largest(Trans_Gazebo[s, a, :]) == max(Trans_Gazebo[s, a, :]):
                        idx_chosen_max = int(np.where(Trans_Gazebo[s, a, :] == second_largest(Trans_Gazebo[s, a, :]))[0][1])
                        T[s, a, idx_chosen_max] = 1
                        for idx_poss_s1 in range(0, len_T):
                            if idx_poss_s1 != idx_chosen_max:
                                T[s, a, idx_poss_s1] = 0
                    else:
                        idx_a_max = np.argmax(Trans_Gazebo[s, a, :])
                        T[s, a, idx_a_max] = 1
                        for idx_poss_s1 in range(0, len_T):
                            if idx_poss_s1 != idx_a_max:
                                T[s, a, idx_poss_s1] = 0
    else: # stochastic matrix
        filename = 'transitions_stochastic.txt'
        for s in range(len_T):
            for a in range(nA):
                somme = np.sum(Trans_Gazebo[s, a, :])
                if somme != 0:
                    T[s, a, :] = Trans_Gazebo[s, a, :] / somme
    with open(filename, 'w') as f:
        for s in range(len_T):
            for a in range(nA):
                for pa in range(len(T[s, a, :])):
                    f.write(str(s) + ',' + str(a) + ',' + str(pa) + ', ' + str(T[s, a, pa]))
                    f.write('\n')
    return T

# Continuous coordinates of all the states discovered by the robot
state_coords = [(0.00320002,0.0059351),(0.310727,0.0241474), (0.593997,0.152759),(0.724483, -0.118826),(0.956215, 0.0761724),
         (-0.0546479, 0.308712),(-0.359653, 0.3537),(-0.465333, 0.651306),(-0.636187 ,0.920479),(-0.325808 ,0.96138),
         (-0.0262775, 0.936864),(-0.12853 ,0.608485),(0.275655, 0.947168),(0.164439 ,1.22664),(-0.588397, 1.21822),
         (-0.779392, 0.652651),(-0.669064, 0.253532),(-0.863761 ,0.0191329),(-0.924563, -0.277856),(-1.08771, 0.22059),
         (-1.23553, 0.490936),(-1.09893 ,0.758705),(-0.293732, 1.30145),(0.208853, 0.454286),(0.522142, 0.755471),
         (0.69987, 0.513679),(-0.0588719, -0.289781),( 0.0889117 ,-0.583929),(-0.120961, -0.804133),(-0.939397, 1.014),
         ( 0.367168, -0.274708),( -0.329299 ,-0.156936),(0.39444, -0.660872),(-0.539525, -0.381784),
         (-1.22956, -0.263396),(-0.504464 ,-0.834182)]

nA = 8 # number of actions (N, S, E, W, NE, NO, NE, NO)
len_T = 36 # number of the common discovered states among the exploration in Gazebo -> sets the number of states in the model
# Reading the real transition matrix from Gazebo
max_len_tMatrix = 60  # max possible number of discovered states (in the Gazebo exploration they are max 43s)
Trans_Gazebo = reading_transitions(max_len_tMatrix, nA)
# Crop the M matrix to the number of "common-to-all-states"
Trans_Gazebo = Trans_Gazebo[0:len_T, :, 0:len_T] 
# Define possible actions 
forbidden_state_action = define_forbidden_state_action(Trans_Gazebo, len_T, nA)
# Define the transitions matrix (deterministic is a boolean to set the properties of the environment)
T_det = define_transitions_matrix(Trans_Gazebo, len_T, nA, deterministic=True) 
T_stoch = define_transitions_matrix(Trans_Gazebo, len_T, nA, deterministic=False) 



# ********************** PARAMETER DICTIONARY ************************ #

# Organise all parameters in a dictionary
params = {'n_individuals': n_individuals, 'n_trials': n_trials,
            'n_individuals_sims':n_individuals_sims, 'n_trials_sims':n_trials_sims,
            'alpha': alpha ,'beta': beta, 'gamma': gamma,
            'alpha_vals':alpha_vals, 'beta_vals':beta_vals, 
            's_start': s_start, 's_rw': s_rw, 
            'starting_points':starting_points, 'reward_states':reward_states, 'trial_change':trial_change,
            'RSS': RSS, 'RR': RR, 'n_seq_d': n_seq_d, 'RSS_d': RSS_d, 
            'epsilon_d': epsilon_d, 'epsilon_p': epsilon_p,
            'perc': perc, 'window': window,
            'n_types_replays': n_types_replays, 'replay_types':replay_types, 'replay_refs':replay_refs,
            'nA': nA, 'nS':len_T, 'state_coords': state_coords, 'forbidden_state_action': forbidden_state_action, 
            'T_det': T_det, 'T_stoch': T_stoch, 'T': T_det}
            # transition matrix deterministic by default, to be set at the beginning of simulation

# Select parameters according to the goal (deterministic vs stochastic, full simulation vs. fast test, types of replays tested)

def set_environment(params, deterministic):
    '''Replaces the entry params['T'] by the stochastic of the deterministic matrix.'''
    if deterministic:
        params['T'] = params['T_det']
        print('Transition matrix set to deterministic in params.')
    else:
        params['T'] = params['T_stoch']
        print('Transition matrix set to stochastic in params.')
    return params

def set_replays(params, replay_refs=replay_refs):
    params['replay_refs'] = replay_refs
    params['n_types_replays'] = len(replay_refs) - 1 # exclude "no replay"
    return params 

def set_simulation(params, sim, gen_test=0, replay_refs=replay_refs):
    '''Updates t=parameter in the dictionary of parmeters.
    :param sim: label of the simulation (str), 'learning'/'generalization'/'fast_test'
    :param gen_test: number of the generalization test (int), 0/1/2
    :param replay_refs: types of replays to be performed
    '''
    params['n_individuals'] = params['n_individuals_sims'][sim]
    params['n_trials'] = params['n_trials_sims'][sim]
    params['beta'] = beta_vals[sim]
    if sim == 'generalization':
        params['s_start'] = params['starting_points']['generalization'][gen_test] # chose the starting point depending on the generalization test
    else:
        params['s_start'] = params['starting_points']['learning'] # reference starting point
    return params


