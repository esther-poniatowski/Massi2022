#!/usr/bin/env python

import numpy as np
from os import path
import random
import bisect 
import matplotlib.pyplot as plt


# ********** BEHAVING AND LEARNING *************** #

def decision_making(s0, Q, params):
    '''Draw a random action, by applying the softmax function to all Qvalues.
    Actions are taken only for licit ones from state s0.'''
    A = [a for a in range(params['nA'])]
    forbidden = params['forbidden_state_action'][s0,:] # list of actions possibilities from s0
    A_probas = np.array([np.exp(params['beta']*Q[s0][a]) if forbidden[a]!=1 else 0 for a in A])
    A_probas /= np.sum(A_probas)
    a_chosen = np.random.choice(A, p=A_probas)
    return a_chosen

def consequence_env(s0, a, params):
    '''Given current state and action taken :
    - go to a new state according to the transition matrix,
    - get a reward according to the reward function.'''
    states = [s for s in range(params['nS'])]
    s1 = np.random.choice(states, p=params['T'][s0,a,:])
    r = int(s1 == params['s_rw'])
    return s1, r

def Temporal_Difference_Learning(Q, s0, s1, a, r, params):
    '''For a model-free agent.
    Computes the new value to update the Q matrix according to the temporal difference learning rule.
    '''
    m = max(Q[s1,:]) # maximum value accessible from the new state
    Qnew = Q[s0][a] + params['alpha']*(r + params['gamma']*m - Q[s0][a])
    return Qnew

def Value_Iteration(Q, hatP, hatR, s0, a, params):
    '''For a model-based agent (implementing prioritized-sweeping).
    Computes the new value to update the Q matrix for the pair (current state, action), with the Value Iteration algorithm. 
    New value = reward obtained for the pair (s0, a) + expected value due to all states accessible from (s0, a), weigthed by the probability to reach them.
    '''
    Qmax = Q.max(axis=1) # best accessible value from each state (vector of length nS)
    trans_probas = hatP[s0,a,:].reshape(params['nS']) # probability to reach each new state by doing action a
    Qnew = hatR[s0,a] + params['gamma']*np.sum(trans_probas*Qmax) # updated value for pair (s0, action)
    return Qnew

def MB_learning(s0, s1, a, r, N, hatP, hatR, params):
    '''Updates *in place* the agent's model of the environment (not its Q matrix).
    :param N: number of visits for the current state-action pair, matrix (nS, nA)
    :param hatP: estimated transition matrix, matrix (nS, nA, nS)
    :param hatR : estimated reward function, matrix (nS, nA)
    '''
    N[s0,a] += 1
    hatP[s0,a,:] = (1 - 1/N[s0,a])*hatP[s0,a,:] + 1/N[s0,a]*(np.arange(params['nS'])==s1).astype(int)
    # hatR[s0, a] = (1 - 1/hatR[s0, a])*hatR[s0, a] + r/N[s0, a]
    hatR[s0,a] = r # keep in mind only the last reward (deterministic)
    return N, hatP, hatR

def policy(Q, params):
    '''Computes the policy pi(a|s).
    :return pi: probabilities to chose an action, in a given state (matrix nS*nA).'''
    pi = np.exp(params['beta']*Q)
    for s in range(params['nS']):
        pi[s,:] /= np.sum(pi[s,:]) # normalization over actions for each state
    return pi

def V_from_Q(Q, params):
    '''Computes the value of each state according to Bellman equation :
    V(s) = sum(pi(a|s)*Q(s,a)) (sum over actions)
    :return V: Value of each state (vector of length nS)'''
    pi = policy(Q, params) # dimension nS*nA, similar to Q
    V =  (pi*Q).sum(axis=1)
    return V


# ********** REPLAY 1 - BACKWARD REPLAY *************** #

def replay_b(M_buffer, Q, params, convergence=True):
    '''Replay type : Backward sequences.
    M_buffer is already ordered in backward order.
    :param M_buffer: memory buffer, storing all the transitions (s0,a,s1,r) to be replayed
    :param Q: Q-value matrix (numpy array)
    :return Q: Updated Q-value matrix (numpy array)
    '''
    for repet in range(params['RR']): # repeating the sequence a number of times
        for (s0, a, s1, r) in M_buffer: 
            Qnew = Temporal_Difference_Learning(Q, s0, s1, a, r, params)
            Q[s0,a] = Qnew
    return Q

def replay_b(M_buffer, Q, params, convergence=True, RRmax=1000):
    '''Replay type : Backward sequences.
    M_buffer is already ordered in backward order.
    :param M_buffer: memory buffer, storing all the transitions (s0,a,s1,r) to be replayed
    :param Q: Q-value matrix (numpy array)
    :return Q: Updated Q-value matrix (numpy array)
    '''
    if convergence:
        RR = RRmax
        DQ = [[] for repet in range(RR)]
        for repet in range(RR):  # repeating the sequence a number of times
            if repet == RR - 1:
                print('More than {} replays repetitions needed in MF algorithm.'.format(RRmax))
                # print(repet, max(DQ[repet-1]), DQ[repet-1])
            if repet == 0 or max(DQ[repet-1]) > params['epsilon_p']:
                for (s0, a, s1, r) in M_buffer:
                    Qnew = Temporal_Difference_Learning(Q, s0, s1, a, r, params)
                    dQ = Qnew - Q[s0, a]
                    Q[s0, a] = Qnew
                    DQ[repet].append(abs(dQ))
            else:
                break
    else:
        RR = params['RR']
        for repet in range(RR):  # repeating the sequence a number of times
            for (s0, a, s1, r) in M_buffer:
                Qnew = Temporal_Difference_Learning(Q, s0, s1, a, r, params)
                Q[s0, a] = Qnew
    return Q

def update_memory_b(M_buffer, H, params):
    '''Update the memory buffer at the end of a trial.
    M_buffer contains the last RSS transitions (s0,a,s1,r), which can run on the previous trials.
    :param H: History of transitions since the beginning of the simulation (all trials).'''
    HR = H.copy()
    HR.reverse()
    M_buffer = HR + M_buffer
    if len(M_buffer) > params['RSS']:
        M_buffer = M_buffer[:params['RSS']]
    return M_buffer


# ********** REPLAY 2 - RANDOM REPLAY *************** #

replay_r = replay_b # identical replay method 

def update_memory_r(M_buffer, H, params, method_r='shuffle'):
    '''Update the memory buffer at the end of a trial.
    Similarly to backward replays, M_buffer contains the last RSS transitions (s0,a,s1,r), which can run on the previous trials.
    For random replays:
    - either transition orders are shuffled
    - or transitions are drawn from the last history
    :param H: History of transitions since the beginning of the simulation (all trials).'''
    # Retain the last transitions from experience, as for backward replays
    M_buffer = update_memory_b(M_buffer, H, params) # start with backward buffer
    if method_r=='shuffle': # shuffle transitions (in place)
        random.shuffle(M_buffer)
    elif method_r=='draw': # draw as many elements as len(M_buffer), with possible duplicates
        M_buffer = random.choices(M_buffer, len(M_buffer))
    return M_buffer


# ********** REPLAY 3 - MEMORY UPDATE WITH DIVERSITY *************** #

def replay_d(M_buffer, Q, params):
    '''Replay type : Most diverse experienced sequences replay.
    :param M_buffer: list of sequences kept in memory for optimizing diversity
    :param Q: Q-value matrix (numpy array)
    :return Q: updated Q-value matrix (numpy array)
    '''
    for repet in range(params['RR']): # each sequence is replayed a number of times
        for i_seq in range(len(M_buffer)): # each sequence
            seq = M_buffer[i_seq]
            for (s0, a, s1, r) in seq:
                Qnew = Temporal_Difference_Learning(Q, s0, s1, a, r, params)
                Q[s0,a] = Qnew
    return Q

def distance_seq(seq1, seq2, params):
    '''Compute how different two sequences are.
    :param seq1: (resp. seq2) List storing a set of transitions (s0,a,s1,r).
    :return d12: Distance between both sequences, d(seq1,seq2) = number of couples (state,a) which differ between seq1 and seq2.
    '''
    M1 = np.zeros((params['nS'], params['nA']))  # will store how many times a couple (s0,a) appears in a sequence
    M2 = M1.copy()
    for k in range(len(seq1)): # handle the first sequence
        s01, a1, s11, r1 = seq1[k]
        M1[s01,a1] += 1
    for k in range(len(seq2)):
        s02, a2, s21, r2 = seq2[k]
        M2[s02,a2] += 1
    count = 0  # count how many (s0,a) couples both sequences have in common
    for i in range(params['nS']):
        for j in range(params['nA']):
            count += min(M1[i,j], M2[i,j])
    d = params['RSS_d'] - count # distance = number of couples which differ between both sequences
    return d

def update_memory_d(M_buffer, D_seq, H, params):
    '''Updates memory, by including or not a new sequence.
    One sequence can be added in memory if :
    1) it is at least at a minimal distance from the other sequences in memory
    2) and :
    - EITHER there is enough space in memory 
    - OR memory is filled, but the distance between pairs of sequences would increase by replacing one sequence by the new sequence
    :param new_seq: sequence to be stored (or not) in memory (list of transitions (s0, a, s1, r))
    :return M_buffer: list of stored sequences, containing at most n_seq_d sequences
    :return D_seq: pairwise distances between sequences in memory (array, dimension n_seq_d x n_seq_d)
    '''
    # 1) Extract a new sequence, of length RSS_d at most
    # print('upd mem', M_buffer)
    if len(H) >= params['RSS_d']: # enough transitions experienced so far 
        new_seq = H[-params['RSS_d']:] # last experienced transitions in history
    else : # otherwise, take the full history since the beginning of the learning phase
        new_seq = H
    # 2) Update, or not, with new sequence
    n = len(M_buffer) # size of memory = number of sequences
    if n == 0: # no sequence in memory
        M_buffer.append(new_seq)
    else:
        d_list = [distance_seq(seq, new_seq, params) for seq in M_buffer] # distances between new_seq and the other sequences in memory
        d_new = min(d_list)
        if d_new >= params['epsilon_d']: # new_seq satisfies the condition to enter in memory
            add = False # 1) decide to append or not the new sequence, False by default
            if n < params['n_seq_d']: # memory not filled
                i_new = n # append sequence at the end of memory
                M_buffer.append(new_seq)
                add = True
            else: # filled memory
                # extract for each stored sequence its distance to its closest neighbour
                D_mins = [min([D_seq[i,j] for i in range(n) if i!=j]) for j in range(n)]
                d_min = min(D_mins)
                if d_new >= d_min: # adding new_seq increases the minimum distance so far
                    i_new = D_mins.index(d_min) # index of the sequence with the current minimal distance to all others
                    M_buffer[i_new] = new_seq # replace it by new_seq
                    add = True
            if add: # 2) update distance matrix
                for i in range(n): # for each sequence previously stored 
                    if i != i_new: # avoid diagonal cell D_seq[i_new,i_new]
                        D_seq[i,i_new] = d_list[i] # update distance between sequence i and new_seq (d_list[i])
                        D_seq[i_new,i] = d_list[i] # symmetric matrix
    return M_buffer, D_seq
    

# ********** REPLAY 4 - UPDATE MEMORY BUFFER WITH SURPRISE *************** #

def replay_p(Q, hatP, hatR, M_buffer, Delta, params):
    '''Replay type : Prioritized sweeping, for MB agent.
    Update in priority states associated with a greater surprise, i.e. reward prediction error, and to their predecessors. 
    :param M_buffer: memory of transitions to be replayed (list of (s0, a, r, s1))
    :param Delta: list of priority values associated with each transition
    WARNING : M_buffer and Delta are sorted in *increasing* order, so replay must start by the last element
    :param Q: Q-value matrix (numpy array)
    :return Q: updated Q-value matrix (numpy array)
    :return M_buffer: updated memory buffer with predecessors
    :return Delta: associated priorities
    '''
    h_repl = []
    for repet in range(params['RR']*params['RSS']): # number of replayed transitions, to be equal with other replay types
        if len(Delta) > 0: # buffer not empty
            s0, a, s1, r = M_buffer[-1] # take the last element, with higher priority
            delta = Delta[-1]
            Qnew = Value_Iteration(Q, hatP, hatR, s0, a, params)
            dQ = Qnew - Q[s0,a]
            Q[s0,a] = Qnew # update Q matrix in place
            dnew = abs(dQ) # new priority = prediction error
            del Delta[-1] # remove last element just been replayed
            del M_buffer[-1]
            update_memory_p(s0, a, s1, r, dnew, M_buffer, Delta, params) # reinsert this transition with new priority
            introduce_predecessors(s0, delta, hatP, hatR, Q, M_buffer, Delta, params) # insert its predecessors
            h_repl.append((s0, a, s1, r))
    return Q, M_buffer, Delta, h_repl

def replay_p(Q, hatP, hatR, M_buffer, Delta, params, convergence=True, RRmax=1000, method_p='predecessor'):
    '''Replay type : Prioritized sweeping, for MB agent.
    Update in priority states associated with a greater surprise, i.e. reward prediction error, and to their predecessors. 
    :param Q: Q-value matrix (numpy array)
    :param M_buffer: memory of transitions to be replayed (list of (s0, a, r, s1))
    :param Delta: list of priority values associated with each transition
    WARNING : M_buffer and Delta are sorted in *increasing* order, so replay must start by the last element
    WARNING : in introduce_predecessors
        if delta is used : a fixed point can be reached and prevent the buffer to evolve
                        this occurs when a state admits itsed as predecessor (s_pred=s0), delta = 1 and hatP[s0,a,s0] > 0 for some a,
                        but the state has been updated and thus it does not need to be appended anymore
                        this can be solved by adding the condition that the predecessor is introduced if its own priority is not null
        if delta_new is used : in some cases delta_new = 0, which can prevent appending predecessors
    :return Q: updated Q-value matrix (numpy array)
    :return M_buffer: updated memory buffer with predecessors
    :return Delta: associated priorities
    '''
    h_repl = []
    delta = 0
    if convergence:
        RR = RRmax
    else:
        RR = params['RR']
    for repet in range(RR):
        if convergence and (repet == RR - 1):
            # raise ValueError('More than 1000 replays repetitions needed in MB algorithm!')
            print('More than 1000 replays repetitions needed in MB algorithm.')
            print(delta)
        if repet == 0 or delta > params['epsilon_p']:
            for rr in range(params['RSS']):
                if len(M_buffer) != 0: # buffer not empty
                    s0, a, s1, r = M_buffer[-1] # take the last element, with higher priority
                    delta = Delta[-1]
                    Qnew = Value_Iteration(Q, hatP, hatR, s0, a, params)
                    dQ = Qnew - Q[s0,a]
                    delta_new = abs(dQ) # new priority = prediction error
                    # print('Mbuffer', M_buffer)
                    # print('Delta', Delta)
                    del Delta[-1] # remove last element just been replayed
                    del M_buffer[-1]
                    Q[s0,a] = Qnew # update Q matrix in place
                    update_memory_p(s0, a, s1, r, delta_new, M_buffer, Delta, params) # reinsert this transition with new priority
                    introduce_predecessors(s0, delta, hatP, hatR, Q, M_buffer, Delta, params, method_p=method_p) # insert its predecessor
                    # print('After predecessors')
                    # print('Mbuffer', M_buffer)
                    # print('Delta', Delta)
                    # print('-------')
                    h_repl.append((s0, a, s1, r))
                else: # no more transition to be replayed -> break the *outer* loop by setting delta = 0
                    delta = 0
        else:
            break
    return Q, M_buffer, Delta, h_repl


def already_stored(s0, a, M_buffer):
    '''Indicates if a transition is already present in the buffer and its position.
    :return i_b: index of this transition in the buffer, None is not present (or empty buffer).'''
    i_b = None
    for i in range(len(M_buffer)):
        s0_b, a_b, s1_b, r_b = M_buffer[i]
        if (s0_b, a_b) == (s0, a) :
            i_b = i
    return i_b

def update_memory_p(s0, a, s1, r, delta, M_buffer, Delta, params):
    '''Introduces (or not) a pair (state, action) in the buffer *in place*, so that optimizing surprise (measured by 'priority').
    Unlike backward and random method, updates have to be made have to be made *during learning*, after each transition.
    Updates are also performed *during replays*, for reinserting updated transitions.
    Conditions for introduction in memory :
    - either (s0, a) is not already in the buffer and priority exceeds threshold, and the buffer is not full,
    - or (s0, a) is already in the buffer, but the new pair (s0, a) has a higher priority than the one previously in the buffer
    :param delta: priority of the transition, delta = abs(RPE(s0,a))
    :return M_buffer: sequence of transitions (s0, a, s1, r) to be replayed (list of length at most RSS)
    :return Delta: associated priorities
    WARNING : By construction (successive insertions) the buffer is sorted in *increasing* order of priority.
    '''
    if delta > params['epsilon_p']: # above threshold to be introduced
        # print('Successful !')
        i_b = already_stored(s0, a, M_buffer) # search if already present in memory
        i_new = bisect.bisect(Delta, delta) # candidate position to insert the new transition (potentially)
        if i_b is None: # not already stored
            if len(M_buffer) < params['RSS']: # memory not full (which encompasses reinsertions, as the replayed transition has just been removed before)
                Delta.insert(i_new, delta)
                M_buffer.insert(i_new, (s0, a, s1, r))
        else: # already stored
            if delta > Delta[i_b]: # higher priority than previously stored experience
                # 1) insert the new experience, which is placed *after* the previous one in the list
                Delta.insert(i_new, delta)
                M_buffer.insert(i_new, (s0, a, s1, r))
                # 2) remove the previous experience, whose index is conserved as the list was not modified ahead of its position
                del Delta[i_b]
                del M_buffer[i_b]
    return M_buffer, Delta

def introduce_predecessors(s0, delta0, hatP, hatR, Q, M_buffer, Delta, params, method_p='predecessor'):
    '''Introduces (or not) predecessors (s, u) of state s0.
    The priority of a predecessor is delta*T[s,u,s0].
    Those updates have to be made during the *replay phase*.
    :param s0: arrival state, taken from the buffer.
    :param delta: priority associated with the transition involving s0.'''
    for u in range(params['nA']): # for all actions which could have led to state s0
        where_pred = hatP[:,u,s0] > 1/params['nS'] # boolean vector of length nS, spotting all previous_states which can lead to state s0 with a probability higher than 1/nS
        # where_pred = hatP[:,u,s0] > 0
        predecessors = np.arange(params['nS'])[where_pred] # array of predecessors' indices
        for s_pred in predecessors :
            Qnew = Value_Iteration(Q, hatP, hatR, s_pred, u, params)
            dQ = Qnew - Q[s_pred,u]
            delta_pred = abs(dQ) # new priority = prediction error
            if method_p == 'arrival': 
                # priority = priority of the arrival state weigthed by the transition probability to s0
                delta = delta0*hatP[s_pred,u,s0]
            elif method_p == 'predecessor': 
                # priority = surprise of the transition
                delta = delta_pred
            r_pred = 0 # as in Mehdi's code, or rather in r_pred = hatR[s_pred, u] ?
            if delta_pred > 0: # to prevent pathological cases, see WARNING in replay_p
                update_memory_p(s_pred, u, s0, r_pred, delta, M_buffer, Delta, params) # insert predecessor in buffer
    return M_buffer, Delta


def introduce_predecessors(s0, delta0, hatP, hatR, Q, M_buffer, Delta, params, method_p='predecessor'):
    '''Introduces (or not) predecessors (s, u) of state s0.
    The priority of a predecessor is delta*T[s,u,s0].
    Those updates have to be made during the *replay phase*.
    :param s0: arrival state, taken from the buffer.
    :param delta: priority associated with the transition involving s0.'''
    metor_p = 'arrival'
    predecessors = []
    r_pred = 0 # as in Mehdi's code, or rather in r_pred = hatR[s_pred, u] ?
    for u in range(params['nA']): # for all actions which could have led to state s0
        all_preds = np.arange(params['nS'])[hatP[:,u,s0]>1/params['nS']] 
        # array of predecessors' states, with boolean indexing
        # inside brackets: boolean vector of length nS, spotting all previous_states which can lead to state s0 with a probability higher than 1/nS
        for s_pred in all_preds:
            predecessors.append((s_pred, u, s0, r_pred))
    random.shuffle(predecessors) # in case of equal priority, to avoid biaising the introduction of the last actions and states to be considered in the loop
    for s_pred, u, s0, r_pred in predecessors :
        Qnew = Value_Iteration(Q, hatP, hatR, s_pred, u, params)
        dQ = Qnew - Q[s_pred,u]
        delta_pred = abs(dQ) # new priority = prediction error
        if method_p == 'arrival': 
            # priority = priority of the arrival state weigthed by the transition probability to s0
            delta = delta0*hatP[s_pred,u,s0]
        elif method_p == 'predecessor': 
            # priority = surprise of the transition
            delta = delta_pred
        if delta_pred > 0: # to prevent pathological cases, see WARNING in replay_p
            update_memory_p(s_pred, u, s0, r_pred, delta, M_buffer, Delta, params) # insert predecessor in buffer
    return M_buffer, Delta


# ********** RUNNING SIMULATIONS *************** #

def trial_exploration(replay_type, params, Q, hatP, hatR, N, M_buffer, Delta):
    '''Phase 1 - Decision-making and Learning, until findind the reward.
    See trial_Q_learning for inputs and outputs.'''
    h_explo = [] # reinitialize the history of transitions for the current trial
    s0 = params['s_start']
    while s0 != params['s_rw']: # while reward not found
        a = decision_making(s0, Q, params)
        s1, r = consequence_env(s0, a, params)
        if replay_type == 4: # model-based agent performing prioritized sweeping (4)
            N, hatP, hatR = MB_learning(s0, s1, a, r, N, hatP, hatR, params) # updates *in place* the agent's model of the environment : N, hatP, hatR
            Qnew = Value_Iteration(Q, hatP, hatR, s0, a, params)
            dQ = Qnew - Q[s0,a]
            delta = abs(dQ) # priority = prediction error, to be used for prioritized sweeping
            Q[s0,a] = Qnew # update Q
            M_buffer, Delta = update_memory_p(s0, a, s1, r, delta, M_buffer, Delta, params) # update *during learning', by inserting the new transition if possible
        else: # model-free agent, performing either no replay (0), or backward replay (1), or random replay (2), or most diverse replays (3)
            Qnew = Temporal_Difference_Learning(Q, s0, s1, a, r, params)
            dQ = Qnew - Q[s0,a]
            Q[s0,a] = Qnew # updates Q
        h_explo.append((s0, a, s1, r)) # save new transition
        s0 = s1
    return Q, hatP, hatR, N, M_buffer, Delta, h_explo

def trial_replays(replay_type, params, H, Q, hatP, hatR, N, M_buffer, D_seq, Delta, convergence=True, method_r='shuffle', method_p='predecessor'):
    '''Phase 2 - Replays during the Inter Trial Interval.
    See trial_Q_learning for inputs and outputs.'''
    # NB : h_repl for the current trial is not initializes, as computed by the following functions
    if replay_type == 0:
        h_repl = []
    if replay_type == 1: # backward replays
        M_buffer = update_memory_b(M_buffer, H, params) # update memory *at the end of the trial*
        Q = replay_b(M_buffer, Q, params, convergence=convergence)
        h_repl = M_buffer.copy()
    elif replay_type == 2 : # random replays
        M_buffer = update_memory_r(M_buffer, H, params, method_r=method_r) # update memory *at the end of the trial*
        Q = replay_r(M_buffer, Q, params, convergence=convergence)
        h_repl = M_buffer.copy()
    elif replay_type == 3: # most diverse experiences replays
        M_buffer, D_seq = update_memory_d(M_buffer, D_seq, H, params)
        Q = replay_d(M_buffer, Q, params)
        h_repl = M_buffer.copy()
    elif replay_type == 4: # prioritized sweeping
        # no need to update buffer, since it was done during learning
        # memory is updated with predecessors during the replay phase
        Q, M_buffer, Delta, h_repl = replay_p(Q, hatP, hatR, M_buffer, Delta, params, convergence=convergence, method_p=method_p)
    return Q, M_buffer, D_seq, Delta, h_repl

def trial_Q_learning(replay_type, params, H=None, Q=None, hatP=None, hatR=None, N=None, M_buffer=None, D_seq=None, Delta=None, convergence=True, method_r='shuffle', method_p='predecessor'):
    '''Performs *one trial*, which encompasses the following steps : 
    1) Decision-making and Learning
    2) Replays
    :param H: history of transitions since the beginning of the learning phase, list storing all the transitions (s0,a,s1,r)
    :param Q: Q-value matrix (numpy array, (dimension nS, nA))
    :param hatP: (model-based agent) estimated transition matrix
    :param hatR: (model-based agent) estimated reward function
    :param N: (model-based agent) number of visits to each pair (s,a)
    :param M_buffer: (if replays 1,2,3,4) memory of transitions to be replayed, list containing at most RSS transitions or n_seq_d sequences
    :param D_seq: (for most diverse sequence replays) distance matrix between the sequences in memory (size (n_seq_d,n_seq_d))
    :param Delta:(for mprioritized sweeping) priority values associated with each transition in memory
    :return h_explo: sequence of transitions performed during the current trial
    Returns also the updated versions of the data, depending on the agent and the type of replay performed.
    '''
    # Initialize data for the first trial (otherwise take that of previous trial)
    if H is None:
        H = [] # history of transitions, storing all the transitions (s0,a,s1,r) during the simulation
    if Q is None:
        Q = np.zeros((params['nS'], params['nA']))
    if N is None:
        N = np.zeros((params['nS'], params['nA']))
    if hatP is None:
        hatP = 0*np.ones((params['nS'], params['nA'], params['nS']))
    if hatR is None:
        hatR = np.zeros((params['nS'], params['nA']))
    if M_buffer is None:
        M_buffer = [] 
        D_seq = np.inf*np.ones((params['n_seq_d'], params['n_seq_d'])) # (for most diverse replays) distance matrix between sequences
        Delta = [] # (for prioritized sweeping) priority values associated with transitions in memory
    # 1) Exploration (active experience)
    # print('Exploration')
    Q, hatP, hatR, N, M_buffer, Delta, h_explo = trial_exploration(replay_type, params, Q, hatP, hatR, N, M_buffer, Delta)
    H += h_explo # append the trial to the full history
    Q_explo = Q.copy() # save the current state of the Q matrix after exploration
    # 2) Replays during the Inter Trial Interval
    # print('Replays')
    Q, M_buffer, D_seq, Delta, h_repl = trial_replays(replay_type, params, H, Q, hatP, hatR, N, M_buffer, D_seq, Delta, convergence=convergence, method_r=method_r, method_p=method_p)
    Q_repl = Q.copy() # save the current state of the Q matrix after replay
    return H, Q, hatP, hatR, N, M_buffer, D_seq, Delta, h_explo, h_repl, Q_explo, Q_repl


# ********** FINDING THE OPTIMAL POLICY *************** #

def get_transitions(params, s, a):
    '''Get all the information regarding the transition matrix and the reward for a give state and action.
    :param s: int representing the starting state
    :param a: int representing the global action
    :return transisions: list of tuples containing the arrival state, the probability to get the arrival state and the reward of the
    arrival state
    '''
    transitions = []
    if s != params['s_rw']:
        reward = [0]*params['nS']
        reward[params['s_rw']] = 1
        for s1 in range(len(params['T'][s,a,:])):
            transitions.append((s1, params['T'][s,a,s1], reward[s1]))
    return transitions

def optimal_value_iteration(params, Q=None, eps=0.01):
    '''Value iteration to obtain the optimal policy.
    :param eps: the threshold to compute convergence
    :return Qopt: np.array with the optimal q-values
    '''
    if Q is None:
        Qopt = np.zeros((params['nS'], params['nA']))
    else:
        Qopt = Q.copy()
    iter = 0
    while True:
        Qold = Qopt.copy()
        for s in range(params['nS']):
            for a in range(params['nA']):
                Qopt[s,a] = 0
                for s1, p, r in get_transitions(params, s, a):
                    Qopt[s,a] += p*(r + max(Qopt[s1,:])*params['gamma'])
        delta = np.max(np.abs(np.array(Qold) - Qopt))
        iter += 1
        if delta < eps:
            break
    return Qopt

