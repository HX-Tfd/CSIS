import numpy as np
import random
import math
from scipy import stats

from model import *

'''
These are default parameters, set them differently in main.py
'''
k_alpha = 1
k_beta = .5
k_gamma = .5
E_profit = 100
K_self_coupling = -2

'''
Create {num_clusters} random clusters
if cluster_specific = True, use initialise_driving_force() to initialise cluster-specific driving force

Returns:
           cluster_elements:  {"cluster_i": numpy array of agent ids in cluster i}
           driving_forces:    numpy array of driving forces (properly-/0-initialised)
           agents:            numpy array of agent intrinsics
           opinions:          a numpy binary array of the opinions

'''
def create_clusters(num_clusters, num_agents, cluster_specific=False):
    assert num_agents >= num_clusters >= 1

    # initialise driving forces
    if cluster_specific:
        driving_forces = initialise_driving_forces(num_agents)
    else:
        driving_forces = np.zeros(num_agents)

    # initialise agents intrinsics
    agents = initialise_agents(num_agents)

    # initialise clusters
    cluster_elements = initialise_clusters(num_agents, num_clusters)
    cluster_elements = np.array(cluster_elements)

    # initialise opinions
    opinions = initialise_opinions(num_agents)

    return [cluster_elements, driving_forces, agents, opinions]


'''
Initialises the clusters
You can define initialization schemes here
currently does uniform partition

Retuns: a numpy array containing the driving force of each agent

'''


def initialise_clusters(num_agents, n):
    l = list(range(num_agents))
    random.shuffle(l)
    res = np.array([l[i::n] for i in range(n)], dtype=object)
    return res


'''
Initialises the driving forces for each agent in {agents}
You can define initialization schemes here

Retuns: a numpy array containing the driving force of each agent

'''


def initialise_driving_forces(num_agents):
    driving_forces = np.random.rand(num_agents)  # hard-coded for now
    return driving_forces


'''
Initialises the intrinsics for each agent in {agents}
You can define initialization schemes here

Retuns: a numpy array containing the intrinsics of each agent
'''


def initialise_agents(num_agents):
    return np.zeros(num_agents)  # hard-coded for now


'''
Initialises the opinions for each agent in {agents}
You can define initialization schemes here

Retuns: a numpy array containing the opinions of each agent
'''


def initialise_opinions(num_agents):
    return np.random.randint(2, size=num_agents)  # hard-coded for now


'''
    updates the agent intrinsics (optional), opinions and driving forces 
'''
def get_updated_driving_forces(clusters, driving_forces, agents, opinions, update_agents=False):
    # update each cluster locally resp. each agent
    updated_driving_force = np.zeros(len(driving_forces))
    for n in range(len(clusters)):
        # agents_ids = clusters["cluster_{}".format(n)]
        clusters = np.array(clusters)
        agents_ids = clusters[n][:]
        cluster_size = len(agents_ids)

        # number of different opinions within the same cluster
        num_diffops = np.zeros(len(opinions))
        for a in agents_ids:
            num_diffops[a] = np.sum([x for x in opinions[np.array(agents_ids).astype(int)] if x != a])

        # update driving forces
        for a in agents_ids:
            updated_driving_force[a] = E_profit + num_diffops[a] * (k_alpha + k_beta + k_gamma)
    return updated_driving_force  # currently it returns the unchanged agent intrinsics


'''
Given the current state, returns a new state

Input: 
       -current_state: (agents' instrinsics, opinions, probabilities of change, driving forces)
       -f:             the state transition function

Returns:
       The new state f(current_state)
'''
def update_state(current_state, has_changed):
    return transition_function(current_state, has_changed)


'''
You can define state transition functions here
[agents, opinions, probs, driving_forces]
'''
def transition_function(state, has_changed):
    agents = state[0]
    state[2] = change_probs(state[3])

    num_changes = 0
    for i in range(len(agents)):
        # if has_changed[i]:
        #     pass
        # else:
        new_opinion = opinion(state[1][i], state[2][i])
        if new_opinion != state[1][i]:
            num_changes += 1
            #     has_changed[i] = True
        state[1][i] = new_opinion

    return state, has_changed, num_changes


'''
Part of modelling the transition function
into [a, b]
'''
def normalise_driving_force(driving_forces, a, b):
    min_val = np.amin(driving_forces)
    #max_val = np.amax(driving_forces)
    max_val = ((len(driving_forces) * (k_alpha+k_beta+k_gamma)) + E_profit)
    #return (b-a) * (driving_forces-min_val)/(max_val-min_val) - a
    return ((b-a)* driving_forces/max_val)+a

'''
compute the probability of change given a driving force    
'''
def change_probs(driving_forces, use_nn=False):
    #probs = a_0+a_1*driving_forces+a_2*driving_forces**2+a_3*driving_forces**3
    #probs = np.repeat(0.01, len(driving_forces))
    #probs = driving_forces/(len(driving_forces) * k_alpha + E_profit)
    if use_nn:
        pass
        # TODO: get probs_ from the dataset and keep the
        #  trained model somewhere so that we don't have to
        #  load it in each simulation step
        #  an example initialization of the model is given in visualization.py
    else:
        driving_forces = normalise_driving_force(driving_forces, -2, 2) # be aware of the normalization range for different distributions
        probs = stats.norm.cdf(driving_forces)

    return probs


'''
The new opinion {x} of an agent given the prob. of change {p}

Can serve as a state transition function
'''
def opinion(x, p):
    res = (np.random.rand(1))
    return 1-x if res < p else x
