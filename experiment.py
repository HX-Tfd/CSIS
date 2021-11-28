import numpy as np
import random
import math

k_alpha = 1
k_beta = .5
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

        # number of different opinions
        num_diffops = np.zeros(len(opinions))
        for a in agents_ids:
            num_diffops[a] = np.sum([x for x in opinions[agents_ids] if x != a])

        # update driving forces
        for a in agents_ids:
            # driving_forces
            updated_driving_force[a] = E_profit + num_diffops[a] * k_alpha
            #print(updated_driving_force)
            #print("agent {} obtained a driving force of {}".format(a, updated_driving_force[a]))
        #print("resulting global df: {}\n".format(updated_driving_force))

    return updated_driving_force  # currently it returns the unchanged agent intrinsics


'''
Given the current state, returns a new state

Input: 
       -current_state: (agents' instrinsics, opinions, probabilities of change, driving forces)
       -f:             the state transition function

Returns:
       The new state f(current_state)
'''
def update_state(current_state):
    return transition_function(current_state)


'''
You can define state transition functions here
'''
def transition_function(state):
    agents = state[0]
    state[2] = change_probs(state[3])

    for i in range(len(agents)):
        new_opinion = opinion(state[1][i], state[2][i])
        state[1][i] = new_opinion
    return state


'''
Part of modelling the transition function

'''
def normalise_driving_force(driving_forces, hi):
    return driving_forces/hi


'''
compute the probability of change given a driving force    
'''
def change_probs(driving_forces):
    hi = np.sum(driving_forces) # the maximum possible driving force
    lo = None
    driving_forces = normalise_driving_force(driving_forces, hi )
    probs = [math.erf(driving_forces[i]) if driving_forces[i] < hi else math.erf(hi) for i in range(len(driving_forces)) ]
    return probs


'''
The new opinion {x} of an agent given the prob. of change {p}

Can serve as a state transition function
'''
def opinion(x, p):
    res = (np.random.rand(1))[0]
    return 1-x if res < p else x
