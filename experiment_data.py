from scipy import stats
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


from model_data import SimpleFCNet, train
from dataio import (num_agents,
                   opinion_timeline_agents,
                   max_node_array,
                   opinion_timeline_agents,
                   cluster_range,
                   cluster_size,
                   cluster_names,
                   edge_properties,

                    )
E_profit = 15
k_alpha = 1
k_beta = 2
k_gamma = 3


'''
Get the updated driving force of the model from the parameters and the cluster information
'''
def update_driving_forces_model(opinions, driving_forces, edge_props, cluster_index, k_alpha_model,
                                k_beta_model,
                                k_gamma_model, E_profit_model):
    first_node = 0
    if cluster_index == 0:
        len_cluster = max_node_array[0]
    else:
        len_cluster = max_node_array[cluster_index] - max_node_array[cluster_index - 1]
        first_node = max_node_array[cluster_index - 1]

    updated_driving_force = np.zeros(len_cluster)
    agents_ids = [first_node + i for i in range(len_cluster)]

    for a in agents_ids:
        for edge in edge_props:
            # check that the edge is directed at node a and opinions of agents don't match
            if edge[1] == a and not opinions[a - first_node] == opinions[edge[0] - first_node]:
                updated_driving_force[a - first_node] += edge[2] * k_alpha_model + edge[3] * k_beta_model + edge[
                    4] * k_gamma_model
        updated_driving_force[a - first_node] += E_profit_model

    return updated_driving_force


def get_initial_opinions(cluster_number):
    initial_opinions = []

    for i in range(num_agents):
        initial_opinions.append(opinion_timeline_agents[i][0])

    peoria_initials = initial_opinions[0:max_node_array[0]]
    bloomington_initials = initial_opinions[cluster_range[1][0]:cluster_range[1][1]]
    quincy_initials = initial_opinions[cluster_range[2][0]:cluster_range[2][1]]
    galesburg_initials = initial_opinions[cluster_range[3][0]:cluster_range[3][1]]
    opinion_initials_per_cluster = [peoria_initials, bloomington_initials, quincy_initials, galesburg_initials]

    return opinion_initials_per_cluster[cluster_number]


# # get percentage of opinion 1 at each timepoint with our model
def get_change_simulated(edge_props, cluster_index):
    simulated_percentage_overtime = []

    # initial state
    # initial_state_all = [0,opinion_timeline_agents[:,0]]
    # initial_state_this_cluster = initial_state_all[cluster_range[i][0]:cluster_range[i][1]]

    # simulated_percentage_overtime.append(initial_state_this_cluster)
    initial_state_this_cluster = get_initial_opinions(cluster_index)
    current_state = [0, initial_state_this_cluster]
    simulated_percentage_overtime.append(
        [0, np.count_nonzero(initial_state_this_cluster) / cluster_size[cluster_index]])

    # calculate driving force(cluster specific)
    driving_force = update_driving_forces_cluster_specific(edge_props, cluster_index, current_state[1])

    # later states
    for i in range(98):
        new_opinions, driving_force = update(driving_force, edge_props, cluster_index, current_state[1])
        current_state = [i + 1, new_opinions]
        simulated_percentage_overtime.append([i + 1, np.count_nonzero(new_opinions) / cluster_size[cluster_index]])

    # plotting
    simulation = pd.DataFrame(simulated_percentage_overtime, columns=['time unit', 'percentage'])
    sns.set_theme()
    sns.lineplot(x='time unit', y='percentage', data=simulation, label=cluster_names[cluster_index], ci=None)

'''
update driving forces for a specific cluster
'''
def update_driving_forces_cluster_specific(edge_props, cluster_number, opinions):
    first_node = 0
    if cluster_number == 0:
        len_cluster = max_node_array[0]
    else:
        len_cluster = max_node_array[cluster_number] - max_node_array[cluster_number - 1]
        first_node = max_node_array[cluster_number - 1]

    updated_driving_force = np.zeros(len_cluster)
    agents_ids = [first_node + i for i in range(len_cluster)]

    for a in agents_ids:
        for edge in edge_props:
            # check that the edge is directed at node a and opinions of agents don't match
            if edge[1] == a and not opinions[a - first_node] == opinions[edge[0] - first_node]:
                updated_driving_force[a - first_node] += edge[2] * k_alpha + edge[3] * k_beta + edge[4] * k_gamma
        updated_driving_force[a - first_node] += E_profit

    return updated_driving_force

'''
'''
def prob_of_change(driving_forces):
    driving_forces = normalise_driving_force(driving_forces, -2, 2)
    probs = stats.norm.cdf(driving_forces)
    return probs


def normalise_driving_force(driving_forces, a, b):
    min_val = np.amin(driving_forces)
    # max_val = np.amax(driving_forces)
    max_val = ((len(driving_forces) * (k_alpha + k_beta + k_gamma)) + E_profit)
    # return (b-a) * (driving_forces-min_val)/(max_val-min_val) - a
    return ((b - a) * driving_forces / max_val) + a


def opinion(x, p):
    res = (np.random.rand(1))[0]
    return 1 - x if res < p else x


def opinion_no_return(x, p):
    if (x == 1):
        return 1
    else:
        return opinion(x, p)


def update(driving_forces, edge_props, cluster_index, current_state):
    new_states = []
    prob = prob_of_change(driving_forces)
    for agent in range(cluster_size[cluster_index]):
        # new_state = opinion(current_state[agent], prob[agent])
        new_state_no_return = opinion_no_return(current_state[agent], prob[agent])
        # new_states.append(new_state)
        new_states.append(new_state_no_return)

    new_driving_force = update_driving_forces_cluster_specific(edge_props, cluster_index, current_state)

    return new_states, new_driving_force


'''
Run simulation with the real data
'''
def run_simulation(cluster_index):
    edge_props_this_cluster = edge_properties[cluster_index]
    num_agents_in_cluster = cluster_size[cluster_index]

    k_alpha_model = .1
    k_beta_model = 1
    k_gamma_model = 2
    E_profit_model = 15

    # initialise clusters and agents
    initial_opinion = get_initial_opinions(cluster_index)

    parameters_to_fit = torch.tensor(
        [k_alpha_model,
         k_beta_model,
         k_gamma_model,
         E_profit_model], requires_grad=True
    )

    # initialise driving force
    driving_forces = update_driving_forces_cluster_specific(edge_props_this_cluster, cluster_index, initial_opinion)
    driving_forces = torch.tensor(driving_forces)

    distribution = SimpleFCNet(
        num_layers=1,
        in_dim=num_agents_in_cluster + 4,
        out_dim=num_agents_in_cluster,
        hidden_dim=2 * (num_agents_in_cluster)
    )


    all_opinions = []
    all_opinions.append(initial_opinion)

    for step in tqdm(range(10), desc="simulation step "):
        y = opinion_timeline_agents[cluster_range[cluster_index][0]:cluster_range[cluster_index][1], step]
        y = torch.tensor(y).float()

        # fit the parameters and get new opinions
        new_opinion = train(distribution, parameters_to_fit, driving_forces, y, step, all_opinions, timestep=step)
        all_opinions.append(new_opinion) # can be used later for the global training

        # update driving forces
        driving_forces = update_driving_forces_model(new_opinion, driving_forces, edge_props_this_cluster,
                                                     cluster_index,
                                                     parameters_to_fit[0].float(),
                                                     parameters_to_fit[1].float(),
                                                     parameters_to_fit[2].float(),
                                                     parameters_to_fit[3].float())
        driving_forces = torch.tensor(driving_forces, requires_grad=False)
        print(parameters_to_fit)

def run_simulation_global(cluster_index):
    edge_props_this_cluster = edge_properties[cluster_index]
    num_agents_in_cluster = cluster_size[cluster_index]

    #initial guesses
    k_alpha_model = .1
    k_beta_model = 1
    k_gamma_model = 2
    E_profit_model = 15

    # initialise clusters and agents
    initial_opinion = get_initial_opinions(cluster_index)

    parameters_to_fit = torch.tensor(
        [k_alpha_model,
         k_beta_model,
         k_gamma_model,
         E_profit_model], requires_grad=True
    )

    # initialise driving force
    driving_forces = update_driving_forces_cluster_specific(edge_props_this_cluster, cluster_index, initial_opinion)
    driving_forces = torch.tensor(driving_forces)

    distribution = SimpleFCNet(
        num_layers=1,
        in_dim=num_agents_in_cluster + 4,
        out_dim=num_agents_in_cluster,
        hidden_dim=2 * (num_agents_in_cluster)
    )


    all_opinions = []
    all_opinions.append(initial_opinion)

    for step in tqdm(range(10), desc="simulation step "):
        y = opinion_timeline_agents[cluster_range[cluster_index][0]:cluster_range[cluster_index][1], step]
        y = torch.tensor(y).float()

        # fit the parameters and get new opinions
        new_opinion =1
        all_opinions.append(new_opinion) # can be used later for the global training

        # update driving forces
        driving_forces = update_driving_forces_model(new_opinion, driving_forces, edge_props_this_cluster,
                                                     cluster_index,
                                                     parameters_to_fit[0].float(),
                                                     parameters_to_fit[1].float(),
                                                     parameters_to_fit[2].float(),
                                                     parameters_to_fit[3].float())
        driving_forces = torch.tensor(driving_forces, requires_grad=False)

    #train globally
    train(distribution, parameters_to_fit, driving_forces, y, step, all_opinions)
'''

What experiments to do:

dataset:
    NN, run locally (with, w/o betw cluster edges)
    NN, run globally (with, w/o betw cluster edges)
    Normal, run locally (with, w/o betw cluster edges)
    Normal, run globally (with, w/o betw cluster edges)
    
Random: 
    Normal, run locally (with, w/o betw cluster edges)
    Normal, run globally (with, w/o betw cluster edges)
    
Different loss functions

'''

run_simulation(0)
