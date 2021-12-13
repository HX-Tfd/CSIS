from graph_tool.all import *
import utils, experiment, visualization, model
# import torch
# import torch.nn as nn
# from torch.utils.data import dataloader

'''
A simulation loop can look like (simplified):

init_experiment_components()
init_distribution_network()

for n in num_timesteps:
    prediction = get_opinions_at_step(n)
    ground_truth = get_data_at_step(n)
    train(distribution_network, prediction, ground_truth, n)
    update_state()
'''

def main():
    # set global configurations
    experiment.k_alpha = .1 # discussion
    experiment.k_beta = 1 # friendship
    experiment.k_gamma = 2 # advice
    experiment.E_profit = 15

    # initialization
    num_clusters = 6
    num_agents = 20
    clusters, driving_forces, agents, opinions = experiment.create_clusters(num_clusters, num_agents)
    print("clusters: {}\n\n driving forces: {}\n\n agents:{}\n\n opinions:{}".format(clusters,
                                                                                     driving_forces,
                                                                                     agents,
                                                                                     opinions))
    g = utils.to_graph(clusters=clusters,
                       driving_forces=driving_forces,
                       agents=agents,
                       opinions=opinions,
                       )
    probs = experiment.change_probs(driving_forces)
    initial_state = [agents, opinions, probs, driving_forces]


    # simulation settings
    num_iters = 100
    visualization.run_simulation(g,
                                 current_state=initial_state,
                                 clusters=clusters,
                                 num_iters=num_iters)

if __name__ == "__main__":
    main()
