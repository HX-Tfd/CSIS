from graph_tool.all import *

import graph_tool.all as gt
import os
import matplotlib.pyplot as plt

import numpy as np
import scipy
from scipy.stats import norm
import math
import random
from random import randint

import utils, experiment, visualization


def main():
    '''
    Global configurations
    '''
    k_alpha = 1
    k_beta = .5
    E_profit = 100
    K_self_coupling = -2

    num_clusters = 5
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

    current_state = [agents, opinions, probs, driving_forces]

    num_iters = 100
    visualization.run_simulation(g,
                                 current_state=current_state,
                                 clusters=clusters,
                                 num_iters=num_iters)


if __name__ == "__main__":
    main()
