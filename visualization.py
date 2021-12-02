import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import functools

import graph_tool.all as gt

import experiment, utils, model

def show_graph(g, dynamic=True):
    # draw opinions
    if dynamic:
        plt.switch_backend("GTK3Cairo")
    # layout_list = ['fruchterman_reingold_layout', 'sfdp_layout']
    state = gt.minimize_nested_blockmodel_dl(g)

    # try different layouts
    pos = gt.sfdp_layout(g)
    deg = g.degree_property_map("in")
    # draw_order = np.concatenate(clusters)  # by cluster
    fill_color = g.vp.opinion #g.graph_properties["opinion"]

    if dynamic:
        gt.graph_draw(g, pos=pos,
                      vertex_fill_color=fill_color,  # vorder=draw_order,
                      vertex_size=25,
                      vertex_pen_width=2,
                      edge_pen_width=2,
                      inline=True,
                      mplfig=plt.figure()
                      )

'''
    Run model for {num_iters} iterations
    kwargs are arguments for dynamics
    
    currently draws the opinions
'''
def run_simulation(g, current_state, clusters, num_iters=100, dynamics=None, **kwargs):
    if dynamics is not None:
        state = dynamics(**kwargs)
    else:
        # TODO: try other states
        state = gt.IsingGlauberState(g, beta=1.5 / 10)

    # graph properties
    pos = gt.sfdp_layout(g)
    deg = g.degree_property_map("in")
    centrality = gt.betweenness(g)[1]

    num_agents = len(current_state[0])
    has_changed = [False for i in range(num_agents)]

    # initialise parameterized distribution (for demonstration only, not used for the visualization loop)
    distribution = model.SimpleFCNet(num_layers=1,
                        in_dim=num_agents,
                        out_dim=num_agents,
                        hidden_dim=2 * num_agents)

    # simulation loop
    win = None
    num_changes_list = []
    for i in tqdm(range(num_iters), desc="running simulation ..."):
        state.iterate_sync(niter=num_iters)
        win = gt.graph_draw(g,
                            pos=pos,
                            vertex_fill_color=g.vp.opinion,
                            vcmap=matplotlib.cm.bone_r,
                            window=win,
                            return_window=True,
                            main=False,
                            vertex_size=25,
                            vertex_pen_width=2,
                            edge_pen_width=2,
                            edge_color='k'
                            #output="it{}.pdf".format(i)
                            )
        current_state[3] = experiment.get_updated_driving_forces(clusters=clusters,
                                                                 driving_forces=current_state[1],
                                                                 agents=current_state[0],
                                                                 opinions=current_state[1],
                                                                 update_agents=False)
        current_state, has_changed, num_changes = experiment.update_state(current_state, has_changed)
        if functools.reduce(lambda a, b: a and b, has_changed):
            print("all agents have changed")
            break

        num_changes_list.append(num_changes)

        # locally update graph properties
        utils.update_property(g, prop_name="opinion", opinion=current_state[1])

    # plot simulation statistics
    n_bins_changes = len(has_changed)
    n_bins_iters = num_iters
    fig, axs = plt.subplots(1, 2, tight_layout = True)
    y = num_changes_list
    plt.title(r'$k_\alpha = {}, k_\beta = {}, k_\gamma = {}$'.format(
        experiment.k_alpha, experiment.k_beta, experiment.k_gamma))
    axs[0].hist(y, bins=n_bins_changes)
    axs[0].set_title('number of changes')
    axs[0].set_ylabel('iteration of changes')
    axs[0].set_xlabel('number of agents')

    axs[1].plot(np.linspace(1, n_bins_iters, num=n_bins_iters, dtype=int), num_changes_list)
    axs[0].set_title('number of changes in each iteration')
    axs[1].set_ylabel('number of changed agents')
    axs[1].set_xlabel('iteration')
    plt.savefig("experiment_exp")
    plt.show()

