import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import graph_tool.all as gt

import experiment, utils

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
        #default
        #state = gt.IsingGlauberState(g, beta=1.5 / 10)
        pass

    # graph properties
    pos = gt.sfdp_layout(g)
    deg = g.degree_property_map("in")

    # simulation loop
    win = None
    for i in range(num_iters):
        print("iteration ", i)
        #ret = state.iterate_sync(niter=10)
        win = gt.graph_draw(g,
                            pos=pos,
                            vertex_fill_color=g.vp.opinion,
                            vcmap=matplotlib.cm.bone_r,
                            window=win,
                            return_window=True,
                            main=False,
                            vertex_size=25,
                            vertex_pen_width=2,
                            edge_pen_width=2)
        current_state[3] = experiment.get_updated_driving_forces(clusters=clusters,
                                                                 driving_forces=current_state[1],
                                                                 agents=current_state[0],
                                                                 opinions=current_state[1],
                                                                 update_agents=False)
        current_state = experiment.update_state(current_state)

        # locally update graph properties
        utils.update_property(g, prop_name="opinion", opinion=current_state[1])
