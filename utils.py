import numpy as np
import graph_tool.all as gt
import os
import matplotlib.pyplot as plt

import os

working_dir = '.'  # placeholder

'''
Saves the graph g to {target path} with name {name}
'''
def save_graph(g, name, target_path):
    abs_path = os.path.join(working_dir, target_path)
    g.save("{}.xml.gz".format(name))


'''
Loads the graph g from {source path} 

Returns: the loaded graph 
'''
def load_graph(source_path):
    return load_graph(source_path)


'''
add edges between clusters of the graph g

currently addds random edges between clusters

for every agent, add edge to another agent that is no in the same cluster with some probability
'''
def add_edges_between_clusters(g, cluster_assignment):
    p = 0.05
    num_agents = len(cluster_assignment)
    for a in range(num_agents):
        for a_ in range(num_agents):
            if not (a == a_ or cluster_assignment[a] == cluster_assignment[a_]):
                if p > np.random.rand(1):
                    g.add_edge(g.vertex(a), g.vertex(a_))


'''
converts the following to graphs and output in {target_directory}:

    - agent ids
    - agent intrinsics
    - driving forces
    - opinion

Input:
    agents:           numpy array of agent intrinsics
    clusters:         {"cluster_i": a list of agent ids}
    driving_forces:   numpy array of the driving forces
    opinions:         binary numpy array of the opinions
    dynamic:          use matplotlib for dynamic visualization instead of saving the graph directly
    target_directory: output directory


The graphs this function produces:
    1.Opinion
    2.Driving force
    3.Agent intrinsics (can be many separate features)
'''
def to_graph(clusters, driving_forces, agents, opinions):
    # directed graph, each cluster is a clique
    g = gt.Graph()
    num_agents = len(agents)
    num_clusters = clusters.shape[0]
    g.add_vertex(num_agents)
    cluster_assignment = np.zeros(num_agents)
    for cluster in range(num_clusters):
        agents_in_cluster = clusters[cluster][:]
        cluster_assignment[np.array(agents_in_cluster).astype(int)] = cluster
        for a1 in agents_in_cluster:
            for a2 in agents_in_cluster:
                if a1 != a2:
                    g.add_edge(g.vertex(a1), g.vertex(a2))

    add_edges_between_clusters(g, cluster_assignment)

    # add graph properties
    vprop_opinion = g.new_vertex_property("int")
    g.vertex_properties["opinion"] = vprop_opinion
    for v in g.iter_vertices():
        g.vp.opinion[v] = opinions[v]
    print(g.list_properties())

    #g.graph_properties["driving_force"]
    return g

'''
update property locally in the graph 

specify the data to be used for updating in kwargs, e.g. update_property(g, "opinion", opinions)

properties available:
- opinion=
- driving_force=

'''
def update_property(g, prop_name, **kwargs):
    assert prop_name is not None

    if prop_name == "opinion":
        opinions = kwargs["opinion"]
        for v in g.iter_vertices():
            g.vp.opinion[v] = opinions[v]
    elif prop_name == "driving_force":
        driving_force = kwargs["driving_force"]
        # for v in g.iter_vertices():
        #     g.vp.driving_force[v] = driving_force[v]
    else:
        raise RuntimeError("unknown property name: ", prop_name)