import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv('nodes')
df = df['specialty;"city";"patients";"free_time";"community";"friends";"adoption_date";"proximity";"med_sch_yr";"jours";"clubs";"meetings";"id";"discuss"'].str.split(';', expand=True)

num_agents = df.shape[0]

max_node_array = [116, 166, 210, 246]
cluster_range = [[0, 116], [116, 166], [166, 210], [210, 246]]
cluster_size = [116, 50, 44, 36]
cluster_names = ["Peoria", "Bloomington", "Quincy", "Galesburg"]


def get_edges_array(cluster_max):
    peoria_cluster_edges = []
    bloomington_cluster_edges = []
    quincy_cluster_edges = []
    galesburg_cluster_edges = []

    with open('medical_innovationver3.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        row_num = 0
        for row in reader:
            if row_num == 0:
                row_num += 1
            else:
                edge_properties = []
                args = row[0].split(";")
                node1 = int(args[0]) - 1
                node2 = int(args[1][1:-1]) - 1
                fr_ad_dis = [int(args[2][1:-1]), int(args[3][1:-1]), int(args[5][1:-1])]
                id = int(args[4][1:-1])
                edge_properties = [node1, node2, fr_ad_dis[0], fr_ad_dis[1], fr_ad_dis[2]]
                if (node1 <= cluster_max[0] and node2 <= cluster_max[0]):
                    peoria_cluster_edges.append(edge_properties)
                elif (node1 <= cluster_max[1] and node2 <= cluster_max[1]):
                    bloomington_cluster_edges.append(edge_properties)
                elif (node1 <= cluster_max[2] and node2 <= cluster_max[2]):
                    quincy_cluster_edges.append(edge_properties)
                else:
                    galesburg_cluster_edges.append(edge_properties)
    return peoria_cluster_edges, bloomington_cluster_edges, quincy_cluster_edges, galesburg_cluster_edges


###

# edges arrays arguments:

#   - node1 (from)

#   - node2 (to)

#   - friendship relation

#   - advice relation

#   - discussion relation

#  ###

peoria_edges, bloomington_edges, quincy_edges, galesburg_edges = get_edges_array(max_node_array)
edge_properties = []
edge_properties.append(peoria_edges)
edge_properties.append(bloomington_edges)
edge_properties.append(quincy_edges)
edge_properties.append(galesburg_edges)


'''

'''
ad_time_all_old = df[6]
ad_time_all=[]
for i in range(246):
  ad_time_all.append(int(ad_time_all_old[i][1:-1]))
ad_time_all=np.asarray(ad_time_all)


'''

'''
time_range = ad_time_all.max()

opinion_timeline_agents = []
for i in range(num_agents):
  opinion_timeline_agent_i = np.zeros(99)
  adoption_timepoint = int(ad_time_all[i])
  opinion_timeline_agent_i=[1 if j >= adoption_timepoint-1 else 0 for j in range(time_range) ]
  opinion_timeline_agents.append(opinion_timeline_agent_i)
opinion_timeline_agents=np.asarray(opinion_timeline_agents)


# label : percentage of opinion 1 at each timepoint

def get_change_real(cluster_index):
    real_percentage_overtime = []

    for i in range(99):
        cluster_column_i = opinion_timeline_agents[cluster_range[cluster_index][0]:cluster_range[cluster_index][1], i]
        ones = np.count_nonzero(cluster_column_i)
        percent = ones / cluster_size[cluster_index]

        combined = []
        combined.append(i)
        combined.append(percent)

        real_percentage_overtime.append(combined)

    data = pd.DataFrame(real_percentage_overtime, columns=['time unit', 'percentage'])

    sns.set_theme()
    sns.lineplot(x='time unit', y='percentage', data=data, label=cluster_names[cluster_index], ci=None)

    return real_percentage_overtime


real_cluster_0 = get_change_real(0)
real_cluster_1 = get_change_real(1)
real_cluster_2 = get_change_real(2)
real_cluster_3 = get_change_real(3)