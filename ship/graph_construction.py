import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
from tqdm import tqdm
import networkx as nx
import torch

import os
import psutil
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, log_loss
from sklearn.metrics import precision_recall_curve
from IPython.display import clear_output
import sys
sys.path.append("..")
from opera_tools import plot_graphx, DISTANCE, scattering_estimation_loss
from sklearn.linear_model import TheilSenRegressor
from copy import deepcopy      
from collections import Counter
from torch_geometric.data import Data

import heapq

from create_graph.create_graph import generate_distances
NUM_SHOWERS_IN_BRICK = 200



from math import fabs, sqrt, log
def rms_integral_root_closed_py(basetrack_left, basetrack_right):
    EPS = 1e-6
    dz = basetrack_right['features']['SZ'] - basetrack_left['features']['SZ']
    dx = basetrack_left['features']['SX'] - (basetrack_right['features']['SX'] - basetrack_right['features']['TX'] * dz)
    dy = basetrack_left['features']['SY'] - (basetrack_right['features']['SY'] - basetrack_right['features']['TY'] * dz)
    dtx = (basetrack_left['features']['TX'] - basetrack_right['features']['TX'])
    dty = (basetrack_left['features']['TY'] - basetrack_right['features']['TY'])
    
    a = (dtx * dz) ** 2 + (dty * dz) ** 2
    b = 2 * (dtx * dz * dx +  dty * dz * dy)
    c = dx ** 2 + dy ** 2
    if a == 0.:
        return fabs(sqrt(c))
    discriminant = (b ** 2 - 4 * a * c)
    log_denominator = 2 * sqrt(a) * sqrt(a + b + c) + 2 * a + b + EPS
    log_numerator = 2 * sqrt(a) * sqrt(c) + b + EPS
    first_part = ( (2 * a + b) * sqrt(a + b + c) - b * sqrt(c) ) / (4 * a)
    
    if fabs(discriminant) < EPS:
        return fabs(first_part)
    else: 
        result = fabs((discriminant * log(log_numerator / log_denominator) / (8 * sqrt(a * a * a)) + first_part))
        return result


def class_disbalance_graphx(graphx):
    signal = []
    for _, node in graphx.nodes(data=True):
        signal.append(node['signal'])
    return list(zip(*np.unique(signal, return_counts=True)))

def class_disbalance_graphx__(graphx):
    signal = []
    for _, node in graphx.nodes(data=True):
        signal.append(node['signal'])
    return np.unique(signal, return_counts=True)

#load data
from opera_tools import combine_mc_bg, gen_graphx, gen_x_y_dataset, load_bg, load_mc

def pmc_to_ship_format(pmc):
    showers = []
    scale = 10000
    for idx in pmc.index:
        shower = pmc.loc[idx]
        
        showers.append(
            {
                'TX': shower['BT_X'] / scale,
                'TY': shower['BT_Y'] / scale,
                'TZ': shower['BT_Z'] / scale,
                'PX': shower['BT_SX'],
                'PY': shower['BT_SY'],
                'PZ': np.ones_like(shower['BT_X']),
                'ele_P': shower['ele_P'],
                'ele_TX': shower['ele_x'] / scale,
                'ele_TY': shower['ele_y'] / scale,
                'ele_TZ': shower['ele_z']  / scale,
                'ele_PX': shower['ele_sx'],
                'ele_PY': shower['ele_sy'],
                'ele_PZ': 1.
            }
        )
    return showers


def main():
    process = psutil.Process(os.getpid())

    pmc = load_mc(filename='mcdata_taue2.root', step=1)

    selected_showers = pmc_to_ship_format(pmc)

    selected_showers = [selected_shower for selected_shower in selected_showers if len(selected_shower['PX']) > 70]
    selected_showers = [selected_shower for selected_shower in selected_showers if len(selected_shower['PX']) < 3000]
    bricks = []
    NUM_SHOWERS_IN_BRICK = 200

    scale = 10000
    bricks = []
    for i in range(len(selected_showers) // NUM_SHOWERS_IN_BRICK):
        node_id = 0
        graphx = nx.DiGraph()
        nodes_to_add = []
        showers_data = []
        for j in range(NUM_SHOWERS_IN_BRICK):
            selected_shower = selected_showers[i * NUM_SHOWERS_IN_BRICK + j]
            showers_data.append(
                {
                'numtracks': len(selected_shower['PX']),
                'signal': j,
                'ele_P': selected_shower['ele_P'],
                'ele_SX': selected_shower['ele_TX'] * scale,
                'ele_SY': selected_shower['ele_TY'] * scale,
                'ele_SZ': selected_shower['ele_TZ'] * scale,
                'ele_TX': selected_shower['ele_PX'] / selected_shower['ele_PZ'],
                'ele_TY': selected_shower['ele_PY'] / selected_shower['ele_PZ']
                }
            )
            for k in range(len(selected_shower['PX'])):
                nodes_to_add.append(
                    (
                        node_id,
                        {
                            'features': {
                                'SX': selected_shower['TX'][k] * scale,
                                'SY': selected_shower['TY'][k] * scale,
                                'SZ': selected_shower['TZ'][k] * scale,
                                'TX': selected_shower['PX'][k] / selected_shower['PZ'][k],
                                'TY': selected_shower['PY'][k] / selected_shower['PZ'][k],
                            },
                            'signal': j
                        }
                    )
                )
                node_id += 1
        graphx.add_nodes_from(nodes_to_add)
        graphx.graph['showers_data'] = showers_data
        bricks.append(graphx)


    print(len(bricks))

    def run_gen_graphx(graphx, layers=2, threshold=250):
        graphx_nodes = list(graphx.nodes(data=True))
        edges = list(graphx.edges())
        graphx.remove_edges_from(edges)
        ebunch = generate_distances(graphx_nodes, layers=layers, threshold=threshold)
        graphx.add_edges_from(ebunch)
        return graphx, ebunch

    #save graphx for clusterization saparetly 

    def brick_to_graph(data):

        graphx, ebunch = run_gen_graphx(data, layers=5, threshold=400)

        x = torch.FloatTensor(pd.DataFrame([graphx.nodes(data = True)[i]['features'] 
                                            for i in range(len(graphx.nodes))]).values)    

        edges_from = [ebunch[i][0] for i in range(len(ebunch))]
        edge_to = [ebunch[i][1] for i in range(len(ebunch))]
        edges = np.vstack([edges_from, edge_to])
        edge_index = torch.LongTensor(edges)

        shower_data = torch.FloatTensor(pd.DataFrame(graphx.graph['showers_data']).values)
        #numtracks	signal	ele_P	ele_SX	ele_SY	ele_SZ	ele_TX	ele_TY

        dist = [ebunch[i][2]['weight'] for i in range(len(ebunch))]
        edge_attr = torch.log(torch.FloatTensor(dist).view(-1, 1))

        y = torch.LongTensor([graphx.nodes(data = True)[i]['signal'] for i in range(len(graphx.nodes))])

        shower = Data(
            x=x,
            edge_index=edge_index,
            shower_data=shower_data,
            pos=x,
            edge_attr=edge_attr,
            y=y
        )

        return graphx, shower

    showers = []
    graphs_for_clusterization = []

    for i in tqdm(range(len(bricks))):
        graphx, shower = brick_to_graph(bricks[i])
        graphs_for_clusterization.append(graphx)
        showers.append(shower)


    output_file='./graphx.pt'
    torch.save(showers, output_file)

    output_file='./graph_for_clustering.pt'
    torch.save(graphs_for_clusterization, output_file)

if __name__ == "__main__":
    main()

