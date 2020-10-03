import comet_ml
from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import click
from nets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, average_precision_score
from torch_geometric.data import DataLoader
from preprocessing import preprocess_dataset
from viz_utils import RunningAverageMeter, plot_aucs
from tqdm import tqdm
import networkx as nx
from custom_hdbscan import run_hdbscan_on_brick, run_hdbscan
from custom_hdbscan import run_vanilla_hdbscan, run_vanilla_hdbscan_on_brick
import clustering_metrics
from clustering_metrics import class_disbalance_graphx, class_disbalance_graphx__
from clustering_metrics import estimate_e, estimate_start_xyz, estimate_txty
from sklearn.linear_model import TheilSenRegressor, LinearRegression, HuberRegressor
from sklearn import svm
from sys_utils import get_freer_gpu, str_to_class
import itertools
from operator import itemgetter
import sys
from sklearn import svm


def str_to_class(classname: str):
    """
    Function to get class object by its name signature
    :param classname: str
        name of the class
    :return: class object with the same name signature as classname
    """
    return getattr(sys.modules[__name__], classname)


def predict_one_shower(shower, graph_embedder, edge_classifier):
    # TODO: batch training
    embeddings = graph_embedder(shower)
    edge_labels_true = (~(shower.y[shower.edge_index[0]] == shower.y[shower.edge_index[1]])).view(-1)
    edge_labels_predicted = edge_classifier(shower=shower, embeddings=embeddings, edge_index=shower.edge_index).view(-1)
    return edge_labels_true, torch.clamp(edge_labels_predicted, 1e-6, 1 - 1e-6)


def k_nearest_cut_succ(graphx, k):
    for node_id in tqdm(graphx.nodes()):
        successors = list(graphx.successors(node_id))
        if len(successors) <= k:
            continue
        edges = list(itertools.product([node_id], successors))
        weights = []
        for edge in edges:
            weights.append(graphx[edge[0]][edge[1]]['weight'])
        weights, edges = [list(x) for x in zip(*sorted(zip(weights, edges), key=itemgetter(0)))]
        edges_to_remove = edges[k:]
        if len(edges_to_remove):
            graphx.remove_edges_from(edges_to_remove)
    return graphx


def k_nearest_cut_pred(graphx, k):
    for node_id in tqdm(graphx.nodes()):
        predecessors = list(graphx.predecessors(node_id))
        if len(predecessors) <= k:
            continue
        edges = list(itertools.product(predecessors, [node_id]))
        weights = []
        for edge in edges:
            weights.append(graphx[edge[0]][edge[1]]['weight'])
        weights, edges = [list(x) for x in zip(*sorted(zip(weights, edges), key=itemgetter(0)))]

        edges_to_remove = edges[k:]
        if len(edges_to_remove):
            graphx.remove_edges_from(edges_to_remove)
    return graphx


def preprocess_torch_shower_to_nx(shower, graph_embedder, edge_classifier, add_noise=0., threshold=0.5, baseline=False):
    G = nx.DiGraph()
    nodes_to_add = []
    showers_data = []
    y = shower.y.cpu().detach().numpy()
    x = shower.x.cpu().detach().numpy()
    for shower_id in tqdm(np.unique(y)):
        shower_data = shower.shower_data[y == shower_id].unique(dim=0).detach().cpu().numpy()[0]
        showers_data.append(
            {
                'numtracks': shower_data[-2],
                'signal': shower_id,
                'ele_P': shower_data[0],
                'ele_SX': shower_data[1],
                'ele_SY': shower_data[2],
                'ele_SZ': shower_data[3],
                'ele_TX': shower_data[4],
                'ele_TY': shower_data[5]
            }
        )
    node_id = 0
    for k in range(len(y)):
        nodes_to_add.append(
            (
                node_id,
                {
                    'features': {
                        'SX': x[k, 0],
                        'SY': x[k, 1],
                        'SZ': x[k, 2],
                        'TX': x[k, 3],
                        'TY': x[k, 4],
                    },
                    'signal': y[k]
                }
            )
        )
        node_id += 1

    edges_to_add = []
    if not baseline:
        _, weights = predict_one_shower(shower, graph_embedder=graph_embedder, edge_classifier=edge_classifier)
        weights = weights.detach().cpu().numpy()
        weights = -np.log((1 - weights) / weights)
        edge_index = shower.edge_index.t().detach().cpu().numpy()
        # weights = np.percentile(weights, q=90)
        edge_index = edge_index[weights > threshold]
        weights = -weights[weights > threshold]
    else:
        weights = shower.edge_attr.view(-1).detach().cpu().numpy()
        edge_index = shower.edge_index.t().detach().cpu().numpy()
        weights = np.exp(weights)
        edge_index = edge_index[weights < threshold]
        weights = weights[weights < threshold]
        weights = np.random.randn(len(weights)) * np.std(weights) * add_noise + weights

    for k, (p0, p1) in enumerate(edge_index):
        edges_to_add.append((p0, p1, weights[k]))

    G.add_nodes_from(nodes_to_add)
    G.add_weighted_edges_from(edges_to_add)

    G.graph['showers_data'] = showers_data
    return G


def calc_clustering_metrics(clusterized_bricks, experiment, energy_true_file, energy_file):
    selected_tracks = 0
    total_tracks = 0


    number_of_lost_showers = 0
    number_of_broken_showers = 0
    number_of_stucked_showers = 0
    total_number_of_showers = 0
    number_of_good_showers = 0
    second_to_first_ratios = []

    ER_one = []
    
    E_raw = []
    E_true = []
    E_true_all = []
    E = []

    x_raw = []
    x_true = []

    y_raw = []
    y_true = []

    z_raw = []
    z_true = []

    tx_raw = []
    tx_true = []

    ty_raw = []
    ty_true = []
    n_showers = []
    for clusterized_brick in clusterized_bricks:
        

        showers_data = clusterized_brick['graphx'].graph['showers_data']
        clusters = clusterized_brick['clusters']
        for shower_data in showers_data:
            shower_data['clusters'] = []

        for cluster in clusters:
            selected_tracks += len(cluster)
            for label, label_count in class_disbalance_graphx(cluster):
                print('n showers:', len(showers_data))
                print(label)
                if label_count / showers_data[label]['numtracks'] >= 0.1:
                    showers_data[label]['clusters'].append(cluster)

        for shower_data in showers_data:
            total_tracks += shower_data['numtracks']
            E.append(shower_data['numtracks'])
            E_true_all.append(shower_data['ele_P'])
            n_showers.append(len(showers_data))

        for shower_data in showers_data:
            total_number_of_showers += 1

            signals_per_cluster = []
            idx_cluster = []
            for i, cluster in enumerate(shower_data['clusters']):
                labels, counts = class_disbalance_graphx__(cluster)
                signals_per_cluster.append(counts[labels == shower_data['signal']][0])
                idx_cluster.append(i)
            signals_per_cluster = np.array(signals_per_cluster)

            if len(signals_per_cluster) == 0:
                number_of_lost_showers += 1
                continue
            if len(signals_per_cluster) == 1:
                second_to_first_ratio = 0.
                second_to_first_ratios.append(second_to_first_ratio)
            else:
                second_to_first_ratio = np.sort(signals_per_cluster)[-2] / signals_per_cluster.max()
                second_to_first_ratios.append(second_to_first_ratio)

            cluster = shower_data['clusters'][np.argmax(signals_per_cluster)]

            # not enough signal
            if (signals_per_cluster.max() / shower_data['numtracks']) <= 0.1:
                continue

            labels, counts = class_disbalance_graphx__(cluster)
            counts = counts / counts.sum()
            # high contamination
            if counts[labels == shower_data['signal']] < 0.9:
                number_of_stucked_showers += 1
                continue

            if second_to_first_ratio > 0.3:
                number_of_broken_showers += 1
                continue

            # for good showers
            number_of_good_showers += 1
            # E
            E_raw.append(estimate_e(cluster))
           
            E_true.append(shower_data['ele_P'])
            
            # x, y, z
            #x, y, z = estimate_start_xyz(cluster)

           # x_raw.append(x)
            #x_true.append(shower_data['ele_SX'])

           # y_raw.append(y)
           # y_true.append(shower_data['ele_SY'])

          #  z_raw.append(z)
           # z_true.append(shower_data['ele_SZ'])

            # tx, ty
           # tx, ty = estimate_txty(cluster)

            #tx_raw.append(tx)
            #tx_true.append(shower_data['ele_TX'])

           # ty_raw.append(ty)
         #  ty_true.append(shower_data['ele_TY'])
    # E_raw = np.array(E_raw)
    # np.save(energy_file, E_raw)
    # E_true = np.array(E_true)
    #np.save(energy_true_file, E_true)
    	
        r = HuberRegressor()
        r.fit(X=np.array(E_raw).reshape((-1, 1)), y=E_true)
        
        E_pred = r.predict(np.array(E_raw).reshape((-1, 1)))

        ER_one.append(np.std((np.array(E_true) - E_pred) / np.array(E_true)))
       

    n_showers = np.array(n_showers)
    np.save('n_showers.npy',n_showers)
    E = np.array(E)
    np.save('E.npy',E)
    ER_one = np.array(ER_one)
    np.save('ER.npy', ER_one)
    E_true_all = np.array(E_true_all)
    np.save('E_true_all.npy', E_true_all)
    #experiment.log_metric('Energy resolution', (ER_one.mean()))
    #experiment.log_metric('Energy resolution STD', (ER_one.std()))
    experiment.log_metric('Energy resolution', (np.std((E_true - E_pred) / E_true)))
    print('Energy resolution = {}'.format(np.std((E_true - E_pred) / E_true)))

    experiment.log_metric('Good showers', (number_of_good_showers / total_number_of_showers))
    print('Good showers = {}'.format(number_of_good_showers / total_number_of_showers))
   # E_raw = np.array(E_raw)
   # E_true = np.array(E_true)

   # x_raw = np.array(x_raw)
    #x_true = np.array(x_true)

   # y_raw = np.array(y_raw)
    #y_true = np.array(y_true)

   # z_raw = np.array(z_raw)
    #z_true = np.array(z_true)

   # tx_raw = np.array(tx_raw)
    #tx_true = np.array(tx_true)

    #ty_raw = np.array(ty_raw)
    #ty_true = np.array(ty_true)

   # r = svm.LinearSVR()
   # r.fit(X=E_raw.reshape((-1, 1)), y=E_true)
  #  E_pred = r.predict(E_raw.reshape((-1, 1)))

    scale_mm = 10000
   # print('Energy resolution = {}'.format(np.std((E_true - E_pred) / E_true)))
    print()
   # print('Track efficiency = {}'.format(selected_tracks / total_tracks))
  #  print()
 #   print('Good showers = {}'.format(number_of_good_showers / total_number_of_showers))
 
    #experiment.log_metric('Energy resolution', (np.std((E_true - E_pred) / E_true)))
    print()
   # experiment.log_metric('Track efficiency', (selected_tracks / total_tracks))
    print()
    #experiment.log_metric('Good showers', (number_of_good_showers / total_number_of_showers))
   # experiment.log_metric('Stuck showers', (number_of_stucked_showers / total_number_of_showers))
  #  experiment.log_metric('Broken showers', (number_of_broken_showers / total_number_of_showers))
  #  experiment.log_metric('Lost showers', (number_of_lost_showers / total_number_of_showers))
    print()
    #experiment.log_metric('MAE for x', (np.abs((x_raw * scale_mm - x_true) / scale_mm).mean()))
   # experiment.log_metric('MAE for y', (np.abs((y_raw * scale_mm - y_true) / scale_mm).mean()))
  #  experiment.log_metric('MAE for z', (np.abs((z_raw * scale_mm - z_true) / scale_mm).mean()))
    print()
   # experiment.log_metric('MAE for tx', (np.abs((tx_raw - tx_true)).mean()))
  #  experiment.log_metric('MAE for ty', (np.abs((ty_raw - ty_true)).mean()))



@click.command()
@click.option('--energy_file', type=str, default='./data/E_pred.pt')
@click.option('--energy_true_file', type=str, default='./data/E_true.pt')
@click.option('--datafile', type=str, default='./data/train_200_preprocessed.pt')
@click.option('--project_name', type=str, prompt='Enter project name', default='em_showers_clustering')
@click.option('--workspace', type=str, prompt='Enter workspace name')
@click.option('--min_cl', type=int, default=40)
@click.option('--min_samples_core', type=int, default=4)
@click.option('--cl_size', type=int, default=40)
@click.option('--add_noise', type=float, default=0.)
@click.option('--threshold', type=float, default=0.5)
@click.option('--vanilla_hdbscan', type=bool, default=False)
@click.option('--baseline', type=bool, default=False)
@click.option('--hidden_dim', type=int, default=32)
@click.option('--output_dim', type=int, default=32)
@click.option('--num_layers_emulsion', type=int, default=3)
@click.option('--num_layers_edge_conv', type=int, default=5)
@click.option('--graph_embedder', type=str, default='GraphNN_KNN_v1')
@click.option('--edge_classifier', type=str, default='EdgeClassifier_v1')
@click.option('--graph_embedder_weights', type=str, default='GraphNN_KNN_v1')
@click.option('--edge_classifier_weights', type=str, default='EdgeClassifier_v1')
def main(
        datafile='./data/train_200_preprocessed.pt',
        energy_file='./data/E_pred.npy',
        energy_true_file='./data/E_true.npy',
        hidden_dim=12,
        output_dim=12,
        num_layers_emulsion=3,
        num_layers_edge_conv=3,
        cl_size=40,
        min_cl=40,
        min_samples_core=5,
        vanilla_hdbscan=False,
        threshold=0.9,
        add_noise=0.,
        project_name='em_showers_clustering',
        workspace='schattengenie',
        graph_embedder='GraphNN_KNN_v1',
        edge_classifier='EdgeClassifier_v1',
        baseline=False,
        graph_embedder_weights='GraphNN_KNN_v1', 
        edge_classifier_weights='EdgeClassifier_v1'
):
    experiment = Experiment(project_name=project_name, workspace=workspace)
    #, offline_directory="/home/vbelavin/comet_ml_offline")
    device = torch.device('cpu')
    showers = torch.load(datafile)
    input_dim = showers[0].x.shape[1]
    edge_dim = showers[0].edge_features.shape[1]
    showers = DataLoader(showers, batch_size=1, shuffle=False)
    graph_embedder = str_to_class(graph_embedder)(
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        edge_dim=edge_dim,
        num_layers_emulsion=num_layers_emulsion,
        num_layers_edge_conv=num_layers_edge_conv,
        input_dim=input_dim,
    ).to(device)
    edge_classifier = str_to_class(edge_classifier)(
        input_dim=2 * output_dim + edge_dim,
    ).to(device)
    if not baseline:
        graph_embedder.load_state_dict(torch.load(graph_embedder_weights, map_location=device))
        edge_classifier.load_state_dict(torch.load(edge_classifier_weights, map_location=device))
    graph_embedder.eval()
    edge_classifier.eval()

    clusterized_bricks = []
    for shower in showers:
        G = preprocess_torch_shower_to_nx(
            shower,
            graph_embedder=graph_embedder,
            edge_classifier=edge_classifier,
            threshold=threshold,
            add_noise=add_noise,
            baseline=baseline
        )
        k_nearest_cut_succ(G, 25)
        k_nearest_cut_pred(G, 25)
        if vanilla_hdbscan:
            graphx, clusters, roots = run_vanilla_hdbscan_on_brick(G, min_cl=min_cl, cl_size=cl_size, min_samples_core=min_samples_core)
        else:
            graphx, clusters, roots = run_hdbscan_on_brick(G, min_cl=min_cl, cl_size=cl_size, min_samples_core=min_samples_core)
        clusters_graphx = []
        for cluster in clusters:
            clusters_graphx.append(
                nx.DiGraph(graphx.subgraph(cluster.nodes))
            )
        clusterized_brick = {
            'graphx': graphx,
            'clusters': clusters_graphx,
        }
        clusterized_bricks.append(clusterized_brick)

    calc_clustering_metrics(clusterized_bricks, experiment=experiment, energy_true_file=energy_true_file, energy_file=energy_file)


if __name__ == "__main__":
    main()
