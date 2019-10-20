import comet_ml
from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import click
from nets import GraphNN_KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, average_precision_score
from torch_geometric.data import DataLoader
from preprocessing import preprocess_dataset
from utils import RunningAverageMeter, plot_aucs
from tqdm import tqdm
import networkx as nx
from hdbscan import run_hdbscan_on_brick, run_hdbscan
import clustering_metrics
from clustering_metrics import class_disbalance_graphx, class_disbalance_graphx__
from clustering_metrics import estimate_e, estimate_start_xyz, estimate_txty
from sklearn.linear_model import TheilSenRegressor, LinearRegression, HuberRegressor


def predict_one_shower(shower, graph_embedder, edge_classifier):
    embeddings = graph_embedder(shower)
    edge_labels_true = (shower.y[shower.edge_index[0]] == shower.y[shower.edge_index[1]]).view(-1)
    edge_data = torch.cat([
        embeddings[shower.edge_index[0]],
        embeddings[shower.edge_index[1]]
    ], dim=1)
    edge_labels_predicted = edge_classifier(edge_data).view(-1)

    return edge_labels_true, edge_labels_predicted


def preprocess_torch_shower_to_nx(shower, graph_embedder, edge_classifier, threshold=0.5):
    node_id = 0
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
    _, weights = predict_one_shower(shower, graph_embedder=graph_embedder, edge_classifier=edge_classifier)
    weights = weights.detach().cpu().numpy()
    edge_index = shower.edge_index.t().detach().cpu().numpy()
    edge_index = edge_index[weights > threshold]
    weights = weights[weights > threshold]
    weights = -np.log(weights)  # TODO: which transformation to use?
    print(len(weights))
    for k, (p0, p1) in enumerate(edge_index):
        edges_to_add.append((p0, p1, weights[k]))

    G.add_nodes_from(nodes_to_add)
    G.add_weighted_edges_from(edges_to_add)

    G.graph['showers_data'] = showers_data
    return G


def calc_clustering_metrics(clusterized_bricks, experiment):
    selected_tracks = 0
    total_tracks = 0

    number_of_lost_showers = 0
    number_of_broken_showers = 0
    number_of_stucked_showers = 0
    total_number_of_showers = 0
    number_of_good_showers = 0
    number_of_survived_showers = 0
    second_to_first_ratios = []

    E_raw = []
    E_true = []

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
    for clusterized_brick in clusterized_bricks:
        showers_data = clusterized_brick['graphx'].graph['showers_data']
        clusters = clusterized_brick['clusters']
        for shower_data in showers_data:
            shower_data['clusters'] = []

        for cluster in clusters:
            print(class_disbalance_graphx(cluster))
            selected_tracks += len(cluster)
            for label, label_count in class_disbalance_graphx(cluster):
                if label_count / showers_data[label]['numtracks'] >= 0.1:
                    showers_data[label]['clusters'].append(cluster)

        for shower_data in showers_data:
            total_tracks += shower_data['numtracks']

        for shower_data in showers_data:
            total_number_of_showers += 1

            signals_per_cluster = []
            idx_cluster = []
            for i, cluster in enumerate(shower_data['clusters']):
                labels, counts = class_disbalance_graphx__(cluster)
                signals_per_cluster.append(counts[labels == shower_data['signal']][0])
                idx_cluster.append(i)
            signals_per_cluster = np.array(signals_per_cluster)
            idx_cluster = np.array(idx_cluster)
            second_to_first_ratio = 0.

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
            x, y, z = estimate_start_xyz(cluster)

            x_raw.append(x)
            x_true.append(shower_data['ele_SX'])

            y_raw.append(y)
            y_true.append(shower_data['ele_SY'])

            z_raw.append(z)
            z_true.append(shower_data['ele_SZ'])

            # tx, ty
            tx, ty = estimate_txty(cluster)

            tx_raw.append(tx)
            tx_true.append(shower_data['ele_TX'])

            ty_raw.append(ty)
            ty_true.append(shower_data['ele_TY'])

    E_raw = np.array(E_raw)
    E_true = np.array(E_true)

    x_raw = np.array(x_raw)
    x_true = np.array(x_true)

    y_raw = np.array(y_raw)
    y_true = np.array(y_true)

    z_raw = np.array(z_raw)
    z_true = np.array(z_true)

    tx_raw = np.array(tx_raw)
    tx_true = np.array(tx_true)

    ty_raw = np.array(ty_raw)
    ty_true = np.array(ty_true)

    r = HuberRegressor()
    r.fit(X=E_raw.reshape((-1, 1)), y=E_true, sample_weight=1 / E_true)
    E_pred = r.predict(E_raw.reshape((-1, 1)))

    scale_mm = 10000
    print('Energy resolution = {}'.format(np.std((E_true - E_pred) / E_true)))
    print()
    print('Track efficiency = {}'.format(selected_tracks / total_tracks))
    print()
    print('Good showers = {}'.format(number_of_good_showers / total_number_of_showers))
    print('Stuck showers = {}'.format(number_of_stucked_showers / total_number_of_showers))
    print('Broken showers = {}'.format(number_of_broken_showers / total_number_of_showers))
    print('Lost showers = {}'.format(number_of_lost_showers / total_number_of_showers))
    print()
    print('MAE for x = {}'.format(np.abs((x_raw * scale_mm - x_true) / scale_mm).mean()))
    print('MAE for y = {}'.format(np.abs((y_raw * scale_mm - y_true) / scale_mm).mean()))
    print('MAE for z = {}'.format(np.abs((z_raw * scale_mm - z_true) / scale_mm).mean()))
    print()
    print('MAE for tx = {}'.format(np.abs((tx_raw - tx_true)).mean()))
    print('MAE for ty = {}'.format(np.abs((ty_raw - ty_true)).mean()))

    experiment.log_metric('Energy resolution', (np.std((E_true - E_pred) / E_true)))
    print()
    experiment.log_metric('Track efficiency', (selected_tracks / total_tracks))
    print()
    experiment.log_metric('Good showers', (number_of_good_showers / total_number_of_showers))
    experiment.log_metric('Stuck showers', (number_of_stucked_showers / total_number_of_showers))
    experiment.log_metric('Broken showers', (number_of_broken_showers / total_number_of_showers))
    experiment.log_metric('Lost showers', (number_of_lost_showers / total_number_of_showers))
    print()
    experiment.log_metric('MAE for x', (np.abs((x_raw * scale_mm - x_true) / scale_mm).mean()))
    experiment.log_metric('MAE for y', (np.abs((y_raw * scale_mm - y_true) / scale_mm).mean()))
    experiment.log_metric('MAE for z', (np.abs((z_raw * scale_mm - z_true) / scale_mm).mean()))
    print()
    experiment.log_metric('MAE for tx', (np.abs((tx_raw - tx_true)).mean()))
    experiment.log_metric('MAE for ty', (np.abs((ty_raw - ty_true)).mean()))


@click.command()
@click.option('--datafile', type=str, default='./data/train_.pt')
@click.option('--project_name', type=str, prompt='Enter project name', default='em_showers_clustering')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--dim_out', type=int, default=144)
@click.option('--min_cl', type=int, default=40)
@click.option('--cl_size', type=int, default=40)
@click.option('--threshold', type=float, default=0.9)
def main(
        datafile='./data/train_.pt',
        dim_out=144, min_cl=40, cl_size=40,
        threshold=0.9,
        project_name='em_showers_clustering',
        work_space='schattengenie'
):
    experiment = Experiment(project_name=project_name, workspace=work_space)
    device = torch.device('cpu')
    showers = preprocess_dataset(datafile)

    k = showers[0].x.shape[1]
    print(k)
    graph_embedder = GraphNN_KNN(dim_out=dim_out, k=k).to(device)
    edge_classifier = nn.Sequential(nn.Linear(dim_out * 2, 144),
                                    nn.Tanh(),
                                    nn.Linear(144, 144),
                                    nn.Tanh(),
                                    nn.Linear(144, 32),
                                    nn.Tanh(),
                                    nn.Linear(32, 1),
                                    nn.Sigmoid()).to(device)

    graph_embedder.load_state_dict(torch.load('graph_embedder.pt', map_location=device))
    graph_embedder.eval()
    edge_classifier.load_state_dict(torch.load('edge_classifier.pt', map_location=device))
    edge_classifier.eval()

    clusterized_bricks = []
    for shower in showers:
        G = preprocess_torch_shower_to_nx(shower,
                                          graph_embedder=graph_embedder,
                                          edge_classifier=edge_classifier,
                                          threshold=threshold)
        graphx, clusters, roots = run_hdbscan_on_brick(G, min_cl=min_cl, cl_size=cl_size)
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

    calc_clustering_metrics(clusterized_bricks, experiment=experiment)


if __name__ == "__main__":
    main()
