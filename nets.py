import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Sequential
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, scatter_
import torch_geometric.transforms as T
import torch_cluster
from torch_geometric.nn import NNConv, GCNConv, GraphConv
from torch_geometric.nn import PointConv, EdgeConv, SplineConv
from torch.utils.checkpoint import checkpoint
from functools import partial
import numpy as np

from torch_geometric.nn.inits import reset

RESIDUALS = False


def extract_subgraph(h, adj, edge_features, order):
    adj_selected = adj[:, order]
    edge_features_selected = edge_features[order, :]
    nodes_selected = adj_selected.unique()
    h_selected = h[nodes_selected]
    nodes_selected_new = torch.arange(len(nodes_selected))
    dictionary = dict(zip(nodes_selected.cpu().numpy(), nodes_selected_new.cpu().numpy()))
    adj_selected_new = torch.tensor(np.vectorize(dictionary.get)(adj_selected.cpu().numpy())).long().to(adj)
    return h_selected, nodes_selected, edge_features_selected, adj_selected_new


class EdgeConv_new(MessagePassing):   
    def __init__(self, nn, aggr='max', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()


    def reset_parameters(self):
        reset(self.nn)


    def forward(self, x, edge_index, dist):
        print(x.shape, edge_index.shape, dist.shape)
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
      
        return self.propagate(edge_index, x=x, dist=dist)


    def message(self, x_i, x_j, dist):
        print(x_i.shape, len(dist))
        return self.nn(torch.cat([x_i, x_j - x_i, dist], dim=1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)      


class EmulsionConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=1, direction=0):
        super().__init__(aggr='add')
        self.mp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_dim, out_channels),
            nn.ReLU()
        )
        self._direction = direction  # TODO: defines direction

    def forward(self, x, edge_index, orders, edge_features, orders_preprocessed):
        x = x.clone()
        for i, order in enumerate(orders):
            if order.sum():
                nodes_selected, adj_selected_new = orders_preprocessed[i]
                nodes_selected = nodes_selected.to(x).long()
                adj_selected_new = adj_selected_new.to(x).long()
                x_selected = x[nodes_selected]
                edge_features_selected = edge_features[order, :]
                x_selected = self.message(
                    x_j=x_selected[adj_selected_new[0]],
                    x_i=x_selected[adj_selected_new[1]],
                    edge_features=edge_features_selected
                )
                x_selected = scatter_('add', x_selected, adj_selected_new[self._direction],
                                      dim=0, dim_size=len(nodes_selected))
                x[nodes_selected] = (x[nodes_selected] + x_selected) / 2.
        return x

    def message(self, x_j, x_i, edge_features):
        return self.mp(torch.cat([x_i, x_j - x_i, edge_features], dim=1))

    def update(self, aggr_out, x):
        return aggr_out + x


def init_bias_model(model, b: float):
    for module in model.modules():
        if hasattr(module, 'bias'):
            module.bias.data.fill_(b)


class GraphNN_KNN_v1(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 edge_dim=1,
                 output_dim=10,
                 num_layers_emulsion=3,
                 num_layers_edge_conv=3,
                 bias_init=0., **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        previous_dim = input_dim
        self._layers = nn.ModuleList()
        for i in range(num_layers_emulsion):
            self._layers.append(
                nn.Sequential(nn.Linear(previous_dim, self.hidden_dim), nn.ReLU())
            )
            self._layers.append(
                EmulsionConv(self.hidden_dim, self.hidden_dim, edge_dim=edge_dim)
            )
            previous_dim = self.hidden_dim

        for i in range(num_layers_edge_conv):
            if num_layers_emulsion == 0 and i == 0:
                self._layers.append(
                    nn.Sequential(nn.Linear(previous_dim, self.hidden_dim), nn.ReLU())
                )
            self._layers.append(
                EdgeConv(Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ReLU()), 'max')
            )

        self.output = nn.Linear(self.hidden_dim, output_dim)
        init_bias_model(self, b=0.)

    def forward(self, data):
        x, edge_index, orders, edge_features = data.x, data.edge_index, data.orders, data.edge_features
        orders_preprocessed = data.orders_preprocessed[0]

        x = self._layers[0](x)

        for layer in self._layers[1:]:
            if isinstance(layer, EmulsionConv):
                layer = partial(layer, orders_preprocessed=orders_preprocessed)
                x = checkpoint(layer,
                               x,
                               edge_index,
                               orders,
                               edge_features)
            elif isinstance(layer, EdgeConv):
                x = checkpoint(layer,
                               x,
                               edge_index)
            else:
                x = layer(x)
        return self.output(x)


class EdgeClassifier_v1(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, prior_proba=1. - 0.03, **kwargs):
        super().__init__()
        self._layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ])
        init_bias_model(self, b=0.)
        init_bias_model(self._layers[-2], b=-np.log((1 - prior_proba) / prior_proba))

    def forward(self, shower, embeddings, edge_index):
        embeddings = torch.cat([
            embeddings[edge_index[0]],
            embeddings[edge_index[1]],
            shower.edge_features
        ], dim=1)
        for layer in self._layers:
            embeddings = checkpoint(layer, embeddings)
        return embeddings


class EdgeDenseClassifier(nn.Module):
    def __init__(self, input_dim=10, **kwargs):
        super(EdgeDenseCl, self).__init__()
        self.edge_classifier = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings, edge_index):
        return self.edge_classifier(torch.cat([embeddings[edge_index[0]], embeddings[edge_index[0]]], 1))


class EdgeDenseClassifierEdgeAttribute(nn.Module):
    def __init__(self, input_dim=10, **kwargs):
        super(EdgeDenseClassifierEdgeAttribute, self).__init__()
        self.edge_classifier = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, shower, embeddings, edge_index):
        x = torch.cat([embeddings[edge_index[0]], embeddings[edge_index[1]]], 1)
        x = torch.cat([x, shower.edge_attr], 1)
        return self.edge_classifier(x)
