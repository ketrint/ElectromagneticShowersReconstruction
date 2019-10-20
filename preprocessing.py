import numpy as np
import constants
import torch


def round_Z_coodr(x):
    return constants.Z_centered[np.argmin(np.abs(constants.Z_centered - x))]
round_Z_coodr = np.vectorize(round_Z_coodr)


def create_mask(data):
    z_rounded = round_Z_coodr(data.x[:, 2].detach().cpu().numpy() * np.array([1e4]))
    orders = np.zeros((len(constants.Z_centered), data.edge_index.shape[1]))
    edge_index_0 = data.edge_index[0].detach().cpu().numpy()
    idx = np.arange(len(z_rounded))
    for i, z_i in enumerate(constants.Z_centered):
        orders[i][np.in1d(edge_index_0, idx[z_rounded == z_i])] = 1
    return orders.astype(np.uint8)


def add_new_features(shower):
    data = shower.x
    shape_0 = data.shape[0]
    # azimuthal_angle
    feat_0 = torch.atan(data[:, 1] / (data[:, 0] + 0.00001)).view(shape_0, 1)
    feat_1 = (torch.sqrt(torch.pow(data[:, 1], 2) + torch.pow(data[:, 0], 2))
              / (data[:, 2] + 0.00001)).view(shape_0, 1)
    feat_2 = (data[:, 0] / (data[:, 2] + 0.00001)).view(shape_0, 1)
    feat_3 = (data[:, 1] / (data[:, 2] + 0.00001)).view(shape_0, 1)
    feat_4 = (torch.sin(feat_0) + torch.cos(feat_0)) / (feat_0 + 0.00001)
    shower.x = torch.cat([data, feat_0, feat_1, feat_2, feat_3, feat_4], dim=1)


def preprocess_dataset(datafile):
    showers = list(torch.load(datafile))
    for i in range(len(showers)):
        showers[i].mask = torch.tensor(create_mask(showers[i])).bool()
        add_new_features(showers[i])
    return showers
