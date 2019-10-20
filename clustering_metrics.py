from math import sqrt
import networkx as nx
from sklearn.linear_model import TheilSenRegressor, LinearRegression, HuberRegressor
from copy import deepcopy
from collections import Counter
import numpy as np

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


def estimate_e(cluster, angle=0.05):
    x, y, z = estimate_start_xyz(cluster)
    tx, ty = estimate_txty(cluster)
    n = 0
    for i, node in cluster.nodes(data=True):
        dx = node['features']['SX'] - x
        dy = node['features']['SY'] - y
        dz = node['features']['SZ'] - z
        dx = dx / dz - tx
        dy = dy / dz - ty
        dz = dz / dz
        if sqrt(dx ** 2 + dy ** 2) < angle:
            n += 1

    return n


def estimate_start_xyz(cluster, k=3, shift_x=0., shift_y=0., shift_z=-2000.):
    xs = []
    ys = []
    zs = []

    for i, node in cluster.nodes(data=True):
        xs.append(node['features']['SX'])
        ys.append(node['features']['SY'])
        zs.append(node['features']['SZ'])

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    argosorted_z = np.argsort(zs)

    x = np.median(np.median(xs[argosorted_z][:k])) + shift_x
    y = np.median(np.median(ys[argosorted_z][:k])) + shift_y
    z = np.median(np.median(zs[argosorted_z][:k])) + shift_z

    return x, y, z


def estimate_txty(cluster, k=20):
    xs = []
    ys = []
    zs = []
    tx = []
    ty = []

    for i, node in cluster.nodes(data=True):
        xs.append(node['features']['SX'])
        ys.append(node['features']['SY'])
        zs.append(node['features']['SZ'])
        tx.append(node['features']['TX'])
        ty.append(node['features']['TY'])

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    tx = np.array(tx)
    ty = np.array(ty)

    argosorted_z = np.argsort(zs)

    lr = TheilSenRegressor()
    lr.fit(zs[argosorted_z][:k].reshape((-1, 1)), xs[argosorted_z][:k])
    TX = lr.coef_[0]

    lr.fit(zs[argosorted_z][:k].reshape((-1, 1)), ys[argosorted_z][:k])
    TY = lr.coef_[0]

    return TX, TY
    # return np.median(np.median(tx[argosorted_z][:k])), np.median(np.median(ty[argosorted_z][:k]))