from math import sqrt
import networkx as nx
from sklearn.linear_model import TheilSenRegressor, LinearRegression, HuberRegressor
from copy import deepcopy
from collections import Counter
import numpy as np

def class_disbalance(cluster):
    signal = []
    for node in cluster:
        signal.append(node['signal'])
    return list(zip(*np.unique(signal, return_counts=True)))

def class_disbalance__(cluster):
    signal = []
    for node in cluster:
        signal.append(node['signal'])
    return np.unique(signal, return_counts=True)



def estimate_start_xyz(cluster, k=3, shift_x=0., shift_y=0., shift_z=-2000.):
    xs = []
    ys = []
    zs = []

    for i in range(len(cluster)):
        xs.append(cluster[i]['features']['SX'])
        ys.append(cluster[i]['features']['SY'])
        zs.append(cluster[i]['features']['SZ'])

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

    for i in range(len(cluster)):
        xs.append(cluster[i]['features']['SX'])
        ys.append(cluster[i]['features']['SY'])
        zs.append(cluster[i]['features']['SZ'])
        tx.append(cluster[i]['features']['TX'])
        ty.append(cluster[i]['features']['TY'])

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