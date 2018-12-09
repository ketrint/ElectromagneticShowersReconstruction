import pandas as pd
import numpy as np
import numpy.linalg as la
import numpy
import copy
import networkx as nx
from tqdm import tqdm
import uproot


BT_Z_unique = np.array([     0.,   1293.,   2586.,   3879.,   5172.,   6465.,   7758.,
                          9051.,  10344.,  11637.,  12930.,  14223.,  15516.,  16809.,
                         18102.,  19395.,  20688.,  21981.,  23274.,  24567.,  25860.,
                         27153.,  28446.,  29739.,  31032.,  32325.,  33618.,  34911.,
                         36204.,  37497.,  38790.,  40083.,  41376.,  42669.,  43962.,
                         45255.,  46548.,  47841.,  49134.,  50427.,  51720.,  53013.,
                         54306.,  55599.,  56892.,  58185.,  59478.,  60771.,  62064.,
                         63357.,  64650.,  65943.,  67236.,  68529.,  69822.,  71115.,
                         72408.,  73701.])


BRICK_X_MIN = 0.
BRICK_X_MAX = 103000. # 10.3 cm
BRICK_Y_MIN = 0.
BRICK_Y_MAX = 128000. # 12.8 cm
SAFE_M = 3000.
dZ = 205. # 0.0205 cm emulsion
DISTANCE = 1293.

kwargs = {'bins': 100, 'alpha': 0.8, 'normed': True}

import pandas as pd
def load_mc(filename="mcdata_taue2.root", step=1):
    f = uproot.open(filename)
    mc = f['Data'].pandas.df(["Event_id", "ele_P", "BT_X", "BT_Y",
                              "BT_Z","BT_SX", "BT_SY","ele_x", 
                              "ele_y", "ele_z", "ele_sx", "ele_sy", "chisquare", ], flatten=False)
    pmc = pd.DataFrame(mc)
    pmc['numtracks'] = pmc.BT_X.apply(lambda x: len(x))
    # cuts
    shapechange = [pmc.shape[0]]
    pmc = pmc[pmc.ele_P > 0.1]
    shapechange.append(pmc.shape[0])

    pmc = pmc[pmc.ele_z < 0]
    shapechange.append(pmc.shape[0])

    pmc = pmc[pmc.numtracks > 3]
    shapechange.append(pmc.shape[0])
    print("numtracks reduction by cuts: ", shapechange)
    pmc['m_BT_X'] = pmc.BT_X.apply(lambda x: x.mean())
    pmc['m_BT_Y'] = pmc.BT_Y.apply(lambda x: x.mean())
    pmc['m_BT_Z'] = pmc.BT_Z.apply(lambda x: x.mean())

    print("len(pmc): {len}".format(len=len(pmc)))
    return pmc


# numpy-vectorized function for rounding Z-coordinate
def round_Z_coodr(x):
    return BT_Z_unique[np.argmin(np.abs(BT_Z_unique - x))]


round_Z_coodr = np.vectorize(round_Z_coodr)


def angle(v1, v2):
    cos = (v1*v2).sum(axis=1)
    sin = la.norm(numpy.cross(v1, v2, axis=1), axis=1)
    return numpy.arctan2(sin, cos)


def plot_dataframe(data: pd.DataFrame, azim=-84, elev=10):
    """
    Function for plotting shower
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import matplotlib.pyplot as plt
    x0, y0, z0 = data.SX.values, data.SY.values, data.SZ.values
    sx, sy = data.TX.values, data.TY.values

    x1 = x0 + dZ * sx
    y1 = y0 + dZ * sy
    z1 = z0 + dZ
    
    start_points = np.array([z0, y0, x0]).T.reshape(-1, 3)
    end_points = np.array([z1, y1, x1]).T.reshape(-1, 3)

    C = plt.cm.Blues(0.9)
    lc = Line3DCollection(list(zip(start_points, end_points)), colors=C, alpha=0.9, lw=2)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    ax.view_init(azim=azim, elev=elev)
    ax.add_collection3d(lc)
    
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_zlabel("x") 
    ax.set_xlim(z0.min(), z1.max())
    ax.set_ylim(y0.min(), y1.max())
    ax.set_zlim(x0.min(), x1.max())
    
    plt.show()

def plot_graphx(graphx: nx.DiGraph, azim=-84, elev=10):
    """
    Function for plotting shower
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import matplotlib.pyplot as plt
    x0, y0, z0 = [], [], []
    sx, sy = [], []
    for _, node in graphx.nodes(data=True):
        x0.append(node['features']['SX'])
        y0.append(node['features']['SY'])
        z0.append(node['features']['SZ'])
        sx.append(node['features']['TX'])
        sy.append(node['features']['TY'])
        
    x0, y0, z0 = np.array(x0), np.array(y0), np.array(z0)
    sx, sy = np.array(sx), np.array(sy)

    x1 = x0 + dZ * sx
    y1 = y0 + dZ * sy
    z1 = z0 + dZ
    
    start_points = np.array([z0, y0, x0]).T.reshape(-1, 3)
    end_points = np.array([z1, y1, x1]).T.reshape(-1, 3)

    C = plt.cm.Blues(0.9)
    lc = Line3DCollection(list(zip(start_points, end_points)), colors=C, alpha=0.9, lw=2)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    ax.view_init(azim=azim, elev=elev)
    ax.add_collection3d(lc)
    
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_zlabel("x") 
    ax.set_xlim(z0.min(), z1.max())
    ax.set_ylim(y0.min(), y1.max())
    ax.set_zlim(x0.min(), x1.max())
    
    plt.show()

def gen_graphx(train_np, edges, r_max):
    G = nx.DiGraph()
    
    for i in range(len(train_np)): 
        G.add_node(int(train_np[i][0]), features={
            'SX': train_np[i][1],
            'SY': train_np[i][2],
            'SZ': train_np[i][3],
            'TX': train_np[i][4],
            'TY': train_np[i][5],
            'chi2': train_np[i][6]
        }, signal=train_np[i][7])
        
    for i in tqdm(range(len(edges)), leave=False):
        if edges[i][2] >= r_max:
            continue
        u = int(edges[i][0])
        v = int(edges[i][1])
        u_data = G.node[u]['features']
        v_data = G.node[v]['features']
        G.add_edge(u_of_edge=u, 
                   v_of_edge=v, 
                   features={
                       'dsx': (v_data['SX'] - u_data['SX']) / DISTANCE,
                       'dsy': (v_data['SX'] - u_data['SX']) / DISTANCE,
                       'dsz': (v_data['SZ'] - u_data['SZ']) / DISTANCE,
                       'dsxProjLeft': (v_data['SX'] - u_data['SX'] - (v_data['SZ'] - u_data['SZ']) * u_data['TX']) / (v_data['SZ'] - u_data['SZ']),
                       'dsyProjLeft': (v_data['SY'] - u_data['SY'] - (v_data['SZ'] - u_data['SZ']) * u_data['TY']) / (v_data['SZ'] - u_data['SZ']),
                       'dsxProjRight': (v_data['SX'] - u_data['SX'] - (v_data['SZ'] - u_data['SZ']) * v_data['TX']) / (v_data['SZ'] - u_data['SZ']),
                       'dsyProjRight': (v_data['SY'] - u_data['SY'] - (v_data['SZ'] - u_data['SZ']) * v_data['TY']) / (v_data['SZ'] - u_data['SZ']),
                       'r': edges[i][2]
                   })
    return G


from math import exp
from math import sqrt, log, fabs

def scattering_estimation_loss(basetrack_left, basetrack_right):
    if basetrack_right['features']['SZ'] < basetrack_left['features']['SZ']:
        basetrack_left, basetrack_right = basetrack_right, basetrack_left
    
    EPS = 1e-6
    X0 = 5 * 1000 # mm
    Es = 21 # MeV    
    
    alpha_y = 0
    beta_x = 0
    beta_y = 0
    gamma = 0
    
    dz = basetrack_right['features']['SZ'] - basetrack_left['features']['SZ']

    z = sqrt((basetrack_right['features']['SZ'] - basetrack_left['features']['SZ'])**2 + (basetrack_right['features']['SX'] - basetrack_left['features']['SX'])**2 + (basetrack_right['features']['SZ'] - basetrack_left['features']['SZ'])**2)
    theta_x = basetrack_right['features']['TX'] - basetrack_left['features']['TX']
    theta_y = basetrack_right['features']['TY'] - basetrack_left['features']['TY']
    dx = basetrack_right['features']['SX'] - (basetrack_left['features']['SX'] + basetrack_left['features']['TX'] * dz)
    dy = basetrack_right['features']['SY'] - (basetrack_left['features']['SY'] + basetrack_left['features']['TY'] * dz)
    
    
    z_corrected = X0 * (exp(2 * z / X0) - 1)
    
    alpha_x = 2 * theta_x**2 / (Es**2 * (exp(2 * z / X0) - 1))
    alpha_y = 2 * theta_y**2 / (Es**2 * (exp(2 * z / X0) - 1))
    
    beta_x = 24 * dx**2 / ( Es**2 * X0**3 * (exp(2 * z / X0) - 1)**3)
    beta_y = 24 * dy**2 / ( Es**2 * X0**3 * (exp(2 * z / X0) - 1)**3)
    
    gamma = 2 * (theta_x**2 + theta_y**2)  / (Es**2 * (exp(2 * z / X0) - 1))
    
    E = sqrt(3 / (alpha_x + alpha_y + beta_x + beta_y + gamma))
    
    
    sigma_theta = Es**2 * (exp(2 * z / X0) - 1) / E**2
    sigma_theta_x = sigma_theta / 2
    sigma_theta_y = sigma_theta / 2
    
    sigma_x = Es**2 * (exp(2 * z / X0) - 1)**3 * X0**2 / (48 * E**2)
    sigma_y = Es**2 * (exp(2 * z / X0) - 1)**3 * X0**2 / (48 * E**2)
    
    likelihood = 0.
    likelihood -= ( theta_x**2 / (2 * sigma_theta_x) + log(sigma_theta_x) / 2)
    likelihood -= ( theta_y**2 / (2 * sigma_theta_y) + log(sigma_theta_y) / 2)
    
    likelihood -= ( dx**2 / (2 * sigma_x) + log(sigma_x) / 2 )
    likelihood -= ( dy**2 / (2 * sigma_y) + log(sigma_y) / 2 )
    
    
    likelihood -= (-log(theta_x**2 + theta_y**2) / 2 + log(sigma_theta) + (theta_x**2 + theta_y**2) / sigma_theta)
    return E, likelihood


def rms_integral_root_closed_py(basetrack_left, basetrack_right, 
                             TX_LEFT='TX', TY_LEFT='TY',
                             TX_RIGHT='TX', TY_RIGHT='TY'):
    EPS = 1e-6
    dz = basetrack_right['features']['SZ'] - basetrack_left['features']['SZ']
    dx = basetrack_left['features']['SX'] - (basetrack_right['features']['SX'] - basetrack_right['features'][TX_RIGHT] * dz)
    dy = basetrack_left['features']['SY'] - (basetrack_right['features']['SY'] - basetrack_right['features'][TY_RIGHT] * dz)
    dtx = (basetrack_left['features'][TX_LEFT] - basetrack_right['features'][TX_RIGHT])
    dty = (basetrack_left['features'][TX_LEFT] - basetrack_right['features'][TY_RIGHT])
    # dz can be assigned to arbitrary value, acutally !
    dz = DISTANCE
    a = (dtx * dz) ** 2 + (dty * dz) ** 2
    b = 2 * (dtx * dz * dx +  dty * dz * dy)
    c = dx ** 2 + dy ** 2
    if a == 0.:
        return fabs(sqrt(c))
    discriminant = (b ** 2 - 4 * a * c)
    log_denominator = 2 * sqrt(a) * sqrt(a + b + c) + 2 * a + b 
    log_numerator = 2 * sqrt(a) * sqrt(c) + b + EPS
    first_part = ( (2 * a + b) * sqrt(a + b + c) - b * sqrt(c) ) / (4 * a)
    return fabs((discriminant * log(log_numerator / log_denominator) / (8 * sqrt(a * a * a)) + first_part))



def gen_x_y_dataset(graphx: nx.DiGraph):
    """
    info about 2 best predecessors and 2 best 
    """
    node_ids = []
    X = []
    y = []
    for node_id, node in graphx.nodes(data=True):
        x = [node['features']['chi2'],
             node['features']['TX'],
             node['features']['TY']
            ,]
        successors_list = []
        
        for successor_id in graphx.successors(node_id):
            successors_list.append(
                (
                    graphx.node[successor_id],
                    graphx[node_id][successor_id], 
                    graphx[node_id][successor_id]['features']['r']
                )
            )
        successors_list = sorted(successors_list, key=lambda x: x[2])
        for k in range(2):
            if len(successors_list) > k:
                energy, likelihood = scattering_estimation_loss(
                    node, 
                    successors_list[k][0]
                )
                x.extend([
                    successors_list[k][1]['features']['r'],
                    energy,
                    likelihood,
                    abs(successors_list[k][1]['features']['dsx']),
                    abs(successors_list[k][1]['features']['dsy']),
                    abs(successors_list[k][1]['features']['dsz']),
                    abs(successors_list[k][1]['features']['dsxProjLeft']),
                    abs(successors_list[k][1]['features']['dsyProjLeft']),
                    abs(successors_list[k][1]['features']['dsxProjRight']),
                    abs(successors_list[k][1]['features']['dsyProjRight']),
                    abs(successors_list[k][0]['features']['TX'] - node['features']['TX']),
                    abs(successors_list[k][0]['features']['TY'] - node['features']['TY']),
                    abs(successors_list[k][0]['features']['TX']),
                    abs(successors_list[k][0]['features']['TY']),
                    successors_list[k][0]['features']['chi2']
                ])
            else:
                x.extend([-999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999])
        
        
        predecessors_list = []
        for predecessor_id in graphx.predecessors(node_id):
            predecessors_list.append(
                (
                    graphx.node[predecessor_id], 
                    graphx[predecessor_id][node_id],
                    graphx[predecessor_id][node_id]['features']['r']
                )
            )
        predecessors_list = sorted(predecessors_list, key=lambda x: x[2])
        
        for k in range(2):
            if len(predecessors_list) > k:
                energy, likelihood = scattering_estimation_loss(
                    node, 
                    predecessors_list[k][0]
                )
                x.extend([
                    predecessors_list[k][1]['features']['r'],
                    energy,
                    likelihood,
                    abs(predecessors_list[k][1]['features']['dsx']),
                    abs(predecessors_list[k][1]['features']['dsy']),
                    abs(predecessors_list[k][1]['features']['dsz']),
                    abs(predecessors_list[k][1]['features']['dsxProjLeft']),
                    abs(predecessors_list[k][1]['features']['dsyProjLeft']),
                    abs(predecessors_list[k][1]['features']['dsxProjRight']),
                    abs(predecessors_list[k][1]['features']['dsyProjRight']),
                    abs(predecessors_list[k][0]['features']['TX'] - node['features']['TX']),
                    abs(predecessors_list[k][0]['features']['TY'] - node['features']['TY']),
                    abs(predecessors_list[k][0]['features']['TX']),
                    abs(predecessors_list[k][0]['features']['TY']),
                    predecessors_list[k][0]['features']['chi2']
                ])
            else:
                x.extend([-999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999])
                
        X.append(x)
        y.append(node['signal'])
        node_ids.append(node_id)
    return X, y, node_ids