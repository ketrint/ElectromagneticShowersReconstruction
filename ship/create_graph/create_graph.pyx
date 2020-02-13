cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, fabs, log, exp
import networkx
from libcpp.vector cimport vector
from tqdm import tqdm
#from tools.opera_tools import DISTANCE
cdef double DISTANCE = 1293.
from scipy.integrate import quad



@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double f(double x, double dx, double dy,double dz, double dtx, double dty, double p) nogil:
    return (fabs(dtx * x + dx)**p + fabs(dty * x + dy)**p)**(1/p) / dz

# SX, SY, SZ, TX, TY
#  0,  1,  2,  3,  4
"""
@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double rms_integral_root_closed(vector[double] basetrack_left, vector[double] basetrack_right, double p=2.) nogil:
    cdef double dz = basetrack_right[3] - basetrack_left[3]
    cdef double dx = basetrack_left[1] - (basetrack_right[1] - basetrack_right[4] * dz)
    cdef double dy = basetrack_left[2] - (basetrack_right[2] - basetrack_right[5] * dz)
    cdef double dtx = (basetrack_left[4] - basetrack_right[4])
    cdef double dty = (basetrack_left[5] - basetrack_right[5])
    
    return quad(f, a=0., b=dz, epsabs=1e-3, args=(dx, dy, dz, dtx, dty, p))[0]
    """
@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double rms_integral_root_closed(vector[double] basetrack_left, vector[double] basetrack_right) nogil:
    cdef double dz = basetrack_right[3] - basetrack_left[3]
    cdef double dx = basetrack_left[1] - (basetrack_right[1] - basetrack_right[4] * dz)
    cdef double dy = basetrack_left[2] - (basetrack_right[2] - basetrack_right[5] * dz)
    cdef double dtx = (basetrack_left[4] - basetrack_right[4])
    cdef double dty = (basetrack_left[5] - basetrack_right[5])
    
    cdef double euclidian_distance = sqrt(dx**2 + dy**2 + dz**2)
    cdef double angle_distance = sqrt(dtx**2 + dty**2)
    
    # dz can be assigned to arbitrary value, acutally!
    # dz = DISTANCE
    cdef double a = (dtx * dz) ** 2 + (dty * dz) ** 2
    cdef double b = 2 * (dtx * dz * dx +  dty * dz * dy)
    cdef double c = dx ** 2 + dy ** 2
    if a == 0.:
        return fabs(sqrt(c))# / angle_distance + euclidian_distance
    cdef double discriminant = (b ** 2 - 4 * a * c)
    cdef double log_denominator = 2 * sqrt(a) * sqrt(a + b + c) + 2 * a + b
    cdef double log_numerator = 2 * sqrt(a) * sqrt(c) + b
    cdef double first_part = ( (2 * a + b) * sqrt(a + b + c) - b * sqrt(c) ) / (4 * a)

    return fabs((discriminant * log(log_numerator / log_denominator) / (8 * sqrt(a * a * a)) + first_part))

@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef vector[double] node_to_vector(node, median=False):
    if median:
        return [node['features']['SX'], 
             node['features']['SY'], 
             node['features']['SZ'], 
             node['features']['TX_m'], 
             node['features']['TY_m']]
    else:
        return [node['features']['SX'], 
             node['features']['SY'], 
             node['features']['SZ'], 
             node['features']['TX'], 
             node['features']['TY']]

@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def generate_distances(graphx_nodes, double threshold=2000, int layers=3):
    cdef int N = len(graphx_nodes)
    cdef int i, j
    cdef double r
    ebunch = []
    graphx_nodes_preprocessed = []
    for i in range(N):
        node_id, node = graphx_nodes[i]
        node = node_to_vector(node)
        graphx_nodes_preprocessed.append([node_id] + node)
    cdef vector[vector[double]] graphx_nodes_preprocessed_vec = graphx_nodes_preprocessed
    
    for i in tqdm(range(N)):#prange(N, nogil=True):
        for j in range(N):
            #if 0 < fabs(graphx_nodes_preprocessed_vec[i][3] - graphx_nodes_preprocessed_vec[j][3]) <= DISTANCE * layers:
            r = rms_integral_root_closed(graphx_nodes_preprocessed_vec[i], 
                                             graphx_nodes_preprocessed_vec[j])
            if r < threshold:
                    #with gil:
                ebunch.append((graphx_nodes_preprocessed_vec[i][0], 
                                       graphx_nodes_preprocessed_vec[j][0], 
                                       {'weight': r}))
    return ebunch