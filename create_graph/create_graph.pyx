cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, fabs, log, abs, copysign
BT_Z_LAYERS = np.array([     0.,   1293.,   2586.,   3879.,   5172.,   6465.,   7758.,
                          9051.,  10344.,  11637.,  12930.,  14223.,  15516.,  16809.,
                         18102.,  19395.,  20688.,  21981.,  23274.,  24567.,  25860.,
                         27153.,  28446.,  29739.,  31032.,  32325.,  33618.,  34911.,
                         36204.,  37497.,  38790.,  40083.,  41376.,  42669.,  43962.,
                         45255.,  46548.,  47841.,  49134.,  50427.,  51720.,  53013.,
                         54306.,  55599.,  56892.,  58185.,  59478.,  60771.,  62064.,
                         63357.,  64650.,  65943.,  67236.,  68529.,  69822.,  71115.,
                         72408.,  73701.])
cdef double DISTANCE = 1293.

@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double rms_integral_root(double[:] basetrack_left, double[:] basetrack_right) nogil:
    cdef double dx, dy, dz, dtx, dty
    dz = basetrack_right[3] - basetrack_left[3]
    dx = basetrack_left[1] - (basetrack_right[1] - basetrack_right[4] * dz)
    dy = basetrack_left[2] - (basetrack_right[2] - basetrack_right[5] * dz)
    
    #dtx = basetrack_left[4] # * copysign(1.0, dz)
    dtx = (basetrack_left[4] - basetrack_right[4]) # * copysign(1.0, dz)
    
    #dty = basetrack_left[5] # * copysign(1.0, dz)
    dty = (basetrack_left[5] - basetrack_right[5]) # * copysign(1.0, dz)
    
    dz = DISTANCE
    
    cdef double a = (dtx * dz) ** 2 + (dty * dz) ** 2
    cdef double b = 2 * (dty * dz * dx +  dty * dz * dy)
    cdef double c = dx ** 2 + dy ** 2
    cdef double discriminant = (b ** 2 - 4 * a * c)
    cdef double log_denominator = 2 * sqrt(a) * sqrt(a + b + c) + 2 * a + b 
    cdef double log_numerator = 2 * sqrt(a) * sqrt(c) + b
    cdef double first_part = ( (2 * a + b) * sqrt(a + b + c) - b * sqrt(c) ) / (4 * a)
    return fabs((discriminant * log(log_numerator / log_denominator) / (8 * sqrt(a * a * a)) + first_part))


from scipy.spatial import cKDTree
import sys
import time
from libcpp.vector cimport vector
@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef vector[vector[double]] calculate_features_for_two_layers(double[:, :] from_layer, 
                                                              double[:, :] to_layer,
                                                              vector[vector[int]] points_of_interest,
                                                              double slope,
                                                              double distance,
                                                              double r_max):# nogil:
    """
    iterative lookup based on rms_integral_root function
    """
    cdef vector[double] from_indexes, to_indexes, rs
    cdef int N = from_layer.shape[0]
    cdef int K = to_layer.shape[0]
    cdef vector[vector[double]] result
    
    if K == 0:
        result.push_back(from_indexes)
        result.push_back(to_indexes)
        result.push_back(rs)
        return result
    
    cdef double r_0
    cdef double r_1
    cdef int r_arg_0
    cdef int r_arg_1
    # indexes 
    cdef int i, j, k
    
    cdef double r, cmp_x, cmp_y, cmp_x_middle, cmp_y_middle, diff_abs, high_x, low_x, high_y, low_y
    diff_abs = fabs(distance * slope)
    
    for i in range(N):
        # mapping of current point in next later
        # for x
        cmp_x_middle = distance * from_layer[i, 4]
        
        # for y
        cmp_y_middle = distance * from_layer[i, 5]
        
        # number of points in next layer to check
        K = points_of_interest[i].size()
        for k in range(K):
            # point index
            j = points_of_interest[i][k]
            cmp_x = to_layer[j, 1] - from_layer[i, 1]
            cmp_y = to_layer[j, 2] - from_layer[i, 2]
            if cmp_x - cmp_x_middle <= diff_abs:
                if cmp_x_middle - cmp_x <= diff_abs:
                    if cmp_y - cmp_y_middle <= diff_abs:
                        if cmp_y_middle - cmp_y <= diff_abs:  
                            r = rms_integral_root(from_layer[i], to_layer[j])
                            if r < r_max:
                                from_indexes.push_back(from_layer[i][0])
                                to_indexes.push_back(to_layer[j][0])
                                rs.push_back(r)
                                
    result.push_back(from_indexes)
    result.push_back(to_indexes)
    result.push_back(rs)
    return result


@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef vector[vector[int]] make_kdtree_lookup(double[:, :] layer, kdtree, double distance):
    """
    kdtree loopup
    """
    # in next layer
    locations_to_look_in_layer = []
    # iterate over points in our layer and calculate their
    N = layer.shape[0]
    for i in range(N):
        locations_to_look_in_layer.append([layer[i, 1] + distance * layer[i, 4], 
                                           layer[i, 2] + distance * layer[i, 5]])
        # for each point in layer n looks for ball of points in layer n + 1
    if len(locations_to_look_in_layer) == 0:
        points_of_interest_py = [[]]
    else:
        # points_of_interest_py = kdtree.query_ball_point(locations_to_look_in_layer, r=800, n_jobs=8)
        points_of_interest_py = kdtree.query(locations_to_look_in_layer, k=20, n_jobs=8)[1]
    points_of_interest = <vector[vector[int]]> points_of_interest_py
    return points_of_interest
        

from libc.stdlib cimport malloc, free
from cpython cimport array
@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def create_graph(np.ndarray[double, ndim=2] data, double slope, double r_max):
    """
    create_graph takes data [idx, sx, sy, sz, tx, ty] of basetracks, maximal slope and r_max for cut.
    And returns array [idx_1, idx_2, r_dist], where idx_1 and idx_2 are indices of tracks that are connected 
    and r_dist -- distance between them
    """
    # firstly split data on layers
    layers = []
    for BT_Z in BT_Z_LAYERS:
        layers.append((data[(data[:, 3] == BT_Z), :]))
    # cdef vector[double[:, :]] layers = <vector[double[:, :]]> py_layers
    cdef vector[vector[vector[int]]] points_of_interest_per_layer_next_3
    cdef vector[vector[vector[int]]] points_of_interest_per_layer_next_2
    cdef vector[vector[vector[int]]] points_of_interest_per_layer_next_1
    cdef vector[vector[vector[int]]] points_of_interest_per_layer_previous_3
    cdef vector[vector[vector[int]]] points_of_interest_per_layer_previous_2
    cdef vector[vector[vector[int]]] points_of_interest_per_layer_previous_1
    
    cdef vector[vector[int]] points_of_interest
    
    cdef int N, i
    cdef int NUM_OF_Z = len(BT_Z_LAYERS)
    
    # distance between two emulsion layers 
    cdef double distance = DISTANCE
    cdef double d
    
    cdef vector[vector[double]] result
    results = []
    
    # going from layer = 1 to N layer and processing points with kdtree for fast lookup for
    # tracks that should be connected by edge
    for layer in range(NUM_OF_Z):
        # initiate kdtree on middle layer layer by  sx, sy coordianates
        kdtree = cKDTree(layers[layer][:, 1:3])
        
        # lookup for 1 layer next approx
        if layer >= 1:
            points_of_interest = make_kdtree_lookup(layers[layer - 1], kdtree, 1. * distance)
            points_of_interest_per_layer_next_1.push_back(points_of_interest)
        else:
            points_of_interest = <vector[vector[int]]> [[ ]]
            points_of_interest_per_layer_next_1.push_back(points_of_interest)
            
        # lookup for 2 layers next approx
        if layer >= 2:
            points_of_interest = make_kdtree_lookup(layers[layer - 2], kdtree, 2. * distance)
            points_of_interest_per_layer_next_2.push_back(points_of_interest)
        else:
            points_of_interest = <vector[vector[int]]> [[ ]]
            points_of_interest_per_layer_next_2.push_back(points_of_interest)

        # lookup for 3 layers next approx
        if layer >= 3:
            points_of_interest = make_kdtree_lookup(layers[layer - 3], kdtree, 3. * distance)
            points_of_interest_per_layer_next_3.push_back(points_of_interest)
        else:
            points_of_interest = <vector[vector[int]]> [[ ]]
            points_of_interest_per_layer_next_3.push_back(points_of_interest)
        """    
        ## lookup for 1 layer back approx
        if layer + 1 <= NUM_OF_Z - 1: 
            points_of_interest = make_kdtree_lookup(layers[layer + 1], kdtree, -1. * distance)
            points_of_interest_per_layer_previous_1.push_back(points_of_interest)
        else:
            points_of_interest = <vector[vector[int]]> [[ ]]
            points_of_interest_per_layer_previous_1.push_back(points_of_interest)
            
        # lookup for 2 layer back approx
        if layer + 2 <= NUM_OF_Z - 1: 
            points_of_interest = make_kdtree_lookup(layers[layer + 2], kdtree, -2. * distance)
            points_of_interest_per_layer_previous_2.push_back(points_of_interest)
        else:
            points_of_interest = <vector[vector[int]]> [[ ]]
            points_of_interest_per_layer_previous_2.push_back(points_of_interest)
            
        # lookup for 2 layer back approx
        if layer + 3 <= NUM_OF_Z - 1: 
            points_of_interest = make_kdtree_lookup(layers[layer + 3], kdtree, -3. * distance)
            points_of_interest_per_layer_previous_3.push_back(points_of_interest)
        else:
            points_of_interest = <vector[vector[int]]> [[ ]]
            points_of_interest_per_layer_previous_3.push_back(points_of_interest)
        """    
            
    # parallel calculation of pairs of tracks
    for i in range(NUM_OF_Z):#, nogil=True):
        # lookup for 1 layer next
        if points_of_interest_per_layer_next_1[i].size() != 0 and i >= 1:
            result = calculate_features_for_two_layers(layers[i - 1],
                                                       layers[i],
                                                       points_of_interest_per_layer_next_1[i], slope, 1. * distance, r_max)
            results.append(np.asarray(result).T)
        
        # lookup for 2 layer next
        if points_of_interest_per_layer_next_2[i].size() != 0 and i >= 2:
            result = calculate_features_for_two_layers(layers[i - 2],
                                                       layers[i],
                                                       points_of_interest_per_layer_next_2[i], slope, 2. * distance, r_max)
            results.append(np.asarray(result).T)

        # lookup for 3 layer next
        if points_of_interest_per_layer_next_3[i].size() != 0 and i >= 3:
            result = calculate_features_for_two_layers(layers[i - 3],
                                                       layers[i],
                                                       points_of_interest_per_layer_next_3[i], slope, 3. * distance, r_max)
            results.append(np.asarray(result).T)
            
        """    
        # lookup for 1 layer back
        if points_of_interest_per_layer_previous_1[i].size() != 0 and i + 1 <= NUM_OF_Z - 1:
            result = calculate_features_for_two_layers(layers[i + 1],
                                                       layers[i],
                                                       points_of_interest_per_layer_previous_1[i], slope, -1. * distance, r_max)
            results.append(np.asarray(result).T)
            
        # lookup for 2 layer back
        if points_of_interest_per_layer_previous_2[i].size() != 0 and i + 2 <= NUM_OF_Z - 1:
            result = calculate_features_for_two_layers(layers[i + 2],
                                                       layers[i],
                                                       points_of_interest_per_layer_previous_2[i], slope, -2. * distance, r_max)
            results.append(np.asarray(result).T)
            
        # lookup for 2 layer back
        if points_of_interest_per_layer_previous_3[i].size() != 0 and i + 3 <= NUM_OF_Z - 1:
            result = calculate_features_for_two_layers(layers[i + 3],
                                                       layers[i],
                                                       points_of_interest_per_layer_previous_3[i], slope, -3. * distance, r_max)
            results.append(np.asarray(result).T)
        #with gil:
        """

    return np.concatenate(results)
