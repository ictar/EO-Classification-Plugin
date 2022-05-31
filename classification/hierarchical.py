from re import L
import numpy as np
import math
from .distance import *

# reference: https://www.codetd.com/article/13154567
# find the minimum value in D and return the corresponding index
def min_index(D):
    m = D.shape[0]
    i_index, j_index = 0, 0
    min_val = math.inf
    for i in range(m):
        for j in range(m):
            if i == j: continue
            if D[i, j] < min_val:
                min_val, i_index, j_index = D[i, j], i, j

    return i_index, j_index

'''
convert classes to a ndarry with the class as final comlumn value
'''
def clusters2array(clses):
    for idx in range(len(clses)):
        clses[idx]

# return the cluster who has the largest diameter
def max_diameter_cluster(cls, valid_col_slice):
    idx, max_diameter = -1, -1
    for i in range(len(cls)):
        diameter = cluster_diameter(cls[i][:, valid_col_slice])
        if diameter > max_diameter:
            idx, max_diameter = i, diameter
    return cls[idx], idx

# param data: numpy.array whose shape is (N, dims)
# param point_distance: method to calulate the distance between two points, default is "euclidean distance"
def DIANA(data, point_distance=euclidean_distance):
    N =  data.shape[0] # number of data
    dim = data.shape[1] # data dimension (n bands)

    # distance matrix between each point
    D = points_distance(data, point_distance)

    # number of iteration / current labels ([1..M])
    M = 1
    # labels (N, 1)
    label = np.ones((N, 1)) * M

    while M < N:
        diam = np.zeros((M, 1))
        num = np.zeros((M, 1))
        for k in range(1, M): # compute the diameters of each cluster
            pos = np.where(label[:, M-1] ==k)
            diam[k-1] = D[pos, pos].max(0).max(0)
            num[k-1] = len(pos) # nuber of elements in cluster k

        diamMax, k1 = diam.max(0), diam.argmax(0) # k1 = cluster with largest diameter

        if diamMax == 0:
            numMax, k1 = num.max(0), num.argmax(0)

        if M <= 5 and M > 1:
            print("""
            diameter of cluster:
            {}

            cluster to be splitted: {}
            -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            """.format(diam, k1))


        pos1 = np.where(label[:, M-1] == k1)
        # dissimilarity table for cluster k1
        T = D[pos1, pos1]
        # average distance between one point to the others
        Tmed1 = np.sum(T) / (len(pos1) - 1)
        # pick the largest one which may be separated from cluster k1
        Tmax1, iMax1 = Tmed1.max(0), Tmed1.argmax(0)
        # position of the picked one in D
        i = pos1[iMax1]

        # create a new column to store the new labels in this interation
        np.concatenate((label, label[:, M-1]))

        pos2, Tmed2 = [], 0
        #  separate i if the mean distance for element i in cluster k1 is large then the mean distance for element i in cluster k2 (element i has the largest mean distance in cluster k1)
        while Tmax1 >= Tmed2 and len(pos1) > 1:
            # move element i from k1 to k2
            pos1 = pos1.difference(i)
            pos2 = pos2.union(i)
            # set the element i to a new label
            label[i, M] = M+1
            # confusion: Tmax[M-1] = Tmax1

            T = D[pos1, pos1]

            if len(pos1) > 1:
                Tmed1 = np.sum(T) / len(pos1) - 1
                Tmax1, iMax1 = Tmed1.max(0), Tmed1.argmax(0)
                i = pos1[iMax1] # next candidate
                T = D[i, pos2]
                Tmed2 = np.mean(T)

    return np.concatenate((data, label[:, M-1]))
