from re import L
import numpy as np
import math
from distance import *

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
    print("distance matrix: ", D)

    # number of iteration / current labels ([1..M])
    M = 1
    # labels (N, 1)
    label = np.ones((N, 1)) * M
    continue_to_split = True

    while M < N and continue_to_split:
        diam = np.zeros((M, 1))
        num = np.zeros((M, 1))
        for k in range(1, M+1): # compute the diameters of each cluster
            pos, = np.where(label[:, M-1] ==k)
            #print("pos: {}\nD[pos, pos]:\n{}".format(pos, D[pos][:,pos]))
            diam[k-1] = D[pos][:,pos].max(0).max(0)
            num[k-1] = len(pos) # nuber of elements in cluster k

        diamMax, k1 = diam.max(0), diam.argmax(0)+1 # k1 = cluster with largest diameter

        if M <= 5:
            print("""
            diameter of cluster:
            {}
            number of cluster:
            {}

            cluster to be splitted: {} (diam={})
            -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            """.format(diam, num, k1, diamMax))


        pos1, = np.where(label[:, M-1] == k1)
        # dissimilarity table for cluster k1
        T = D[pos1][:, pos1]
        # average distance between one point to the others
        Tmed1 = np.sum(T, axis=0) / (len(pos1) - 1)
        # pick the largest one which may be separated from cluster k1
        Tmax1, iMax1 = Tmed1.max(0), Tmed1.argmax(0)
        # position of the picked one in D
        i = pos1[iMax1]
        #print("move element {} out of cluster {}".format(i, k1))

        # create a new column to store the new labels in this interation
        label = np.concatenate((label, label[:, -1].reshape(N,1)), 1)
        pos2, Tmed2 = [], 0
        #  separate i if the mean distance for element i in cluster k1 is large then the mean distance for element i in cluster k2 (element i has the largest mean distance in cluster k1)
        continue_to_split = False
        while Tmax1 >= Tmed2 and len(pos1) > 1:
            # move element i from k1 to k2
            pos1 = np.setdiff1d(pos1, i)
            pos2 = np.union1d(pos2, i)
            # set the element i to a new label
            label[i, M] = M+1
            continue_to_split = True
            # confusion: Tmax[M-1] = Tmax1

            T = D[pos1][:, pos1]
            #print("new T: ", T)
            if len(pos1) > 1:
                Tmed1 = np.sum(T, axis=0) / (len(pos1) - 1)
                Tmax1, iMax1 = Tmed1.max(0), Tmed1.argmax(0)
                i = pos1[iMax1] # next candidate
                T = D[i][pos2.astype('int64')]
                Tmed2 = np.mean(T)
                #print("next candidate: ", i, "\nTmed2: ", Tmed2)
        
        #print("M={}\nlabel={}".format(M, label))
        M += 1

    return np.concatenate((data, label[:, -1].reshape(N,1)), 1)
