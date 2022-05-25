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

# data: numpy.array whose shape is (nX*nY, band+2)
# param k: number of cluster
def AGNES(data, k,
          point_distance=euclidean_distance,
          cluster_distance=min_cluster_distance,
          valid_col_slice=np.index_exp[:]
          ):
    m, n = data.shape
    # initial m clusters: every cluster has one element
    cls = [np.array(data[i]) for i in range(0, m)]
    # initial dissimilarity (distance table)
    DT = np.zeros((m, m))
    for i in range(0, m):
        for j in range(0, m):
            if i == j:
                DT[i, j] = 0
                continue
            #print('cls[i]: ', cls[i].shape, type(cls[i]))
            DT[i, j] = point_distance(cls[i][valid_col_slice], cls[j][valid_col_slice])
            DT[j, i] = DT[i, j]

    # number of current clusters
    count, can_continue = m, True
    step_mindist = []
    while count > k and can_continue:
        # find the two closest clusters
        l, r = min_index(DT)
        step, min_distance = m-count + 1, DT[(l, r)]
        step_mindist.append([step, min_distance])
        # TODO: stop when elbow of step-min_distance curve is just passed
        # merge them
        cls[l] = np.concatenate((cls[l], cls[r]), axis=0)
        ## delete the origin one
        cls = np.delete(cls, r, axis=0)
        ## delete the corresponding class in dissimilarity table
        DT = np.delete(DT, r, axis=0)
        DT = np.delete(DT, r, axis=1)
        # update the dissimilarity table
        for j in range(0, count-1):
            DT[l, j] = cluster_distance(cls[l][valid_col_slice], cls[j][valid_col_slice])
            DT[j, l] = DT[l, j]

        count -= 1

    return cls, np.array(step_mindist)

# return the cluster who has the largest diameter
def max_diameter_cluster(cls, valid_col_slice):
    idx, max_diameter = -1, -1
    for i in range(len(cls)):
        diameter = cluster_diameter(cls[i][:, valid_col_slice])
        if diameter > max_diameter:
            idx, max_diameter = i, diameter
    return cls[idx], idx

# data: numpy.array whose shape is (nX*nY, band+2)
# param k: number of cluster
def DIANA(data, k, 
        distinct=avg_distinct,
        valid_col_slice=np.index_exp[:]
        ):
    m, n = data.shape
    # Init: all points are considered as part of the same cluster
    cls = [data]
    ## number of the current clusters
    count = 1
    iter_distance = []
    # Loop: subdivide the largest cluster into two clusters
    while count < k:
        # find the largest cluster
        C, idx = max_diameter_cluster(cls, valid_col_slice)
        # find the point which has the maximum distinction => C[j]
        max_val, j = 0, -1
        for i in range(C.shape[0]):
            diff = distinct(C[i][valid_col_slice], np.delete(C, i, axis=0))
            if max_val < diff:
                max_val, j = diff, i

        iter_distance.append([count, max_val])
        # divide it into two clusters
        cls_new = C[j].reshape((1,n))
        cls_old = np.delete(C, j, axis=0)
        while True:
            update = 0
            for i in range(cls_old.shape[0]):
                l, r = distinct(cls_old[i][valid_col_slice], cls_new), distinct(cls_old[i][valid_col_slice], np.delete(cls_old, i, axis=0))
                if l < r:
                    update += 1
                    cls_new = np.concatenate((cls_new, [cls_old[i]]), axis=0)
                    cls_old = np.delete(cls_old, i, axis=0)
                    break
            if update == 0: break

        cls.pop(idx)
        cls.append(cls_old)
        cls.append(cls_new)

        count += 1

    return cls, np.array(iter_distance)