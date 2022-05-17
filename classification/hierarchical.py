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

# param k: number of cluster
def AGNES(data, k, distance=min_cluster_distance):
    m, n = data.shape # m = number of data
    # initial m clusters
    cls = [np.array(data[i]).reshape((1,n)) for i in range(m)]
    # initial dissimilarity (distance table)
    D = np.zeros((m, m))
    for i in range (0, m):
        for j in range (0, m):
            if i == j:
                D[i, j] = 0
                continue
            #print('cls[i]: ', cls[i].shape, type(cls[i]))
            D[i, j] = distance(cls[i], cls[j])
            D[j, i] = D[i, j]

    # number of current clusters
    count, can_continue = m, True
    step_mindist = []
    while count > k and can_continue:
        # find the two closest clusters
        l, r = min_index(D)
        step, min_distance = m-count + 1, D[(l, r)]
        step_mindist.append([step, min_distance])
        # TODO: stop when elbow of step-min_distance curve is just passed
        # merge them
        cls[l] = np.concatenate((cls[l], cls[r]), axis=0)
        ## delete the origin one
        cls = np.delete(cls, r, axis=0)
        ## delete the corresponding class in dissimilarity table
        D = np.delete(D, r, axis=0)
        D = np.delete(D, r, axis=1)
        # update the dissimilarity table
        for j in range(0, count-1):
            D[l, j] = distance(cls[l], cls[j])
            D[j, l] = D[l, j]

        count -= 1

    return cls, np.array(step_mindist)

# param k: number of cluster
def DIANA(data, k, distinct=avg_distinct):
    m, n = data.shape
    # Init: all points are considered as part of the same cluster
    cls = [data]
    ## number of the current clusters
    count = 1
    # Loop: subdivide the largest cluster into two clusters
    while count < k:
        # find the largest cluster
        C, idx = max_diameter_cluster(cls)
        # find the point which has the maximum distinction => C[j]
        max_val, j = 0, -1
        for i in range(len(C.shape[0])):
            diff = distinct(C[i], np.delete(C, i, axis=0))
            if max_val < diff:
                max_val, j = diff, i

        # divide it into two clusters
        cls_new = C[j].reshape((1,n))
        cls_old = np.delete(C, j, axis=0)
        while True:
            update = 0
            for i in range(cls_old.shape[0]):
                l, r = distinct(cls_old[i], cls_new), distinct(cls_old[i], np.delete(cls_old, i, axis=0))
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

    return cls

# param k: number of cluster
def KMeans(data, k):
    pass