import numpy as np
import math
from distance import *

# reference: https://www.codetd.com/article/13154567
# find the minimum value in D and return the corresponding index
def min_index(D):
    return np.unravel_index(np.argmin(D, axis=None), D.shape)

# param k: number of cluster
# param min_dist: stop criterion, the mininum distance between two clusters, -1 means "not used"
def AGNES(data, k, min_dist=-1, distance=min_cluster_distance):
    m, n = data.shape # m = number of data
    # initial m clusters
    cls = [np.array(data[i]) for i in range(0, m)]
    # initial dissimilarity (distance table)
    D = np.zeros((m, n))
    for i in range (0, m):
        for j in range (0, m):
            D[i, j] = distance(cls[i], cls[j])
            D[j, i] = D[i, j]

    # number of current clusters
    count = m
    lrdist = math.inf
    while count > k and lrdist > min_dist:
        # find the two closest clusters
        l, r = min_index(D)
        lrdist = D[l,r]
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

    return cls

# param k: number of cluster
# param min_dist: stop criterion, the mininum distance between two clusters, -1 means "not used"
def DIANA(data, k, min_dist=-1, distinct=avg_distinct):
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
    # initial k clusters
    # initial m clusters
    cls = [np.array(data[i]) for i in range(0, k)]