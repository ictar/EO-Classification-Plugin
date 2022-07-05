import numpy as np
import math

# definition of distance between two point
euclidean_distance = lambda x, y: np.sqrt(np.sum(np.square(x-y)))
cityblock_distance = lambda x, y: np.sum(np.abs(x-y), axis=0)

# return the distance between each point in data
def points_distance(data, distance=euclidean_distance):
    N = data.shape[0]
    results = np.ones((N, N)) * -1

    for i in range(N):
        for j in range(N):
            if i == j:
                results[i, j] = 0
                continue
            if results[i, j] != -1: continue
            results[i, j] = distance(data[i], data[j])
            results[j, i] = results[i, j]
    return results

# ----------------------------- Begin: distance between cluster and cluster -----------------------------
# return the average distance between set I and set J
def avg_cluster_distance(I, J, distance=euclidean_distance):
    n1, n2 = I.shape[0], J.shape[0]
    total = 0.0
    for i in I:
        for j in J:
            total += distance(i, j)
    return total / (n1*n2)

# return the minimum distance between set I and set J
def min_cluster_distance(I, J, distance=euclidean_distance):
    min_dist = math.inf
    for i in I:
        for j in J:
            #print("i: ", type(i), i.shape)
            dist = distance(i, j)
            if dist < min_dist:
                min_dist = dist

    return min_dist

# return the max distance between set I and set J
def max_cluster_distance(I, J, distance=euclidean_distance):
    max_dist = -math.inf
    for i in I:
        for j in J:
            dist = distance(i, j)
            if dist > max_dist:
                max_dist = dist

    return max_dist

# ----------------------------- End: distance between cluster and cluster -----------------------------

# return the cluster diameter
def cluster_diameter(data, distance=euclidean_distance):
    max_dist = -1
    for x in data:
        for y in data:
            if (x == y).all(): continue
            d = distance(x, y)
            if d > max_dist:
                max_dist = d
    return max_dist

# return the average distance between a point and a cluster
def avg_distinct(x, C, distance=euclidean_distance):
    total = 0
    for c in C:
        total += distance(x, c)
    return total/C.shape[0]

