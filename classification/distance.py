import numpy as np
import math

# definition of distance
euclidean_distance = lambda x, y: math.sqrt((x-y)@(x-y).T)

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
    n1, n2 = I.shape[0], J.shape[0]
    min_dist = math.inf
    for i in I:
        for j in J:
            dist = distance(i, j)
            if dist < min_dist:
                min_dist = dist

    return min_dist

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

# return the cluster who has the largest diameter
def max_diameter_cluster(cls):
    idx, max_diameter = -1, -1
    for i in range(len(cls)):
        diameter = cluster_diameter(cls[i])
        if diameter > max_diameter:
            idx, max_diameter = i, diameter
    return idx, max_diameter

# return the average distinct between a point and a cluster)
def avg_distinct(x, C, distance=euclidean_distance):
    total = 0
    for c in C:
        total += distance(x, c)
    return total/C.shape[0]

