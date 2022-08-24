import numpy as np
from .distance import *
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

# stop criterion
# param data: numpy.array whose shape is (N, dims)
# param point_distance: method to calulate the distance between two points, default is "euclidean distance"
def DIANA_raw(data, n_clusters=-1, point_distance=euclidean_distance):
    N =  data.shape[0] # number of data
    dim = data.shape[1] # data dimension (n bands)

    # distance matrix between each point
    D = points_distance(data, point_distance)
    print("distance matrix: ", D)

    # number of iteration / current labels ([1..M])
    M = 1
    # labels (N, 1)
    label = np.ones((N, 1)) * M

    maxSilM, sils = 0, []
    while M < N and M < n_clusters:
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
        while Tmax1 >= Tmed2 and len(pos1) > 1:
            # move element i from k1 to k2
            pos1 = np.setdiff1d(pos1, i)
            pos2 = np.union1d(pos2, i)
            # set the element i to a new label
            label[i, M] = M+1
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

        # silhouette index
        ## dist[i,k] = distance from element i to cluster k
        dist = np.zeros((N, M))
        a, b, sil = np.zeros(N), np.zeros(N), np.zeros(N)
        for n in range(N):
            n_in_cluster = int(label[n, -1])
            for k in range(1, M+1):
                pos, = np.where(label[:, M-1] == k)
                dist[n,k-1] = np.sum(D[n][pos.astype('int64')])
                
                if k == n_in_cluster and len(pos)>1:
                    dist[n, k-1] = dist[n, k-1] / (len(pos)-1)
                elif len(pos) > 0:
                    dist[n, k-1] = dist[n, k-1] / len(pos)
            
            # a[n] = distance from n to other elements in the same cluster
            a[n] = dist[n, n_in_cluster-1]
            # b[n] = minimum distance from n to other clusters
            other_clusters = np.setdiff1d(np.arange(M), n_in_cluster-1)
            #print(n, n_in_cluster, other_clusters, type(other_clusters), other_clusters.shape, dist[n])
            if other_clusters.size > 0:
                b[n] = np.min(dist[n][other_clusters])
            else: b[n] = 0

            ab = np.max([a[n], b[n]])
            if ab != 0:
                sil[n] = (b[n]-a[n])/ab
        silM = np.mean(sil)
        print("silhouette index: ", silM)
        sils.append(silM)

        if silM > maxSilM: maxSilM = silM
        else:
            # abandom the last trial
            print("silhouette index is worse in iter ", M-1)
            return np.concatenate((data, label[:, -2].reshape(N,1)), 1), sils, M-1


    return np.concatenate((data, label[:, -1].reshape(N,1)), 1), M


# DIANA using method offered by sklearn
def DIANA_v0(data, n_clusters=-1, point_distance='euclidean'):
    N =  data.shape[0] # number of data

    if n_clusters == -1:
        n_clusters = N

    # distance matrix between each point
    D = cdist(data, data, point_distance)
    #np.savetxt("D.csv", D, delimiter=",")
    # number of iteration / current labels ([1..M]) / current number of clusters
    M = 1
    # labels (N, 1)
    label = np.ones((N, 2))

    while M < N and M < n_clusters:
        label[:, 0] = label[:, 1].copy()
        diam = np.zeros((M, 1))
        num = np.zeros((M, 1))
        for k in range(1, M+1): # compute the diameters of each cluster
            pos, = np.where(label[:, 0] ==k)
            diam[k-1] = D[pos][:,pos].max(0).max(0)
            num[k-1] = len(pos) # number of elements in cluster k

        diamMax, k1 = diam.max(0), diam.argmax(0)+1 # k1 = cluster with largest diameter
        if diamMax == 0:
            numMax, k1 = num.max(0), num.argmax(0)+1

        pos1, = np.where(label[:, 0] == k1)
        # dissimilarity table for cluster k1
        T = D[pos1][:, pos1]
        # average distance between one point to the others
        Tmed1 = np.sum(T, axis=0) / (len(pos1) - 1)
        # pick the largest one which may be separated from cluster k1
        Tmax1, iMax1 = Tmed1.max(0), Tmed1.argmax(0)
        # position of the picked one in D
        i = pos1[iMax1]

        # create a new column to store the new labels in this interation
        label[:, 1] = label[:, 0].copy()
        pos2, Tmed2 = [], 0
        #  separate i if the mean distance for element i in cluster k1 ("Tmax1") is larger then the mean distance for element i in cluster k2 ("Tmed2") (element i has the largest mean distance in cluster k1)
        if M == 2:
            print(M, i, data[i], Tmax1, Tmed2)
        while Tmax1 >= Tmed2 and len(pos1) > 1:
            # move element i from k1 to k2
            pos1 = np.setdiff1d(pos1, i) #k1
            pos2 = np.union1d(pos2, i) #k2
            # set the element i to a new label
            label[i, 1] = M+1

            T = D[pos1][:, pos1]
            #if M==2:np.savetxt("T_{}.csv".format(i), T, delimiter=",")
            if len(pos1) > 1: # could be split
                # find the point that has the greatest average distance from k1: i
                Tmed1 = np.sum(T, axis=0) / (len(pos1) - 1)
                Tmax1, iMax1 = Tmed1.max(0), Tmed1.argmax(0)
                i = pos1[iMax1] # next candidate
                # average distance from point "i" to cluster k2
                T = D[i][pos2.astype('int64')]
                Tmed2 = np.mean(T) # distance to k2
            if M == 2:
                print('--> ', i, data[i], Tmax1, Tmed2, T)
                
        M += 1

    return np.concatenate((data, label[:, 1].reshape(N,1)), 1), M

# DIANA using method offered by sklearn
# when seperating cluster B from cluster A: distance of points in cluster B should be smaller than their distance in cluster A 
# refernece: https://stackoverflow.com/questions/32229055/divisive-analysis-diana-hierarchical-clustering
def DIANA_v1(data, n_clusters=-1, point_distance='euclidean'):
    N =  data.shape[0] # number of data

    if n_clusters == -1:
        n_clusters = N

    # distance matrix between each point
    D = cdist(data, data, point_distance)
    # number of iteration / current labels ([1..M]) / current number of clusters
    M = 1
    # labels (N, 1)
    label = np.ones((N, 2))

    while M < N and M < n_clusters:
        label[:, 0] = label[:, 1].copy()
        diam = np.zeros((M, 1))
        num = np.zeros((M, 1))
        for k in range(1, M+1): # compute the diameters of each cluster
            pos, = np.where(label[:, 0] ==k)
            diam[k-1] = D[pos][:,pos].max(0).max(0)
            num[k-1] = len(pos) # number of elements in cluster k

        diamMax, k1 = diam.max(0), diam.argmax(0)+1 # k1 = cluster with largest diameter
        if diamMax == 0:
            numMax, k1 = num.max(0), num.argmax(0)+1

        pos1, = np.where(label[:, 0] == k1)
        # dissimilarity table for cluster k1
        T = D[pos1][:, pos1]
        # average distance between one point to the others
        Tmed1 = np.sum(T, axis=0) / (len(pos1) - 1)
        # pick the largest one which may be separated from cluster k1
        Tmax1, iMax1 = Tmed1.max(0), Tmed1.argmax(0)
        # position of the picked one in D
        i = pos1[iMax1]

        # create a new column to store the new labels in this interation
        label[:, 1] = label[:, 0].copy()
        pos2, Tmed2 = [], 0
        #  separate i if the mean distance for element i in cluster k1 ("Tmax1") is larger then the mean distance for element i in cluster k2 ("Tmed2") (element i has the largest mean distance in cluster k1)
        while len(pos1) > 1:
            # move element i from k1 to k2
            pos1 = np.setdiff1d(pos1, i) #k1
            pos2 = np.union1d(pos2, i) #k2
            # set the element i to a new label
            label[i, 1] = M+1

            T = D[pos1][:, pos1]
            if len(pos1) > 1: # could be split
                # find the point that closer to cluster k2 than cluster k1: i
                Tmed1 = np.sum(T, axis=0) / (len(pos1) - 1)
                Tmax1, iMax1 = 0, -1
                for idx in range(len(pos1)):
                    ti = pos1[idx]
                    Tmed2 = np.mean(D[ti][pos2.astype('int')]) #distance to k2
                    if Tmed1[idx] - Tmed2 > Tmax1:
                        Tmax1 = Tmed1[idx] - Tmed2
                        iMax1 = idx

                if Tmax1 <= 0: break

                i = pos1[iMax1] # next candidate
                
                
        M += 1

    return np.concatenate((data, label[:, 1].reshape(N,1)), 1), M

def DIANA(data, n_clusters=-1, point_distance='euclidean'):
    return DIANA_v1(data, n_clusters)