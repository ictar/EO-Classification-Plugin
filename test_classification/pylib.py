'''
classification method using python library
'''
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

#------------------------- BEGIN FUZZY -------------------------
def fuzzy(data, k, prec):
    # skfuzzy.cluster.cmeans(data, c, m, error, maxiter, init=None, seed=None)
    return fuzz.cluster.cmeans(data, k, 2, prec, maxiter=1000)



def show_fuzzy(dataset):
    
    k = int(fname.split('_')[2])
    prec = 0.01

    fig1, ax = plt.subplots(1, 2)
    # visualize the test data
    show_raw(ax[0], dataset, k)

    # begin to cluster
    data = np.vstack((xpts, ypts))
    
    center, u, u0, d, jm, p, fpc = fuzzy(data, k, prec)


    # cntr:2d array, (S, N), Cluster centers. Data for each center along each feature provided for every cluster (of the c requested clusters).
    # u: 2d array, (S, N), Final fuzzy c-partitioned matrix.
    # fpc: Final fuzzy partition coefficient.

    # for 2d data only:
    if data.shape[0] == 2:
        xpts, ypts = data[0], data[1]
        # plot assigned clusters, for each data point in training set
        cluster_membership = np.argmax(u, axis=0)
        print(xpts.shape, ypts.shape, cluster_membership.shape)
        for j in range(k):
            ax[1].plot(
                xpts[cluster_membership == j],
                ypts[cluster_membership == j],
                '.',
                color = colors[j]
                )
        # mark the center of each cluster
        for pt in center:
            ax[1].plot(pt[0], pt[1], 'rs')
        
        ax[1].set_title('Center = {}'.format(k))

        plt.show()


#------------------------- END FUZZY -------------------------

#------------------------- BEGIN DIANA -------------------------
from scipy.cluster.hierarchy import dendrogram, linkage

def draw_dendrogram(data):
    N = data.shape[0]
    linked = linkage(data, 'single')
    labelList = range(N)
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
        orientation='top',
        labels=labelList,
        distance_sort='descending',
        show_leaf_counts=True
        )
    plt.show()

from sklearn.cluster import AgglomerativeClustering
def skAgglomerative(data, k):
    cluster = AgglomerativeClustering(
        n_clusters=k,
        affinity='euclidean',
        linkage='ward'
        )
    cluster.fit_predict(data)
    return cluster.labels_

def show_diana(dataset, k):
    data = dataset[:, :2]
    draw_dendrogram(data)

    fig1, ax = plt.subplots(1, 2)
    # visualize the test data
    show_raw(ax[0], dataset, k)

    labels = skAgglomerative(data, k)

    # show result
    ax[1].scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')

    plt.show()


def show_raw(ax, dataset, k):
    # visualize the test data
    xpts, ypts, labels = dataset[:, 0], dataset[:, 1], dataset[:, -1]

    for label in range(k):
        ax.plot(xpts[labels==label], ypts[labels == label], '.', color=colors[label])
    ax.set_title("Test data: 200 points x {} clusters.".format(k))

#------------------------- END DIANA -------------------------


if __name__ == '__main__':
    fname = r'./data/data_2_3_601.txt'
    # load data
    dataset = np.loadtxt(fname)

    #show_fuzzy(dataset)
    show_diana(dataset, 3)
