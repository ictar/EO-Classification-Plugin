import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from pylab import fuzzy as plfuzzy
from pylab import skAgglomerative, show_raw

from classification.hierarchical import DIANA
from classification.optimization import FUZZY
from classification.statistics import fuzzy_misclassified_number, diana_misclassified_number

import time, json
from scipy.io import loadmat

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

#------------------------------ FUZZY --------------------------
# compare the reulst between truth, FUZZY and skfuzzy
def compare_fuzzy_skfuzzy():
    fname = r'./data/data_2_3_601.txt'
    k = int(fname.split('_')[2])
    prec = 0.01
    # load data and visualize
    dataset = np.loadtxt(fname)

    # plot raw data
    fig1, ax = plt.subplots(1, 3)
    # visualize the test data
    xpts, ypts, labels = dataset[:, 0], dataset[:, 1], dataset[:, -1]

    for label in range(k):
        ax[0].plot(xpts[labels==label], ypts[labels == label], '.', color=colors[label])
    ax[0].set_title("Test data: 200 points x {} clusters.".format(k))

    # run FUZZY and visualize
    flabels, weights, m = FUZZY(dataset[:, :2], k, prec)
    for i in flabels:
        ax[1].plot(
            i[0], i[1],
            '.', color=colors[int(i[2])-1]
        )
    for pt in m:
        ax[1].plot(pt[0], pt[1], 'rs')
        ax[1].text(pt[0]+0.2, pt[1], "({:.4f},{:.4f})".format(pt[0], pt[1]), horizontalalignment='left', size='medium', color='black')

    print("center in FUZZY: ", m)
    ax[1].set_title('using FUZZY')

    # run fuzzy in pylab and visualize
    data = np.vstack((xpts, ypts))
    center, u, u0, d, jm, p, fpc = plfuzzy(data, k, prec)

    cluster_membership = np.argmax(u, axis=0)
    for j in range(k):
        ax[2].plot(
            xpts[cluster_membership == j],
            ypts[cluster_membership == j],
            '.',
            color = colors[j]
            )
    # mark the center of each cluster
    for pt in center:
        ax[2].plot(pt[0], pt[1], 'rs')
        ax[2].text(pt[0]+0.2, pt[1], "({:.4f},{:.4f})".format(pt[0], pt[1]), horizontalalignment='left', size='medium', color='black')


    print("center in skfuzzy: ", center)
    
    ax[2].set_title('using skfuzzy')

    plt.show()
    # compare result between truth and FUZZY, FUZZY and fuzzy
    
# TODO: evaluate the performance of FUZZY on a single dataset, including running time, accuracy compared with truth.
def performance_fuzzy(dataset, tlabels, k, tcenters, prec=0.01, prec_decimals=2):
    performance = {}

    # time
    start = time.process_time_ns()
    labels, weights, m = FUZZY(data, k, prec)
    end = time.process_time_ns()
    performance['elapse_time'] = end - start

    # accuracy
    performance['misclassified_number'] = fuzzy_misclassified_number(tlabels, tcenters, labels[:, -1], m, prec_decimals)

    return performance

#---------------------------------------------------------------

#------------------------------ DIANA --------------------------
# TOFIX: the result returned from DIANA is wrong!!!
def compare_diana_sklearn():
    fname = r'./data/data_2_3_601.txt'
    k = int(fname.split('_')[2])

    # load data and visualize
    dataset = np.loadtxt(fname)

    # plot raw data
    fig1, ax = plt.subplots(1, 4)
    # visualize the test data
    show_raw(ax[0], dataset, k)

    data = dataset[:, :2]
    # run DIANA and visualize
    dlabels, idx, n_cluster = DIANA(data, 8)
    print(idx, n_cluster)
    ax[1].scatter(dlabels[:, 0], dlabels[:, 1], c=dlabels[:, 2], cmap="rainbow")
    ax[1].set_title('using DIANA')
    ## show index
    ax[2].plot(idx)

    # run skAgglomerative and visualize
    sklabels = skAgglomerative(data, k)

    # show result
    ax[3].scatter(data[:, 0], data[:, 1], c=sklabels, cmap='rainbow')
    ax[1].set_title('using Agglomerative in sklearn with n_clusters={}'.format(k))
    
    plt.show()

# TODO: evaluate the performance of DIANA, including running time, accuracy compared with truth.
def performance_DIANA(dataset, tlabels):
    performance = {}

    # time
    start = time.process_time_ns()
    dlabels, idx, n_cluster = DIANA(dataset)
    end = time.process_time_ns()
    performance['elapse_time'] = end - start

    # accuracy
    performance['misclassified_number'] = diana_misclassified_number(tlabels, dlabels[:, -1])

    return performance

#---------------------------------------------------------------

def compare_fuzzy_diana(dir, save_to):
    performance = {}
    # for each data file in dir
    for fname in os.listdir(dir):
        mat = None
        # if it is .mat, conver it to readable
        if fname.endswith(".mat"):
            mat = loadmat(fname)
            # dict_keys(['__header__', '__version__', '__globals__', 'C_cl', 'C_ts', 'label', 'mix', 'mu_cl', 'mu_ts', 'p_cl', 'p_ts', 'ts'])
        else: continue

        if mat is None:
            print(fname, " is invalid!! Skip!")
            continue
        dataset = mat['mix']
        tlabels = mat['label']
        k = mat['C_cl'].shape[0]
        tcenters = mat['mu_ts']
        tp = {
            'C_cl': mat['C_cl'], # model cov. Matrices of cluster    (Ncl x Nb x Nb)
            'mu_ts': mat['mu_ts'], # estimated mean clusters                          (Ncl x Nb)
            'n_cluster': k,
        }
        # play FUZZY
        tp['fuzzy'] = performance_fuzzy(dataset, tlabels, k, tcenters)
        # play DIANA
        tp['diana'] = performance_DIANA(dataset, tlabels)

        performance[fname] = tp

    # save to file
    with open(save_to, "w") as f:
        json.dump(performance, f)

if __name__ == '__main__':
    #compare_fuzzy_skfuzzy()
    #compare_diana_sklearn()
    compare_fuzzy_diana(r'./data/compare', "compare.json")