import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)

import numpy as np
import matplotlib.pyplot as plt
from pylib import fuzzy as plfuzzy
from pylib import skAgglomerative, show_raw

from classification.hierarchical import DIANA
from classification.optimization import FUZZY
from classification.statistics import fuzzy_misclassified_number, diana_misclassified_number

import time, json
from scipy.io import loadmat

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

#------------------------------ FUZZY --------------------------
# compare the reulst between truth, FUZZY and skfuzzy
def compare_fuzzy_skfuzzy():
    
    fname = SCRIPT_DIR + r'/data/data_2_3_601.txt'
    k = int(fname.split('_')[-2]) 
    # load data and visualize
    dataset = np.loadtxt(fname)
    labels = dataset[:, -1]
    '''
    mat = loadmat(r'data/compare/data_4_2.mat')
    dataset, k, labels = mat['mix'], mat['C_cl'].shape[0], mat['label'].reshape(-1)
    '''

    prec = 0.01
    # plot raw data
    fig1, ax = plt.subplots(1, 3, figsize=(15,8))
    # visualize the test data
    xpts, ypts = dataset[:, 0], dataset[:, 1]

    for label in range(k+1):
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

    #print("center in FUZZY: ", m)
    ax[1].set_title('using FUZZY')

    # run fuzzy in pylib and visualize
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


    #print("center in skfuzzy: ", center)
    
    ax[2].set_title('using skfuzzy')

    plt.show()
    # compare result between truth and FUZZY, FUZZY and fuzzy
  
# evaluate the performance of FUZZY on a single dataset, including running time, accuracy compared with truth.
def performance_fuzzy(dataset, tlabels, k, tcenters, prec=0.01, prec_decimals=2):
    performance = {}

    # time
    start = time.process_time_ns()
    labels, weights, m = FUZZY(dataset, k, prec)
    end = time.process_time_ns()
    performance['elapse_time'] = end - start
    performance['centers'] = str(m)
    # accuracy
    performance['misclassified_number'] = fuzzy_misclassified_number(tlabels, tcenters, labels[:, -1], m, prec_decimals)

    return performance

#---------------------------------------------------------------

#------------------------------ DIANA --------------------------
# compare the reulst between truth, DIANA
def compare_diana():
    '''
    fname = SCRIPT_DIR + r'/data/data_2_3_601.txt'
    k = int(fname.split('_')[-2]) 
    # load data and visualize
    dataset = np.loadtxt(fname)
    labels = dataset[:, -1]
    '''
    mat = loadmat(SCRIPT_DIR + r'/data/compare/data_4_1.mat')
    dataset, k, labels = mat['mix'], mat['C_cl'].shape[0], mat['label'].reshape(-1)

    prec = 0.01
    # plot raw data
    fig1, ax = plt.subplots(1, 3, figsize=(15,8))
    # visualize the test data
    xpts, ypts = dataset[:, 0], dataset[:, 1]

    for label in range(k+1):
        ax[0].plot(xpts[labels==label], ypts[labels == label], '.', color=colors[label])
    ax[0].set_title("Test data: 200 points x {} clusters.".format(k))

    # run DIANA and visualize
    k = 2
    dlabels, n_cluster = DIANA(dataset[:,:2], k)
    for i in dlabels:
        ax[1].plot(
            i[0], i[1],
            '.', color=colors[int(i[2])]
        )

    ax[1].set_title('using DIANA: k='+str(k))

    k = 3
    dlabels, n_cluster = DIANA(dataset[:,:2], k)
   # miscnt, dlabels_min = diana_misclassified_number(labels, dlabels[:, -1], k)
    #print(miscnt)
    #dlabels[:,-1] = dlabels_min
    for i in dlabels:
        ax[2].plot(
            i[0], i[1],
            '.', color=colors[int(i[2])]
        )

    ax[2].set_title('using DIANA: k='+str(k))

    plt.show()

# evaluate the performance of DIANA, including running time, accuracy compared with truth.
def performance_DIANA(dataset, tlabels, k):
    performance = {}

    # time
    start = time.process_time_ns()
    dlabels, n_cluster = DIANA(dataset, k)
    end = time.process_time_ns()
    performance['elapse_time'] = end - start

    # accuracy
    performance['misclassified_number'], _ = diana_misclassified_number(tlabels, dlabels[:, -1], k)

    return performance

#---------------------------------------------------------------

def compare_fuzzy_diana(dir, save_to):
    performance = {}
    # for each data file in dir
    for fname in os.listdir(dir):
        mat = None
        # if it is .mat, conver it to readable
        if fname.endswith(".mat"):
            mat = loadmat(dir + fname)
            # dict_keys(['__header__', '__version__', '__globals__', 'C_cl', 'C_ts', 'label', 'mix', 'mu_cl', 'mu_ts', 'p_cl', 'p_ts', 'ts'])
        else: continue

        if mat is None:
            print(fname, " is invalid!! Skip!")
            continue
        print("[BEGIN]", fname)
        dataset = mat['mix']
        tlabels = mat['label'].reshape(-1)
        k = mat['C_cl'].shape[0]
        tcenters = mat['mu_cl']
        tp = {
            'C_cl': mat['C_cl'].tolist(), # model cov. Matrices of cluster    (Ncl x Nb x Nb)
            'mu_cl': mat['mu_cl'].tolist(), # estimated mean clusters                          (Ncl x Nb)
            'n_cluster': k,
            'n_samples': dataset.shape[0],
        }
        # play FUZZY
        tp['fuzzy'] = performance_fuzzy(dataset, tlabels, k, tcenters)
        # play DIANA
        tp['diana'] = performance_DIANA(dataset, tlabels, k)

        performance[fname] = tp
        print("[END]", fname)
        import gc
        gc.collect()

    # save to file
    with open(save_to, "w") as f:
        json.dump(performance, f)

if __name__ == '__main__':
    #compare_fuzzy_skfuzzy()
    #compare_diana()
    compare_fuzzy_diana(r'./data/compare/', "compare.json")
    #compare_fuzzy_diana(r'./data/performance/', "performance.json")