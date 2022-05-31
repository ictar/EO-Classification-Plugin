import numpy as np
from .distance import *

# param data: (nX*nY, band+2)
# param k: number of cluster
# param prec: precision
def FUZZY(data, k, prec):
    N, dim = data.shape
    # initial: weight / cluster membership
    w = np.random.rand(N, k)
    somma = np.sum(w, axis=1).reshape((N, 1))
    # normalization
    w = w /somma
    # initial: cluster representative
    m = np.zeros((k, dim))
    w_old = np.zeros((N, k))
    #print("w: {}\nm: {}".format(w, m)) 
    i = 0
    while np.max(np.max(w-w_old)) > prec:
        w_old = w

        # update m
        for c in range(k):
            w_c = w[:, c]
            w_c = np.power(w_c, 2).reshape((N,1))
            m[c, :] = np.sum(w_c * data) / np.sum(w_c)

        # update w
        for c in range(k):
            m_c = m[c, :]
            w[:, c] = 1 / np.sum(np.power(data-m_c, 2), axis=1)


        somma = np.sum(w, axis=1).reshape((N, 1))
        # normalization
        w = w /somma

        #print("iter {}\nw: {}\nm: {}".format(i, w, m))

        i += 1

    w_max = w.max(1).reshape((N, 1))
    label = np.where(w == w_max)[1].reshape((N, 1))
    #print(w_max, np.where(w == w_max)[1] )
    return np.concatenate((data, label), 1), w_max
