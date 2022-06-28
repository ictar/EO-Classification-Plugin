import numpy as np

from scipy.spatial.distance import cdist

# FUZZY: return the number of misclassified points
# INPUT:
#   tlabels = true label
#   tcenters = true cluster center
#   flabels = label identified by FUZZY
#   fcenters = cluster center identified by FUZZY
def fuzzy_misclassified_number(tlabels, tcenters, flabels, fcenters, prec):
    # modify both labels according to prec
    tcenters = np.around(tcenters, prec)
    fcenters = np.around(fcenters, prec)
    # correct the label value of FUZZY
    lmapping = {}
    D = cdist(fcenters, tcenters, 'euclidean')
    tmp = D.argmin(0)
    for i in range(len(tmp)):
        lmapping[i] = tmp[i]+1

    flabels_corr = np.array([lmapping[i-1] for i in flabels]).reshape(-1)
    return np.size(tlabels) - np.count_nonzero(tlabels == flabels_corr)


# DIANA: return the number of misclassified points
# INPUT:
#   tlabels = true label
#   dlabels = label identified by DIANA
def diana_misclassified_number(tlabels, dlabels):
    return np.size(tlabels) - np.count_nonzero(tlabels == dlabels)