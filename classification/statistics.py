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

    flabels_corr = np.array([lmapping[i-1] for i in flabels]).reshape(tlabels.shape)

    #print(tlabels.shape, flabels_corr.shape, (tlabels == flabels_corr).shape, np.count_nonzero(tlabels == flabels_corr))
    return np.size(tlabels) - np.count_nonzero(tlabels == flabels_corr)


# DIANA: return the number of misclassified points
# INPUT:
#   tlabels = true label
#   dlabels = label identified by DIANA
def diana_misclassified_number(tlabels, dlabels):
    dlabels = dlabels.reshape(tlabels.shape)
    #print(tlabels.shape, dlabels.shape, (tlabels == dlabels).shape, np.count_nonzero(tlabels == dlabels))
    return np.size(tlabels) - np.count_nonzero(tlabels == dlabels)


# index
from sklearn.metrics import silhouette_score
def silhouette(data, label, metric='euclidean'):
    return silhouette_score(data, label, metric)
