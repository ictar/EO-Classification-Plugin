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
    # D[i, j] = disance between fcenters[i] and tcenters[j]
    D = cdist(fcenters, tcenters, 'euclidean')
    # axis 1 is the axis that runs horizontally across the columns.
    tmp = D.argmin(1)
    for i in range(len(tmp)):
        lmapping[i] = tmp[i]+1

    flabels_corr = np.array([lmapping[i-1] for i in flabels]).reshape(tlabels.shape)

    #print(lmapping, tlabels.shape, flabels_corr.shape, (tlabels == flabels_corr).shape, np.count_nonzero(tlabels == flabels_corr))

    return np.size(tlabels) - np.count_nonzero(tlabels == flabels_corr)

from itertools import permutations
def _combination(arr1, arr2):
    lst = list(permutations(arr1))
    n_arr2 = len(arr2)
    for i in lst:
        yield {arr2[j]: i[j] for j in range(n_arr2)}
    
# DIANA: return the number of misclassified points
# INPUT:
#   tlabels = true label
#   dlabels = label identified by DIANA
#   k = number of clusters, k == -1 means the number of clusters equals to the number of elements
def diana_misclassified_number(tlabels, dlabels, k=-1):
    dlabels = dlabels.reshape(tlabels.shape)
    #print(tlabels.shape, dlabels.shape, (tlabels == dlabels).shape, np.count_nonzero(tlabels == dlabels))
    if k == -1:
        return np.size(tlabels) - np.count_nonzero(tlabels == dlabels), dlabels
    
    min_cnt, dlabels_min, lmapping_min = np.Inf, None, None
    arr1, arr2 = list(set(list(tlabels))), list(set(list(dlabels)))
    for lmapping in _combination(arr1, arr2):
        dlabels_corr = np.array([lmapping[i] for i in dlabels]).reshape(tlabels.shape)
        cnt = np.size(tlabels) - np.count_nonzero(tlabels == dlabels_corr)
        if cnt < min_cnt:
            min_cnt = cnt
            dlabels_min = dlabels_corr.copy()
            lmapping_min = lmapping

    #print(lmapping_min,np.size(tlabels), np.count_nonzero(tlabels == dlabels_min))
    return min_cnt, dlabels_min


# index
from sklearn.metrics import silhouette_score
def silhouette(data, label, metric='euclidean'):
    return silhouette_score(data, label, metric)
