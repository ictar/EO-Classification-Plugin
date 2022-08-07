import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import unittest
import numpy as np

from scipy.io import loadmat

from classification.optimization import FUZZY
from classification.statistics import fuzzy_misclassified_number

class TestOptimization(unittest.TestCase):


    def test_FUZZY_1D(self):
        print("test FUZZY with 1D data")
        data = np.array([-0.308, -0.179, 0.21, 0.421, 1.224, 1.579, 1.681, 1.717]).reshape((8,1))
        k = 2
        prec = 0.01
        labels, weights, m = FUZZY(data, k, prec)
        print("labels: {}\nweights:{}\nm:{}".format(labels, weights, m))
    
        # in matlab: [label, w, med] = FUZZY(data, 2, 0.01)
        tlabels = np.array([2, 2, 2, 2, 1, 1, 1, 1])
        tcenters = np.array([1.5569,0.0241]).reshape((2,1))
        self.assertEqual(fuzzy_misclassified_number(tlabels, tcenters,labels[:, -1], m, 2), 0)

    def test_FUZZY_2D_8(self):
        print("test FUZZY with 2D data (#data=8)")
        data = np.array([[1.0908, 0.2894], [2.2991, 3.6485], [2.5934, 5.4505], [1.1632, 1.6596], [0.9466, -0.7360], [1.2054, 0.7978], [3.6348, 4.1629], [2.9563, 4.5428]]).reshape((8, 2))
        k = 2
        prec = 0.01
        labels, weights, m = FUZZY(data, k, prec)
        print("labels: {}\nweights:{}\nm:{}".format(labels, weights, m))
        
        # in matlab: [label, w, med] = FUZZY(data, 2, 0.01)
        tlabels = np.array([2, 1, 1, 2, 2, 2, 1, 1])
        tcenters = np.array([[2.8800, 4.4556], [1.1048, 0.4741]]).reshape((2,2))
        self.assertEqual(fuzzy_misclassified_number(tlabels, tcenters,labels[:, -1], m, 2), 0)


    def test_FUZZY_2D_100(self):
        print("test FUZZY with 2D data (#data=100)")
        data = np.array([[1.0908, 0.2894], [2.2991, 3.6485], [2.5934, 5.4505], [1.1632, 1.6596], [0.9466, -0.7360], [1.2054, 0.7978], [3.6348, 4.1629], [2.9563, 4.5428], [4.6725, 3.8615], [2.4276, 0.4553], [0.3472, 1.9508], [0.6830, 2.6983], [4.8593, 4.0894], [1.2683, 2.0437], [3.2508, 3.2661], [4.4164, 2.6229], [1.2119, 2.5480], [3.7974, 3.3708], [3.2405, 4.1045], [3.2168, 3.4492], [3.8448, 6.4812], [3.4729, 4.6836], [-1.7417, -1.3826], [4.7463, 4.8391], [3.3499, 3.9292], [-0.1083, -0.0999], [0.1107, 3.3254], [4.9015, 3.4347], [4.5196, 4.1858], [3.8471, 3.7286], [3.0700, 2.0664], [0.7684, 1.3149], [1.0654, 2.1259], [4.2176, 4.1642], [1.4764, 0.8993], [2.2495, 0.8417], [0.5755, -0.7325], [1.8702, 3.1997], [-0.3172, 1.2089], [4.0299, 3.9867], [5.1867, 2.5643], [2.6177, 1.8295], [4.3929, 3.8827], [3.9746, 3.8916], [3.4109, 4.7330], [1.8530, 0.3852], [3.2161, 4.8263], [1.2614, 0.5442], [2.7628, 1.2396], [3.2382, 6.3681], [4.2738, 4.8909], [0.5717, 0.6846], [0.5671, 1.0730], [3.7795, 3.6815], [-0.6641, 0.4760], [2.1305, 1.3846], [2.9787, 4.1951], [3.9804, 3.7318], [0.1516, 2.4687], [1.9569, 0.5859], [1.4925, -0.1924], [0.3986, 0.5474], [4.6874, 5.9676], [1.2923, 2.5975], [1.9123, 1.0726], [0.1386, 2.5929], [-0.1881, 0.4515], [3.8684, 4.1645], [3.1039, 3.0067], [4.6681, 2.7472], [1.1911, 1.6443], [1.2889, 0.6344], [0.4602, 0.5527], [5.3251, 3.8831], [2.7125, 1.0870], [0.8716, 0.1523], [0.9660, -0.5726], [3.2994, 2.9291], [4.3538, 4.0200], [-0.1409, 2.1295], [5.9160, 4.7556], [3.7950, 2.1445], [2.9782, 0.4112], [0.4173, 2.5145], [3.0004, -0.0709], [0.8923, -0.1464], [1.0014, -0.4828], [4.5340, 4.0500], [2.8601, 3.7910], [4.5940, 4.5938], [0.0989, 1.3197], [1.2610, 2.5079], [0.4306, 0.6049], [5.0821, 3.3541], [4.0327, 5.0711], [3.6979, 5.0649], [4.3587, 3.9390], [2.1878, 3.0217], [3.7223, 6.1014], [5.3108, 2.2086]]).reshape((100, 2))
        k = 2
        prec = 0.01
        labels, weights, m = FUZZY(data, k, prec)
        #print("labels: {}\nweights:{}\nm:{}".format(labels, weights, m))

        # in matlab: [label, w, med] = FUZZY(data, 2, 0.01)
        tlabels = np.array([2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1])
        tcenters = np.array([[3.9171, 4.0407], [1.0349, 0.9986]]).reshape((2,2))
        self.assertEqual(fuzzy_misclassified_number(tlabels, tcenters,labels[:, -1], m, 2), 0)


    def test_FUZZY_mat(self):
        mp = r'data/compare/data_4_2.mat'
        print('teste FUZZY with matfile ' + mp)
        data, tlabels, tcenters, k = self._load_mat(mp)
        prec = 0.01
        labels, weights, m = FUZZY(data, k, prec)
        print("m:{},\ntrue m: {}".format(m, tcenters))
        # plot
        import matplotlib.pyplot as plt
        colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
        for i in labels:
            plt.plot(
            i[0], i[1],
            '.', color=colors[int(i[2])-1]
        )
        for pt in m:
            plt.plot(pt[0], pt[1], 'rs')
        plt.text(pt[0]+0.2, pt[1], "({:.4f},{:.4f})".format(pt[0], pt[1]), horizontalalignment='left', size='medium', color='black')
        plt.show()

        self.assertEqual(fuzzy_misclassified_number(tlabels, tcenters,labels[:, -1], m, 2), 0)


    """
    dataset, tlabels, tcenters, k = self._load_mat(mp)
    """
    def _load_mat(self, matpath):
        mat = loadmat(matpath)
        return mat['mix'], mat['label'], mat['mu_cl'], mat['C_cl'].shape[0]


if __name__ == '__main__':
    unittest.main()
