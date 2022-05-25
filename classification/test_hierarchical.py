import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from hierarchical import AGNES, DIANA, FUZZY

class TestHierarchical(unittest.TestCase):

    def test_AGNES_simple(self):
        data = np.array([-0.308, -0.179, 0.210, 0.421, 1.224, 1.579, 1.681, 1.717]).reshape((8, 1))
        classes, step_mindist = AGNES(data, n_clusters)

    def test_AGNES(self):
        iris = datasets.load_iris()

        # train
        n_clusters = 3
        #print(iris.data.shape)
        classes, step_mindist = AGNES(iris.data, n_clusters)

        # show the result
        fig, axs = plt.subplots(2)
        fig.suptitle('AGNES')

        label = ['r.', 'go', 'b*']
        i = 0
        for cls in classes:
            print(i, cls.shape)
            axs[0].plot(cls[:, 0], cls[:, 1], label[i])
            i += 1

        axs[0].set(xlabel='petal length', ylabel='petal width')

        axs[1].plot(step_mindist[:, 0], step_mindist[:, 1], 's-')
        axs[1].set(xlabel='step', ylabel='Min. Distance')

        plt.show()


    def test_DIANA(self):
        iris = datasets.load_iris()

        # train
        n_clusters = 2
        classes, iter_distance = DIANA(iris.data, n_clusters)

        # show the result
        fig, axs = plt.subplots(2)
        fig.suptitle('DIANA')

        label = ['r.', 'go', 'b*']
        i = 0
        for cls in classes:
            print(i, cls.shape)
            axs[0].plot(cls[:, 0], cls[:, 1], label[i])
            i += 1

        axs[0].set(xlabel='petal length', ylabel='petal width')

        axs[1].plot(iter_distance[:, 0], iter_distance[:, 1], 's-')
        axs[1].set(xlabel='iteration', ylabel='Min. Distance')

        plt.show()


if __name__ == '__main__':
    unittest.main()
