import unittest
import matplotlib.pyplot as plt
from sklearn import datasets

from hierarchical import AGNES

class TestHierarchical(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
