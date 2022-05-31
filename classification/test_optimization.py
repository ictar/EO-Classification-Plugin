import unittest
import numpy as np

from optimization import FUZZY

class TestOptimization(unittest.TestCase):

    def test_FUZZY(self):
        data = np.array([-0.308, -0.179, 0.21, 0.421, 1.224, 1.579, 1.681, 1.717]).reshape((8,1))
        k = 2
        prec = 0.001
        labels, weights = FUZZY(data, k, prec)
        print(labels)


if __name__ == '__main__':
    unittest.main()
