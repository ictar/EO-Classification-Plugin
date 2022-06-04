import unittest
import numpy as np

from optimization import FANNY

class TestOptimization(unittest.TestCase):

    def test_FANNY_1D(self):
        data = np.array([-0.308, -0.179, 0.21, 0.421, 1.224, 1.579, 1.681, 1.717]).reshape((8,1))
        k = 2
        prec = 0.001
        labels, weights, m = FANNY(data, k, prec)
        print("labels: {}\nweights:{}\nm:{}".format(labels, weights, m))
        """labels: [[-0.308  2.   ]
                    [-0.179  2.   ]
                    [ 0.21   2.   ]
                    [ 0.421  2.   ]
                    [ 1.224  1.   ]
                    [ 1.579  1.   ]
                    [ 1.681  1.   ]
                    [ 1.717  1.   ]]
            weights:[[0.9697]
                    [0.9868]
                    [0.9808]
                    [0.8899]
                    [0.9289]
                    [0.9998]
                    [0.9944]
                    [0.9911]]
                m:[[1.5566]
                    [0.0215]]"""

    def test_FANNY_2D(self):
        data = np.array([])
        k = 2
        prec = 0.01
        labels, weights, m = FANNY(data, k, prec)
        print("labels: {}\nweights:{}\nm:{}".format(labels, weights, m))



if __name__ == '__main__':
    unittest.main()
