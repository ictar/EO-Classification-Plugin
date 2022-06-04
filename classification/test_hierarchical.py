import unittest
import numpy as np

from hierarchical import DIANA

class TestHierarchical(unittest.TestCase):

    def test_DIANA_1D(self):
        data = np.array([-0.308, -0.179, 0.21, 0.421, 1.224, 1.579, 1.681, 1.717]).reshape((8,1))

        label = DIANA(data)
        print("label: ", label)
        """label:  [[-0.308  7.   ]
                    [-0.179  2.   ]
                    [ 0.21   5.   ]
                    [ 0.421  3.   ]
                    [ 1.224  4.   ]
                    [ 1.579  6.   ]
                    [ 1.681  8.   ]
                    [ 1.717  1.   ]]"""


    def test_DIANA_2D(self):
        pass


if __name__ == '__main__':
    unittest.main()
