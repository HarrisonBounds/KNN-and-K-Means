import numpy as np
from distance_metrics import euclidean


def test_cosim():
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([1, 1, 0])
    a4 = np.array([0, 0, 1])

    assert np.round(euclidean(a1, a2), 3) == 1.414
    assert np.round(euclidean(a1, a3), 3) == 1.0
    assert euclidean(a1, a4) == np.linalg.norm(a1-a4)
