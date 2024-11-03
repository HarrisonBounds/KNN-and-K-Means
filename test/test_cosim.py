import numpy as np
from distance_metrics import cosim


def test_cosim():
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([1, 1, 0])
    a4 = np.array([0, 0, 1])

    assert cosim(a1, a2) == 0.0
    assert np.round(cosim(a1, a3), 3) == 0.707
    assert np.round(cosim(a2, a3), 3) == 0.707
    assert cosim(a3, a4) == 0.0

    a5 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a6 = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    assert np.round(cosim(a5, a6), 3) == 1.0

    a7 = np.array([1, 2, 3, 4, 5])
    a8 = np.array([-1, -2, -3, -4, -5])

    assert cosim(a7, a8) == -1.0
