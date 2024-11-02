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
