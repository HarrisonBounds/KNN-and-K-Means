import numpy as np
from starter import reduce_data, reduce_query

data = [["1", np.array([0, 1, 2])],
        ["1", np.array([0, 3, 5])],
        ["1", np.array([0, 8, 4])],
        ["1", np.array([0, 12, 7])],
        ["2", np.array([0, 32, 89])],
        ["2", np.array([0, 7, 1])],
        ["2", np.array([0, 6, 18])],
        ["2", np.array([0, 56, 33])]]


def test_reduce_data():
    reduced_data = reduce_data(data)
    print(reduced_data)
    assert len(reduced_data) == 8
    assert len(reduced_data[1]) == 2
