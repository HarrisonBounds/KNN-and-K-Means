from k_nearest_neighbors import knn
import pytest
from starter import read_data, show


def test_knn():
    train = [
        [1, [1, 0, 1, 0]],
        [0, [0, 1, 0, 1]],
        [1, [1, 1, 1, 1]],
        [0, [1, 0, 0, 1]],
        [0, [1, 0, 1, 1]],
        [0, [0, 1, 1, 0]],
        [1, [1, 0, 0, 0]],
        [1, [1, 0, 1, 1]],
        [0, [1, 1, 0, 1]],
        [1, [0, 1, 1, 1]],
        [1, [1, 0, 1, 0]],
        [0, [0, 1, 1, 0]],
    ]
    query = [[1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1]]
    assert knn(train=train, query=query, metric='euclidean', k=3) == [1, 1, 0]
