

from starter import in_same_dimension, hamming
import sklearn
import sklearn.metrics


def test_hamming():
    a = [1, 0, 1, 0]
    b = [0, 1, 0, 1]
    assert hamming(a, b) == 4
    a = [1, 1, 1, 1]
    b = [1, 1, 1, 1]
    assert hamming(a, b) == 0
    a = [1, 1, 1, 1]
    b = [0, 0, 0, 0]
    assert hamming(a, b) == 4
    a = [1, 0, 1, 0]
    b = [1, 0, 1, 0]
    assert hamming(a, b) == 0


def test_hamming_vs_sklearn():
    a = [1, 0, 1, 0]
    b = [0, 1, 0, 1]
    assert hamming(a, b) == sklearn.metrics.hamming_loss(a, b) * len(a)
    a = [1, 1, 1, 1]
    b = [1, 1, 1, 1]
    assert hamming(a, b) == sklearn.metrics.hamming_loss(a, b) * len(a)
    a = [1, 1, 1, 1]
    b = [0, 0, 0, 0]
    assert hamming(a, b) == sklearn.metrics.hamming_loss(a, b) * len(a)
