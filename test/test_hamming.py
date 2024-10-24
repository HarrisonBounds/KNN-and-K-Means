import pytest
import numpy as np
from starter import hamming


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


def test_hamming_vs_np():
    a = [1, 0, 1, 0]
    b = [0, 1, 0, 1]
    assert hamming(a, b) == np.count_nonzero(np.array(a) != np.array(b))
