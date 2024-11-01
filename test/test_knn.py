from k_nearest_neighbors import knn
import pytest
from starter import read_data, show


@pytest.fixture
def format_data():
    # Data files end with .csv
    mnist_training_data = read_data("mnist_train.csv")
    mnist_testing_data = read_data("mnist_test.csv")
    mnist_validation_data = read_data("mnist_valid.csv")
    # show('mnist_valid.csv', mode='pixels')


def test_knn(format_data):
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
    assert knn(train, query, metric='euclidean', k=3) == [1, 1, 0]
