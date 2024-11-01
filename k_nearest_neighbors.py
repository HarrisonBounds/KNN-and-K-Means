from distance_metrics import euclidean, cosim
from starter import read_data
import numpy as np
# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.


def knn(train: list, query: list, metric: str, k: int = 5) -> list:
    # For the given query, find the closest k examples in the training set
    # Assign the most common label among those collected to that given query
    # Do this for all queries
    # query is list(pixels) 1-D array of length 784 (img_size)
    # train is list[label, list(pixels)]
    # Find k closest neighbors by sorting
    f_d = None
    if metric == 'euclidean':
        f_d = euclidean
    elif metric == 'cosim':
        f_d = cosim
    print(
        f'K-Nearest Neighbors using {metric} distance metric and k={k}\n' +
        f'{len(train)} training examples and {len(query)} queries'
    )
    labels = []
    for i, q in enumerate(query):
        if len(q) != len(train[i][1]):
            raise ValueError('Invalid query length')
        # Sort neighbors using distance metric
        nearest_neighbors = sorted(
            [t for t in train], key=lambda x: f_d(x[1], q)
        )[:k]
        # Find the most common label among the k closest neighbors
        labels_for_neighbor = [t[0] for t in nearest_neighbors]
        # Find the most common label among the k closest neighbors
        # and assign it to the query
        most_common_label = np.argmax(np.bincount(labels_for_neighbor))
        print(
            f'Query {q}\n' +
            f'Nearest neighbors: {nearest_neighbors}\n' +
            f'Labels for neighbors: {labels_for_neighbor}\n' +
            f'Most common label: {most_common_label}'
        )
        labels.append(most_common_label)
    return labels


def run_knn():
    mnist_training_data = read_data("mnist_train.csv")
    mnist_testing_data = read_data("mnist_test.csv")
    mnist_validation_data = read_data("mnist_valid.csv")
    print(
        f'Training Data Size: {len(mnist_training_data)}\n' +
        f'Testing Data Size: {len(mnist_testing_data)}\n' +
        f'Validation Data Size: {len(mnist_validation_data)}\n'
    )
    # Run training data through KNN and receive the labels for each query
    # May have to modify KNN so the query is [label, list(pixels)] instead of just list(pixels)
    # so that we can compare the assigned label to the actual label
    # Not actually sure if this is how we do this

    # Create a confusion matrix which shows the number of correct and incorrect labels
    # True positive, true negative, false positive, false negative
    # We need to do so for each label so we should have a 10x10 matrix
    # Use the confusion matrix to calculate:
    # Accuracy, Precision, Recall, F1 Score
