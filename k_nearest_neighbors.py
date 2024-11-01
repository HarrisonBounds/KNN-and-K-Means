from distance_metrics import euclidean, cosim
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
    ls = []
    for i, q in enumerate(query):
        if len(q) != len(train[i][1]):
            raise ValueError('Invalid query length')
        nearest_neighbors = sorted(
            [t for t in train], key=lambda x: f_d(x[1], q)
        )[:k]
        # Find the most common label among the k closest neighbors
        ls = [t[0] for t in nearest_neighbors]
        # Find the most common label among the k closest neighbors
        # and assign it to the query
        most_common_label = np.argmax(np.bincount(ls))
        print(
            f'Query {q}\n' +
            f'Nearest neighbors: {nearest_neighbors}\n' +
            f'Labels: {ls}\n' +
            f'Most common label: {most_common_label}'
        )
        ls.append(most_common_label)
        print(f'Label[{i}]: {ls[i]}')
    print(f'Return: {ls}')
    return ls
