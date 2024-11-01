from distance_metrics import euclidean, cosim

# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.


def knn(train: list, query: list, metric: str):
    # For the given query, find the closest k examples in the training set
    # Assign the most common label among those collected to that given query
    # Do this for all queries
    # query is list(pixels) 1-D array of length 784 (img_size)
    # train is list[label, list(pixels)]
    k = 5
    labels = []
    # Find k closest neighbors by sorting
    f_d = None
    if metric == 'euclidean':
        f_d = euclidean
    elif metric == 'cosim':
        f_d = cosim
    # Find the k closest neighbors ()
    nearest_neighbors = sorted(
        [t for t in train], key=lambda x: f_d(x[1], query))[:k]
    # Find the most common label among the k closest neighbors
    labels = [t[0] for t in nearest_neighbors]
    # Find the most common label among the k closest neighbors
    # and assign it to the query
    most_common_label = max(set(labels), key=labels.count)
    labels.append(most_common_label)
    return labels
