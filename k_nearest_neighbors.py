from distance_metrics import euclidean, cosim
from starter import read_data
import numpy as np
import statistics

# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train: list, query: list, metric: str, k: int = 5) -> list:
    """
    Returns a list of labels for the query dataset based upon observations in the train dataset.

    Note that the length of the labels returned is the same as the length of the query dataset
    since each query is assigned a label.

    Args:
        train (list): The training dataset or examples that KNN will utilize to calculate distance and assign labels
        query (list): The dataset of queries that KNN will assign labels to [list(pixels)]
        metric (str): The distance metric to use for KNN. Either 'euclidean' or 'cosim'
        k (int, optional): The number of neighbors to consider. Defaults to 5.

    Raises:
        ValueError: If the distance metric is not 'euclidean' or 'cosim'
        ValueError: If the query data is not the same size as the data in the training set.

    Returns:
        list: The labels assigned to each query in the query dataset
    """
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
    else:
        raise ValueError('Invalid distance metric given')
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

def acuracy_calculation(labels:list, query: list):
    # For 100% accuracy, diagonal elements of confusino matrix need to be non-zero and rest all needs to be 0.
    expected_result = [row[0] for row in query]
    confusion_matrix = generate_confision_matrix(labels, expected_result)
    # Generating binary classification matrices.
    n = 10
    # binary_matrices is a dict[key as label : 0-9] of each bm. 
    binary_matrices = {}
    accuracy_indv, precision_indv, recall_indv, f1_score_indv = {}, {}, {}, {}
    for i in range(n):
        binary_matrix = np.zeros((2,2))
        # True positive
        binary_matrix[0][0] = confusion_matrix[i][i]
        for j in range(n):
            for k in range(n):
                if (j != i and k != i):
                    # True negetive
                    binary_matrix[1][1] += confusion_matrix[j][k]        
        for j in range(n):
            if (j!= i):
                # False negetive
                binary_matrix[1][0] += confusion_matrix[j][i]        
                # False positive
                binary_matrix[0][1] += confusion_matrix[i][j]
        binary_matrices[i] = binary_matrix
        total_sum = binary_matrix[0][0] + binary_matrix[0][1] +binary_matrix[1][0] + binary_matrix[1][1] 
        accuracy_indv[i] = (binary_matrix[0][0] + binary_matrix[1][1]) / total_sum
        precision_indv[i] = binary_matrix[0][0] / (binary_matrix[0][0] + binary_matrix[0][1])
        recall_indv[i] = binary_matrix[0][0] / (binary_matrix[0][0] + binary_matrix[1][0])
        f1_score_indv[i] = (2 * precision_indv[i] * recall_indv[i])  / (precision_indv[i] + recall_indv[i])
        
    # Calculate metrics for each and consider their mean as the system metric.   
    accuracy = statistics.mean(accuracy_indv.values()) 
    precision = statistics.mean(precision_indv.values()) 
    recall = statistics.mean(recall_indv.values()) 
    f1_score = statistics.mean(f1_score_indv.values()) 

    display(confusion_matrix)
    print(f"Accuracy of knn: {accuracy}")
    print(f"Precision of knn: {precision}")
    print(f"Recall of knn: {recall}")
    print(f"F1 Score of knn: {f1_score}")

def generate_confision_matrix(labels:list, expected_result: list):
    # Confusion matrix: is a square (n*n) matrix, with n = number of label options, as a result of knn and actual label of the querry set.
    # In this case, if the input data is sufficiently large: CM -> 10*10 
    # To generalise: Replace 10 with n.
    confusion_matrix = np.zeros((10,10))
    for i in range(len(labels)):
        confusion_matrix[expected_result[i]][labels[i]] += 1
    return confusion_matrix

def display(confusion_matrix):
    # To display the matirx, needs to be called for each dist metric.
    pass

def test_knn():
    print("test function printing")
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

    labels = knn(train=train, query=query, metric='euclidean', k=3)
    acuracy_calculation(labels, query)

def run_knn():
    mnist_training_data = read_data("mnist_train.csv")
    mnist_testing_data = read_data("mnist_test.csv")
    mnist_validation_data = read_data("mnist_valid.csv")
    print(
        f'Training Data Size: {len(mnist_training_data)}\n' +
        f'Testing Data Size: {len(mnist_testing_data)}\n' +
        f'Validation Data Size: {len(mnist_validation_data)}'
    )
    # For examples of running KNN, see test/test_knn.py where some
    # hardcoded training and query data was used to test the function

    # Before using training data, we may need to run dimensionality reduction on it
    # to reduce the number of features. We should try reduce() that we wrote
    # but we should try other methods that the assignment reccomends as well:
    # grayscale to binary, dimension scaling, etc.

    # Run training data through KNN and receive the labels for each query
    # We might have to modify KNN so the query is [label, list(pixels)] instead of just list(pixels)
    # so that we can compare the assigned label to the actual label
    # Not actually sure if this is how we do this


test_knn()