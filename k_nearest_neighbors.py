from distance_metrics import euclidean, cosim
from starter import read_data, reduce_data, reduce_query
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.


def knn(train: list, query: list, metric: str, k: int = 5, debug: bool = False) -> list:
    """
    Returns a list of labels for the query dataset based upon observations in the train dataset.

    Note that the length of the labels returned is the same as the length of the query dataset
    since each query is assigned a label.

    Args:
        train (list): The training dataset or examples that KNN will utilize to calculate distance and assign labels
        query (list): The dataset of queries that KNN will assign labels to [query_label, [list(pixels)]]
        metric (str): The distance metric to use for KNN. Either 'euclidean' or 'cosim'
        k (int, optional): The number of neighbors to consider. Defaults to 5.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Raises:
        ValueError: If the distance metric is not 'euclidean' or 'cosim'
        ValueError: If the query data is not the same size as the data in the training set.

    Returns:
        list: The labels assigned to each query in the query dataset
    """
    f_d = None
    if metric == 'euclidean':
        f_d = euclidean
    elif metric == 'cosim':
        f_d = cosim
    else:
        raise ValueError('Invalid distance metric given')
    if debug:
        print(
            f'K-Nearest Neighbors using {metric} distance metric and k={k}, ' +
            f'{len(train)} training examples and {len(query)} queries:'
        )
    labels = []
    for q_label, q in query:
        # Sort neighbors using distance metric and take the first k entries
        nearest_neighbors = sorted(
            [t for t in train], key=lambda x: f_d(x[1], q)
        )[:k]
        labels_for_neighbor = [int(t[0]) for t in nearest_neighbors]
        # Find the most common label among the k closest neighbors
        most_common_label = np.argmax(np.bincount(labels_for_neighbor))
        if debug:
            print(
                f'Query {q}\n' +
                f'Nearest neighbors: {nearest_neighbors}\n' +
                f'Labels for neighbors: {labels_for_neighbor}\n' +
                f'Most common label: {most_common_label}\n' +
                f'Expected label: {q_label}'
            )
        labels.append(most_common_label)
    return labels


def evaluate_knn_accuracy(labels: list, query: list) -> tuple:
    """
    Calculates and prints metrics, i.e. Accuracy, Precision, Recall and F1 Score for a trained KNN model.

    Args:
        labels (list): The labels assigned to each query in the query dataset by the KNN model, whose accuracy is being measured.
        query (list): The dataset of queries that KNN will assign labels. [query_label, [list(pixels)]]

    Returns:
        tuple: A tuple containing the accuracy, precision, recall, and F1 score of the KNN model

    """
    # For 100% accuracy, diagonal elements of confusion matrix need to be non-zero and rest all needs to be 0.
    expected_result = [row[0] for row in query]
    confusion_matrix = generate_confision_matrix(labels, expected_result)
    num_labels = confusion_matrix.shape[0]
    metrics = []  # (accuracy, precision, recall, f1_score)
    for i in range(num_labels):
        # True Positives: Diagonal elements (Correctly classified)
        tp = confusion_matrix[i][i]
        # False Negatives: Everything in this row except TP since it is not classified as i
        fn = np.sum(confusion_matrix[i, :]) - tp
        # False Positives: Everything in this column except TP since it is not classified as i
        fp = np.sum(confusion_matrix[:, i]) - tp
        # True Negatives: Everything except TP, FN, FP
        tn = np.sum(confusion_matrix) - (tp + fn + fp)
        accuracy = (tp + tn) / np.sum(confusion_matrix)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = (2 * precision * recall) / \
            (precision + recall) if precision + recall > 0 else 0
        metrics.append(
            [accuracy, precision, recall, f1_score]
        )
    accuracy, precision, recall, f1_score = np.mean(metrics, axis=0)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1_score: {f1_score}")
    return (accuracy, precision, recall, f1_score)


def generate_confision_matrix(labels: list, expected_result: list):
    """
    Returns the confusion matrix with the input of label and expected result.

    Args:
        labels (list): The labels assigned to each query in the query dataset by the KNN model, whose accuracy is being measured.
        expected_result (list): The correct label values of the query dataset.

    Returns: Confusion matrix (a 2D array): is a square (n*n) matrix, with n = number of label options, as the union of knn and actual label of the querry set.
             In this case, if the input data is sufficiently large: CM -> 10*10               
    """
    n = len(set(expected_result))
    confusion_matrix = np.zeros((n, n), dtype=int)
    for expected_label, predicted_label in zip(expected_result, labels):
        confusion_matrix[int(expected_label)][int(predicted_label)] += 1
    return confusion_matrix


def display_confusion_matrices(train_matrix, validation_matrix, metric):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axes = axs.flatten()
    sns.heatmap(train_matrix, annot=True, fmt='d', ax=axes[0])
    axes[0].set_xlabel("Predicted Labels")
    axes[0].set_ylabel("True Labels")
    axes[0].set_title(f"Training Set Confusion Matrix ({metric})")
    sns.heatmap(validation_matrix, annot=True, fmt='d', ax=axes[1])
    axes[1].set_xlabel("Predicted Labels")
    axes[1].set_ylabel("True Labels")
    axes[1].set_title(f"Validation Set Confusion Matrix ({metric})")
    plt.tight_layout()
    plt.show()


def run_knn():
    # Parse the MNIST dataset
    mnist_training_data = read_data("mnist_train.csv")
    mnist_testing_data = read_data("mnist_test.csv")
    mnist_validation_data = read_data("mnist_valid.csv")

    print(
        f'Training Data Size: {len(mnist_training_data)}\n' +
        f'Testing Data Size: {len(mnist_testing_data)}\n' +
        f'Validation Data Size: {len(mnist_validation_data)}'
    )

    # Before using training data, we may need to run dimensionality reduction on it
    # to reduce the number of features. We should try reduce() that we wrote
    # but we should try other methods that the assignment reccomends as well:
    # grayscale to binary, dimension scaling, etc.
    reduced_training_data, train_features = reduce_data(mnist_training_data)
    reduced_testing_data, test_features = reduce_data(mnist_testing_data)
    reduced_validation_data, valid_features = reduce_data(
        mnist_validation_data)

    test_query = reduce_query(mnist_testing_data, train_features)
    valid_query = reduce_query(mnist_validation_data, train_features)

    # Run training data through KNN and receive the labels for each query
    # We might have to modify KNN so the query is [label, list(pixels)] instead of just list(pixels)
    # so that we can compare the assigned label to the actual label
    # Not actually sure if this is how we do this
    predicted_labels = knn(
        train=reduced_training_data,
        query=test_query,
        metric='euclidean',
        k=5
    )

    training_matrix = generate_confision_matrix(
        predicted_labels, [q[0] for q in mnist_testing_data]
    )

    (accuracy, precision, recall, f1_score) = evaluate_knn_accuracy(
        labels=predicted_labels,
        query=test_query
    )

    print(
        f'Test Data Metrics (euclidean):\n' +
        f'Accuracy: {accuracy}\n' +
        f'Precision: {precision}\n' +
        f'Recall: {recall}\n' +
        f'F1 Score: {f1_score}'
    )

    validation_matrix = generate_confision_matrix(
        predicted_labels, [q[0] for q in mnist_validation_data]
    )

    (accuracy, precision, recall, f1_score) = evaluate_knn_accuracy(
        labels=predicted_labels,
        query=mnist_validation_data
    )

    print(
        f'Validation Data Metrics (euclidean):\n' +
        f'Accuracy: {accuracy}\n' +
        f'Precision: {precision}\n' +
        f'Recall: {recall}\n' +
        f'F1 Score: {f1_score}'
    )

    display_confusion_matrices(training_matrix, validation_matrix, 'euclidean')

    predicted_labels = knn(
        train=reduced_training_data,
        query=test_query,
        metric='cosim',
        k=5
    )

    training_matrix = generate_confision_matrix(
        predicted_labels, [q[0] for q in mnist_testing_data]
    )

    (accuracy, precision, recall, f1_score) = evaluate_knn_accuracy(
        labels=predicted_labels,
        query=reduced_testing_data
    )

    print(
        f'Test Data Metrics (cosim):\n' +
        f'Accuracy: {accuracy}\n' +
        f'Precision: {precision}\n' +
        f'Recall: {recall}\n' +
        f'F1 Score: {f1_score}'
    )

    validation_matrix = generate_confision_matrix(
        predicted_labels, [q[0] for q in mnist_validation_data]
    )

    (accuracy, precision, recall, f1_score) = evaluate_knn_accuracy(
        labels=predicted_labels,
        query=reduced_validation_data
    )

    print(
        f'Validation Data Metrics (cosim):\n' +
        f'Accuracy: {accuracy}\n' +
        f'Precision: {precision}\n' +
        f'Recall: {recall}\n' +
        f'F1 Score: {f1_score}'
    )

    display_confusion_matrices(training_matrix, validation_matrix, 'cosim')


if __name__ == "__main__":
    run_knn()
