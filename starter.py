import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from distance_metrics import euclidean, cosim, pearson, hamming
from copy import deepcopy

dist = 0
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

removed_features = []


def reduce(data_set):
    """ Returns the reduced dataset using variance thresholding

    Args:
        data_set (ndarray(int, ndarray)): processed data

    Returns:
        ndarray(int, ndarray): Reduced dataset

    """
    data_cp = deepcopy(data_set)
    features = np.array([feature[1] for feature in data_cp])
    variances = np.var(features, axis=0)
    threshold = 0.01
    removed_features = [index for index, variance in enumerate(
        variances) if variance < threshold]

    for ind in sorted(removed_features, reverse=True):
        for entry in data_cp:
            del entry[1][ind]

    return data_cp


def initialize_centroids(k):

    clusters = []

    for _ in range(k):
        randx = np.random.randint(0, IMAGE_WIDTH)
        randy = np.random.randint(0, IMAGE_WIDTH)
        clusters.append((randx, randy))

    return clusters


def update_centroids():
    pass


# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric):
    k = 5
    max_iters = 100
    labels = []
    centroids = initialize_centroids(k)
    x_sum = 0
    y_sum = 0
    num_pixels = 0

    cluster_assignments = {}

    print("Centroids: ", centroids)

    for _ in range(max_iters):
        total_error = 0
        for example in train:
            # print("Example :", example)
            # print("Example[0]: ", example[0])
            for i in range(len(example)):
                min_dist = IMAGE_HEIGHT * IMAGE_WIDTH

                for j, centroid in enumerate(centroids):
                    # print("Example[i]: ", example[i])
                    # print("centroid: ", centroid)
                    dist = euclidean(example[i], centroid)
                    # print("Dist: ", dist)
                    if dist < min_dist:
                        min_dist = dist
                        assigned_centroid = j

                if assigned_centroid in cluster_assignments.keys():
                    cluster_assignments[assigned_centroid].append(example[i])
                else:
                    cluster_assignments[assigned_centroid] = [example[i]]

        # Update centroids
        new_centroids = []
        for j in range(len(centroids)):
            # print("Centroid: ", centroid)
            for pixel in cluster_assignments[j]:
                # print("pixel: ", pixel)
                x_sum += pixel[0]
                y_sum += pixel[1]
                num_pixels += 1
                new_centroid = (x_sum / num_pixels, y_sum / num_pixels)
            new_centroids.append(new_centroid)

        # Calculate error
        for j in range(len(centroids)):
            dist = euclidean(centroids[j], new_centroids[j])
            total_error += dist

        print("Total error: ", total_error)

        if total_error < 0.01:
            break

        centroids = new_centroids

    return labels


def read_data(file_name: str) -> list:

    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label, attribs])
    return (data_set)


def show(file_name, mode):

    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ', end='')
                else:
                    print('*', end='')
            else:
                print('%4s ' % data_set[obs][1][idx], end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0], end='')
        print(' ')


def main():
    # show('valid.csv','pixels')
    r = 250
    k = 5

    mnist_training_data = read_data("mnist_train.csv")
    mnist_testing_data = read_data("mnist_test.csv")
    mnist_validation_data = read_data("mnist_valid.csv")
    print(
        f"Training data: {len(mnist_training_data)} Testing data: {len(
            mnist_testing_data)} Validation data: {len(mnist_validation_data)}"
    )
    # Training data is a list of lists
    # [[label, [pixels]]

    # Testing distance metrics
    a = np.array([1, 2])
    b = np.array([3, 4])
    cosim(a, b)

    data = pd.read_csv("mnist_train.csv")

    print("Shape of data: ", data.shape)

    X = data.drop(data.columns[0], axis=1)  # first column is class
    X = data.drop(data.columns[-1], axis=1)  # last column is nan?
    y = data[data.columns[0]]  # labels

    X = np.array(X)

    print("Shape of original X: ", X.shape)
    print("Shape of original y: ", y.shape)

    X_array = []

    # Convert each pixel index to (x, y) coordinates for each image
    for example in X:
        pixel_coords = []
        for i in range(len(example)):
            pixel_x = i % IMAGE_WIDTH
            pixel_y = np.floor(i / IMAGE_HEIGHT)
            pixel_coords.append((int(pixel_x), int(pixel_y)))
        X_array.append(pixel_coords)

    labels = kmeans(X_array, 0, "None")

    # print("Kmeans labels: ", labels)

    # kmeans_sklearn = KMeans(n_clusters=5, random_state=0, n_init='auto').fit(X_array)

    # print("Kmeans labels: ", kmeans_sklearn.labels_)
    # cluster_assignments_1 = kmeans(X[:500], None, None) #Only the first 500 examples

    # Only the first 500 examples
    # cluster_assignments_1 = kmeans(X[:500], None, None)


    # X_reduced = reduce(X, r)
    # print("X Reduced shape:", X_reduced.shape)
if __name__ == "__main__":
    main()
