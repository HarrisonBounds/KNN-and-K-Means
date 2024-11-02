import numpy as np
from distance_metrics import euclidean, cosim, pearson, hamming
from copy import deepcopy

dist = 0
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

removed_features = []

def reduce_data(data_set):
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
    global removed_features
    removed_features = [index for index, variance in enumerate(
        variances) if variance < threshold]

    for entry in data_cp:
        entry[1] = np.delete(entry[1], sorted(removed_features, reverse=True))

    return data_cp


def reduce_query(data_set):
    """ Returns the reduced query point

    Args:
        (int, ndarray): image

    Returns:
        (int, ndarray): Reduced image

    """
    query_cp = deepcopy(data_set)
    for entry in query_cp:
        entry[1] = np.delete(entry[1], sorted(removed_features, reverse=True))

    return query_cp


def initialize_centroids(k, data):
    centroids = []

    ind = np.random.randint(0, len(data))
    centroids.append(data[ind][1])

    for i in range(1, k):
        distances = []
        for x in data:
            to_centroid = []
            for c in centroids[:i]:
                to_centroid.append(euclidean(x[1], c))
            distances.append(np.min(to_centroid))

        distances = np.array(distances)

        probabilities = distances**2
        probabilities /= np.sum(probabilities)

        next_centroid = np.random.choice(len(data), p=probabilities)
        centroids.append(data[next_centroid][1])

    return np.array(centroids)


# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric):
    k = 10
    max_iters = 100
    labels = []
    train_reduced = reduce_data(train)
    centroids = initialize_centroids(k, train_reduced)
    cluster_assignments = {}

    for _ in range(max_iters):
        distances = []
        for c in centroids:
            centroid_dist = []
            for x in train_reduced:
                if metric == "euclidean":
                    centroid_dist.append(euclidean(c, x[1]))
                elif metric == "cosim":
                    centroid_dist.append(cosim(c, x[1]))
            distances.append(np.array(centroid_dist))
        distances = np.array(distances)

        # Assign clusters based on minimum distance to centroids
        clusters = np.argmin(distances, axis=0)

        new_centroids = []
        for i in range(k):
            cluster_group = []
            for j in range(len(train_reduced)):
                if clusters[j] == i:
                    cluster_group.append(train_reduced[j][1])

            cluster_group = np.array(cluster_group)

            if len(cluster_group) > 0:
                centroid = cluster_group.mean(axis=0)
            else:
                centroid = np.zeros(len(train_reduced[0][1]))

            new_centroids.append(centroid)

        new_centroids = np.array(new_centroids)

        if np.all(new_centroids == centroids):
            break
        else:
            centroids = new_centroids

    query_reduced = reduce_query(query)

    query_distances = []
    for c in centroids:
        centroid_dist = []
        for q in query_reduced:
            if metric == "euclidean":
                centroid_dist.append(euclidean(q[1], c))
            elif metric == "cosim":
                centroid_dist.append(cosim(q[1], c))
        query_distances.append(np.array(centroid_dist))
    query_distances = np.array(query_distances)

    classes = np.argmin(query_distances, axis=0)

    for c in classes:
        labels.append(int(c))

    return labels

    # # Harrison code
    # for _ in range(max_iters):
    #     total_error = 0
    #     # Assign Points
    #     for example in train:
    #         for i in range(len(example)):

    #             for j, centroid in enumerate(centroids):
    #                 if metric == "euclidean":
    #                     dist = euclidean(example[i], centroid)
    #                 elif metric == "cosim":
    #                     dist = cosim(example[i], centroid)

    #                 if dist < min_dist:
    #                     min_dist = dist
    #                     assigned_centroid = j

    #             if assigned_centroid in cluster_assignments.keys():
    #                 cluster_assignments[assigned_centroid].append(example[i])
    #             else:
    #                 cluster_assignments[assigned_centroid] = [example[i]]

    #     # Update centroids
    #     new_centroids = []
    #     for j in range(len(centroids)):
    #         for pixel in cluster_assignments[j]:
    #             x_sum += pixel[0]
    #             y_sum += pixel[1]
    #             num_pixels += 1
    #             new_centroid = (x_sum / num_pixels, y_sum / num_pixels)
    #             centroids[j], new_centroids[j]
    #         new_centroids.append(new_centroid)

    #     # Calculate error
    #     for j in range(len(centroids)):
    #         if metric == "euclidean":
    #             dist = euclidean(centroids[j], new_centroids[j])
    #         elif metric == "cosim":
    #             cosim(centroids[j], new_centroids[j])

    #         total_error += dist

    #     print("Total error: ", total_error)

    #     if total_error < 0.01:
    #         break

    #     centroids = new_centroids

    # # Run the query set
    # for example in query:
    #     for i in range(len(example)):
    #         min_dist = IMAGE_HEIGHT * IMAGE_WIDTH

    #         for j, centroid in enumerate(centroids):
    #             if metric == "euclidean":
    #                 dist = euclidean(example[i], centroid)
    #             elif metric == "cosim":
    #                 dist = cosim(example[i], centroid)

    #             if dist < min_dist:
    #                 min_dist = dist
    #                 assigned_centroid = j

    #         if assigned_centroid in cluster_assignments.keys():
    #             cluster_assignments[assigned_centroid].append(example[i])
    #         else:
    #             cluster_assignments[assigned_centroid] = [example[i]]

    # return labels


def accuracy(labels, test_data):
    correct = 0
    for i in range(len(labels)):
        if int(test_data[i][0]) == labels[i]:
            correct += 1

    return correct / len(labels)


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
            data_set.append([label, np.array(attribs, dtype=float)])
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

    labels = kmeans(mnist_training_data, mnist_testing_data, "euclidean")
    print(accuracy(labels, mnist_testing_data))

if __name__ == "__main__":
    main()
