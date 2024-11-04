import numpy as np
from distance_metrics import euclidean, cosim
from starter import reduce_data, reduce_query
from copy import deepcopy


np.random.seed(30)

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
def kmeans(train, query, metric, k=10, threshold=0.01):
    max_iters = 100
    labels = []
    train_reduced, removed_features = reduce_data(train, threshold=threshold)
    centroids = initialize_centroids(k, train_reduced)

    for it in range(max_iters):
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
        if metric == "euclidean":
            clusters = np.argmin(distances, axis=0)
        elif metric == "cosim":
            clusters = np.argmax(distances, axis=0)

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
            print(f"Exited at {it}")
            break
        else:
            centroids = new_centroids

    query_reduced = reduce_query(query, removed_features)

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

    if metric == "euclidean":
        classes = np.argmin(query_distances, axis=0)
    elif metric == "cosim":
        classes = np.argmax(query_distances, axis=0)

    for c in classes:
        labels.append(int(c))

    return labels


def calculate_clustering_accuracy(labels, test_data, k=10):
    label_mapping = {}
    correct = 0
    true_labels = []
    for x in test_data:
        true_labels.append(int(x[0]))

    for c in range(k):
        indices = []
        for i, x in enumerate(labels):
            if x == c:
                indices.append(i)
        cluster_labels = []
        for x in indices:
            cluster_labels.append(true_labels[x])
        if len(cluster_labels) > 0:
            vals, count = np.unique(
                np.array(cluster_labels), return_counts=True)
            common = vals[np.argmax(count)]
            label_mapping[c] = [common, cluster_labels]

    for key in label_mapping.keys():
        id = label_mapping[key][0]
        values = label_mapping[key][1]
        for val in values:
            if int(id) == val:
                correct += 1

    acc = correct / len(true_labels)
    print(f"K-Means Clustering Accuracy: {acc}")
    return acc