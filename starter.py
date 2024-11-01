import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


dist = 0
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
# returns Euclidean distance between vectors and b


def euclidean(a, b):
    dist = np.sqrt(np.sum(np.subtract(a, b)**2))
    return dist

# returns Cosine Similarity between vectors a and b


def cosim(a, b):
    # Change to vectors
    a = np.array(a)
    b = np.array(b)

    print("a: ", a)
    print("b: ", b)
    print("np.dot(a, b): ", np.dot(a, b))

    numerator = np.dot(a, b)
    denominator = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))

    if numerator == 0 or denominator == 0:
        return 0
    # Generalize to higher dimensions
    dist = numerator / denominator

    return (dist)


def in_same_dimension(a: np.ndarray, b: np.ndarray) -> bool:
    """ Determines if the two given vectors are in the 
        same dimension or not

    Args:
        a (np.ndarray): The first vector of any dimension
        b (np.ndarray): The second vector of any dimension

    Returns:
        bool: Whether the 2 vectors are the same dimension or not
    """
    return np.shape(a) == np.shape(b)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """ Returns the Hamming distance between vectors a and b

    Args:
        a (np.ndarray): A vector of any dimension
        b (np.ndarray): A vector of any dimension

    Returns:
        int: The Hamming distance between the two vectors

    Raises:
        ValueError: If the given vectors are different dimensions
    """
    # Ensure that the two vectors occupy the same dimension
    if not in_same_dimension(a, b):
        print(
            f"Given vectors have different shapes: " +
            f"{np.shape(a)} != {np.shape(b)}"
        )
        raise ValueError("Hamming requires 2 identically-shaped vectors")
    # Create a vector
    comparison_vector = np.array([ai != bi for ai, bi in zip(a, b)])
    return comparison_vector.sum()


def reduce(examples, r):
    # Step 1: Center the data (subtract the mean)
    X_centered = examples - np.mean(examples, axis=0)

    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]

    # Step 5: Select the top r eigenvectors
    # Number of components to retain
    top_eigenvectors = eigenvectors[:, :r]

    # Step 6: Project the data onto the top k eigenvectors
    X_reduced = np.dot(X_centered, top_eigenvectors)

    return X_reduced


def initialize_centroids(k):

    clusters = []

    for _ in range(k):
        randx = np.random.randint(0, IMAGE_WIDTH)
        randy = np.random.randint(0, IMAGE_WIDTH)
        clusters.append((randx, randy))

    return clusters


def update_centroids():
    pass

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.


def knn(train, query, metric):
    return (labels)

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


def read_data(file_name):

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

    # X_reduced = reduce(X, r)

    # print("X Reduced shape:", X_reduced.shape)

    main()
