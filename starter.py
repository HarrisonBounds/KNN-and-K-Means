
def hamming(a, b):
    """ Returns the Hamming distance between vectors a and b

    Args:
        a (_type_): A vector
        b (_type_): A vector
    """
    edits = 0
    if len(a) != len(b):
        print("Vectors must be the same length")
        return -1
    for i in range(len(a)):
        if a[i] != b[i]:
            edits += 1
    return edits


# returns Euclidean distance between vectors and b
def euclidean(a, b):

    return (dist)

# returns Cosine Similarity between vectors and b


def cosim(a, b):

    return (dist)

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
    return (labels)


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
    show('valid.csv', 'pixels')


if __name__ == "__main__":
    main()
