from starter import *
import numpy as np
import unittest

np.random.seed(30)

s1 = np.random.normal(loc=0.0, scale=1.0, size=(50, 2))
s2 = np.random.normal(loc=10.0, scale=1.0, size=(50, 2))
s3 = np.random.normal(loc=20.0, scale=1.0, size=(50, 2))

t1 = np.random.normal(loc=0.0, scale=1.0, size=(5, 2))
t2 = np.random.normal(loc=10.0, scale=1.0, size=(5, 2))
t3 = np.random.normal(loc=20.0, scale=1.0, size=(5, 2))

train_set = []
test_set = []

for x in s1:
    train_set.append(["0", x])

for x in s2:
    train_set.append(["1", x])

for x in s3:
    train_set.append(["2", x])

for x in t1:
    test_set.append(["0", x])

for x in t2:
    test_set.append(["1", x])

for x in t3:
    test_set.append(["2", x])


def test_initialize_centroids():
    centroids = initialize_centroids(k=3, data=train_set)
    assert centroids.shape == (3, 2)


def test_kmeans():
    labels = kmeans(train=train_set, query=test_set, metric='euclidean', k=3)
    print(labels)
    assert len(labels) == len(test_set)

    acc = accuracy(labels, test_set, k=10)
    assert acc == 1.0
