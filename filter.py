import numpy as np
import csv
import random
from functools import reduce
from distance_metrics import cosim
from k_nearest_neighbors import knn

user_ids = []
movie_ids = []


def read_movie_data(file_name: str) -> dict:

    data_set = {}
    with open(file_name, 'rt') as f:
        f.readline()
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split('\t')
            user = int(tokens[0])

            if user in data_set:
                data_set[user]['movies'][int(tokens[1])] = (
                    int(tokens[2]), tokens[4])
            else:
                attribs = {}
                attribs['age'] = tokens[5]
                attribs['gender'] = tokens[6]
                attribs['occupation'] = tokens[7]
                attribs['movies'] = {}
                attribs['movies'][int(tokens[1])] = (int(tokens[2]), tokens[4])
                data_set[user] = attribs
    return (data_set)


def build_cf(train_data: list, user_test: dict, K=5, M=5):
    movies = []
    users = []
    for user in train_data.keys():
        users.append(user)
        for movie in list(train_data[user]['movies'].keys()):
            if movie not in movies:
                movies.append(movie)

    feature_mat = []
    for user in users:
        ratings = []
        for movie in movies:
            if movie in list(train_data[user]['movies'].keys()):
                ratings.append(train_data[user]['movies'][movie][0])
            else:
                ratings.append(0)
        feature_mat.append([user, np.array(ratings)])

    user_vector = []
    user_movies = []
    user = list(user_test.keys())[0]
    for movie in movies:
        if movie in list(user_test[user]['movies'].keys()):
            user_vector.append(user_test[user]['movies'][movie][0])
            user_movies.append(movie)
        else:
            user_vector.append(0)
    user_vector = [user, np.array(user_vector)]

    nearest_neighbors = sorted(
        [t for t in feature_mat], key=lambda x: cosim(x[1], user_vector[1])
    )[:K]
    closest_users = [int(t[0]) for t in nearest_neighbors]

    recommended = []
    for user in closest_users:
        for mov in list(train_data[user]['movies'].keys()):
            if mov not in user_movies and train_data[user]['movies'][mov][0] > 3:
                recommended.append(mov)
                if len(recommended) == M:
                    return recommended

    return recommended


def common_movies(users, train_data):
    common = set(train_data[users[0]]['movies'].keys())

    for user in users:
        print(train_data[user]['movies'].keys())
        common.intersection_update(list(train_data[user]['movies'].keys()))
        print(common)

    print(common)
    return common


def build_feature_matrix(train_data):
    movies = []
    users = []
    for user in train_data.keys():
        users.append(user)
        for movie in list(train_data[user]['movies'].keys()):
            if movie not in movies:
                movies.append(movie)

    feature_mat = []
    for user in users:
        ratings = []
        for movie in movies:
            if movie in list(train_data[user]['movies'].keys()):
                ratings.append(train_data[user]['movies'][movie][0])
            else:
                ratings.append(0)
        feature_mat.append([user, np.array(ratings)])

    return feature_mat


def main():
    movielens = 'movielens.txt'
    data = read_movie_data(movielens)
    train_a = read_movie_data('train_a.txt')
    valid_a = read_movie_data('valid_a.txt')
    test_a = read_movie_data('test_a.txt')
    recommended = build_cf(data, train_a)
    print(recommended)


if __name__ == "__main__":
    main()
