import numpy as np
import pandas as pd
from distance_metrics import cosim

# Hyperparameters
K = 10
M = 7
rating_threshold = 4
min = float('inf')

target_users = [405, 655, 13]
user_ids = []
similar_users = []
similarity_dict = {}
ratings_vector_1 = []
ratings_vector_2 = []


total_matrix = pd.read_csv("movielens.txt", delimiter='\t')

user_a_train = pd.read_csv("train_a.txt", delimiter='\t')
user_b_train = pd.read_csv("train_b.txt", delimiter='\t')
user_c_train = pd.read_csv("train_c.txt", delimiter='\t')

user_a_valid = pd.read_csv("valid_a.txt", delimiter='\t')
user_b_valid = pd.read_csv("valid_b.txt", delimiter='\t')
user_c_valid = pd.read_csv("valid_c.txt", delimiter='\t')

user_a_test = pd.read_csv("test_a.txt", delimiter='\t')
user_b_test = pd.read_csv("test_b.txt", delimiter='\t')
user_c_test = pd.read_csv("test_c.txt", delimiter='\t')

combined_data = pd.concat([user_a_train, user_b_train, user_c_train])
combined_data_valid = pd.concat([user_a_valid, user_b_valid, user_c_valid])
combined_data_test = pd.concat([user_a_test, user_b_test, user_c_test])


user_total_matrix = total_matrix.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
user_item_matrix = combined_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
user_valid_matrix = combined_data_valid.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
user_test_matrix = combined_data_test.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)


# print("user item matrix: ", user_item_matrix)
# print("user total matrix: ", user_total_matrix)

print("user ids: ", user_ids)

for id1 in target_users:
    print("id1: ", id1)
    # ratings_vector1 = user_item_matrix.loc[id1]
    # print("vector1 Shape: ", ratings_vector1.shape)
    # print("Vector1: ", ratings_vector1)
    
    for id2 in user_total_matrix.index:
        if id1 == id2:
            continue
        for movie_id in user_total_matrix.columns:
            if movie_id in user_item_matrix.columns and movie_id in user_total_matrix.columns:
                rating1 = user_item_matrix.loc[id1, movie_id]
                rating2 = user_total_matrix.loc[id2, movie_id]
                if rating1 > 0 and rating2 > 0:
                    ratings_vector_1.append(rating1)
                    ratings_vector_2.append(rating2)
                
        # Calculate the similarity
        user_sim = cosim(np.array(ratings_vector_1), np.array(ratings_vector_2))
        similar_users.append((id2, user_sim))

    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    similarity_dict[id1] = similar_users[:K]  # Keep only top K similar users

#print("Similarity Dictionary: ", similarity_dict)
# Recommend movies for each user
true_positive = 0
false_positive = 0
false_negative = 0

for user_id in target_users:
    weighted_ratings = {}
    
    #similar_users = user_similarity.loc[user_id].sort_values(ascending=False).drop(user_id)
    top_k_users = similarity_dict[user_id][:K]
    #print("Top K Users: ", top_k_users)
    #target_user_ratings = user_item_matrix.loc[user_id]
    
    for sim_user_id in top_k_users:
        #print("Sim User Id[0]: ", sim_user_id[0])
        #print("Sim User Id[1]: ", sim_user_id[1])
        sim_user_ratings = user_total_matrix.loc[sim_user_id[0]]
        #print("Sim User Ratings: ", sim_user_ratings)
        similarity_score = sim_user_id[1]
        
        for movie_id in user_total_matrix.columns:
            rating = sim_user_ratings[movie_id]
            if movie_id in user_item_matrix.columns:
                if user_item_matrix.loc[user_id][movie_id] == 0 and rating > 0:
                    if movie_id not in weighted_ratings:
                        weighted_ratings[movie_id] = 0.0
                    weighted_ratings[movie_id] += rating * similarity_score
                
    best_movies = sorted(weighted_ratings, key=weighted_ratings.get, reverse=True)[:M]
    
    print(f"The best movies for user {user_id} based off of {K} similar users: {best_movies}")
    
    # Evaluate using the validation set
    for best_movie_id in best_movies:
        if best_movie_id in user_test_matrix.columns:
            actual_rating = user_test_matrix.loc[user_id, best_movie_id]
            if actual_rating >= rating_threshold:
                true_positive += 1
            else:
                false_positive += 1
            
            user_ratings = user_test_matrix.loc[user_id]
            for test_movie_id, user_rating in user_ratings.items():
                if test_movie_id not in best_movies and user_rating >= rating_threshold:
                    false_negative += 1

# Evaluation Metrics
precision = true_positive / (true_positive + false_positive) 
recall = true_positive / (true_positive + false_negative) 
f1_score = (precision * recall) / (precision + recall)

print(f"Precision: {precision}\nRecall: {recall}, \nF1 Score: {f1_score}")
