import numpy as np
import pandas as pd
from starter import cosim

# Users and their IDs
#users = ["A", "B", "C"]
user_ids = []

# Hyperparameters
K = 1  # Number of similar users to consider
M = 10  # Number of movies to recommend
rating_threshold = 3  # Rating threshold for positive recommendations
# Load the full MovieLens dataset
total_matrix = pd.read_csv("movielens.txt", delimiter='\t')

# Create train, validation, and test sets from the full dataset
# You can adjust the splitting logic as needed
train_data = total_matrix.sample(frac=0.8, random_state=1)  # 80% for training
temp_data = total_matrix.drop(train_data.index)  # Remaining data
valid_data = temp_data.sample(frac=0.5, random_state=1)  # 10% for validation
test_data = temp_data.drop(valid_data.index)  # 10% for testing

# Create user-item matrices
user_item_matrix = train_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
user_test_matrix = test_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
user_valid_matrix = valid_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

for user_id in user_item_matrix.index:
    user_ids.append(user_id)
    
#print("User ids: ", user_ids)

# Calculate user similarity matrix
user_similarity = pd.DataFrame(index=user_ids, columns=user_ids)

# Calculate similarity between users
for id1 in user_ids:
    ratings_vector1 = user_item_matrix.loc[id1]
    for id2 in user_ids:
        if id1 == id2:
            continue
        ratings_vector2 = user_item_matrix.loc[id2]
        user_sim = cosim(ratings_vector1, ratings_vector2)
        user_similarity.loc[id1, id2] = user_sim

# Recommend movies for each user
true_positive = 0
false_positive = 0
false_negative = 0

for user_id in user_ids:
    weighted_ratings = {}
    
    similar_users = user_similarity.loc[user_id].sort_values(ascending=False).drop(user_id)
    top_k_users = similar_users.head(K)
    target_user_ratings = user_item_matrix.loc[user_id]
    
    for sim_user_id in top_k_users.index:
        sim_user_ratings = user_item_matrix.loc[sim_user_id]
        similarity_score = top_k_users[sim_user_id]
        
        for movie_id in sim_user_ratings.index:
            rating = sim_user_ratings[movie_id]
            if target_user_ratings[movie_id] == 0.0 and rating > 0:
                if movie_id not in weighted_ratings:
                    weighted_ratings[movie_id] = 0
                weighted_ratings[movie_id] += rating * similarity_score
                
    best_movies = sorted(weighted_ratings, key=weighted_ratings.get, reverse=True)[:M]
    
    print(f"The best movies for user {user_id} based off of {K} similar users: {best_movies}")
    
    # Evaluate using the validation set
    for best_movie_id in best_movies:
        if best_movie_id in user_valid_matrix.columns:
            actual_rating = user_valid_matrix.loc[user_id, best_movie_id]
            if actual_rating >= rating_threshold:
                true_positive += 1
            else:
                false_positive += 1
            
            user_ratings = user_valid_matrix.loc[user_id]
            for test_movie_id, user_rating in user_ratings.items():
                if test_movie_id not in best_movies and user_rating >= rating_threshold:
                    false_negative += 1

# Evaluation Metrics
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_score = (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision}\nRecall: {recall}, \nF1 Score: {f1_score}")
