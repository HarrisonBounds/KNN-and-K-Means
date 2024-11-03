import numpy as np
import pandas as pd
from starter import cosim

users = ["A", "B", "C"]
user_ids = [405, 655, 13]

K = 2
M = 5
rating_threshold = 4
true_postive = 0
false_positive = 0
true_negative = 0
false_negative = 0

user_a_train = pd.read_csv("train_a.txt", delimiter='\t')
user_b_train = pd.read_csv("train_b.txt", delimiter='\t')
user_c_train = pd.read_csv("train_c.txt", delimiter='\t')

user_a_test = pd.read_csv("test_a.txt", delimiter='\t')
user_b_test = pd.read_csv("test_b.txt", delimiter='\t')
user_c_test = pd.read_csv("test_c.txt", delimiter='\t')

combined_data = pd.concat([user_a_train, user_b_train, user_c_train])

combined_test_data = pd.concat([user_a_test, user_b_test, user_c_test])

user_item_matrix = combined_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

user_test_matrix = combined_test_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

#print("\nUser-Item Matrix:")
#print(user_item_matrix)

user_similarity = pd.DataFrame(index=user_ids, columns=user_ids)

#Calculate similarity between users
for id1 in user_ids:
    ratings_vector1 = user_item_matrix.loc[id1]
    for id2 in user_ids:
        if id1 == id2:
            continue
        ratings_vector2 = user_item_matrix.loc[id2]
        #print(f"Comparing User {id1} and User {id2}")
        
        user_sim = cosim(ratings_vector1, ratings_vector2)
        user_similarity.loc[id1, id2] = user_sim
        
        #print(f"Distance between users {user_sim}")
        
#print("User similarity: ", user_similarity)
        

for user_id in user_ids:
    weighted_ratings = {}
    
    similar_users = user_similarity.loc[user_id].sort_values(ascending=False) #Start with the largest similar user
    similar_users = similar_users.drop(user_id)
    top_k_users = similar_users.head(K)
    target_user_ratings = user_item_matrix.loc[user_id]
    
    #print(f"Top {K} Similar Users to User {user_id}: {top_k_users.index}")
    
    for i, sim_user_id in enumerate(top_k_users.index):
        sim_user_ratings = user_item_matrix.loc[sim_user_id]
        similarity_score = top_k_users[sim_user_id]
        
        # print("Sim user: ", sim_user_id)
        # print("Sim User Ratings: ", sim_user_ratings)
        # print("Simimlarity score between target user and current similar user: ", similarity_score)
        
        for movie_id in sim_user_ratings.index:
            rating = sim_user_ratings[movie_id]
            
            if target_user_ratings[movie_id] == 0.0 and rating > 0: #Target user hasnt rated the movie but similar user has
                if movie_id not in weighted_ratings:
                    weighted_ratings[movie_id] = 0
                    
                weighted_ratings[movie_id] += rating * similarity_score
                
    best_movies = sorted(weighted_ratings, key=weighted_ratings.get, reverse=True)[:M]
    
    print(f"The best movies for user {user_id} based off of {K} similar users: {best_movies}")
    
    #Use test set to apply evaluation metrics
    for best_movie_id in best_movies:
        actual_rating = user_test_matrix.loc[user_id, best_movie_id]
        
        if actual_rating >= rating_threshold:
            true_postive += 1
        else: 
            false_positive += 1
            
        
            
            
            
            
        
            
            
            

            
    


                
            
        
        
    
    
    
        

