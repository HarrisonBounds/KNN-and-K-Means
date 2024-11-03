import numpy as np
import pandas as pd
from starter import cosim



# Hyper-parameters
K = 2
M = 5
rating_weight = 3
age_weight = 3


# Initialisations
users = ["A", "B", "C"]
user_ids = [405, 655, 13]
user_age = {}
K = 2
M = 5

user_a_train = pd.read_csv("train_a.txt", delimiter='\t')

# Can similarly be used for other metrics also.
user_age[user_a_train['user_id'].values[0]] = user_a_train['age'].values[0]
user_b_train = pd.read_csv("train_b.txt", delimiter='\t')
user_age[user_b_train['user_id'].values[0]] = user_b_train['age'].values[0]
user_c_train = pd.read_csv("train_c.txt", delimiter='\t')
user_age[user_c_train['user_id'].values[0]] = user_c_train['age'].values[0]


combined_data = pd.concat([user_a_train, user_b_train, user_c_train])
user_item_matrix = combined_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
user_similarity = pd.DataFrame(index=user_ids, columns=user_ids)

print("\nUser-Item Matrix:")
print(user_item_matrix)

# Can be generalised by using input as str array of the metrics.
def normalise_metric_similarity(user_id_first:int, user_id_second:int):
    max_age_difference = max(user_age.values()) - min(user_age.values())
    age_similarity = 1 - abs(user_age[user_id_first] - user_age[user_id_second]) / max_age_difference
    return age_similarity

#Calculate similarity between users
for id1 in user_ids:
    ratings_vector1 = user_item_matrix.loc[id1]
    for id2 in user_ids:
        if id1 == id2:
            continue
        ratings_vector2 = user_item_matrix.loc[id2]
        # print(f"Comparing User {id1} and User {id2}")

        # CAlculating similaity between different metrics.
        rating_sim = cosim(ratings_vector1, ratings_vector2)
        age_sim = normalise_metric_similarity(id1,id2)

        user_sim = rating_sim * rating_weight + age_sim * age_weight 
        user_similarity.loc[id1, id2] = user_sim
        
        print(f"Distance between users {user_sim}")
        
print("User similarity: ", user_similarity)        

for user_id in user_ids:
    weighted_ratings = {}
    
    similar_users = user_similarity.loc[user_id].sort_values(ascending=False) #Start with the largest similar user
    similar_users = similar_users.drop(user_id)
    top_k_users = similar_users.head(K)
    target_user_ratings = user_item_matrix.loc[user_id]
    
    print(f"Top {K} Similar Users to User {user_id}: {top_k_users.index}")
    
    for i, sim_user_id in enumerate(top_k_users.index):
        # sim_user_ratings: series of movie ratings by the user who is in top K most similar to the user who we are going to recommend.
        sim_user_ratings = user_item_matrix.loc[sim_user_id]
        similarity_score = top_k_users[sim_user_id]
        
        
        print("Sim user: ", sim_user_id)
        print("Sim User Ratings: ", sim_user_ratings)
        print("Simimlarity score between target user and current similar user: ", similarity_score)
        
        for movie_id in sim_user_ratings.index:
            rating = sim_user_ratings[movie_id]
            
            if target_user_ratings[movie_id] == 0.0 and rating > 0: #Target user hasnt rated the movie but similar user has
                # this loop can be replaced by initialing before.
                if movie_id not in weighted_ratings:
                    weighted_ratings[movie_id] = 0
                    
                weighted_ratings[movie_id] += rating * similarity_score
                
    best_movies = sorted(weighted_ratings, key=weighted_ratings.get, reverse=True)[:M]
    
    print(f"The best movies for user {user_id} based off of {K} similar users, decided on metric of movie ratings: {best_movies}")
                
            
        
        
    
    
    
        

