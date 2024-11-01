import numpy as np
import pandas as pd
from starter import cosim

user_a_train = pd.read_csv("train_a.txt", delimiter='\t')
user_b_train = pd.read_csv("train_b.txt", delimiter='\t')
user_c_train = pd.read_csv("train_c.txt", delimiter='\t')

combined_data = pd.concat([user_a_train, user_b_train, user_c_train])

user_item_matrix = combined_data.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

print("\nUser-Item Matrix:")
print(user_item_matrix)

print(user_item_matrix[1])

similarity_results = {}

for movie1 in user_item_matrix.columns:
    similairty_row = []
    movie1_vector = user_item_matrix[movie1].values
    
    for movie2 in user_item_matrix.columns:
        if movie1 != movie2:
            movie2_vector = user_item_matrix[movie2].values
            
            similarity_score = cosim(movie1_vector, movie2_vector)
            
            similairty_row.append((movie2, similarity_score))
            
    similarity_results[movie1] = similairty_row
    

        
        
        



