import pandas as pd
from read_funcs import *

random.seed(42)
distance_matrix = pd.read_csv(r'./results/random_sample/distance_matrix_body.csv', header=None)
neg_limit = 15
pos_limit = 200
margin = 0.3
# distance_matrix_norm = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())

trip_inds = []
already_query = []
file_root = r'./data/out_k=50.txt'
read_triplets(file_root, already_query, distance_matrix, trip_inds, margin, neg_limit, pos_limit)
file_root = r'./data/out_k=30.txt'
read_triplets(file_root, already_query, distance_matrix, trip_inds, margin, neg_limit, pos_limit)
file_root = r'./data/out_k=10.txt'
read_triplets(file_root, already_query, distance_matrix, trip_inds, margin, neg_limit, pos_limit)

trip_inds_pd = pd.DataFrame(trip_inds, index=None)
trip_inds_pd.to_csv(r'./fine_tune/data/triplets_body.csv', index=False, header=False)
