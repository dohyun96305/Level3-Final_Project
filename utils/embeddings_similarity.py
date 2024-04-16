import csv
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 전역 변수로 데이터를 저장할 딕셔너리

def parse_csv(file_path):
    vectors = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            id = row['id']
            vector_str = row['embeddings']
            vector = ast.literal_eval(vector_str)
            vectors[id] = vector

    return vectors

def compute_similarity(base_ids, vectors):
    similarity_results = []
    for base_id in base_ids:
        base_vector = np.array([vectors[base_id]])
        for id, vector in vectors.items():
            if id != base_id:
                similarity = cosine_similarity(base_vector, [np.array(vector)])[0][0]
                similarity_results.append({'source': id, 'target': base_id, 'distance': similarity})
    return similarity_results

def get_top_similarity_results(similarity_results, top_n=5):
    similarity_results = sorted(similarity_results, key=lambda x: x['distance'], reverse=True)
    return similarity_results[:top_n]
