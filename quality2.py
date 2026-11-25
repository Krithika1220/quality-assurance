import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# 1. Keep only users with >= 200 ratings
user_counts = df_ratings['user'].value_counts()
df_ratings = df_ratings[df_ratings['user'].isin(user_counts[user_counts >= 200].index)]

# 2. Keep only books with >= 100 ratings
book_counts = df_ratings['isbn'].value_counts()
df_ratings = df_ratings[df_ratings['isbn'].isin(book_counts[book_counts >= 100].index)]

# 3. Attach titles (drop duplicate ISBNs so the merge doesn't multiply rows)
df = df_ratings.merge(df_books.drop_duplicates('isbn'), on='isbn')

# Some titles map to several ISBNs (different editions): average their ratings per user
df = df.groupby(['title', 'user'], as_index=False)['rating'].mean()

# 4. Book x user matrix (rows = books), missing ratings = 0
book_user = df.pivot(index='title', columns='user', values='rating').fillna(0)
sparse_matrix = csr_matrix(book_user.values)

# 5. Fit the KNN model with cosine distance
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_matrix)

titles = list(book_user.index)

# 6. Recommendation function
def get_recommends(book=""):
    idx = titles.index(book)
    distances, indices = model.kneighbors(sparse_matrix[idx], n_neighbors=6)

    # First result is the book itself (distance 0), so skip it
    recs = []
    for dist, i in zip(distances[0][1:], indices[0][1:]):
        recs.append([titles[i], float(dist)])

    # Expected order is farthest to closest
    recs = recs[::-1]

    return [book, recs]

print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
