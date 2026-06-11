import pandas as pd
import numpy as np
import ast
import pickle
from scipy.sparse import save_npz, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
 
# ── 1. Load raw data ──────────────────────────────────────────────────────────
 
print("Loading raw data …")
 
tmdb_movies  = pd.read_csv("tmdb_5000_movies.csv")
tmdb_credits = pd.read_csv("tmdb_5000_credits.csv")
 
ratings   = pd.read_csv("ratings.csv")
movies_ml = pd.read_csv("movies.csv")   # MovieLens movies
 
# ── 2. Content-based: build tag vectors ───────────────────────────────────────
 
print("Building content vectors …")
 
tmdb = tmdb_movies.merge(tmdb_credits, left_on="id", right_on="movie_id")
tmdb = tmdb[["movie_id", "title_x", "genres", "keywords", "overview", "cast", "crew"]]
tmdb.rename(columns={"title_x": "title"}, inplace=True)
tmdb.dropna(inplace=True)
 
 
def _extract_names(text, limit=None):
    items = [i["name"].replace(" ", "") for i in ast.literal_eval(text)]
    return items[:limit] if limit else items
 
 
def _director(text):
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            return [i["name"].replace(" ", "")]
    return []
 
 
tmdb["genres"]   = tmdb["genres"].apply(_extract_names)
tmdb["keywords"] = tmdb["keywords"].apply(_extract_names)
tmdb["cast"]     = tmdb["cast"].apply(lambda x: _extract_names(x, 3))
tmdb["crew"]     = tmdb["crew"].apply(_director)
tmdb["overview"] = tmdb["overview"].apply(lambda x: x.split())
 
tmdb["tags"] = (
    tmdb["genres"] + tmdb["keywords"] + tmdb["cast"]
    + tmdb["crew"] + tmdb["overview"]
)
tmdb["tags"] = tmdb["tags"].apply(lambda x: " ".join(x).lower())
 
movie_list = tmdb[["movie_id", "title"]].reset_index(drop=True)
 
cv = CountVectorizer(max_features=5000, stop_words="english")
content_matrix = cv.fit_transform(tmdb["tags"])   # sparse (N × 5000)
 
# L2-normalise rows so cosine_similarity reduces to a dot product (faster)
content_matrix_norm = normalize(content_matrix, norm="l2")
 
print(f"  Content matrix shape : {content_matrix_norm.shape}")
 
# ── 3. Collaborative: build user-rating pivot (sparse) ────────────────────────
 
print("Building collaborative matrix …")
 
movie_ratings = ratings.merge(movies_ml, on="movieId")
 
movie_pivot = movie_ratings.pivot_table(
    index="title", columns="userId", values="rating"
).fillna(0)
 
collab_sparse = csr_matrix(movie_pivot.values)
collab_norm   = normalize(collab_sparse, norm="l2")   # L2-normalise rows
 
movies_ml["clean_title"] = movies_ml["title"].apply(lambda t: t.split(" (")[0])
 
print(f"  Collab matrix shape  : {collab_sparse.shape}")
 
# ── 4. Save lightweight artifacts ─────────────────────────────────────────────
 
print("Saving artifacts …")
 
# Sparse content vectors (the CountVectorizer output)
save_npz("content_vectors.npz", content_matrix_norm)
 
# Sparse collaborative matrix
save_npz("collab_matrix.npz", collab_norm)
 
# Small DataFrames / index lists
with open("movie_list.pkl", "wb") as f:
    pickle.dump(movie_list, f)
 
with open("movies_ml.pkl", "wb") as f:
    # Only keep the columns app.py needs
    pickle.dump(
        movies_ml[["movieId", "title", "clean_title"]].reset_index(drop=True), f
    )
 
# Save the pivot row-index (list of movie titles) so app.py can map titles → row
collab_titles = list(movie_pivot.index)
with open("collab_titles.pkl", "wb") as f:
    pickle.dump(collab_titles, f)
 
print("\n✅  Done!  Artifacts written:")
print("   content_vectors.npz  (content-based, sparse, L2-normalised)")
print("   collab_matrix.npz    (collaborative, sparse, L2-normalised)")
print("   movie_list.pkl       (TMDB title index)")
print("   movies_ml.pkl        (MovieLens title + clean_title)")
print("   collab_titles.pkl    (row-order of collab_matrix)")
