import streamlit as st
import pickle
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ── Load lightweight artifacts (cached across sessions) ───────────────────────

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Loads all artifacts once per server lifetime.

    Memory footprint (approximate):
      content_vectors.npz  → sparse float32 (N=4800, d=5000) ≈ 4–8 MB
      collab_matrix.npz    → sparse float32 (M≈3500, k≈700)  ≈ 2–5 MB
      pickle files         → negligible
    """
    content_matrix = load_npz("content_vectors.npz")   # (N × 5000) sparse
    collab_matrix  = load_npz("collab_matrix.npz")     # (M × users) sparse

    with open("movie_list.pkl",   "rb") as f: movie_list    = pickle.load(f)
    with open("movies_ml.pkl",    "rb") as f: movies_ml     = pickle.load(f)
    with open("collab_titles.pkl","rb") as f: collab_titles = pickle.load(f)

    # Build a fast title→row-index lookup for the collab matrix
    collab_title_to_idx = {t: i for i, t in enumerate(collab_titles)}

    return content_matrix, collab_matrix, movie_list, movies_ml, collab_title_to_idx


content_matrix, collab_matrix, movie_list, movies_ml, collab_title_to_idx = (
    load_artifacts()
)

# ── On-demand similarity helpers ──────────────────────────────────────────────

def get_content_scores(movie: str, top_n: int = 20) -> dict[str, float]:
    """
    Computes cosine similarity between *one* query row and all content rows.
    Because rows are L2-normalised, this is just a dot product → very fast.
    Cost: O(N · d_nnz)  where N≈4800, d_nnz ≈ average non-zeros per row.
    """
    idx = movie_list[movie_list["title"] == movie].index
    if len(idx) == 0:
        return {}
    row_idx = idx[0]

    query_vec = content_matrix[row_idx]                       # (1 × 5000) sparse
    sims      = cosine_similarity(query_vec, content_matrix).flatten()  # (N,)

    # Exclude self (index row_idx) and return top_n
    top_indices = np.argsort(sims)[::-1]
    scores = {}
    for i in top_indices:
        if i == row_idx:
            continue
        scores[movie_list.iloc[i]["title"]] = float(sims[i])
        if len(scores) == top_n:
            break
    return scores


def get_collab_scores(movie: str, top_n: int = 20) -> dict[str, float]:
    """
    Looks up the MovieLens full title, then computes cosine similarity between
    that one row and all rows in the collab matrix.
    Cost: O(M · k_nnz)  where M≈3500, k_nnz ≈ ratings per movie (sparse).
    """
    row = movies_ml[movies_ml["clean_title"] == movie]
    if row.empty:
        return {}

    full_title = row.iloc[0]["title"]
    idx = collab_title_to_idx.get(full_title)
    if idx is None:
        return {}

    query_vec = collab_matrix[idx]                            # (1 × users) sparse
    sims      = cosine_similarity(query_vec, collab_matrix).flatten()  # (M,)

    top_indices = np.argsort(sims)[::-1]
    collab_titles_list = list(collab_title_to_idx.keys())

    scores = {}
    for i in top_indices:
        if i == idx:
            continue
        clean = collab_titles_list[i].split(" (")[0]
        scores[clean] = float(sims[i])
        if len(scores) == top_n:
            break
    return scores


def hybrid_recommend(movie: str, top_n: int = 10) -> list[tuple[str, float]]:
    content_scores = get_content_scores(movie)
    collab_scores  = get_collab_scores(movie)

    all_movies = set(content_scores) | set(collab_scores)
    final_scores = {
        m: 0.7 * content_scores.get(m, 0.0) + 0.3 * collab_scores.get(m, 0.0)
        for m in all_movies
    }
    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CineFusion — AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.block-container { padding-top:2rem; padding-left:3rem; padding-right:3rem; }
.stApp {
    background: linear-gradient(135deg, #B8FFF9 0%, #DDFCFB 30%, #FFE6D5 70%, #FFD8BE 100%);
    min-height: 100vh;
}
div[data-testid="stSelectbox"] label { color:#A04527 !important; font-size:28px !important; font-weight:800 !important; }
.hero {
    background: linear-gradient(145deg,
        rgba(252,255,254,0.96) 0%,
        rgba(240,253,251,0.91) 40%,
        rgba(255,241,230,0.95) 100%);
    backdrop-filter: blur(28px) saturate(200%) brightness(1.04);
    -webkit-backdrop-filter: blur(28px) saturate(200%) brightness(1.04);
    padding: 60px 56px;
    border-radius: 36px;
    box-shadow:
        0 32px 72px rgba(0,0,0,0.14),
        0 8px 24px rgba(229,9,20,0.07),
        0 2px 6px rgba(0,0,0,0.06),
        inset 0 1.5px 0 rgba(255,255,255,0.95),
        inset 0 -1px 0 rgba(160,69,39,0.06);
    border: 1px solid rgba(255,255,255,0.82);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 36px;
    background: linear-gradient(135deg,
        rgba(255,255,255,0.18) 0%,
        transparent 50%,
        rgba(255,179,138,0.06) 100%);
    pointer-events: none;
}
.big-font { font-size:72px; font-weight:900; letter-spacing:-2px; text-align:center; color:#E50914 !important; margin-bottom:8px; }
.sub-font  { text-align:center; font-size:22px; font-weight:500; color:#475569; }
.stat-card {
    background:white; padding:30px; border-radius:20px;
    text-align:center; font-size:22px; font-weight:700; color:#2563EB;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06); transition:0.3s;
}
.stat-card:hover { transform:translateY(-5px); box-shadow: 0 12px 25px rgba(0,0,0,0.10); }
.movie-card {
    background: linear-gradient(135deg, #0F172A, #1E293B);
    height:260px; color:white; border-radius:24px; padding:25px;
    display:flex; flex-direction:column; justify-content:center;
    align-items:center; text-align:center;
    transition:0.3s; box-shadow: 0 10px 25px rgba(0,0,0,0.15); margin-bottom:30px;
}
.movie-card:hover { transform:translateY(-8px); box-shadow: 0 15px 35px rgba(15,23,42,0.35); }
.movie-card h3 { font-size:22px; line-height:1.3; font-weight:700; font-family:'Oswald',sans-serif; letter-spacing:1px; margin-bottom:15px; }
.movie-card p  { font-size:16px; color:#CBD5E1; }
.top-pick-badge {
    display: inline-block;
    background: linear-gradient(90deg, #FFB38A, #FF9B85);
    color: #0F172A;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 20px;
    margin-top: 4px;
}
.empty-state {
    background: linear-gradient(135deg,
        rgba(248,255,254,0.85) 0%,
        rgba(255,243,234,0.85) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1.5px dashed rgba(160,69,39,0.25);
    border-radius: 24px;
    padding: 52px 40px;
    text-align: center;
    margin: 32px auto;
    max-width: 640px;
}
.empty-state-icon { font-size: 52px; margin-bottom: 16px; }
.empty-state h3 {
    color: #A04527;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 12px;
}
.empty-state p {
    color: #64748B;
    font-size: 17px;
    line-height: 1.65;
    max-width: 460px;
    margin: 0 auto;
}
.loading-msg {
    text-align: center;
    color: #A04527;
    font-size: 18px;
    font-weight: 600;
    margin-top: 12px;
}
.loading-sub {
    text-align: center;
    color: #64748B;
    font-size: 15px;
    margin-top: 4px;
}
.stButton > button {
    background: linear-gradient(90deg, #FFB38A, #FF9B85);
    color:#0F172A; font-weight:700; height:55px;
    border:none; border-radius:16px; transition:0.3s;
}
.stButton > button:hover { transform:scale(1.05); }
.stSelectbox div[data-baseweb="select"] { border-radius:16px; min-height:60px; font-size:18px; }
.footer {
    text-align: center;
    margin-top: 48px;
    padding-top: 28px;
    border-top: 1px solid rgba(160,69,39,0.12);
    color: #475569;
    font-size: 16px;
    line-height: 1.8;
}
.footer-links { margin-top: 12px; display: flex; justify-content: center; align-items: center; gap: 6px; }
.footer-links a {
    color: #A04527;
    text-decoration: none;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 8px;
    transition: color 0.2s, background 0.2s;
}
.footer-links a:hover { color: #E50914; background: rgba(229,9,20,0.06); }
            /* How CineFusion Works - Expander */

[data-testid="stExpander"] {
    background: #B8F0A1 !important;
    border: none !important;
    border-radius: 18px !important;
    overflow: hidden;
    margin-top: 20px;
}

[data-testid="stExpander"] summary {
    background: #B8F0A1 !important;
    color: #7D1907 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    padding: 14px 18px !important;
    border-radius: 18px !important;
}

[data-testid="stExpander"] summary:hover {
    background: #AEDF9A !important;
}

[data-testid="stExpander"] svg {
    color: #7D1907 !important;
    fill: #7D1907 !important;
}

/* Opened content area */
[data-testid="stExpander"] div[role="region"] {
    background: #DDF8D1 !important;
    border-top: 1px solid rgba(125,25,7,0.1);
}
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
  <h1 class="big-font">🎬 CineFusion</h1>
  <p class="sub-font">Discover Your Next Favorite Movie With AI</p>
  <p style="text-align:center;font-size:18px;font-weight:500;color:#64748B;margin-top:10px;letter-spacing:0.3px;">
    Powered by Content-Based Filtering + Collaborative Intelligence
  </p>
  <div style="margin-top:22px;text-align:center;display:flex;justify-content:center;gap:12px;flex-wrap:wrap;">
    <span style="background:rgba(160,69,39,0.09);color:#A04527;font-weight:700;font-size:14px;
                 padding:7px 18px;border-radius:999px;letter-spacing:0.4px;">
      ✨ Personalized Recommendations
    </span>
    <span style="background:rgba(160,69,39,0.09);color:#A04527;font-weight:700;font-size:14px;
                 padding:7px 18px;border-radius:999px;letter-spacing:0.4px;">
      🎭 Genre-Aware Matching
    </span>
    <span style="background:rgba(160,69,39,0.09);color:#A04527;font-weight:700;font-size:14px;
                 padding:7px 18px;border-radius:999px;letter-spacing:0.4px;">
      🤖 Hybrid AI Engine
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# Stats
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="stat-card">🎥 4,800+ Movies Available</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-card">👥 600+ User Ratings</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card">🤖 Hybrid AI Engine</div>', unsafe_allow_html=True)

st.divider()

# Movie select
left, center, right = st.columns([1, 3, 1])
with center:
    st.markdown("""
    <h3 style="color:#A04527;font-size:34px;font-weight:700;margin-bottom:10px;">
      🎥 Choose a Movie
    </h3>""", unsafe_allow_html=True)

    selected_movie = st.selectbox(
        "",
        movie_list["title"].values,
        index=None,
        placeholder="🔍 Search for a movie...",
        label_visibility="collapsed",
    )
    recommend_btn = st.button("🍿 Find Similar Movies")

# Recommendations
if recommend_btn and selected_movie:
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(
            '<p class="loading-msg">🎬 CineFusion is analyzing thousands of movies…</p>'
            '<p class="loading-sub">Finding the best recommendations for you…</p>',
            unsafe_allow_html=True,
        )

    with st.spinner(""):
        recommendations = hybrid_recommend(selected_movie)

    loading_placeholder.empty()

    st.divider()
    st.markdown("""
    <h2 style="color:#A04527;font-size:58px;font-weight:900;margin-bottom:25px;
               margin-top:10px;font-family:cursive;">
      🎯 Recommended Movies
    </h2>""", unsafe_allow_html=True)

    cols = st.columns(5)
    for i, (movie, score) in enumerate(recommendations):
        with cols[i % 5]:
            st.markdown(
                f"<div class='movie-card'>"
                f"<h3>{movie}</h3>"
                f"<span class='top-pick-badge'>★ Top Pick</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

elif recommend_btn and not selected_movie:
    st.warning("Please select a movie first.")

else:
    # Empty state — shown before any recommendation is triggered
    st.markdown("""
    <div style="display:flex;justify-content:center;">
      <div class="empty-state">
        <div class="empty-state-icon">🎬</div>
        <h3>Ready to Discover Your Next Favorite Movie?</h3>
        <p>Search for a movie above and let CineFusion generate personalized
           recommendations powered by a Hybrid AI Engine.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br><br><br>", unsafe_allow_html=True)

with st.expander("📖 How Does CineFusion Work?"):
    st.markdown("""
<div style="color:#A04527; line-height:1.75; font-size:16px;">
  <h3 style="font-size:20px; font-weight:700; margin-bottom:6px;">🎭 Content-Based Filtering</h3>
  <p style="color:#475569; margin-bottom:20px;">
    Analyzes genres, cast, keywords and movie descriptions to find films with similar characteristics.
  </p>

  <h3 style="font-size:20px; font-weight:700; margin-bottom:6px;">👥 Collaborative Filtering</h3>
  <p style="color:#475569; margin-bottom:20px;">
    Learns from ratings and viewing preferences of 600+ users to identify movies enjoyed by similar audiences.
  </p>

  <h3 style="font-size:20px; font-weight:700; margin-bottom:6px;">🤖 Hybrid Recommendation</h3>
  <p style="color:#475569;">
    Combines both approaches with a 70 / 30 weighting to deliver more accurate and personalized recommendations.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  🎬 <strong>CineFusion</strong> &nbsp;·&nbsp; Built by <strong>Satyarth Pandey</strong><br>
  <span style="font-size:14px; color:#94A3B8; letter-spacing:0.4px;">
    Machine Learning &nbsp;·&nbsp; Recommendation Systems &nbsp;·&nbsp; NLP
  </span>
  <div class="footer-links">
    <a href="https://github.com/Satyarth740" target="_blank" rel="noopener noreferrer">
      🔗 GitHub
    </a>
    <span style="color:#CBD5E1;">|</span>
    <a href="https://www.linkedin.com/in/satyarth-pandey740/" target="_blank" rel="noopener noreferrer">
      💼 LinkedIn
    </a>
  </div>
</div>
""", unsafe_allow_html=True)
