import streamlit as st
import pickle
import pandas as pd

movie_list = pickle.load(
    open('movie_list.pkl', 'rb')
)

content_similarity = pickle.load(
    open('content_similarity.pkl', 'rb')
)

collab_similarity_df = pickle.load(
    open('collab_similarity.pkl', 'rb')
)

movies_ml = pickle.load(
    open('movies_ml.pkl', 'rb')
)

def get_content_scores(movie):

    movie_index = movie_list[
        movie_list['title'] == movie
    ].index[0]

    movies_list = sorted(
        list(enumerate(content_similarity[movie_index])),
        reverse=True,
        key=lambda x: x[1]
    )

    scores = {}

    for i in movies_list[1:21]:

        title = movie_list.iloc[i[0]].title
        score = float(i[1])

        scores[title] = score

    return scores

def get_collab_scores(movie):

    movie_row = movies_ml[
        movies_ml['clean_title'] == movie
    ]

    if len(movie_row) == 0:
        return {}

    full_title = movie_row.iloc[0]['title']

    similar_movies = collab_similarity_df[
        full_title
    ].sort_values(
        ascending=False
    )[1:21]

    scores = {}

    for title, score in similar_movies.items():

        clean_name = title.split(" (")[0]

        scores[clean_name] = float(score)

    return scores

def hybrid_recommend(movie):

    content_scores = get_content_scores(movie)

    collab_scores = get_collab_scores(movie)

    all_movies = set(content_scores.keys()) | set(collab_scores.keys())

    final_scores = {}

    for m in all_movies:

        content = content_scores.get(m, 0)

        collab = collab_scores.get(m, 0)

        score = (0.7 * content) + (0.3 * collab)

        final_scores[m] = score

    recommendations = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:10]

st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}
.block-container{
    padding-top:2rem;
    padding-left:3rem;
    padding-right:3rem;
}
/* Background */

html, body, [data-testid="stAppViewContainer"] {
    height: 100%;
}

.stApp {
    min-height: 100vh;
}

.main .block-container {
    min-height: 80vh;
}            

.stApp {

background:
linear-gradient(
135deg,
#B8FFF9 0%,
#DDFCFB 30%,
#FFE6D5 70%,
#FFD8BE 100%
);

}
div[data-testid="stSelectbox"] label {

color:#A04527 !important;

font-size:28px !important;

font-weight:800 !important;
}
/* Hero */

.hero {

background:
linear-gradient(
135deg,
#F8FFFE 0%,
#F2FCFB 50%,
#FFF3EA 100%
);

padding:45px;

border-radius:30px;

box-shadow:
0 15px 35px rgba(0,0,0,0.08);

border:1px solid rgba(255,255,255,0.6);
}
/* Title */

.big-font {

font-size:72px;

font-weight:900;

letter-spacing:-2px;

text-align:center;

color:#E50914 !important;

margin-bottom:8px;
}
            
.sub-font {

text-align:center;

font-size:22px;

font-weight:500;

color:#475569;
}
.stat-card {

background:white;

padding:30px;

border-radius:20px;

text-align:center;

font-size:22px;

font-weight:700;

color:#2563EB;

box-shadow:
0 8px 20px rgba(0,0,0,0.06);

transition:0.3s;
}

.stat-card:hover {

transform:translateY(-5px);

box-shadow:
0 12px 25px rgba(0,0,0,0.10);
}
            
.stat-card:hover {

transform:translateY(-5px);

box-shadow:
0 12px 25px rgba(0,0,0,0.10);
}            
/* Stats Cards */

[data-testid="stAlert"] {

background:white;

border:none;

border-radius:20px;

padding:18px;

min-height:120px;

display:flex;

align-items:center;

box-shadow:
0 8px 20px rgba(0,0,0,0.06);
}

/* Movie Cards */

.movie-card {

background:
linear-gradient(
135deg,
#0F172A,
#1E293B
);
height: 260px;   
color:white;

border:none;

border-radius:24px;

padding:25px;

min-height:120px;

display:flex;
margin-bottom:30px;
            
flex-direction:column;

justify-content:center;

align-items:center;

text-align:center;

transition:0.3s;

box-shadow:
0 10px 25px rgba(0,0,0,0.15);
}

.movie-card:hover {

transform:
translateY(-8px);

box-shadow:
0 15px 35px rgba(15,23,42,0.35);
}

.movie-card h3 {

font-size:28px;
line-height:1.3;
font-weight:700;
font-family: 'Oswald', sans-serif;
letter-spacing: 1px;
margin-bottom:15px;
}

.movie-card p {

font-size:16px;

color:#CBD5E1;
}

/* Button */
/* Expander Container */

[data-testid="stExpander"] {

background: ##DDF4C8 !important;

border: none !important;

border-radius: 16px !important;

overflow: hidden;
}

/* Expander Header */

[data-testid="stExpander"] summary {

background: ##DDF4C8!important;

color: #7D1907 !important;

font-size: 20px !important;

font-weight: 700 !important;

padding: 12px 18px !important;

border-radius: 16px !important;
}

/* Arrow */

[data-testid="stExpander"] svg {

color: #4A4D0E !important;

fill: #4A4D0E !important;
}
.stButton > button {

background:
linear-gradient(
90deg,
#FFB38A,
#FF9B85
);

color:#0F172A;

font-weight:700;

height:55px;

border:none;

border-radius:16px;

transition:0.3s;
}

.stButton > button:hover {

transform:scale(1.05);
}

/* Selectbox */

.stSelectbox div[data-baseweb="select"] {

border-radius:16px;

min-height:60px;

font-size:18px;
}

/* Footer */

.footer {

text-align:center;

margin-top:40px;

color:#475569;

font-size:16px;
}

</style>
""", unsafe_allow_html=True)
# HERO

# HERO

st.markdown("""
<div class="hero">

<h1 class="big-font">
🎬 CineFusion
</h1>

<p class="sub-font">
Discover Your Next Favorite Movie With AI
</p>

<p style="
text-align:center;
font-size:20px;
font-weight:500;
color:#64748B;
margin-top:12px;
">
Powered by Content-Based Filtering + Collaborative Intelligence
</p>

<div style="
margin-top:20px;
text-align:center;
font-size:16px;
color:#A04527;
font-weight:600;
">

✨ Personalized Recommendations &nbsp; • &nbsp;
🎭 Genre-Aware Matching &nbsp; • &nbsp;
🤖 Hybrid AI Engine

</div>

</div>
""", unsafe_allow_html=True)

# STATS

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stat-card">
    🎥 4,800+ Movies Available
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
    👥 600+ User Ratings
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
    🤖 Hybrid AI Engine
    </div>
    """, unsafe_allow_html=True)

st.divider()


# MOVIE SELECT

left, center, right = st.columns([1,3,1])

with center:

    st.markdown("""
    <h3 style="
    color:#A04527;
    font-size:34px;
    font-weight:700;
    margin-bottom:10px;
    ">
    🎥 Choose a Movie
    </h3>
    """, unsafe_allow_html=True)

    selected_movie = st.selectbox(
    "",
    movie_list['title'].values,
    index=None,
    placeholder="🔍 Search for a movie...",
    label_visibility="collapsed"
)
    recommend_btn = st.button(
        "🍿 Find Similar Movies"
    )

# RECOMMENDATIONS

if recommend_btn:

    with st.spinner(
        "🎬 CineFusion is analyzing movies..."
    ):
        recommendations = hybrid_recommend(
            selected_movie
        )

    st.divider()

    st.markdown("""
    <h2 style="
    color:#A04527;
    font-size:58px;
    font-weight:900;
    margin-bottom:25px;
    margin-top:10px;
    font-family:'cursive'            
    ">
    🎯 Recommended Movies
    </h2>
    """, unsafe_allow_html=True)

    cols = st.columns(5)

    for i, (movie, score) in enumerate(recommendations):

        with cols[i % 5]:

            st.markdown(
                f"""
                <div class='movie-card'>
                    <h3>{movie}</h3>
                    
                </div>
                """,
                unsafe_allow_html=True
            )
st.markdown("<br><br><br><br>", unsafe_allow_html=True)            
with st.expander("📖 How Does CineFusion Work?"):


    st.markdown("""
          
<div style="color:#A04527;">

<h3>Content-Based Filtering</h3>
Uses movie genres, cast, keywords and overview.

<br>

<h3>Collaborative Filtering</h3>
Uses ratings and preferences from 600+ users.

<br>

<h3>Hybrid Recommendation</h3>
Combines both approaches to generate smarter recommendations.

</div>
""",
    unsafe_allow_html=True)     
st.markdown("""
<div class="footer">

🎬 CineFusion

Built by Satyarth Pandey

Machine Learning • Recommendation Systems • NLP

</div>
""", unsafe_allow_html=True)