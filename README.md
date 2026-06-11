# 🎬 CineFusion

### Discover Your Next Favorite Movie With AI

CineFusion is a Hybrid Movie Recommendation System that combines **Content-Based Filtering** and **Collaborative Filtering** to generate personalized movie recommendations.

Built using **Python**, **Machine Learning**, **NLP**, and **Streamlit**, CineFusion helps users discover movies similar to their interests through an intelligent recommendation engine.

---

## 🚀 Live Demo

🔗 Add your deployed Streamlit link here

---

## ✨ Features

* 🎥 Search from 4,800+ movies
* 🤖 Hybrid AI recommendation engine
* 🎭 Content-Based Filtering using movie metadata
* 👥 Collaborative Filtering using user ratings
* ⚡ Lightweight deployment-friendly architecture
* 🎨 Modern Streamlit UI with glassmorphism design
* 🔍 Smart movie search functionality
* 📱 Responsive web interface

---

## 🧠 How CineFusion Works

### Content-Based Filtering

Analyzes movie genres, cast, keywords, and descriptions to identify movies with similar characteristics.

### Collaborative Filtering

Learns from user ratings and preferences to recommend movies enjoyed by similar audiences.

### Hybrid Recommendation

Combines both approaches using weighted scoring to generate more accurate and personalized recommendations.

---

## 🛠️ Tech Stack

### Machine Learning

* Scikit-Learn
* Cosine Similarity
* CountVectorizer
* NLP Feature Engineering

### Data Processing

* Pandas
* NumPy
* SciPy Sparse Matrices

### Frontend

* Streamlit
* HTML
* CSS

### Dataset

* TMDB 5000 Movies Dataset
* MovieLens Ratings Dataset

---

## 📊 Project Architecture

Input Movie
↓
Content-Based Recommendation

Genres

Cast

Keywords

Overview

↓

Collaborative Recommendation

User Ratings

User Preferences

↓

Hybrid Scoring Engine

↓

Top Movie Recommendations

---

## ⚡ Optimization

The initial implementation used precomputed similarity matrices:

* Content Similarity Matrix: ~175 MB
* Collaborative Similarity Matrix: ~720 MB

To make deployment practical, CineFusion was redesigned to use:

* Sparse Matrices (.npz)
* On-demand Cosine Similarity
* Cached Artifacts

Result:

* Storage reduced from ~900 MB to ~1.6 MB
* Faster deployment
* GitHub-friendly repository size
* Streamlit Cloud compatible

---

## 📂 Project Structure

```text
CineFusion/
│
├── app.py
├── requirements.txt
│
├── content_vectors.npz
├── collab_matrix.npz
├── movie_list.pkl
├── movies_ml.pkl
├── collab_titles.pkl
│
└── README.md
```

---

## 🖥️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/CineFusion.git

cd CineFusion

pip install -r requirements.txt

streamlit run app.py
```

---

## 📸 Screenshots

Add screenshots of:

* Home Page
* Movie Search
* Recommendation Results

---

## 👨‍💻 Author

**Satyarth Pandey**

Machine Learning • Recommendation Systems • NLP

GitHub: https://github.com/YOUR_USERNAME

LinkedIn: https://linkedin.com/in/YOUR_PROFILE

---

## ⭐ Future Improvements

* Movie Poster Integration
* Genre-Based Filtering
* Trending Movies Section
* User Accounts & Watchlists
* Deep Learning Recommendation Models
* Real-Time Movie APIs

---

### If you like this project, consider giving it a ⭐ on GitHub.
