# 🎬 Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![ML](https://img.shields.io/badge/ML-Content--Based%20Filtering-green)](https://scikit-learn.org)

A content-based movie recommendation system built with Python and Streamlit.
Type any movie and instantly get 5 similar recommendations with posters.

## 🚀 Live Demo
[👉 Click here to try it live](#) 

## 🧠 How It Works
1. Each movie is converted into a tags vector (genres + cast + director + keywords + overview)
2. TF-IDF vectorization converts text into 5000-dimensional numeric vectors
3. Cosine similarity finds the most similar movies
4. Top 5 results are returned with real posters via TMDB API

## 📁 Project Structure
movie-recommender/
├── app.py              # Streamlit frontend
├── recommender.py      # ML model and data preprocessing
├── requirements.txt    # Dependencies
└── README.md

## 🛠️ Run Locally
git clone https://github.com/tharunreddy1922/movie-recommender
cd movie-recommender
conda create -n movie_recom python=3.10
conda activate movie_recom
conda install pandas numpy scikit-learn nltk pillow -y
pip install streamlit requests
python recommender.py
streamlit run app.py

## 📊 Dataset
TMDB 5000 Movie Dataset from Kaggle

## 💡 Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn (TF-IDF, Cosine Similarity)
- NLTK (Stemming)
- Streamlit
- TMDB API
```

Press **Ctrl+S** to save.

## Step 2 — Create .gitignore file

Create a new file called `.gitignore` and paste:
```
venv/
__pycache__/
*.pyc
.env
data/
*.pkl