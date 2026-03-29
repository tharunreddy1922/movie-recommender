import streamlit as st
import pickle
import requests
import pandas as pd
import numpy as np
import ast
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Build model from CSV files
@st.cache_data
def build_model():
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id','title','overview',
                     'genres','keywords','cast','crew']]
    movies.dropna(inplace=True)

    def extract_names(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def extract_cast(obj):
        return [i['name'] for i in ast.literal_eval(obj)][:3]

    def extract_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres']   = movies['genres'].apply(extract_names)
    movies['keywords'] = movies['keywords'].apply(extract_names)
    movies['cast']     = movies['cast'].apply(extract_cast)
    movies['crew']     = movies['crew'].apply(extract_director)

    def collapse(lst):
        return [i.replace(" ","") for i in lst]

    movies['genres']   = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)
    movies['cast']     = movies['cast'].apply(collapse)
    movies['crew']     = movies['crew'].apply(collapse)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['tags'] = (movies['overview'] + movies['genres'] +
                      movies['keywords'] + movies['cast'] +
                      movies['crew'])
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

    ps = PorterStemmer()
    def stem(text):
        return " ".join([ps.stem(w) for w in text.split()])

    movies['tags'] = movies['tags'].apply(stem)

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)

    final = movies[['movie_id','title']].reset_index(drop=True)
    return final, similarity

# Fetch poster
def fetch_poster(movie_id):
    try:
        API_KEY = st.secrets["TMDB_API_KEY"]
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    except:
        pass
    return "https://placehold.co/500x750/1a1a2e/white?text=No+Poster"

# Recommend
def recommend(movie, movies, similarity):
    idx = movies[movies['title'] == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)),
                        reverse=True, key=lambda x: x[1])[1:6]
    names, posters = [], []
    for i, _ in movie_list:
        movie_id = movies.iloc[i]['movie_id']
        names.append(movies.iloc[i]['title'])
        posters.append(fetch_poster(movie_id))
    return names, posters

# UI
st.title("🎬 Movie Recommendation System")
st.markdown("*Find movies similar to your favourites — powered by ML*")

with st.spinner("Loading model... (first load takes ~30 seconds)"):
    movies, similarity = build_model()

st.divider()

selected_movie = st.selectbox(
    "Select a movie you like:",
    movies['title'].values,
    index=None,
    placeholder="e.g. Avatar, Inception, The Dark Knight..."
)

if st.button("🍿 Get Recommendations", type="primary"):
    if selected_movie:
        with st.spinner("Finding similar movies..."):
            names, posters = recommend(selected_movie, movies, similarity)
        st.subheader(f"Because you liked **{selected_movie}**:")
        cols = st.columns(5)
        for col, name, poster in zip(cols, names, posters):
            with col:
                st.image(poster, width=150)
                st.caption(f"**{name}**")
    else:
        st.warning("Please select a movie first!")

st.divider()
st.markdown(
    "<div style='text-align:center;color:gray'>Built with Streamlit · Data from TMDB · ML with scikit-learn</div>",
    unsafe_allow_html=True
)
